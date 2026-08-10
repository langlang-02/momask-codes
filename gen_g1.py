#!/usr/bin/env python3
"""Generate native Unitree G1 control trajectories from text prompts."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from models.mask_transformer.transformer import MaskTransformer, ResidualTransformer
from models.vq.model import RVQVAE
from options.eval_option import EvalG1Options
from utils.fixseed import fixseed
from utils.get_opt import get_opt


PREPARE_DIR = Path(__file__).resolve().parent / "dataset" / "bones-seed_prepare"
if str(PREPARE_DIR) not in sys.path:
    sys.path.insert(0, str(PREPARE_DIR))

from g1_io import G1_FEATURE_DIM, resample_features, write_g1_csv


CLIP_VERSION = "ViT-B/32"


def checkpoint_path(model_dir: Path, requested: str, fallbacks: tuple[str, ...]) -> Path:
    requested_name = requested if requested.endswith(".tar") else f"{requested}.tar"
    candidates = (requested_name, *fallbacks)
    for name in candidates:
        path = model_dir / name
        if path.exists():
            return path
    choices = ", ".join(candidates)
    raise FileNotFoundError(f"No checkpoint found in {model_dir}; tried: {choices}")


def load_vq_model(vq_opt, device: torch.device) -> RVQVAE:
    vq_model = RVQVAE(
        vq_opt,
        G1_FEATURE_DIM,
        vq_opt.nb_code,
        vq_opt.code_dim,
        vq_opt.output_emb_width,
        vq_opt.down_t,
        vq_opt.stride_t,
        vq_opt.width,
        vq_opt.depth,
        vq_opt.dilation_growth_rate,
        vq_opt.vq_act,
        vq_opt.vq_norm,
    )
    model_dir = Path(vq_opt.checkpoints_dir) / vq_opt.dataset_name / vq_opt.name / "model"
    path = checkpoint_path(model_dir, "net_best_val", ("net_best_fid.tar", "latest.tar"))
    checkpoint = torch.load(path, map_location=device)
    model_key = "vq_model" if "vq_model" in checkpoint else "net"
    vq_model.load_state_dict(checkpoint[model_key])
    print(f"Loading G1 RVQ model from {path}")
    return vq_model


def load_mask_transformer(model_opt, device: torch.device, requested: str) -> MaskTransformer:
    transformer = MaskTransformer(
        code_dim=model_opt.code_dim,
        cond_mode="text",
        latent_dim=model_opt.latent_dim,
        ff_size=model_opt.ff_size,
        num_layers=model_opt.n_layers,
        num_heads=model_opt.n_heads,
        dropout=model_opt.dropout,
        clip_dim=512,
        cond_drop_prob=model_opt.cond_drop_prob,
        clip_version=CLIP_VERSION,
        opt=model_opt,
    )
    model_dir = Path(model_opt.checkpoints_dir) / model_opt.dataset_name / model_opt.name / "model"
    path = checkpoint_path(model_dir, requested, ("net_best_val.tar", "net_best_acc.tar", "latest.tar"))
    checkpoint = torch.load(path, map_location=device)
    state_key = "t2m_transformer" if "t2m_transformer" in checkpoint else "trans"
    missing, unexpected = transformer.load_state_dict(checkpoint[state_key], strict=False)
    if unexpected or any(not key.startswith("clip_model.") for key in missing):
        raise RuntimeError(f"Unexpected Mask Transformer checkpoint keys: missing={missing}, unexpected={unexpected}")
    print(f"Loading G1 Mask Transformer from {path}")
    return transformer


def load_residual_transformer(res_opt, vq_opt, device: torch.device, requested: str) -> ResidualTransformer:
    res_opt.num_quantizers = vq_opt.num_quantizers
    res_opt.num_tokens = vq_opt.nb_code
    transformer = ResidualTransformer(
        code_dim=vq_opt.code_dim,
        cond_mode="text",
        latent_dim=res_opt.latent_dim,
        ff_size=res_opt.ff_size,
        num_layers=res_opt.n_layers,
        num_heads=res_opt.n_heads,
        dropout=res_opt.dropout,
        clip_dim=512,
        shared_codebook=vq_opt.shared_codebook,
        cond_drop_prob=res_opt.cond_drop_prob,
        share_weight=res_opt.share_weight,
        clip_version=CLIP_VERSION,
        opt=res_opt,
    )
    model_dir = Path(res_opt.checkpoints_dir) / res_opt.dataset_name / res_opt.name / "model"
    path = checkpoint_path(model_dir, requested, ("net_best_val.tar", "net_best_loss.tar", "latest.tar"))
    checkpoint = torch.load(path, map_location=device)
    missing, unexpected = transformer.load_state_dict(checkpoint["res_transformer"], strict=False)
    if unexpected or any(not key.startswith("clip_model.") for key in missing):
        raise RuntimeError(f"Unexpected Residual Transformer checkpoint keys: missing={missing}, unexpected={unexpected}")
    print(f"Loading G1 Residual Transformer from {path}")
    return transformer


def read_prompts(text_prompt: str, text_path: str, motion_length: int) -> tuple[list[str], list[int]]:
    if text_prompt:
        if motion_length <= 0:
            raise ValueError("G1 generation requires --motion_length because no G1 length estimator is trained.")
        return [text_prompt], [motion_length]

    if not text_path:
        raise ValueError("Provide --text_prompt with --motion_length, or --text_path with caption#length lines.")

    prompts, lengths = [], []
    with Path(text_path).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("#")
            if len(parts) < 2:
                raise ValueError(f"{text_path}:{line_number}: expected caption#motion_length")
            try:
                length = int(parts[-1])
            except ValueError as exc:
                raise ValueError(f"{text_path}:{line_number}: motion length must be an integer") from exc
            if length <= 0:
                raise ValueError(f"{text_path}:{line_number}: motion length must be positive")
            prompts.append(parts[0])
            lengths.append(length)
    if not prompts:
        raise ValueError(f"No prompts found in {text_path}")
    return prompts, lengths


def load_dataset_info(data_root: Path, default_model_fps: float, default_scale: float) -> tuple[float, float]:
    info_path = data_root / "dataset_info.json"
    if not info_path.exists():
        return default_model_fps, default_scale
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    return float(info.get("target_fps", default_model_fps)), float(
        info.get("root_position_scale", default_scale)
    )


def main() -> None:
    opt = EvalG1Options().parse()
    if opt.dataset_name != "bones_g1":
        raise ValueError("gen_g1.py only supports --dataset_name bones_g1")
    fixseed(opt.seed)
    device = torch.device("cpu" if opt.gpu_id == -1 else f"cuda:{opt.gpu_id}")

    model_root = Path(opt.checkpoints_dir) / opt.dataset_name / opt.name
    model_opt = get_opt(model_root / "opt.txt", device=device)
    if model_opt.dataset_name != "bones_g1":
        raise ValueError(f"Mask Transformer {opt.name} is not a bones_g1 model")

    vq_opt = get_opt(
        Path(opt.checkpoints_dir) / opt.dataset_name / model_opt.vq_name / "opt.txt", device=device
    )
    if vq_opt.dataset_name != "bones_g1":
        raise ValueError(f"RVQ model {model_opt.vq_name} is not a bones_g1 model")
    model_opt.num_tokens = vq_opt.nb_code
    model_opt.num_quantizers = vq_opt.num_quantizers
    model_opt.code_dim = vq_opt.code_dim

    res_opt = get_opt(Path(opt.checkpoints_dir) / opt.dataset_name / opt.res_name / "opt.txt", device=device)
    if res_opt.dataset_name != "bones_g1":
        raise ValueError(f"Residual Transformer {opt.res_name} is not a bones_g1 model")
    if res_opt.vq_name != model_opt.vq_name:
        raise ValueError("Mask and Residual Transformer checkpoints reference different RVQ models")

    vq_model = load_vq_model(vq_opt, device).to(device).eval()
    mask_transformer = load_mask_transformer(model_opt, device, opt.which_epoch).to(device).eval()
    residual_transformer = load_residual_transformer(res_opt, vq_opt, device, opt.which_epoch).to(device).eval()

    unit_length = vq_opt.stride_t ** vq_opt.down_t
    prompts, requested_lengths = read_prompts(opt.text_prompt, opt.text_path, opt.motion_length)
    motion_lengths = []
    for length in requested_lengths:
        rounded = (length // unit_length) * unit_length
        if rounded < unit_length:
            raise ValueError(f"motion length {length} is shorter than one RVQ token ({unit_length} frames)")
        if rounded > model_opt.max_motion_length:
            raise ValueError(
                f"motion length {rounded} exceeds model maximum {model_opt.max_motion_length} frames"
            )
        if rounded != length:
            print(f"Rounding requested length {length} down to {rounded} for RVQ alignment")
        motion_lengths.append(rounded)
    token_lengths = torch.tensor(motion_lengths, device=device, dtype=torch.long) // unit_length

    data_root = Path(opt.g1_data_root)
    model_fps, root_position_scale = load_dataset_info(
        data_root, opt.model_fps, opt.root_position_scale
    )
    mean = np.load(Path(vq_opt.checkpoints_dir) / vq_opt.dataset_name / vq_opt.name / "meta" / "mean.npy")
    std = np.load(Path(vq_opt.checkpoints_dir) / vq_opt.dataset_name / vq_opt.name / "meta" / "std.npy")
    if mean.shape != (G1_FEATURE_DIM,) or std.shape != (G1_FEATURE_DIM,):
        raise ValueError("RVQ normalization metadata does not match the 38-D G1 representation")

    data_min = data_max = None
    if not opt.disable_data_range_clip:
        min_path, max_path = data_root / "Min.npy", data_root / "Max.npy"
        if min_path.exists() and max_path.exists():
            data_min, data_max = np.load(min_path), np.load(max_path)
        else:
            print("G1 Min.npy/Max.npy not found; generated controls will not be range-clipped")

    result_root = Path("generation") / opt.ext
    motion_dir = result_root / "motions"
    csv_dir = result_root / "g1_csv"
    motion_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    for repeat_index in range(opt.repeat_times):
        with torch.no_grad():
            token_ids = mask_transformer.generate(
                prompts,
                token_lengths,
                timesteps=opt.time_steps,
                cond_scale=opt.cond_scale,
                temperature=opt.temperature,
                topk_filter_thres=opt.topkr,
                gsample=opt.gumbel_sample,
            )
            token_ids = residual_transformer.generate(
                token_ids, prompts, token_lengths, temperature=1.0, cond_scale=opt.cond_scale
            )
            generated = vq_model.forward_decoder(token_ids).cpu().numpy()

        generated = generated * std + mean
        for sample_index, (caption, length) in enumerate(zip(prompts, motion_lengths)):
            controls_20hz = generated[sample_index, :length].astype(np.float32)
            if data_min is not None and data_max is not None:
                controls_20hz = np.clip(controls_20hz, data_min, data_max)
            controls_output = resample_features(controls_20hz, model_fps, opt.output_fps)

            stem = f"sample{sample_index:03d}_repeat{repeat_index:02d}_len{length}"
            np.save(motion_dir / f"{stem}.npy", controls_20hz)
            write_g1_csv(csv_dir / f"{stem}.csv", controls_output, root_position_scale)
            print(
                f"Generated {stem}: {len(controls_20hz)} frames at {model_fps:g} FPS, "
                f"{len(controls_output)} frames written at {opt.output_fps:g} FPS. Prompt: {caption}"
            )


if __name__ == "__main__":
    main()

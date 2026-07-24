#!/usr/bin/env python3
"""Reconstruct a native 35-D G1 trajectory with a trained MoMask RVQ model."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PREPARE_DIR = REPO_ROOT / "dataset" / "bones-seed_prepare"
if str(PREPARE_DIR) not in sys.path:
    sys.path.insert(0, str(PREPARE_DIR))

from g1_io import (  # noqa: E402
    G1_FEATURE_DIM,
    csv_values_to_model_features,
    downsample_features,
    read_g1_csv,
    resample_features,
    write_g1_csv,
)
from models.vq.model import RVQVAE  # noqa: E402
from utils.get_opt import get_opt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct a BONES-SEED G1 CSV or a 35-D MoMask NPY with a trained RVQ model."
    )
    parser.add_argument("--motion", type=Path, required=True, help="Input G1 .csv or 35-D model-space .npy.")
    parser.add_argument("--vq_name", default="bones_g1_rvq_nq6", help="RVQ experiment name.")
    parser.add_argument("--checkpoints_dir", type=Path, default=Path("checkpoints"))
    parser.add_argument(
        "--checkpoint",
        default="latest",
        help="Checkpoint filename or stem under model/. Use latest for the most recent saved state or net_best_val.",
    )
    parser.add_argument("--device", default="cpu", help="PyTorch device, e.g. cpu or cuda:0.")
    parser.add_argument("--output_dir", type=Path, default=Path("generation/rvq_reconstruction"))
    parser.add_argument("--output_fps", type=float, default=120.0, help="Output CSV frame rate.")
    parser.add_argument(
        "--input_fps",
        type=float,
        default=None,
        help="Source frame rate. Defaults to 120 for CSV and 20 for NPY.",
    )
    parser.add_argument(
        "--model_fps",
        type=float,
        default=20.0,
        help="RVQ training-data frame rate. Entire input trajectories are resampled to this rate before reconstruction.",
    )
    parser.add_argument(
        "--load_retries",
        type=int,
        default=5,
        help="Retries when latest.tar is being written by an active training job.",
    )
    parser.add_argument("--retry_delay", type=float, default=2.0, help="Seconds between checkpoint-load retries.")
    return parser.parse_args()


def load_motion(path: Path) -> tuple[np.ndarray, float]:
    if not path.is_file():
        raise FileNotFoundError(f"Motion file not found: {path}")
    if path.suffix.lower() == ".csv":
        return csv_values_to_model_features(read_g1_csv(path)), 120.0
    if path.suffix.lower() == ".npy":
        features = np.load(path).astype(np.float32)
        if features.ndim != 2 or features.shape[1] != G1_FEATURE_DIM:
            raise ValueError(f"{path} must have shape (frames, {G1_FEATURE_DIM}), got {features.shape}")
        if len(features) == 0 or not np.isfinite(features).all():
            raise ValueError(f"{path} is empty or contains non-finite values")
        return features, 20.0
    raise ValueError(f"Unsupported motion format {path.suffix!r}; use .csv or .npy")


def checkpoint_file(model_dir: Path, requested: str) -> Path:
    requested_path = Path(requested)
    if requested_path.is_absolute() or requested_path.parent != Path("."):
        return requested_path
    filename = requested if requested.endswith(".tar") else f"{requested}.tar"
    return model_dir / filename


def load_checkpoint_with_retry(path: Path, device: torch.device, retries: int, retry_delay: float) -> dict:
    if retries < 1:
        raise ValueError("--load_retries must be at least 1")
    if retry_delay < 0:
        raise ValueError("--retry_delay cannot be negative")

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            before = path.stat()
            checkpoint = torch.load(path, map_location=device)
            after = path.stat()
            if before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns:
                return checkpoint
            raise RuntimeError("checkpoint changed while it was being loaded")
        except (EOFError, OSError, RuntimeError, ValueError) as error:
            last_error = error
            if attempt == retries:
                break
            print(f"Checkpoint load attempt {attempt}/{retries} failed ({error}); retrying in {retry_delay:g}s")
            time.sleep(retry_delay)
    raise RuntimeError(f"Could not read a stable checkpoint after {retries} attempts: {path}") from last_error


def build_model(vq_opt, device: torch.device) -> RVQVAE:
    if vq_opt.dataset_name != "bones_g1":
        raise ValueError(f"{vq_opt.name!r} is not a bones_g1 RVQ experiment")
    model = RVQVAE(
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
    return model.to(device).eval()


def pad_to_unit_length(features: np.ndarray, unit_length: int) -> tuple[np.ndarray, int]:
    if unit_length <= 0:
        raise ValueError(f"Invalid RVQ unit length: {unit_length}")
    original_length = len(features)
    padded_length = ((original_length + unit_length - 1) // unit_length) * unit_length
    if padded_length == original_length:
        return features, original_length
    padding = np.repeat(features[-1:], padded_length - original_length, axis=0)
    print(f"Padding {original_length} input frames to {padded_length} for RVQ token alignment")
    return np.concatenate((features, padding), axis=0), original_length


def resample_for_rvq(features: np.ndarray, source_fps: float, model_fps: float) -> np.ndarray:
    if source_fps <= 0 or model_fps <= 0:
        raise ValueError("Source and model frame rates must be positive")
    if np.isclose(source_fps, model_fps):
        return features
    ratio = source_fps / model_fps
    if source_fps > model_fps and np.isclose(ratio, round(ratio)):
        print(
            f"Decimating the full trajectory from {source_fps:g} FPS to RVQ "
            f"{model_fps:g} FPS (keeping every {round(ratio)}th source frame)"
        )
        return downsample_features(features, source_fps, model_fps)
    print(f"Resampling the full trajectory from {source_fps:g} FPS to RVQ {model_fps:g} FPS")
    return resample_features(features, source_fps, model_fps)


def checkpoint_epoch_label(checkpoint: dict) -> str:
    """Return a filesystem-safe label for the actual checkpoint epoch."""
    epoch = checkpoint.get("ep")
    try:
        return f"ep{int(epoch):04d}"
    except (TypeError, ValueError):
        return "epunknown"


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {args.device}, but CUDA is not available")
    device = torch.device(args.device)
    source_features, default_input_fps = load_motion(args.motion)
    source_fps = args.input_fps or default_input_fps
    if source_fps <= 0 or args.model_fps <= 0 or args.output_fps <= 0:
        raise ValueError("--input_fps, --model_fps and --output_fps must be positive")
    features = resample_for_rvq(source_features, source_fps, args.model_fps)

    experiment_dir = args.checkpoints_dir / "bones_g1" / args.vq_name
    opt_path = experiment_dir / "opt.txt"
    if not opt_path.is_file():
        raise FileNotFoundError(f"RVQ options not found: {opt_path}")
    vq_opt = get_opt(str(opt_path), device)
    model = build_model(vq_opt, device)
    model_dir = experiment_dir / "model"
    checkpoint_path = checkpoint_file(model_dir, args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"RVQ checkpoint not found: {checkpoint_path}")

    checkpoint = load_checkpoint_with_retry(
        checkpoint_path, device, args.load_retries, args.retry_delay
    )
    model_key = "vq_model" if "vq_model" in checkpoint else "net"
    model.load_state_dict(checkpoint[model_key])
    print(
        f"Loaded {checkpoint_path} "
        f"(epoch={checkpoint.get('ep', 'unknown')}, iteration={checkpoint.get('total_it', 'unknown')})"
    )

    mean = np.load(experiment_dir / "meta" / "mean.npy").astype(np.float32)
    std = np.load(experiment_dir / "meta" / "std.npy").astype(np.float32)
    if mean.shape != (G1_FEATURE_DIM,) or std.shape != (G1_FEATURE_DIM,):
        raise ValueError("RVQ normalization metadata does not match the 35-D G1 representation")

    unit_length = vq_opt.stride_t ** vq_opt.down_t
    padded_features, original_length = pad_to_unit_length(features, unit_length)
    normalized = (padded_features - mean) / std
    motion_tensor = torch.from_numpy(normalized).unsqueeze(0).to(device)
    with torch.no_grad():
        token_ids, _ = model.encode(motion_tensor)
        reconstruction = model.forward_decoder(token_ids).squeeze(0).cpu().numpy()
    reconstruction = reconstruction[:original_length] * std + mean
    token_ids = token_ids.squeeze(0).cpu().numpy()

    stem = args.motion.stem
    checkpoint_stem = checkpoint_path.stem
    epoch_label = checkpoint_epoch_label(checkpoint)
    output_dir = args.output_dir / f"{stem}_{args.vq_name}_{checkpoint_stem}_{epoch_label}"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "input.npy", features)
    if not np.isclose(source_fps, args.model_fps):
        np.save(output_dir / "source_input.npy", source_features)
    np.save(output_dir / "reconstruction.npy", reconstruction.astype(np.float32))
    np.save(output_dir / "tokens.npy", token_ids)
    write_g1_csv(
        output_dir / "input.csv",
        resample_features(features, args.model_fps, args.output_fps),
    )
    write_g1_csv(
        output_dir / "reconstruction.csv",
        resample_features(reconstruction, args.model_fps, args.output_fps),
    )
    print(f"Wrote reconstruction, input and RVQ tokens to {output_dir}")
    print(f"RECONSTRUCTION_DIR={output_dir.resolve()}")


if __name__ == "__main__":
    main()

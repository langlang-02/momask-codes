#!/usr/bin/env python3
"""Convert BONES-SEED SOMA Uniform BVH files to MoMask/HumanML3D format."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from bvh_utils import downsample_indices, global_positions, load_bvh, select_joints
from soma_hml3d_mapping import (
    HML3D_FACE_JOINTS,
    HML3D_FOOT_LEFT,
    HML3D_FOOT_RIGHT,
    HML3D_LEFT_LEG,
    SOMA_TO_HML3D,
)
from split_utils import write_split_files
from text_utils import captions_from_metadata, clean_caption, format_momask_text_line

from common.skeleton import Skeleton
from utils import motion_process
from utils.paramUtil import t2m_kinematic_chain, t2m_raw_offsets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare BONES-SEED SOMA Uniform data for MoMask route-A training."
    )
    parser.add_argument("--bones-root", type=Path, default=Path("bones-seed"))
    parser.add_argument("--metadata", type=Path, default=Path("bones-seed/metadata/seed_metadata_v004.csv"))
    parser.add_argument("--temporal-labels", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("dataset/BONES-SOMA"))
    parser.add_argument("--target-fps", type=float, default=20.0)
    parser.add_argument("--scale", type=float, default=0.01, help="BVH unit scale; SOMA BVH is usually centimeters.")
    parser.add_argument("--max-feature-frames", type=int, default=196)
    parser.add_argument("--min-feature-frames", type=int, default=40)
    parser.add_argument("--feet-threshold", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--limit", type=int, default=0, help="Convert at most N metadata rows; 0 means all.")
    parser.add_argument("--drop-mirrors", action="store_true", help="Skip rows marked as mirrored.")
    parser.add_argument("--use-temporal-labels", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--print-joints", action="store_true", help="Print joints from the first readable BVH and exit.")
    return parser.parse_args()


def read_metadata(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_temporal_labels(path: Path | None) -> dict[str, list[dict[str, object]]]:
    if path is None or not path.exists():
        return {}
    labels = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            labels[item["filename"]] = item.get("events", [])
    return labels


def is_true(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def safe_id(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name)


def configure_hml3d_preprocess(example_positions: np.ndarray) -> None:
    """Set globals expected by utils.motion_process.process_file."""
    motion_process.l_idx1, motion_process.l_idx2 = HML3D_LEFT_LEG
    motion_process.fid_r = HML3D_FOOT_RIGHT
    motion_process.fid_l = HML3D_FOOT_LEFT
    motion_process.face_joint_indx = HML3D_FACE_JOINTS
    motion_process.n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    motion_process.kinematic_chain = t2m_kinematic_chain

    target_skeleton = Skeleton(motion_process.n_raw_offsets, t2m_kinematic_chain, "cpu")
    motion_process.tgt_offsets = target_skeleton.get_offsets_joints(
        torch.from_numpy(example_positions[0]).float()
    )


def load_hml3d_positions(bvh_path: Path, target_fps: float, scale: float) -> tuple[np.ndarray, list[str]]:
    motion = load_bvh(bvh_path)
    positions = global_positions(motion, scale=scale)
    indices = downsample_indices(motion.frametime, len(positions), target_fps)
    positions = positions[indices]
    positions = select_joints(positions, motion.names, SOMA_TO_HML3D)
    return positions.astype(np.float32), motion.names


def iter_segments(
    positions: np.ndarray,
    captions: list[str],
    events: list[dict[str, object]],
    use_temporal_labels: bool,
    target_fps: float,
) -> list[tuple[np.ndarray, list[str]]]:
    if use_temporal_labels and events:
        segments = []
        for event in events:
            caption = clean_caption(event.get("description"))
            if not caption:
                continue
            start = int(round(float(event.get("start_time", 0.0)) * target_fps))
            end = int(round(float(event.get("end_time", 0.0)) * target_fps))
            start = max(0, min(start, len(positions)))
            end = max(start, min(end, len(positions)))
            if end > start:
                segments.append((positions[start:end], [caption]))
        if segments:
            return segments
    return [(positions, captions)]


def iter_windows(
    positions: np.ndarray,
    max_feature_frames: int,
    min_feature_frames: int,
) -> list[np.ndarray]:
    max_position_frames = max_feature_frames + 1
    min_position_frames = min_feature_frames + 1
    if len(positions) < min_position_frames:
        return []
    if len(positions) <= max_position_frames:
        return [positions]

    windows = []
    stride = max_feature_frames
    for start in range(0, len(positions) - min_position_frames + 1, stride):
        end = min(start + max_position_frames, len(positions))
        if end - start >= min_position_frames:
            windows.append(positions[start:end])
    return windows


def write_text_file(path: Path, captions: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for caption in captions:
            f.write(format_momask_text_line(caption) + "\n")


def main() -> None:
    args = parse_args()
    rows = read_metadata(args.metadata)
    temporal = read_temporal_labels(args.temporal_labels)

    output_root = args.output_root
    motion_dir = output_root / "new_joint_vecs"
    text_dir = output_root / "texts"
    motion_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    sum_vec = None
    sum_sq_vec = None
    frame_count = 0
    written_ids: list[str] = []
    configured = False
    converted_rows = 0
    skipped = 0

    def add_feature_stats(feature: np.ndarray) -> None:
        nonlocal sum_vec, sum_sq_vec, frame_count
        batch_sum = feature.sum(axis=0, dtype=np.float64)
        batch_sq = np.square(feature.astype(np.float64)).sum(axis=0)
        if sum_vec is None:
            sum_vec = np.zeros_like(batch_sum)
            sum_sq_vec = np.zeros_like(batch_sq)
        sum_vec += batch_sum
        sum_sq_vec += batch_sq
        frame_count += feature.shape[0]

    iterator = rows if args.limit <= 0 else rows[: args.limit]
    for row in tqdm(iterator, desc="Converting BONES-SOMA"):
        if args.drop_mirrors and is_true(row.get("is_mirror")):
            continue

        rel_bvh = row.get("move_soma_uniform_path") or ""
        if not rel_bvh:
            skipped += 1
            continue
        bvh_path = args.bones_root / rel_bvh
        if not bvh_path.exists():
            skipped += 1
            continue

        captions = captions_from_metadata(row)
        if not captions:
            skipped += 1
            continue

        try:
            positions, names = load_hml3d_positions(bvh_path, args.target_fps, args.scale)
            if args.print_joints:
                print("\n".join(names))
                return
            if not configured:
                configure_hml3d_preprocess(positions)
                configured = True
        except Exception as exc:
            print(f"[skip] failed to read/map {bvh_path}: {exc}", file=sys.stderr)
            skipped += 1
            continue

        filename = row.get("filename") or Path(rel_bvh).stem
        segments = iter_segments(
            positions,
            captions,
            temporal.get(filename, []),
            args.use_temporal_labels,
            args.target_fps,
        )

        sample_idx = 0
        for segment_idx, (segment_positions, segment_captions) in enumerate(segments):
            for window_idx, window_positions in enumerate(
                iter_windows(segment_positions, args.max_feature_frames, args.min_feature_frames)
            ):
                sample_id = safe_id(f"{filename}_seg{segment_idx:03d}_win{window_idx:03d}")
                motion_path = motion_dir / f"{sample_id}.npy"
                text_path = text_dir / f"{sample_id}.txt"
                if motion_path.exists() and text_path.exists() and not args.overwrite:
                    add_feature_stats(np.load(motion_path))
                    written_ids.append(sample_id)
                    continue

                try:
                    processed = motion_process.process_file(window_positions.copy(), args.feet_threshold)
                    feature = processed[0] if isinstance(processed, tuple) else processed
                    feature = feature.astype(np.float32)
                except Exception as exc:
                    print(f"[skip] process_file failed for {sample_id}: {exc}", file=sys.stderr)
                    skipped += 1
                    continue

                np.save(motion_path, feature)
                write_text_file(text_path, segment_captions)
                written_ids.append(sample_id)
                sample_idx += 1
                add_feature_stats(feature)

        converted_rows += 1

    if not written_ids:
        raise RuntimeError("No samples were written. Check paths, mapping, and metadata filters.")

    if frame_count > 0 and sum_vec is not None and sum_sq_vec is not None:
        mean = sum_vec / frame_count
        var = np.maximum(sum_sq_vec / frame_count - np.square(mean), 1e-12)
        std = np.sqrt(var)
        std[std < 1e-6] = 1.0
        np.save(output_root / "Mean.npy", mean.astype(np.float32))
        np.save(output_root / "Std.npy", std.astype(np.float32))

    write_split_files(output_root, sorted(set(written_ids)), args.seed, args.val_ratio, args.test_ratio)

    print(f"Output root: {output_root}")
    print(f"Samples written/reused: {len(set(written_ids))}")
    print(f"Metadata rows converted: {converted_rows}")
    print(f"Skipped items: {skipped}")
    print(f"Feature frames used for stats: {frame_count}")


if __name__ == "__main__":
    # Keep BLAS from oversubscribing CPUs on shared Slurm nodes.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()

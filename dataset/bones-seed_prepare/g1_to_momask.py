#!/usr/bin/env python3
"""Convert BONES-SEED Unitree G1 CSV trajectories to native MoMask features."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from g1_io import G1_FEATURE_DIM, csv_values_to_model_features, downsample_features, read_g1_csv
from split_utils import write_split_files
from text_utils import captions_from_metadata, clean_caption, format_momask_text_line


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare BONES-SEED Unitree G1 trajectories for native MoMask training."
    )
    parser.add_argument("--bones-root", type=Path, default=Path("bones-seed"))
    parser.add_argument("--metadata", type=Path, default=Path("bones-seed/metadata/seed_metadata_v004.csv"))
    parser.add_argument("--temporal-labels", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("dataset/BONES-G1-rot6d"))
    parser.add_argument("--source-fps", type=float, default=120.0)
    parser.add_argument("--target-fps", type=float, default=20.0)
    parser.add_argument(
        "--root-position-scale",
        type=float,
        default=0.01,
        help="Scale BONES-SEED root translations from centimeters to meters.",
    )
    parser.add_argument("--max-motion-frames", type=int, default=196)
    parser.add_argument("--min-motion-frames", type=int, default=40)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--limit", type=int, default=0, help="Convert at most N metadata rows; 0 means all.")
    parser.add_argument("--start-index", type=int, default=0, help="Start from this metadata row index.")
    parser.add_argument("--drop-mirrors", action="store_true", help="Skip rows marked as mirrored.")
    parser.add_argument("--use-temporal-labels", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--rebuild-stats",
        action="store_true",
        help="Rebuild all statistics and split files from paired output files after conversion.",
    )
    return parser.parse_args()


def read_metadata(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_temporal_labels(path: Path | None) -> dict[str, list[dict[str, object]]]:
    if path is None or not path.exists():
        return {}
    labels: dict[str, list[dict[str, object]]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                labels[item["filename"]] = item.get("events", [])
    return labels


def is_true(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def safe_id(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name)


def iter_segments(
    motion: np.ndarray,
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
            start = max(0, min(start, len(motion)))
            end = max(start, min(end, len(motion)))
            if end > start:
                segments.append((motion[start:end], [caption]))
        if segments:
            return segments
    return [(motion, captions)]


def iter_windows(motion: np.ndarray, max_frames: int, min_frames: int) -> list[np.ndarray]:
    if min_frames <= 0 or max_frames < min_frames:
        raise ValueError("max-motion-frames must be at least min-motion-frames and both must be positive")
    if len(motion) < min_frames:
        return []
    if len(motion) <= max_frames:
        return [motion]

    windows = []
    for start in range(0, len(motion) - min_frames + 1, max_frames):
        end = min(start + max_frames, len(motion))
        if end - start >= min_frames:
            windows.append(motion[start:end])
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

    sum_vec: np.ndarray | None = None
    sum_sq_vec: np.ndarray | None = None
    min_vec: np.ndarray | None = None
    max_vec: np.ndarray | None = None
    frame_count = 0
    written_ids: list[str] = []
    converted_rows = 0
    skipped = 0

    def add_feature_stats(feature: np.ndarray) -> None:
        nonlocal sum_vec, sum_sq_vec, min_vec, max_vec, frame_count
        batch_sum = feature.sum(axis=0, dtype=np.float64)
        batch_sq = np.square(feature.astype(np.float64)).sum(axis=0)
        if sum_vec is None:
            sum_vec = np.zeros_like(batch_sum)
            sum_sq_vec = np.zeros_like(batch_sq)
            min_vec = feature.min(axis=0).astype(np.float64)
            max_vec = feature.max(axis=0).astype(np.float64)
        else:
            min_vec = np.minimum(min_vec, feature.min(axis=0))
            max_vec = np.maximum(max_vec, feature.max(axis=0))
        sum_vec += batch_sum
        sum_sq_vec += batch_sq
        frame_count += feature.shape[0]

    def rebuild_stats_from_existing() -> tuple[list[str], int]:
        existing_ids = []
        bad_files = 0
        for motion_path in tqdm(sorted(motion_dir.glob("*.npy")), desc="Rebuilding BONES-G1 stats", miniters=1000):
            sample_id = motion_path.stem
            if not (text_dir / f"{sample_id}.txt").exists():
                bad_files += 1
                continue
            try:
                feature = np.load(motion_path)
            except Exception as exc:
                print(f"[skip] failed to load {motion_path}: {exc}", file=sys.stderr)
                bad_files += 1
                continue
            if (
                feature.ndim != 2
                or feature.shape[1] != G1_FEATURE_DIM
                or feature.shape[0] <= 0
                or not np.isfinite(feature).all()
            ):
                print(f"[skip] invalid existing feature {motion_path}: shape={feature.shape}", file=sys.stderr)
                bad_files += 1
                continue
            add_feature_stats(feature)
            existing_ids.append(sample_id)
        return existing_ids, bad_files

    start_index = max(args.start_index, 0)
    iterator = rows[start_index:]
    if args.limit > 0:
        iterator = iterator[: args.limit]

    for row in tqdm(iterator, desc="Converting BONES-G1", miniters=100):
        if args.drop_mirrors and is_true(row.get("is_mirror")):
            continue

        relative_path = row.get("move_g1_path") or ""
        csv_path = args.bones_root / relative_path
        if not relative_path or not csv_path.exists():
            skipped += 1
            continue

        captions = captions_from_metadata(row)
        if not captions:
            skipped += 1
            continue

        try:
            csv_values = read_g1_csv(csv_path)
            motion = csv_values_to_model_features(csv_values, args.root_position_scale)
            motion = downsample_features(motion, args.source_fps, args.target_fps)
        except Exception as exc:
            print(f"[skip] failed to read {csv_path}: {exc}", file=sys.stderr)
            skipped += 1
            continue

        filename = row.get("filename") or Path(relative_path).stem
        segments = iter_segments(
            motion,
            captions,
            temporal.get(filename, []),
            args.use_temporal_labels,
            args.target_fps,
        )
        for segment_index, (segment, segment_captions) in enumerate(segments):
            for window_index, window in enumerate(
                iter_windows(segment, args.max_motion_frames, args.min_motion_frames)
            ):
                sample_id = safe_id(f"{filename}_seg{segment_index:03d}_win{window_index:03d}")
                motion_path = motion_dir / f"{sample_id}.npy"
                text_path = text_dir / f"{sample_id}.txt"
                if motion_path.exists() and text_path.exists() and not args.overwrite:
                    if not args.rebuild_stats:
                        add_feature_stats(np.load(motion_path))
                        written_ids.append(sample_id)
                    continue

                np.save(motion_path, window.astype(np.float32))
                write_text_file(text_path, segment_captions)
                if not args.rebuild_stats:
                    add_feature_stats(window)
                    written_ids.append(sample_id)
        converted_rows += 1

    bad_existing = 0
    if args.rebuild_stats:
        sum_vec = sum_sq_vec = min_vec = max_vec = None
        frame_count = 0
        written_ids, bad_existing = rebuild_stats_from_existing()

    if not written_ids or sum_vec is None or sum_sq_vec is None:
        raise RuntimeError("No G1 samples were written. Check the extracted g1 directory and metadata paths.")

    mean = sum_vec / frame_count
    variance = np.maximum(sum_sq_vec / frame_count - np.square(mean), 1e-12)
    std = np.sqrt(variance)
    std[std < 1e-6] = 1.0
    np.save(output_root / "Mean.npy", mean.astype(np.float32))
    np.save(output_root / "Std.npy", std.astype(np.float32))
    np.save(output_root / "Min.npy", min_vec.astype(np.float32))
    np.save(output_root / "Max.npy", max_vec.astype(np.float32))
    write_split_files(output_root, sorted(set(written_ids)), args.seed, args.val_ratio, args.test_ratio)

    dataset_info = {
        "format": "bones_seed_g1_root_rot6d_v2",
        "feature_dim": G1_FEATURE_DIM,
        "feature_units": {
            "root_translation": "meters",
            "root_rotation": "continuous_6d",
            "joint_angles": "radians",
        },
        "source_fps": args.source_fps,
        "target_fps": args.target_fps,
        "root_position_scale": args.root_position_scale,
    }
    with (output_root / "dataset_info.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Output root: {output_root}")
    print(f"Samples written/reused: {len(set(written_ids))}")
    print(f"Metadata start index: {start_index}")
    print(f"Metadata rows converted: {converted_rows}")
    print(f"Skipped items: {skipped}")
    print(f"Bad existing files skipped while rebuilding stats: {bad_existing}")
    print(f"Feature frames used for stats: {frame_count}")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()

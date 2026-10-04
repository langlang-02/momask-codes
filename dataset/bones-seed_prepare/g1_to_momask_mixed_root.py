#!/usr/bin/env python3
"""Convert BONES-SEED Unitree G1 CSV trajectories to mixed-root MoMask features."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import tarfile
from pathlib import Path

import numpy as np
from tqdm import tqdm


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from g1_io import (  # noqa: E402
    G1_CSV_FEATURE_DIM,
    G1_FEATURE_COLUMNS,
    G1_FEATURE_DIM,
    G1_MIXED_ROOT_ABS_XYZ_SLICE,
    G1_MIXED_ROOT_FEATURE_DIM,
    G1_MIXED_ROOT_FEATURE_NAMES,
    G1_MIXED_ROOT_HEIGHT_SLICE,
    G1_MIXED_ROOT_JOINT_SLICE,
    G1_MIXED_ROOT_ROLL_PITCH_6D_SLICE,
    G1_MIXED_ROOT_XY_VEL_BODY_SLICE,
    G1_MIXED_ROOT_YAW_VEL_SLICE,
    csv_values_to_model_features,
    downsample_features,
    model_features_to_mixed_root_features,
    read_g1_csv,
)
from g1_to_momask import (  # noqa: E402
    captions_from_metadata,
    is_true,
    iter_segments,
    iter_windows,
    read_metadata,
    read_temporal_labels,
    safe_id,
    write_text_file,
)
from split_utils import write_split_files  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare BONES-SEED Unitree G1 trajectories with yaw/planar velocity plus absolute XYZ anchors."
    )
    parser.add_argument("--bones-root", type=Path, default=Path("bones-seed"))
    parser.add_argument(
        "--g1-tar",
        type=Path,
        default=None,
        help="Optional g1.tar.gz. When set, CSV files are streamed directly from the archive.",
    )
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--temporal-labels", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("dataset/BONES-G1-mixed-root"))
    parser.add_argument("--source-fps", type=float, default=120.0)
    parser.add_argument("--target-fps", type=float, default=20.0)
    parser.add_argument(
        "--velocity-unit",
        choices=("per_frame", "per_second"),
        default="per_frame",
        help="Store root_yaw_vel/root_xy_vel_body as per-frame deltas or per-second rates.",
    )
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
        help="Rebuild statistics and split files from paired output files after conversion.",
    )
    return parser.parse_args()


def resolve_metadata_path(args: argparse.Namespace) -> Path:
    if args.metadata is not None:
        return args.metadata
    return args.bones_root / "metadata" / "seed_metadata_v004.csv"


def resolve_temporal_labels_path(args: argparse.Namespace) -> Path | None:
    if args.temporal_labels is not None:
        return args.temporal_labels
    candidate = args.bones_root / "metadata" / "seed_metadata_v002_temporal_labels.jsonl"
    return candidate if candidate.exists() else None


def feature_layout() -> dict[str, list[int]]:
    return {
        "root_yaw_vel": [G1_MIXED_ROOT_YAW_VEL_SLICE.start, G1_MIXED_ROOT_YAW_VEL_SLICE.stop],
        "root_xy_vel_body": [G1_MIXED_ROOT_XY_VEL_BODY_SLICE.start, G1_MIXED_ROOT_XY_VEL_BODY_SLICE.stop],
        "root_height": [G1_MIXED_ROOT_HEIGHT_SLICE.start, G1_MIXED_ROOT_HEIGHT_SLICE.stop],
        "root_roll_pitch_rot6d": [
            G1_MIXED_ROOT_ROLL_PITCH_6D_SLICE.start,
            G1_MIXED_ROOT_ROLL_PITCH_6D_SLICE.stop,
        ],
        "root_abs_xyz": [G1_MIXED_ROOT_ABS_XYZ_SLICE.start, G1_MIXED_ROOT_ABS_XYZ_SLICE.stop],
        "joint_angles": [G1_MIXED_ROOT_JOINT_SLICE.start, G1_MIXED_ROOT_JOINT_SLICE.stop],
    }


def validate_mixed_feature(feature: np.ndarray, path: Path | None = None) -> bool:
    valid = (
        feature.ndim == 2
        and feature.shape[1] == G1_MIXED_ROOT_FEATURE_DIM
        and feature.shape[0] > 0
        and np.isfinite(feature).all()
    )
    if not valid and path is not None:
        print(f"[skip] invalid existing feature {path}: shape={feature.shape}", file=sys.stderr)
    return valid


def read_g1_csv_from_binary_stream(stream: object, source: str) -> np.ndarray:
    text_stream = io.StringIO(stream.read().decode("utf-8"))
    reader = csv.DictReader(text_stream)
    if reader.fieldnames is None:
        raise ValueError(f"{source}: missing CSV header")
    missing = [name for name in G1_FEATURE_COLUMNS if name not in reader.fieldnames]
    if missing:
        raise ValueError(f"{source}: missing G1 columns: {', '.join(missing)}")

    rows = []
    for line_number, row in enumerate(reader, start=2):
        try:
            rows.append([float(row[name]) for name in G1_FEATURE_COLUMNS])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{source}:{line_number}: invalid numeric value") from exc
    csv_values = np.asarray(rows, dtype=np.float32)
    if csv_values.ndim != 2 or csv_values.shape[1] != G1_CSV_FEATURE_DIM:
        raise ValueError(f"{source}: expected (frames, {G1_CSV_FEATURE_DIM}), got {csv_values.shape}")
    if len(csv_values) == 0 or not np.isfinite(csv_values).all():
        raise ValueError(f"{source}: empty or non-finite G1 CSV trajectory")
    return csv_values


def main() -> None:
    args = parse_args()
    metadata_path = resolve_metadata_path(args)
    temporal_labels_path = resolve_temporal_labels_path(args)
    rows = read_metadata(metadata_path)
    temporal = read_temporal_labels(temporal_labels_path)

    output_root = args.output_root
    motion_dir = output_root / "new_joint_vecs"
    text_dir = output_root / "texts"
    metadata_dir = output_root / "metadata"
    motion_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    sum_vec: np.ndarray | None = None
    sum_sq_vec: np.ndarray | None = None
    min_vec: np.ndarray | None = None
    max_vec: np.ndarray | None = None
    frame_count = 0
    written_ids: list[str] = []
    window_metadata: list[dict[str, object]] = []
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
        for motion_path in tqdm(sorted(motion_dir.glob("*.npy")), desc="Rebuilding BONES-G1 mixed stats", miniters=1000):
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
            if not validate_mixed_feature(feature, motion_path):
                bad_files += 1
                continue
            add_feature_stats(feature)
            existing_ids.append(sample_id)
        return existing_ids, bad_files

    start_index = max(args.start_index, 0)
    iterator = rows[start_index:]
    if args.limit > 0:
        iterator = iterator[: args.limit]
    indexed_rows = [
        (row_index, row)
        for row_index, row in enumerate(iterator, start=start_index)
        if not (args.drop_mirrors and is_true(row.get("is_mirror")))
    ]

    def process_row_motion(row_index: int, row: dict[str, str], motion: np.ndarray, relative_path: str) -> bool:
        nonlocal converted_rows
        captions = captions_from_metadata(row)
        if not captions:
            return False
        if motion.shape[1] != G1_FEATURE_DIM:
            raise ValueError(f"Expected {G1_FEATURE_DIM} native features, got {motion.shape}")
        motion = downsample_features(motion, args.source_fps, args.target_fps)

        filename = row.get("filename") or Path(relative_path).stem
        segments = iter_segments(
            motion,
            captions,
            temporal.get(filename, []),
            args.use_temporal_labels,
            args.target_fps,
        )
        wrote_any = False
        for segment_index, (segment, segment_captions) in enumerate(segments):
            for window_index, window in enumerate(iter_windows(segment, args.max_motion_frames, args.min_motion_frames)):
                sample_id = safe_id(f"{filename}_seg{segment_index:03d}_win{window_index:03d}")
                motion_path = motion_dir / f"{sample_id}.npy"
                text_path = text_dir / f"{sample_id}.txt"
                if motion_path.exists() and text_path.exists() and not args.overwrite:
                    if not args.rebuild_stats:
                        existing_feature = np.load(motion_path)
                        if validate_mixed_feature(existing_feature, motion_path):
                            add_feature_stats(existing_feature)
                            written_ids.append(sample_id)
                            wrote_any = True
                    continue

                mixed, mixed_metadata = model_features_to_mixed_root_features(
                    window,
                    target_fps=args.target_fps,
                    velocity_unit=args.velocity_unit,
                    return_metadata=True,
                )
                np.save(motion_path, mixed.astype(np.float32))
                write_text_file(text_path, segment_captions)
                if not args.rebuild_stats:
                    add_feature_stats(mixed)
                    written_ids.append(sample_id)
                    wrote_any = True
                    window_metadata.append(
                        {
                            "sample_id": sample_id,
                            "metadata_row_index": row_index,
                            "filename": filename,
                            "move_g1_path": relative_path,
                            "segment_index": segment_index,
                            "window_index": window_index,
                            "frames": int(mixed.shape[0]),
                            "captions": segment_captions,
                            **mixed_metadata,
                        }
                    )
        converted_rows += 1
        return wrote_any

    if args.g1_tar is None:
        for row_index, row in tqdm(indexed_rows, desc="Converting BONES-G1 mixed root", miniters=100):
            relative_path = row.get("move_g1_path") or ""
            csv_path = args.bones_root / relative_path
            if not relative_path or not csv_path.exists():
                skipped += 1
                continue

            try:
                csv_values = read_g1_csv(csv_path)
                motion = csv_values_to_model_features(csv_values, args.root_position_scale)
                if not process_row_motion(row_index, row, motion, relative_path):
                    skipped += 1
            except Exception as exc:
                print(f"[skip] failed to read {csv_path}: {exc}", file=sys.stderr)
                skipped += 1
                continue
    else:
        rows_by_path: dict[str, list[tuple[int, dict[str, str]]]] = {}
        for row_index, row in indexed_rows:
            relative_path = (row.get("move_g1_path") or "").lstrip("./")
            if not relative_path:
                skipped += 1
                continue
            rows_by_path.setdefault(relative_path, []).append((row_index, row))

        seen_paths: set[str] = set()
        with tarfile.open(args.g1_tar, "r|gz") as tar:
            for member in tqdm(tar, desc="Streaming BONES-G1 tar", miniters=1000):
                if not member.isfile() or not member.name.endswith(".csv"):
                    continue
                relative_path = member.name.lstrip("./")
                matches = rows_by_path.get(relative_path)
                if not matches:
                    continue
                extracted = tar.extractfile(member)
                if extracted is None:
                    skipped += len(matches)
                    continue
                try:
                    csv_values = read_g1_csv_from_binary_stream(extracted, relative_path)
                    motion = csv_values_to_model_features(csv_values, args.root_position_scale)
                except Exception as exc:
                    print(f"[skip] failed to read {relative_path} from {args.g1_tar}: {exc}", file=sys.stderr)
                    skipped += len(matches)
                    continue
                seen_paths.add(relative_path)
                for row_index, row in matches:
                    try:
                        if not process_row_motion(row_index, row, motion, relative_path):
                            skipped += 1
                    except Exception as exc:
                        print(f"[skip] failed to convert {relative_path}: {exc}", file=sys.stderr)
                        skipped += 1
                if len(seen_paths) == len(rows_by_path):
                    break
        missing_paths = set(rows_by_path) - seen_paths
        skipped += sum(len(rows_by_path[path]) for path in missing_paths)
        if missing_paths:
            print(f"[skip] metadata G1 paths not found in tar: {len(missing_paths)}", file=sys.stderr)

    bad_existing = 0
    if args.rebuild_stats:
        sum_vec = sum_sq_vec = min_vec = max_vec = None
        frame_count = 0
        written_ids, bad_existing = rebuild_stats_from_existing()

    if not written_ids or sum_vec is None or sum_sq_vec is None:
        raise RuntimeError("No mixed-root G1 samples were written. Check G1 CSV and metadata paths.")

    mean = sum_vec / frame_count
    variance = np.maximum(sum_sq_vec / frame_count - np.square(mean), 1e-12)
    std = np.sqrt(variance)
    std[std < 1e-6] = 1.0
    np.save(output_root / "Mean.npy", mean.astype(np.float32))
    np.save(output_root / "Std.npy", std.astype(np.float32))
    np.save(output_root / "Min.npy", min_vec.astype(np.float32))
    np.save(output_root / "Max.npy", max_vec.astype(np.float32))
    write_split_files(output_root, sorted(set(written_ids)), args.seed, args.val_ratio, args.test_ratio)

    if window_metadata:
        mode = "w" if args.overwrite else "a"
        with (metadata_dir / "mixed_root_windows.jsonl").open(mode, encoding="utf-8") as f:
            for item in window_metadata:
                json.dump(item, f, sort_keys=True)
                f.write("\n")

    dataset_info = {
        "format": "bones_seed_g1_mixed_root_absxyz_v1",
        "native_format": "bones_seed_g1_root_rot6d_v2",
        "feature_dim": G1_MIXED_ROOT_FEATURE_DIM,
        "native_feature_dim": G1_FEATURE_DIM,
        "feature_names": G1_MIXED_ROOT_FEATURE_NAMES,
        "feature_layout": feature_layout(),
        "feature_units": {
            "root_yaw_vel": f"radians_{args.velocity_unit}",
            "root_xy_vel_body": f"meters_{args.velocity_unit}",
            "root_height": "meters",
            "root_roll_pitch_rot6d": "continuous_6d_after_removing_world_z_yaw",
            "root_abs_xyz": "meters",
            "joint_angles": "radians",
        },
        "source_fps": args.source_fps,
        "target_fps": args.target_fps,
        "root_position_scale": args.root_position_scale,
        "velocity_unit": args.velocity_unit,
        "metadata": str(metadata_path),
        "temporal_labels": str(temporal_labels_path) if temporal_labels_path is not None else None,
        "use_temporal_labels": bool(args.use_temporal_labels),
        "g1_tar": str(args.g1_tar) if args.g1_tar is not None else None,
        "reconstruction_note": (
            "Use mixed_root_features_to_model_features(..., use_absolute_position=True) to reconstruct 38-D "
            "features with absolute XY anchors. Pass root_yaw0_rad from metadata/mixed_root_windows.jsonl "
            "for exact ground-truth orientation."
        ),
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
    print(f"Feature dim: {G1_MIXED_ROOT_FEATURE_DIM}")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()

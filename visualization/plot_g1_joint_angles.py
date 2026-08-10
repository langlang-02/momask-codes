#!/usr/bin/env python3
"""Plot native Unitree G1 joint-angle trajectories and reconstruction errors."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PREPARE_DIR = REPO_ROOT / "dataset" / "bones-seed_prepare"
if str(PREPARE_DIR) not in sys.path:
    sys.path.insert(0, str(PREPARE_DIR))

from g1_io import (  # noqa: E402
    G1_FEATURE_COLUMNS,
    G1_FEATURE_DIM,
    G1_JOINT_SLICE,
    csv_values_to_model_features,
    read_g1_csv,
)


JOINT_START = G1_JOINT_SLICE.start
JOINT_NAMES = G1_FEATURE_COLUMNS[JOINT_START:]


def validate_features(features: np.ndarray, source: str) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != G1_FEATURE_DIM:
        raise ValueError(
            f"{source} must have shape (frames, {G1_FEATURE_DIM}), got {features.shape}."
        )
    if len(features) == 0 or not np.isfinite(features).all():
        raise ValueError(f"{source} is empty or contains non-finite values.")
    return features


def load_model_features(path: str | Path) -> np.ndarray:
    """Load a model-space NPY or convert a BONES-SEED CSV to model units."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return csv_values_to_model_features(read_g1_csv(path))
    if path.suffix.lower() == ".npy":
        return validate_features(np.load(path), str(path))
    raise ValueError(f"Unsupported trajectory format: {path}. Use .npy or .csv.")


def _joint_label(name: str) -> str:
    return name.removesuffix("_joint_dof").replace("_", " ")


def _write_error_table(
    path: Path,
    ground_truth_deg: np.ndarray,
    prediction_deg: np.ndarray,
) -> None:
    difference = prediction_deg - ground_truth_deg
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("joint", "mae_deg", "rmse_deg", "max_abs_error_deg"))
        for index, name in enumerate(JOINT_NAMES):
            error = difference[:, index]
            writer.writerow(
                (
                    name,
                    f"{np.mean(np.abs(error)):.6f}",
                    f"{np.sqrt(np.mean(np.square(error))):.6f}",
                    f"{np.max(np.abs(error)):.6f}",
                )
            )


def plot_joint_angle_comparison(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    output_dir: str | Path,
    *,
    stem: str = "comparison",
    fps: float = 20.0,
    prediction_label: str = "RVQ reconstruction",
) -> tuple[Path, Path]:
    """Save a 29-DoF angle comparison image and per-joint error table.

    Inputs use model units: root position in meters, root orientation in 6-D,
    and 29 hinge-joint angles in radians. Only the hinge joints are reported.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    ground_truth = validate_features(ground_truth, "ground_truth")
    prediction = validate_features(prediction, "prediction")
    frame_count = min(len(ground_truth), len(prediction))
    if frame_count == 0:
        raise ValueError("No overlapping frames to plot.")
    if len(ground_truth) != len(prediction):
        print(
            f"Warning: trajectory lengths differ ({len(ground_truth)} vs {len(prediction)}); "
            f"plotting the first {frame_count} frames."
        )

    ground_truth_deg = np.rad2deg(ground_truth[:frame_count, JOINT_START:])
    prediction_deg = np.rad2deg(prediction[:frame_count, JOINT_START:])
    absolute_error_deg = np.abs(prediction_deg - ground_truth_deg)
    time_axis = np.arange(frame_count, dtype=np.float32) / fps

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"{stem}_joint_angles.png"
    table_path = output_dir / f"{stem}_joint_angle_errors.csv"

    fig, axes = plt.subplots(6, 5, figsize=(24, 18), sharex=True)
    for index, axis in enumerate(axes.flat):
        if index >= len(JOINT_NAMES):
            axis.axis("off")
            continue
        axis.plot(time_axis, ground_truth_deg[:, index], color="#1f2937", linewidth=1.0)
        axis.plot(time_axis, prediction_deg[:, index], color="#007f87", linewidth=1.0, linestyle="--")
        error_axis = axis.twinx()
        error_axis.plot(time_axis, absolute_error_deg[:, index], color="#d05a1e", linewidth=0.8, alpha=0.8)
        error_axis.tick_params(axis="y", labelsize=6, colors="#d05a1e")
        axis.set_title(_joint_label(JOINT_NAMES[index]), fontsize=8)
        axis.tick_params(axis="both", labelsize=7)
        axis.grid(alpha=0.25, linewidth=0.5)
        if index % 5 == 0:
            axis.set_ylabel("angle (deg)", fontsize=7)
        if index >= 25:
            axis.set_xlabel("time (s)", fontsize=7)

    legend = (
        Line2D((0,), (0,), color="#1f2937", label="Ground truth"),
        Line2D((0,), (0,), color="#007f87", linestyle="--", label=prediction_label),
        Line2D((0,), (0,), color="#d05a1e", label="Absolute error (right axis)"),
    )
    fig.legend(handles=legend, loc="upper center", ncol=3, frameon=False, fontsize=10)
    fig.suptitle("G1 joint-angle comparison", y=0.985, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(image_path, dpi=180)
    plt.close(fig)

    _write_error_table(table_path, ground_truth_deg, prediction_deg)
    return image_path, table_path


def plot_joint_angle_batch(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    output_dir: str | Path,
    *,
    fps: float = 20.0,
    prediction_label: str = "RVQ reconstruction",
) -> list[tuple[Path, Path]]:
    """Plot one comparison image per paired batch item."""
    ground_truth = np.asarray(ground_truth)
    prediction = np.asarray(prediction)
    if ground_truth.ndim != 3 or prediction.ndim != 3:
        raise ValueError(f"Batched trajectories must have shape (batch, frames, {G1_FEATURE_DIM}).")
    sample_count = min(len(ground_truth), len(prediction))
    return [
        plot_joint_angle_comparison(
            ground_truth[index],
            prediction[index],
            output_dir,
            stem=f"sample{index:02d}",
            fps=fps,
            prediction_label=prediction_label,
        )
        for index in range(sample_count)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot G1 joint angles and prediction error.")
    parser.add_argument("--ground_truth", required=True, help="Ground-truth .npy or BONES-SEED .csv.")
    parser.add_argument("--prediction", required=True, help="Predicted .npy or BONES-SEED .csv.")
    parser.add_argument("--output_dir", required=True, help="Directory for PNG and per-joint CSV metrics.")
    parser.add_argument("--fps", type=float, default=20.0, help="Frame rate of both input trajectories.")
    parser.add_argument("--prediction_label", default="Generated", help="Label used in the plot legend.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path, table_path = plot_joint_angle_comparison(
        load_model_features(args.ground_truth),
        load_model_features(args.prediction),
        args.output_dir,
        fps=args.fps,
        prediction_label=args.prediction_label,
    )
    print(f"Wrote {image_path}")
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()

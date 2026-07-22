"""I/O helpers for BONES-SEED Unitree G1 CSV trajectories.

The BONES-SEED CSV files store root translation in centimeters and all
orientation / joint values in degrees.  MoMask trains on meters and radians,
then converts back to the original CSV convention for generated trajectories.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


G1_FRAME_COLUMN = "Frame"
G1_FEATURE_COLUMNS = (
    "root_translateX",
    "root_translateY",
    "root_translateZ",
    "root_rotateX",
    "root_rotateY",
    "root_rotateZ",
    "left_hip_pitch_joint_dof",
    "left_hip_roll_joint_dof",
    "left_hip_yaw_joint_dof",
    "left_knee_joint_dof",
    "left_ankle_pitch_joint_dof",
    "left_ankle_roll_joint_dof",
    "right_hip_pitch_joint_dof",
    "right_hip_roll_joint_dof",
    "right_hip_yaw_joint_dof",
    "right_knee_joint_dof",
    "right_ankle_pitch_joint_dof",
    "right_ankle_roll_joint_dof",
    "waist_yaw_joint_dof",
    "waist_roll_joint_dof",
    "waist_pitch_joint_dof",
    "left_shoulder_pitch_joint_dof",
    "left_shoulder_roll_joint_dof",
    "left_shoulder_yaw_joint_dof",
    "left_elbow_joint_dof",
    "left_wrist_roll_joint_dof",
    "left_wrist_pitch_joint_dof",
    "left_wrist_yaw_joint_dof",
    "right_shoulder_pitch_joint_dof",
    "right_shoulder_roll_joint_dof",
    "right_shoulder_yaw_joint_dof",
    "right_elbow_joint_dof",
    "right_wrist_roll_joint_dof",
    "right_wrist_pitch_joint_dof",
    "right_wrist_yaw_joint_dof",
)
G1_FEATURE_DIM = len(G1_FEATURE_COLUMNS)


def _validate_features(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != G1_FEATURE_DIM:
        raise ValueError(
            f"Expected a (frames, {G1_FEATURE_DIM}) G1 trajectory, got {features.shape}."
        )
    if len(features) == 0 or not np.isfinite(features).all():
        raise ValueError("G1 trajectory is empty or contains non-finite values.")
    return features


def read_g1_csv(path: str | Path) -> np.ndarray:
    """Read a BONES-SEED G1 CSV in its original centimeter/degree convention."""
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header")
        missing = [name for name in G1_FEATURE_COLUMNS if name not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path}: missing G1 columns: {', '.join(missing)}")

        rows = []
        for line_number, row in enumerate(reader, start=2):
            try:
                rows.append([float(row[name]) for name in G1_FEATURE_COLUMNS])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid numeric value") from exc
    return _validate_features(np.asarray(rows, dtype=np.float32))


def csv_values_to_model_features(
    csv_values: np.ndarray,
    root_position_scale: float = 0.01,
) -> np.ndarray:
    """Convert BONES-SEED centimeters/degrees into MoMask meters/radians."""
    if root_position_scale <= 0:
        raise ValueError("root_position_scale must be positive")
    features = _validate_features(csv_values).copy()
    features[:, :3] *= root_position_scale
    features[:, 3:] = np.deg2rad(features[:, 3:])
    return features.astype(np.float32)


def model_features_to_csv_values(
    features: np.ndarray,
    root_position_scale: float = 0.01,
) -> np.ndarray:
    """Convert model meters/radians back into the BONES-SEED CSV convention."""
    if root_position_scale <= 0:
        raise ValueError("root_position_scale must be positive")
    csv_values = _validate_features(features).astype(np.float64, copy=True)
    csv_values[:, :3] /= root_position_scale
    csv_values[:, 3:] = np.rad2deg(csv_values[:, 3:])
    return csv_values


def downsample_features(features: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    """Downsample a G1 trajectory without interpolating or changing its values."""
    features = _validate_features(features)
    if source_fps <= 0 or target_fps <= 0 or target_fps > source_fps:
        raise ValueError("FPS values must be positive and target_fps cannot exceed source_fps")

    ratio = source_fps / target_fps
    step = int(round(ratio))
    if not np.isclose(ratio, step):
        raise ValueError(
            f"source_fps/target_fps must be an integer for decimation, got {source_fps}/{target_fps}."
        )
    return features[::step].copy()


def resample_features(features: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    """Linearly resample model-space trajectories, unwrapping angular channels first."""
    features = _validate_features(features)
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("FPS values must be positive")
    if np.isclose(source_fps, target_fps) or len(features) == 1:
        return features.copy()

    duration = (len(features) - 1) / source_fps
    target_count = int(round(duration * target_fps)) + 1
    source_times = np.arange(len(features), dtype=np.float64) / source_fps
    target_times = np.arange(target_count, dtype=np.float64) / target_fps
    target_times[-1] = min(target_times[-1], duration)

    interpolated = features.astype(np.float64, copy=True)
    interpolated[:, 3:] = np.unwrap(interpolated[:, 3:], axis=0)
    output = np.empty((target_count, G1_FEATURE_DIM), dtype=np.float64)
    for feature_index in range(G1_FEATURE_DIM):
        output[:, feature_index] = np.interp(
            target_times, source_times, interpolated[:, feature_index]
        )
    return output.astype(np.float32)


def write_g1_csv(
    path: str | Path,
    model_features: np.ndarray,
    root_position_scale: float = 0.01,
) -> None:
    """Write model-space features as a MuJoCo-compatible BONES-SEED G1 CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    csv_values = model_features_to_csv_values(model_features, root_position_scale)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow((G1_FRAME_COLUMN, *G1_FEATURE_COLUMNS))
        for frame_index, values in enumerate(csv_values):
            writer.writerow([frame_index, *[f"{value:.9f}" for value in values]])

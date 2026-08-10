"""I/O helpers for BONES-SEED Unitree G1 trajectories.

CSV files retain BONES-SEED's centimetre/degree Euler convention.  MoMask
uses a 38-D model representation instead: root translation in metres, a
continuous 6-D root rotation, and the 29 single-DoF joint angles in radians.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


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
G1_CSV_FEATURE_DIM = len(G1_FEATURE_COLUMNS)
G1_ROOT_POSITION_DIM = 3
G1_ROOT_ROTATION_6D_DIM = 6
G1_JOINT_DIM = G1_CSV_FEATURE_DIM - 6
G1_ROOT_ROTATION_SLICE = slice(G1_ROOT_POSITION_DIM, G1_ROOT_POSITION_DIM + G1_ROOT_ROTATION_6D_DIM)
G1_JOINT_SLICE = slice(G1_ROOT_ROTATION_SLICE.stop, G1_ROOT_ROTATION_SLICE.stop + G1_JOINT_DIM)
G1_FEATURE_DIM = G1_JOINT_SLICE.stop


def _validate_model_features(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != G1_FEATURE_DIM:
        raise ValueError(
            f"Expected a (frames, {G1_FEATURE_DIM}) G1 trajectory, got {features.shape}."
        )
    if len(features) == 0 or not np.isfinite(features).all():
        raise ValueError("G1 trajectory is empty or contains non-finite values.")
    return features


def _validate_csv_values(csv_values: np.ndarray) -> np.ndarray:
    csv_values = np.asarray(csv_values, dtype=np.float32)
    if csv_values.ndim != 2 or csv_values.shape[1] != G1_CSV_FEATURE_DIM:
        raise ValueError(
            f"Expected a (frames, {G1_CSV_FEATURE_DIM}) G1 CSV trajectory, got {csv_values.shape}."
        )
    if len(csv_values) == 0 or not np.isfinite(csv_values).all():
        raise ValueError("G1 CSV trajectory is empty or contains non-finite values.")
    return csv_values


def rotation_matrix_to_6d(rotation_matrices: np.ndarray) -> np.ndarray:
    """Encode rotation matrices by their first two columns (Zhou et al. 6-D)."""
    matrices = np.asarray(rotation_matrices, dtype=np.float64)
    if matrices.ndim < 2 or matrices.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotation matrices ending in (3, 3), got {matrices.shape}")
    return np.concatenate((matrices[..., :, 0], matrices[..., :, 1]), axis=-1).astype(np.float32)


def rotation_6d_to_matrix(rotation_6d: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Decode 6-D rotations with Gram--Schmidt orthogonalization.

    Model outputs are not guaranteed to be orthonormal.  Degenerate vectors
    use a deterministic perpendicular fallback so visualization remains safe.
    """
    values = np.asarray(rotation_6d, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != G1_ROOT_ROTATION_6D_DIM:
        raise ValueError(f"Expected rotation values ending in ({G1_ROOT_ROTATION_6D_DIM},), got {values.shape}")

    first = values[..., :3]
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    fallback_first = np.zeros_like(first)
    fallback_first[..., 0] = 1.0
    first = np.where(first_norm > eps, first / np.maximum(first_norm, eps), fallback_first)

    second = values[..., 3:] - np.sum(values[..., 3:] * first, axis=-1, keepdims=True) * first
    second_norm = np.linalg.norm(second, axis=-1, keepdims=True)
    least_aligned_axis = np.argmin(np.abs(first), axis=-1)
    reference = np.zeros_like(first)
    np.put_along_axis(reference, least_aligned_axis[..., None], 1.0, axis=-1)
    fallback_second = np.cross(first, reference)
    second = np.where(second_norm > eps, second / np.maximum(second_norm, eps), fallback_second)
    second /= np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), eps)

    third = np.cross(first, second)
    return np.stack((first, second, third), axis=-1)


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
    return _validate_csv_values(np.asarray(rows, dtype=np.float32))


def csv_values_to_model_features(
    csv_values: np.ndarray,
    root_position_scale: float = 0.01,
    root_euler_order: str = "xyz",
) -> np.ndarray:
    """Convert BONES-SEED CSV values into 38-D MoMask model features."""
    if root_position_scale <= 0:
        raise ValueError("root_position_scale must be positive")
    csv_values = _validate_csv_values(csv_values)
    root_rotation = Rotation.from_euler(root_euler_order, np.deg2rad(csv_values[:, 3:6])).as_matrix()
    features = np.empty((len(csv_values), G1_FEATURE_DIM), dtype=np.float32)
    features[:, :G1_ROOT_POSITION_DIM] = csv_values[:, :G1_ROOT_POSITION_DIM] * root_position_scale
    features[:, G1_ROOT_ROTATION_SLICE] = rotation_matrix_to_6d(root_rotation)
    features[:, G1_JOINT_SLICE] = np.deg2rad(csv_values[:, 6:])
    return features


def model_features_to_csv_values(
    features: np.ndarray,
    root_position_scale: float = 0.01,
    root_euler_order: str = "xyz",
) -> np.ndarray:
    """Convert 38-D MoMask model features back into BONES-SEED CSV values."""
    if root_position_scale <= 0:
        raise ValueError("root_position_scale must be positive")
    features = _validate_model_features(features)
    csv_values = np.empty((len(features), G1_CSV_FEATURE_DIM), dtype=np.float64)
    csv_values[:, :G1_ROOT_POSITION_DIM] = features[:, :G1_ROOT_POSITION_DIM] / root_position_scale
    root_rotation = Rotation.from_matrix(rotation_6d_to_matrix(features[:, G1_ROOT_ROTATION_SLICE]))
    csv_values[:, 3:6] = np.rad2deg(root_rotation.as_euler(root_euler_order))
    csv_values[:, 6:] = np.rad2deg(features[:, G1_JOINT_SLICE])
    return _validate_csv_values(csv_values)


def downsample_features(features: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    """Downsample a G1 trajectory without interpolating or changing its values."""
    features = _validate_model_features(features)
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
    """Resample positions/joints linearly and root orientation with SLERP."""
    features = _validate_model_features(features)
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("FPS values must be positive")
    if np.isclose(source_fps, target_fps) or len(features) == 1:
        return features.copy()

    duration = (len(features) - 1) / source_fps
    target_count = int(round(duration * target_fps)) + 1
    source_times = np.arange(len(features), dtype=np.float64) / source_fps
    target_times = np.arange(target_count, dtype=np.float64) / target_fps
    target_times[-1] = min(target_times[-1], duration)

    output = np.empty((target_count, G1_FEATURE_DIM), dtype=np.float64)
    for feature_index in range(G1_ROOT_POSITION_DIM):
        output[:, feature_index] = np.interp(
            target_times, source_times, features[:, feature_index]
        )
    root_rotation = Rotation.from_matrix(rotation_6d_to_matrix(features[:, G1_ROOT_ROTATION_SLICE]))
    output[:, G1_ROOT_ROTATION_SLICE] = rotation_matrix_to_6d(Slerp(source_times, root_rotation)(target_times).as_matrix())
    joint_values = np.unwrap(features[:, G1_JOINT_SLICE].astype(np.float64), axis=0)
    for feature_index in range(G1_JOINT_DIM):
        output[:, G1_JOINT_SLICE.start + feature_index] = np.interp(
            target_times, source_times, joint_values[:, feature_index]
        )
    return output.astype(np.float32)


def write_g1_csv(
    path: str | Path,
    model_features: np.ndarray,
    root_position_scale: float = 0.01,
    root_euler_order: str = "xyz",
) -> None:
    """Write model-space features as a MuJoCo-compatible BONES-SEED G1 CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    csv_values = model_features_to_csv_values(model_features, root_position_scale, root_euler_order)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow((G1_FRAME_COLUMN, *G1_FEATURE_COLUMNS))
        for frame_index, values in enumerate(csv_values):
            writer.writerow([frame_index, *[f"{value:.9f}" for value in values]])

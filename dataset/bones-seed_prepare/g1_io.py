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

G1_MIXED_ROOT_YAW_VEL_DIM = 1
G1_MIXED_ROOT_XY_VEL_BODY_DIM = 2
G1_MIXED_ROOT_HEIGHT_DIM = 1
G1_MIXED_ROOT_ROLL_PITCH_6D_DIM = 6
G1_MIXED_ROOT_ABS_XYZ_DIM = 3

G1_MIXED_ROOT_YAW_VEL_SLICE = slice(0, 1)
G1_MIXED_ROOT_XY_VEL_BODY_SLICE = slice(1, 3)
G1_MIXED_ROOT_HEIGHT_SLICE = slice(3, 4)
G1_MIXED_ROOT_ROLL_PITCH_6D_SLICE = slice(4, 10)
G1_MIXED_ROOT_ABS_XYZ_SLICE = slice(10, 13)
G1_MIXED_ROOT_JOINT_SLICE = slice(13, 42)
G1_MIXED_ROOT_FEATURE_DIM = G1_MIXED_ROOT_JOINT_SLICE.stop
G1_MIXED_ROOT_FEATURE_NAMES = (
    ["root_yaw_vel_rad", "root_x_vel_body_m", "root_y_vel_body_m", "root_height_m"]
    + [f"root_roll_pitch_rot6d_{idx}" for idx in range(G1_MIXED_ROOT_ROLL_PITCH_6D_DIM)]
    + ["root_abs_x_m", "root_abs_y_m", "root_abs_z_m"]
    + list(G1_FEATURE_COLUMNS[6:])
)


G1_BANK_LAYOUT = {}
_bank_offset = 0
for _name, _width in (("joint_angles", G1_JOINT_DIM), ("joint_vel", G1_JOINT_DIM),
                     ("root_roll_pitch_rot6d", 6), ("root_roll_pitch_sincos", 4),
                     ("root_yaw_vel", 1), ("root_xy_vel_body", 2), ("root_height", 1)):
    G1_BANK_LAYOUT[_name] = [_bank_offset, _bank_offset + _width]
    _bank_offset += _width
G1_BANK_DIM = _bank_offset

G1_MOTION_AE_EXTENSION_LAYOUT = {
    "joint_velocity": [0, 29],
    "root_rotation_anchor6d": [29, 35],
    "root_linear_velocity_body": [35, 38],
    "root_angular_velocity_body": [38, 41],
}


def extract_motion_ae_heading(rotation: np.ndarray) -> np.ndarray:
    """motion_ae yaw_quat uses quaternion w/z projection, not Euler yaw."""
    quat = Rotation.from_matrix(rotation).as_quat()
    if np.any(np.hypot(quat[..., 2], quat[..., 3]) < 1e-8):
        raise ValueError('motion_ae heading is singular for this inverted root pose')
    return np.unwrap(2 * np.arctan2(quat[..., 2], quat[..., 3]), axis=0)


def model_features_to_motion_ae_extension(features: np.ndarray, fps: float) -> np.ndarray:
    """Per-second backward velocities, expressed in the current yaw-only frame."""
    features = _validate_model_features(features)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("FPS must be finite and positive")
    rotation = rotation_6d_to_matrix(features[:, G1_ROOT_ROTATION_SLICE])
    heading_inv = np.swapaxes(yaw_to_matrix(extract_motion_ae_heading(rotation)), -1, -2)
    xyz = features[:, :3].astype(np.float64)
    joints = np.unwrap(features[:, G1_JOINT_SLICE].astype(np.float64), axis=0)
    linear = np.diff(xyz, axis=0, prepend=xyz[:1]) * fps
    angular = np.zeros_like(linear)
    if len(features) > 1:
        angular[1:] = Rotation.from_matrix(rotation[1:] @ np.swapaxes(rotation[:-1], -1, -2)).as_rotvec() * fps
    return np.concatenate((np.diff(joints, axis=0, prepend=joints[:1]) * fps,
                           rotation_matrix_to_6d(heading_inv @ rotation),
                           np.einsum('tij,tj->ti', heading_inv, linear),
                           np.einsum('tij,tj->ti', heading_inv, angular)), axis=1).astype(np.float32)


def motion_ae_features_to_model_features(features: np.ndarray, anchor: np.ndarray,
                                       fps: float) -> np.ndarray:
    """Decode 70D using predicted tilt and angular increments, not omega-z as yaw rate."""
    features = np.asarray(features, dtype=np.float64)
    anchor = np.asarray(anchor, dtype=np.float64)
    if (features.ndim != 2 or features.shape[1] != 70 or not len(features)
            or not np.isfinite(features).all() or anchor.shape != (4,)
            or not np.isfinite(anchor).all() or not np.isfinite(fps) or fps <= 0):
        raise ValueError("Invalid motion_ae features, initial state or FPS")
    tilt = rotation_6d_to_matrix(features[:, 58:64])
    # Project predicted tilt back to yaw-free rotations before heading recovery.
    tilt = np.swapaxes(yaw_to_matrix(extract_motion_ae_heading(tilt)), -1, -2) @ tilt
    previous_in_current = Rotation.from_rotvec(-features[:, 67:70] / fps).as_matrix() @ tilt
    heading_delta = -extract_motion_ae_heading(previous_in_current)
    heading_delta = (heading_delta + np.pi) % (2 * np.pi) - np.pi
    yaw = np.full(len(features), anchor[3])
    yaw[1:] += np.cumsum(heading_delta[1:])
    heading = yaw_to_matrix(yaw)
    world_delta = np.einsum('tij,tj->ti', heading, features[:, 64:67]) / fps
    xyz = np.repeat(anchor[None, :3], len(features), axis=0)
    xyz[1:] += np.cumsum(world_delta[1:], axis=0)
    return np.concatenate((xyz, rotation_matrix_to_6d(heading @ tilt), features[:, :29]), axis=1).astype(np.float32)


def model_features_to_feature_bank(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Native poses -> bank and per-frame [XYZ, yaw] anchors.

    Deltas use the preceding frame and current heading. Compute before slicing.
    """
    features = _validate_model_features(features)
    rotation = rotation_6d_to_matrix(features[:, G1_ROOT_ROTATION_SLICE])
    yaw = extract_root_yaw(rotation)
    rp_matrix = np.swapaxes(yaw_to_matrix(yaw), -1, -2) @ rotation
    roll = np.arctan2(rp_matrix[:, 2, 1], rp_matrix[:, 2, 2])
    pitch = np.arctan2(-rp_matrix[:, 2, 0], np.hypot(rp_matrix[:, 0, 0], rp_matrix[:, 1, 0]))
    xyz = features[:, :3].astype(np.float64)
    delta = np.diff(xyz[:, :2], axis=0, prepend=xyz[:1, :2])
    c, s = np.cos(yaw), np.sin(yaw)
    xy_body = np.column_stack((c * delta[:, 0] + s * delta[:, 1],
                              -s * delta[:, 0] + c * delta[:, 1]))
    joints = features[:, G1_JOINT_SLICE]
    unwrapped = np.unwrap(joints.astype(np.float64), axis=0)
    bank = np.concatenate((joints, np.diff(unwrapped, axis=0, prepend=unwrapped[:1]),
                           rotation_matrix_to_6d(rp_matrix),
                           np.column_stack((np.sin(roll), np.cos(roll) - 1,
                                            np.sin(pitch), np.cos(pitch) - 1)),
                           np.diff(yaw, prepend=yaw[0])[:, None], xy_body, xyz[:, 2:3]), axis=1)
    return bank.astype(np.float32), np.column_stack((xyz, yaw)).astype(np.float64)


def local_features_to_model_features(features: np.ndarray, fields: list[str],
                                    anchor: np.ndarray) -> np.ndarray:
    """Decode with first-frame [XYZ, yaw]; ignore the first delta and joint deltas."""
    import warnings

    groups = {}
    offset = 0
    for name in fields:
        start, end = G1_BANK_LAYOUT[name]
        groups[name] = features[:, offset:offset + end - start]
        offset += end - start
    if features.ndim != 2 or features.shape[1] != offset or not len(features) or not np.isfinite(features).all():
        raise ValueError("Invalid local G1 features")
    anchor = np.asarray(anchor, dtype=np.float64)
    if anchor.shape != (4,) or not np.isfinite(anchor).all():
        raise ValueError("Expected finite initial [x, y, z, yaw]")
    yaw = np.full(len(features), anchor[3], dtype=np.float64)
    yaw[1:] += np.cumsum(groups["root_yaw_vel"][1:, 0], dtype=np.float64)
    delta = groups["root_xy_vel_body"]
    c, s = np.cos(yaw), np.sin(yaw)
    world = np.column_stack((c * delta[:, 0] - s * delta[:, 1],
                             s * delta[:, 0] + c * delta[:, 1]))
    xyz = np.zeros((len(features), 3), dtype=np.float64)
    xyz[:, :2] = anchor[:2]
    xyz[1:, :2] += np.cumsum(world[1:], axis=0)
    xyz[:, 2] = groups["root_height"][:, 0]
    if "root_roll_pitch_rot6d" in groups:
        rp = rotation_6d_to_matrix(groups["root_roll_pitch_rot6d"])
    else:
        pairs = groups["root_roll_pitch_sincos"].reshape(-1, 2, 2).astype(np.float64).copy()
        pairs[:, :, 1] += 1
        bad = np.linalg.norm(pairs, axis=-1) < 1e-8
        if bad.any():
            warnings.warn(f"Zero-angle fallback for {int(bad.sum())} degenerate sincos pairs")
            pairs[bad] = [0, 1]
        angles = np.arctan2(pairs[:, :, 0], pairs[:, :, 1])
        rp = Rotation.from_euler("xyz", np.column_stack((angles, np.zeros(len(features))))).as_matrix()
    return np.concatenate((xyz, rotation_matrix_to_6d(yaw_to_matrix(yaw) @ rp),
                           groups["joint_angles"]), axis=1).astype(np.float32)


def _validate_model_features(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != G1_FEATURE_DIM:
        raise ValueError(
            f"Expected a (frames, {G1_FEATURE_DIM}) G1 trajectory, got {features.shape}."
        )
    if len(features) == 0 or not np.isfinite(features).all():
        raise ValueError("G1 trajectory is empty or contains non-finite values.")
    return features


def _validate_mixed_root_features(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != G1_MIXED_ROOT_FEATURE_DIM:
        raise ValueError(
            f"Expected a (frames, {G1_MIXED_ROOT_FEATURE_DIM}) mixed-root G1 trajectory, got {features.shape}."
        )
    if len(features) == 0 or not np.isfinite(features).all():
        raise ValueError("Mixed-root G1 trajectory is empty or contains non-finite values.")
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


def yaw_to_matrix(yaw: np.ndarray | float) -> np.ndarray:
    """Build world-Z yaw rotation matrices."""
    yaw_values = np.asarray(yaw, dtype=np.float64)
    cos_yaw = np.cos(yaw_values)
    sin_yaw = np.sin(yaw_values)
    matrices = np.zeros(yaw_values.shape + (3, 3), dtype=np.float64)
    matrices[..., 0, 0] = cos_yaw
    matrices[..., 0, 1] = -sin_yaw
    matrices[..., 1, 0] = sin_yaw
    matrices[..., 1, 1] = cos_yaw
    matrices[..., 2, 2] = 1.0
    return matrices


def extract_root_yaw(root_rotation_matrices: np.ndarray) -> np.ndarray:
    """Extract continuous heading yaw from the root rotation's forward axis."""
    matrices = np.asarray(root_rotation_matrices, dtype=np.float64)
    if matrices.ndim < 2 or matrices.shape[-2:] != (3, 3):
        raise ValueError(f"Expected rotation matrices ending in (3, 3), got {matrices.shape}")
    return np.unwrap(np.arctan2(matrices[..., 1, 0], matrices[..., 0, 0]), axis=0)


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


def _velocity_storage_scale(target_fps: float, velocity_unit: str) -> float:
    if target_fps <= 0:
        raise ValueError("target_fps must be positive")
    if velocity_unit == "per_frame":
        return 1.0
    if velocity_unit == "per_second":
        return float(target_fps)
    raise ValueError("velocity_unit must be 'per_frame' or 'per_second'")


def model_features_to_mixed_root_features(
    features: np.ndarray,
    target_fps: float = 20.0,
    velocity_unit: str = "per_frame",
    return_metadata: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Convert 38-D native G1 features to the mixed-root training layout.

    The mixed-root layout keeps heading and planar translation as local frame
    deltas while also storing absolute XYZ anchors:

    [root_yaw_vel, root_xy_vel_body, root_height, root_roll_pitch_rot6d,
     root_abs_xyz, 29 joint angles]
    """
    features = _validate_model_features(features)
    velocity_scale = _velocity_storage_scale(target_fps, velocity_unit)

    root_xyz = features[:, :G1_ROOT_POSITION_DIM].astype(np.float64)
    root_rotation = rotation_6d_to_matrix(features[:, G1_ROOT_ROTATION_SLICE])
    root_yaw = extract_root_yaw(root_rotation)

    yaw_delta = np.zeros(len(features), dtype=np.float64)
    if len(features) > 1:
        yaw_delta[1:] = np.diff(root_yaw)

    world_delta_xy = np.zeros((len(features), 2), dtype=np.float64)
    if len(features) > 1:
        world_delta_xy[1:] = np.diff(root_xyz[:, :2], axis=0)
    previous_yaw = np.concatenate(([root_yaw[0]], root_yaw[:-1]))
    cos_yaw = np.cos(previous_yaw)
    sin_yaw = np.sin(previous_yaw)
    body_delta_xy = np.empty_like(world_delta_xy)
    body_delta_xy[:, 0] = cos_yaw * world_delta_xy[:, 0] + sin_yaw * world_delta_xy[:, 1]
    body_delta_xy[:, 1] = -sin_yaw * world_delta_xy[:, 0] + cos_yaw * world_delta_xy[:, 1]

    yaw_rotation = yaw_to_matrix(root_yaw)
    roll_pitch_rotation = np.matmul(np.swapaxes(yaw_rotation, -1, -2), root_rotation)

    mixed = np.empty((len(features), G1_MIXED_ROOT_FEATURE_DIM), dtype=np.float32)
    mixed[:, G1_MIXED_ROOT_YAW_VEL_SLICE] = (yaw_delta[:, None] * velocity_scale).astype(np.float32)
    mixed[:, G1_MIXED_ROOT_XY_VEL_BODY_SLICE] = (body_delta_xy * velocity_scale).astype(np.float32)
    mixed[:, G1_MIXED_ROOT_HEIGHT_SLICE] = root_xyz[:, 2:3].astype(np.float32)
    mixed[:, G1_MIXED_ROOT_ROLL_PITCH_6D_SLICE] = rotation_matrix_to_6d(roll_pitch_rotation)
    mixed[:, G1_MIXED_ROOT_ABS_XYZ_SLICE] = root_xyz.astype(np.float32)
    mixed[:, G1_MIXED_ROOT_JOINT_SLICE] = features[:, G1_JOINT_SLICE]

    if not return_metadata:
        return mixed

    metadata = {
        "root_yaw0_rad": float(root_yaw[0]),
        "root_xyz0_m": [float(value) for value in root_xyz[0]],
        "root_yaw_last_rad": float(root_yaw[-1]),
        "root_xyz_last_m": [float(value) for value in root_xyz[-1]],
        "target_fps": float(target_fps),
        "velocity_unit": velocity_unit,
    }
    return mixed, metadata


def mixed_root_features_to_model_features(
    mixed_features: np.ndarray,
    root_yaw0: float = 0.0,
    root_xy0: np.ndarray | None = None,
    target_fps: float = 20.0,
    velocity_unit: str = "per_frame",
    use_absolute_position: bool = True,
) -> np.ndarray:
    """Convert mixed-root G1 features back to the 38-D native G1 layout.

    When ``use_absolute_position`` is true, absolute XY anchors are used for
    root translation to avoid integration drift.  The separate root_height
    channel remains canonical for Z.
    """
    mixed_features = _validate_mixed_root_features(mixed_features)
    velocity_scale = _velocity_storage_scale(target_fps, velocity_unit)

    yaw_delta = mixed_features[:, G1_MIXED_ROOT_YAW_VEL_SLICE.start] / velocity_scale
    root_yaw = np.empty(len(mixed_features), dtype=np.float64)
    root_yaw[0] = float(root_yaw0)
    if len(mixed_features) > 1:
        root_yaw[1:] = float(root_yaw0) + np.cumsum(yaw_delta[1:])

    root_xyz = np.empty((len(mixed_features), G1_ROOT_POSITION_DIM), dtype=np.float64)
    if use_absolute_position:
        root_xyz[:] = mixed_features[:, G1_MIXED_ROOT_ABS_XYZ_SLICE]
        root_xyz[:, 2] = mixed_features[:, G1_MIXED_ROOT_HEIGHT_SLICE.start]
    else:
        if root_xy0 is None:
            root_xy = mixed_features[0, G1_MIXED_ROOT_ABS_XYZ_SLICE.start:G1_MIXED_ROOT_ABS_XYZ_SLICE.start + 2]
        else:
            root_xy = np.asarray(root_xy0, dtype=np.float64)
            if root_xy.shape != (2,):
                raise ValueError(f"root_xy0 must have shape (2,), got {root_xy.shape}")
        root_xyz[0, :2] = root_xy
        root_xyz[0, 2] = mixed_features[0, G1_MIXED_ROOT_HEIGHT_SLICE.start]
        body_delta_xy = mixed_features[:, G1_MIXED_ROOT_XY_VEL_BODY_SLICE] / velocity_scale
        for frame_index in range(1, len(mixed_features)):
            cos_yaw = np.cos(root_yaw[frame_index - 1])
            sin_yaw = np.sin(root_yaw[frame_index - 1])
            local_x, local_y = body_delta_xy[frame_index]
            root_xyz[frame_index, 0] = root_xyz[frame_index - 1, 0] + cos_yaw * local_x - sin_yaw * local_y
            root_xyz[frame_index, 1] = root_xyz[frame_index - 1, 1] + sin_yaw * local_x + cos_yaw * local_y
            root_xyz[frame_index, 2] = mixed_features[frame_index, G1_MIXED_ROOT_HEIGHT_SLICE.start]

    roll_pitch_rotation = rotation_6d_to_matrix(mixed_features[:, G1_MIXED_ROOT_ROLL_PITCH_6D_SLICE])
    root_rotation = np.matmul(yaw_to_matrix(root_yaw), roll_pitch_rotation)

    features = np.empty((len(mixed_features), G1_FEATURE_DIM), dtype=np.float32)
    features[:, :G1_ROOT_POSITION_DIM] = root_xyz.astype(np.float32)
    features[:, G1_ROOT_ROTATION_SLICE] = rotation_matrix_to_6d(root_rotation)
    features[:, G1_JOINT_SLICE] = mixed_features[:, G1_MIXED_ROOT_JOINT_SLICE]
    return features


def is_mixed_root_features(features: np.ndarray) -> bool:
    values = np.asarray(features)
    return values.ndim == 2 and values.shape[1] == G1_MIXED_ROOT_FEATURE_DIM


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

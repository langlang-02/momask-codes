"""Physical-space G1 reconstruction metrics, independent of model features."""

import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dataset/bones-seed_prepare'))
from g1_io import (G1_BANK_LAYOUT, G1_FEATURE_COLUMNS, rotation_6d_to_matrix,
                   rotation_matrix_to_6d, yaw_to_matrix, extract_root_yaw)

JOINT_NAMES = tuple(c.removesuffix('_dof') for c in G1_FEATURE_COLUMNS[6:])
ENDPOINT_NAMES = ('left_wrist_yaw_joint', 'right_wrist_yaw_joint',
                  'left_ankle_roll_joint', 'right_ankle_roll_joint')


def ground_truth_native(bank, anchors):
    if bank.ndim != 2 or bank.shape[1] not in (72, 113) or anchors.shape != (len(bank), 4):
        raise ValueError('Invalid feature-bank trajectory or anchors')
    start, end = G1_BANK_LAYOUT['root_roll_pitch_rot6d']
    rotation = yaw_to_matrix(anchors[:, 3]) @ rotation_6d_to_matrix(bank[:, start:end])
    return np.concatenate((anchors[:, :3], rotation_matrix_to_6d(rotation), bank[:, :29]), axis=1)


class G1Kinematics:
    def __init__(self, mjcf):
        from g1_visualize.replay_g1 import qpos_addresses
        self.model = mujoco.MjModel.from_xml_path(str(mjcf))
        self.data = mujoco.MjData(self.model)
        self.root, self.addresses = qpos_addresses(self.model)
        self.ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
        self.endpoints = [JOINT_NAMES.index(n) for n in ENDPOINT_NAMES]

    def positions(self, native):
        rotations = rotation_6d_to_matrix(native[:, 3:9])
        quaternions = Rotation.from_matrix(rotations).as_quat()[:, [3, 0, 1, 2]]
        result = np.empty((len(native), len(self.ids), 3), dtype=np.float32)
        for i, frame in enumerate(native):
            self.data.qpos[:] = self.model.qpos0
            self.data.qpos[self.root:self.root + 3] = frame[:3]
            self.data.qpos[self.root + 3:self.root + 7] = quaternions[i]
            self.data.qpos[self.addresses] = frame[9:]
            # Only positions are needed: no contacts, dynamics, rendering or stepping.
            mujoco.mj_kinematics(self.model, self.data)
            result[i] = self.data.xanchor[self.ids]
        return result


def metric_values(gt, pred, gt_points, pred_points, fps, endpoints):
    if gt.shape != pred.shape or len(gt) < 3 or fps <= 0:
        raise ValueError('Metrics require matching trajectories of at least three frames')
    if not all(np.isfinite(a).all() for a in (gt, pred, gt_points, pred_points)):
        raise ValueError('Nonfinite reconstruction')
    position = np.linalg.norm(pred_points - gt_points, axis=-1) * 1000
    relative = np.linalg.norm((pred_points - pred[:, None, :3]) -
                              (gt_points - gt[:, None, :3]), axis=-1) * 1000
    root = np.linalg.norm(pred[:, :3] - gt[:, :3], axis=-1)
    gt_rotation = rotation_6d_to_matrix(gt[:, 3:9])
    pred_rotation = rotation_6d_to_matrix(pred[:, 3:9])
    angle = Rotation.from_matrix(np.swapaxes(gt_rotation, -1, -2) @ pred_rotation).magnitude()
    yaw_delta = extract_root_yaw(pred_rotation) - extract_root_yaw(gt_rotation)
    # These are limited hinge coordinates; do not wrap away joint-limit errors.
    joints = np.rad2deg(pred[:, 9:] - gt[:, 9:])
    gt_unwrapped, pred_unwrapped = (np.unwrap(x[:, 9:], axis=0) for x in (gt, pred))
    velocity = (np.diff(pred_unwrapped, axis=0) - np.diff(gt_unwrapped, axis=0)) * fps
    acceleration = (np.diff(pred_unwrapped, n=2, axis=0) - np.diff(gt_unwrapped, n=2, axis=0)) * fps ** 2
    values = dict(global_mpjpe_mm=position, root_relative_mpjpe_mm=relative,
                  root_ade_m=root, root_fde_m=root[-1:],
                  root_rotation_deg=np.rad2deg(angle),
                  root_yaw_deg=np.rad2deg(np.abs(np.arctan2(np.sin(yaw_delta), np.cos(yaw_delta)))),
                  joint_mae_deg=np.abs(joints), joint_rmse_deg=joints,
                  joint_velocity_rmse_rad_s=velocity,
                  joint_acceleration_rmse_rad_s2=acceleration)
    for label, index in zip(('left_wrist', 'right_wrist', 'left_ankle', 'right_ankle'), endpoints):
        values[label + '_position_mm'] = position[:, index]
    return values


def reduce_values(name, values):
    return float(np.sqrt(np.mean(np.square(values))) if 'rmse' in name else np.mean(values))

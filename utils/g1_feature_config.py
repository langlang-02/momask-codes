"""Helpers for resolving Unitree G1 feature layout from prepared datasets."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np


def resolve_representation(data_root, config_path):
    """Resolve an ordered feature view; never infer representation from dimension."""
    import yaml

    info = load_g1_dataset_info(data_root)
    with open(config_path) as stream:
        config = yaml.safe_load(stream)
    if config.get('extension'):
        return _resolve_motion_ae_extension(data_root, config, info)
    for key, actual in [('bank_format', info.get('format')), ('fps', info.get('target_fps')),
                        ('velocity_unit', info.get('velocity_unit')),
                        ('velocity_frame', info.get('velocity_frame'))]:
        if config.get(key) != actual:
            raise ValueError(f'Representation/dataset mismatch: {key}')
    if not info.get('complete'):
        raise ValueError('Feature bank conversion is incomplete')
    fields = config['fields']
    if not fields or len(set(fields)) != len(fields):
        raise ValueError('Representation fields must be nonempty and unique')
    required = {'joint_angles', 'root_yaw_vel', 'root_xy_vel_body', 'root_height'}
    if not required.issubset(fields) or sum(k in fields for k in ('root_roll_pitch_rot6d', 'root_roll_pitch_sincos')) != 1:
        raise ValueError('Local decoding requires joints, root motion, height and exactly one rotation representation')
    indices, layout = [], {}
    for name in fields:
        start, end = info['feature_layout'][name]
        layout[name] = [len(indices), len(indices) + end - start]
        indices.extend(range(start, end))
    config.update(indices=indices, feature_dim=len(indices), feature_layout=layout,
                  bank_feature_dim=info['feature_dim'])
    digest = hashlib.sha256()
    for name in ('Mean.npy', 'Std.npy', 'train.txt', 'val.txt', 'test.txt'):
        digest.update((Path(data_root) / name).read_bytes())
    digest.update(json.dumps(info, sort_keys=True).encode())
    config['dataset_signature'] = digest.hexdigest()
    return config


def feature_bank_signature(data_root):
    """Same fingerprint as existing v1 specs; extension files are excluded."""
    digest = hashlib.sha256()
    for name in ('Mean.npy', 'Std.npy', 'train.txt', 'val.txt', 'test.txt'):
        digest.update((Path(data_root) / name).read_bytes())
    digest.update(json.dumps(load_g1_dataset_info(data_root), sort_keys=True).encode())
    return digest.hexdigest()


def _resolve_motion_ae_extension(data_root, config, info):
    root = Path(data_root)
    extension = config['extension']
    if extension != 'extensions/motion_ae_v1':
        raise ValueError('Unsupported G1 extension')
    ext_root = root / extension
    ext = load_g1_dataset_info(ext_root)
    fields = ['joint_angles', 'joint_velocity', 'root_rotation_anchor6d',
              'root_linear_velocity_body', 'root_angular_velocity_body']
    extra_layout = {'joint_velocity': [0, 29], 'root_rotation_anchor6d': [29, 35],
                    'root_linear_velocity_body': [35, 38], 'root_angular_velocity_body': [38, 41]}
    if (config.get('fields') != fields or config.get('bank_format') != info.get('format')
            or info.get('format') != 'bones_g1_feature_bank_v1' or info.get('feature_dim') != 72
            or ext.get('format') != 'bones_g1_motion_ae_extension_v1'
            or ext.get('feature_dim') != 41 or ext.get('feature_layout') != extra_layout
            or not info.get('complete') or not ext.get('complete') or ext.get('partial')):
        raise ValueError('Invalid or incomplete motion_ae feature bank/extension')
    for key, expected in [('fps', 20), ('velocity_unit', 'per_second'),
                          ('velocity_frame', 'current_heading'),
                          ('heading_definition', 'quaternion_wz_projection')]:
        if config.get(key) != expected or ext.get('target_fps' if key == 'fps' else key) != expected:
            raise ValueError(f'Extension/config mismatch: {key}')
    if info.get('target_fps') != 20 or info.get('velocity_frame') != 'current_heading':
        raise ValueError('Incompatible base feature bank')
    base_signature = feature_bank_signature(root)
    if ext.get('base_signature') != base_signature:
        raise ValueError('Extension was built from a different base dataset')
    indices, layout = [], {}
    for field in fields:
        if field in extra_layout:
            start, end = extra_layout[field]
            start, end = start + 72, end + 72
        else:
            start, end = info['feature_layout'][field]
        layout[field] = [len(indices), len(indices) + end - start]
        indices.extend(range(start, end))
    digest = hashlib.sha256(base_signature.encode())
    digest.update(json.dumps(ext, sort_keys=True).encode())
    for name in ('Mean.npy', 'Std.npy'):
        digest.update((ext_root / name).read_bytes())
    config.update(indices=indices, feature_dim=70, feature_layout=layout, bank_feature_dim=72,
                  extension_feature_dim=41, dataset_signature=digest.hexdigest())
    return config


def load_representation_stats(data_root, representation):
    root = Path(data_root)
    values = []
    for name in ('Mean', 'Std'):
        array = np.load(root / f'{name}.npy')
        if representation.get('extension'):
            extra = np.load(root / representation['extension'] / f'{name}.npy')
            if extra.shape != (representation['extension_feature_dim'],):
                raise ValueError('Invalid extension normalization shape')
            array = np.concatenate((array, extra))
        values.append(array[representation['indices']])
    if not all(np.isfinite(x).all() for x in values) or (values[1] <= 0).any():
        raise ValueError('Invalid representation normalization values')
    return tuple(values)


def select_representation_features(base, data_root, sample, representation):
    if base.ndim != 2 or base.shape[1] not in (representation['bank_feature_dim'], 113):
        raise ValueError(f'Invalid feature-bank shape: {sample} {base.shape}')
    if representation.get('extension') and base.shape[1] != 113:
        extra = np.load(Path(data_root) / representation['extension'] / 'features' / f'{sample}.npy')
        if extra.shape != (len(base), representation['extension_feature_dim']):
            raise ValueError(f'Extension shape/frame mismatch: {sample} {extra.shape}')
        base = np.concatenate((base, extra), axis=1)
    view = np.ascontiguousarray(base[:, representation['indices']])
    if not np.isfinite(view).all():
        raise ValueError(f'Nonfinite feature-bank sample: {sample}')
    return view


def representation_motion_dir(data_root, representation):
    """Storage packing must not change the logical checkpoint representation."""
    root = Path(data_root)
    marker = root / 'merged_storage.json'
    if not marker.exists():
        return root / 'new_joint_vecs'
    storage = json.loads(marker.read_text())
    if (storage.get('format') != 'bones_g1_merged_storage_v1'
            or not storage.get('complete') or storage.get('feature_dim') != 113
            or storage.get('base_signature') != feature_bank_signature(root)
            or storage.get('extension_sha256') != motion_ae_storage_signature(root)):
        raise ValueError('Invalid or stale merged feature storage')
    if representation['bank_feature_dim'] != 72:
        raise ValueError('Merged storage requires the v1 feature bank')
    directory = root / 'merged_joint_vecs'
    if not directory.is_dir():
        raise ValueError('Merged feature storage is missing')
    return directory


def motion_ae_storage_signature(data_root):
    root = Path(data_root) / 'extensions/motion_ae_v1'
    digest = hashlib.sha256()
    for name in ('dataset_info.json', 'Mean.npy', 'Std.npy', 'Min.npy', 'Max.npy'):
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def load_g1_dataset_info(data_root: str | Path) -> dict[str, object]:
    info_path = Path(data_root) / "dataset_info.json"
    if not info_path.exists():
        return {}
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    if not isinstance(info, dict):
        raise ValueError(f"{info_path} must contain a JSON object")
    return info


def resolve_g1_feature_dim(data_root: str | Path, fallback: int = 38) -> int:
    root = Path(data_root)
    info = load_g1_dataset_info(root)
    if "feature_dim" in info:
        feature_dim = int(info["feature_dim"])
        if feature_dim <= 0:
            raise ValueError(f"{root / 'dataset_info.json'} has invalid feature_dim={feature_dim}")
        return feature_dim

    mean_path = root / "Mean.npy"
    if mean_path.exists():
        mean = np.load(mean_path, mmap_mode="r")
        if mean.ndim == 1 and mean.shape[0] > 0:
            return int(mean.shape[0])
        raise ValueError(f"{mean_path} has invalid shape {mean.shape}")

    return fallback

"""Run with .venv/bin/python -m unittest discover -s tests -p test_g1_feature_bank.py."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import warnings

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dataset' / 'bones-seed_prepare'))
from g1_io import (G1_BANK_LAYOUT, G1_BANK_DIM, G1_FEATURE_DIM, G1_ROOT_ROTATION_SLICE,
                   G1_JOINT_SLICE, model_features_to_feature_bank,
                   local_features_to_model_features, rotation_matrix_to_6d)
from utils.g1_feature_config import resolve_representation


class FeatureBankTest(unittest.TestCase):
    def test_round_trip_and_window(self):
        for n in (1, 100):
            x = np.zeros((n, G1_FEATURE_DIM), dtype=np.float32)
            x[:, :3] = np.column_stack((np.arange(n) * .03, np.sin(np.arange(n) / 10), np.ones(n)))
            euler = np.column_stack((.2 * np.sin(np.arange(n) / 10),
                                    .3 * np.cos(np.arange(n) / 10), np.linspace(2, 5, n)))
            x[:, G1_ROOT_ROTATION_SLICE] = rotation_matrix_to_6d(Rotation.from_euler('xyz', euler).as_matrix())
            x[:, G1_JOINT_SLICE.start] = (np.linspace(2, 5, n) + np.pi) % (2 * np.pi) - np.pi
            bank, anchors = model_features_to_feature_bank(x)
            for rotation in ('root_roll_pitch_rot6d', 'root_roll_pitch_sincos'):
                fields = ['joint_angles', 'joint_vel', rotation, 'root_yaw_vel', 'root_xy_vel_body', 'root_height']
                indices = [i for k in fields for i in range(*G1_BANK_LAYOUT[k])]
                for start in (0, n // 2):
                    result = local_features_to_model_features(bank[start:, indices], fields, anchors[start])
                    np.testing.assert_allclose(result, x[start:], atol=2e-6)
            self.assertTrue(np.all(bank[0, 29:58] == 0))
            if n > 1:
                np.testing.assert_allclose(bank[1:, 29], 3 / (n - 1), atol=1e-6)
            # Current-heading delta rotated back must equal the world delta.
            a, b = G1_BANK_LAYOUT['root_xy_vel_body']
            d = bank[:, a:b]
            c, s = np.cos(anchors[:, 3]), np.sin(anchors[:, 3])
            world = np.column_stack((c*d[:, 0]-s*d[:, 1], s*d[:, 0]+c*d[:, 1]))
            np.testing.assert_allclose(world[1:], np.diff(x[:, :2], axis=0), atol=1e-6)

    def test_neutral_and_degenerate_sincos(self):
        x = np.zeros((2, G1_FEATURE_DIM), np.float32)
        x[:, G1_ROOT_ROTATION_SLICE] = rotation_matrix_to_6d(np.repeat(np.eye(3)[None], 2, axis=0))
        bank, anchors = model_features_to_feature_bank(x)
        np.testing.assert_array_equal(bank[:, 64:68], 0)
        fields = [k for k in G1_BANK_LAYOUT if k != 'root_roll_pitch_rot6d']
        indices = [i for k in fields for i in range(*G1_BANK_LAYOUT[k])]
        bank[:, 65] = -1
        with warnings.catch_warnings(record=True) as messages:
            y = local_features_to_model_features(bank[:, indices], fields, anchors[0])
        self.assertEqual(len(messages), 1)
        np.testing.assert_allclose(y, x)

    def test_config_and_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            info = dict(format='bones_g1_feature_bank_v1', complete=True, target_fps=20,
                        velocity_unit='per_frame', velocity_frame='current_heading',
                        feature_dim=G1_BANK_DIM, feature_layout=G1_BANK_LAYOUT)
            (root / 'dataset_info.json').write_text(json.dumps(info))
            for name in ('Mean', 'Std'):
                np.save(root / f'{name}.npy', np.arange(G1_BANK_DIM, dtype=np.float32) + 1)
            for split in ('train', 'val', 'test'):
                (root / f'{split}.txt').write_text('sample\n')
            for name, dim in [('rot6d68', 68), ('sincos66', 66)]:
                config = f'configs/representations/g1_local_{name}_v1.yaml'
                spec = resolve_representation(root, config)
                self.assertEqual(spec['feature_dim'], dim)
                self.assertEqual(len(set(spec['indices'])), dim)
                full = np.arange(3 * G1_BANK_DIM).reshape(3, G1_BANK_DIM)
                mean = np.load(root / 'Mean.npy')
                std = np.load(root / 'Std.npy')
                idx = spec['indices']
                np.testing.assert_allclose(((full - mean) / std)[:, idx], (full[:, idx]-mean[idx])/std[idx])
            info['velocity_frame'] = 'previous_heading'
            (root / 'dataset_info.json').write_text(json.dumps(info))
            with self.assertRaises(ValueError):
                resolve_representation(root, config)


if __name__ == '__main__':
    unittest.main()

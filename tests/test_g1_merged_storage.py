import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dataset/bones-seed_prepare'))
import g1_io
from merge_g1_feature_bank import merge
from utils.g1_feature_config import (
    resolve_representation, load_representation_stats, representation_motion_dir,
    select_representation_features,
)


class MergedStorageTest(unittest.TestCase):
    def test_merge_preserves_views_stats_and_reconstruction(self):
        from data.t2m_dataset import MotionDataset
        from g1_visualize.reconstruct_g1_rvq import load_motion, representation_input_features

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'new_joint_vecs').mkdir()
            (root / 'anchors').mkdir()
            ext = root / 'extensions/motion_ae_v1'
            (ext / 'features').mkdir(parents=True)
            info = dict(format='bones_g1_feature_bank_v1', complete=True, target_fps=20,
                        velocity_unit='per_frame', velocity_frame='current_heading',
                        feature_dim=72, feature_layout=g1_io.G1_BANK_LAYOUT)
            (root / 'dataset_info.json').write_text(json.dumps(info))
            for split in ('train', 'val', 'test'):
                (root / f'{split}.txt').write_text(split + '\n')
            for name in ('Mean', 'Std', 'Min', 'Max'):
                np.save(root / f'{name}.npy', np.ones(72, np.float32))
                np.save(ext / f'{name}.npy', np.ones(41, np.float32))
            configs = [f'configs/representations/{name}.yaml' for name in
                       ('g1_local_rot6d68_v1', 'g1_local_sincos66_v1', 'g1_motion_ae70_v1')]
            old = resolve_representation(root, configs[0])
            extra_info = dict(format='bones_g1_motion_ae_extension_v1', complete=True,
                              partial=False, base_signature=old['dataset_signature'], target_fps=20,
                              velocity_unit='per_second', velocity_frame='current_heading',
                              heading_definition='quaternion_wz_projection', feature_dim=41,
                              feature_layout=g1_io.G1_MOTION_AE_EXTENSION_LAYOUT)
            (ext / 'dataset_info.json').write_text(json.dumps(extra_info))
            pose = np.zeros((12, 38), np.float32)
            pose[:, 3:9] = [1, 0, 0, 0, 1, 0]
            pose[:, 0] = np.arange(12) * .02
            base, anchors = g1_io.model_features_to_feature_bank(pose)
            extra = g1_io.model_features_to_motion_ae_extension(pose, 20)
            for sample in ('train', 'val', 'test'):
                np.save(root / 'new_joint_vecs' / f'{sample}.npy', base[3:])
                np.save(root / 'anchors' / f'{sample}.npy', anchors[3:])
                np.save(ext / 'features' / f'{sample}.npy', extra[3:])
            specs = [resolve_representation(root, c) for c in configs]
            stats = [load_representation_stats(root, s) for s in specs]
            expected = [select_representation_features(base[3:], root, 'train', s) for s in specs]
            self.assertEqual(representation_motion_dir(root, specs[0]), root / 'new_joint_vecs')
            with patch.object(Path, 'rename', side_effect=OSError('EFS directory rename unavailable')):
                report = merge(root, 2)
            self.assertEqual(report['samples'], 3)
            self.assertTrue((root / 'merged_joint_vecs').is_symlink())
            self.assertFalse((root / 'merged_joint_vecs').readlink().is_absolute())
            # No sidecar trajectory may be opened after publication.
            with patch('utils.g1_feature_config.np.load', side_effect=AssertionError('sidecar read')):
                for spec, values in zip(specs, expected):
                    np.testing.assert_array_equal(
                        select_representation_features(np.concatenate((base[3:], extra[3:]), axis=1),
                                                       root, 'train', spec), values)
            for config, spec, stat, values in zip(configs, specs, stats, expected):
                self.assertEqual(resolve_representation(root, config), spec)
                for a, b in zip(load_representation_stats(root, spec), stat):
                    np.testing.assert_array_equal(a, b)
                path = representation_motion_dir(root, spec) / 'train.npy'
                native, fps = load_motion(path, .01, 'xyz')
                resolved_native, _ = load_motion(path.resolve(), .01, 'xyz')
                np.testing.assert_array_equal(native, resolved_native)
                actual, _ = representation_input_features(native, spec, path, fps, fps)
                np.testing.assert_array_equal(actual, values)
                opt = SimpleNamespace(joints_num=29, representation=spec, data_root=root,
                                      motion_dir=root/'new_joint_vecs', window_size=4, is_train=False)
                dataset = MotionDataset(opt, *stat, root/'train.txt')
                np.testing.assert_array_equal(dataset.data[0], values)
            with self.assertRaises(FileExistsError):
                merge(root, 2)
            np.save(ext / 'Std.npy', np.ones(41, np.float32) * 2)
            with self.assertRaises(ValueError):
                representation_motion_dir(root, specs[0])


if __name__ == '__main__':
    unittest.main()

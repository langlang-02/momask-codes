import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'dataset' / 'bones-seed_prepare'))
import g1_io
from utils import g1_feature_config as config


class MotionAEExtensionTest(unittest.TestCase):
    def test_bank_window_reconstruction_keeps_boundary_velocities(self):
        from g1_visualize import reconstruct_g1_rvq as reconstruction
        self.assertTrue(hasattr(reconstruction, 'representation_input_features'))
        pose = np.zeros((20, 38), np.float32)
        pose[:, 3:9] = [1, 0, 0, 0, 1, 0]
        pose[:, 0] = np.arange(20)*.03
        pose[:, 9] = np.arange(20)*.04
        bank, _ = g1_io.model_features_to_feature_bank(pose)
        extra = g1_io.model_features_to_motion_ae_extension(pose, 20)
        spec = dict(extension='extensions/motion_ae_v1', bank_feature_dim=72,
                    extension_feature_dim=41, indices=list(range(29))+list(range(72,113)),
                    heading_definition='quaternion_wz_projection')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'new_joint_vecs').mkdir()
            ext = root/spec['extension']
            (ext/'features').mkdir(parents=True)
            path = root/'new_joint_vecs/window.npy'
            np.save(path, bank[5:])
            np.save(ext/'features/window.npy', extra[5:])
            info = dict(complete=True, partial=False, format='bones_g1_motion_ae_extension_v1',
                        feature_dim=41, target_fps=20, velocity_unit='per_second',
                        velocity_frame='current_heading', heading_definition='quaternion_wz_projection')
            (ext/'dataset_info.json').write_text(json.dumps(info))
            values, _ = reconstruction.representation_input_features(pose[5:], spec, path, 20, 20)
            np.testing.assert_array_equal(values, np.concatenate((bank[5:,:29], extra[5:]), axis=1))
            self.assertGreater(values[0,29], 0)
            self.assertGreater(values[0,64], 0)
            with self.assertRaises(ValueError):
                reconstruction.representation_input_features(pose[5:], spec, path, 40, 20)

    def test_static_and_single_frame(self):
        for n in (1, 5):
            pose = np.zeros((n, 38), np.float32)
            pose[:, 3:9] = [1, 0, 0, 0, 1, 0]
            extra = g1_io.model_features_to_motion_ae_extension(pose, 20)
            np.testing.assert_array_equal(extra[:, :29], 0)
            np.testing.assert_array_equal(extra[:, 35:], 0)
            np.testing.assert_array_equal(extra[:, 29:35], pose[:, 3:9])

    def test_motion_ae_reference(self):
        repo = Path(__file__).resolve().parents[2] / 'motion_ae'
        if not (repo / 'motion_ae/feature_builder.py').exists():
            self.skipTest('Sibling motion_ae checkout unavailable')
        sys.path.insert(0, str(repo))
        from motion_ae.feature_builder import build_features
        from motion_ae.config import NpzKeysConfig, PelvisConfig
        t = np.arange(100) / 20
        pose = np.zeros((100, 38), np.float32)
        pose[:, :3] = np.column_stack((np.sin(t), np.cos(t), t*.1))
        pose[:, 3:9] = g1_io.rotation_matrix_to_6d(Rotation.from_euler(
            'xyz', np.column_stack((.2*np.sin(t), .3*np.cos(t), 3+t))).as_matrix())
        extra = g1_io.model_features_to_motion_ae_extension(pose, 20)
        rot = g1_io.rotation_6d_to_matrix(pose[:, 3:9])
        xyz = pose[:, :3].astype(np.float64)
        linear = np.diff(xyz, axis=0, prepend=xyz[:1]) * 20
        angular = np.zeros_like(linear)
        angular[1:] = Rotation.from_matrix(rot[1:] @ rot[:-1].transpose(0, 2, 1)).as_rotvec() * 20
        raw = dict(joint_pos=pose[:, 9:], joint_vel=extra[:, :29],
                   body_quat_w=Rotation.from_matrix(rot).as_quat()[:, None, [3, 0, 1, 2]],
                   body_lin_vel_w=linear[:, None], body_ang_vel_w=angular[:, None])
        reference, _ = build_features(raw, NpzKeysConfig(), PelvisConfig())
        np.testing.assert_allclose(np.concatenate((pose[:, 9:], extra), axis=1), reference, atol=1e-6)

    def test_physical_velocities_and_roundtrip(self):
        self.assertTrue(hasattr(g1_io, 'model_features_to_motion_ae_extension'))
        t = np.arange(100) / 20
        pose = np.zeros((100, 38), np.float32)
        pose[:, :3] = np.column_stack((t, t * 2, 1 + t * .1))
        rotation = Rotation.from_euler('xyz', np.column_stack((.2*np.sin(t), .3*np.cos(t), 3+t)))
        pose[:, 3:9] = g1_io.rotation_matrix_to_6d(rotation.as_matrix())
        pose[:, 9] = (3 + t + np.pi) % (2*np.pi) - np.pi
        bank, anchors = g1_io.model_features_to_feature_bank(pose)
        extra = g1_io.model_features_to_motion_ae_extension(pose, 20)
        self.assertEqual(extra.shape, (100, 41))
        np.testing.assert_array_equal(extra[0, :29], 0)
        np.testing.assert_array_equal(extra[0, 35:], 0)
        np.testing.assert_allclose(extra[1:, 0], 1, atol=1e-5)
        anchors[:, 3] = g1_io.extract_motion_ae_heading(rotation.as_matrix())
        yaw_rotation = g1_io.yaw_to_matrix(anchors[:, 3])
        world_v = np.einsum('tij,tj->ti', yaw_rotation, extra[:, 35:38])
        np.testing.assert_allclose(world_v[1:], np.broadcast_to([1, 2, .1], (99, 3)), atol=2e-5)
        world_w = np.einsum('tij,tj->ti', yaw_rotation, extra[:, 38:41])
        expected_w = (rotation[1:] * rotation[:-1].inv()).as_rotvec() * 20
        np.testing.assert_allclose(world_w[1:], expected_w, atol=5e-6)
        features = np.concatenate((bank[:, :29], extra), axis=1)
        for start in (0, 33):
            recovered = g1_io.motion_ae_features_to_model_features(features[start:], anchors[start], 20)
            np.testing.assert_allclose(recovered, pose[start:], atol=5e-5)
        with self.assertRaises(ValueError):
            g1_io.model_features_to_motion_ae_extension(pose, 0)

    def test_sidecar_loading_and_legacy_signature(self):
        self.assertTrue(hasattr(config, 'load_representation_stats'))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'new_joint_vecs').mkdir()
            info = dict(format='bones_g1_feature_bank_v1', complete=True, target_fps=20,
                        velocity_unit='per_frame', velocity_frame='current_heading',
                        feature_dim=72, feature_layout=g1_io.G1_BANK_LAYOUT)
            (root/'dataset_info.json').write_text(json.dumps(info))
            for split in ('train', 'val', 'test'):
                (root/f'{split}.txt').write_text('sample\n')
            for name in ('Mean', 'Std'):
                np.save(root/f'{name}.npy', np.ones(72, np.float32))
            old = config.resolve_representation(root, 'configs/representations/g1_local_rot6d68_v1.yaml')
            ext = root/'extensions/motion_ae_v1'
            (ext/'features').mkdir(parents=True)
            sidecar = dict(format='bones_g1_motion_ae_extension_v1', complete=True, partial=False,
                           base_signature=old['dataset_signature'], target_fps=20,
                           velocity_unit='per_second', velocity_frame='current_heading',
                           heading_definition='quaternion_wz_projection',
                           feature_dim=41, feature_layout=g1_io.G1_MOTION_AE_EXTENSION_LAYOUT)
            (ext/'dataset_info.json').write_text(json.dumps(sidecar))
            for name in ('Mean', 'Std'):
                np.save(ext/f'{name}.npy', np.full(41, 2, np.float32))
            spec = config.resolve_representation(root, 'configs/representations/g1_motion_ae70_v1.yaml')
            self.assertEqual(spec['feature_dim'], 70)
            self.assertEqual(old, config.resolve_representation(root, 'configs/representations/g1_local_rot6d68_v1.yaml'))
            mean, std = config.load_representation_stats(root, spec)
            np.testing.assert_array_equal(mean[:29], 1)
            np.testing.assert_array_equal(mean[29:58], 2)
            base = np.ones((10, 72), np.float32)
            np.save(ext/'features/sample.npy', np.full((10, 41), 2, np.float32))
            view = config.select_representation_features(base, root, 'sample', spec)
            self.assertEqual(view.shape, (10, 70))
            np.testing.assert_array_equal(view[0], mean)
            np.save(ext/'features/sample.npy', np.zeros((9, 41), np.float32))
            with self.assertRaises(ValueError):
                config.select_representation_features(base, root, 'sample', spec)
            sidecar['complete'] = False
            (ext/'dataset_info.json').write_text(json.dumps(sidecar))
            with self.assertRaises(ValueError):
                config.resolve_representation(root, 'configs/representations/g1_motion_ae70_v1.yaml')


if __name__ == '__main__':
    unittest.main()

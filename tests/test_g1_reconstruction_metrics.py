import unittest

import numpy as np

from evaluation.g1_metrics import metric_values, ground_truth_native


class ReconstructionMetricTests(unittest.TestCase):
    def setUp(self):
        self.pose = np.zeros((8, 38), dtype=np.float64)
        self.pose[:, 3:9] = [1, 0, 0, 0, 1, 0]
        self.points = np.zeros((8, 29, 3))

    def test_identical(self):
        result = metric_values(self.pose, self.pose, self.points, self.points, 20, [0, 1, 2, 3])
        for values in result.values():
            np.testing.assert_allclose(values, 0, atol=1e-8)

    def test_translation(self):
        prediction = self.pose.copy()
        prediction[:, 0] += 1
        points = self.points.copy()
        points[:, :, 0] += 1
        values = metric_values(self.pose, prediction, self.points, points, 20, [0, 1, 2, 3])
        np.testing.assert_allclose(values['global_mpjpe_mm'], 1000)
        np.testing.assert_allclose(values['root_relative_mpjpe_mm'], 0)
        np.testing.assert_allclose(values['root_ade_m'], 1)
        np.testing.assert_allclose(values['root_fde_m'], 1)

    def test_rotation_and_time_units(self):
        from g1_io import rotation_matrix_to_6d
        from scipy.spatial.transform import Rotation
        prediction = self.pose.copy()
        prediction[:, 3:9] = rotation_matrix_to_6d(Rotation.from_euler('z', np.full((8, 1), 90), degrees=True).as_matrix())
        prediction[:, 9:] = np.arange(8)[:, None] * 0.05
        values = metric_values(self.pose, prediction, self.points, self.points, 20, [0, 1, 2, 3])
        np.testing.assert_allclose(values['root_rotation_deg'], 90)
        np.testing.assert_allclose(values['root_yaw_deg'], 90)
        np.testing.assert_allclose(values['joint_velocity_rmse_rad_s'], 1)
        np.testing.assert_allclose(values['joint_acceleration_rmse_rad_s2'], 0, atol=1e-10)

    def test_gt_uses_all_anchors(self):
        from g1_io import model_features_to_feature_bank
        bank, anchors = model_features_to_feature_bank(self.pose)
        anchors[:, 0] = np.arange(8)
        gt = ground_truth_native(bank, anchors)
        np.testing.assert_array_equal(gt[:, 0], anchors[:, 0])

    def test_batch_padding_is_removed(self):
        from evaluation.eval_g1_rvq import reconstruct_batch

        class IdentityModel:
            def encode(self, x):
                self.shape = tuple(x.shape)
                return x, None

            def forward_decoder(self, tokens):
                return tokens

        model = IdentityModel()
        features = np.arange(5 * 3, dtype=np.float32).reshape(5, 3)
        output = reconstruct_batch(model, [features, features], np.zeros(3), np.ones(3), 4, 'cpu')
        self.assertEqual(model.shape, (2, 8, 3))
        self.assertEqual(output.shape, (2, 5, 3))
        np.testing.assert_array_equal(output[0], features)
        with self.assertRaises(ValueError):
            reconstruct_batch(model, [features, features[:4]], np.zeros(3), np.ones(3), 4, 'cpu')


if __name__ == '__main__':
    unittest.main()

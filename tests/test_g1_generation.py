from pathlib import Path
import unittest

import numpy as np

from gen_g1 import to_native_g1_features
from g1_io import model_features_to_feature_bank, G1_BANK_LAYOUT


class GenerationTests(unittest.TestCase):
    def test_local_representation_decode(self):
        import yaml
        spec = yaml.safe_load(Path('configs/representations/g1_local_rot6d68_v1.yaml').read_text())
        spec['feature_dim'] = 68
        gt = np.zeros((8, 38), dtype=np.float32)
        gt[:, 2] = 0.8
        gt[:, 3:9] = [1, 0, 0, 0, 1, 0]
        gt[:, 0] = np.arange(8) * 0.02
        bank, _ = model_features_to_feature_bank(gt)
        indices = [i for f in spec['fields'] for i in range(*G1_BANK_LAYOUT[f])]
        result = to_native_g1_features(bank[:, indices], 20, 'per_frame', spec)
        np.testing.assert_allclose(result, gt, atol=1e-6)
        with self.assertRaises(ValueError):
            to_native_g1_features(bank, 20, 'per_frame', spec)

    def test_legacy(self):
        native = np.zeros((4, 38), dtype=np.float32)
        np.testing.assert_array_equal(to_native_g1_features(native, 20, 'per_frame'), native)


if __name__ == '__main__':
    unittest.main()

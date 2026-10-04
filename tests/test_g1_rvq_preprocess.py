"""Contracts for offline RVQ conditioning; run with unittest discover."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from argparse import Namespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'dataset' / 'bones-seed_prepare'))
from g1_io import model_features_to_feature_bank
from models.vq.model import RVQVAE


class PreprocessTest(unittest.TestCase):
    def setUp(self):
        path = ROOT / 'dataset' / 'bones-seed_prepare' / 'g1_preprocess_rvq.py'
        self.assertTrue(path.exists(), 'Offline RVQ preprocessor must exist')
        spec = importlib.util.spec_from_file_location('preprocess', path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def test_reference_uses_each_ground_truth_anchor_not_velocity_integration(self):
        native = np.zeros((5, 38), np.float32)
        native[:, :3] = [[0, 0, 1], [1, 0, 1], [2, 1, 1], [2, 2, 1], [3, 2, 1]]
        native[:, 3:9] = [1, 0, 0, 0, 1, 0]
        native[:, 9:] = .25
        bank, anchors = model_features_to_feature_bank(native)
        bank[:, 68:71] = 123  # Auxiliary deltas must not corrupt original references.
        np.testing.assert_allclose(self.module.native_reference(bank, anchors), native, atol=1e-6)
        with self.assertRaises(ValueError):
            self.module.native_reference(bank, anchors[:-1])

    def test_padded_tail_and_mixed_lengths_match_independent_encoding(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        args = Namespace(num_quantizers=2, shared_codebook=False, quantize_dropout_prob=.5, mu=.99)
        model = RVQVAE(args, 3, 8, 8, 8, 2, 2, 8, 1, 1).eval()
        for layer in model.quantizer.layers:
            layer.codebook.copy_(torch.randn_like(layer.codebook))
        seqs = [np.random.default_rng(n).normal(size=(n, 3)).astype(np.float32) for n in (5, 16, 6)]
        mean, std = np.ones(3, np.float32), np.full(3, 2, np.float32)
        result = self.module.encode_sequences(model, seqs, mean, std, 4, 8)
        for seq, (tokens, latent) in zip(seqs, result):
            padded = np.pad(seq, ((0, (-len(seq)) % 4), (0, 0)), mode='edge')
            with torch.inference_mode():
                expected, _ = model.encode(torch.from_numpy((padded - mean) / std)[None])
                embedding = model.quantizer.get_codes_from_indices(expected).sum(0)[0].numpy()
            np.testing.assert_array_equal(tokens, expected[0].numpy())
            np.testing.assert_allclose(latent, embedding, atol=1e-6)
            self.assertEqual(len(tokens), (len(seq) + 3) // 4)
        with self.assertRaises(ValueError):
            self.module.encode_sequences(model, [np.empty((0, 3), np.float32)], mean, std, 4, 8)

    def test_interrupted_resume_preserves_unvisited_checksums(self):
        self.assertTrue(hasattr(self.module, 'publish_shard'), 'Resume must preserve all published records')
        first = {'path': 'train-00000.npz', 'sha256': 'first'}
        later = {'path': 'val-00000.npz', 'sha256': 'original'}
        manifest = {'complete': True, 'shards': [first, later]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'manifest.json'
            self.module.publish_shard(path, manifest, first)
            # Simulate process interruption and a fresh resume from disk.
            reloaded = json.loads(path.read_text())
            self.assertFalse(reloaded['complete'])
            self.assertEqual(reloaded['shards'], [first, later])
            with self.assertRaises(ValueError):
                self.module.publish_shard(path, reloaded, dict(later, sha256='changed'))

    def test_shard_rejects_bad_offsets_and_out_of_range_tokens(self):
        rows = [{'sample_id': 'clip', 'frames': 5}]
        arrays = dict(sample_ids=np.array(['clip']), frame_offsets=np.array([0, 5]),
                      token_offsets=np.array([0, 2]), reference_native38=np.zeros((5, 38), np.float32),
                      tokens=np.zeros((2, 2), np.int32), latent=np.zeros((2, 8), np.float32))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'shard.npz'
            np.savez(path, **arrays)
            self.assertEqual(self.module.validate_shard(path, rows, 4, 2, 8, 8), (5, 2))
            arrays['token_offsets'] = np.array([0, 1])
            np.savez(path, **arrays)
            with self.assertRaises(AssertionError):
                self.module.validate_shard(path, rows, 4, 2, 8, 8)
            arrays['token_offsets'] = np.array([0, 2])
            arrays['tokens'][0, 0] = 8
            np.savez(path, **arrays)
            with self.assertRaises(ValueError):
                self.module.validate_shard(path, rows, 4, 2, 8, 8)


if __name__ == '__main__':
    unittest.main()

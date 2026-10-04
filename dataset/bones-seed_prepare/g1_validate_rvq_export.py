#!/usr/bin/env python3
"""Validate exported shard coverage/embeddings and sampled raw CSV correspondence."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from g1_preprocess_rvq import load_records, sha256, validate_shard, write_json
from g1_io import read_g1_csv, csv_values_to_model_features
from g1_visualize.reconstruct_g1_rvq import build_model
from utils.get_opt import get_opt
from utils.g1_feature_config import select_representation_features, representation_motion_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_root', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--samples-per-split', type=int, default=3)
    parser.add_argument('--data-root', type=Path, help='Override source bank mount path')
    parser.add_argument('--bones-root', type=Path, help='Override original CSV mount path')
    parser.add_argument('--checkpoint', type=Path, help='Override checkpoint mount path')
    args = parser.parse_args()
    if args.samples_per_split < 1:
        parser.error('--samples-per-split must be positive')
    out = args.output_root
    manifest = json.loads((out / 'manifest.json').read_text())
    if not manifest['complete']:
        raise ValueError('Export is not complete')
    root = args.data_root or Path(manifest['data_source'])
    checkpoint = args.checkpoint or Path(manifest['checkpoint_source'])
    bones = args.bones_root or Path(json.loads((root/'dataset_info.json').read_text())['source_root'])
    representation = manifest['identity']['representation']
    if sha256(checkpoint) != manifest['identity']['checkpoint_sha256']:
        raise ValueError('Checkpoint hash mismatch')
    unit = manifest['token_stride_frames']
    splits = load_records(root)
    limit = manifest['identity']['limit_per_split']
    if limit:
        splits = {split: rows[:limit] for split, rows in splits.items()}
    rows_by_id = {row['sample_id']: row for rows in splits.values() for row in rows}
    selected = set()
    for rows in splits.values():
        ordered = sorted(rows, key=lambda row: (row['frames'], row['sample_id']))
        for index in np.linspace(0, len(ordered)-1, min(len(ordered), args.samples_per_split), dtype=int):
            selected.add(ordered[index]['sample_id'])
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    options = get_opt(str(checkpoint.parent.parent/'opt.txt'), device)
    model = build_model(options, representation['feature_dim'], device)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(saved['vq_model'], strict=True)
    del saved
    mean, std = np.load(out/'mean.npy'), np.load(out/'std.npy')
    for name in ('mean.npy', 'std.npy'):
        if sha256(out/name) != manifest['identity']['normalization_sha256'][name]:
            raise ValueError(f'Normalization hash mismatch: {name}')
    books = np.load(out/'codebooks.npy')
    np.testing.assert_array_equal(books, model.quantizer.codebooks.detach().cpu().numpy())
    motion_dir = representation_motion_dir(root, representation)
    seen, checked = set(), []
    frame_count = token_count = 0
    embedding_error = 0.0
    for shard in manifest['shards']:
        path = out/shard['path']
        if sha256(path) != shard['sha256']:
            raise ValueError(f'Shard checksum mismatch: {path}')
        with np.load(path, allow_pickle=False) as data:
            names = data['sample_ids'].tolist()
            rows = [rows_by_id[name] for name in names]
            if seen.intersection(names) or len(names) != len(set(names)):
                raise ValueError('Duplicate samples in export')
            if any(row['split'] != shard['split'] for row in rows):
                raise ValueError('Export split mismatch')
            seen.update(names)
            nf, nt = validate_shard(path, rows, unit, options.num_quantizers, options.code_dim, options.nb_code)
            frame_count += nf
            token_count += nt
            tokens, latent = data['tokens'], data['latent']
            # Small blocks avoid allocating another full shard per RVQ layer.
            for start in range(0, nt, 8192):
                ids = tokens[start:start+8192]
                terms = [books[q, ids[:, q]].astype(np.float64) for q in range(options.num_quantizers)]
                expected = sum(terms)
                actual = latent[start:start+8192]
                # Float32 parallel GPU reduction and sequential NumPy sum differ under cancellation.
                # Bound error against a float64 sum by gamma_n * sum(abs(terms)).
                n_eps = options.num_quantizers * np.finfo(np.float32).eps
                bound = n_eps / (1 - n_eps) * sum(np.abs(term) for term in terms)
                error = np.abs(actual - expected)
                if np.any(error > bound + np.finfo(np.float32).tiny):
                    raise ValueError(f'Embedding differs from codebook sum beyond float32 rounding: {path}')
                embedding_error = max(embedding_error, float(np.max(error)))
            for index, name in enumerate(names):
                if name not in selected:
                    continue
                f0, f1 = data['frame_offsets'][index:index+2]
                t0, t1 = data['token_offsets'][index:index+2]
                reference = data['reference_native38'][f0:f1]
                bank = np.load(motion_dir/f'{name}.npy')
                anchors = np.load(root/'anchors'/f'{name}.npy')
                np.testing.assert_array_equal(reference[:, :3], anchors[:, :3])
                np.testing.assert_array_equal(reference[:, 9:], bank[:, :29])
                features = select_representation_features(bank, root, name, representation)
                padded = np.pad(features, ((0, (-len(features)) % unit), (0, 0)), mode='edge')
                with torch.inference_mode():
                    ids, _ = model.encode(torch.from_numpy((padded-mean)/std)[None].to(device))
                np.testing.assert_array_equal(ids[0].cpu().numpy(), tokens[t0:t1])
                raw = csv_values_to_model_features(read_g1_csv(bones/rows[index]['source']))[::6]
                candidates = np.where(np.max(np.abs(raw-reference[0]), axis=1) < 1e-4)[0]
                error = min((float(np.max(np.abs(raw[j:j+len(reference)]-reference)))
                             for j in candidates if j+len(reference) <= len(raw)), default=float('inf'))
                if error >= 1e-4:
                    raise ValueError(f'Original CSV mismatch: {name}, {error}')
                checked.append({'sample': name, 'split': shard['split'], 'raw_max_abs_error': error,
                                'standalone_tokens_equal': True})
        print(f'Validated {path.name}: samples={len(seen)}', flush=True)
    if (seen != rows_by_id.keys() or len(seen) != manifest['samples']
            or frame_count != manifest['frames'] or token_count != manifest['tokens']
            or len(checked) != len(selected)):
        raise ValueError('Export coverage/count mismatch')
    report = {'passed': True, 'partial': manifest['partial'], 'samples': len(seen),
              'frames': frame_count, 'tokens': token_count, 'shards': len(manifest['shards']),
              'all_embeddings_checked': True, 'max_embedding_abs_error': embedding_error,
              'checkpoint_sha256': manifest['identity']['checkpoint_sha256'], 'sample_checks': checked}
    write_json(out/'validation_report.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

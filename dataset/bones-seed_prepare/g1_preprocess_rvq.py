#!/usr/bin/env python3
"""Freeze G1 feature-bank windows into sharded RVQ conditions and GT references."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from g1_io import (G1_BANK_LAYOUT, G1_FEATURE_COLUMNS, G1_FEATURE_DIM,
                   G1_JOINT_SLICE, G1_ROOT_ROTATION_SLICE,
                   rotation_6d_to_matrix, rotation_matrix_to_6d, yaw_to_matrix)
from g1_visualize.reconstruct_g1_rvq import build_model
from utils.get_opt import get_opt
from utils.g1_feature_config import (feature_bank_signature, load_representation_stats,
                                      representation_motion_dir, select_representation_features)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)



def publish_shard(path, manifest, record):
    """Retain unvisited published hashes even if a resume is interrupted again."""
    published = {item['path']: item for item in manifest['shards']}
    previous = published.get(record['path'])
    if previous is not None and previous != record:
        raise ValueError(f"Published shard changed: {record['path']}")
    published[record['path']] = record
    manifest['shards'] = list(published.values())
    manifest['complete'] = False
    write_json(path, manifest)


def native_reference(bank, anchors):
    """Use every original XYZ/yaw anchor, never integrate RVQ or bank velocities."""
    if (bank.ndim != 2 or bank.shape[1] not in (72, 113) or not len(bank)
            or anchors.shape != (len(bank), 4)
            or not np.isfinite(bank).all() or not np.isfinite(anchors).all()):
        raise ValueError('Invalid feature bank or per-frame ground-truth anchors')
    result = np.empty((len(bank), G1_FEATURE_DIM), np.float32)
    result[:, :3] = anchors[:, :3]
    tilt = rotation_6d_to_matrix(bank[:, slice(*G1_BANK_LAYOUT['root_roll_pitch_rot6d'])])
    result[:, G1_ROOT_ROTATION_SLICE] = rotation_matrix_to_6d(yaw_to_matrix(anchors[:, 3]) @ tilt)
    result[:, G1_JOINT_SLICE] = bank[:, slice(*G1_BANK_LAYOUT['joint_angles'])]
    return result


@torch.inference_mode()
def encode_sequences(model, sequences, mean, std, unit, batch_size):
    """Group by padded length: longer neighbours must not change convolution edges."""
    if unit <= 0 or batch_size <= 0:
        raise ValueError('unit and batch_size must be positive')
    groups = defaultdict(list)
    result = [None] * len(sequences)
    for index, seq in enumerate(sequences):
        if seq.ndim != 2 or not len(seq) or seq.shape[1:] != mean.shape or not np.isfinite(seq).all():
            raise ValueError('Empty, nonfinite or incorrectly shaped RVQ input')
        groups[(len(seq) + unit - 1) // unit * unit].append(index)
    device = next(model.parameters()).device
    for length, indices in groups.items():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            batch = np.stack([np.pad(sequences[i], ((0, length-len(sequences[i])), (0, 0)),
                                     mode='edge') for i in batch_indices])
            x = torch.from_numpy((batch - mean) / std).to(device)
            tokens, _ = model.encode(x)
            latent = model.quantizer.get_codes_from_indices(tokens).sum(0).cpu().numpy()
            tokens = tokens.cpu().numpy()
            if (tokens < 0).any() or (tokens >= model.num_code).any() or not np.isfinite(latent).all():
                raise ValueError('Invalid RVQ output')
            for row, index in enumerate(batch_indices):
                result[index] = (tokens[row].astype(np.int32), latent[row].copy())
    return result


def load_records(root):
    records = {}
    with (root / 'metadata/windows.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            sample = row['sample_id']
            if sample in records or Path(sample).name != sample or row['frames'] <= 0:
                raise ValueError(f'Invalid or duplicate window: {sample}')
            records[sample] = row
    seen = set()
    splits = {}
    for split in ('train', 'val', 'test'):
        names = (root / f'{split}.txt').read_text().splitlines()
        for sample in names:
            if sample in seen or sample not in records or records[sample]['split'] != split:
                raise ValueError(f'Split membership mismatch: {sample}')
            seen.add(sample)
        splits[split] = [records[name] for name in names]
    if seen != records.keys():
        raise ValueError('Window manifest and splits have different coverage')
    return splits


def validate_shard(path, rows, unit, num_quantizers, code_dim, nb_code):
    """Fully read the saved arrays (including ZIP CRC); check all offsets and values."""
    lengths = np.array([row['frames'] for row in rows], dtype=np.int64)
    with np.load(path, allow_pickle=False) as data:
        np.testing.assert_array_equal(data['sample_ids'], [r['sample_id'] for r in rows])
        np.testing.assert_array_equal(data['frame_offsets'], np.r_[0, np.cumsum(lengths)])
        np.testing.assert_array_equal(data['token_offsets'], np.r_[0, np.cumsum((lengths+unit-1)//unit)])
        nframes, ntokens = int(data['frame_offsets'][-1]), int(data['token_offsets'][-1])
        refs, tokens, latent = data['reference_native38'], data['tokens'], data['latent']
        if (refs.shape != (nframes, G1_FEATURE_DIM) or tokens.shape != (ntokens, num_quantizers)
                or latent.shape != (ntokens, code_dim) or refs.dtype != np.float32
                or latent.dtype != np.float32 or tokens.dtype != np.int32
                or not np.isfinite(refs).all() or not np.isfinite(latent).all()
                or (tokens < 0).any() or (tokens >= nb_code).any()):
            raise ValueError(f'Invalid shard arrays: {path}')
    return nframes, ntokens


def run(args):
    checkpoint = args.checkpoint.resolve()
    experiment = checkpoint.parent.parent
    representation = json.loads((experiment / 'meta/representation.json').read_text())
    if representation['id'] != 'g1_local_rot6d68_v1':
        raise ValueError('This export contract currently requires g1_local_rot6d68_v1')
    info = json.loads((args.data_root / 'dataset_info.json').read_text())
    if not info.get('complete') or info.get('partial'):
        raise ValueError('Source feature bank is incomplete or partial')
    if feature_bank_signature(args.data_root) != representation['dataset_signature']:
        raise ValueError('Feature bank differs from RVQ training data')
    mean = np.load(experiment / 'meta/mean.npy')
    std = np.load(experiment / 'meta/std.npy')
    expected_mean, expected_std = load_representation_stats(args.data_root, representation)
    np.testing.assert_array_equal(mean, expected_mean)
    np.testing.assert_array_equal(std, expected_std)
    if mean.shape != (representation['feature_dim'],) or std.shape != mean.shape:
        raise ValueError('Normalization shape mismatch')
    motion_dir = representation_motion_dir(args.data_root, representation)
    options = get_opt(str(experiment / 'opt.txt'), torch.device(args.device))
    unit = options.stride_t ** options.down_t
    model = build_model(options, len(mean), torch.device(args.device))
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved.get('representation') != representation:
        raise ValueError('Checkpoint and metadata representation differ')
    model.load_state_dict(saved['vq_model'], strict=True)
    model.requires_grad_(False)
    epoch, iteration = saved.get('ep'), saved.get('total_it')
    del saved
    # Keep the existing training split; sort by length only to improve inference batching.
    splits = load_records(args.data_root)
    full_count = sum(map(len, splits.values()))
    if full_count != info['samples']:
        raise ValueError('Source sample count mismatch')
    if args.limit:
        splits = {name: rows[:args.limit] for name, rows in splits.items()}
    for rows in splits.values():
        rows.sort(key=lambda r: ((r['frames']+unit-1)//unit, r['sample_id']))
    code_paths = [Path(__file__), ROOT/'dataset/bones-seed_prepare/g1_io.py',
                  ROOT/'utils/g1_feature_config.py', ROOT/'utils/get_opt.py',
                  ROOT/'g1_visualize/reconstruct_g1_rvq.py', *sorted((ROOT/'models/vq').glob('*.py'))]
    identity = {
        'format': 'g1_rvq_tracking_v1', 'checkpoint_sha256': sha256(checkpoint),
        'representation': representation, 'normalization_sha256': {
            name: sha256(experiment / 'meta' / name) for name in ('mean.npy', 'std.npy')},
        'options_sha256': sha256(experiment / 'opt.txt'),
        'windows_sha256': sha256(args.data_root / 'metadata/windows.jsonl'),
        'code_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in code_paths},
        'shard_size': args.shard_size, 'batch_size': args.batch_size, 'limit_per_split': args.limit,
        'torch': torch.__version__, 'device_type': torch.device(args.device).type,
        'precision': 'float32; no autocast; tf32 disabled',
    }
    manifest_path = args.output_root / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['identity'] != identity:
            raise ValueError('Existing export has different inputs/options/code; use a new output directory')
        if not args.resume:
            raise ValueError('Output exists; pass --resume to validate and reuse completed shards')
    else:
        if any(args.output_root.glob('*.npz')):
            raise ValueError('Untracked shards in output directory')
        manifest = {
            'identity': identity, 'complete': False, 'partial': bool(args.limit),
            'checkpoint_source': str(checkpoint), 'checkpoint_epoch': epoch,
            'checkpoint_iteration': iteration, 'data_source': str(args.data_root.resolve()),
            'source_sample_count': full_count, 'reference_fps': representation['fps'],
            'token_stride_frames': unit, 'latent_fps': representation['fps']/unit,
            'num_quantizers': options.num_quantizers, 'code_dim': options.code_dim,
            'nb_code': options.nb_code, 'latent_definition': 'sum of all quantized codebook embeddings',
            'temporal_mapping': 'frame=floor(t*reference_fps); token=frame//token_stride_frames; phase=(frame%token_stride_frames)/token_stride_frames',
            'tail_padding': 'repeat last model frame to next stride multiple; reference excludes padding',
            'encoder_context': 'noncausal; independent full window, not streaming',
            'reference_layout': 'root_xyz_m(3), root_rot6d_columns(6), joint_angles_rad(29)',
            'joint_names': [name.removesuffix('_dof') for name in G1_FEATURE_COLUMNS[6:]],
            'reference_source': 'original bank joint angles/roll-pitch and per-frame GT XYZ/yaw anchors; no RVQ decoder',
            'shards': [],
        }
        write_json(manifest_path, manifest)
    for name in ('mean.npy', 'std.npy', 'representation.json'):
        shutil.copyfile(experiment / 'meta' / name, args.output_root / name)
    np.save(args.output_root / 'codebooks.npy', model.quantizer.codebooks.detach().cpu().numpy())
    started = time.monotonic()
    completed = []

    def load(row):
        sample = row['sample_id']
        bank = np.load(motion_dir / f'{sample}.npy', allow_pickle=False)
        anchors = np.load(args.data_root / 'anchors' / f'{sample}.npy', allow_pickle=False)
        if len(bank) != row['frames']:
            raise ValueError(f'Wrong frame count: {sample}')
        return (select_representation_features(bank, args.data_root, sample, representation),
                native_reference(bank, anchors))

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for split, rows in splits.items():
            for start in range(0, len(rows), args.shard_size):
                selected = rows[start:start + args.shard_size]
                path = args.output_root / f'{split}-{start//args.shard_size:05d}.npz'
                existing = path.exists()
                if not existing:
                    loaded = list(pool.map(load, selected))
                    encoded = encode_sequences(model, [item[0] for item in loaded], mean, std,
                                               unit, args.batch_size)
                    frame_offsets = np.r_[0, np.cumsum([len(item[1]) for item in loaded])]
                    token_offsets = np.r_[0, np.cumsum([len(item[0]) for item in encoded])]
                    temp = path.with_suffix('.npz.tmp')
                    with temp.open('wb') as stream:
                        np.savez(stream, sample_ids=np.array([r['sample_id'] for r in selected]),
                                 source_paths=np.array([r['source'] for r in selected]),
                                 frame_offsets=frame_offsets, token_offsets=token_offsets,
                                 reference_native38=np.concatenate([item[1] for item in loaded]),
                                 tokens=np.concatenate([item[0] for item in encoded]),
                                 latent=np.concatenate([item[1] for item in encoded]))
                    # Only publish a fully readable shard; stale temporary files are overwritten on resume.
                    validate_shard(temp, selected, unit, options.num_quantizers, options.code_dim, options.nb_code)
                    temp.replace(path)
                    del loaded, encoded
                frames, tokens = validate_shard(path, selected, unit, options.num_quantizers,
                                                options.code_dim, options.nb_code)
                record = {'path': path.name, 'split': split, 'samples': len(selected),
                          'frames': frames, 'tokens': tokens, 'sha256': sha256(path)}
                publish_shard(manifest_path, manifest, record)
                completed.append(record)
                count = sum(r['samples'] for r in completed)
                print(f'{path.name}: {"reused" if existing else "encoded"}, samples={count}, '
                      f'elapsed={time.monotonic()-started:.1f}s', flush=True)
    manifest.update(complete=True, samples=sum(r['samples'] for r in completed),
                    frames=sum(r['frames'] for r in completed), tokens=sum(r['tokens'] for r in completed),
                    split_counts={k: len(v) for k, v in splits.items()})
    write_json(manifest_path, manifest)
    print(json.dumps({k: manifest[k] for k in ('complete', 'partial', 'samples', 'frames', 'tokens', 'split_counts')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--shard-size', type=int, default=2048)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--limit', type=int, default=0, help='Smoke only: at most N samples from each split')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if min(args.batch_size, args.shard_size, args.workers) <= 0 or args.limit < 0:
        parser.error('Batch/shard/workers must be positive; limit must be nonnegative')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / '.preprocess.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


if __name__ == '__main__':
    main()

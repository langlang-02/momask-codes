#!/usr/bin/env python3
"""Evaluate native-G1 RVQ checkpoints on a shared feature-bank split."""

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.g1_metrics import G1Kinematics, JOINT_NAMES, ENDPOINT_NAMES, ground_truth_native, metric_values, reduce_values
from g1_visualize.reconstruct_g1_rvq import build_model
from g1_visualize.replay_g1 import DEFAULT_MJCF
from g1_io import (local_features_to_model_features, motion_ae_features_to_model_features,
                   extract_motion_ae_heading, rotation_6d_to_matrix)
from utils.g1_feature_config import resolve_representation, representation_motion_dir, select_representation_features
from utils.get_opt import get_opt
from utils.transformer_training import file_sha256

DEFAULT_EXPERIMENTS = ['bones_g1_rvq_rot6d68_v1_nq6', 'bones_g1_rvq_sincos66_v1_nq6',
                       'bones_g1_rvq_motion_ae70_v1_nq6']


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def reconstruct_batch(model, features, mean, std, unit, device):
    length = len(features[0])
    if any(len(x) != length for x in features):
        raise ValueError('Batch must contain equal-length trajectories')
    array = np.stack(features)
    padding = (-length) % unit
    if padding:
        array = np.concatenate((array, np.repeat(array[:, -1:], padding, axis=1)), axis=1)
    tensor = torch.as_tensor((array - mean) / std, dtype=torch.float32, device=device)
    with torch.inference_mode():
        tokens, _ = model.encode(tensor)
        result = model.forward_decoder(tokens).cpu().numpy()
    return result[:, :length] * std + mean


def decode(features, spec, gt, anchors, fps):
    anchor = anchors[0].copy()
    if spec.get('extension'):
        anchor[3] = extract_motion_ae_heading(rotation_6d_to_matrix(gt[:1, 3:9]))[0]
        return motion_ae_features_to_model_features(features, anchor, fps)
    return local_features_to_model_features(features, spec['fields'], anchor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--checkpoints-dir', type=Path, required=True)
    parser.add_argument('--experiments', nargs='+', default=DEFAULT_EXPERIMENTS)
    parser.add_argument('--checkpoints', nargs='+', default=['net_best_val', 'latest'])
    parser.add_argument('--split', default='test')
    parser.add_argument('--mjcf', type=Path, default=DEFAULT_MJCF)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1 or args.limit < 0:
        parser.error('batch-size must be positive and limit must be nonnegative')
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(3407)
    np.random.seed(3407)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    started = time.time()
    split = args.data_root / f'{args.split}.txt'
    names = split.read_text().splitlines()
    if len(set(names)) != len(names) or not names:
        raise ValueError('Empty or duplicate split IDs')
    if args.limit:
        names = names[:args.limit]
    manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(),
                    arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    split_sha256=file_sha256(split), sample_count=len(names), fps=20,
                    joint_names=JOINT_NAMES, endpoint_names=ENDPOINT_NAMES,
                    mjcf_sha256=file_sha256(args.mjcf), initial_state='GT first frame only',
                    metrics_protocol='g1_rvq_reconstruction_v1', complete=False)
    manifest['precision'] = 'float32; TF32 disabled; deterministic cuDNN'
    write_json(out / 'manifest.json', manifest)
    (out / 'samples.txt').write_text('\n'.join(names) + '\n')
    fk = G1Kinematics(args.mjcf)
    experiments = []
    for experiment in args.experiments:
        directory = args.checkpoints_dir / 'bones_g1' / experiment
        spec_path = directory / 'meta/representation.json'
        spec = json.loads(spec_path.read_text())
        # The resolver accepts JSON as YAML, so the checkpoint spec is the config.
        if resolve_representation(args.data_root, spec_path) != spec:
            raise ValueError(f'Data/representation mismatch: {experiment}')
        motion_dir = representation_motion_dir(args.data_root, spec)
        experiments.append((experiment, directory, spec))
    samples, groups = {}, defaultdict(list)
    print(f'Loading GT for {len(names)} samples from {motion_dir}', flush=True)
    for i, name in enumerate(names):
        bank = np.load(motion_dir / f'{name}.npy')
        anchors = np.load(args.data_root / 'anchors' / f'{name}.npy')
        if not np.isfinite(bank).all() or not np.isfinite(anchors).all():
            raise ValueError(f'Nonfinite sample {name}')
        gt = ground_truth_native(bank, anchors)
        samples[name] = (bank, anchors, gt, fk.positions(gt))
        groups[len(bank)].append(name)
        if (i + 1) % 1000 == 0:
            print(f'GT {i + 1}/{len(names)}, elapsed={time.time() - started:.1f}s', flush=True)
    comparison = []
    for experiment, directory, spec in experiments:
        mean, std = (np.load(directory / 'meta' / f'{n}.npy') for n in ('mean', 'std'))
        if (mean.shape != (spec['feature_dim'],) or std.shape != mean.shape or
                not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any()):
            raise ValueError(f'Invalid normalization: {experiment}')
        opt = get_opt(str(directory / 'opt.txt'), torch.device(args.device))
        unit = opt.stride_t ** opt.down_t
        for checkpoint_name in args.checkpoints:
            path = directory / 'model' / (checkpoint_name.removesuffix('.tar') + '.tar')
            checksum = file_sha256(path)
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            if checkpoint.get('representation') != spec:
                raise ValueError(f'Checkpoint representation mismatch: {path}')
            model = build_model(opt, spec['feature_dim'], torch.device(args.device))
            model.load_state_dict(checkpoint.get('vq_model', checkpoint.get('net')))
            epoch = int(checkpoint['ep'])
            del checkpoint
            run = out / f'{experiment}_{path.stem}_ep{epoch:04d}'
            run.mkdir()
            identity = dict(experiment=experiment, checkpoint=str(path), checkpoint_sha256=checksum,
                            epoch=epoch, representation=spec, mean_sha256=file_sha256(directory / 'meta/mean.npy'),
                            std_sha256=file_sha256(directory / 'meta/std.npy'))
            write_json(run / 'identity.json', identity)
            totals, rows = {}, []
            joint_abs, joint_sq, joint_count = np.zeros(29), np.zeros(29), 0
            roundtrip_max = 0.0
            batch_delta = 0.0
            run_start = time.time()
            checked_batch = False
            with (run / 'per_sample.csv').open('w') as stream:
                writer = None
                for length, ids in sorted(groups.items()):
                    for start in range(0, len(ids), args.batch_size):
                        batch_ids = ids[start:start + args.batch_size]
                        inputs = [select_representation_features(samples[n][0], args.data_root, n, spec) for n in batch_ids]
                        predictions = reconstruct_batch(model, inputs, mean, std, unit, args.device)
                        if not checked_batch and len(batch_ids) > 1:
                            single = reconstruct_batch(model, inputs[:1], mean, std, unit, args.device)[0]
                            batch_delta = float(np.max(np.abs(single - predictions[0])))
                            np.testing.assert_allclose(single, predictions[0], atol=1e-4, rtol=1e-4)
                            checked_batch = True
                        for name, features, prediction in zip(batch_ids, inputs, predictions):
                            _, anchors, gt, gt_points = samples[name]
                            recovered = decode(features, spec, gt, anchors, manifest['fps'])
                            error = float(np.max(np.abs(recovered - gt)))
                            roundtrip_max = max(roundtrip_max, error)
                            if error > 1e-3:
                                raise ValueError(f'Representation roundtrip failed: {name}, max error={error}')
                            pred = decode(prediction, spec, gt, anchors, manifest['fps'])
                            values = metric_values(gt, pred, gt_points, fk.positions(pred), manifest['fps'], fk.endpoints)
                            row = dict(sample_id=name, frames=length)
                            for key, value in values.items():
                                row[key] = reduce_values(key, value)
                                total, count = totals.get(key, (0.0, 0))
                                total += float(np.square(value).sum() if 'rmse' in key else value.sum())
                                totals[key] = (total, count + value.size)
                            angles = (pred[:, 9:] - gt[:, 9:]) * (180 / np.pi)
                            joint_abs += np.abs(angles).sum(axis=0)
                            joint_sq += np.square(angles).sum(axis=0)
                            joint_count += length
                            if writer is None:
                                writer = csv.DictWriter(stream, fieldnames=list(row))
                                writer.writeheader()
                            writer.writerow(row)
                            rows.append(row)
                        if len(rows) % 500 < len(batch_ids):
                            stream.flush()
                            print(f'{experiment}/{path.stem}: {len(rows)}/{len(names)}, elapsed={time.time()-run_start:.1f}s', flush=True)
            if len(rows) != len(names) or file_sha256(path) != checksum:
                raise ValueError('Incomplete coverage or checkpoint changed during evaluation')
            summary = dict(epoch=epoch, samples=len(rows), frames=joint_count, seconds=time.time()-run_start,
                           roundtrip_max_abs=roundtrip_max, batch_single_max_abs=batch_delta,
                           batch_single_checked=checked_batch, metrics={}, complete=True)
            for key, (total, count) in totals.items():
                weighted = total / count
                if 'rmse' in key:
                    weighted = np.sqrt(weighted)
                summary['metrics'][key] = dict(action_mean=float(np.mean([r[key] for r in rows])),
                    action_p95=float(np.percentile([r[key] for r in rows], 95)),
                    pooled=float(weighted), observations=int(count))
            write_json(run / 'summary.json', summary)
            with (run / 'per_joint.csv').open('w') as stream:
                writer = csv.writer(stream)
                writer.writerow(['joint', 'mae_deg', 'rmse_deg'])
                writer.writerows(zip(JOINT_NAMES, joint_abs / joint_count, np.sqrt(joint_sq / joint_count)))
            comparison.append(dict(experiment=experiment, checkpoint=path.stem, epoch=epoch,
                                   **{k: v['action_mean'] for k, v in summary['metrics'].items()}))
            write_json(out / 'comparison.json', comparison)
            print(json.dumps(dict(experiment=experiment, checkpoint=path.stem, **summary)), flush=True)
            del model
            torch.cuda.empty_cache()
    with (out / 'comparison.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    manifest.update(complete=True, elapsed_seconds=time.time()-started)
    write_json(out / 'manifest.json', manifest)
    print(f'COMPLETE: {out}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)

"""Build one shared feature bank, preserving existing sample IDs and captions."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import subprocess

import numpy as np
from tqdm import tqdm

from g1_io import (G1_BANK_DIM, G1_BANK_LAYOUT, csv_values_to_model_features,
                   downsample_features, model_features_to_feature_bank, read_g1_csv)
from g1_to_momask import (captions_from_metadata, iter_segments, iter_windows,
                          read_metadata, read_temporal_labels, safe_id)
from split_utils import split_ids


def convert_source(job):
    row, events, bones, output, expected, split, reference = job
    source = row['move_g1_path']
    try:
        native = downsample_features(csv_values_to_model_features(read_g1_csv(Path(bones) / source)), 120, 20)
        bank, anchors = model_features_to_feature_bank(native)
        combined = np.concatenate((bank, anchors), axis=1)
        segments = iter_segments(combined, captions_from_metadata(row), events, True, 20)
        records = []
        total = np.zeros(G1_BANK_DIM, dtype=np.float64)
        squares = total.copy()
        low = np.full(G1_BANK_DIM, np.inf)
        high = -low.copy()
        frames = 0
        for si, (segment, _) in enumerate(segments):
            for wi, window in enumerate(iter_windows(segment, 196, 40)):
                sample = safe_id(f"{row['filename']}_seg{si:03d}_win{wi:03d}")
                if sample not in expected:
                    continue
                features = window[:, :G1_BANK_DIM].astype(np.float32)
                if not np.isfinite(features).all():
                    raise ValueError(f'Nonfinite features: {sample}')
                np.save(Path(output) / 'new_joint_vecs' / f'{sample}.npy', features)
                np.save(Path(output) / 'anchors' / f'{sample}.npy', window[:, G1_BANK_DIM:])
                (Path(output) / 'texts' / f'{sample}.txt').write_bytes(
                    (Path(reference) / 'texts' / f'{sample}.txt').read_bytes())
                records.append({'sample_id': sample, 'source': source, 'frames': len(window),
                                'split': split, 'initial_state_source': 'ground_truth'})
                if split == 'train':
                    total += features.sum(axis=0, dtype=np.float64)
                    squares += np.square(features.astype(np.float64)).sum(axis=0)
                    low = np.minimum(low, features.min(axis=0))
                    high = np.maximum(high, features.max(axis=0))
                    frames += len(features)
        return records, total, squares, low, high, frames, None
    except Exception as exc:
        return [], None, None, None, None, 0, {'source': source, 'error': str(exc)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bones-root', type=Path, required=True)
    p.add_argument('--reference-root', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--limit', type=int, default=0, help='Source limit for an explicitly partial smoke dataset')
    p.add_argument('--resume-incomplete', action='store_true', help='Rebuild this converter\'s unfinished output')
    args = p.parse_args()
    root = args.output_root
    if root.exists() and any(root.iterdir()):
        previous = json.loads((root / 'dataset_info.json').read_text())
        if not args.resume_incomplete or previous.get('complete') or previous.get('format') != 'bones_g1_feature_bank_v1':
            raise ValueError('Only an explicitly requested incomplete bank can be rebuilt')
        if previous['source_root'] != str(args.bones_root) or previous['reference_root'] != str(args.reference_root):
            raise ValueError('Incomplete bank source paths do not match')
    for name in ('new_joint_vecs', 'anchors', 'texts', 'metadata'):
        (root / name).mkdir(parents=True, exist_ok=True)
    info = {'format': 'bones_g1_feature_bank_v1', 'complete': False,
            'feature_dim': G1_BANK_DIM, 'feature_layout': G1_BANK_LAYOUT,
            'target_fps': 20, 'source_fps': 120, 'velocity_unit': 'per_frame',
            'velocity_frame': 'current_heading', 'difference': 'backward; full-source first frame zero',
            'rotation_convention': 'Rz(yaw) Ry(pitch) Rx(roll)',
            'sincos_order': ['sin_roll', 'cos_roll_minus_one', 'sin_pitch', 'cos_pitch_minus_one'],
            'stats_split': 'train', 'root_position_scale': 0.01,
            'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'source_root': str(args.bones_root), 'reference_root': str(args.reference_root)}
    info['converter_sha256'] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in (Path(__file__), Path(__file__).with_name('g1_io.py'))}
    def save_info():
        (root / 'dataset_info.json').write_text(json.dumps(info, indent=2) + '\n')
    save_info()
    expected = {}
    source_splits = {}
    for split in ('train', 'val', 'test'):
        for sample in (args.reference_root / f'{split}.txt').read_text().splitlines():
            source = sample.rsplit('_seg', 1)[0]
            expected.setdefault(source, set()).add(sample)
            source_splits.setdefault(source.removesuffix('_M'), set()).add(split)
    leakage = sum(len(splits) > 1 for splits in source_splits.values())
    if leakage:
        groups = split_ids(sorted(source_splits), 3407, 0.05, 0.05)
        assignments = {source: split for split, sources in zip(('train', 'val', 'test'), groups) for source in sources}
    else:
        assignments = {source: next(iter(splits)) for source, splits in source_splits.items()}
    info['reference_leaking_source_groups'] = leakage
    info['split_policy'] = 'source_and_mirror_group_seed3407' if leakage else 'preserved_reference'
    metadata = args.bones_root / 'metadata' / 'seed_metadata_v004.csv'
    labels = args.bones_root / 'metadata' / 'seed_metadata_v002_temporal_labels.jsonl'
    info['source_metadata_sha256'] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                      for path in (metadata, labels)}
    temporal = read_temporal_labels(labels)
    rows = []
    seen = set()
    for row in read_metadata(metadata):
        key = safe_id(row.get('filename', ''))
        if key in expected and key not in seen:
            rows.append(row)
            seen.add(key)
    if args.limit:
        rows = rows[:args.limit]
    selected = {safe_id(row['filename']) for row in rows}
    wanted = set().union(*(expected[key] for key in (selected if args.limit else expected)))
    jobs = [(row, temporal.get(row['filename'], []), str(args.bones_root), str(root),
             expected[safe_id(row['filename'])], assignments[safe_id(row['filename']).removesuffix('_M')],
             str(args.reference_root)) for row in rows]
    sums = np.zeros(G1_BANK_DIM, dtype=np.float64)
    squares = sums.copy()
    low, high = np.full(G1_BANK_DIM, np.inf), np.full(G1_BANK_DIM, -np.inf)
    frames, count = 0, 0
    errors, found = [], set()
    splits = {key: [] for key in ('train', 'val', 'test')}
    with (root / 'metadata' / 'windows.jsonl').open('w') as manifest, ProcessPoolExecutor(
            args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        # Bound outstanding jobs/results instead of queuing the entire corpus.
        def results():
            for start in range(0, len(jobs), 64):
                yield from pool.map(convert_source, jobs[start:start + 64], chunksize=1)
        for records, total, sq, lo, hi, n, error in tqdm(results(), total=len(jobs), mininterval=10):
            if error:
                errors.append(error)
                continue
            sums += total
            squares += sq
            low, high = np.minimum(low, lo), np.maximum(high, hi)
            frames += n
            for record in records:
                sample = record['sample_id']
                manifest.write(json.dumps(record) + '\n')
                found.add(sample)
                splits[record['split']].append(sample)
                count += record['frames']
    report = {'samples': len(found), 'sources': len(rows), 'frames': count, 'train_frames': frames,
              'missing_samples': sorted(wanted - found), 'errors': errors,
              'reference_leaking_source_groups': leakage, 'partial_requested': bool(args.limit)}
    (root / 'conversion_report.json').write_text(json.dumps(report, indent=2) + '\n')
    if errors or wanted - found or not frames:
        raise RuntimeError(f'Incomplete conversion; see {root}/conversion_report.json')
    mean = sums / frames
    std = np.sqrt(np.maximum(squares / frames - mean ** 2, 0))
    std[std < 1e-6] = 1
    for name, values in [('Mean', mean), ('Std', std), ('Min', low), ('Max', high)]:
        np.save(root / f'{name}.npy', values.astype(np.float32))
    for split, samples in splits.items():
        (root / f'{split}.txt').write_text(''.join(sample + '\n' for sample in sorted(samples)))
    info.update(complete=True, partial=bool(args.limit), samples=len(found), train_frames=frames,
                split_counts={k: len(v) for k, v in splits.items()})
    info['split_sha256'] = {k: hashlib.sha256((root / f'{k}.txt').read_bytes()).hexdigest() for k in splits}
    save_info()
    print(json.dumps({k: v for k, v in report.items() if k not in ('missing_samples', 'errors')}, indent=2))


if __name__ == '__main__':
    main()

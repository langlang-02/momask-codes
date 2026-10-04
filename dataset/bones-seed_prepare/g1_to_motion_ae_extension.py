"""Add a motion_ae sidecar without changing any existing feature-bank files."""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from g1_io import (G1_FEATURE_COLUMNS, G1_MOTION_AE_EXTENSION_LAYOUT, csv_values_to_model_features,
                   downsample_features, model_features_to_feature_bank,
                   model_features_to_motion_ae_extension)
from g1_to_momask import (captions_from_metadata, iter_segments, iter_windows,
                          read_metadata, read_temporal_labels, safe_id)
from utils.g1_feature_config import feature_bank_signature, load_g1_dataset_info


def read_source_csv(path):
    """Map canonical columns by header, parse numeric rows with NumPy."""
    with path.open(encoding='utf-8-sig', newline='') as stream:
        header = next(csv.reader(stream))
        columns = [header.index(name) for name in G1_FEATURE_COLUMNS]
        return np.loadtxt(stream, delimiter=',', usecols=columns, dtype=np.float32, ndmin=2)


def convert_source(job):
    row, events, bones, base, output, expected = job
    source = row['move_g1_path']
    try:
        native = downsample_features(csv_values_to_model_features(read_source_csv(Path(bones) / source)), 120, 20)
        bank, _ = model_features_to_feature_bank(native)
        extra = model_features_to_motion_ae_extension(native, 20)
        combined = np.concatenate((bank, extra), axis=1)
        found = []
        total = np.zeros(41, np.float64)
        squares = total.copy()
        low, high = np.full(41, np.inf), np.full(41, -np.inf)
        train_frames = 0
        for si, (segment, _) in enumerate(iter_segments(combined, captions_from_metadata(row), events, True, 20)):
            for wi, window in enumerate(iter_windows(segment, 196, 40)):
                sample = safe_id(f"{row['filename']}_seg{si:03d}_win{wi:03d}")
                if sample not in expected:
                    continue
                original = np.load(Path(base) / 'new_joint_vecs' / f'{sample}.npy')
                if original.shape != (len(window), 72) or not np.array_equal(original, window[:, :72]):
                    raise ValueError(f'Base window differs from source conversion: {sample}')
                values = window[:, 72:].astype(np.float32)
                if values.shape != (expected[sample]['frames'], 41) or not np.isfinite(values).all():
                    raise ValueError(f'Invalid extension: {sample}')
                path = Path(output) / 'features' / f'{sample}.npy'
                temporary = path.with_suffix('.tmp')
                with temporary.open('wb') as stream:
                    np.save(stream, values)
                temporary.replace(path)
                found.append(sample)
                if expected[sample]['split'] == 'train':
                    double = values.astype(np.float64)
                    total += double.sum(axis=0)
                    squares += np.square(double).sum(axis=0)
                    low, high = np.minimum(low, double.min(axis=0)), np.maximum(high, double.max(axis=0))
                    train_frames += len(values)
        missing = set(expected) - set(found)
        if missing:
            raise ValueError(f'Missing {len(missing)} windows for {source}')
        return found, total, squares, low, high, train_frames, None
    except Exception as exc:
        return [], None, None, None, None, 0, {'source': source, 'error': str(exc)}


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-root', type=Path, required=True)
    parser.add_argument('--bones-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, help='Default: BASE/extensions/motion_ae_v1')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--limit', type=int, default=0, help='Partial smoke conversion only')
    parser.add_argument('--resume-incomplete', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or args.limit < 0:
        parser.error('workers must be positive and limit nonnegative')
    base = args.base_root.resolve()
    root = (args.output_root or base / 'extensions/motion_ae_v1').resolve()
    if root == base or root in base.parents or (root.is_relative_to(base) and root != base / 'extensions/motion_ae_v1'):
        parser.error('Output must be the extension directory or a separate smoke directory')
    info = load_g1_dataset_info(base)
    if info.get('format') != 'bones_g1_feature_bank_v1' or not info.get('complete') or info.get('partial'):
        raise ValueError('A complete v1 base dataset is required')
    signature = feature_bank_signature(base)
    protected = ['dataset_info.json', 'Mean.npy', 'Std.npy', 'Min.npy', 'Max.npy',
                 'train.txt', 'val.txt', 'test.txt', 'metadata/windows.jsonl']
    hashes = {name: hashlib.sha256((base / name).read_bytes()).hexdigest() for name in protected}
    if root.exists() and any(root.iterdir()):
        previous = load_g1_dataset_info(root)
        if (not args.resume_incomplete or previous.get('complete')
                or previous.get('format') != 'bones_g1_motion_ae_extension_v1'
                or previous.get('base_signature') != signature
                or previous.get('source_root') != str(args.bones_root.resolve())
                or previous.get('source_limit') != args.limit):
            raise ValueError('Refusing to overwrite existing or incompatible extension output')
    (root / 'features').mkdir(parents=True, exist_ok=True)
    lock = root / '.conversion.lock'
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    try:
        metadata = dict(format='bones_g1_motion_ae_extension_v1', complete=False, partial=bool(args.limit),
                        base_signature=signature, base_file_sha256=hashes,
                        source_root=str(args.bones_root.resolve()), source_limit=args.limit,
                        source_fps=120, target_fps=20, feature_dim=41,
                        feature_layout=G1_MOTION_AE_EXTENSION_LAYOUT,
                        velocity_unit='per_second', velocity_frame='current_heading',
                        heading_definition='quaternion_wz_projection',
                        units={'joint_velocity': 'rad/s', 'root_linear_velocity_body': 'm/s',
                               'root_angular_velocity_body': 'rad/s', 'root_rotation_anchor6d': 'dimensionless'},
                        difference='backward on complete downsampled source; first source frame zero',
                        angular_difference='log(R[t] @ R[t-1].T) * fps, rotated by current heading inverse',
                        stats_split='train', converter_sha256={
                            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (Path(__file__), Path(__file__).with_name('g1_io.py'))})
        write_json(root / 'dataset_info.json', metadata)
        expected = {}
        with (base / 'metadata/windows.jsonl').open() as stream:
            for line in stream:
                record = json.loads(line)
                expected.setdefault(record['source'], {})[record['sample_id']] = record
        rows = []
        seen = set()
        for row in read_metadata(args.bones_root / 'metadata/seed_metadata_v004.csv'):
            source = row.get('move_g1_path')
            if source in expected and source not in seen:
                rows.append(row)
                seen.add(source)
        if not args.limit and seen != set(expected):
            raise ValueError('Raw metadata does not cover all base sources')
        if args.limit:
            rows = rows[:args.limit]
        temporal = read_temporal_labels(args.bones_root / 'metadata/seed_metadata_v002_temporal_labels.jsonl')
        sums = np.zeros(41, np.float64)
        squares = sums.copy()
        low, high = np.full(41, np.inf), np.full(41, -np.inf)
        errors, found = [], set()
        frames = 0
        with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            progress = tqdm(total=len(rows), mininterval=30)
            for start in range(0, len(rows), 16):
                jobs = [(row, temporal.get(row['filename'], []), str(args.bones_root), str(base), str(root),
                         expected[row['move_g1_path']]) for row in rows[start:start + 16]]
                for samples, total, sq, lo, hi, n, error in pool.map(convert_source, jobs, chunksize=1):
                    progress.update()
                    if error:
                        errors.append(error)
                        continue
                    found.update(samples)
                    sums += total
                    squares += sq
                    low, high = np.minimum(low, lo), np.maximum(high, hi)
                    frames += n
            progress.close()
        wanted = {s for row in rows for s in expected[row['move_g1_path']]}
        unchanged = all(hashlib.sha256((base / name).read_bytes()).hexdigest() == digest for name, digest in hashes.items())
        report = dict(samples=len(found), sources=len(rows), train_frames=frames, errors=errors,
                      missing_samples=sorted(wanted - found), base_metadata_unchanged=unchanged,
                      all_base_windows_equal=True if not errors else False)
        write_json(root / 'conversion_report.json', report)
        if errors or wanted != found or not frames or not unchanged:
            raise RuntimeError(f'Incomplete extension: {root}/conversion_report.json')
        mean = sums / frames
        std = np.sqrt(np.maximum(squares / frames - mean ** 2, 0))
        std[std < 1e-6] = 1
        for name, values in [('Mean', mean), ('Std', std), ('Min', low), ('Max', high)]:
            np.save(root / f'{name}.npy', values.astype(np.float32))
        metadata.update(complete=True, samples=len(found), sources=len(rows), train_frames=frames)
        write_json(root / 'dataset_info.json', metadata)
        print(json.dumps(report, indent=2))
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()

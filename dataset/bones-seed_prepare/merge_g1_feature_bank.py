"""Pack v1 base and motion_ae sidecars without changing checkpoint signatures."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.g1_feature_config import (
    feature_bank_signature, motion_ae_storage_signature, resolve_representation,
)


def merge(root, workers=8):
    root = Path(root)
    # flock releases automatically on interruption; concurrent publishers are forbidden.
    with (root / '.merge.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return merge_locked(root, workers)


def merge_locked(root, workers):
    repo = Path(__file__).resolve().parents[2]
    resolve_representation(root, repo / 'configs/representations/g1_motion_ae70_v1.yaml')
    marker = root / 'merged_storage.json'
    if marker.exists():
        raise FileExistsError('Merged storage is already published; refusing to overwrite')
    if (root / '.merged_storage.json.tmp').exists():
        return publish_verified_storage(root)
    base_signature = feature_bank_signature(root)
    extension_signature = motion_ae_storage_signature(root)
    extra_root = root / 'extensions/motion_ae_v1'
    ids = []
    for split in ('train', 'val', 'test'):
        ids.extend((root / f'{split}.txt').read_text().splitlines())
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError('Empty dataset or overlapping split IDs')
    for directory in (root / 'new_joint_vecs', extra_root / 'features'):
        if {p.stem for p in directory.glob('*.npy')} != set(ids):
            raise ValueError(f'Sample set mismatch: {directory}')
    stage = root / '.merged_joint_vecs.incomplete'
    final = root / 'merged_joint_vecs'
    if final.exists():
        raise FileExistsError('Unpublished merged_joint_vecs exists; inspect before retrying')
    stage.mkdir(exist_ok=True)

    def pack(sample):
        base = np.load(root / 'new_joint_vecs' / f'{sample}.npy')
        extra = np.load(extra_root / 'features' / f'{sample}.npy')
        if (base.ndim != 2 or base.shape[1] != 72 or extra.shape != (len(base), 41)
                or base.dtype != np.float32 or extra.dtype != np.float32):
            raise ValueError(f'Invalid shape/dtype: {sample}')
        combined = np.concatenate((base, extra), axis=1)
        if not np.isfinite(combined).all():
            raise ValueError(f'Nonfinite values: {sample}')
        path = stage / f'{sample}.npy'
        if not path.exists():
            temporary = path.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                np.save(stream, combined)
            temporary.replace(path)
        stored = np.load(path)
        if stored.dtype != combined.dtype or not np.array_equal(stored, combined):
            raise ValueError(f'Merged file differs from original: {sample}')
        return len(base), hashlib.sha256(stored.tobytes()).hexdigest()

    frames = 0
    digest = hashlib.sha256()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Bounded batches avoid retaining 187k futures and arrays in memory.
        for start in range(0, len(ids), 1024):
            batch = ids[start:start + 1024]
            for sample, (count, checksum) in zip(batch, pool.map(pack, batch)):
                frames += count
                digest.update(f'{sample}:{checksum}\n'.encode())
            print(f'Verified {min(start + 1024, len(ids))}/{len(ids)} samples', flush=True)
    if {p.stem for p in stage.glob('*.npy')} != set(ids):
        raise ValueError('Unexpected samples in staging directory')
    if (base_signature != feature_bank_signature(root)
            or extension_signature != motion_ae_storage_signature(root)):
        raise ValueError('Source metadata changed during merge')
    info = json.loads((root / 'dataset_info.json').read_text())
    ext = json.loads((extra_root / 'dataset_info.json').read_text())
    layout = dict(info['feature_layout'])
    layout.update({key: [a + 72, b + 72] for key, (a, b) in ext['feature_layout'].items()})
    report = dict(format='bones_g1_merged_storage_v1', complete=True, feature_dim=113,
                  feature_layout=layout, base_signature=base_signature,
                  extension_sha256=extension_signature, samples=len(ids), frames=frames,
                  content_sha256=digest.hexdigest(), all_samples_exact=True,
                  normalization='Original base/extension statistics preserved without recomputation',
                  velocity_conventions={'base': 'per_frame, Euler current heading',
                                        'extension': 'per_second, quaternion w/z current heading'})
    final_path = root / '.merged_storage.json.tmp'
    final_path.write_text(json.dumps(report, indent=2) + '\n')
    return publish_verified_storage(root)


def publish_verified_storage(root):
    """EFS may reject a directory rename across its internal storage shards."""
    receipt = root / '.merged_storage.json.tmp'
    report = json.loads(receipt.read_text())
    if (not report.get('complete') or not report.get('all_samples_exact')
            or report.get('base_signature') != feature_bank_signature(root)
            or report.get('extension_sha256') != motion_ae_storage_signature(root)):
        raise ValueError('Invalid or stale verified merge receipt')
    stage = root / '.merged_joint_vecs.incomplete'
    final = root / 'merged_joint_vecs'
    if not stage.is_dir():
        raise FileNotFoundError(stage)
    if final.is_symlink():
        if final.readlink() != Path(stage.name):
            raise ValueError('Merged storage link points to an unexpected location')
    else:
        final.symlink_to(stage.name, target_is_directory=True)
    receipt.replace(root / 'merged_storage.json')
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    merge(args.data_root, args.workers)

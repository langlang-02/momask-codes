"""Shared RVQ input contract and scheduling for both Transformer stages."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from utils.g1_feature_config import (
    load_g1_dataset_info, representation_motion_dir, resolve_g1_feature_dim,
    resolve_representation,
)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare_g1_inputs(opt):
    meta = Path(opt.checkpoints_dir) / opt.dataset_name / opt.vq_name / 'meta'
    spec_path = meta / 'representation.json'
    opt.representation = None
    if spec_path.exists():
        if not opt.representation_config:
            raise ValueError('Feature-bank RVQ requires --representation_config')
        spec = resolve_representation(opt.data_root, opt.representation_config)
        if spec != json.loads(spec_path.read_text()):
            raise ValueError('RVQ representation or dataset signature mismatch')
        opt.representation = spec
        opt.motion_dir = str(representation_motion_dir(opt.data_root, spec))
        dim = spec['feature_dim']
    else:
        if opt.representation_config or load_g1_dataset_info(opt.data_root).get('format') == 'bones_g1_feature_bank_v1':
            raise ValueError('Feature bank requires RVQ representation metadata')
        dim = resolve_g1_feature_dim(opt.data_root)
    mean, std = (np.load(meta / name) for name in ('mean.npy', 'std.npy'))
    if (mean.shape != (dim,) or std.shape != (dim,) or
            not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any()):
        raise ValueError('Invalid RVQ normalization or feature dimension')
    return dim


def select_vq_checkpoint(opt):
    directory = Path(opt.checkpoints_dir) / opt.dataset_name / opt.vq_name / 'model'
    if opt.vq_checkpoint:
        path = Path(opt.vq_checkpoint)
        if not path.is_absolute():
            path = directory / path
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    for name in ('net_best_val.tar', 'net_best_fid.tar', 'latest.tar'):
        path = directory / name
        if path.is_file():
            return path
    raise FileNotFoundError(f'No RVQ checkpoint in {directory}')


def bind_vq_checkpoint(opt, path, checkpoint):
    if getattr(opt, 'representation', None) is not None and checkpoint.get('representation') != opt.representation:
        raise ValueError('RVQ checkpoint representation differs from its metadata')
    meta = Path(opt.checkpoints_dir) / opt.dataset_name / opt.vq_name / 'meta'
    binding = dict(vq_name=opt.vq_name, sha256=file_sha256(path),
                   epoch=int(checkpoint.get('ep', -1)),
                   representation=getattr(opt, 'representation', None),
                   mean_sha256=file_sha256(meta / 'mean.npy'),
                   std_sha256=file_sha256(meta / 'std.npy'))
    opt.training_binding = binding
    if getattr(opt, 'is_main_process', True):
        output = Path(opt.save_root) / 'meta'
        output.mkdir(parents=True, exist_ok=True)
        record = output / 'training_binding.json'
        if record.exists() and json.loads(record.read_text()) != binding:
            raise ValueError('Experiment RVQ binding changed; use a new experiment name')
        record.write_text(json.dumps(binding, indent=2) + '\n')
        for name in ('mean.npy', 'std.npy', 'representation.json'):
            if (meta / name).exists():
                (output / name).write_bytes((meta / name).read_bytes())
    print(f'RVQ frozen: {path}, epoch={binding["epoch"]}, sha256={binding["sha256"]}')


def check_resume_binding(checkpoint, binding):
    if binding is not None and checkpoint.get('training_binding') != binding:
        raise ValueError('Resume checkpoint was trained with a different RVQ/data contract')


def make_scheduler(optimizer, opt, steps_per_epoch):
    if steps_per_epoch <= 0:
        raise ValueError('Empty training loader')
    if opt.warm_up_epochs is None:
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, opt.milestones, gamma=opt.gamma)
    if opt.warm_up_epochs < 0 or any(e <= opt.warm_up_epochs for e in opt.decay_epochs):
        raise ValueError('Decay epochs must follow warmup')
    warmup = opt.warm_up_epochs * steps_per_epoch
    milestones = [e * steps_per_epoch for e in opt.decay_epochs]
    print(f'Epoch schedule: steps/epoch={steps_per_epoch}, warmup={warmup}, milestones={milestones}, '
          f'base_lr={optimizer.param_groups[0]["lr"]:.3e}')

    def factor(step):
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        return opt.gamma ** sum(step >= milestone for milestone in milestones)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)

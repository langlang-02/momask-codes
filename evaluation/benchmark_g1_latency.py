#!/usr/bin/env python3
"""Benchmark the standard G1 text-to-motion pipeline on a resident GPU model."""
import argparse
import json
import sys
import time
from pathlib import Path

START = time.perf_counter()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import gen_g1 as g


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoints-dir', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--res-name', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--runs', type=int, default=20)
    p.add_argument('--warmup', type=int, default=3)
    args = p.parse_args()
    device = torch.device('cuda:0')
    g.fixseed(3407)
    root = args.checkpoints_dir / 'bones_g1'
    m = g.get_opt(root / args.name / 'opt.txt', device)
    v = g.get_opt(root / m.vq_name / 'opt.txt', device)
    r = g.get_opt(root / args.res_name / 'opt.txt', device)
    binding = json.loads((root / args.name / 'meta/training_binding.json').read_text())
    assert binding == json.loads((root / args.res_name / 'meta/training_binding.json').read_text())
    assert r.vq_name == m.vq_name
    rep = binding['representation']
    meta = root / m.vq_name / 'meta'
    assert rep == json.loads((meta / 'representation.json').read_text())
    for key in ('mean', 'std'):
        assert g.file_sha256(meta / (key + '.npy')) == binding[key + '_sha256']
    for opt in (m, v, r):
        opt.checkpoints_dir = str(args.checkpoints_dir)
    m.training_binding = r.training_binding = binding
    m.num_tokens, m.num_quantizers, m.code_dim = v.nb_code, v.num_quantizers, v.code_dim
    v.dim_pose = rep['feature_dim']
    load_start = time.perf_counter()
    vq = g.load_vq_model(v, device, m.vq_checkpoint, binding).to(device).eval()
    mask = g.load_mask_transformer(m, device, 'net_best_val').to(device).eval()
    res = g.load_residual_transformer(r, v, device, 'net_best_val').to(device).eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start
    ready_seconds = time.perf_counter() - START
    mean, std = (np.load(meta / (key + '.npy')) for key in ('mean', 'std'))
    lo, hi = (np.load(args.data_root / (key + '.npy'))[rep['indices']] for key in ('Min', 'Max'))
    prompts = ['A person dances with jazz hands.', 'A person walks forward.', 'A person waves their right hand.']

    @torch.no_grad()
    def generate(length, prompt):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        lengths = torch.tensor([length // (v.stride_t ** v.down_t)], device=device)
        tokens = mask.generate([prompt], lengths, timesteps=18, cond_scale=4,
                               temperature=1., topk_filter_thres=.9, gsample=False)
        tokens = res.generate(tokens, [prompt], lengths, temperature=1., cond_scale=4)
        decoded = vq.forward_decoder(tokens).cpu().numpy()
        t1 = time.perf_counter()
        features = np.clip((decoded * std + mean)[0, :length].astype(np.float32), lo, hi)
        native = g.to_native_g1_features(features, 20, 'per_frame', rep)
        output = g.resample_features(native, 20, 120)
        t2 = time.perf_counter()
        assert np.isfinite(output).all()
        return {'model_ms': (t1-t0)*1000, 'postprocess_ms': (t2-t1)*1000,
                'total_ms': (t2-t0)*1000}

    report = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  cpu_threads=torch.get_num_threads(), batch_size=1, timesteps=18,
                  cond_scale=4, precision='standard model dtypes, no autocast',
                  prompts=prompts, warmup=args.warmup, runs=args.runs,
                  model_load_seconds=load_seconds, script_to_models_ready_seconds=ready_seconds,
                  checkpoints={k: x.generation_checkpoint for k,x in [('rvq',vq),('mask',mask),('residual',res)]},
                  scope='Text encoding + both Transformers + RVQ + CPU transfer + denormalization/clipping + native conversion + 120Hz resampling; excludes disk writes, rendering and RL',results=[])
    report['first_request'] = generate(60, prompts[0])
    for length in (60, 100, 196):
        for i in range(args.warmup):
            generate(length, prompts[i % len(prompts)])
        samples = [generate(length, prompts[i % len(prompts)]) for i in range(args.runs)]
        stats = {key: {label: float(fn([s[key] for s in samples])) for label,fn in
                      [('mean',np.mean),('median',np.median),('p95',lambda x:np.percentile(x,95)),('min',np.min),('max',np.max)]}
                 for key in samples[0]}
        result = dict(frames=length, duration_seconds=length/20, statistics=stats, samples=samples)
        report['results'].append(result)
        print(json.dumps({k:val for k,val in result.items() if k != 'samples'}), flush=True)
    report['peak_gpu_allocated_mib'] = torch.cuda.max_memory_allocated()/2**20
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print('REPORT', args.output, 'LOAD_SECONDS', load_seconds, 'FIRST', report['first_request'], flush=True)


if __name__ == '__main__':
    main()

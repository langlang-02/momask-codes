# G1 RVQ reconstruction evaluation

Run from the repository root with the training `.venv`. No cloud task,
rendering, EGL context or physics rollout is required.

```bash
.venv/bin/python evaluation/eval_g1_rvq.py \
  --data-root /EFS/users/zhaohui.yang/dataset/BONES-G1-feature-bank-v1 \
  --checkpoints-dir /EFS/users/zhaohui.yang/dataset/momask-checkpoints \
  --device cuda:0 --batch-size 32 \
  --output-dir /EFS/users/zhaohui.yang/dataset/momask-rvq-eval/my_run
```

The output directory must not exist. Defaults evaluate all test IDs for
rot6d68, sincos66 and motion_ae70, with both `net_best_val` and `latest`.
Use `--limit 16` for a smoke test, or `--experiments NAME --checkpoints latest`
to select runs. `--mjcf` overrides the sibling Unitree scene_29dof.xml.

## Protocol

- Uses checkpoint representation and normalization, validated against the
  feature-bank signature. Model is in eval mode; all RVQ levels are decoded.
- GT root position/yaw come from **all GT anchors**, with base-bank root tilt
  and joint angles. Predictions receive **only the first GT anchor**. For
  motion_ae the initial heading is the quaternion-w/z projection, as in its
  converter. rot6d/sincos height is predicted; motion_ae height is integrated.
- Equal-length batches; repeat the last frame only to a multiple of RVQ
  downsampling. Padding is discarded before metrics. No random crop or
  generated-frame resampling. Evaluation stays at 20 Hz.
- Float32 inference with TF32 disabled and deterministic cuDNN. A multi-sample
  batch is compared with single-sample inference for each checkpoint; the
  maximum absolute deviation and whether the check ran are in the summary.
- FK uses MuJoCo `mj_kinematics`, without dynamics, limit clipping or contact
  correction. Position metrics use the ordered 29 named hinge **joint anchors**,
  not SMPL joints or arbitrary body centers. Coincident joint anchors are kept.
- Global MPJPE is world-space mean Euclidean error (mm). Root-relative MPJPE
  subtracts each sequence's own root translation per frame, but not rotation.
- Root ADE/FDE use 3D Euclidean distance (m); FDE is the last valid frame.
  Root rotation uses SO(3) geodesic angle, yaw uses wrapped Euler-yaw error
  (degrees). Euler yaw may be ill-conditioned near vertical root orientations.
- Wrist/ankle errors refer to left/right wrist-yaw and ankle-roll joint anchors,
  not palm/sole contact points. Units are mm.
- Limited-hinge angle errors are direct differences (degrees), not modulo 2pi.
  Velocity/acceleration errors use first/second differences of temporally
  unwrapped decoded joint angles, scaled by FPS/FPS squared (rad/s, rad/s^2).
  Stored representation velocity channels are not used as metric predictions.

## Outputs and aggregation

- `manifest.json`: protocol, sample count, split/MJCF hashes, arguments, completion.
- `samples.txt`: exact ordered IDs. `comparison.csv/json`: action-mean comparison.
- Per-checkpoint directory: `identity.json`, `per_sample.csv`, `per_joint.csv`,
  `summary.json`. Identity includes checkpoint SHA256/epoch and representation.
- `action_mean` and `action_p95` summarize per-clip metrics (equal clip weights).
  `pooled` aggregates all valid scalar observations, not a mean of per-clip RMSEs.
  FDE has one observation per clip. Velocity excludes the first frame and
  acceleration excludes the first two. Per-joint results pool valid frames.
- No videos or reconstructed trajectories are written by default. Invalid
  data, changed checkpoints and failed roundtrip/batch checks terminate with
  a traceback; `manifest.complete` remains false. Partial results are not a
  completed benchmark. Do not compare incomplete runs as full-test scores.

The split contains short converted windows, not entire source CSV motions;
its FDE does not establish long-horizon drift or physical trackability.

```bash
.venv/bin/python -m unittest discover -s tests -p test_g1_reconstruction_metrics.py
```

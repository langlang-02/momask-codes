# G1 shared feature bank

The bank stores 72 float32 channels once. Both views use 20 Hz, backward
per-frame deltas, and the **current** frame's yaw-only heading coordinates.
Velocity-named fields are not per-second rates. Multiply by 20 to obtain rates.

| Field | Bank columns | rot6d view | sincos view |
| --- | --- | --- | --- |
| joint_angles (rad) | 0:29 | 29 | 29 |
| joint_vel (rad increment) | 29:58 | 29 | 29 |
| root_roll_pitch_rot6d | 58:64 | 6 | - |
| root_roll_pitch_sincos | 64:68 | - | 4 |
| root_yaw_vel (rad increment) | 68:69 | 1 | 1 |
| root_xy_vel_body (m increment) | 69:71 | 2 | 2 |
| root_height (m) | 71:72 | 1 | 1 |
| Total | 72 | 68 | 66 |

Sincos order is `[sin(roll), cos(roll)-1, sin(pitch), cos(pitch)-1]`.
Rotation convention is `Rz(yaw) Ry(pitch) Rx(roll)`, with canonical pitch.
The neutral sincos vector is zero. Decoding uses atan2(sin, shifted_cos+1);
degenerate pairs produce a warning and a zero-angle fallback.

## Convert

Set DATASET_ROOT to persistent storage; run from the repository root:

```bash
.venv/bin/python dataset/bones-seed_prepare/g1_to_feature_bank.py \
  --bones-root "$DATASET_ROOT/bones-seed-g1" \
  --reference-root "$DATASET_ROOT/BONES-G1-mixed-root" \
  --output-root "$DATASET_ROOT/BONES-G1-feature-bank-v1" --workers 16
```

The converter reads original CSVs, computes features on full downsampled
motions, then recreates reference windows and copies their original captions.
It audits the reference split by source and mirror group. If leakage exists,
it produces a source-group split with seed 3407 and 90/5/5 proportions.
Only training windows contribute to statistics. Constant columns use std=1.
The report lists missing samples and source errors; incomplete data is rejected
by training. `--limit N` creates a explicitly partial smoke bank. To rebuild a
failed conversion, use `--resume-incomplete`; completed banks are never replaced.

`anchors/*.npy` contains per-frame ground-truth `[x,y,z,yaw]`, separate from model
inputs. A cropped window uses its own first-frame anchor and ignores the first
delta. Position and yaw are integrated; height and joint angles are direct.
Joint deltas are auxiliary and are not integrated to reconstruct joint angles.

## Train either view

```bash
REP=g1_local_rot6d68_v1  # alternatively: g1_local_sincos66_v1
.venv/bin/python train_vq.py --dataset_name bones_g1 \
  --g1_data_root "$DATASET_ROOT/BONES-G1-feature-bank-v1" \
  --representation_config "configs/representations/$REP.yaml" \
  --name "rvq_${REP}_seed3407" \
  --checkpoints_dir "$DATASET_ROOT/momask-checkpoints" \
  --num_quantizers 6 --window_size 64 --batch_size 256 --max_epoch 500
```

Only selected columns are retained in dataset memory, and their mean/std are
selected in the same order. The unchanged RVQ loss is normalized SmoothL1 +
0.5 times all-feature temporal-difference loss + 0.02 times commitment loss.
Temporal differences of velocity fields do not represent velocity itself.

Volcano's RVQ entrypoint accepts `REPRESENTATION_CONFIG`; also set G1_DATA_ROOT
and a new VQ_NAME. Use paths as mounted **inside the training container**.
Existing queue/resource YAML files are not changed or submitted by this work.
DDP batch size remains per GPU. Keep global batch and update budget fixed for
comparisons. Transformer training is not yet adapted to the feature-bank views.

## Reconstruct and verify

```bash
.venv/bin/python g1_visualize/reconstruct_g1_rvq.py "$RAW_CSV" \
  --vq_name "rvq_${REP}_seed3407" \
  --checkpoints_dir "$DATASET_ROOT/momask-checkpoints" \
  --checkpoint net_best_val --device cuda:0 --output_dir "$DATASET_ROOT/reconstruction"
.venv/bin/python -m unittest discover -s tests -p test_g1_feature_bank.py
```

Reconstruction automatically reads `meta/representation.json`; bank NPY input
also works when its dataset and anchors remain alongside it. Output explicitly
records use of the input's ground-truth initial XYZ/yaw. Compare decoded physical
errors, not raw validation losses across dimensions. Checkpoints embed the
resolved view and dataset signature; mismatched resume is rejected. Old 38/42D
datasets and checkpoints keep their original path when no view is specified.

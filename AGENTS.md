# MoMask + BONES-SEED Repository Guide

This is a MoMask/PyTorch repository extended for BONES-SEED data, especially
native Unitree G1 text-to-motion. Read this file before changing data formats,
training entry points, Slurm scripts, or G1 visualization.

Current default is local execution. Do not use `srun`, `sbatch`, or Slurm
wrappers unless the user explicitly asks for cluster scripts.

## Repository Map

| Area | Key files | Purpose |
| --- | --- | --- |
| Base MoMask | `models/`, `data/`, `motion_loaders/`, `utils/`, `common/` | RVQ, Mask/Residual Transformers, datasets, normalization and motion utilities. |
| Generic entry points | `train_vq.py`, `train_t2m_transformer.py`, `train_res_transformer.py`, `gen_t2m.py`, `edit_t2m.py` | Original HumanML3D/KIT and BONES-SOMA workflows. |
| G1 conversion | `dataset/bones-seed_prepare/g1_to_momask.py`, `g1_io.py` | BONES-SEED CSV to/from native G1 model features. |
| G1 generation | `gen_g1.py` | Text -> RVQ tokens -> native G1 control trajectory / CSV. |
| G1 visualization | `g1_visualize/` | MuJoCo replay, RVQ reconstruction comparison, and Slurm helpers. |
| Cluster jobs | `slurm/*.sbatch`, `slurm/slurm_note.md` | Dataset conversion, all G1 training stages, generation, and job notes. |
| Runtime state | `dataset/`, `checkpoints/`, `generation/`, `log/`, `bones-seed/` | Large/generated data; ignored by Git. |

The conversion helpers live in a directory named `bones-seed_prepare` (with a
hyphen), so scripts that import them add that directory to `sys.path`. Do not
rename it without updating every such entry point.

## Data Formats and G1 Invariants

### Native G1 model representation (current)

`dataset/BONES-G1-rot6d/new_joint_vecs/*.npy` stores a `(frames, 38)`
`float32` trajectory at 20 Hz:

```text
[root_xyz_m (3), root_rotation_6d (6), hinge_joint_angles_rad (29)]
```

- The 29 named G1 joints are single-DoF MuJoCo hinge joints. Keep them as one
  scalar angle each.
- The floating base is the only 3D rotation. It uses the continuous 6D
  representation (the first two columns of a rotation matrix), not Euler
  angles or a quaternion.
- `dataset/bones-seed_prepare/g1_io.py` is the canonical definition of this
  layout. Use its `G1_FEATURE_DIM`, slices, CSV conversion, 6D decoding, and
  `resample_features`; do not hard-code `38`, `9`, or rotation conversion in
  a second location.
- Root 6D rotation is orthogonalized by Gram--Schmidt before MuJoCo/CSV output;
  root orientation interpolation uses SLERP. Joint angles are interpolated
  after angle unwrapping.

### BONES-SEED CSV

BONES-SEED CSV columns are defined by `G1_FEATURE_COLUMNS` in `g1_io.py`:

```text
root translation: centimetres
root_rotateX/Y/Z: degrees, XYZ Euler convention
29 joint columns: degrees
```

`csv_values_to_model_features()` converts CSV to 38D model features. The
reverse `write_g1_csv()` is the only supported way to export a model output.
Do not directly treat a CSV as model-space NPY data.

### Legacy 35D data and checkpoints

`dataset/BONES-G1/` is the old 35D Euler-root dataset and is deliberately kept
separate. It and checkpoints trained from it are incompatible with the current
38D pipeline. Do not resume a 35D RVQ/Transformer after switching to 6D.

Current 6D defaults are:

```text
dataset root: dataset/BONES-G1-rot6d
RVQ:          bones_g1_rvq_rot6d_nq6
Mask:         bones_g1_mtrans_rot6d
Residual:     bones_g1_res_rot6d
```

## G1 Workflow and Dependency Order

1. Convert raw BONES-SEED G1 CSV data into the 38D dataset.
2. Train the RVQ. It creates normalization metadata and the token space.
3. Train Mask Transformer and Residual Transformer after RVQ. These two stages
   are independent of each other and may run in parallel once RVQ is fixed.
4. Generate from text with all three models, then render with MuJoCo.

The main scripts and defaults are:

```bash
# 1. Data conversion: defaults to dataset/BONES-G1-rot6d
python dataset/bones-seed_prepare/g1_to_momask.py \
  --bones-root bones-seed \
  --metadata bones-seed/metadata/seed_metadata_v004.csv \
  --temporal-labels bones-seed/metadata/seed_metadata_v002_temporal_labels.jsonl \
  --output-root dataset/BONES-G1-rot6d \
  --use-temporal-labels \
  --rebuild-stats

# 2. First RVQ run. Use --is_continue only when latest.tar exists.
python train_vq.py \
  --dataset_name bones_g1 \
  --name bones_g1_rvq_rot6d_nq6 \
  --g1_data_root dataset/BONES-G1-rot6d \
  --gpu_id 0 \
  --batch_size 2048 \
  --max_epoch 500 \
  --num_quantizers 6 \
  --window_size 64

# 3. Run both after RVQ has finished.
python train_t2m_transformer.py \
  --dataset_name bones_g1 \
  --name bones_g1_mtrans_rot6d \
  --vq_name bones_g1_rvq_rot6d_nq6 \
  --g1_data_root dataset/BONES-G1-rot6d \
  --gpu_id 0 \
  --batch_size 2048 \
  --max_epoch 500
python train_res_transformer.py \
  --dataset_name bones_g1 \
  --name bones_g1_res_rot6d \
  --vq_name bones_g1_rvq_rot6d_nq6 \
  --g1_data_root dataset/BONES-G1-rot6d \
  --gpu_id 0 \
  --batch_size 2048 \
  --max_epoch 500 \
  --cond_drop_prob 0.2 \
  --share_weight

# 4. Generate, then render an MP4.
RUN_EXT=g1_t2m/local_test
TEXT_PROMPT='A person brushes off dust from their right shoulder with the left arm.'
python gen_g1.py \
  --dataset_name bones_g1 \
  --name bones_g1_mtrans_rot6d \
  --res_name bones_g1_res_rot6d \
  --g1_data_root dataset/BONES-G1-rot6d \
  --gpu_id 0 \
  --ext "$RUN_EXT" \
  --text_prompt "$TEXT_PROMPT" \
  --motion_length 64 \
  --output_fps 120
MUJOCO_GL=egl python g1_visualize/replay_g1.py \
  --motion generation/$RUN_EXT/g1_csv/sample000_repeat00_len64.csv \
  --output_mp4 generation/$RUN_EXT/sample000_repeat00_len64.mp4
```

Use `--g1_data_root` to override the default dataset location. Checkpoint
experiment names can similarly be overridden with `--name`, `--vq_name`, and
`--res_name`.

`latest.tar` is the resumable training state. Use `--is_continue`/`IS_CONTINUE=1`
only with the same experiment and feature layout. `net_best_val.tar` is the
best native-G1 validation model and is the preferred checkpoint for inference.

For G1, the standard HumanML3D evaluator/FID/R-Precision path is disabled.
Monitor validation reconstruction loss for RVQ and validation token loss /
accuracy for both Transformers; validate text-to-motion semantically by
generating held-out English captions and inspecting MuJoCo videos.

## G1 Generation, Reconstruction, and Visualization

- `gen_g1.py` loads `net_best_val` by default, invokes CLIP text conditioning,
  samples coarse Mask tokens, fills residual RVQ tokens, decodes them, clips
  values to the dataset range when available, and writes 20 Hz NPY plus 120 Hz
  CSV.
- `gen_g1.py` performs text-conditioned G1 generation. Render the generated CSV
  with `g1_visualize/replay_g1.py`.
- `g1_visualize/replay_g1.py` is strictly kinematic: it writes `qpos` and calls
  `mujoco.mj_forward`; it never calls `mj_step`. Ground contact is visual only.
- `g1_visualize/reconstruct_g1_rvq.py` reconstructs a raw G1 CSV or 38D NPY
  using RVQ, with retry-safe checkpoint loading. It puts the actual checkpoint
  epoch in its output directory name.
- `g1_visualize/reconstruct_g1_local.sh` runs reconstruction then a fixed-camera,
  side-by-side original-vs-reconstruction MuJoCo render locally.
- `g1_visualize/compare_g1_reconstruction.py` is the two-panel renderer;
  `visualization/plot_g1_joint_angles.py` writes joint-angle/error plots.

Example RVQ reconstruction after a 6D RVQ checkpoint exists:

```bash
RAW_CSV=<RAW_G1_CSV> \
VQ_NAME=bones_g1_rvq_rot6d_nq6 \
CHECKPOINT=net_best_val \
bash g1_visualize/reconstruct_g1_local.sh
```

## Moving to Another Server

Do not copy generated data blindly into Git. Transfer or recreate these items
outside version control:

```text
bones-seed/                         raw BONES-SEED archive/extracted data
dataset/BONES-G1-rot6d/             converted 38D G1 dataset
checkpoints/bones_g1/               RVQ and Transformer checkpoints
generation/, log/                   optional outputs and TensorBoard logs
../unitree_mujoco/unitree_robots/g1/scene_29dof.xml
```

The default MuJoCo XML is a sibling repository path. On another server either
preserve that sibling layout or set `MJCF=/absolute/path/to/scene_29dof.xml` in
the rendering/reconstruction environment.

Historical Slurm wrappers are kept under `slurm/` for cluster runs, but local
commands are the default in this workspace.

Two environments are used:

- `momask-h800`: PyTorch/CLIP/MoMask training, conversion, generation, and RVQ
  reconstruction. Recreate from `environment.yml`/`requirements.txt` as a base
  and ensure NumPy, SciPy, PyTorch, tensorboard, einops, and CLIP dependencies
  are installed.
- `g1-mujoco`: MuJoCo rendering environment. Recreate with
  `conda env create -f g1_visualize/environment.yml`. On headless GPU nodes use
  `MUJOCO_GL=egl` before importing MuJoCo.

## Development Rules

- Use 4-space Python indentation and standard import grouping. Prefer
  `snake_case` functions/variables and `CamelCase` classes.
- Keep reusable G1 math in `g1_io.py`; entry scripts should stay argument-driven.
- Preserve feature-layout compatibility checks. A change to `G1_FEATURE_DIM`,
  CSV convention, normalization, or tokenization requires a new dataset root
  and new checkpoint experiment names.
- Do not commit datasets, checkpoints, raw BONES-SEED archives, generated MP4s,
  TensorBoard logs, or machine-specific paths/credentials.
- Use `rg` to find references before changing a data layout. For a layout change,
  check converter, training entry points, `utils/get_opt.py`, generation,
  reconstruction, visualization, local wrappers, and documentation together.

## Minimum Validation Before Handoff

For a 6D/G1 data-path change, run the smallest relevant checks:

```bash
# Syntax/import validation in the training environment
python -m py_compile dataset/bones-seed_prepare/g1_io.py \
  dataset/bones-seed_prepare/g1_to_momask.py gen_g1.py \
  train_vq.py train_t2m_transformer.py train_res_transformer.py

# Validate CSV -> 38D -> MuJoCo named-joint mapping; no window or physics step.
python g1_visualize/replay_g1.py --motion <G1_CSV> --dry_run

# Shell syntax for changed local wrappers
bash -n g1_visualize/reconstruct_g1_local.sh
```

Run actual EGL MP4 rendering only on a machine with a working MuJoCo EGL setup.

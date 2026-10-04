# G1 MuJoCo kinematic replay

`replay_g1.py` visualizes native 38-D G1 trajectories with the 29-DoF Unitree
MuJoCo model. It is a kinematic replay tool: it writes base and joint `qpos`
directly, calls `mujoco.mj_forward`, and never calls `mj_step`. The scene floor
is therefore visual only and cannot alter the reconstructed motion.

## Environment

Use a local Python environment with MuJoCo, PyTorch and the MoMask dependencies
installed. If you use Conda:

```bash
conda activate g1-mujoco
```

To create the MuJoCo-only environment:

```bash
conda env create -f g1_visualize/environment.yml
```

For reconstruction plus rendering in one command, use an environment that has
both the training dependencies and MuJoCo. Override the executable with
`MOMASK_PYTHON=/path/to/python` if needed.

The default XML is the sibling Unitree asset:
`../unitree_mujoco/unitree_robots/g1/scene_29dof.xml`. Pass `--mjcf` when it is
located elsewhere.

## Interactive replay

MoMask `.npy` files use 38-D features (root translation in metres, root 6-D
rotation, and 29 joint angles in radians) and default to 20 Hz:

```bash
python g1_visualize/replay_g1.py \
  --motion generation/g1_walk/motions/sample000_repeat00_len120.npy
```

Generated or source G1 CSVs retain BONES-SEED centimetres/degrees and default to 120 Hz:

```bash
python g1_visualize/replay_g1.py \
  --motion generation/g1_walk/g1_csv/sample000_repeat00_len120.csv
```

Use `--loop` to repeat the motion. `--root_euler_order xyz` controls only how
CSV `root_rotateX/Y/Z` values are converted into the model's 6-D root rotation.

## Headless MP4 rendering

On a GPU compute node, set the renderer backend before Python imports MuJoCo:

```bash
MUJOCO_GL=egl python g1_visualize/replay_g1.py \
  --motion generation/g1_walk/g1_csv/sample000_repeat00_len120.csv \
  --output_mp4 generation/g1_walk/g1_replay.mp4 \
  --orbit_degrees 180
```

The default video camera follows the `pelvis` body and starts from the MJCF
scene's configured azimuth (`-130` degrees) and elevation (`-20` degrees).
Set `--camera_azimuth 50 --orbit_degrees 0` for a fixed 50-degree view.
`--orbit_degrees 180` rotates it by 180 degrees around the robot over the full
clip. Use a negative value for the opposite direction. `--camera_elevation`
and `--camera_distance` override the remaining viewpoint parameters.

## Text-to-motion generation and rendering locally

Run `gen_g1.py` directly, then render the generated CSV with `replay_g1.py`.
Each run writes to `generation/g1_t2m/<name>/`.

```bash
RUN_EXT=g1_t2m/local_test

TEXT_PROMPT='A person slowly walks backward and turns to the left side.'
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
  --output_mp4 generation/$RUN_EXT/sample000_repeat00_len64.mp4 \
  --playback_fps 120 \
  --camera_azimuth 50 \
  --camera_elevation -20 \
  --camera_distance 3.0
```

## RVQ reconstruction of a specific action

Run reconstruction with the local Python environment that contains PyTorch and
the MoMask model code. The script reads the latest *completed* checkpoint; if
another process is currently writing `latest.tar`, it retries automatically.

```bash
python g1_visualize/reconstruct_g1_rvq.py \
  bones-seed/g1/csv/230227/come_down_50cm_box_R_002__A219_M.csv \
  --vq_name bones_g1_rvq_rot6d_nq6 \
  --checkpoint latest \
  --device cpu
```

It writes `input.npy`, `reconstruction.npy`, `tokens.npy`, plus 120 Hz CSVs to
`generation/rvq_reconstruction/<motion>_<vq_name>_<checkpoint>_ep####/`. The
`ep####` portion is read from the checkpoint actually loaded, so outputs from
different RVQ epochs never overwrite one another. To reconstruct with the best
native-G1 validation model instead, pass `--checkpoint
net_best_val`. The output `reconstruction.npy` can be rendered with
`replay_g1.py`; switch back to `g1-mujoco` only for MuJoCo viewing or MP4
rendering.

An entire original G1 CSV can be passed directly as the first argument or via
`--raw_csv`. The script
keeps the full duration, resamples its 120 Hz source trajectory to the RVQ's
20 Hz training rate for reconstruction, then writes both input and
reconstruction CSVs at 120 Hz. `input.npy` and `reconstruction.npy` are both
at 20 Hz and are frame-aligned for side-by-side comparison; `source_input.npy`
preserves the pre-resampling trajectory when the input rate differs.

### Reconstruct and render locally

`reconstruct_g1_local.sh` runs RVQ reconstruction and EGL rendering
consecutively on this machine.
The video is always named after the reconstructed NPY: `reconstruction.npy`
and `reconstruction.mp4` are written beside each other.

```bash
VQ_NAME=bones_g1_rvq_rot6d_nq6 \
CHECKPOINT=latest \
bash g1_visualize/reconstruct_g1_local.sh \
  bones-seed/g1/csv/230227/come_down_50cm_box_R_002__A219_M.csv
```

The script defaults to a labelled side-by-side comparison: Ground Truth on the
left and RVQ Reconstruction on the right. It uses the same fixed 50-degree
camera and fixed scene-center look-at point for both panels, so it does not
follow the robot's motion. Override the camera with `CAMERA_AZIMUTH=...`; the
rendered video uses 20 FPS by default, matching the reconstructed MoMask NPY.

Check an input's shape, XML loading and 29 named-joint mapping without opening
a window or rendering:

```bash
python g1_visualize/replay_g1.py \
  --motion dataset/BONES-G1-rot6d/new_joint_vecs/<sample>.npy \
  --dry_run
```

# G1 MuJoCo kinematic replay

`replay_g1.py` visualizes native 38-D G1 trajectories with the 29-DoF Unitree
MuJoCo model. It is a kinematic replay tool: it writes base and joint `qpos`
directly, calls `mujoco.mj_forward`, and never calls `mj_step`. The scene floor
is therefore visual only and cannot alter the reconstructed motion.

## Environment

An isolated `g1-mujoco` Conda environment has been created for this workspace.
Activate it with:

```bash
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate g1-mujoco
```

To reproduce it elsewhere:

```bash
conda env create -f g1_visualize/environment.yml
```

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

For SuperPOD, submit the included GPU rendering job. `MOTION` is required:

```bash
sbatch --export=ALL,\
MOTION=generation/g1_walk/g1_csv/sample000_repeat00_len120.csv,\
OUTPUT_MP4=generation/g1_walk/g1_replay.mp4,\
ORBIT_DEGREES=180 \
g1_visualize/render_g1.sbatch
```

The login node has no GPU/EGL context, so use this Slurm script (or an
interactive GPU allocation) for `MUJOCO_GL=egl` rendering.

## Text-to-motion generation and rendering in one job

`slurm/generate_bones_g1.sbatch` uses one GPU allocation: it first runs
`gen_g1.py` with the Mask Transformer, Residual Transformer and RVQ, then
switches to `g1-mujoco` and renders the generated CSV. Each job writes to a
time-named directory under `generation/g1_t2m/`, including
`generation_info.txt` with the prompt and requested/generated motion length.

```bash
TEXT_PROMPT='A person slowly walks backward and turns to the left side.' \
sbatch --export=ALL,MASK_NAME=bones_g1_mtrans_rot6d,RES_NAME=bones_g1_res_rot6d,MOTION_LENGTH=64 \
  slurm/generate_bones_g1.sbatch
```

## RVQ reconstruction of a specific action

Run reconstruction with the training environment, which contains PyTorch and
the MoMask model code. The script reads the latest *completed* checkpoint; if
the active trainer is currently writing `latest.tar`, it retries automatically.

```bash
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate momask-h800

python g1_visualize/reconstruct_g1_rvq.py \
  --motion dataset/BONES-G1-rot6d/new_joint_vecs/eating_apple_R_003__A235_M_seg001_win000.npy \
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

An entire original G1 CSV can be passed directly to `--motion`. The script
keeps the full duration, resamples its 120 Hz source trajectory to the RVQ's
20 Hz training rate for reconstruction, then writes both input and
reconstruction CSVs at 120 Hz. `input.npy` and `reconstruction.npy` are both
at 20 Hz and are frame-aligned for side-by-side comparison; `source_input.npy`
preserves the pre-resampling trajectory when the input rate differs.

### Reconstruct and render with `srun`

`reconstruct_g1_srun.sh` requests no resources itself; run it in a one-GPU
Slurm allocation. It runs RVQ reconstruction and EGL rendering consecutively.
The video is always named after the reconstructed NPY: `reconstruction.npy`
and `reconstruction.mp4` are written beside each other.

```bash
srun --account=peilab --partition=normal --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --gpus=1 --time=00:20:00 \
  --export=ALL,MOTION=dataset/BONES-G1-rot6d/new_joint_vecs/eating_apple_R_003__A235_M_seg001_win000.npy,VQ_NAME=bones_g1_rvq_rot6d_nq6,CHECKPOINT=latest \
  bash g1_visualize/reconstruct_g1_srun.sh
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

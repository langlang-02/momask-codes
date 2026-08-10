#!/usr/bin/env bash
#SBATCH --job-name=g1_replay
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail

cd /project/peilab/wengzhenglang/momask-codes
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate g1-mujoco

export MUJOCO_GL=egl

MOTION=${MOTION:?Pass a .csv or 38-D .npy trajectory with --export=ALL,MOTION=...}
MJCF=${MJCF:-../unitree_mujoco/unitree_robots/g1/scene_29dof.xml}
OUTPUT_MP4=${OUTPUT_MP4:-g1_visualize/output/g1_replay.mp4}
ROOT_EULER_ORDER=${ROOT_EULER_ORDER:-xyz}
INPUT_FPS=${INPUT_FPS:-}
CLIP_JOINT_LIMITS=${CLIP_JOINT_LIMITS:-0}
ORBIT_DEGREES=${ORBIT_DEGREES:-0}
CAMERA_AZIMUTH=${CAMERA_AZIMUTH:-}
CAMERA_ELEVATION=${CAMERA_ELEVATION:-}
CAMERA_DISTANCE=${CAMERA_DISTANCE:-}

args=(
  --motion "$MOTION"
  --mjcf "$MJCF"
  --output_mp4 "$OUTPUT_MP4"
  --root_euler_order "$ROOT_EULER_ORDER"
  --orbit_degrees "$ORBIT_DEGREES"
)
if [ -n "$INPUT_FPS" ]; then
  args+=(--input_fps "$INPUT_FPS")
fi
if [ -n "$CAMERA_AZIMUTH" ]; then
  args+=(--camera_azimuth "$CAMERA_AZIMUTH")
fi
if [ -n "$CAMERA_ELEVATION" ]; then
  args+=(--camera_elevation "$CAMERA_ELEVATION")
fi
if [ -n "$CAMERA_DISTANCE" ]; then
  args+=(--camera_distance "$CAMERA_DISTANCE")
fi
if [ "$CLIP_JOINT_LIMITS" = "1" ]; then
  args+=(--clip_joint_limits)
fi

python g1_visualize/replay_g1.py "${args[@]}"

#!/usr/bin/env bash
# Run this script inside a one-GPU Slurm allocation with srun.
# It reconstructs a G1 trajectory with RVQ, then renders reconstruction.npy as
# reconstruction.mp4 in the same output directory.

set -euo pipefail

cd /project/peilab/wengzhenglang/momask-codes
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh

MOTION=${MOTION:?Pass a G1 .csv or 38-D .npy trajectory with MOTION=...}
VQ_NAME=${VQ_NAME:-bones_g1_rvq_rot6d_nq6}
CHECKPOINT=${CHECKPOINT:-latest}
DEVICE=${DEVICE:-cuda:0}
OUTPUT_ROOT=${OUTPUT_ROOT:-generation/rvq_reconstruction}
OUTPUT_FPS=${OUTPUT_FPS:-120}
LOAD_RETRIES=${LOAD_RETRIES:-5}
RETRY_DELAY=${RETRY_DELAY:-2}
MJCF=${MJCF:-../unitree_mujoco/unitree_robots/g1/scene_29dof.xml}
CAMERA_AZIMUTH=${CAMERA_AZIMUTH:-50}
CAMERA_ELEVATION=${CAMERA_ELEVATION:--20}
CAMERA_DISTANCE=${CAMERA_DISTANCE:-3.0}
PLAYBACK_FPS=${PLAYBACK_FPS:-20}
WIDTH=${WIDTH:-960}
HEIGHT=${HEIGHT:-720}

motion_filename=$(basename "$MOTION")
motion_stem=${motion_filename%.*}
reconstruction_log=$(mktemp)
trap 'rm -f "$reconstruction_log"' EXIT

conda activate momask-h800
python g1_visualize/reconstruct_g1_rvq.py \
  --motion "$MOTION" \
  --vq_name "$VQ_NAME" \
  --checkpoint "$CHECKPOINT" \
  --device "$DEVICE" \
  --output_dir "$OUTPUT_ROOT" \
  --output_fps "$OUTPUT_FPS" \
  --load_retries "$LOAD_RETRIES" \
  --retry_delay "$RETRY_DELAY" | tee "$reconstruction_log"

reconstruction_dir=$(awk -F= '/^RECONSTRUCTION_DIR=/{path=$2} END{print path}' "$reconstruction_log")
if [ -z "$reconstruction_dir" ]; then
  echo "Reconstruction did not report its output directory" >&2
  exit 1
fi
reconstruction_npy="$reconstruction_dir/reconstruction.npy"
video_path="${reconstruction_npy%.npy}.mp4"

if [ ! -f "$reconstruction_npy" ]; then
  echo "Expected reconstruction was not written: $reconstruction_npy" >&2
  exit 1
fi

conda activate g1-mujoco
export MUJOCO_GL=egl
python g1_visualize/compare_g1_reconstruction.py \
  --reference_motion "$reconstruction_dir/input.npy" \
  --prediction_motion "$reconstruction_npy" \
  --mjcf "$MJCF" \
  --output_mp4 "$video_path" \
  --playback_fps "$PLAYBACK_FPS" \
  --camera_azimuth "$CAMERA_AZIMUTH" \
  --camera_elevation "$CAMERA_ELEVATION" \
  --camera_distance "$CAMERA_DISTANCE" \
  --width "$WIDTH" \
  --height "$HEIGHT"

echo "Reconstruction NPY: $reconstruction_npy"
echo "Side-by-side reconstruction video: $video_path"

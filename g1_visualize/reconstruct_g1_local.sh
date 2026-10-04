#!/usr/bin/env bash
# Reconstruct a G1 trajectory with RVQ, then render a side-by-side MP4.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

VQ_NAME=${VQ_NAME:-bones_g1_rvq_mixed_root_absxyz_nq6}
CHECKPOINTS_DIR=${CHECKPOINTS_DIR:-checkpoints}
CHECKPOINT=${CHECKPOINT:-latest}
DEVICE=${DEVICE:-cuda:0}
OUTPUT_ROOT=${OUTPUT_ROOT:-generation/rvq_reconstruction}
OUTPUT_FPS=${OUTPUT_FPS:-120}
ROOT_POSITION_SCALE=${ROOT_POSITION_SCALE:-0.01}
ROOT_EULER_ORDER=${ROOT_EULER_ORDER:-xyz}
LOAD_RETRIES=${LOAD_RETRIES:-5}
RETRY_DELAY=${RETRY_DELAY:-2}
MJCF=${MJCF:-../unitree_mujoco/unitree_robots/g1/scene_29dof.xml}
CAMERA_AZIMUTH=${CAMERA_AZIMUTH:-50}
CAMERA_ELEVATION=${CAMERA_ELEVATION:--20}
CAMERA_DISTANCE=${CAMERA_DISTANCE:-3.0}
PLAYBACK_FPS=${PLAYBACK_FPS:-20}
WIDTH=${WIDTH:-960}
HEIGHT=${HEIGHT:-486}
if [ -z "${MOMASK_PYTHON:-}" ]; then
  if [ -x "${REPO_ROOT}/.venv/bin/python" ]; then
    MOMASK_PYTHON=${REPO_ROOT}/.venv/bin/python
  else
    MOMASK_PYTHON=python
  fi
fi

if [ "$#" -gt 1 ]; then
  echo "Usage: $0 [RAW_G1_CSV_OR_NPY]" >&2
  exit 1
fi
if [ "$#" -eq 1 ]; then
  if [ -n "${RAW_CSV:-}" ] || [ -n "${MOTION:-}" ]; then
    echo "Pass input either as the first argument, RAW_CSV, or MOTION, not multiple" >&2
    exit 1
  fi
  case "$1" in
    *.csv|*.CSV) RAW_CSV=$1 ;;
    *) MOTION=$1 ;;
  esac
fi

if [ -n "${RAW_CSV:-}" ]; then
  if [ -n "${MOTION:-}" ]; then
    echo "Pass either RAW_CSV or MOTION, not both" >&2
    exit 1
  fi
  motion_args=(--raw_csv "$RAW_CSV")
else
  MOTION=${MOTION:?Pass RAW_CSV=... for original G1 CSV, or MOTION=... for .csv/.npy}
  motion_args=(--motion "$MOTION")
fi

reconstruction_log=$(mktemp)
trap 'rm -f "$reconstruction_log"' EXIT

"${MOMASK_PYTHON}" g1_visualize/reconstruct_g1_rvq.py \
  "${motion_args[@]}" \
  --vq_name "$VQ_NAME" \
  --checkpoints_dir "$CHECKPOINTS_DIR" \
  --checkpoint "$CHECKPOINT" \
  --device "$DEVICE" \
  --output_dir "$OUTPUT_ROOT" \
  --output_fps "$OUTPUT_FPS" \
  --root_position_scale "$ROOT_POSITION_SCALE" \
  --root_euler_order "$ROOT_EULER_ORDER" \
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

export MUJOCO_GL=${MUJOCO_GL:-egl}
"${MOMASK_PYTHON}" g1_visualize/compare_g1_reconstruction.py \
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

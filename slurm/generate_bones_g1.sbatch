#!/usr/bin/env bash
#SBATCH --job-name=bones_g1_generate
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail
cd /project/peilab/wengzhenglang/momask-codes
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate momask-h800

MASK_NAME=${MASK_NAME:-bones_g1_mtrans_rot6d}
RES_NAME=${RES_NAME:-bones_g1_res_rot6d}
G1_DATA_ROOT=${G1_DATA_ROOT:-dataset/BONES-G1-rot6d}
TEXT_PROMPT=${TEXT_PROMPT:-A person is walking forward.}
MOTION_LENGTH=${MOTION_LENGTH:-120}
OUTPUT_FPS=${OUTPUT_FPS:-120}
RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
MJCF=${MJCF:-../unitree_mujoco/unitree_robots/g1/scene_29dof.xml}
CAMERA_AZIMUTH=${CAMERA_AZIMUTH:-50}
CAMERA_ELEVATION=${CAMERA_ELEVATION:--20}
CAMERA_DISTANCE=${CAMERA_DISTANCE:-3.0}
PLAYBACK_FPS=${PLAYBACK_FPS:-120}
WIDTH=${WIDTH:-1280}
HEIGHT=${HEIGHT:-720}

EXT="g1_t2m/${RUN_TIMESTAMP}"
RUN_DIR="generation/${EXT}"
mkdir -p "$RUN_DIR"
{
  printf 'text_prompt: %s\n' "$TEXT_PROMPT"
  printf 'requested_motion_length_20hz: %s\n' "$MOTION_LENGTH"
  printf 'mask_transformer: %s\n' "$MASK_NAME"
  printf 'residual_transformer: %s\n' "$RES_NAME"
  printf 'g1_data_root: %s\n' "$G1_DATA_ROOT"
  printf 'requested_output_fps: %s\n' "$OUTPUT_FPS"
  printf 'run_timestamp: %s\n' "$RUN_TIMESTAMP"
} > "$RUN_DIR/generation_info.txt"

python gen_g1.py \
  --dataset_name bones_g1 \
  --name "$MASK_NAME" \
  --res_name "$RES_NAME" \
  --g1_data_root "$G1_DATA_ROOT" \
  --gpu_id 0 \
  --ext "$EXT" \
  --text_prompt "$TEXT_PROMPT" \
  --motion_length "$MOTION_LENGTH" \
  --output_fps "$OUTPUT_FPS"

shopt -s nullglob
generated_csvs=("$RUN_DIR"/g1_csv/*.csv)
if [ "${#generated_csvs[@]}" -ne 1 ]; then
  echo "Expected exactly one generated CSV in $RUN_DIR/g1_csv, found ${#generated_csvs[@]}" >&2
  exit 1
fi
generated_csv=${generated_csvs[0]}
generated_filename=$(basename "$generated_csv")
generated_length=${generated_filename##*_len}
generated_length=${generated_length%.csv}
video_path="$RUN_DIR/${generated_filename%.csv}.mp4"

printf 'generated_motion_length_20hz: %s\n' "$generated_length" >> "$RUN_DIR/generation_info.txt"
printf 'generated_csv: %s\n' "$generated_csv" >> "$RUN_DIR/generation_info.txt"
printf 'video: %s\n' "$video_path" >> "$RUN_DIR/generation_info.txt"

conda activate g1-mujoco
export MUJOCO_GL=egl
python g1_visualize/replay_g1.py \
  --motion "$generated_csv" \
  --mjcf "$MJCF" \
  --output_mp4 "$video_path" \
  --playback_fps "$PLAYBACK_FPS" \
  --camera_azimuth "$CAMERA_AZIMUTH" \
  --camera_elevation "$CAMERA_ELEVATION" \
  --camera_distance "$CAMERA_DISTANCE" \
  --width "$WIDTH" \
  --height "$HEIGHT"

echo "Generation metadata: $RUN_DIR/generation_info.txt"
echo "Generated video: $video_path"

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

MASK_NAME=${MASK_NAME:-bones_g1_mtrans}
RES_NAME=${RES_NAME:-bones_g1_res}
TEXT_PROMPT=${TEXT_PROMPT:-A person is walking forward.}
MOTION_LENGTH=${MOTION_LENGTH:-120}
EXT=${EXT:-bones_g1_generation}
OUTPUT_FPS=${OUTPUT_FPS:-120}

python gen_g1.py \
  --dataset_name bones_g1 \
  --name "$MASK_NAME" \
  --res_name "$RES_NAME" \
  --gpu_id 0 \
  --ext "$EXT" \
  --text_prompt "$TEXT_PROMPT" \
  --motion_length "$MOTION_LENGTH" \
  --output_fps "$OUTPUT_FPS"

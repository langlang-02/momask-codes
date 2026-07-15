#!/usr/bin/env bash
#SBATCH --job-name=bones_vq
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail
cd /project/peilab/wengzhenglang/momask-codes
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate momask-h800

export OMP_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/mplconfig-${USER}-${SLURM_JOB_ID}
mkdir -p "$MPLCONFIGDIR"

NAME=${NAME:-bones_rvq_nq6}
BATCH_SIZE=${BATCH_SIZE:-256}
MAX_EPOCH=${MAX_EPOCH:-50}
NUM_QUANTIZERS=${NUM_QUANTIZERS:-6}
WINDOW_SIZE=${WINDOW_SIZE:-64}

python train_vq.py \
  --dataset_name bones \
  --name "$NAME" \
  --gpu_id 0 \
  --batch_size "$BATCH_SIZE" \
  --max_epoch "$MAX_EPOCH" \
  --num_quantizers "$NUM_QUANTIZERS" \
  --window_size "$WINDOW_SIZE"

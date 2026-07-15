#!/usr/bin/env bash
#SBATCH --job-name=bones_res
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail
cd /project/peilab/wengzhenglang/momask-codes
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate momask-h800

export OMP_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/mplconfig-${USER}-${SLURM_JOB_ID}
mkdir -p "$MPLCONFIGDIR"

NAME=${NAME:-bones_res}
VQ_NAME=${VQ_NAME:-bones_rvq_nq6}
BATCH_SIZE=${BATCH_SIZE:-64}
MAX_EPOCH=${MAX_EPOCH:-500}
COND_DROP_PROB=${COND_DROP_PROB:-0.2}
SHARE_WEIGHT=${SHARE_WEIGHT:-1}

args=(
  --dataset_name bones
  --name "$NAME"
  --vq_name "$VQ_NAME"
  --gpu_id 0
  --batch_size "$BATCH_SIZE"
  --max_epoch "$MAX_EPOCH"
  --cond_drop_prob "$COND_DROP_PROB"
)
if [ "$SHARE_WEIGHT" = "1" ]; then
  args+=(--share_weight)
fi

python train_res_transformer.py "${args[@]}"

#!/usr/bin/env bash
#SBATCH --job-name=bones_g1_mtrans
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=16:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail
cd /project/peilab/wengzhenglang/momask-codes
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate momask-h800

export OMP_NUM_THREADS=1
NAME=${NAME:-bones_g1_mtrans}
VQ_NAME=${VQ_NAME:-bones_g1_rvq_nq6}
BATCH_SIZE=${BATCH_SIZE:-64}
MAX_EPOCH=${MAX_EPOCH:-500}
IS_CONTINUE=${IS_CONTINUE:-1}

args=(
  --dataset_name bones_g1
  --name "$NAME"
  --vq_name "$VQ_NAME"
  --gpu_id 0
  --batch_size "$BATCH_SIZE"
  --max_epoch "$MAX_EPOCH"
)
if [ "$IS_CONTINUE" = "1" ]; then
  args+=(--is_continue)
fi

python train_t2m_transformer.py "${args[@]}"

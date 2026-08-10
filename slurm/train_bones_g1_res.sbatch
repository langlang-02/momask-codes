#!/usr/bin/env bash
#SBATCH --job-name=bones_g1_res
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
NAME=${NAME:-bones_g1_res_rot6d}
VQ_NAME=${VQ_NAME:-bones_g1_rvq_rot6d_nq6}
G1_DATA_ROOT=${G1_DATA_ROOT:-dataset/BONES-G1-rot6d}
BATCH_SIZE=${BATCH_SIZE:-2048}
MAX_EPOCH=${MAX_EPOCH:-500}
COND_DROP_PROB=${COND_DROP_PROB:-0.2}
SHARE_WEIGHT=${SHARE_WEIGHT:-1}
IS_CONTINUE=${IS_CONTINUE:-0}

args=(
  --dataset_name bones_g1
  --name "$NAME"
  --vq_name "$VQ_NAME"
  --g1_data_root "$G1_DATA_ROOT"
  --gpu_id 0
  --batch_size "$BATCH_SIZE"
  --max_epoch "$MAX_EPOCH"
  --cond_drop_prob "$COND_DROP_PROB"
)
if [ "$SHARE_WEIGHT" = "1" ]; then
  args+=(--share_weight)
fi
if [ "$IS_CONTINUE" = "1" ]; then
  args+=(--is_continue)
fi

python train_res_transformer.py "${args[@]}"

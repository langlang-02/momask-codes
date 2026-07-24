#!/usr/bin/env bash
#SBATCH --job-name=bones_g1_vq
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
export MPLCONFIGDIR=/tmp/mplconfig-${USER}-${SLURM_JOB_ID}
mkdir -p "$MPLCONFIGDIR"
NAME=${NAME:-bones_g1_rvq_nq6}
BATCH_SIZE=${BATCH_SIZE:-1024}
MAX_EPOCH=${MAX_EPOCH:-500}
NUM_QUANTIZERS=${NUM_QUANTIZERS:-6}
WINDOW_SIZE=${WINDOW_SIZE:-64}
IS_CONTINUE=${IS_CONTINUE:-1}
G1_VIS_EVERY_E=${G1_VIS_EVERY_E:-10}
G1_VIS_SAMPLES=${G1_VIS_SAMPLES:-1}

args=(
  --dataset_name bones_g1
  --name "$NAME"
  --gpu_id 0
  --batch_size "$BATCH_SIZE"
  --max_epoch "$MAX_EPOCH"
  --num_quantizers "$NUM_QUANTIZERS"
  --window_size "$WINDOW_SIZE"
  --g1_vis_every_e "$G1_VIS_EVERY_E"
  --g1_vis_samples "$G1_VIS_SAMPLES"
)
if [ "$IS_CONTINUE" = "1" ]; then
  args+=(--is_continue)
fi

python train_vq.py "${args[@]}"

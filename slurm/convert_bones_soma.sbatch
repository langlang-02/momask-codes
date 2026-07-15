#!/usr/bin/env bash
#SBATCH --job-name=bones_convert
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail
cd /project/peilab/wengzhenglang/momask-codes
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate momask-h800

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

BONES_ROOT=${BONES_ROOT:-bones-seed}
METADATA=${METADATA:-bones-seed/metadata/seed_metadata_v004.csv}
TEMPORAL_LABELS=${TEMPORAL_LABELS:-bones-seed/metadata/seed_metadata_v002_temporal_labels.jsonl}
OUTPUT_ROOT=${OUTPUT_ROOT:-dataset/BONES-SOMA}
LIMIT=${LIMIT:-0}
VAL_RATIO=${VAL_RATIO:-0.05}
TEST_RATIO=${TEST_RATIO:-0.05}
SEED=${SEED:-3407}
USE_TEMPORAL=${USE_TEMPORAL:-1}
DROP_MIRRORS=${DROP_MIRRORS:-0}
OVERWRITE=${OVERWRITE:-0}

if [ ! -d "$BONES_ROOT/soma_uniform" ]; then
  echo "soma_uniform/ not found; extracting $BONES_ROOT/soma_uniform.tar.gz"
  tar -xzf "$BONES_ROOT/soma_uniform.tar.gz" -C "$BONES_ROOT"
fi

args=(
  --bones-root "$BONES_ROOT"
  --metadata "$METADATA"
  --output-root "$OUTPUT_ROOT"
  --limit "$LIMIT"
  --val-ratio "$VAL_RATIO"
  --test-ratio "$TEST_RATIO"
  --seed "$SEED"
)

if [ -f "$TEMPORAL_LABELS" ]; then
  args+=(--temporal-labels "$TEMPORAL_LABELS")
fi
if [ "$USE_TEMPORAL" = "1" ]; then
  args+=(--use-temporal-labels)
fi
if [ "$DROP_MIRRORS" = "1" ]; then
  args+=(--drop-mirrors)
fi
if [ "$OVERWRITE" = "1" ]; then
  args+=(--overwrite)
fi

python dataset/bones-seed_prepare/soma_to_hml3d.py "${args[@]}"

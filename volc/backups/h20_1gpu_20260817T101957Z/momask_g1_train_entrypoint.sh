#!/usr/bin/env bash
set -euo pipefail

stage="${1:-}"
if [ -z "${stage}" ]; then
  echo "Usage: $0 {rvq|mask|res}" >&2
  exit 2
fi

export MOMASK_ROOT="${MOMASK_ROOT:-/root/code/momask-codes}"
if [ ! -f "${MOMASK_ROOT}/train_vq.py" ]; then
  for candidate in \
    /root/efs/users/zhaohui.yang/momask-codes \
    /root/efs/users/zhaohui.yang/code/momask-codes \
    /root/code/momask-codes \
    /EFS/users/zhaohui.yang/momask-codes \
    /DRoboticsAILab/zhaohui.yang/momask-codes
  do
    if [ -f "${candidate}/train_vq.py" ]; then
      export MOMASK_ROOT="${candidate}"
      break
    fi
  done
fi

export MOMASK_PYTHON="${MOMASK_PYTHON:-${MOMASK_ROOT}/.venv/bin/python}"
export MOMASK_PIP="${MOMASK_PIP:-${MOMASK_PYTHON} -m pip}"
export G1_DATA_ROOT="${G1_DATA_ROOT:-/root/efs/users/zhaohui.yang/dataset/BONES-G1-mixed-root}"
if [ ! -f "${G1_DATA_ROOT}/dataset_info.json" ]; then
  for candidate in \
    /root/efs/users/zhaohui.yang/dataset/BONES-G1-mixed-root \
    /EFS/users/zhaohui.yang/dataset/BONES-G1-mixed-root \
    /DRoboticsAILab/zhaohui.yang/dataset/BONES-G1-mixed-root \
    /root/code/dataset/BONES-G1-mixed-root
  do
    if [ -f "${candidate}/dataset_info.json" ]; then
      export G1_DATA_ROOT="${candidate}"
      break
    fi
  done
fi

export CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-/root/efs/users/zhaohui.yang/dataset/momask-checkpoints}"
export MOMASK_LOG_ROOT="${MOMASK_LOG_ROOT:-/root/efs/users/zhaohui.yang/dataset/momask-log}"
export MOMASK_HOME="${MOMASK_HOME:-/root/efs/users/zhaohui.yang/dataset/momask-home}"
export PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/momask_pycache}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export HF_HOME="${HF_HOME:-${MOMASK_HOME}/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TORCH_HOME="${TORCH_HOME:-${MOMASK_HOME}/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${MOMASK_HOME}/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${MOMASK_HOME}/pip-cache}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}"
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"
export HOME="${MOMASK_HOME}"

require_path() {
  local path="$1"
  if [ ! -e "${path}" ]; then
    echo "ERROR: required path is missing: ${path}" >&2
    echo "Mounted /root/code contents:" >&2
    find /root/code -maxdepth 3 -mindepth 1 2>/dev/null | sort | head -200 >&2 || true
    echo "Mounted /EFS contents:" >&2
    find /EFS -maxdepth 4 -mindepth 1 2>/dev/null | sort | head -200 >&2 || true
    exit 2
  fi
}

require_any_path() {
  local path
  for path in "$@"; do
    if [ -e "${path}" ]; then
      return 0
    fi
  done
  echo "ERROR: none of these required paths exists:" >&2
  printf '  %s\n' "$@" >&2
  exit 2
}

detect_gpu_count() {
  if [ -n "${MLP_WORKER_GPU:-}" ]; then
    echo "${MLP_WORKER_GPU}"
  elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    python3 - <<'PY'
import os

value = os.environ.get("CUDA_VISIBLE_DEVICES", "")
print(len([item for item in value.split(",") if item.strip()]))
PY
  else
    nvidia-smi -L 2>/dev/null | wc -l
  fi
}

export DDP="${DDP:-0}"
if [ "${DDP}" = "1" ] || [ "${DDP}" = "true" ]; then
  export NPROC_PER_NODE="${NPROC_PER_NODE:-$(detect_gpu_count)}"
else
  export NPROC_PER_NODE=1
fi

run_train() {
  if { [ "${DDP}" = "1" ] || [ "${DDP}" = "true" ]; } && [ "${NPROC_PER_NODE}" -gt 1 ]; then
    "${MOMASK_PYTHON}" -m torch.distributed.run \
      --standalone \
      --nnodes=1 \
      --nproc_per_node="${NPROC_PER_NODE}" \
      "$@"
  else
    "${MOMASK_PYTHON}" "$@"
  fi
}

cd "${MOMASK_ROOT}"

venv_original_root="/DRoboticsAILab/zhaohui.yang/momask-codes"
if [ "${MOMASK_ROOT}" != "${venv_original_root}" ] && [ ! -e "${venv_original_root}" ]; then
  mkdir -p "$(dirname "${venv_original_root}")"
  ln -s "${MOMASK_ROOT}" "${venv_original_root}" 2>/dev/null || true
fi

require_path "${MOMASK_ROOT}/train_vq.py"
require_path "${MOMASK_ROOT}/train_t2m_transformer.py"
require_path "${MOMASK_ROOT}/train_res_transformer.py"
require_path "${G1_DATA_ROOT}/dataset_info.json"
require_path "${G1_DATA_ROOT}/Mean.npy"
require_path "${G1_DATA_ROOT}/Std.npy"
require_path "${G1_DATA_ROOT}/train.txt"
require_path "${G1_DATA_ROOT}/val.txt"

if [ ! -x "${MOMASK_PYTHON}" ]; then
  echo "ERROR: MOMASK_PYTHON is not executable: ${MOMASK_PYTHON}" >&2
  echo "Create or mount ${MOMASK_ROOT}/.venv before submitting the task." >&2
  exit 2
fi
if ! "${MOMASK_PYTHON}" -m pip --version >/dev/null; then
  echo "ERROR: pip module is not available through MOMASK_PYTHON: ${MOMASK_PYTHON}" >&2
  exit 2
fi

mkdir -p "${CHECKPOINTS_DIR}" "${MOMASK_LOG_ROOT}" "${MOMASK_HOME}" \
  "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${TORCH_HOME}" "${XDG_CACHE_HOME}/clip" \
  "${PIP_CACHE_DIR}"

if [ -n "${TENSORBOARD_LOG_PATH:-}" ]; then
  mkdir -p "${TENSORBOARD_LOG_PATH}"
  if [ -L log ] || [ ! -e log ]; then
    ln -sfn "${TENSORBOARD_LOG_PATH}" log
  fi
else
  if [ -L log ] || [ ! -e log ]; then
    ln -sfn "${MOMASK_LOG_ROOT}" log
  fi
fi

echo "=== MoMask BONES-G1 ${stage} preflight ==="
echo "MOMASK_ROOT=${MOMASK_ROOT}"
echo "MOMASK_PYTHON=${MOMASK_PYTHON}"
echo "MOMASK_PIP=${MOMASK_PIP}"
echo "MOMASK_PYTHON_REAL=$(readlink -f "${MOMASK_PYTHON}" 2>/dev/null || true)"
echo "VENV_ORIGINAL_ROOT=${venv_original_root}"
echo "G1_DATA_ROOT=${G1_DATA_ROOT}"
echo "CHECKPOINTS_DIR=${CHECKPOINTS_DIR}"
echo "MOMASK_HOME=${MOMASK_HOME}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "DDP=${DDP}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
nvidia-smi || true

if ! "${MOMASK_PYTHON}" - <<'PY'
import importlib

required = {
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "yaml": "PyYAML",
}
missing = []
for module, package in required.items():
    try:
        importlib.import_module(module)
    except Exception:
        missing.append(package)
if missing:
    print("missing packages:", " ".join(missing))
    raise SystemExit(1)
PY
then
  "${MOMASK_PYTHON}" -m pip install scipy scikit-learn matplotlib PyYAML
fi

"${MOMASK_PYTHON}" - <<'PY'
import json
import os
import torch

root = os.environ["G1_DATA_ROOT"]
with open(os.path.join(root, "dataset_info.json"), "r", encoding="utf-8") as f:
    info = json.load(f)
print("feature_dim", info.get("feature_dim"), "format", info.get("format"))
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu0", torch.cuda.get_device_name(0))
import scipy  # noqa: F401
from scipy import linalg  # noqa: F401
import sklearn  # noqa: F401
import yaml  # noqa: F401
import clip  # noqa: F401
import einops  # noqa: F401
import vector_quantize_pytorch  # noqa: F401
print("scipy", scipy.__version__)
PY

continue_arg=()
if [ "${IS_CONTINUE:-0}" = "1" ]; then
  continue_arg=(--is_continue)
fi

case "${stage}" in
  rvq)
    run_train train_vq.py \
      --dataset_name bones_g1 \
      --g1_data_root "${G1_DATA_ROOT}" \
      --checkpoints_dir "${CHECKPOINTS_DIR}" \
      --name "${VQ_NAME:-bones_g1_rvq_mixed_root_absxyz_nq6}" \
      --gpu_id "${GPU_ID:-0}" \
      --batch_size "${RVQ_BATCH_SIZE:-256}" \
      --max_epoch "${RVQ_MAX_EPOCH:-50}" \
      --num_quantizers "${RVQ_NUM_QUANTIZERS:-6}" \
      --log_every "${LOG_EVERY:-50}" \
      --save_latest "${SAVE_LATEST:-500}" \
      --g1_vis_every_e "${G1_VIS_EVERY_E:-0}" \
      "${continue_arg[@]}"
    ;;
  mask)
    require_any_path \
      "${CHECKPOINTS_DIR}/bones_g1/${VQ_NAME:-bones_g1_rvq_mixed_root_absxyz_nq6}/model/net_best_val.tar" \
      "${CHECKPOINTS_DIR}/bones_g1/${VQ_NAME:-bones_g1_rvq_mixed_root_absxyz_nq6}/model/latest.tar"
    run_train train_t2m_transformer.py \
      --dataset_name bones_g1 \
      --g1_data_root "${G1_DATA_ROOT}" \
      --checkpoints_dir "${CHECKPOINTS_DIR}" \
      --vq_name "${VQ_NAME:-bones_g1_rvq_mixed_root_absxyz_nq6}" \
      --name "${MASK_NAME:-bones_g1_mtrans_mixed_root_absxyz}" \
      --gpu_id "${GPU_ID:-0}" \
      --batch_size "${MASK_BATCH_SIZE:-64}" \
      --max_epoch "${MASK_MAX_EPOCH:-500}" \
      --log_every "${LOG_EVERY:-50}" \
      --save_latest "${SAVE_LATEST:-500}" \
      "${continue_arg[@]}"
    ;;
  res)
    require_any_path \
      "${CHECKPOINTS_DIR}/bones_g1/${VQ_NAME:-bones_g1_rvq_mixed_root_absxyz_nq6}/model/net_best_val.tar" \
      "${CHECKPOINTS_DIR}/bones_g1/${VQ_NAME:-bones_g1_rvq_mixed_root_absxyz_nq6}/model/latest.tar"
    run_train train_res_transformer.py \
      --dataset_name bones_g1 \
      --g1_data_root "${G1_DATA_ROOT}" \
      --checkpoints_dir "${CHECKPOINTS_DIR}" \
      --vq_name "${VQ_NAME:-bones_g1_rvq_mixed_root_absxyz_nq6}" \
      --name "${RES_NAME:-bones_g1_res_mixed_root_absxyz}" \
      --gpu_id "${GPU_ID:-0}" \
      --batch_size "${RES_BATCH_SIZE:-64}" \
      --max_epoch "${RES_MAX_EPOCH:-500}" \
      --log_every "${LOG_EVERY:-50}" \
      --save_latest "${SAVE_LATEST:-500}" \
      "${continue_arg[@]}"
    ;;
  *)
    echo "ERROR: unknown stage: ${stage}" >&2
    exit 2
    ;;
esac

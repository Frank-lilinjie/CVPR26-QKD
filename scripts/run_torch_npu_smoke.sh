#!/usr/bin/env bash
# Legacy PyTorch/torch_npu smoke launcher. Kept for reference reproduction.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

ASCEND_ENV="${ASCEND_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
if [[ -f "${ASCEND_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${ASCEND_ENV}"
fi

export QKD_BACKEND="${QKD_BACKEND:-npu}"
export QKD_DATA_ROOT="${QKD_DATA_ROOT:-${ROOT_DIR}/data}"
export QKD_CHECKPOINT_DIR="${QKD_CHECKPOINT_DIR:-${ROOT_DIR}/checkpoints}"
export QKD_NUM_WORKERS="${QKD_NUM_WORKERS:-0}"

PYTHON_BIN="${PYTHON_BIN:-python}"
QKD_DEVICE="${QKD_DEVICE:-npu:0}"

exec "${PYTHON_BIN}" main.py \
  --config exps/smoke/quan_adapter.json \
  --exp torch_npu_smoke \
  --device "${QKD_DEVICE}" \
  --backend "${QKD_BACKEND}" \
  --num-workers "${QKD_NUM_WORKERS}" \
  "$@"

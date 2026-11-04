#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

ASCEND_ENV="${ASCEND_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
if [[ -f "${ASCEND_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${ASCEND_ENV}"
else
  echo "[WARN] CANN environment script not found: ${ASCEND_ENV}" >&2
  echo "[WARN] Source the CANN set_env.sh for the target host before running." >&2
fi

export QKD_DATA_ROOT="${QKD_DATA_ROOT:-${ROOT_DIR}/data}"
export QKD_CHECKPOINT_DIR="${QKD_CHECKPOINT_DIR:-${ROOT_DIR}/checkpoints}"
mkdir -p "${QKD_DATA_ROOT}" "${QKD_CHECKPOINT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
QKD_DEVICE="${QKD_DEVICE:-npu:0}"
QKD_DEVICE_TARGET="${QKD_DEVICE_TARGET:-ascend}"
QKD_MS_MODE="${QKD_MS_MODE:-pynative}"
QKD_NUM_WORKERS="${QKD_NUM_WORKERS:-0}"

echo "[INFO] framework=mindspore target=${QKD_DEVICE_TARGET} device=${QKD_DEVICE} mode=${QKD_MS_MODE}"
echo "[INFO] data_root=${QKD_DATA_ROOT}"
echo "[INFO] checkpoint_dir=${QKD_CHECKPOINT_DIR}"

exec "${PYTHON_BIN}" -m qkd_mindspore.main \
  --config exps/smoke/mindspore_synthetic.json \
  --exp npu_smoke \
  --device "${QKD_DEVICE}" \
  --device-target "${QKD_DEVICE_TARGET}" \
  --ms-mode "${QKD_MS_MODE}" \
  --num-workers "${QKD_NUM_WORKERS}" \
  "$@"

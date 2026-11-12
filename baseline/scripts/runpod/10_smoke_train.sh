#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

prepare_storage
require_command uv
SMOKE_CORPUS="${WIKIDATA5M_CORPUS}"
if [[ ! -f "${SMOKE_CORPUS}/manifest.json" ]]; then
  SMOKE_CORPUS="${WIKIDATA5M_SMOKE_CORPUS}"
  if [[ ! -f "${SMOKE_CORPUS}/manifest.json" ]]; then
    echo "Full corpus is absent; building a bounded smoke corpus"
    cd "${BASELINE_DIR}"
    uv run wikidata5m-preprocess build \
      "${WIKIDATA5M_ROOT}" \
      --output "${SMOKE_CORPUS}" \
      --stop-id "${SMOKE_PREPROCESS_STOP_ID:-100000}" \
      --workers "${PREPROCESS_WORKERS:-8}" \
      --max-context-blanks 0
  fi
  uv run wikidata5m-preprocess validate "${SMOKE_CORPUS}"
fi

RUN_NAME="${RUN_NAME:-oreo-qwen3-4b-smoke-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OREO_RUN_ROOT}/${RUN_NAME}"
mkdir -p "${OUTPUT_DIR}"

cd "${BASELINE_DIR}"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run oreo-qwen-train \
  --corpus "${SMOKE_CORPUS}" \
  --index "${WIKIDATA5M_ROOT}/indexed" \
  --output "${OUTPUT_DIR}" \
  --embedding-cache "${OREO_EMBEDDING_CACHE}" \
  --model-name "${QWEN_MODEL_NAME}" \
  --embedding-model-name "${QWEN_EMBEDDING_MODEL_NAME}" \
  --kil-boundary 34 \
  --max-length "${MAX_LENGTH:-1024}" \
  --batch-size 1 \
  --embedding-batch-size "${EMBEDDING_BATCH_SIZE:-16}" \
  --dtype bfloat16 \
  --embedding-device "${EMBEDDING_DEVICE:-cuda:0}" \
  --local-files-only \
  --smoke-one-batch \
  2>&1 | tee "${OUTPUT_DIR}/smoke.log"

if [[ "${UPLOAD_TRAINING_RUN:-1}" == "1" ]]; then
  require_command aws
  upload_run_to_s3 "${OUTPUT_DIR}" "${RUN_NAME}"
fi
echo "Smoke run completed: ${OUTPUT_DIR}"


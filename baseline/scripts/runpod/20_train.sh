#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

prepare_storage
require_command uv
[[ -f "${WIKIDATA5M_CORPUS}/manifest.json" ]] || {
  echo "Missing corpus. Run 01_download_data.sh and 02_prepare_corpus.sh first." >&2
  exit 2
}

RUN_NAME="${RUN_NAME:-oreo-qwen3-4b-late-kil-v1}"
OUTPUT_DIR="${OREO_RUN_ROOT}/${RUN_NAME}"
mkdir -p "${OUTPUT_DIR}"

cd "${BASELINE_DIR}"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run oreo-qwen-train \
  --corpus "${WIKIDATA5M_CORPUS}" \
  --index "${WIKIDATA5M_ROOT}/indexed" \
  --output "${OUTPUT_DIR}" \
  --embedding-cache "${OREO_EMBEDDING_CACHE}" \
  --model-name "${QWEN_MODEL_NAME}" \
  --embedding-model-name "${QWEN_EMBEDDING_MODEL_NAME}" \
  --kil-boundary 34 \
  --max-length "${MAX_LENGTH:-2048}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
  --epochs "${EPOCHS:-1}" \
  --learning-rate "${LEARNING_RATE:-2e-4}" \
  --weight-decay "${WEIGHT_DECAY:-0.01}" \
  --warmup-ratio "${WARMUP_RATIO:-0.03}" \
  --max-grad-norm "${MAX_GRAD_NORM:-1.0}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --embedding-batch-size "${EMBEDDING_BATCH_SIZE:-64}" \
  --save-steps "${SAVE_STEPS:-500}" \
  --save-total-limit "${SAVE_TOTAL_LIMIT:-3}" \
  --dtype bfloat16 \
  --embedding-device "${EMBEDDING_DEVICE:-cuda:0}" \
  --local-files-only \
  2>&1 | tee -a "${OUTPUT_DIR}/train.log"

if [[ "${UPLOAD_TRAINING_RUN:-1}" == "1" ]]; then
  require_command aws
  upload_run_to_s3 "${OUTPUT_DIR}" "${RUN_NAME}"
fi
echo "Training completed: ${OUTPUT_DIR}"


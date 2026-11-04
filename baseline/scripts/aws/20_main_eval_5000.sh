#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
require_workspace

INPUT_NAME="raw-qwen-native-main5000-v1"
RUN_NAME="qwen3-0.6b-native-main5000-v1"
INPUT_DIR="${RAW_EVAL_INPUT_ROOT}/${INPUT_NAME}"
RUN_DIR="${RAW_EVAL_RUN_ROOT}/${RUN_NAME}"

cd "${BASELINE_DIR}"

uv run raw-qwen-eval prepare \
  "${WIKIDATA5M_ROOT}" \
  "${INPUT_DIR}" \
  --sample-size 5000 \
  --seed 20261104 \
  --shard-size 250

uv run raw-qwen-eval validate "${INPUT_DIR}"
sync_directory_to_s3 "${INPUT_DIR}" "${KGREALM_S3_URI}/eval-inputs/${INPUT_NAME}"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run raw-qwen-eval run \
  "${INPUT_DIR}" \
  "${RUN_DIR}" \
  --batch-size 4 \
  --max-input-tokens 8192 \
  --max-new-tokens 512

uv run raw-qwen-eval score "${RUN_DIR}"
sync_directory_to_s3 "${RUN_DIR}" "${KGREALM_S3_URI}/eval-runs/${RUN_NAME}"

echo "Main metrics: ${RUN_DIR}/rescored_metrics.json"

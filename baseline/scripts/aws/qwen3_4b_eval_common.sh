#!/usr/bin/env bash
set -euo pipefail

QWEN3_4B_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${QWEN3_4B_SCRIPT_DIR}/common.sh"

QWEN3_4B_MODEL="Qwen/Qwen3-4B"
QWEN3_4B_BATCH_SIZE="${QWEN3_4B_BATCH_SIZE:-2}"

ensure_shared_eval_input() {
  local input_name="$1"
  local input_dir="$2"
  local sample_size="$3"
  local shard_size="$4"

  mkdir -p "${input_dir}"
  if [[ ! -f "${input_dir}/manifest.json" ]]; then
    echo "Downloading shared eval inputs from S3: ${input_name}"
    aws s3 sync \
      "${KGREALM_S3_URI}/eval-inputs/${input_name}/" \
      "${input_dir}/" \
      --only-show-errors
  fi

  if [[ ! -f "${input_dir}/manifest.json" ]]; then
    echo "Shared inputs are absent in S3; preparing them from Wikidata5M."
    uv run raw-qwen-eval prepare \
      "${WIKIDATA5M_ROOT}" \
      "${input_dir}" \
      --sample-size "${sample_size}" \
      --seed 20261104 \
      --shard-size "${shard_size}"
  fi

  uv run raw-qwen-eval validate "${input_dir}"
  sync_directory_to_s3 \
    "${input_dir}" \
    "${KGREALM_S3_URI}/eval-inputs/${input_name}"
}

run_qwen3_4b_eval() {
  local input_name="$1"
  local run_name="$2"
  local sample_size="$3"
  local shard_size="$4"
  local review_limit="${5:-}"
  local input_dir="${RAW_EVAL_INPUT_ROOT}/${input_name}"
  local run_dir="${RAW_EVAL_RUN_ROOT}/${run_name}"

  require_command aws
  require_command uv
  require_workspace
  cd "${BASELINE_DIR}"

  ensure_shared_eval_input \
    "${input_name}" \
    "${input_dir}" \
    "${sample_size}" \
    "${shard_size}"

  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run raw-qwen-eval run \
    "${input_dir}" \
    "${run_dir}" \
    --model "${QWEN3_4B_MODEL}" \
    --batch-size "${QWEN3_4B_BATCH_SIZE}" \
    --max-input-tokens 8192 \
    --max-new-tokens 512

  uv run raw-qwen-eval score "${run_dir}"
  if [[ -n "${review_limit}" ]]; then
    uv run raw-qwen-eval review \
      "${run_dir}/rescored_metrics.json" \
      --limit "${review_limit}"
  else
    uv run raw-qwen-eval review "${run_dir}/rescored_metrics.json"
  fi

  sync_directory_to_s3 \
    "${run_dir}" \
    "${KGREALM_S3_URI}/eval-runs/${run_name}"

  echo "Metrics: ${run_dir}/rescored_metrics.json"
  echo "Review: ${run_dir}/review.html"
}

#!/usr/bin/env bash
set -euo pipefail

RUNPOD_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${RUNPOD_SCRIPT_DIR}/runpod_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${RUNPOD_SCRIPT_DIR}/runpod_env.sh"
fi

export BASELINE_DIR="${BASELINE_DIR:-$(cd "${RUNPOD_SCRIPT_DIR}/../.." && pwd)}"
export RUNPOD_VOLUME_ROOT="${RUNPOD_VOLUME_ROOT:-/workspace}"
export KGREALM_STORAGE="${KGREALM_STORAGE:-${RUNPOD_VOLUME_ROOT}/kgrealm-storage}"
export KGREALM_S3_URI="${KGREALM_S3_URI:-s3://kgrealm-oreo-data-589498925064-us-east-1-an/kgrealm-oreo-data}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${KGREALM_STORAGE}/venv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${KGREALM_STORAGE}/cache/uv}"
export UV_TOOL_DIR="${UV_TOOL_DIR:-${KGREALM_STORAGE}/tools/uv}"
export UV_TOOL_BIN_DIR="${UV_TOOL_BIN_DIR:-${KGREALM_STORAGE}/tools/bin}"
export HF_HOME="${HF_HOME:-${KGREALM_STORAGE}/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_XET_CACHE="${HF_XET_CACHE:-${HF_HOME}/xet}"
export TORCH_HOME="${TORCH_HOME:-${KGREALM_STORAGE}/cache/torch}"
export TMPDIR="${TMPDIR:-${KGREALM_STORAGE}/tmp}"
export PATH="${UV_TOOL_BIN_DIR}:${PATH}"

export WIKIDATA5M_ROOT="${WIKIDATA5M_ROOT:-${KGREALM_STORAGE}/datasets/Wikidata5M}"
export WIKIDATA5M_CORPUS="${WIKIDATA5M_CORPUS:-${WIKIDATA5M_ROOT}/preprocessed/wiki5m_reconstruction_v3}"
export WIKIDATA5M_SMOKE_CORPUS="${WIKIDATA5M_SMOKE_CORPUS:-${WIKIDATA5M_ROOT}/preprocessed/wiki5m_reconstruction_smoke_v3}"
export OREO_RUN_ROOT="${OREO_RUN_ROOT:-${KGREALM_STORAGE}/training-runs}"
export OREO_EMBEDDING_CACHE="${OREO_EMBEDDING_CACHE:-${KGREALM_STORAGE}/embedding-cache/qwen3-embedding-0.6b.sqlite3}"
export QWEN_MODEL_NAME="${QWEN_MODEL_NAME:-Qwen/Qwen3-4B}"
export QWEN_EMBEDDING_MODEL_NAME="${QWEN_EMBEDDING_MODEL_NAME:-Qwen/Qwen3-Embedding-0.6B}"

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Required command is missing: $1" >&2
    exit 2
  }
}

prepare_storage() {
  mkdir -p \
    "${KGREALM_STORAGE}" \
    "${UV_CACHE_DIR}" \
    "${UV_TOOL_DIR}" \
    "${UV_TOOL_BIN_DIR}" \
    "${HF_HUB_CACHE}" \
    "${HF_XET_CACHE}" \
    "${TORCH_HOME}" \
    "${TMPDIR}" \
    "${WIKIDATA5M_ROOT}" \
    "${OREO_RUN_ROOT}" \
    "$(dirname "${OREO_EMBEDDING_CACHE}")"
}

run_aws_s3() {
  if [[ "${S3_NO_SIGN_REQUEST:-0}" == "1" ]]; then
    aws "$@" --no-sign-request
  else
    aws "$@"
  fi
}

upload_run_to_s3() {
  local source="$1"
  local run_name="$2"
  run_aws_s3 s3 sync \
    "${source}/" \
    "${KGREALM_S3_URI}/training-runs/${run_name}/" \
    --exclude ".*.tmp/*" \
    --only-show-errors
}

print_storage_summary() {
  echo "Storage root: ${KGREALM_STORAGE}"
  df -h "${RUNPOD_VOLUME_ROOT}" || true
  du -sh "${KGREALM_STORAGE}" 2>/dev/null || true
}


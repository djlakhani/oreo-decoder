#!/usr/bin/env bash
set -euo pipefail

AWS_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${AWS_SCRIPT_DIR}/aws_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${AWS_SCRIPT_DIR}/aws_env.sh"
fi

: "${KGREALM_REPO:?Set KGREALM_REPO or create scripts/aws/aws_env.sh}"
: "${KGREALM_SCRATCH:?Set KGREALM_SCRATCH}"
: "${KGREALM_S3_URI:?Set KGREALM_S3_URI}"
: "${HF_HUB_CACHE:?Set HF_HUB_CACHE to the existing Qwen cache}"
if [[ "${HF_HUB_CACHE}" == "/REPLACE/WITH/EXISTING/HUGGINGFACE/HUB/CACHE" ]]; then
  echo "Set HF_HUB_CACHE in the environment or scripts/aws/aws_env.sh" >&2
  exit 2
fi

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${KGREALM_SCRATCH}/venv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${KGREALM_SCRATCH}/cache/uv}"
export TMPDIR="${TMPDIR:-${KGREALM_SCRATCH}/tmp}"
export WIKIDATA5M_ROOT="${WIKIDATA5M_ROOT:-${KGREALM_SCRATCH}/datasets/Wikidata5M}"
export RAW_EVAL_INPUT_ROOT="${RAW_EVAL_INPUT_ROOT:-${KGREALM_SCRATCH}/eval-inputs}"
export RAW_EVAL_RUN_ROOT="${RAW_EVAL_RUN_ROOT:-${KGREALM_SCRATCH}/eval-runs}"
export BASELINE_DIR="${KGREALM_REPO}/oreo-decoder/baseline"
export PATH="${KGREALM_SCRATCH}/tools/bin:${PATH}"

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Required command is missing: $1" >&2
    exit 2
  }
}

require_workspace() {
  local mount_source
  mount_source="$(findmnt -n -o SOURCE --target /opt/dlami/nvme)"
  if [[ -z "${mount_source}" ]]; then
    echo "/opt/dlami/nvme is not a mounted filesystem" >&2
    exit 2
  fi
  mkdir -p \
    "${KGREALM_SCRATCH}/datasets" \
    "${RAW_EVAL_INPUT_ROOT}" \
    "${RAW_EVAL_RUN_ROOT}" \
    "${KGREALM_SCRATCH}/tools" \
    "${TMPDIR}" \
    "${UV_CACHE_DIR}"
}

sync_directory_to_s3() {
  local source="$1"
  local destination="$2"
  aws s3 sync "${source}/" "${destination}/" --only-show-errors
}


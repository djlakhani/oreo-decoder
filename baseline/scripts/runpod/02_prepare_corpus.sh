#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

prepare_storage
require_command uv

cd "${BASELINE_DIR}"
if [[ -f "${WIKIDATA5M_CORPUS}/manifest.json" ]]; then
  echo "Validating existing v3 reconstruction corpus"
  uv run wikidata5m-preprocess validate "${WIKIDATA5M_CORPUS}"
else
  echo "Building resumable v3 reconstruction corpus"
  uv run wikidata5m-preprocess build \
    "${WIKIDATA5M_ROOT}" \
    --output "${WIKIDATA5M_CORPUS}" \
    --workers "${PREPROCESS_WORKERS:-8}"
  uv run wikidata5m-preprocess validate "${WIKIDATA5M_CORPUS}"
fi

if [[ "${UPLOAD_PREPROCESSED:-1}" == "1" ]]; then
  require_command aws
  echo "Uploading corpus to S3 without --delete"
  run_aws_s3 s3 sync \
    "${WIKIDATA5M_CORPUS}/" \
    "${KGREALM_S3_URI}/datasets/Wikidata5M/preprocessed/wiki5m_reconstruction_v3/" \
    --only-show-errors
fi

print_storage_summary


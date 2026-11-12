#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

prepare_storage
require_command aws
require_command uv

echo "Downloading WikiData5M from S3 without deleting local or remote objects"
run_aws_s3 s3 sync \
  "${KGREALM_S3_URI}/datasets/Wikidata5M/" \
  "${WIKIDATA5M_ROOT}/" \
  --exclude ".DS_Store" \
  --only-show-errors

cd "${BASELINE_DIR}"
uv run wikidata5m-index validate \
  "${WIKIDATA5M_ROOT}/indexed" \
  --source "${WIKIDATA5M_ROOT}" \
  --full

print_storage_summary


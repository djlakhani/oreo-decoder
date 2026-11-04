#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_command aws
require_command uv
require_workspace

aws s3 sync \
  "${KGREALM_S3_URI}/datasets/Wikidata5M/" \
  "${WIKIDATA5M_ROOT}/" \
  --only-show-errors

cd "${BASELINE_DIR}"
uv run wikidata5m-index validate \
  "${WIKIDATA5M_ROOT}/indexed" \
  --source "${WIKIDATA5M_ROOT}" \
  --full

du -sh "${WIKIDATA5M_ROOT}"


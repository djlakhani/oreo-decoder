#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=qwen3_4b_eval_common.sh
source "${SCRIPT_DIR}/qwen3_4b_eval_common.sh"

run_qwen3_4b_eval \
  "raw-qwen-native-main5000-v1" \
  "qwen3-4b-native-main5000-v1" \
  5000 \
  250 \
  200

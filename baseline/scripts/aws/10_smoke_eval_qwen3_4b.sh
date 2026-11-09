#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=qwen3_4b_eval_common.sh
source "${SCRIPT_DIR}/qwen3_4b_eval_common.sh"

run_qwen3_4b_eval \
  "raw-qwen-native-smoke32-v1" \
  "qwen3-4b-native-smoke32-v1" \
  32 \
  16

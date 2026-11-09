#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The generic setup installs the locked project environment, downloads this
# public checkpoint to NVMe, and verifies config/tokenizer access offline.
export QWEN_MODEL_NAME="Qwen/Qwen3-4B"
export QWEN_DOWNLOAD_MODEL=1

exec "${SCRIPT_DIR}/01_setup_instance.sh"

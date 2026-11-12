#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

prepare_storage
require_command curl
require_command nvidia-smi
require_command python3

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="${UV_TOOL_BIN_DIR}" sh
fi
require_command uv

if ! command -v aws >/dev/null 2>&1; then
  uv tool install awscli
fi
require_command aws

echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
print_storage_summary

cd "${BASELINE_DIR}"
uv sync --frozen --extra test
# common.sh defines the future dataset location for the other RunPod scripts.
# Do not let that default opt into tests which require the dataset to have
# already been downloaded; 01_download_data.sh owns that step.
env -u WIKIDATA5M_ROOT uv run pytest -q

echo "Downloading model snapshots to the persistent volume cache"
uv run python - <<'PY'
import os
from huggingface_hub import snapshot_download

for name in (os.environ["QWEN_MODEL_NAME"], os.environ["QWEN_EMBEDDING_MODEL_NAME"]):
    print(name, snapshot_download(repo_id=name, cache_dir=os.environ["HF_HUB_CACHE"]))
PY

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python - <<'PY'
import os
import torch
from transformers import AutoConfig, AutoTokenizer

for name in (os.environ["QWEN_MODEL_NAME"], os.environ["QWEN_EMBEDDING_MODEL_NAME"]):
    config = AutoConfig.from_pretrained(name, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
    print(name, config.model_type, len(tokenizer))
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("bf16", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
PY

if [[ "${S3_NO_SIGN_REQUEST:-0}" != "1" ]]; then
  aws sts get-caller-identity
fi
run_aws_s3 s3 ls "${KGREALM_S3_URI}/datasets/Wikidata5M/" --summarize
print_storage_summary


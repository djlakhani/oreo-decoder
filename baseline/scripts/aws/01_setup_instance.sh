#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

QWEN_MODEL_NAME="${QWEN_MODEL_NAME:-Qwen/Qwen3-0.6B}"
QWEN_DOWNLOAD_MODEL="${QWEN_DOWNLOAD_MODEL:-0}"
export QWEN_MODEL_NAME

require_command aws
require_command nvidia-smi
require_command findmnt
require_command python3
require_workspace

if ! command -v uv >/dev/null 2>&1; then
  python3 -m venv "${KGREALM_SCRATCH}/tools"
  "${KGREALM_SCRATCH}/tools/bin/python" -m pip install --upgrade pip uv
fi
require_command uv

echo "NVMe workspace:"
df -hT /opt/dlami/nvme

echo "Root volume:"
df -hT /

echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "S3 identity:"
aws sts get-caller-identity

echo "S3 prefix:"
aws s3 ls "${KGREALM_S3_URI}/"

cd "${BASELINE_DIR}"
uv sync --frozen --no-dev

uv run python - <<'PY'
import torch
import transformers

print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY

if [[ "${QWEN_DOWNLOAD_MODEL}" == "1" ]]; then
  echo "Downloading ${QWEN_MODEL_NAME} into ${HF_HUB_CACHE}:"
  uv run python - <<'PY'
import os
from huggingface_hub import snapshot_download

name = os.environ["QWEN_MODEL_NAME"]
snapshot = snapshot_download(repo_id=name, cache_dir=os.environ["HF_HUB_CACHE"])
print("model_snapshot", snapshot)
PY
fi

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python - <<'PY'
import os
from transformers import AutoConfig, AutoTokenizer

name = os.environ["QWEN_MODEL_NAME"]
config = AutoConfig.from_pretrained(name, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
print("cached_model", name)
print("cached_model_type", config.model_type)
print("cached_tokenizer_size", len(tokenizer))
PY

echo "Hugging Face cache usage:"
du -sh "${HF_HOME}"


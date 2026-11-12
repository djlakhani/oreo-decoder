#!/usr/bin/env bash
# Copy to runpod_env.sh. Store AWS/Hugging Face credentials as RunPod secrets,
# not in this file or in git.

export RUNPOD_VOLUME_ROOT="/workspace"
export KGREALM_STORAGE="${RUNPOD_VOLUME_ROOT}/kgrealm-storage"
export KGREALM_S3_URI="s3://kgrealm-oreo-data-589498925064-us-east-1-an/kgrealm-oreo-data"
export AWS_DEFAULT_REGION="us-east-1"

# Optional overrides:
# export HF_TOKEN="..."
# export AWS_ACCESS_KEY_ID="..."
# export AWS_SECRET_ACCESS_KEY="..."
# export AWS_SESSION_TOKEN="..."
# export S3_NO_SIGN_REQUEST=1  # public read only; cannot upload checkpoints


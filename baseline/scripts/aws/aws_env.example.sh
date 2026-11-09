#!/usr/bin/env bash
# Copy this file to aws_env.sh. Model and package caches live on NVMe so a
# fresh instance does not consume the comparatively small root volume.

export OREO_DECODER_REPO="/home/ubuntu/oreo-decoder"
export KGREALM_SCRATCH="/opt/dlami/nvme/kgrealm"
export KGREALM_S3_URI="s3://kgrealm-oreo-data-589498925064-us-east-1-an/kgrealm-oreo-data"
export HF_HOME="${KGREALM_SCRATCH}/cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_XET_CACHE="${HF_HOME}/xet"

# Keep new environments, package caches, and temporary files off the root disk.
export UV_PROJECT_ENVIRONMENT="${KGREALM_SCRATCH}/venv"
export UV_CACHE_DIR="${KGREALM_SCRATCH}/cache/uv"
export TMPDIR="${KGREALM_SCRATCH}/tmp"

export WIKIDATA5M_ROOT="${KGREALM_SCRATCH}/datasets/Wikidata5M"
export RAW_EVAL_INPUT_ROOT="${KGREALM_SCRATCH}/eval-inputs"
export RAW_EVAL_RUN_ROOT="${KGREALM_SCRATCH}/eval-runs"


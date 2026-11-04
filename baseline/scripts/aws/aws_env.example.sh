#!/usr/bin/env bash
# Copy this file to aws_env.sh and set HF_HUB_CACHE to the existing Qwen cache.

export KGREALM_REPO="/home/ubuntu/KGReaLM"
export KGREALM_SCRATCH="/opt/dlami/nvme/kgrealm"
export KGREALM_S3_URI="s3://kgrealm-oreo-data-589498925064-us-east-1-an/kgrealm-oreo-data"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/REPLACE/WITH/EXISTING/HUGGINGFACE/HUB/CACHE}"

# Keep new environments, package caches, and temporary files off the root disk.
export UV_PROJECT_ENVIRONMENT="${KGREALM_SCRATCH}/venv"
export UV_CACHE_DIR="${KGREALM_SCRATCH}/cache/uv"
export TMPDIR="${KGREALM_SCRATCH}/tmp"

export WIKIDATA5M_ROOT="${KGREALM_SCRATCH}/datasets/Wikidata5M"
export RAW_EVAL_INPUT_ROOT="${KGREALM_SCRATCH}/eval-inputs"
export RAW_EVAL_RUN_ROOT="${KGREALM_SCRATCH}/eval-runs"


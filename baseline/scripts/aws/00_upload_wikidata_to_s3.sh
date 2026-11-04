#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /absolute/path/to/datasets/Wikidata5M" >&2
  exit 2
fi

SOURCE_ROOT="$1"
S3_URI="${KGREALM_S3_URI:-s3://kgrealm-oreo-data-589498925064-us-east-1-an/kgrealm-oreo-data}"

if [[ ! -f "${SOURCE_ROOT}/indexed/manifest.json" ]]; then
  echo "Wikidata5M index manifest not found under ${SOURCE_ROOT}" >&2
  exit 2
fi

aws s3 sync \
  "${SOURCE_ROOT}/" \
  "${S3_URI}/datasets/Wikidata5M/" \
  --exclude ".DS_Store" \
  --only-show-errors

aws s3 ls "${S3_URI}/datasets/Wikidata5M/indexed/manifest.json"


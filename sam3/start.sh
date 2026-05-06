#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$(cd "${SCRIPT_DIR}/../data" && pwd)"
CACHE_DIR="${HF_CACHE_DIR:-${HOME}/.cache/huggingface}"

mkdir -p "${CACHE_DIR}"

docker run \
  --gpus "device=0" \
  --rm \
  -it \
  --name sam3_inference \
  -v "${CACHE_DIR}:/root/.cache/huggingface" \
  -v "${SCRIPT_DIR}:/workspace/sam3" \
  -w /workspace/sam3 \
  -v "${DATA_DIR}:/workspace/sam3/data" \
  sam3:latest \
  "${@:-bash}"

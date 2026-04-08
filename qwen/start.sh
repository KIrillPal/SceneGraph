#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-qwen-transformers}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen-transformers}"
PORT="${PORT:-8000}"
CACHE_DIR="${HF_CACHE_DIR:-${HOME}/.cache/huggingface}"

mkdir -p "${CACHE_DIR}"

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

docker run --rm -it \
  --name "${CONTAINER_NAME}" \
  --gpus "device=0" \
  --ipc=host \
  -p "${PORT}:8000" \
  -v "${CACHE_DIR}:/root/.cache/huggingface" \
  -v "${REPO_DIR}:/workspace" \
  -w /workspace \
  "${IMAGE_NAME}" \
  bash

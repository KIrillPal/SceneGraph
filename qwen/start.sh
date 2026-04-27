#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-qwen-vllm}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen-vllm}"
PORT="${PORT:-8000}"
CACHE_DIR="${HF_CACHE_DIR:-${HOME}/.cache/huggingface}"
ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-/workspace}"

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
  --model Qwen/Qwen3.5-4B \
  --allowed-local-media-path "${ALLOWED_LOCAL_MEDIA_PATH}"

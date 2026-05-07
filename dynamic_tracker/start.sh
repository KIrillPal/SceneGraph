#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-dynamic-tracker}"
CONTAINER_NAME="${CONTAINER_NAME:-dynamic-tracker}"
GPU_DEVICE="${GPU_DEVICE:-0}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${HOME}/.cache/huggingface}"
COTRACKER_CACHE_DIR="${COTRACKER_CACHE_DIR:-${HOME}/.cache/cotracker}"
COTRACKER_CHECKPOINT="${COTRACKER_CHECKPOINT:-/root/.cache/cotracker/scaled_online.pth}"

mkdir -p "${HF_CACHE_DIR}" "${COTRACKER_CACHE_DIR}"

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

docker run --rm -it \
  --name "${CONTAINER_NAME}" \
  --user "$(id -u):$(id -g)" \
  --gpus "device=${GPU_DEVICE}" \
  --ipc=host \
  -e "COTRACKER_CHECKPOINT=${COTRACKER_CHECKPOINT}" \
  -v "${REPO_DIR}:/workspace" \
  -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
  -v "${COTRACKER_CACHE_DIR}:/root/.cache/cotracker" \
  -w /workspace \
  "${IMAGE_NAME}" \
  "${@:-bash}"

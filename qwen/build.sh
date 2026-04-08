#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${IMAGE_NAME:-qwen-transformers}"

docker build \
  --tag "${IMAGE_NAME}" \
  "${SCRIPT_DIR}"

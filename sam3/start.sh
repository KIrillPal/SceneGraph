#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker run \
  --gpus all \
  --rm \
  -it \
  --name sam3_inference \
  -v "${SCRIPT_DIR}:/workspace/sam3" \
  -w /workspace/sam3 \
  sam3:latest \
  "${@:-bash}"

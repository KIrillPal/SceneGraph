#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker run \
  --gpus "device=0" \
  --rm \
  -it \
  --name depth_anything_3_streaming \
  -v "${SCRIPT_DIR}:/workspace/da3_streaming" \
  -v "${SCRIPT_DIR}/.cache:/workspace/.cache" \
  depth_anything_3_streaming:latest

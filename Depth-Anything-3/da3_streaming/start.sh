#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve the path to the 'data' folder (2 levels up from SCRIPT_DIR)
DATA_DIR="$(cd "${SCRIPT_DIR}/../../data" && pwd)"

docker run \
  --gpus "device=0" \
  --rm \
  -it \
  --name depth_anything_3_streaming \
  -v "${SCRIPT_DIR}:/workspace/da3_streaming" \
  -v "${SCRIPT_DIR}/.cache:/workspace/.cache" \
  -v "${DATA_DIR}:/workspace/da3_streaming/data" \
  depth_anything_3_streaming:latest

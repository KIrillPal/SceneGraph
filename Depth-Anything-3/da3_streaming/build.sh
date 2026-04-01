#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

docker build \
  -t depth_anything_3_streaming:latest \
  -f "${SCRIPT_DIR}/Dockerfile" \
  "${REPO_ROOT}"

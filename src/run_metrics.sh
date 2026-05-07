#!/usr/bin/env bash
set -euo pipefail

# Dockerized recon_metrics CLI — same argv as: python -m recon_metrics
#
# Code: bind-mount `src/metrics` as `recon_metrics` in the image (includes TAE); not the whole repo.
# Data: bind-mounts only paths from --scene-dir, --da3-dir, --batch-da3-dir, --output-dir
# at the same absolute paths inside the container (Linux).
# GPU: docker run --gpus all (override with DOCKER_GPUS=0 on hosts without NVIDIA toolkit).
#
# Example:
#   bash src/run_metrics.sh \
#     --scene-dir /home/you/mipt/data/.../Validation \
#     --batch-da3-dir /home/you/mipt/data/.../da3 \
#     --output-dir /home/you/mipt/tmp/batch \
#     --plot-charts --plot-rrd

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_RS="${ROOT_DIR}/src/metrics"
IMAGE_NAME="${IMAGE_NAME:-scenegraph-metrics:latest}"
DOCKER_GPUS="${DOCKER_GPUS:-all}"

_abs_path() {
	local p="$1"
	if command -v realpath >/dev/null 2>&1; then
		realpath -m "$p"
	else
		python3 -c "import os, sys; print(os.path.abspath(sys.argv[1]))" "$p"
	fi
}

MOUNT_SEEN=()
EXTRA_VOLS=()
add_bind() {
	local ap="$1"
	[[ -z "$ap" ]] && return
	local s
	for s in "${MOUNT_SEEN[@]}"; do
		[[ "$s" == "$ap" ]] && return
	done
	MOUNT_SEEN+=("$ap")
	EXTRA_VOLS+=(-v "$ap:$ap")
}

# Parse "$@" for path flags and add -v host:host (deduped).
args=("$@")
n=${#args[@]}
i=0
while ((i < n)); do
	case "${args[i]}" in
	--scene-dir | --da3-dir | --batch-da3-dir | --output-dir)
		key="${args[i]}"
		((i++)) || true
		val="${args[i]:-}"
		if ((i < n)) && [[ -n "$val" && "${val:0:1}" != - ]]; then
			if [[ "$key" == --output-dir ]]; then
				mkdir -p "$val" 2>/dev/null || true
			fi
			add_bind "$(_abs_path "$val")"
		fi
		;;
	esac
	((i++)) || true
done

[[ -d "$CODE_RS" ]] || {
	echo "run_metrics.sh: missing package dir: $CODE_RS" >&2
	exit 1
}

DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-0}" docker build \
	-f "${CODE_RS}/Dockerfile.recon_metrics" \
	-t "$IMAGE_NAME" \
	"${CODE_RS}"

GPU_FLAGS=()
if [[ "$DOCKER_GPUS" != "0" && "$DOCKER_GPUS" != "false" ]]; then
	GPU_FLAGS=(--gpus "$DOCKER_GPUS")
fi

docker run --rm \
	"${GPU_FLAGS[@]}" \
	-v "${CODE_RS}:/workspace/tmp_scripts/recon_metrics" \
	"${EXTRA_VOLS[@]}" \
	-w /workspace \
	"$IMAGE_NAME" "$@"

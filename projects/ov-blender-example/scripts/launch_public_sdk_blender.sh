#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
task_source_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
task_blender="${OV_BLENDER_SPIKE_BLENDER:-${BLENDER_COMMAND:-blender}}"
export OV_BLENDER_PUBLIC_SDK_PYTHON="${OV_BLENDER_PUBLIC_SDK_PYTHON:-$task_source_root/../out/public-sdk-venv/bin/python}"
if ! command -v "$task_blender" >/dev/null 2>&1; then
  echo 'Blender was not found. Set OV_BLENDER_SPIKE_BLENDER or BLENDER_COMMAND to your Blender 5.2 executable.' >&2
  exit 2
fi
if [[ ! -x "$OV_BLENDER_PUBLIC_SDK_PYTHON" ]]; then
  echo 'External SDK Python was not found. Set OV_BLENDER_PUBLIC_SDK_PYTHON to the verified runtime interpreter.' >&2
  exit 2
fi
export OV_BLENDER_PUBLIC_SDK_OUTPUT="${OV_BLENDER_PUBLIC_SDK_OUTPUT:-$task_source_root/../out/public-sdk-spike}"
export BLENDER_USER_CONFIG="$OV_BLENDER_PUBLIC_SDK_OUTPUT/blender-config"
export BLENDER_USER_SCRIPTS="$OV_BLENDER_PUBLIC_SDK_OUTPUT/blender-scripts"
export OV_BLENDER_EXAMPLE_SCENE_GENERATION_DIR="$OV_BLENDER_PUBLIC_SDK_OUTPUT/scene-generations"
export OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR="$OV_BLENDER_PUBLIC_SDK_OUTPUT/render-composition"
if [[ -z "${OV_BLENDER_PUBLIC_SDK_GPU+x}" && -z "${OVRTX_ACTIVE_CUDA_GPUS+x}" ]]; then
  export OV_BLENDER_PUBLIC_SDK_GPU=0
fi
export PYTHONPATH="$task_source_root/addon${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$BLENDER_USER_CONFIG" "$BLENDER_USER_SCRIPTS" "$OV_BLENDER_PUBLIC_SDK_OUTPUT"
exec "$task_blender" --factory-startup "$@"

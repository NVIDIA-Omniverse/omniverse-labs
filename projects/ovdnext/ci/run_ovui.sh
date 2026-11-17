#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
python_binary=${OVD_PYTHON:-"$root/_ci/venv/bin/python"}
native_root=${OVD_NATIVE_ROOT:-"$root/_ci/build/ovui/python"}
export PYTHONPATH="$root/python:$native_root${PYTHONPATH:+:$PYTHONPATH}"
export OVUIINSPECT_ENABLED=${OVUIINSPECT_ENABLED:-0}
# GLVND may choose Mesa for a virtual X display even on an NVIDIA host.
# Prefer the installed NVIDIA GLX vendor, while preserving an explicit override.
if [[ -z "${__GLX_VENDOR_LIBRARY_NAME+x}" && -d /proc/driver/nvidia/gpus ]]; then
    export __GLX_VENDOR_LIBRARY_NAME=nvidia
fi
exec "$python_binary" -m ovd_ui "$@"

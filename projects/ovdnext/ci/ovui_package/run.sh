#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
bundle=$(cd "$(dirname "$0")" && pwd)
python_binary=${OVD_PYTHON:-"$bundle/venv/bin/python"}
if [[ -z "${__GLX_VENDOR_LIBRARY_NAME+x}" && -d /proc/driver/nvidia/gpus ]]; then
    export __GLX_VENDOR_LIBRARY_NAME=nvidia
fi
exec "$python_binary" -I -m ovd_ui "$@"

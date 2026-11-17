#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
bundle=$(cd "$(dirname "$0")" && pwd)
destination=${1:-"$bundle/venv"}
destination=$(realpath -m -- "$destination")
[[ ! -e "$destination" && ! -L "$destination" ]] || { echo 'Choose a new venv destination.' >&2; exit 1; }
cd "$bundle"
sha256sum --check SHA256SUMS
uv venv --python "${OVD_INSTALL_PYTHON:-3.12}" --no-python-downloads "$destination"
uv pip install --python "$destination/bin/python" --offline --no-index --require-hashes \
    --find-links "$bundle/wheelhouse" -r "$bundle/requirements.lock"
uv pip check --python "$destination/bin/python"
printf 'Launch with %s/bin/pvd-ovui CAPTURE, or use run.sh for the default venv.\n' "$destination"

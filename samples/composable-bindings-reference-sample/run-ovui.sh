#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${OVUI_PYTHON:-python3.12}"
export PYTHONPATH="$ROOT/source${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" -m clients.ovui_client "$@"

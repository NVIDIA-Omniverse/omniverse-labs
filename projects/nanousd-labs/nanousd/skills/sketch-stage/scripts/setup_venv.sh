#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Create / refresh the sketch-stage venv with required deps.
#
# Idempotent: if the venv already exists with all packages, it's a no-op.
# Otherwise it creates the venv and installs from requirements.txt.
#
# Override location with:
#   SKETCH_STAGE_VENV=/path/to/venv bash setup_venv.sh
set -euo pipefail

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_VENV="${XDG_CACHE_HOME:-$HOME/.cache}/sketch-stage/venv"
VENV_DIR="${SKETCH_STAGE_VENV:-$DEFAULT_VENV}"
REQ_FILE="$SKILL_DIR/requirements.txt"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[setup_venv] creating venv at $VENV_DIR" >&2
  python3 -m venv "$VENV_DIR"
fi

PYBIN="$VENV_DIR/bin/python"
"$PYBIN" -m pip install --upgrade pip >/dev/null

# Re-install only when something is missing or requirements.txt is newer
NEEDS_INSTALL=0
if [[ ! -f "$VENV_DIR/.requirements_hash" ]]; then
  NEEDS_INSTALL=1
else
  CURRENT_HASH=$(sha256sum "$REQ_FILE" | awk '{print $1}')
  STORED_HASH=$(cat "$VENV_DIR/.requirements_hash")
  [[ "$CURRENT_HASH" != "$STORED_HASH" ]] && NEEDS_INSTALL=1
fi

if [[ "$NEEDS_INSTALL" = "1" ]]; then
  echo "[setup_venv] installing requirements from $REQ_FILE" >&2
  "$PYBIN" -m pip install -r "$REQ_FILE"
  sha256sum "$REQ_FILE" | awk '{print $1}' > "$VENV_DIR/.requirements_hash"
fi

echo "$VENV_DIR"

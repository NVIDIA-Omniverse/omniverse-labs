#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# build.sh — clean release build of this repo.
#
# *** macOS only.*** Metal is Apple-platform; on Linux/other this script
# exits 0 with a "skipped" message so the workspace build_all keeps passing.
#
# Usage:
#   ./build.sh              # clean release build (default)
#   ./build.sh --debug      # clean debug build
#   ./build.sh --no-clean   # incremental build (re-uses build/ tree)
#   ./build.sh --help       # show this help
#
# Environment overrides (all optional):
#   BUILD_TYPE=Release|Debug    (default Release)
#   JOBS=N                      (default $(nproc) / sysctl -n hw.ncpu)
#   PREFIX=path                 (default <workspace>/.local)
#   VENV=path                   (default <workspace>/.venv)
#
# Conventions across the fleet repos:
#   - <workspace> is the parent of this repo.
#   - C/C++ repos install to $PREFIX (default ../.local).
#   - Python repos install (editable) into $VENV.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
WORKSPACE="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO="$(basename "$SCRIPT_DIR")"

CLEAN=1
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
PREFIX="${PREFIX:-$WORKSPACE/.local}"
VENV="${VENV:-$WORKSPACE/.venv}"

for arg in "$@"; do
  case "$arg" in
    --debug)    BUILD_TYPE=Debug ;;
    --release)  BUILD_TYPE=Release ;;
    --no-clean) CLEAN=0 ;;
    -h|--help)  sed -n '2,23p' "$0"; exit 0 ;;
    *) echo "build.sh: unknown arg: $arg (try --help)" >&2; exit 2 ;;
  esac
done

# *** Platform gate — Metal is macOS-only. ***
if [[ "$(uname -s)" != "Darwin" ]]; then
  echo ">>> [$REPO] build.sh: skipped (Metal is macOS-only; uname=$(uname -s))"
  exit 0
fi

T0=$(date +%s)
echo ">>> [$REPO] build.sh: type=$BUILD_TYPE jobs=$JOBS clean=$CLEAN prefix=$PREFIX"

# Native build (libnusd_renderer.dylib + Metal shaders).
[[ "$CLEAN" -eq 1 ]] && rm -rf build
cmake -B build \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_PREFIX_PATH="$PREFIX"
cmake --build build --parallel "$JOBS"

# Python bindings (editable install of python/nusd_renderer) if present.
if [[ -f python/pyproject.toml ]] && [[ -f "$VENV/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate" || true
  [[ -d "$VENV/bin" ]] && export PATH="$VENV/bin:$PATH"
  python -m pip install -e python --upgrade
fi

T=$(( $(date +%s) - T0 ))
echo ">>> [$REPO] build.sh: OK (${T}s) — dylib at build/, py module 'nusd_renderer'"

#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# build.sh — clean release build of this repo.
#
# Usage:
#   ./build.sh              # clean release build (default)
#   ./build.sh --debug      # clean debug build
#   ./build.sh --no-clean   # incremental build (re-uses build/ tree)
#   ./build.sh --help       # show this help
#
# Environment overrides (all optional):
#   BUILD_TYPE=Release|Debug    (default Release)
#   JOBS=N                      (default $(nproc))
#   PREFIX=path                 (default <workspace>/.local)
#   VENV=path                   (default <workspace>/.venv312 if present,
#                                else <workspace>/.venv)
#
# Conventions across the fleet repos:
#   - <workspace> is the parent of this repo.
#   - C/C++ repos install to $PREFIX (default ../.local).
#   - Python repos install (editable) into $VENV.
#
# *** This is scaffolding. *** Until src/ + a real CMakeLists.txt lands, the
# build only verifies cmake configuration. See ../UNIFIED_VIEWER_PLAN.md.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
WORKSPACE="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO="$(basename "$SCRIPT_DIR")"

CLEAN=1
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
PREFIX="${PREFIX:-$WORKSPACE/.local}"
if [[ -z "${VENV:-}" ]]; then
  if [[ -d "$WORKSPACE/.venv312" ]]; then
    VENV="$WORKSPACE/.venv312"
  else
    VENV="$WORKSPACE/.venv"
  fi
fi

for arg in "$@"; do
  case "$arg" in
    --debug)    BUILD_TYPE=Debug ;;
    --release)  BUILD_TYPE=Release ;;
    --no-clean) CLEAN=0 ;;
    -h|--help)  sed -n '2,21p' "$0"; exit 0 ;;
    *) echo "build.sh: unknown arg: $arg (try --help)" >&2; exit 2 ;;
  esac
done

T0=$(date +%s)
echo ">>> [$REPO] build.sh: type=$BUILD_TYPE jobs=$JOBS clean=$CLEAN prefix=$PREFIX"

if [[ -f "$VENV/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate" || true
fi
[[ -d "$VENV/bin" ]] && export PATH="$VENV/bin:$PATH"

if ! python -c "import PySide6" >/dev/null 2>&1; then
  echo ">>> [$REPO] installing PySide6 dependency"
  python -m pip install "PySide6>=6.6,<7" --upgrade
fi

if [[ -d "$WORKSPACE/nanousd-python" ]] && \
   ! python -c "import nanousd, nanousd.pxr_compat" >/dev/null 2>&1; then
  echo ">>> [$REPO] installing nanousd-python dependency"
  (cd "$WORKSPACE/nanousd-python" && VENV="$VENV" ./build.sh)
fi

[[ "$CLEAN" -eq 1 ]] && rm -rf build
cmake -B build \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_PREFIX_PATH="$PREFIX"
cmake --build build --parallel "$JOBS"

T=$(( $(date +%s) - T0 ))
echo ">>> [$REPO] build.sh: OK (${T}s) — Python launcher configured at build/nanousdview"

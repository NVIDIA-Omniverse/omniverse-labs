#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# build.sh — clean release build of this repo.
#
# Usage:
#   ./build.sh              # clean release build (default)
#   ./build.sh --debug      # clean debug build
#   ./build.sh --no-clean   # incremental build (re-uses build/ tree)
#   ./build.sh --mdl        # enable NVIDIA MDL SDK bridge
#   ./build.sh --help       # show this help
#
# Environment overrides (all optional):
#   BUILD_TYPE=Release|Debug    (default Release)
#   JOBS=N                      (default nproc/sysctl/4)
#   PREFIX=path                 (default <workspace>/.local)
#   VENV=path                   (default <workspace>/.venv)
#   NANOUSD_DIR=path            (auto-detected from ../nanousd by default)
#   NUSD_ENABLE_MDL_SDK=ON|OFF  (default OFF; generic .mdl fallback is always built)
#   MDL_SDK_ROOT=path           (optional NVIDIA MDL SDK install/build prefix)
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
NANOUSD_DIR="${NANOUSD_DIR:-}"
NUSD_ENABLE_MDL_SDK="${NUSD_ENABLE_MDL_SDK:-OFF}"
MDL_SDK_ROOT="${MDL_SDK_ROOT:-}"

for arg in "$@"; do
  case "$arg" in
    --debug)    BUILD_TYPE=Debug ;;
    --release)  BUILD_TYPE=Release ;;
    --no-clean) CLEAN=0 ;;
    --mdl)      NUSD_ENABLE_MDL_SDK=ON ;;
    --mdl-sdk-root=*) MDL_SDK_ROOT="${arg#*=}"; NUSD_ENABLE_MDL_SDK=ON ;;
    -h|--help)  sed -n '2,23p' "$SCRIPT_DIR/build.sh"; exit 0 ;;
    *) echo "build.sh: unknown arg: $arg (try --help)" >&2; exit 2 ;;
  esac
done

T0=$(date +%s)
echo ">>> [$REPO] build.sh: type=$BUILD_TYPE jobs=$JOBS clean=$CLEAN prefix=$PREFIX"

[[ "$CLEAN" -eq 1 ]] && rm -rf build
if [[ -z "$NANOUSD_DIR" ]] && [[ -f "$WORKSPACE/nanousd/CMakeLists.txt" ]]; then
    NANOUSD_DIR="$WORKSPACE/nanousd"
fi
CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    -DCMAKE_PREFIX_PATH="$PREFIX"
    -DNUSD_ENABLE_MDL_SDK="$NUSD_ENABLE_MDL_SDK"
)
[[ -n "$NANOUSD_DIR" ]] && CMAKE_ARGS+=(-DNANOUSD_DIR="$NANOUSD_DIR")
[[ -n "${MDL_SDK_ROOT:-}" ]] && CMAKE_ARGS+=(-DMDL_SDK_ROOT="$MDL_SDK_ROOT")
echo ">>> [$REPO] cmake: NANOUSD_DIR=${NANOUSD_DIR:-<auto/system>} MDL_SDK=${NUSD_ENABLE_MDL_SDK} MDL_SDK_ROOT=${MDL_SDK_ROOT:-<auto/system>}"
cmake -B build "${CMAKE_ARGS[@]}"
cmake --build build --parallel "$JOBS"

RENDERER_LIB="build/libnusd_renderer_opengl.so"
if [[ -f build/libnusd_renderer_opengl.dylib ]]; then
    RENDERER_LIB="build/libnusd_renderer_opengl.dylib"
fi

T=$(( $(date +%s) - T0 ))
echo ">>> [$REPO] build.sh: OK (${T}s) — OpenGL backend at $RENDERER_LIB"

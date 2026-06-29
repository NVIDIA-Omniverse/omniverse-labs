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
#   JOBS=N                      (default $(nproc))
#   PREFIX=path                 (default <workspace>/.local)
#   VENV=path                   (default <workspace>/.venv)
#   NANOUSD_DIR=path            (auto-detected from ../nanousd by default)
#   GLSLC=path                  (auto-detected from PATH/Vulkan SDK/local tools)
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
JOBS="${JOBS:-$(nproc)}"
PREFIX="${PREFIX:-$WORKSPACE/.local}"
VENV="${VENV:-$WORKSPACE/.venv}"
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

detect_nanousd_dir() {
  local build_lc
  build_lc="$(printf '%s' "$BUILD_TYPE" | tr '[:upper:]' '[:lower:]')"
  local candidates=(
    "$WORKSPACE/nanousd/_install/$build_lc"
    "$WORKSPACE/nanousd/_install/release"
    "$WORKSPACE/nanousd/_install/debug"
    "$WORKSPACE/nanousd"
  )
  local d
  for d in "${candidates[@]}"; do
    [[ -f "$d/include/nanousd/nanousdapi.h" ]] || continue
    if [[ -e "$d/lib/libnanousdapi.so" ]] || \
       [[ -e "$d/lib/libnanousdapi.a" ]] || \
       [[ -e "$d/_build/release/Release/libnanousdapi.so" ]] || \
       [[ -e "$d/build/Release/libnanousdapi.so" ]] || \
       [[ -e "$d/build/libnanousdapi.so" ]]; then
      printf '%s\n' "$d"
      return 0
    fi
  done
  return 1
}

detect_glslc() {
  if command -v glslc >/dev/null 2>&1; then
    command -v glslc
    return 0
  fi
  local candidates=(
    "${VULKAN_SDK:-}/bin/glslc"
    "$HOME/blender/lib/linux_x64/shaderc/bin/glslc"
    "$HOME/build_linux/deps_x64/Release/shaderc/bin/glslc"
  )
  local p
  for p in "${candidates[@]}"; do
    [[ -n "$p" ]] && [[ -x "$p" ]] && { printf '%s\n' "$p"; return 0; }
  done
  return 1
}

if [[ -z "${NANOUSD_DIR:-}" ]] && NANOUSD_DIR="$(detect_nanousd_dir)"; then
  export NANOUSD_DIR
fi

if [[ -z "${GLSLC:-}" ]] && GLSLC="$(detect_glslc)"; then
  export GLSLC
fi

CMAKE_ARGS=(
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
  -DCMAKE_PREFIX_PATH="$PREFIX"
  -DNUSD_ENABLE_MDL_SDK="$NUSD_ENABLE_MDL_SDK"
)
[[ -n "${NANOUSD_DIR:-}" ]] && CMAKE_ARGS+=(-DNANOUSD_DIR="$NANOUSD_DIR")
[[ -n "${GLSLC:-}" ]] && CMAKE_ARGS+=(-DGLSLC="$GLSLC")
[[ -n "${MDL_SDK_ROOT:-}" ]] && CMAKE_ARGS+=(-DMDL_SDK_ROOT="$MDL_SDK_ROOT")

echo ">>> [$REPO] cmake: NANOUSD_DIR=${NANOUSD_DIR:-<auto/system>} GLSLC=${GLSLC:-<auto/system>} MDL_SDK=${NUSD_ENABLE_MDL_SDK} MDL_SDK_ROOT=${MDL_SDK_ROOT:-<auto/system>}"

# C build (renders + libnusd_renderer.so).
[[ "$CLEAN" -eq 1 ]] && rm -rf build
cmake -B build "${CMAKE_ARGS[@]}"
cmake --build build --parallel "$JOBS"

# OVRTX facade plus private backend bindings.
if [[ -f python/pyproject.toml ]] && [[ -f "$VENV/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate" || true
  [[ -d "$VENV/bin" ]] && export PATH="$VENV/bin:$PATH"
  python -m pip install -e python --upgrade
fi

T=$(( $(date +%s) - T0 ))
echo ">>> [$REPO] build.sh: OK (${T}s) — binary at build/, Python API 'ovrtx.Renderer'"

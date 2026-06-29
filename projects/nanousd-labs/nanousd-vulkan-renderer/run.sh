#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# run.sh — run the canonical smoke test for this repo.
#
# Usage:
#   ./run.sh                # default smoke run (ctest)
#   ./run.sh --help         # show this help
#   ./run.sh -- <args>      # forward extra args to ctest
#
# Environment overrides (all optional):
#   VENV=path               (default <workspace>/.venv)
#   PREFIX=path             (default <workspace>/.local)
#   DISPLAY=:N              (default :1 — required for headless Vulkan on DGXC)
#   XAUTHORITY=path         (default /run/user/1000/gdm/Xauthority)
#
# Auto-builds via ./build.sh if the build artifact is missing.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
WORKSPACE="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO="$(basename "$SCRIPT_DIR")"

VENV="${VENV:-$WORKSPACE/.venv}"
PREFIX="${PREFIX:-$WORKSPACE/.local}"
export DISPLAY="${DISPLAY:-:1}"
export XAUTHORITY="${XAUTHORITY:-/run/user/1000/gdm/Xauthority}"

EXTRA=()
PASSTHROUGH=0
for arg in "$@"; do
  if [[ $PASSTHROUGH -eq 1 ]]; then EXTRA+=("$arg"); continue; fi
  case "$arg" in
    --) PASSTHROUGH=1 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) EXTRA+=("$arg") ;;
  esac
done

if [[ ! -d build ]]; then
  echo ">>> [$REPO] run.sh: no build/ — running ./build.sh first"
  ./build.sh
fi

echo ">>> [$REPO] run.sh: ctest (vulkan RT smoke + correctness)"
ctest --test-dir build --output-on-failure "${EXTRA[@]}"
echo ">>> [$REPO] run.sh: OK"

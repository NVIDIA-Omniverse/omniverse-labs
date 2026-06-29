#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# run.sh — open the shared OVRTX viewer on the OpenGL implementation.
#
# Usage:
#   ./run.sh                       # opens nanousdview on test_cube.usda
#   ./run.sh path/to/scene.usd     # opens nanousdview on the given scene
#   ./run.sh --screenshot out.ppm --width 640 --height 480
#   ./run.sh --help                # show this help
#   ./run.sh -- <args>             # forward extra args to nanousdview
#
# Environment overrides (all optional):
#   VENV=path               (default <workspace>/.venv312 if present, else <workspace>/.venv)
#   PREFIX=path             (default <workspace>/.local)
#   DISPLAY=:N              (default :1)
#   XAUTHORITY=path         (default /run/user/1000/gdm/Xauthority)
#
# Auto-builds via ./build.sh if the OpenGL renderer library is missing.

set -euo pipefail

CALLER_CWD="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
cd "$SCRIPT_DIR"
WORKSPACE="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO="$(basename "$SCRIPT_DIR")"

if [[ -z "${VENV:-}" ]]; then
  if [[ -d "$WORKSPACE/.venv312" ]]; then
    VENV="$WORKSPACE/.venv312"
  else
    VENV="$WORKSPACE/.venv"
  fi
fi
PREFIX="${PREFIX:-$WORKSPACE/.local}"
export DISPLAY="${DISPLAY:-:1}"
export XAUTHORITY="${XAUTHORITY:-/run/user/1000/gdm/Xauthority}"

SCENE=""
EXTRA=()
PASSTHROUGH=0
i=0
args=("$@")
resolve_path_arg() {
  local path_arg="$1"
  if [[ "$path_arg" == /* ]]; then
    printf '%s\n' "$path_arg"
  else
    printf '%s\n' "$CALLER_CWD/$path_arg"
  fi
  return 0
}
append_extra_value() {
  local opt="$1"
  local path_value="${2:-0}"
  local value=""

  if [[ "$opt" == *=* ]]; then
    value="${opt#*=}"
    if [[ "$path_value" -eq 1 ]]; then
      value="$(resolve_path_arg "$value")"
    fi
    EXTRA+=("${opt%%=*}=$value")
    return
  fi

  i=$((i+1))
  if [[ $i -ge ${#args[@]} ]]; then
    echo ">>> [$REPO] run.sh: option requires a value: $opt" >&2
    exit 2
  fi
  value="${args[$i]}"
  if [[ "$path_value" -eq 1 ]]; then
    value="$(resolve_path_arg "$value")"
  fi
  EXTRA+=("$opt" "$value")
}
while [[ $i -lt $# ]]; do
  arg="${args[$i]}"
  if [[ $PASSTHROUGH -eq 1 ]]; then EXTRA+=("$arg"); i=$((i+1)); continue; fi
  case "$arg" in
    --) PASSTHROUGH=1 ;;
    --backend)
      i=$((i+1))
      if [[ $i -ge ${#args[@]} ]]; then
        echo ">>> [$REPO] run.sh: option requires a value: --backend" >&2
        exit 2
      fi
      if [[ "${args[$i]}" != "opengl" ]]; then
        echo ">>> [$REPO] run.sh: this wrapper only supports --backend opengl" >&2
        exit 2
      fi
      ;;
    --backend=opengl) ;;
    --backend=*)
      echo ">>> [$REPO] run.sh: this wrapper only supports --backend opengl" >&2
      exit 2
      ;;
    -h|--help) sed -n '2,18p' "$SCRIPT_PATH"; exit 0 ;;
    --screenshot|--capture-window) append_extra_value "$arg" 1 ;;
    --screenshot=*|--capture-window=*) append_extra_value "$arg" 1 ;;
    --width|--height|--camera|--frame|--cf|--ff|--lf|--render-mode|--capture-delay)
      append_extra_value "$arg"
      ;;
    --width=*|--height=*|--camera=*|--frame=*|--cf=*|--ff=*|--lf=*|--render-mode=*|--capture-delay=*)
      append_extra_value "$arg"
      ;;
    --qa-interactions) EXTRA+=("$arg") ;;
    --*) EXTRA+=("$arg") ;;
    *)
      if [[ -z "$SCENE" ]]; then SCENE="$(resolve_path_arg "$arg")"; else EXTRA+=("$arg"); fi
      ;;
  esac
  i=$((i+1))
done

if [[ -n "$SCENE" ]] && [[ ! -f "$SCENE" ]]; then
  echo ">>> [$REPO] run.sh: scene not found: $SCENE" >&2
  exit 2
fi
SCENE="${SCENE:-$SCRIPT_DIR/test_cube.usda}"
if [[ ! -f "$SCENE" ]]; then
  echo ">>> [$REPO] run.sh: default scene not found: $SCENE" >&2
  exit 2
fi

if [[ ! -f build/libnusd_renderer_opengl.so ]] && [[ ! -f build/libnusd_renderer_opengl.dylib ]]; then
  echo ">>> [$REPO] run.sh: no build/libnusd_renderer_opengl.{so,dylib} — running ./build.sh first"
  ./build.sh
fi

echo ">>> [$REPO] run.sh: nanousdview --backend opengl $SCENE ${EXTRA[*]:-}"
cd "$CALLER_CWD"
exec "$WORKSPACE/nanousdview/run.sh" --backend opengl "$SCENE" "${EXTRA[@]}"

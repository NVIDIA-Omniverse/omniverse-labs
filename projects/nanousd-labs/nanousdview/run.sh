#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# run.sh — open the viewer on a USD scene.
#
# Backend matrix (default in []):
#   Linux: [vulkan], opengl, ovrtx
#   macOS: [metal],  opengl
#
# Usage:
#   ./run.sh                       # platform default backend, default scene
#   ./run.sh path/to/scene.usd     # platform default backend, given scene
#   ./run.sh --backend opengl      # OpenGL (Linux + macOS)
#   ./run.sh --backend vulkan      # Vulkan (Linux only)
#   ./run.sh --backend metal       # Metal  (macOS only)
#   ./run.sh --backend ovrtx       # NVIDIA OmniRTX (Linux only; requires `ovrtx` pip wheel)
#   ./run.sh --backend opengl --screenshot out.ppm --width 640 --height 480
#   ./run.sh --backend opengl --capture-window out.png --qa-camera-controls --qa-report report.json
#   ./run.sh --help                # show this help
#   ./run.sh -- <args>             # forward extra args to nanousdview
#
# Environment overrides (all optional):
#   VENV=path               (default <workspace>/.venv312 if present, else <workspace>/.venv)
#   PYTHON=path             (default: python, then python3 from PATH)
#   PREFIX=path             (default <workspace>/.local)
#   DISPLAY=:N              (Linux only — default :1 for headless Vulkan on DGXC)
#   XAUTHORITY=path         (Linux only)
#   NANOUSD_VIEW_BACKEND=name
#                           (default = platform default; --backend overrides)
#   NANOUSD_LIB=path        (override libnanousdapi location)
#   AIUSD_LIB_PATH=path     (legacy alias for NANOUSD_LIB)

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

# Platform default backend.
case "$(uname -s)" in
  Darwin) DEFAULT_BACKEND=metal ;;
  Linux)
    DEFAULT_BACKEND=vulkan
    export DISPLAY="${DISPLAY:-:1}"
    export XAUTHORITY="${XAUTHORITY:-/run/user/1000/gdm/Xauthority}"
    ;;
  *) DEFAULT_BACKEND=vulkan ;;
esac

BACKEND="${NANOUSD_VIEW_BACKEND:-$DEFAULT_BACKEND}"
SCENE=""
EXTRA=()
PASSTHROUGH=0
i=0
args=("$@")
resolve_scene_arg() {
  local scene_arg="$1"
  if [[ "$scene_arg" == /* ]]; then
    printf '%s\n' "$scene_arg"
  else
    printf '%s\n' "$CALLER_CWD/$scene_arg"
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
      value="$(resolve_scene_arg "$value")"
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
    value="$(resolve_scene_arg "$value")"
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
      BACKEND="${args[$i]}"
      ;;
    --backend=*) BACKEND="${arg#*=}" ;;
    -h|--help) sed -n '2,22p' "$SCRIPT_PATH"; exit 0 ;;
    --screenshot|--capture-window|--qa-report) append_extra_value "$arg" 1 ;;
    --screenshot=*|--capture-window=*|--qa-report=*) append_extra_value "$arg" 1 ;;
    --width|--height|--camera|--frame|--cf|--ff|--lf|--render-mode|--capture-delay|--aov|--envmap|--envmap-intensity)
      append_extra_value "$arg"
      ;;
    --width=*|--height=*|--camera=*|--frame=*|--cf=*|--ff=*|--lf=*|--render-mode=*|--capture-delay=*|--aov=*|--envmap=*|--envmap-intensity=*)
      append_extra_value "$arg"
      ;;
    --qa-interactions|--qa-camera-controls) EXTRA+=("$arg") ;;
    --*) EXTRA+=("$arg") ;;
    *)
      if [[ -z "$SCENE" ]]; then SCENE="$(resolve_scene_arg "$arg")"; else EXTRA+=("$arg"); fi
      ;;
  esac
  i=$((i+1))
done

# Locate a default scene if none given. nanousdview itself ships none;
# probe sibling repos for a known test_cube.usda.
if [[ -n "$SCENE" ]] && [[ ! -f "$SCENE" ]]; then
  echo ">>> [$REPO] run.sh: scene not found: $SCENE" >&2
  exit 2
fi
if [[ -z "$SCENE" ]]; then
  for cand in \
    "$WORKSPACE/nanousd-opengl-renderer/test_cube.usda" \
  ; do
    if [[ -f "$cand" ]]; then SCENE="$cand"; break; fi
  done
fi
if [[ -z "$SCENE" ]] || [[ ! -f "$SCENE" ]]; then
  echo ">>> [$REPO] run.sh: no scene found. Pass a USD path:" >&2
  echo "    ./run.sh path/to/scene.usda" >&2
  exit 2
fi

# Activate the workspace .venv. Some local venv activate scripts can carry
# stale prompt/path metadata, so prepend $VENV/bin ourselves regardless of
# activate's success.
if [[ -f "$VENV/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate" || true
fi
export PATH="$VENV/bin:$PATH"

PYTHON_BIN="${PYTHON:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  fi
fi
if [[ -z "$PYTHON_BIN" ]] || ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo ">>> [$REPO] run.sh: python not on PATH (looked in $VENV/bin and PATH)" >&2
  exit 2
fi

# Make the nanousdview package importable. The selected OVRTX implementation
# is configured inside __main__.py based on --backend.
export PYTHONPATH="$SCRIPT_DIR/python${PYTHONPATH:+:$PYTHONPATH}"

# libnanousdapi auto-discovery: __main__.py probes $PREFIX/lib + nanousd/build.
# Honour an explicit override but otherwise let the entry point find it.
if [[ -n "${NANOUSD_LIB:-}" ]]; then
  export NANOUSD_LIB
elif [[ -n "${AIUSD_LIB_PATH:-}" ]]; then
  export AIUSD_LIB_PATH
fi

# --- ovrtx-specific runtime setup ---------------------------------------
# ovrtx 0.3.0's bundled libgpu.foundation.plugin.so calls g_string_copy
# (added in GLib 2.86); the system glibc on dgxc/Ubuntu still ships GLib
# 2.80 without that symbol, so loading the plugin against the system libglib
# fails with `undefined symbol g_string_copy`. ovrtx ships a newer libglib
# in its plugin dir — preload it (and libgobject which has the same skew)
# so the dynamic linker resolves the new symbols before falling back to
# /lib/x86_64-linux-gnu. Also tell ovrtx to skip its own pxr-presence
# guard since nanousdview registers a `pxr_compat` shim that the adapter
# already steps around at import time.
if [[ "$BACKEND" = "ovrtx" ]]; then
  OVRTX_PLUGIN_DIR="$VENV/lib/python3.12/site-packages/ovrtx/bin/plugins/gpu.foundation"
  if [[ -f "$OVRTX_PLUGIN_DIR/libglib-2.0.so.0" ]]; then
    export LD_PRELOAD="$OVRTX_PLUGIN_DIR/libglib-2.0.so.0:$OVRTX_PLUGIN_DIR/libgobject-2.0.so.0${LD_PRELOAD:+:$LD_PRELOAD}"
  fi
  export OVRTX_SKIP_USD_CHECK="${OVRTX_SKIP_USD_CHECK:-1}"
fi

echo ">>> [$REPO] run.sh: $PYTHON_BIN -m nanousdview --backend $BACKEND $SCENE ${EXTRA[*]:-}"
if [[ ${#EXTRA[@]} -gt 0 ]]; then
  exec "$PYTHON_BIN" -m nanousdview --backend "$BACKEND" "$SCENE" "${EXTRA[@]}"
else
  exec "$PYTHON_BIN" -m nanousdview --backend "$BACKEND" "$SCENE"
fi

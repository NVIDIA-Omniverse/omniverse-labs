#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# setup.sh - install the Python runtime for ov-fmi and build the demo FMUs.
#
# Defaults:
#   - create/use demoapp/.venv with access to base-environment packages
#   - require user-installed ovrtx, ovstage, and ovphysx packages
#   - install ov-fmi and its Python dependencies
#   - create/use demoapp/.usd_venv for usd-core parsing
#   - build generated FMU and SSP archives into demoapp/usd/
#
# Optional environment variables:
#   SKIP_FMU_BUILD=1
#   INSTALL_CUDA_PYTHON=1
#
# Optional fmi_usd_helper build:
#   OVPHYSX_SDK_DIR=/path/to/ovphysx-sdk
#   OPENUSD_INCLUDE_DIR=/path/to/openusd_src

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OVFMI_DIR="$REPO_ROOT"
APP_VENV="$SCRIPT_DIR/.venv"
APP_PYTHON="$APP_VENV/bin/python"
USD_VENV="$SCRIPT_DIR/.usd_venv"
USD_PYTHON="$USD_VENV/bin/python"

echo "Creating app venv at $APP_VENV ..."
if [ ! -x "$APP_PYTHON" ]; then
    python3 -m venv --system-site-packages "$APP_VENV"
elif ! grep -Eq '^include-system-site-packages[[:space:]]*=[[:space:]]*true[[:space:]]*$' "$APP_VENV/pyvenv.cfg"; then
    echo "ERROR: demoapp/.venv is isolated and cannot see the user-installed ov packages." >&2
    echo "Delete demoapp/.venv and rerun setup." >&2
    exit 1
fi

echo "Installing Python packages..."
"$APP_PYTHON" -m pip install --upgrade pip
if ! "$APP_PYTHON" - <<'EOF'
from importlib.metadata import PackageNotFoundError, version

missing = []
for name in ("ovrtx", "ovstage", "ovphysx"):
    try:
        print(f"Using {name}=={version(name)}")
    except PackageNotFoundError:
        missing.append(name)

if missing:
    raise SystemExit("Missing required package(s): " + ", ".join(missing))
EOF
then
    echo "Install the packages in the base Python environment first:" >&2
    echo "  python3 -m pip install ovrtx ovstage ovphysx" >&2
    exit 1
fi

OVRTX_BIN="$("$APP_PYTHON" - <<'EOF'
from pathlib import Path
import ovrtx

print(Path(ovrtx.__file__).resolve().parent / "bin")
EOF
)"
if [ ! -f "$OVRTX_BIN/libovrtx-dynamic.so" ]; then
    echo ""
    echo "ERROR: Installed ovrtx package does not contain libovrtx-dynamic.so at $OVRTX_BIN."
    echo ""
    exit 1
fi
echo "Using ovrtx native library from: $OVRTX_BIN"

"$APP_PYTHON" -m pip install -e "$OVFMI_DIR"
"$APP_PYTHON" -m pip install -e "$SCRIPT_DIR"

if [ "${INSTALL_CUDA_PYTHON:-0}" = "1" ]; then
    "$APP_PYTHON" -m pip install cuda-python
else
    echo "Skipping cuda-python. Set INSTALL_CUDA_PYTHON=1 to enable CUDA/OpenGL zero-copy display support."
fi

# --- optionally build fmi_usd_helper ----------------------------------------
HELPER_DIR="$SCRIPT_DIR/fmi_usd_helper"
HELPER_BUILD="$HELPER_DIR/build"

if [ -n "${OVPHYSX_SDK_DIR:-}" ] && [ -n "${OPENUSD_INCLUDE_DIR:-}" ]; then
    echo "Building fmi_usd_helper..."
    cmake -B "$HELPER_BUILD" \
          -S "$HELPER_DIR" \
          -DOVPHYSX_SDK_DIR="$OVPHYSX_SDK_DIR" \
          -DOPENUSD_INCLUDE_DIR="$OPENUSD_INCLUDE_DIR" \
          -DCMAKE_BUILD_TYPE=Release
    cmake --build "$HELPER_BUILD" --config Release
    echo "fmi_usd_helper built at: $HELPER_BUILD/fmi_usd_helper"
    echo "  -> usd-core subprocess fallback will not be used."
else
    echo "Skipping fmi_usd_helper; usd-core subprocess fallback will be used for USD parsing."
fi

# --- isolated usd-core venv (ovrtx refuses to load when pxr is installed) ----
echo "Creating isolated usd-core venv at $USD_VENV ..."
if [ ! -x "$USD_PYTHON" ]; then
    python3 -m venv "$USD_VENV"
fi
echo "Installing usd-core into isolated venv..."
"$USD_PYTHON" -m pip install --upgrade pip
"$USD_PYTHON" -m pip install --quiet usd-core

# --- emit a small env file --------------------------------------------------
ENV_FILE="$SCRIPT_DIR/.env"
cat > "$ENV_FILE" <<ENVEOF
# Source this file before running ov-fmi with a different Python executable:
#   source demoapp/.env
export OVRTX_LIBRARY_PATH_HINT="$OVRTX_BIN"
export OVSTAGE_LIBRARY_PATH_HINT="$OVRTX_BIN"
export PYTHONPATH="$SCRIPT_DIR:\${PYTHONPATH:-}"
export USD_PYTHON="$USD_PYTHON"
ENVEOF

# --- build generated FMU and SSP archives ----------------------------------
if [ "${SKIP_FMU_BUILD:-0}" = "1" ]; then
    echo "Skipping FMU/SSP build because SKIP_FMU_BUILD=1."
else
    echo "Building demo FMUs and SSPs..."
    if [ -n "${CXX:-}" ] || command -v g++ >/dev/null 2>&1; then
        "$APP_PYTHON" "$SCRIPT_DIR/build_fmu.py"
    elif command -v clang++ >/dev/null 2>&1; then
        CXX=clang++ "$APP_PYTHON" "$SCRIPT_DIR/build_fmu.py"
    else
        echo ""
        echo "ERROR: Cannot build FMUs because neither g++ nor clang++ was found."
        echo "       Install a C++17 compiler, or rerun with SKIP_FMU_BUILD=1."
        echo ""
        exit 1
    fi
fi

echo ""
echo "Setup complete."
echo "Wrote $ENV_FILE"
echo ""
echo "Run:"
echo "  $APP_PYTHON $SCRIPT_DIR/main.py $SCRIPT_DIR/usd/fmi_parser_test.usda"

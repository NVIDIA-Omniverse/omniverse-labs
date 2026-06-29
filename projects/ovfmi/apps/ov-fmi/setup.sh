#!/usr/bin/env bash
# setup.sh - install the Python runtime for ov-fmi and build the demo FMUs.
#
# Defaults:
#   - create/use apps/ov-fmi/.venv
#   - install ovrtx 0.3 from NVIDIA's Python package index
#   - install ov-fmi and its Python dependencies
#   - install ovphysx 0.4.9 for physics demos
#   - create/use apps/ov-fmi/.usd_venv for usd-core parsing
#   - build generated FMU and SSP archives into usd/ov-fmi/
#
# Optional environment variables:
#   OVRTX_DIR=/path/to/extracted/ovrtx-package
#   SKIP_OVPHYSX=1
#   SKIP_FMU_BUILD=1
#   INSTALL_CUDA_PYTHON=1
#
# Optional fmi_usd_helper build:
#   OVPHYSX_SDK_DIR=/path/to/ovphysx-sdk
#   OPENUSD_INCLUDE_DIR=/path/to/openusd_src

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
APP_VENV="$SCRIPT_DIR/.venv"
APP_PYTHON="$APP_VENV/bin/python"
USD_VENV="$SCRIPT_DIR/.usd_venv"
USD_PYTHON="$USD_VENV/bin/python"
OVRTX_PYTHON_SRC="$REPO_ROOT/third-party/ovrtx/python"
OVRTX_WHEEL_VERSION="${OVRTX_WHEEL_VERSION:-0.3.0.312915}"
OVPHYSX_VERSION="${OVPHYSX_VERSION:-0.4.9}"

echo "Creating app venv at $APP_VENV ..."
if [ ! -x "$APP_PYTHON" ]; then
    python3 -m venv "$APP_VENV"
fi

echo "Installing Python packages..."
"$APP_PYTHON" -m pip install --upgrade pip

if [ -n "${OVRTX_DIR:-}" ]; then
    OVRTX_BIN="$OVRTX_DIR/bin"
    if [ ! -f "$OVRTX_BIN/libovrtx-dynamic.so" ]; then
        echo ""
        echo "ERROR: OVRTX_DIR must point to an extracted ovrtx package root containing bin/libovrtx-dynamic.so."
        echo ""
        exit 1
    fi
    "$APP_PYTHON" -m pip install -e "$OVRTX_PYTHON_SRC"
else
    "$APP_PYTHON" -m pip install "ovrtx==$OVRTX_WHEEL_VERSION" --extra-index-url https://pypi.nvidia.com
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
fi

echo "Using ovrtx native library from: $OVRTX_BIN"

"$APP_PYTHON" -m pip install -e "$SCRIPT_DIR"

if [ "${SKIP_OVPHYSX:-0}" = "1" ]; then
    echo "Skipping ovphysx because SKIP_OVPHYSX=1."
else
    "$APP_PYTHON" -m pip install "ovphysx==$OVPHYSX_VERSION" --extra-index-url https://pypi.nvidia.com
fi

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
#   source apps/ov-fmi/.env
export OVRTX_LIBRARY_PATH_HINT="$OVRTX_BIN"
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
echo "  $APP_PYTHON $SCRIPT_DIR/main.py usd/ov-fmi/fmi_parser_test.usda"

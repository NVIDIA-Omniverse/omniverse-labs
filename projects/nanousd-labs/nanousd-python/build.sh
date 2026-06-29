#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# build.sh — editable install for nanousd-python.
#
# Usage:
#   ./build.sh              # install into the workspace venv
#   ./build.sh --help       # show this help
#
# Environment overrides:
#   VENV=path               (default <workspace>/.venv312 if present, else <workspace>/.venv)
#   NANOUSD_DIR=path        (default <workspace>/nanousd when present)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
WORKSPACE="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ -z "${VENV:-}" ]]; then
  if [[ -d "$WORKSPACE/.venv312" ]]; then
    VENV="$WORKSPACE/.venv312"
  else
    VENV="$WORKSPACE/.venv"
  fi
fi

for arg in "$@"; do
  case "$arg" in
    -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
    *) echo "build.sh: unknown arg: $arg (try --help)" >&2; exit 2 ;;
  esac
done

if [[ -f "$VENV/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate" || true
fi
[[ -d "$VENV/bin" ]] && export PATH="$VENV/bin:$PATH"

if [[ -z "${NANOUSD_DIR:-}" ]] && [[ -d "$WORKSPACE/nanousd" ]]; then
  export NANOUSD_DIR="$WORKSPACE/nanousd"
fi

python -m pip install -e . --upgrade
python - <<'PY'
import nanousd
import nanousd.pxr_compat
from pxr import Sdf, Usd

assert nanousd.Stage is not None
assert Sdf.Path("/") is not None
assert Usd.Stage is not None
PY
echo ">>> [nanousd-python] build.sh: OK - package imports verified"

# OpenUSD conformance regression gate. Runs only when an oracle interpreter with
# usd-core is available (set NANOUSD_CONFORMANCE_USDCORE_PYTHON); otherwise the
# gate is skipped with a hint. Fails the build on NEW divergences vs the frozen
# tests/conformance_baseline.json (regenerate with NANOUSD_CONFORMANCE_UPDATE_BASELINE=1).
if [[ -n "${NANOUSD_CONFORMANCE_USDCORE_PYTHON:-}" ]] && python -c "import pytest" >/dev/null 2>&1; then
    echo ">>> [nanousd-python] running OpenUSD conformance gate"
    if python -m pytest "$SCRIPT_DIR/tests/test_usdphysics_conformance.py" \
                        "$SCRIPT_DIR/tests/test_stage_conformance.py" \
                        "$SCRIPT_DIR/tests/test_openusd_schema_conformance.py" \
                        "$SCRIPT_DIR/tests/test_matrix_rotation_conformance.py" \
                        "$SCRIPT_DIR/tests/test_instancing_conformance.py" \
                        -q -p no:cacheprovider; then
        echo ">>> [nanousd-python] conformance gate: OK"
    else
        echo ">>> [nanousd-python] conformance gate: FAILED (new divergences vs baseline)" >&2
        exit 1
    fi
else
    echo ">>> [nanousd-python] conformance gate skipped (set NANOUSD_CONFORMANCE_USDCORE_PYTHON to a usd-core python + install pytest to enable)"
fi

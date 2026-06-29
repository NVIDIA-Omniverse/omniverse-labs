#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# compare_smoke.sh — verify compare_viewers.py exit codes.
#
# 0  if both reference and test are equal       → PASS or OK
# 1  if RMSE is large enough for WARN/FAIL
# 2  if invocation is bad / dimensions mismatch in --strict mode
set -u

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="${PYTHON:-python3}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Build small synthetic images we can fully control.
"$PY" - <<EOF
try:
    import numpy as np
    from PIL import Image
except ModuleNotFoundError as exc:
    print(f"SKIP: Python dependency missing: {exc.name}")
    raise SystemExit(77)
red   = np.zeros((64, 64, 3), dtype=np.uint8); red[:,:,0] = 255
red2  = red.copy()
blue  = np.zeros((64, 64, 3), dtype=np.uint8); blue[:,:,2] = 255
small = np.zeros((32, 32, 3), dtype=np.uint8)
Image.fromarray(red).save("$TMP/red.ppm")
Image.fromarray(red2).save("$TMP/red2.ppm")
Image.fromarray(blue).save("$TMP/blue.ppm")
Image.fromarray(small).save("$TMP/small.ppm")
EOF
rc=$?
if [[ "$rc" -eq 77 ]]; then
    exit 77
fi
if [[ "$rc" -ne 0 ]]; then
    exit "$rc"
fi

fails=0
expect() {
    local label="$1" want="$2"; shift 2
    "$@" >/dev/null 2>&1
    local got=$?
    if [[ "$got" -eq "$want" ]]; then
        echo "ok:   $label (exit $got)"
    else
        echo "FAIL: $label expected exit $want got $got"
        fails=$((fails+1))
    fi
    return 0
}

# Identical → PASS → exit 0
expect "identical-images-exit-0"  0  "$PY" "$REPO/compare_viewers.py" "$TMP/red.ppm" "$TMP/red2.ppm"
# Different colors at full saturation → FAIL → exit 1
expect "fail-images-exit-1"       1  "$PY" "$REPO/compare_viewers.py" "$TMP/red.ppm" "$TMP/blue.ppm"
# Bad invocation → exit 2
expect "no-args-exit-2"           2  "$PY" "$REPO/compare_viewers.py"
# Missing input → exit 2
expect "missing-file-exit-2"      2  "$PY" "$REPO/compare_viewers.py" "$TMP/red.ppm" "$TMP/_nope_.ppm"
# Lenient mismatch → resize, compare, FAIL because content differs → exit 1
expect "lenient-dim-mismatch-1"   1  "$PY" "$REPO/compare_viewers.py" "$TMP/red.ppm" "$TMP/small.ppm"
# Strict mismatch → exit 2 regardless of content
expect "strict-dim-mismatch-2"    2  "$PY" "$REPO/compare_viewers.py" --strict "$TMP/red.ppm" "$TMP/small.ppm"

if [[ "$fails" -eq 0 ]]; then
    echo "PASS: 0 failures"
    exit 0
else
    echo "FAIL: $fails failures"
    exit 1
fi

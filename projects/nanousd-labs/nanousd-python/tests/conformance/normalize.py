# SPDX-FileCopyrightText: Copyright (c) 2026 nanousd
# SPDX-License-Identifier: Apache-2.0
"""Shared value normalization for the OpenUSD ⇄ nanousd differential dumps.

Both backends represent the same USD value differently (int-enum vs token,
``Gf.Vec3f`` vs tuple, ``Vt.Array`` vs list, quaternion component order). This
coerces any USD value to a stable JSON-comparable form so the downstream diff
flags *semantic* divergences rather than encoding noise.
"""
from __future__ import annotations

import numbers

# Large geometry arrays (mesh points, faceVertexIndices, ...) are summarized
# rather than expanded element-by-element: full per-element Python recursion over
# thousands of points is pathologically slow and dominates a full-stage walk.
# Length + sampled endpoints still flag truncation / reordering / scaling diffs.
_MAX_ARRAY = 32

# axis token <-> int-enum is a pure representation difference.
AXIS_TOKENS = {0: "X", 1: "Y", 2: "Z", "X": "X", "Y": "Y", "Z": "Z"}


def _array_checksum(items: list):
    """Cheap content fingerprint so mid-array divergences aren't hidden by the cap.

    sum + abs-sum together catch interior value changes and sign flips that
    head/tail sampling alone would miss. Best-effort: 'n/a' for non-numeric arrays.
    """
    try:
        import numpy as _np

        a = _np.asarray(items, dtype=_np.float64)
        return [round(float(_np.nansum(a)), 3), round(float(_np.nansum(_np.abs(a))), 3)]
    except Exception:
        return "n/a"


def _summarize_seq(items: list, depth: int):
    if len(items) > _MAX_ARRAY:
        head = [normalize(x, depth + 1) for x in items[:8]]
        tail = [normalize(x, depth + 1) for x in items[-8:]]
        # head/tail + checksum: catches truncation, reordering at the ends, AND
        # interior value divergence (via the sum fingerprint).
        return ["array", len(items), head, tail, _array_checksum(items)]
    return [normalize(x, depth + 1) for x in items]


def normalize(v, depth: int = 0):
    """Coerce an arbitrary USD value to a JSON-comparable, backend-neutral form."""
    if depth > 8:
        return "<deep>"
    if v is None or isinstance(v, (bool, str)):
        return v
    # numbers.* catches Python AND numpy scalars (the shim returns numpy floats);
    # without this they fall into the struct fallback and explode over dir().
    if isinstance(v, numbers.Integral):
        return int(v)
    if isinstance(v, numbers.Real):
        f = float(v)
        if f > 1e37:
            return "+INF"
        if f < -1e37:
            return "-INF"
        return round(f, 5)
    tn = type(v).__name__
    if tn in ("SdfPath", "Path", "_Path") or hasattr(v, "pathString"):
        return str(v)
    # Quaternion (Gf.Quat*): canonicalize to ["quat", w, x, y, z].
    if hasattr(v, "GetReal") and hasattr(v, "GetImaginary"):
        return ["quat", round(float(v.GetReal()), 5)] + [round(float(x), 5) for x in v.GetImaginary()]
    if isinstance(v, (list, tuple)):
        return _summarize_seq(list(v), depth)
    if isinstance(v, dict):
        return {str(k): normalize(val, depth + 1) for k, val in v.items()}
    # Iterable numerics: Gf.Vec*, Gf.Matrix*, Vt.Array.
    try:
        items = list(v)
    except TypeError:
        items = None
    if items is not None:
        return _summarize_seq(items, depth)
    # Only USD-ish objects reach the struct fallback; never recurse over the ~160
    # dir() attributes of a numpy/builtin scalar (that explodes to the depth cap).
    if type(v).__module__ in ("numpy", "builtins"):
        return str(v)
    # Nested struct (JointDrive, JointLimit, ...): extract public data attributes.
    struct = {}
    for n in dir(v):
        if n.startswith("_"):
            continue
        try:
            a = getattr(v, n)
        except Exception:
            continue
        if callable(a):
            continue
        struct[n] = normalize(a, depth + 1)
    return struct if struct else str(v)


def canon_field(name: str, value):
    """Canonicalize representation-only differences for a named field."""
    if name in ("axis", "Axis") and isinstance(value, (int, str)) and not isinstance(value, bool):
        return AXIS_TOKENS.get(value, value)
    return value

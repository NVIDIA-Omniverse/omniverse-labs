# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure-Python curve-width derivation and USD ``BasisCurves`` validation.

Shared by the live add-on runtime (``scene_generation``) and the offline
fixture-prep exporter (``tests/fixtures/blender_export/curves``) so both paths
derive hair/curve widths and validate curve topology the same way. This module
must not import ``bpy`` or ``pxr`` at load time — it operates on plain numbers
and arrays so it runs in headless Python, the add-on runtime, and Blender's own
interpreter alike. Blender/USD callers extract attributes and pass numeric
values in.

Three concerns:

1. **Particle-hair widths** — reconstruct root/tip endpoint widths from Blender
   hair-shape settings and taper each strand root-to-tip.
2. **Physical curve widths (issue #83)** — derive a physically meaningful USD
   width from evaluated bevel geometry and scene units, generalizing the
   closed-#57 Classroom relation with no fixture-specific constants.
3. **BasisCurves validity** — the array-length and per-curve rules a renderer
   needs so cubic strands are not dropped or rendered as blobs.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# ``vstep`` per cubic basis (OpenUSD UsdGeomBasisCurves): bezier advances the
# control-point window by 3 per segment, bspline/catmullRom by 1.
CUBIC_BASIS_VSTEP: dict[str, int] = {"bezier": 3, "bspline": 1, "catmullRom": 1}

# The default target this spec normalizes to (see spec Goals and Scope).
DEFAULT_TYPE = "cubic"
DEFAULT_BASIS = "catmullRom"
DEFAULT_WRAP = "nonperiodic"

# Defensive width floors (USD diameters, scene units). These keep strands
# visible/finite; they are not Blender semantics. Mirrors the values the
# fixture-prep exporter learned so both paths agree.
_MIN_ROOT_WIDTH = 0.001
_MIN_TIP_FRACTION_CLOSED = 0.05
_MIN_TIP_FRACTION_BLUNT = 0.20


# --------------------------------------------------------------------------
# Particle-hair widths
# --------------------------------------------------------------------------
def particle_width_range(
    radius_scale: float,
    root_radius: float,
    tip_radius: float,
    *,
    use_close_tip: bool = False,
) -> tuple[float, float]:
    """Return ``(root_width, tip_width)`` USD diameters from hair settings.

    Blender particle hair has no per-key width; it stores a global
    ``radius_scale`` times separate root/tip radius multipliers. We reconstruct
    the two endpoint widths and let :func:`fill_hair_widths` taper between them.
    The clamps are defensive floors (a literal 0 width would drop the strand or
    produce NaNs), not Blender semantics; ``use_close_tip`` off means the artist
    wants blunt tips, so the tip floor rises.
    """

    radius_scale = float(radius_scale or 0.005)
    root_radius = float(root_radius or 1.0)
    tip_radius = float(tip_radius or 0.2)

    root_width = max(radius_scale * root_radius, _MIN_ROOT_WIDTH)
    tip_width = max(radius_scale * tip_radius, root_width * _MIN_TIP_FRACTION_CLOSED)
    if not use_close_tip:
        tip_width = max(tip_width, root_width * _MIN_TIP_FRACTION_BLUNT)
    return root_width, tip_width


def hair_sample_count(
    render_step: int,
    hair_step: int = 1,
    *,
    cap: int = 64,
    minimum: int = 3,
) -> int:
    """Points-per-strand for exported hair, derived from ``render_step``.

    Blender evaluates a hair strand's curve at ``2**render_step`` subdivisions;
    the exported strand keeps that native resolution (``2**render_step + 1``
    points) rather than a fixed uniform count, bounded by ``cap`` so child
    strands cannot explode the USD, and floored at ``minimum``. ``render_step``
    is the render-time subdivision; ``hair_step`` is the viewport fallback. The
    exponent is clamped to ``[1, 20]`` to tolerate malformed files.
    """

    step = int(render_step or 0)
    if step <= 0:
        step = int(hair_step or 1)
    step = min(max(step, 1), 20)
    return max(int(minimum), min((1 << step) + 1, int(cap)))


def fill_hair_widths(
    vertex_counts: Sequence[int],
    root_width: float,
    tip_width: float,
) -> np.ndarray:
    """Build a flat per-point width array tapering each strand root-to-tip.

    ``vertex_counts`` holds one point-count per strand; the returned array is
    concatenated to match the flat ``points`` array (one width per point), which
    is why the USD ``widths`` interpolation is authored ``vertex``. Each strand
    lerps from ``root_width`` to ``tip_width`` across a normalized parameter, so
    strands taper at the same relative rate regardless of point count.
    """

    counts = np.asarray(vertex_counts, dtype=np.int64)
    total = int(counts.sum()) if counts.size else 0
    widths = np.empty(total, dtype=np.float32)
    idx = 0
    for count in counts:
        c = int(count)
        if c <= 0:
            continue
        if c == 1:
            widths[idx] = root_width
            idx += 1
            continue
        t = np.linspace(0.0, 1.0, c, dtype=np.float32)
        widths[idx : idx + c] = root_width * (1.0 - t) + tip_width * t
        idx += c
    return widths[:idx]


# --------------------------------------------------------------------------
# Physical curve widths (issue #83)
# --------------------------------------------------------------------------
def physical_curve_width(bevel_depth_meters: float, point_radius: float) -> float:
    """USD width (diameter, meters) for a beveled-curve control point.

    Generalizes the closed-#57 Classroom relation
    ``width = 2 * bevel_depth * point_radius`` into a scene-independent rule.
    ``bevel_depth_meters`` must already be expressed in meters (the caller
    multiplies the Blender bevel depth by the scene ``metersPerUnit``); the
    point radius is the per-point radius factor from the evaluated spline. No
    object scale, name, or fixture constant enters here.
    """

    return 2.0 * float(bevel_depth_meters) * float(point_radius)


def width_is_finite_positive(width: float) -> bool:
    """True when ``width`` is a finite, strictly positive number."""

    w = float(width)
    return np.isfinite(w) and w > 0.0


def width_is_implausible(
    authored_width: float,
    reference_width: float,
    *,
    tolerance_factor: float = 100.0,
) -> bool:
    """True when ``authored_width`` is physically implausible vs a reference.

    ``reference_width`` is the source-derived physical width
    (:func:`physical_curve_width`). A width is flagged when it is non-finite,
    non-positive, or exceeds the reference by more than ``tolerance_factor`` (a
    scene-independent *ratio*, not an absolute meter constant — the Classroom
    blinds were ~1000x their physical width). ``tolerance_factor`` is a tunable
    relative bound, deliberately not a fixture value.
    """

    a = float(authored_width)
    if not np.isfinite(a) or a <= 0.0:
        return True
    r = float(reference_width)
    if not np.isfinite(r) or r <= 0.0:
        # No usable reference — only the finite/positive check applies.
        return False
    return a > r * float(tolerance_factor)


# --------------------------------------------------------------------------
# BasisCurves validity
# --------------------------------------------------------------------------
def points_match_counts(num_points: int, vertex_counts: Sequence[int]) -> bool:
    """True when ``num_points == sum(vertex_counts)`` (the core USD invariant)."""

    return int(num_points) == int(np.asarray(vertex_counts, dtype=np.int64).sum())


def curve_count_valid(
    vertex_count: int,
    curve_type: str = DEFAULT_TYPE,
    basis: str = DEFAULT_BASIS,
    wrap: str = DEFAULT_WRAP,
) -> bool:
    """Validate one curve's control-point count for ``type``/``basis``/``wrap``.

    Encodes the OpenUSD UsdGeomBasisCurves rules: linear nonperiodic needs
    ``vc >= 2`` (periodic ``vc > 2``); cubic nonperiodic needs
    ``(vc - 4) % vstep == 0`` (so bspline/catmullRom need ``vc >= 4``), cubic
    periodic ``vc % vstep == 0``, cubic pinned ``vc >= 2`` (phantom endpoints).
    """

    vc = int(vertex_count)
    if curve_type == "linear":
        return vc > 2 if wrap == "periodic" else vc >= 2
    vstep = CUBIC_BASIS_VSTEP.get(basis, 1)
    if wrap == "periodic":
        return vc >= vstep and vc % vstep == 0
    if wrap == "pinned":
        return vc >= 2
    return vc >= 4 and (vc - 4) % vstep == 0


def min_points_for_cubic(basis: str = DEFAULT_BASIS, wrap: str = DEFAULT_WRAP) -> int:
    """Minimum control points a cubic curve needs to be valid."""

    if wrap == "pinned":
        return 2
    if wrap == "periodic":
        return CUBIC_BASIS_VSTEP.get(basis, 1)
    return 4


def widths_length_valid(
    interpolation: str,
    widths_length: int,
    num_points: int,
    num_curves: int,
    num_segments: int | None = None,
) -> bool:
    """Validate a ``widths`` array length against its declared interpolation.

    ``constant`` -> 1, ``uniform`` -> num_curves, ``vertex`` -> num_points.
    ``varying`` depends on type/wrap; when ``num_segments`` is given it must be
    ``num_segments + num_curves`` (linear / cubic nonperiodic / pinned) — the
    cubic-periodic ``num_segments`` case is left to the caller.
    """

    wl = int(widths_length)
    if interpolation == "constant":
        return wl == 1
    if interpolation == "uniform":
        return wl == int(num_curves)
    if interpolation == "vertex":
        return wl == int(num_points)
    if interpolation == "varying":
        if num_segments is None:
            return True
        return wl == int(num_segments) + int(num_curves)
    return False


def invalid_curve_indices(
    vertex_counts: Sequence[int],
    curve_type: str = DEFAULT_TYPE,
    basis: str = DEFAULT_BASIS,
    wrap: str = DEFAULT_WRAP,
) -> list[int]:
    """Return indices of curves whose vertex counts are invalid for the target."""

    return [
        i
        for i, vc in enumerate(np.asarray(vertex_counts, dtype=np.int64))
        if not curve_count_valid(int(vc), curve_type, basis, wrap)
    ]


def short_strand_indices(
    vertex_counts: Sequence[int],
    basis: str = DEFAULT_BASIS,
    wrap: str = DEFAULT_WRAP,
) -> list[int]:
    """Indices of strands too short to be valid cubic curves (need padding).

    The Phase 1 short-strand guard resamples/pads these (or the caller may switch
    them to ``pinned`` wrap) so variable-length evaluated strands are not dropped.
    """

    threshold = min_points_for_cubic(basis, wrap)
    return [
        i
        for i, vc in enumerate(np.asarray(vertex_counts, dtype=np.int64))
        if int(vc) < threshold
    ]


__all__ = [
    "CUBIC_BASIS_VSTEP",
    "DEFAULT_TYPE",
    "DEFAULT_BASIS",
    "DEFAULT_WRAP",
    "particle_width_range",
    "hair_sample_count",
    "fill_hair_widths",
    "physical_curve_width",
    "width_is_finite_positive",
    "width_is_implausible",
    "points_match_counts",
    "curve_count_valid",
    "min_points_for_cubic",
    "widths_length_valid",
    "invalid_curve_indices",
    "short_strand_indices",
]

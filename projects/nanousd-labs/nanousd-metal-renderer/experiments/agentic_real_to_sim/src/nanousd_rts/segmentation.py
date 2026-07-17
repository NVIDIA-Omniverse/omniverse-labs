"""Reference-preserving visual segmentation for articulated Gaussian parts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .core import AXES, Bounds, RealToSimError
from .gaussian import GaussianScene, select_bounds


PLANAR_REFINER = "nanousd-rts-planar-reference-v1"
MINIMUM_PANEL_GAUSSIANS = 24


@dataclass(frozen=True, slots=True)
class PlanarSelection:
    """A stable working-PLY selection plus auditable refinement diagnostics."""

    indices: np.ndarray
    diagnostics: dict[str, Any]


_DEFAULT_DEPTH_BANDS: dict[str, tuple[float, float]] = {
    "cabinet-door": (0.038, 0.035),
    "drawer": (0.045, 0.045),
    "oven-door": (0.080, 0.120),
    "refrigerator-door": (0.055, 0.035),
}


def _front_plane(
    positions: np.ndarray,
    *,
    proposal: Bounds,
    axis: int,
    outward_sign: int,
) -> tuple[float, dict[str, Any]]:
    points = np.asarray(positions, dtype=np.float64)
    values = points[:, axis]
    extent = float(np.ptp(values))
    if len(values) < MINIMUM_PANEL_GAUSSIANS or extent <= 1e-6:
        raise RealToSimError("planar segmentation needs a non-degenerate point proposal")

    # Five-to-eight millimetre bins are fine enough to separate a panel face from
    # the cabinet carcass at the Home Scan scale, while remaining stable at LOD5.
    bins = int(np.clip(np.ceil(extent / 0.006), 12, 96))
    counts, edges = np.histogram(values, bins=bins)
    centers = (edges[:-1] + edges[1:]) * 0.5
    normalized = (centers - centers.min()) / max(float(np.ptp(centers)), 1e-9)
    outward = normalized if outward_sign > 0 else 1.0 - normalized
    # Prefer a dense mode near the outward side without blindly taking an AABB
    # extreme, where reconstruction floaters commonly live.
    prior = 0.35 + 0.65 * np.exp(-np.square((outward - 0.70) / 0.28))
    # A dense cabinet edge can beat a comparatively sparse door face in a pure
    # 1D histogram. Score a shallow slab around each depth mode by how much of
    # the two tangent dimensions it actually covers. This is the visual signal
    # that distinguishes a panel from trim, handles, and vertical scan streaks.
    tangent_axes = [index for index in range(3) if index != axis]
    proposal_minimum = np.asarray(proposal.minimum, dtype=np.float64)
    proposal_size = np.asarray(proposal.size, dtype=np.float64)
    bin_width = float(edges[1] - edges[0])
    scoring_half_width = max(0.018, bin_width * 2.5)
    scores = np.zeros(len(centers), dtype=np.float64)
    local_counts = np.zeros(len(centers), dtype=np.int64)
    occupancies = np.zeros(len(centers), dtype=np.float64)
    coverage_geomeans = np.zeros(len(centers), dtype=np.float64)
    for index, center in enumerate(centers):
        local = points[np.abs(values - center) <= scoring_half_width]
        local_counts[index] = len(local)
        if len(local) < 4:
            continue
        normalized = (
            (local[:, tangent_axes] - proposal_minimum[tangent_axes])
            / proposal_size[tangent_axes]
        )
        cells = np.floor(np.clip(normalized, 0.0, 0.999999) * 6).astype(np.int64)
        occupancies[index] = len(np.unique(cells, axis=0)) / 36.0
        coverage = np.clip(
            np.ptp(local[:, tangent_axes], axis=0) / proposal_size[tangent_axes],
            0.0,
            1.0,
        )
        coverage_geomeans[index] = float(np.sqrt(np.prod(coverage)))
        scores[index] = (
            local_counts[index]
            * (0.15 + occupancies[index])
            * (0.20 + coverage_geomeans[index])
            * prior[index]
        )
    peak = int(np.argmax(scores))
    return float(centers[peak]), {
        "histogram_bins": bins,
        "peak_count": int(counts[peak]),
        "peak_local_count": int(local_counts[peak]),
        "peak_tangent_occupancy": float(occupancies[peak]),
        "peak_tangent_coverage_geomean": float(coverage_geomeans[peak]),
        "peak_score": float(scores[peak]),
        "scoring_half_width": scoring_half_width,
        "peak_outward_position": float(outward[peak]),
    }


def _front_alignment(scene: GaussianScene, indices: np.ndarray, axis: int) -> np.ndarray:
    """Return absolute cosine between each Gaussian minor axis and panel normal."""

    quaternions = scene.orientations[indices]
    w, x, y, z = quaternions.T
    rotations = np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=1,
    ).reshape(-1, 3, 3)
    minor_axes = np.argmin(scene.scales[indices], axis=1)
    normals = rotations[np.arange(len(indices)), :, minor_axes]
    return np.abs(normals[:, axis])


def refine_planar_selection(
    scene: GaussianScene,
    proposal: Bounds,
    *,
    front_axis: str,
    outward_sign: int,
    kind: str,
    inward_band: float | None = None,
    outward_band: float | None = None,
) -> PlanarSelection:
    """Turn a coarse AABB proposal into a stable positive/negative panel labeling.

    The AABB remains the transfer neighborhood. Only points in the robust front
    plane band become positive references; every other working-PLY point in that
    neighborhood is retained as a negative reference by the high-resolution SOG
    transfer stage.
    """

    if front_axis not in AXES:
        raise RealToSimError("front_axis must be X, Y, or Z")
    if outward_sign not in {-1, 1}:
        raise RealToSimError("outward_sign must be -1 or 1")
    if kind not in _DEFAULT_DEPTH_BANDS:
        raise RealToSimError(f"unsupported planar segmentation kind: {kind}")
    default_inward, default_outward = _DEFAULT_DEPTH_BANDS[kind]
    inward = default_inward if inward_band is None else float(inward_band)
    outward = default_outward if outward_band is None else float(outward_band)
    if not 0.005 <= inward <= 0.25 or not 0.005 <= outward <= 0.25:
        raise RealToSimError("planar segmentation bands must be within [0.005, 0.25]")

    candidates = select_bounds(scene, proposal).astype(np.uint32)
    if len(candidates) < MINIMUM_PANEL_GAUSSIANS:
        raise RealToSimError(
            f"planar proposal is too sparse: {len(candidates)} Gaussians"
        )
    axis = AXES[front_axis]
    candidate_positions = scene.positions[candidates]
    coordinates = candidate_positions[:, axis]
    plane, histogram = _front_plane(
        candidate_positions,
        proposal=proposal,
        axis=axis,
        outward_sign=outward_sign,
    )
    oriented_depth = (coordinates - plane) * outward_sign
    selected_mask = (oriented_depth >= -inward) & (oriented_depth <= outward)
    selected = candidates[selected_mask]
    negative_count = int(len(candidates) - len(selected))
    if len(selected) < MINIMUM_PANEL_GAUSSIANS:
        raise RealToSimError(
            f"planar refinement is too sparse: {len(selected)} positive references"
        )
    if negative_count == 0:
        raise RealToSimError("planar refinement produced no negative references")

    selected_positions = scene.positions[selected]
    alignment = _front_alignment(scene, selected, axis)
    tangent_axes = [index for index in range(3) if index != axis]
    proposal_size = np.asarray(proposal.size, dtype=np.float64)
    selected_extent = np.ptp(selected_positions, axis=0)
    tangent_coverage = {
        "XYZ"[index]: float(selected_extent[index] / proposal_size[index])
        for index in tangent_axes
    }
    normalized_tangent = (
        (selected_positions[:, tangent_axes] - np.asarray(proposal.minimum)[tangent_axes])
        / proposal_size[tangent_axes]
    )
    tangent_cells = np.floor(
        np.clip(normalized_tangent, 0.0, 0.999999) * 8
    ).astype(np.int64)
    tangent_occupancy = float(len(np.unique(tangent_cells, axis=0)) / 64.0)
    diagnostics: dict[str, Any] = {
        "schema_version": 1,
        "refiner": PLANAR_REFINER,
        "kind": kind,
        "front_axis": front_axis,
        "outward_sign": outward_sign,
        "proposal_bounds": proposal.to_json(),
        "candidate_gaussians": int(len(candidates)),
        "positive_references": int(len(selected)),
        "negative_references": negative_count,
        "positive_fraction": float(len(selected) / len(candidates)),
        "front_plane": plane,
        "inward_band": inward,
        "outward_band": outward,
        "depth_residual_q95": float(np.quantile(np.abs(oriented_depth[selected_mask]), 0.95)),
        "front_alignment_median": float(np.median(alignment)),
        "front_alignment_q10": float(np.quantile(alignment, 0.10)),
        "tangent_coverage": tangent_coverage,
        "tangent_occupancy_8x8": tangent_occupancy,
        "selected_bounds": Bounds(
            tuple(selected_positions.min(axis=0)),
            tuple(selected_positions.max(axis=0)),
        ).to_json(),
        **histogram,
    }
    return PlanarSelection(np.ascontiguousarray(selected, dtype=np.uint32), diagnostics)

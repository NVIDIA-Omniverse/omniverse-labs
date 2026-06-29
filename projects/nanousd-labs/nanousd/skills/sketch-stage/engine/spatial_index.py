# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""3D spatial index for placements + OBB collision via SAT.

Each placement is stored as an oriented bounding box (OBB):
  center (cx, cy, cz), half-extents (hw, hd, hh), yaw rotation about Z.

Broad phase: 3D spatial-hash grid. Each item registers in every cell its
3D AABB (the conservative envelope of the OBB) touches. A collision query
walks only the cells the query AABB overlaps.

Narrow phase: OBB-vs-OBB collision via separating-axis theorem (SAT).
Rotation is yaw-only (about Z), so SAT reduces to a Z interval test plus
2D SAT on XY rotated rectangles (4 axes). This catches skewed cases the
old AABB-of-rotated-rect would miss or over-report.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass
class IndexedPlacement:
    id: str
    archetype: str
    posM: tuple[float, float, float]
    slotM: tuple[float, float, float]
    yawDeg: float
    assetPath: str | None = None  # explicit asset (bypasses best-fit at realize)
    scaleM: tuple[float, float, float] | None = None  # per-axis scale
    parentZoneId: str | None = None  # zone this placement belongs to (template view)
    sourcePath: str | None = None  # original USD prim path from the absorbed source (if any)


# ---------- OBB primitives (yaw-only rotation about Z) ---------- #

def _obb_axes_xy(yaw_deg: float) -> tuple[tuple[float, float], tuple[float, float]]:
    r = math.radians(yaw_deg)
    c, s = math.cos(r), math.sin(r)
    return (c, s), (-s, c)  # local X axis, local Y axis (unit vectors in world)


def _obb_aabb_3d(pos, slot, yaw_deg) -> tuple[float, float, float, float, float, float]:
    """Conservative axis-aligned 3D envelope of the rotated OBB."""
    cx, cy, cz = pos
    w, d, h = slot
    hw, hd, hh = w / 2, d / 2, h / 2
    r = math.radians(yaw_deg)
    c, s = abs(math.cos(r)), abs(math.sin(r))
    rw = hw * c + hd * s
    rd = hw * s + hd * c
    return (cx - rw, cy - rd, cz - hh, cx + rw, cy + rd, cz + hh)


def _obb_overlap(a_pos, a_slot, a_yaw, b_pos, b_slot, b_yaw) -> bool:
    """OBB-vs-OBB SAT collision (yaw-only). True iff actually overlapping."""
    # Z interval test (cheap reject)
    a_hh = a_slot[2] / 2
    b_hh = b_slot[2] / 2
    if abs(a_pos[2] - b_pos[2]) > a_hh + b_hh:
        return False
    # 2D SAT on XY: 4 candidate separating axes (each box's local X and Y).
    a_x, a_y = _obb_axes_xy(a_yaw)
    b_x, b_y = _obb_axes_xy(b_yaw)
    a_hw, a_hd = a_slot[0] / 2, a_slot[1] / 2
    b_hw, b_hd = b_slot[0] / 2, b_slot[1] / 2
    dx = b_pos[0] - a_pos[0]
    dy = b_pos[1] - a_pos[1]
    for ax, ay in (a_x, a_y, b_x, b_y):
        # Project center-to-center distance onto axis
        t = abs(dx * ax + dy * ay)
        # Project box A extent onto axis: hw·|axis·A_x| + hd·|axis·A_y|
        ra = a_hw * abs(ax * a_x[0] + ay * a_x[1]) + a_hd * abs(ax * a_y[0] + ay * a_y[1])
        rb = b_hw * abs(ax * b_x[0] + ay * b_x[1]) + b_hd * abs(ax * b_y[0] + ay * b_y[1])
        if t > ra + rb:
            return False  # separating axis found
    return True


def _aabb_overlap_volume_3d(a, b) -> float:
    """Approximate overlap volume of two 3D AABBs (used as a sortable score)."""
    ox = max(0.0, min(a[3], b[3]) - max(a[0], b[0]))
    oy = max(0.0, min(a[4], b[4]) - max(a[1], b[1]))
    oz = max(0.0, min(a[5], b[5]) - max(a[2], b[2]))
    return ox * oy * oz


def _obb_overlap_batch(q_pos, q_slot, q_yaw_deg,
                       p_pos: np.ndarray, p_slot: np.ndarray,
                       p_yaw_deg: np.ndarray) -> np.ndarray:
    """Vectorized OBB-vs-OBB SAT. One query OBB vs N candidate OBBs.

    Args:
        q_pos: (3,) query center
        q_slot: (3,) query (w, d, h)
        q_yaw_deg: scalar
        p_pos: (N, 3)
        p_slot: (N, 3)
        p_yaw_deg: (N,)
    Returns:
        bool array (N,) — True where the OBBs actually overlap.
    """
    n = p_pos.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)

    # Z interval test (cheap reject) — short-circuits a lot of candidates
    q_hh = q_slot[2] * 0.5
    p_hh = p_slot[:, 2] * 0.5
    z_ok = np.abs(p_pos[:, 2] - q_pos[2]) <= (q_hh + p_hh)
    if not z_ok.any():
        return z_ok  # all-False

    # Build local axes for every box (yaw rotation about Z)
    q_r = math.radians(q_yaw_deg)
    qcx, qsx = math.cos(q_r), math.sin(q_r)
    q_x = np.array([qcx, qsx])               # (2,)
    q_y = np.array([-qsx, qcx])              # (2,)

    p_r = np.radians(p_yaw_deg)
    pcx, psx = np.cos(p_r), np.sin(p_r)      # (N,)
    p_x = np.stack([pcx, psx], axis=1)       # (N, 2)
    p_y = np.stack([-psx, pcx], axis=1)      # (N, 2)

    q_hw, q_hd = q_slot[0] * 0.5, q_slot[1] * 0.5
    p_hw = p_slot[:, 0] * 0.5
    p_hd = p_slot[:, 1] * 0.5

    d = p_pos[:, :2] - np.array(q_pos[:2])   # (N, 2)

    # 4 candidate separating axes: q_x, q_y, p_x[i], p_y[i]
    # Stack to (N, 4, 2) so we can vectorize the rest.
    axes = np.empty((n, 4, 2), dtype=float)
    axes[:, 0, :] = q_x
    axes[:, 1, :] = q_y
    axes[:, 2, :] = p_x
    axes[:, 3, :] = p_y

    # Project center-to-center on each axis: t = |d · axis|, shape (N, 4)
    t = np.abs(np.einsum("ni,nai->na", d, axes))

    # Project box extents onto each axis. For each axis a:
    #   r = hw * |a · X| + hd * |a · Y|  where X,Y are the box's local axes
    # Q's projection is the same array of axes dotted with q_x, q_y.
    rq = (q_hw * np.abs(np.einsum("nai,i->na", axes, q_x)) +
          q_hd * np.abs(np.einsum("nai,i->na", axes, q_y)))
    rp = (p_hw[:, None] * np.abs(np.einsum("nai,ni->na", axes, p_x)) +
          p_hd[:, None] * np.abs(np.einsum("nai,ni->na", axes, p_y)))

    # SAT: separated on any axis ⇒ no overlap. So "no separation on any axis" ⇒ overlap.
    separated = t > (rq + rp)               # (N, 4)
    xy_ok = ~separated.any(axis=1)
    return z_ok & xy_ok


# ---------- Spatial index ---------- #

class PlacementIndex:
    """3D spatial-hash grid + OBB-SAT narrow phase.

    Cells are CELL_SIZE meters in X, Y, and Z; each placement registers
    under every cell its conservative 3D AABB touches. Collision queries
    walk only the cells the query AABB overlaps and confirm with an exact
    OBB-OBB SAT test, so yaw-rotated rects don't false-overlap and stacked
    placements at different Z don't collide.
    """

    CELL_SIZE = 5.0

    def __init__(self) -> None:
        self._items: dict[str, IndexedPlacement] = {}
        self._grid: dict[tuple[int, int, int], set[str]] = defaultdict(set)
        # Cache the 3D AABB per item so we can quickly unindex / score overlaps.
        self._aabb_by_id: dict[str, tuple[float, float, float, float, float, float]] = {}

    def __len__(self) -> int:
        return len(self._items)

    @staticmethod
    def _aabb(pos: tuple[float, float, float],
              slot: tuple[float, float, float],
              yaw_deg: float = 0.0) -> tuple[float, float, float, float, float, float]:
        return _obb_aabb_3d(pos, slot, yaw_deg)

    def _cells_for_aabb(self, ax0, ay0, az0, ax1, ay1, az1):
        cs = self.CELL_SIZE
        i0, j0, k0 = math.floor(ax0 / cs), math.floor(ay0 / cs), math.floor(az0 / cs)
        i1, j1, k1 = math.floor(ax1 / cs), math.floor(ay1 / cs), math.floor(az1 / cs)
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                for k in range(k0, k1 + 1):
                    yield (i, j, k)

    def _index_aabb(self, pid: str, aabb):
        self._aabb_by_id[pid] = aabb
        for cell in self._cells_for_aabb(*aabb):
            self._grid[cell].add(pid)

    def _unindex_aabb(self, pid: str):
        aabb = self._aabb_by_id.pop(pid, None)
        if aabb is None:
            return
        for cell in self._cells_for_aabb(*aabb):
            self._grid[cell].discard(pid)

    def insert(self, p: IndexedPlacement) -> None:
        if p.id in self._items:
            self._unindex_aabb(p.id)
        self._items[p.id] = p
        self._index_aabb(p.id, self._aabb(p.posM, p.slotM, p.yawDeg))

    def remove(self, pid: str) -> bool:
        if pid not in self._items:
            return False
        self._unindex_aabb(pid)
        del self._items[pid]
        return True

    def update(self, pid: str, **fields) -> bool:
        p = self._items.get(pid)
        if not p:
            return False
        for k, v in fields.items():
            if hasattr(p, k):
                setattr(p, k, v)
        # Re-index: pose-affecting fields (posM/slotM/yawDeg) require AABB refresh
        self._unindex_aabb(pid)
        self._index_aabb(pid, self._aabb(p.posM, p.slotM, p.yawDeg))
        return True

    def get(self, pid: str) -> "IndexedPlacement | None":
        return self._items.get(pid)

    def all(self) -> list[IndexedPlacement]:
        return list(self._items.values())

    def query_nearby(self, point: tuple[float, float],
                     radius_m: float,
                     archetype: str | None = None,
                     limit: int = 200) -> list[IndexedPlacement]:
        """XY-distance nearest-first. Z is intentionally ignored — this is
        the LLM-facing 'what's around me on the floor' query."""
        px, py = point
        out: list[tuple[float, IndexedPlacement]] = []
        for p in self._items.values():
            if archetype and p.archetype != archetype:
                continue
            dx = p.posM[0] - px
            dy = p.posM[1] - py
            d = math.hypot(dx, dy)
            if d <= radius_m:
                out.append((d, p))
        out.sort(key=lambda t: t[0])
        return [p for _, p in out[:limit]]

    def query_collision(self, posM: tuple[float, float, float],
                        slotM: tuple[float, float, float],
                        yaw_deg: float = 0.0,
                        ignore_id: str | None = None) -> list[tuple[IndexedPlacement, float]]:
        """OBB-vs-OBB collision via SAT; broad-phase via 3D spatial hash.

        Returns placements whose OBBs *actually* overlap, scored by 3D AABB
        overlap volume (cheap proxy used purely for ranking — the gate
        uses the boolean SAT result, not the score).

        Narrow phase is vectorized in numpy: one matmul-style pass over all
        candidates instead of a Python-level OBB-vs-OBB loop.
        """
        q_aabb = self._aabb(posM, slotM, yaw_deg)
        ax0, ay0, az0, ax1, ay1, az1 = q_aabb
        candidates: set[str] = set()
        for cell in self._cells_for_aabb(ax0, ay0, az0, ax1, ay1, az1):
            candidates.update(self._grid.get(cell, ()))
        if ignore_id:
            candidates.discard(ignore_id)
        if not candidates:
            return []

        # Cheap AABB reject before stuffing into numpy arrays.
        ids: list[str] = []
        b_aabbs: list[tuple] = []
        for pid in candidates:
            b = self._aabb_by_id[pid]
            if (ax1 < b[0] or ax0 > b[3] or
                ay1 < b[1] or ay0 > b[4] or
                az1 < b[2] or az0 > b[5]):
                continue
            ids.append(pid)
            b_aabbs.append(b)
        if not ids:
            return []

        n = len(ids)
        p_pos = np.empty((n, 3), dtype=float)
        p_slot = np.empty((n, 3), dtype=float)
        p_yaw = np.empty(n, dtype=float)
        for i, pid in enumerate(ids):
            p = self._items[pid]
            p_pos[i] = p.posM
            p_slot[i] = p.slotM
            p_yaw[i] = p.yawDeg

        hits = _obb_overlap_batch(np.asarray(posM, dtype=float),
                                  np.asarray(slotM, dtype=float),
                                  yaw_deg, p_pos, p_slot, p_yaw)
        out: list[tuple[IndexedPlacement, float]] = []
        for i, hit in enumerate(hits):
            if not hit:
                continue
            score = _aabb_overlap_volume_3d(q_aabb, b_aabbs[i])
            out.append((self._items[ids[i]], score))
        out.sort(key=lambda t: -t[1])
        return out


# Public re-exports for callers that want the OBB primitives without
# instantiating the index (e.g., absorb-time dedup).
__all__ = [
    "IndexedPlacement",
    "PlacementIndex",
    "obb_overlap",
    "obb_aabb_3d",
]


def obb_overlap(a_pos, a_slot, a_yaw, b_pos, b_slot, b_yaw) -> bool:
    return _obb_overlap(a_pos, a_slot, a_yaw, b_pos, b_slot, b_yaw)


def obb_aabb_3d(pos, slot, yaw_deg):
    return _obb_aabb_3d(pos, slot, yaw_deg)

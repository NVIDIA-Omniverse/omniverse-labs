# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GlesRendererAdapter — RendererAdapter that drives libnusd_gles.so.

ovgear's viewport widget hands us a 4×4 view matrix and 4×4 projection
matrix every frame; we forward them to the headless OpenGL renderer and
return the rendered frame as ``(H, W, 4)`` uint8 RGBA. Picking is
implemented in Python via ray–AABB intersection over the cached prim
bounds, mirroring the strategy used by ``OvRtxRendererAdapter`` so the
selection workflow keeps working without a GPU pick pass.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from ovgear.adapters import RendererAdapter

from nusd_gles._bindings import GlesViewer
from nusd_gles._nanousd import NanousdStage, _Prim


def _eye_from_view(view: np.ndarray) -> np.ndarray:
    """Recover the camera world-space origin from a GL view matrix.

    A GL view matrix maps world → view; the camera origin in world space
    is ``-R^T @ t`` where ``R`` is the upper-left 3×3 and ``t`` is the
    last column of the view matrix.
    """
    v = np.asarray(view, dtype=np.float64).reshape(4, 4)
    R = v[:3, :3]
    t = v[:3, 3]
    eye = -R.T @ t
    return np.asarray(eye, dtype=np.float32)


class GlesRendererAdapter(RendererAdapter):
    """Concrete RendererAdapter backed by the OpenGL ES nanoUSD viewer."""

    _DEBUG_FRAME_PATH = None  # set to a path to dump each frame for triage

    def __init__(self) -> None:
        self._viewer: Optional[GlesViewer] = None
        self._stage: Optional[NanousdStage] = None
        self._stage_path: Optional[str] = None
        self._width: int = 1280
        self._height: int = 720
        self._selected_paths: List[str] = []
        # Last (view, proj) supplied by the viewport for picking.
        self._last_view: Optional[np.ndarray] = None
        self._last_proj: Optional[np.ndarray] = None
        # Scratch buffer the C side writes into. We always copy out
        # before returning to the embedder — ImageBridge / ByteImageProvider
        # holds the returned numpy array across frames, so handing back
        # the same buffer would overwrite the pixels it is currently
        # presenting.
        self._scratch: Optional[np.ndarray] = None

    # ── Stage loading ─────────────────────────────────────────────────────

    def load_stage(self, stage: Any) -> None:
        """Load a stage. Accepts a NanousdStage or a file path string."""
        if isinstance(stage, NanousdStage):
            self._stage = stage
            self._stage_path = stage.filepath
        elif isinstance(stage, (str, Path)):
            self._stage_path = str(stage)
            self._stage = NanousdStage(self._stage_path)
        else:
            raise TypeError(
                f"GlesRendererAdapter.load_stage expected NanousdStage or "
                f"str path, got {type(stage).__name__}"
            )

        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        self._viewer = GlesViewer(
            usd_path=self._stage_path,
            width=self._width,
            height=self._height,
        )
        # Drop the cached frame buffer — its shape may not match.
        self._scratch = None

    # ── Per-frame rendering ───────────────────────────────────────────────

    def render_frame(
        self,
        width: int,
        height: int,
        view_matrix: Any,
        proj_matrix: Any,
    ) -> np.ndarray:
        w = max(int(width), 1)
        h = max(int(height), 1)
        if self._viewer is None:
            return np.zeros((h, w, 4), dtype=np.uint8)
        eye = _eye_from_view(view_matrix)
        # Cache the matrices for picking.
        try:
            self._last_view = np.asarray(view_matrix, dtype=np.float64)
            self._last_proj = np.asarray(proj_matrix, dtype=np.float64)
        except Exception:
            pass
        if self._scratch is None or self._scratch.shape != (h, w, 4):
            self._scratch = np.empty((h, w, 4), dtype=np.uint8)
        try:
            self._viewer.render(
                w, h,
                np.asarray(view_matrix, dtype=np.float32),
                np.asarray(proj_matrix, dtype=np.float32),
                eye,
                out=self._scratch,
            )
        except Exception:
            self._scratch.fill(0)
        # Return a fresh copy so the embedder can safely retain the
        # array across frames without us overwriting it next render.
        out = self._scratch.copy()
        # Optional debug dump — set the env var ``NUSD_GLES_DUMP_FRAME`` to
        # a path prefix, and the next frame is written to <prefix>_NNN.ppm.
        import os
        dump = os.environ.get("NUSD_GLES_DUMP_FRAME")
        if dump:
            cnt = getattr(self, "_dbg_count", 0)
            if cnt < 3:  # first few frames only
                with open(f"{dump}_{cnt:03d}.ppm", "wb") as f:
                    f.write(f"P6\n{w} {h}\n255\n".encode())
                    f.write(out[..., :3].tobytes())
                self._dbg_count = cnt + 1
        return out

    def set_resolution(self, width: int, height: int) -> None:
        self._width = max(int(width), 1)
        self._height = max(int(height), 1)
        if self._viewer is not None:
            self._viewer.resize(self._width, self._height)
        self._scratch = None

    # ── Picking ───────────────────────────────────────────────────────────

    def pick(
        self,
        x: float,
        y: float,
        callback: Callable[[Optional[str], Optional[Tuple[float, float, float]]], None],
        query_name: str,
    ) -> None:
        try:
            from ovuiviewport.pick_ray import pick_closest, screen_ndc_to_ray
            cands = self._gather_pick_candidates()
            if not cands or self._last_view is None or self._last_proj is None:
                callback(None, None)
                return
            origin, direction = screen_ndc_to_ray(
                float(x), float(y), self._last_view, self._last_proj,
            )
            hit = pick_closest(origin, direction, cands)
        except Exception:
            callback(None, None)
            return
        if hit is None:
            callback(None, None)
        else:
            path, world_point = hit
            callback(path, world_point)

    def cancel_pick(self, query_name: str) -> None:
        return

    def pick_rect(
        self,
        x0: float, y0: float, x1: float, y1: float,
        callback: Callable[[List[str]], None],
    ) -> None:
        try:
            from ovuiviewport.pick_ray import pick_rect_paths
            cands = self._gather_pick_candidates()
            if not cands or self._last_view is None or self._last_proj is None:
                callback([])
                return
            rect = (
                min(float(x0), float(x1)),
                min(float(y0), float(y1)),
                max(float(x0), float(x1)),
                max(float(y0), float(y1)),
            )
            paths = pick_rect_paths(rect, self._last_view, self._last_proj, cands)
        except Exception:
            callback([])
            return
        callback(paths)

    def _gather_pick_candidates(self) -> list:
        """Walk the stage and produce ``[(path, ((min), (max)))]`` for
        every Gprim with a usable extent, transformed to world space.
        """
        if self._stage is None:
            return []
        stage = self._stage
        gprim_types = {
            "Mesh", "Cube", "Sphere", "Cylinder", "Cone", "Capsule", "Plane",
            "Points", "BasisCurves", "NurbsCurves", "NurbsPatch",
        }
        results: list = []
        # Lazy import — keep the module importable without numpy.linalg
        # at module load.
        from nusd_gles.transform_adapter import NanousdTransformAdapter
        ta = NanousdTransformAdapter(stage)

        n = stage.n_prims()
        for i in range(n):
            prim = stage.prim_at_index(i)
            if prim is None or prim.type_name not in gprim_types:
                continue
            # Visibility check — invisible prims should not pick.
            try:
                if prim.get_attrib_str("visibility", "inherited") == "invisible":
                    continue
            except Exception:
                pass
            extent_min, extent_max = self._read_extent(prim)
            if extent_min is None:
                continue
            world = ta.get_world_transform(prim.path)
            wmin, wmax = self._transform_aabb(extent_min, extent_max, world)
            results.append((prim.path, (wmin, wmax)))
        return results

    @staticmethod
    def _read_extent(prim) -> tuple:
        """Return ``((minx, miny, minz), (maxx, maxy, maxz))`` from the
        prim's ``extent`` attribute (preferred) or a type-based fallback
        derived from radius/size/height. ``(None, None)`` if neither is
        available.
        """
        try:
            arr = prim.get_attrib_arrayv3f("extent", 2)
        except Exception:
            arr = []
        if len(arr) >= 2:
            return (arr[0], arr[1])
        type_name = prim.type_name
        if type_name == "Sphere":
            r = prim.get_attrib_float("radius", 1.0) or 1.0
            return ((-r, -r, -r), (r, r, r))
        if type_name == "Cube":
            s = (prim.get_attrib_float("size", 2.0) or 2.0) * 0.5
            return ((-s, -s, -s), (s, s, s))
        if type_name in ("Cylinder", "Cone", "Capsule"):
            r = prim.get_attrib_float("radius", 1.0) or 1.0
            h = prim.get_attrib_float("height", 2.0) or 2.0
            return ((-r, -h * 0.5, -r), (r, h * 0.5, r))
        if type_name == "Mesh":
            # No extent and no fallback parameters — use a unit-ish box
            # so the prim is still pickable from the centre of view.
            return ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
        return (None, None)

    @staticmethod
    def _transform_aabb(bmin, bmax, world):
        """Transform a local AABB by a 4x4 row-major world matrix and
        return the axis-aligned enclosing world-space AABB.
        """
        w = np.asarray(world, dtype=np.float64).reshape(4, 4)
        corners = np.array([
            [bmin[0], bmin[1], bmin[2], 1.0],
            [bmax[0], bmin[1], bmin[2], 1.0],
            [bmin[0], bmax[1], bmin[2], 1.0],
            [bmax[0], bmax[1], bmin[2], 1.0],
            [bmin[0], bmin[1], bmax[2], 1.0],
            [bmax[0], bmin[1], bmax[2], 1.0],
            [bmin[0], bmax[1], bmax[2], 1.0],
            [bmax[0], bmax[1], bmax[2], 1.0],
        ], dtype=np.float64)
        # Row-major: world . point — world stores translation in the
        # last column for column-vector multiplication, but ovgear's
        # transforms use row-vector convention with translation in
        # the last row. Try both and pick whichever yields the larger
        # finite AABB. (NanousdTransformAdapter.get_world_transform
        # returns the multiplied chain of nanousd_get_local_transform
        # outputs which are USD's row-major form.)
        out = corners @ w
        out_xyz = out[:, :3]
        return (
            (float(out_xyz[:, 0].min()), float(out_xyz[:, 1].min()), float(out_xyz[:, 2].min())),
            (float(out_xyz[:, 0].max()), float(out_xyz[:, 1].max()), float(out_xyz[:, 2].max())),
        )

    def set_selection_highlight(self, paths: List[str]) -> None:
        # The OpenGL renderer doesn't draw a highlight overlay yet; just
        # remember the selection in case a future build adds one.
        self._selected_paths = list(paths)

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def get_scene_bounds(self) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """Return the loaded scene's world-space AABB or None."""
        if self._viewer is None:
            return None
        return self._viewer.get_scene_bounds()

    def shutdown(self) -> None:
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None
        if self._stage is not None:
            try:
                self._stage.close()
            except Exception:
                pass
            self._stage = None

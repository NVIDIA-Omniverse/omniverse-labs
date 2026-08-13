# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scan a USD stage for variant sets and cameras (pxr). Off the render thread."""
from __future__ import annotations

from pxr import Usd, UsdGeom

from dev_variant_presenter.models import CameraInfo, StageInfo, VariantSetInfo
from dev_variant_presenter.usd_guard import USD_LOCK


def _open_composed(usd_path: str, extra_layers: tuple[str, ...] = ()):
    """Open the user stage, optionally composed under sidecar edits layers (turntable
    camera etc.) so their prims and time-range opinions are visible like authored content."""
    if not extra_layers:
        return Usd.Stage.Open(usd_path)
    from pathlib import Path
    from pxr import Sdf
    root = Sdf.Layer.CreateAnonymous(".usda")
    root.subLayerPaths = ([Path(e).absolute().as_posix() for e in extra_layers]
                          + [Path(usd_path).absolute().as_posix()])
    user = Sdf.Layer.FindOrOpen(Path(usd_path).absolute().as_posix())
    if user is not None and user.defaultPrim:
        root.defaultPrim = user.defaultPrim   # scanner scopes variant discovery by it
    from dev_variant_presenter.render.composer import hoist_time_metadata
    hoist_time_metadata(root)   # root-only time metadata + tps must match (see composer)
    return Usd.Stage.Open(root)


def read_camera(usd_path: str, camera_path: str,
                extra_layers: tuple[str, ...] = ()) -> tuple[tuple[float, ...], float]:
    """Return (world 4x4 row-major as 16 floats, focusDistance) for an authored camera.
    Used by the API to seed/snap the viewer camera. pxr — off the render thread, so it
    holds USD_LOCK (opening/composing a stage races concurrent pxr authoring)."""
    with USD_LOCK:
        stage = _open_composed(usd_path, extra_layers)
        prim = stage.GetPrimAtPath(camera_path)
        if not prim or not prim.IsValid():
            ident = (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
            return tuple(float(v) for v in ident), 0.0
        m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        xf = tuple(float(m[i][j]) for i in range(4) for j in range(4))
        focus = UsdGeom.Camera(prim).GetFocusDistanceAttr().Get() or 0.0
        return xf, float(focus)


def _xform_animated(prim) -> bool:
    """Time-sampled xform ops on the prim or ANY ancestor (turntable rigs animate the
    parent pivot, not the camera prim itself)."""
    while prim and prim.IsValid() and not prim.IsPseudoRoot():
        xf = UsdGeom.Xformable(prim)
        if xf:
            for op in xf.GetOrderedXformOps():
                if op.GetNumTimeSamples() > 1:
                    return True
        prim = prim.GetParent()
    return False


def scan_stage(usd_path: str, extra_layers: tuple[str, ...] = ()) -> StageInfo:
    with USD_LOCK:   # opening/composing a stage races concurrent pxr authoring elsewhere
        return _scan_stage(usd_path, extra_layers)


def _scan_stage(usd_path: str, extra_layers: tuple[str, ...] = ()) -> StageInfo:
    stage = _open_composed(usd_path, extra_layers)
    if not stage:
        raise ValueError(f"Failed to open USD stage: {usd_path}")
    dp = stage.GetDefaultPrim()
    default = dp.GetName() if dp else ""
    root = f"/{default}" if default else ""
    vsets: list[VariantSetInfo] = []
    cams: list[CameraInfo] = []
    seen: set[tuple[str, str]] = set()
    for prim in stage.Traverse():
        p = str(prim.GetPath())
        if prim.GetTypeName() == "Camera":
            cams.append(CameraInfo(p, prim.GetName(), animated=_xform_animated(prim)))
        if root and not p.startswith(root):
            continue
        vs = prim.GetVariantSets()
        for sn in vs.GetNames():
            if (p, sn) in seen:
                continue
            seen.add((p, sn))
            s = vs.GetVariantSet(sn)
            names = tuple(s.GetVariantNames())
            if names:
                vsets.append(VariantSetInfo(sn, p, names, s.GetVariantSelection()))
    return StageInfo(
        usd_path=usd_path,
        default_prim=default,
        up_axis=stage.GetMetadata("upAxis") or "Y",
        start_time=stage.GetStartTimeCode(),
        end_time=stage.GetEndTimeCode(),
        fps=stage.GetTimeCodesPerSecond() or 24.0,
        variant_sets=tuple(vsets),
        cameras=tuple(cams),
        meters_per_unit=float(UsdGeom.GetStageMetersPerUnit(stage) or 1.0),
    )

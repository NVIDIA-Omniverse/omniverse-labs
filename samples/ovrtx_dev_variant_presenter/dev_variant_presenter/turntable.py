# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Author a turntable camera as a SIDECAR edits layer — the source stage is never
written (mandatory for read-only / mirrored content). The layer holds
/Turntable/Camera with one full 360° revolution around a pivot as time-sampled
transforms, plus the time-range metadata. Composites sublayer it ABOVE the user
stage, so the camera behaves like authored content everywhere: the camera dropdown,
the Grid's camera list, and the Grid's animation range (start/end resolve from this
layer's opinions).

The orbit looks AT the pivot every frame; matrices use the app's row-vector
convention (rows = right / up / -forward / eye). Y-up and Z-up are both handled.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

import numpy as np

from dev_variant_presenter.usd_guard import USD_LOCK

PIVOT_PATH = "/TurntableRig"
CAMERA_PATH = "/TurntableRig/Turntable"   # leaf name is the camera dropdown's display name


def edit_layer_path(user_usd: str, data_root: str = "data") -> str:
    """Stable sidecar location for a stage's edits layer, keyed on the stage path."""
    # Case-fold ONLY on Windows, where `C:\A.usd` and `c:\a.usd` are the same file and must
    # share one sidecar. On POSIX they are DIFFERENT stages, so folding would collide them
    # onto one turntable.usda and let one stage's rig overwrite the other's.
    abs_path = str(Path(user_usd).absolute())
    key = hashlib.sha1(
        (abs_path.lower() if os.name == "nt" else abs_path).encode()).hexdigest()[:12]
    return str(Path(data_root) / "_edits" / key / "turntable.usda")


def orbit_matrix(pivot, radius: float, height: float, theta: float, up_axis: str = "Y") -> np.ndarray:
    """Row-vector 4x4 world transform for an eye orbiting `pivot` at angle `theta`,
    offset `height` along the up axis, always looking at the pivot."""
    p = np.asarray(pivot, dtype=np.float64)
    if up_axis.upper() == "Z":
        eye = p + np.array([radius * math.cos(theta), radius * math.sin(theta), height])
        world_up = np.array([0.0, 0.0, 1.0])
    else:
        eye = p + np.array([radius * math.sin(theta), height, radius * math.cos(theta)])
        world_up = np.array([0.0, 1.0, 0.0])
    fwd = p - eye
    n = np.linalg.norm(fwd)
    fwd = fwd / n if n > 1e-9 else np.array([0.0, 0.0, -1.0])
    right = np.cross(fwd, world_up)
    rn = np.linalg.norm(right)
    right = right / rn if rn > 1e-9 else np.array([1.0, 0.0, 0.0])
    up = np.cross(right, fwd)
    m = np.eye(4, dtype=np.float64)
    m[0, :3] = right
    m[1, :3] = up
    m[2, :3] = -fwd
    m[3, :3] = eye
    return m


def author_turntable(user_usd: str, *, pivot, radius: float, height: float = 0.0,
                     frames: int = 120, fps: float = 30.0, focal_length: float = 35.0,
                     up_axis: str = "Y", start_deg: float = 0.0,
                     camera_world=None, data_root: str = "data") -> str:
    """Write (or rewrite) the edits layer with the turntable RIG and return the layer path.

    Proper DCC rig semantics (not baked eye positions): /Turntable is an Xform AT the
    pivot whose up-axis rotation is animated 0->360 across the frame range; the camera
    is its CHILD at a static local offset. The parent's spin carries the camera around —
    and the rig stays editable: retime by re-animating one attr, reframe by moving one
    offset.

    camera_world (16 floats, row-vector world matrix): author the user's EXACT current
    pose as the camera — expressed in the pivot's frame-0 coordinate system, so frame 0
    reproduces the composed view INCLUDING any pan offset, and the pivot point holds the
    same screen position for the whole revolution. Without it, the camera falls back to
    a pivot-centered orbit framing (radius out, height up, looking at the pivot)."""
    from pxr import Gf, Sdf, Usd, UsdGeom
    out = Path(edit_layer_path(user_usd, data_root))
    out.parent.mkdir(parents=True, exist_ok=True)
    frames = max(2, int(frames))
    with USD_LOCK:
        # the layer may still be ALIVE in pxr's process-global registry (composed scans hold
        # references) — CreateNew would throw 'layer already exists' even after deleting the
        # file. Reuse-and-clear instead of recreate.
        layer = Sdf.Layer.Find(str(out)) or (Sdf.Layer.FindOrOpen(str(out)) if out.exists() else None)
        if layer is not None:
            layer.Clear()
        else:
            layer = Sdf.Layer.CreateNew(str(out))
        stage = Usd.Stage.Open(layer)
        stage.SetStartTimeCode(0)
        stage.SetEndTimeCode(frames - 1)
        stage.SetTimeCodesPerSecond(float(fps))
        stage.SetFramesPerSecond(float(fps))
        rig = UsdGeom.Xform.Define(stage, PIVOT_PATH)
        rig.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in pivot]))
        if up_axis.upper() == "Z":
            spin = rig.AddRotateZOp(opSuffix="spin")
        else:
            spin = rig.AddRotateYOp(opSuffix="spin")
        start = float(start_deg)                   # frame 0 = the azimuth the user composed from
        spin.Set(start)                            # default-time pose == frame 0 (read_camera seeds it)
        spin.Set(start, 0.0)
        spin.Set(start + 360.0, float(frames))     # linear interp: frame i -> start+360*i/frames,
                                                   # so rendering 0..frames-1 is one seamless loop
        cam = UsdGeom.Camera.Define(stage, CAMERA_PATH)
        cam.CreateFocalLengthAttr(float(focal_length))
        if camera_world is not None and len(tuple(camera_world)) == 16:
            # exact-pose framing: local = world x inverse(pivot frame 0). The pivot's
            # local transform composes spin-then-translate for row vectors (xformOpOrder
            # ["translate", "spin"]), so frame 0 is R(start_deg) * T(pivot).
            axis = Gf.Vec3d(0, 0, 1) if up_axis.upper() == "Z" else Gf.Vec3d(0, 1, 0)
            p0 = (Gf.Matrix4d().SetRotate(Gf.Rotation(axis, start))
                  * Gf.Matrix4d().SetTranslate(Gf.Vec3d(*[float(v) for v in pivot])))
            world = Gf.Matrix4d(*[float(v) for v in camera_world])
            local_m = world * p0.GetInverse()
        else:
            local = orbit_matrix([0.0, 0.0, 0.0], float(radius), float(height), 0.0, up_axis)
            local_m = Gf.Matrix4d(*[float(v) for v in local.reshape(-1)])
        UsdGeom.Xformable(cam.GetPrim()).MakeMatrixXform().Set(local_m)
        layer.Save()
    return str(out)


def rig_info(user_usd: str, data_root: str = "data") -> dict | None:
    """Read the authored rig back for UI rehydration: {pivot, frames, fps, start_deg}.
    The pivot lives in the rig itself (the /TurntableRig translate), so a page reload
    or project load can restore the pivot gizmo + tools without re-picking. None if no
    edits layer exists."""
    from pxr import Sdf, Usd, UsdGeom
    out = Path(edit_layer_path(user_usd, data_root))
    if not out.exists():
        return None
    with USD_LOCK:
        layer = Sdf.Layer.FindOrOpen(str(out))
        if layer is None:
            return None
        stage = Usd.Stage.Open(layer)
        rig = stage.GetPrimAtPath(PIVOT_PATH)
        if not rig or not rig.IsValid():
            return None
        pivot = None
        start_deg = 0.0
        frames = 0
        for op in UsdGeom.Xformable(rig).GetOrderedXformOps():
            name = op.GetOpName()
            if name == "xformOp:translate":
                pivot = [float(v) for v in op.Get()]
            elif ":spin" in name and op.GetNumTimeSamples() > 1:
                times = op.GetTimeSamples()
                start_deg = float(op.Get(Usd.TimeCode(times[0])))
                frames = int(round(times[-1]))
        if pivot is None:
            return None
        return {"pivot": pivot, "frames": frames,
                "fps": float(layer.timeCodesPerSecond) if layer.HasTimeCodesPerSecond() else 0.0,
                "start_deg": start_deg}


def remove_turntable(user_usd: str, data_root: str = "data") -> bool:
    """Delete the turntable edits layer (clear the registry-held layer object too, so a
    later re-author starts clean). Returns True if anything was removed."""
    from pxr import Sdf
    out = Path(edit_layer_path(user_usd, data_root))
    with USD_LOCK:
        lay = Sdf.Layer.Find(str(out))
        if lay is not None:
            lay.Clear()
        if out.exists():
            out.unlink()
            return True
    return lay is not None

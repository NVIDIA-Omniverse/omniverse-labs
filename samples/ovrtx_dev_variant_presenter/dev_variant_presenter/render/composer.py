# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Author a composite USDA that sublayers the user scene and adds the viewer pipeline.

Never mutates the user USD. Sublayers by absolute posix path, which keeps the
composite location-independent. Authors a viewer Camera (seeded from a target authored camera,
aperture matched to the render resolution), a RenderProduct (camera rel + LdrColor
RenderVar + resolution + omni:rtx render-mode attrs), and the variant selections.
Uses multi-line USDA prim bodies (pxr authoring), not one-line bodies.
"""
from __future__ import annotations

from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom

from dev_variant_presenter.models import QualitySpec, Selection
from dev_variant_presenter.render.modes import rendermode_attrs

_USD_TYPE = {
    "token": Sdf.ValueTypeNames.Token,
    "int": Sdf.ValueTypeNames.Int,
    "float": Sdf.ValueTypeNames.Float,
    "bool": Sdf.ValueTypeNames.Bool,
}


LAST_APERTURE: float = 20.955   # horizontal aperture used by the most recent composite
                                # (read by the gizmo projection; single render thread writes it)


def hoist_time_metadata(layer: Sdf.Layer) -> None:
    """Copy the strongest sublayer's time metadata onto `layer` (the composition root).
    start/endTimeCode resolve from the ROOT layer only (USD rule), and a sublayer whose
    timeCodesPerSecond differs from the root's (default 24) gets an implicit layer-offset
    SCALE — a 60tps turntable rig composed under a bare root plays 2.5x fast, finishes by
    frame 48, then holds. Root time must equal content time. Shared by the scanner and
    the composite builder so stage_info and the renderer agree on the timebase."""
    for lp in list(layer.subLayerPaths):
        sub = Sdf.Layer.FindOrOpen(lp)
        if sub is not None and sub.HasStartTimeCode():
            layer.startTimeCode = sub.startTimeCode
            layer.endTimeCode = sub.endTimeCode
            if sub.HasTimeCodesPerSecond():
                layer.timeCodesPerSecond = sub.timeCodesPerSecond
            if sub.HasFramesPerSecond():
                layer.framesPerSecond = sub.framesPerSecond
            return


def build_composite(
    user_usd: str,
    selections: Selection,
    *,
    camera_path: str,
    render_product_path: str,
    quality: QualitySpec,
    out_path: str,
    viewer_camera_path: str = "/Viewer/Camera",
    camera: dict | None = None,
    extra_sublayers: tuple[str, ...] = (),
    use_stage_camera: bool = False,
) -> str:
    """Write the composite USDA to out_path and return out_path. `camera` optionally
    overrides UsdGeomCamera attrs on the viewer camera: focal_length (mm, FOV),
    f_stop + focus_distance (depth of field), exposure (stops)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    layer = Sdf.Layer.CreateNew(str(out))
    # absolute(), NOT resolve(): resolve() follows junctions, and the remote-stage mirror
    # depends on a short junction path to stay under Windows MAX_PATH — expanding it back
    # to the real (300-char) path silently breaks pxr/MDL file access for every layer anchor
    for extra in extra_sublayers:      # e.g. the turntable edits layer — STRONGER than the
        if extra:                      # user stage so its time-range opinions win
            layer.subLayerPaths.append(Path(extra).absolute().as_posix())
    layer.subLayerPaths.append(Path(user_usd).absolute().as_posix())
    hoist_time_metadata(layer)   # root tps must match the content or USD time-scales it
    stage = Usd.Stage.Open(layer)

    # 1) variant selections (grouped by prim)
    by_prim: dict[str, dict[str, str]] = {}
    for c in selections:
        by_prim.setdefault(c.prim_path, {})[c.set_name] = c.variant
    for prim_path, sets in by_prim.items():
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            prim = stage.OverridePrim(prim_path)
        for sn, v in sets.items():
            prim.GetVariantSets().GetVariantSet(sn).SetVariantSelection(v)

    # 2) viewer camera, seeded from the target authored camera; aperture matches resolution
    w, h = quality.resolution
    src_prim = stage.GetPrimAtPath(camera_path) if camera_path else None
    src_cam = UsdGeom.Camera(src_prim) if src_prim and src_prim.IsValid() else None
    h_ap = (src_cam.GetHorizontalApertureAttr().Get() if src_cam else None) or 20.955
    global LAST_APERTURE
    LAST_APERTURE = float(h_ap)
    # use_stage_camera: the RenderProduct shoots through the AUTHORED camera prim itself —
    # required for ANIMATED cameras (turntable rigs): the viewer copy below is a static
    # default-time snapshot and would freeze the camera for the whole animation.
    if use_stage_camera and src_cam is not None:
        over = stage.OverridePrim(camera_path)
        over.CreateAttribute("horizontalAperture", Sdf.ValueTypeNames.Float).Set(float(h_ap))
        over.CreateAttribute("verticalAperture", Sdf.ValueTypeNames.Float).Set(float(h_ap) * h / w)
        if camera:
            if camera.get("focal_length"):
                over.CreateAttribute("focalLength", Sdf.ValueTypeNames.Float).Set(float(camera["focal_length"]))
            if camera.get("f_stop"):
                over.CreateAttribute("fStop", Sdf.ValueTypeNames.Float).Set(float(camera["f_stop"]))
            if camera.get("focus_distance"):
                over.CreateAttribute("focusDistance", Sdf.ValueTypeNames.Float).Set(float(camera["focus_distance"]))
            if camera.get("iso"):
                over.CreateAttribute("exposure:iso", Sdf.ValueTypeNames.Float).Set(float(camera["iso"]))
        rp = stage.DefinePrim(render_product_path, "RenderProduct")
        rp.CreateRelationship("camera").SetTargets([Sdf.Path(camera_path)])
        rv_path = render_product_path + "/LdrColor"
        rp.CreateRelationship("orderedVars").SetTargets([Sdf.Path(rv_path)])
        rp.CreateAttribute("resolution", Sdf.ValueTypeNames.Int2).Set(Gf.Vec2i(w, h))
        for name, (usd_type, value) in rendermode_attrs(quality).items():
            rp.CreateAttribute(name, _USD_TYPE[usd_type]).Set(value)
        rv = stage.DefinePrim(rv_path, "RenderVar")
        rv.CreateAttribute("sourceName", Sdf.ValueTypeNames.String).Set("LdrColor")
        stage.GetRootLayer().Save()
        return str(out)

    vcam = UsdGeom.Camera.Define(stage, viewer_camera_path)
    vcam.CreateHorizontalApertureAttr(float(h_ap))
    vcam.CreateVerticalApertureAttr(float(h_ap) * h / w)
    if src_cam is not None:
        m = UsdGeom.Xformable(src_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        UsdGeom.Xformable(vcam.GetPrim()).MakeMatrixXform().Set(m)
    if camera:   # optional display overrides (FOV / depth of field / exposure)
        if camera.get("focal_length"):
            vcam.CreateFocalLengthAttr(float(camera["focal_length"]))
        if camera.get("f_stop"):
            vcam.CreateFStopAttr(float(camera["f_stop"]))
        if camera.get("focus_distance"):
            vcam.CreateFocusDistanceAttr(float(camera["focus_distance"]))
        if camera.get("exposure") is not None:
            vcam.CreateExposureAttr(float(camera["exposure"]))
        if camera.get("iso"):
            vcam.GetPrim().CreateAttribute("exposure:iso", Sdf.ValueTypeNames.Float).Set(float(camera["iso"]))

    # 3) RenderProduct + LdrColor RenderVar + render-mode attrs
    rp = stage.DefinePrim(render_product_path, "RenderProduct")
    rp.CreateRelationship("camera").SetTargets([Sdf.Path(viewer_camera_path)])
    rv_path = render_product_path + "/LdrColor"
    rp.CreateRelationship("orderedVars").SetTargets([Sdf.Path(rv_path)])
    rp.CreateAttribute("resolution", Sdf.ValueTypeNames.Int2).Set(Gf.Vec2i(w, h))
    for name, (usd_type, value) in rendermode_attrs(quality).items():
        rp.CreateAttribute(name, _USD_TYPE[usd_type]).Set(value)
    rv = stage.DefinePrim(rv_path, "RenderVar")
    rv.CreateAttribute("sourceName", Sdf.ValueTypeNames.String).Set("LdrColor")

    stage.GetRootLayer().Save()
    return str(out)

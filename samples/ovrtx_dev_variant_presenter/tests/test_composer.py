# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest
from pxr import Sdf, Usd

from dev_variant_presenter.models import QualitySpec, VariantChoice
from dev_variant_presenter.render.composer import build_composite

SCENE = str(Path(__file__).resolve().parents[1] / "data" / "ConceptCar" / "product_configurator_base.usd")
pytestmark = pytest.mark.skipif(
    not Path(SCENE).exists(), reason="local ConceptCar mirror not present (data/ ships outside the repo)")
LOOKS = "/World/ConfigurableAssets/ConceptCar/Looks"
MAIN_CAM = "/World/Cameras/Cameras/Cameras_ALL/Main_Cam_01"


def test_composite_has_pipeline_and_selection(tmp_path):
    sel = (VariantChoice(LOOKS, "Carpaint", "Sakura"),)
    out = str(tmp_path / "comp.usda")
    p = build_composite(SCENE, sel, camera_path=MAIN_CAM,
                        render_product_path="/Render/Viewport",
                        quality=QualitySpec(resolution=(800, 450)), out_path=out)
    stage = Usd.Stage.Open(p)

    rp = stage.GetPrimAtPath("/Render/Viewport")
    assert rp and rp.GetTypeName() == "RenderProduct"
    # RP points at the viewer-owned camera (authored cameras stay pristine)
    assert rp.GetRelationship("camera").GetTargets() == [Sdf.Path("/Viewer/Camera")]
    assert rp.GetAttribute("omni:rtx:rendermode").Get() == "RealTimePathTracing"

    rv = stage.GetPrimAtPath("/Render/Viewport/LdrColor")
    assert rv.GetAttribute("sourceName").Get() == "LdrColor"

    looks = stage.GetPrimAtPath(LOOKS)
    assert looks.GetVariantSets().GetVariantSet("Carpaint").GetVariantSelection() == "Sakura"

    cam = stage.GetPrimAtPath("/Viewer/Camera")
    assert cam and cam.GetTypeName() == "Camera"
    h_ap = cam.GetAttribute("horizontalAperture").Get()
    v_ap = cam.GetAttribute("verticalAperture").Get()
    assert abs(v_ap - h_ap * 450 / 800) < 1e-3   # aperture tracks the 800x450 resolution


def test_user_usd_untouched(tmp_path):
    import os
    before = os.path.getmtime(SCENE)
    build_composite(SCENE, (), camera_path=MAIN_CAM, render_product_path="/Render/Viewport",
                    quality=QualitySpec(), out_path=str(tmp_path / "c.usda"))
    assert os.path.getmtime(SCENE) == before  # never modified

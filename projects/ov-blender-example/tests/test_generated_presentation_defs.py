# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generated presentation layers must define the sensor-path ancestor chain.

A ``def RenderProduct`` nested under ``over`` ancestors composes as an
undefined-ancestor prim that default USD stage traversal never reaches; the
OVRTX worker then rejects the sensor set with
``Render product prim not found`` (observed against the real worker on the
live-authored route, 2026-07-07). Live-authored scenes only define /World,
so the generated layer must define /Render and friends itself."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import ovrtx_scene_composition  # noqa: E402

SENSOR = "/Render/OmniverseKit/HydraTextures/ViewportTexture0"


def _compose(tmp_path: Path, *, generated: bool) -> ovrtx_scene_composition.OvrtxSceneComposition:
    source = tmp_path / "scene.usda"
    source.write_text('#usda 1.0\n\ndef Xform "World"\n{\n}\n', encoding="utf-8")
    import os

    os.environ["OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR"] = str(tmp_path / "work")
    try:
        return ovrtx_scene_composition.compose(
            source_scene_path=str(source),
            camera_prim_path="/World/OVRTXCamera",
            sensor_paths=(SENSOR,),
            width=320,
            height=240,
            camera_projection=None,
            material_scene_layer=None,
            generate_scene_presentation=generated,
        )
    finally:
        os.environ.pop("OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR", None)


def _presentation_layer_text(
    composition: ovrtx_scene_composition.OvrtxSceneComposition,
) -> str:
    paths = [
        str(record["path"])
        for record in composition.presentation_layers
        if str(record.get("source", "")) == "viewport_camera_projection"
    ]
    assert len(paths) == 1
    return Path(paths[0]).read_text(encoding="utf-8")


def test_generated_presentation_defines_the_sensor_ancestor_chain(tmp_path: Path) -> None:
    text = _presentation_layer_text(_compose(tmp_path, generated=True))
    assert 'def "Render"' in text
    assert 'def "OmniverseKit"' in text
    assert 'def "HydraTextures"' in text
    assert 'def RenderProduct "ViewportTexture0"' in text
    assert 'over "Render"' not in text
    # Ancestor defs are typeless: no type opinion that could override a
    # source layer's typing.
    assert "def Scope" not in text
    # The camera's ancestor (/World) is also defined — typeless, so the
    # source layer's Xform typing stays authoritative.
    assert 'def "World"' in text
    assert 'def Camera "OVRTXCamera"' in text


def test_generated_presentation_defines_camera_ancestors_outside_world(tmp_path: Path) -> None:
    """A scene-setting camera path outside /World (e.g. the stale
    ``/Camera/Camera`` from an old fixture workflow) must still compose as
    a defined camera — an undefined ``/Camera`` ancestor made every
    viewport camera value write fail with worker INTERNAL ("path or
    attribute not found in stage", observed 2026-07-07)."""

    import os

    source = tmp_path / "scene.usda"
    source.write_text('#usda 1.0\n\ndef Xform "World"\n{\n}\n', encoding="utf-8")
    os.environ["OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR"] = str(tmp_path / "work")
    try:
        composition = ovrtx_scene_composition.compose(
            source_scene_path=str(source),
            camera_prim_path="/Camera/Camera",
            sensor_paths=(SENSOR,),
            width=320,
            height=240,
            camera_projection=None,
            material_scene_layer=None,
            generate_scene_presentation=True,
        )
    finally:
        os.environ.pop("OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR", None)
    text = _presentation_layer_text(composition)
    assert 'def "Camera"' in text
    assert 'def Camera "Camera"' in text
    assert 'over "Camera"' not in text


def test_override_presentation_keeps_existing_prims_as_overs(tmp_path: Path) -> None:
    """The fixture-stage variant overrides the existing camera prim, while the
    render product it (re)defines against the declared sensor identity gets
    typeless-def ancestors — traversable, but with no type opinion that could
    redefine typing a source layer authors (regression guard)."""

    text = _presentation_layer_text(_compose(tmp_path, generated=False))
    # The camera chain stays pure overs: the source stage owns those prims.
    assert 'over "World"' in text
    assert 'over "OVRTXCamera"' in text
    assert 'def "World"' not in text
    # The render product is defined so it composes even when the source
    # stage lacks /Render; its ancestors are typeless defs, never typed.
    assert 'def RenderProduct "ViewportTexture0"' in text
    assert 'def "Render"' in text
    assert "def Scope" not in text
    assert "def Xform" not in text


def test_composed_sublayer_references_are_drive_qualified(tmp_path: Path, monkeypatch) -> None:
    """The composed root layer must reference its presentation layers with
    fully resolved absolute paths. On Windows the POSIX-style default work
    dir expanded to a drive-relative ``\\tmp\\...`` reference that USD
    silently dropped from a ``file:///`` context — losing the render
    product (observed against the real worker, 2026-07-07)."""

    import os
    import re

    source = tmp_path / "scene.usda"
    source.write_text('#usda 1.0\n\ndef Xform "World"\n{\n}\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    # A relative work dir must come out fully resolved (drive-qualified on
    # Windows), exactly like the drive-relative default expansion.
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR", "work-relative")
    composition = ovrtx_scene_composition.compose(
        source_scene_path=str(source),
        camera_prim_path="/World/OVRTXCamera",
        sensor_paths=(SENSOR,),
        width=320,
        height=240,
        camera_projection=None,
        material_scene_layer=None,
        generate_scene_presentation=True,
    )
    text = Path(composition.composed_scene_path).read_text(encoding="utf-8")
    references = [
        match.replace("\\\\", "\\")
        for match in re.findall(r"@([^@]+)@", text)
    ]
    assert references
    for reference in references:
        assert os.path.isabs(reference), reference
        if os.name == "nt":
            assert os.path.splitdrive(reference)[0], (
                f"sublayer reference is drive-relative: {reference}"
            )


def test_generated_presentation_composes_a_defined_traversable_product(tmp_path: Path) -> None:
    try:
        from pxr import Usd  # type: ignore
    except Exception:
        import pytest

        pytest.skip("pxr (OpenUSD) is not available in this environment")
    composition = _compose(tmp_path, generated=True)
    stage = Usd.Stage.Open(composition.composed_scene_path)
    assert stage
    prim = stage.GetPrimAtPath(SENSOR)
    assert prim and prim.IsDefined()
    traversed = {p.GetPath().pathString for p in stage.Traverse()}
    assert SENSOR in traversed

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RTPT quality attribute authoring on the generated RenderProduct.

Spec render-quality-color-controls, task01-03: the four documented RTPT quality
attributes are authored onto the generated ``RenderProduct`` definition on every
session composition -- viewport and F12 alike -- with values sourced from the
scene property group via ``RTPT_RENDER_SETTINGS`` (task01-01), so the artist's
values are deterministic session state that survives session replacement.

The composition assertions run in the plain (non-Blender) pytest lane against
``ovrtx_session.build_spec`` (the shared viewport/F12 entry point) and the
``RenderRequestTranslator`` settings-snapshot plumbing.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import ovrtx_scene_composition, ovrtx_session  # noqa: E402
from ovrtx_blender_example.blender_signal_translation import (  # noqa: E402
    RenderRequestTranslator,
)
from ovrtx_blender_example.blender_signals import (  # noqa: E402
    BlenderRenderIntent,
    BlenderRenderSignal,
    BlenderRenderSignalSource,
)
from ovrtx_blender_example.properties import RTPT_RENDER_SETTINGS  # noqa: E402
from ovrtx_blender_example.render_requests import RenderRequest  # noqa: E402


# Documented runtime (wire) defaults, expressed as the exact authored USD lines.
# These are the wire values OVRTX consumes; the UI defaults are 1/3/15/true and
# Max Bounces adds +2, so the authored default line stays maxBounces = 3.
_DEFAULT_LINES = (
    "int omni:rtx:rtpt:maxBounces = 3",
    "int omni:rtx:rtpt:maxSpecularAndTransmissionBounces = 3",
    "int omni:rtx:rtpt:maxVolumeBounces = 15",
    "bool omni:rtx:rtpt:fireflyFilter:enabled = true",
)

# ``rtpt_quality`` carries artist-facing UI values.
_NON_DEFAULT_QUALITY = {
    "rtpt_max_bounces": 7,
    "rtpt_max_specular_and_transmission_bounces": 2,
    "rtpt_max_volume_bounces": 0,
    "rtpt_firefly_filter_enabled": False,
}

# The authored lines carry the wire values: Max Bounces UI 7 -> wire 9 (+2);
# the sub-caps pass through unchanged.
_NON_DEFAULT_LINES = (
    "int omni:rtx:rtpt:maxBounces = 9",
    "int omni:rtx:rtpt:maxSpecularAndTransmissionBounces = 2",
    "int omni:rtx:rtpt:maxVolumeBounces = 0",
    "bool omni:rtx:rtpt:fireflyFilter:enabled = false",
)


def _live_request(tmp_path: Path, **changes: object) -> RenderRequest:
    """A live-authored (``.blend``) request: the generated-presentation route."""

    request = RenderRequest(
        input_usd_path=str(tmp_path / "scene.usda"),
        current_scene_generation=True,
        sensor_paths=("/Render/OmniverseKit/HydraTextures/ViewportTexture0",),
        selected_sensor_paths=(
            "/Render/OmniverseKit/HydraTextures/ViewportTexture0",
        ),
        width=320,
        height=180,
        camera_prim_path="/World/OVRTXCamera",
    )
    return replace(request, **changes)


def _presentation_text(
    composition: ovrtx_scene_composition.OvrtxSceneComposition,
) -> str:
    record = next(
        item
        for item in composition.presentation_layers
        if item["source"] == "viewport_camera_projection"
    )
    return Path(str(record["path"])).read_text(encoding="utf-8")


def _render_product_block(text: str) -> str:
    marker = "def RenderProduct"
    assert marker in text, text
    return text[text.index(marker):]


# --- Settings-snapshot plumbing (task01-01 mapping -> RenderRequest) --------


def _render_scene(**rtpt: object) -> SimpleNamespace:
    return SimpleNamespace(
        render=SimpleNamespace(
            resolution_x=640,
            resolution_y=360,
            resolution_percentage=50,
        ),
        ovrtx_example=SimpleNamespace(
            render_product_path="/Render/Test/Product",
            min_samples=1,
            max_samples=128,
            camera_prim_path="/World/Camera",
            sync_viewport_camera=False,
            simulation_reset_token=0,
            **rtpt,
        ),
        frame_current=1,
        frame_start=1,
        frame_end=1,
    )


def test_translator_carries_scene_rtpt_values_into_the_request() -> None:
    scene = _render_scene(**_NON_DEFAULT_QUALITY)

    request = RenderRequestTranslator().translate(
        BlenderRenderSignal(
            BlenderRenderSignalSource.VIEW_UPDATE,
            BlenderRenderIntent.VIEWPORT,
            scene,
            "",
            camera_prim_path="/Generated/Camera",
            render_product_path="/Generated/Product",
            context=SimpleNamespace(),
            current_scene_generation=True,
        )
    )

    assert request.rtpt_quality == _NON_DEFAULT_QUALITY


@pytest.mark.parametrize(
    "intent, expected_route",
    [
        (BlenderRenderIntent.VIEWPORT, True),
        (BlenderRenderIntent.FINAL_RENDER, False),
    ],
)
def test_translator_sets_rtpt_value_route_for_live_viewport_only(
    intent: BlenderRenderIntent,
    expected_route: bool,
) -> None:
    # task01-04: the live viewport routes RTPT quality changes as runtime
    # attribute writes (excluded from the composition digest so a change does
    # not replace the session); F12 keeps them in the digest. Pins the exact
    # acceptance boundary against a flipped condition.
    scene = _render_scene(**_NON_DEFAULT_QUALITY)

    request = RenderRequestTranslator().translate(
        BlenderRenderSignal(
            BlenderRenderSignalSource.VIEW_UPDATE,
            intent,
            scene,
            "",
            camera_prim_path="/Generated/Camera",
            render_product_path="/Generated/Product",
            context=SimpleNamespace(),
            current_scene_generation=True,
        )
    )

    assert request.rtpt_value_route is expected_route


def test_translator_uses_documented_defaults_when_settings_absent() -> None:
    # A scene whose ``ovrtx_example`` group is missing the RTPT attributes
    # (older saved file / partial stub) falls back to the documented defaults.
    scene = _render_scene()

    request = RenderRequestTranslator().translate(
        BlenderRenderSignal(
            BlenderRenderSignalSource.VIEW_UPDATE,
            BlenderRenderIntent.VIEWPORT,
            scene,
            "",
            camera_prim_path="/Generated/Camera",
            render_product_path="/Generated/Product",
            context=SimpleNamespace(),
            current_scene_generation=True,
        )
    )

    assert request.rtpt_quality == {
        name: spec.default for name, spec in RTPT_RENDER_SETTINGS.items()
    }


# --- Composition authoring (shared viewport/F12 build_spec path) ------------


def test_non_default_values_author_exact_names_types_and_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR", str(tmp_path / "work"))
    spec = ovrtx_session.build_spec(
        _live_request(tmp_path, rtpt_quality=_NON_DEFAULT_QUALITY)
    )
    block = _render_product_block(_presentation_text(spec.ovrtx_scene_composition))

    for line in _NON_DEFAULT_LINES:
        assert line in block, block
    # Authored on the RenderProduct prim, not a camera / RenderVar / separate
    # RenderSettings prim.
    assert "def RenderSettings" not in block
    for line in _NON_DEFAULT_LINES:
        assert block.index(line) < block.index('def RenderVar "LdrColor"')


def test_default_values_author_the_documented_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR", str(tmp_path / "work"))
    # No rtpt_quality on the request (empty default): the documented runtime
    # defaults are authored, so out-of-the-box output is unchanged.
    spec = ovrtx_session.build_spec(_live_request(tmp_path))
    block = _render_product_block(_presentation_text(spec.ovrtx_scene_composition))

    for line in _DEFAULT_LINES:
        assert line in block, block


def test_viewport_and_final_render_both_author_the_attributes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR", str(tmp_path / "work"))
    # Viewport request: runtime camera pose override.
    viewport = ovrtx_session.build_spec(
        _live_request(
            tmp_path,
            rtpt_quality=_NON_DEFAULT_QUALITY,
            camera_matrix=(
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
        )
    )
    # F12 request: composed scene-camera pose (no runtime override).
    final_render = ovrtx_session.build_spec(
        _live_request(
            tmp_path,
            rtpt_quality=_NON_DEFAULT_QUALITY,
            scene_camera_matrix=(
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, -1.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
        )
    )

    assert viewport.camera_pose_source == ovrtx_session.RUNTIME_UPDATE
    assert final_render.camera_pose_source == ovrtx_session.COMPOSED_SCENE
    for spec in (viewport, final_render):
        block = _render_product_block(
            _presentation_text(spec.ovrtx_scene_composition)
        )
        for line in _NON_DEFAULT_LINES:
            assert line in block, block


def test_rekeyed_composition_carries_current_values_and_new_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR", str(tmp_path / "work"))
    first = ovrtx_session.build_spec(_live_request(tmp_path))
    second = ovrtx_session.build_spec(
        _live_request(tmp_path, rtpt_quality=_NON_DEFAULT_QUALITY)
    )

    first_block = _render_product_block(
        _presentation_text(first.ovrtx_scene_composition)
    )
    second_block = _render_product_block(
        _presentation_text(second.ovrtx_scene_composition)
    )
    for line in _DEFAULT_LINES:
        assert line in first_block
    for line in _NON_DEFAULT_LINES:
        assert line in second_block

    # A changed quality value folds into the composition digest so
    # reuse_decision replaces the session that composes the new opinions.
    assert (
        first.ovrtx_scene_composition.digest
        != second.ovrtx_scene_composition.digest
    )
    assert (
        ovrtx_session.reuse_decision(first, second).reason
        == "scene_composition_changed"
    )


def test_live_value_route_keeps_session_identity_across_quality_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # task01-04: on the live viewport route (``rtpt_value_route=True``) a
    # quality change is applied as a runtime attribute write on the render
    # thread, so it must NOT change session identity — the RTPT values are
    # excluded from the composition digest and ``reuse_decision`` keeps the
    # running session. The layer body still authors the current values so any
    # (re)placed session composes them fresh.
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR", str(tmp_path / "work"))
    first = ovrtx_session.build_spec(_live_request(tmp_path, rtpt_value_route=True))
    second = ovrtx_session.build_spec(
        _live_request(
            tmp_path,
            rtpt_value_route=True,
            rtpt_quality=_NON_DEFAULT_QUALITY,
        )
    )

    assert (
        first.ovrtx_scene_composition.digest
        == second.ovrtx_scene_composition.digest
    )
    decision = ovrtx_session.reuse_decision(first, second)
    assert decision.reuse is True
    assert decision.reason == "same_session"
    # The body still carries the changed values for a fresh composition.
    second_block = _render_product_block(
        _presentation_text(second.ovrtx_scene_composition)
    )
    for line in _NON_DEFAULT_LINES:
        assert line in second_block, second_block


def test_direct_usd_override_route_does_not_author_rtpt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The fixture/direct-USD validation route does
    # not generate a RenderProduct at all; it must not emit RTPT lines, so the
    # digest and body stay consistent with "generated route only" authoring.
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR", str(tmp_path / "work"))
    source = tmp_path / "scene.usda"
    source.write_text('#usda 1.0\n\ndef Xform "World"\n{\n}\n', encoding="utf-8")
    composition = ovrtx_scene_composition.compose(
        source_scene_path=str(source),
        camera_prim_path="/World/Camera",
        sensor_paths=("/Render/OmniverseKit/HydraTextures/ViewportTexture0",),
        width=320,
        height=180,
        camera_projection=None,
        material_scene_layer=None,
        generate_scene_presentation=False,
        rtpt_quality=_NON_DEFAULT_QUALITY,
    )
    text = _presentation_text(composition)
    assert "omni:rtx:rtpt:" not in text


def test_artifact_diagnostics_roundtrip_authored_quality(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The artifact diagnostics recompute contributions from the request; the
    # authored RTPT values must round-trip through the digest validation.
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR", str(tmp_path / "work"))
    request = _live_request(tmp_path, rtpt_quality=_NON_DEFAULT_QUALITY)
    spec = ovrtx_session.build_spec(request)

    evidence = ovrtx_scene_composition.diagnostics(
        spec.ovrtx_scene_composition,
        request=request,
    )

    assert evidence["enabled"] is True
    assert evidence["conflict_count"] == 0

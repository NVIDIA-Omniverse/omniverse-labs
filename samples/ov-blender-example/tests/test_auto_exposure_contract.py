# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scene-linear ownership contract, step 2: renderer auto-exposure is disabled.

Spec render-quality-color-controls, task02-04. The scene-linear ownership
contract (validated by the fresh-open work in #65) is:

1. Request OVRTX ``HdrColor`` as scene-linear RGBA16F.
2. **Disable renderer auto-exposure for that path.**
3. Insert linear pixels into Blender's render result / viewport presentation.
4. Apply Blender View Transform, Look, Exposure, and Gamma exactly once.

Step 2 is satisfied at authoring time: the generated ``RenderProduct``
definition (``ovrtx_scene_composition.py``) unconditionally authors
``bool omni:rtx:autoExposure:enabled = false``. If that opinion ever flips to
``true`` (or is dropped), the renderer would auto-brighten/darken the
scene-linear ``HdrColor`` pass before Blender's display transform runs, so the
View Transform / Look / Exposure / Gamma would no longer be the single owner of
tone — the Junk Shop-style presentation regression this contract guards
against. These assertions pin the opinion so a future composition edit cannot
silently regress contract step 2.

The opinion is authored unconditionally and is mode-independent (it stays
disabled in both the LDR-passthrough and scene-linear presentation modes), so
the assertions here are mode-independent. They run in the plain (non-Blender)
pytest lane against ``ovrtx_session.build_spec`` — the shared viewport/F12 entry
point — covering both the runtime-camera viewport route and the composed-scene
final-render (F12) route.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import (  # noqa: E402
    color_presentation,
    ovrtx_scene_composition,
    ovrtx_session,
)
from ovrtx_blender_example.render_requests import RenderRequest  # noqa: E402


# The exact authored USD line contract step 2 requires. Pinned as concrete text
# (not derived) so a name/type/value drift fails loudly.
_AUTO_EXPOSURE_DISABLED_LINE = "bool omni:rtx:autoExposure:enabled = false"

_IDENTITY_MATRIX = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
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


def test_viewport_composition_disables_auto_exposure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Viewport request: a runtime camera pose override (RUNTIME_UPDATE). The
    # generated render product must author auto-exposure disabled so contract
    # step 2 holds on the interactive path.
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR", str(tmp_path / "work"))
    spec = ovrtx_session.build_spec(
        _live_request(tmp_path, camera_matrix=_IDENTITY_MATRIX)
    )

    assert spec.camera_pose_source == ovrtx_session.RUNTIME_UPDATE
    block = _render_product_block(_presentation_text(spec.ovrtx_scene_composition))
    assert _AUTO_EXPOSURE_DISABLED_LINE in block, block


def test_final_render_composition_disables_auto_exposure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # F12 request: the composed scene-camera pose (COMPOSED_SCENE, no runtime
    # override). The generated render product must author auto-exposure
    # disabled so contract step 2 holds on the final-render path too.
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR", str(tmp_path / "work"))
    spec = ovrtx_session.build_spec(
        _live_request(tmp_path, scene_camera_matrix=_IDENTITY_MATRIX)
    )

    assert spec.camera_pose_source == ovrtx_session.COMPOSED_SCENE
    block = _render_product_block(_presentation_text(spec.ovrtx_scene_composition))
    assert _AUTO_EXPOSURE_DISABLED_LINE in block, block


def test_auto_exposure_stays_disabled_in_both_presentation_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The opinion is authored unconditionally, so the presentation mode must
    # not gate it: both the default LDR-passthrough mode and an explicit
    # scene-linear selection keep auto-exposure disabled. Pins mode-independence
    # so a future mode-conditional edit cannot flip it on for one mode.
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR", str(tmp_path / "work"))
    for mode in (
        color_presentation.MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH,
        color_presentation.MODE_SCENE_LINEAR_HDR,
    ):
        spec = ovrtx_session.build_spec(
            _live_request(
                tmp_path,
                camera_matrix=_IDENTITY_MATRIX,
                color_presentation=color_presentation.presentation_from_scene(
                    None,
                    requested_mode=mode,
                ),
            )
        )
        block = _render_product_block(
            _presentation_text(spec.ovrtx_scene_composition)
        )
        assert _AUTO_EXPOSURE_DISABLED_LINE in block, (mode, block)

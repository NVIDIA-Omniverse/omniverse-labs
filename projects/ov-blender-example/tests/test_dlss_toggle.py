# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DLSS Super-Resolution toggle (user-requested addition).

A real-GPU A/B on this RealTimePathTracing worker build proved:
  * DLSS-SR is ON by default and NO exposed setting fully disables it; the only
    honored DLSS knob is ``/rtx/post/dlss/execMode`` (a quality/perf preset).
  * That knob is honored on the launch-only ``ovrtx.config.json`` carb channel
    AND — unlike the ignored ``omni:rtx:rtpt:*`` family — as
    ``omni:rtx:post:dlss:execMode`` on the generated ``RenderProduct`` at SESSION
    creation. So folding the toggle into the composition digest re-keys the
    session and applies the change with NO worker restart.

Semantics: ``dlss_enabled=True`` (default) leaves the engine default; ``False``
authors the Performance-preset execMode value on both channels. These plain
(non-Blender) tests exercise the config authoring, the composition/digest
participation, the change-triggers-apply re-key, and the source-level panel row.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from blender_test_support import blender_executable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import (  # noqa: E402
    ovrtx_scene_composition,
    ovrtx_session,
    rtpt_worker_config,
)
from ovrtx_blender_example.blender_signal_translation import (  # noqa: E402
    RenderRequestTranslator,
)
from ovrtx_blender_example.blender_signals import (  # noqa: E402
    BlenderRenderIntent,
    BlenderRenderSignal,
    BlenderRenderSignalSource,
)
from ovrtx_blender_example.properties import (  # noqa: E402
    DLSS_DISABLED_EXECMODE,
    DLSS_EXECMODE_ATTRIBUTE,
)
from ovrtx_blender_example.render_requests import RenderRequest  # noqa: E402


_DLSS_OFF_LINE = f"int {DLSS_EXECMODE_ATTRIBUTE} = {int(DLSS_DISABLED_EXECMODE)}"
_DLSS_CARB_PATH = ("rtx", "post", "dlss", "execMode")


def _live_request(tmp_path: Path, **changes: object) -> RenderRequest:
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


def _nested(tree: dict, path: tuple[str, ...]):
    cursor = tree
    for key in path:
        assert isinstance(cursor, dict) and key in cursor, (path, tree)
        cursor = cursor[key]
    return cursor


# --- Request plumbing -------------------------------------------------------


def _render_scene(**ovrtx: object) -> SimpleNamespace:
    return SimpleNamespace(
        render=SimpleNamespace(
            resolution_x=640, resolution_y=360, resolution_percentage=50
        ),
        ovrtx_example=SimpleNamespace(
            render_product_path="/Render/Test/Product",
            min_samples=1,
            max_samples=128,
            camera_prim_path="/World/Camera",
            sync_viewport_camera=False,
            simulation_reset_token=0,
            **ovrtx,
        ),
        frame_current=1,
        frame_start=1,
        frame_end=1,
    )


def test_request_default_dlss_enabled_true() -> None:
    assert RenderRequest(input_usd_path="x").dlss_enabled is True


def test_translator_carries_dlss_toggle_into_request() -> None:
    for value in (True, False):
        scene = _render_scene(dlss_enabled=value)
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
        assert request.dlss_enabled is value


def test_translator_defaults_dlss_on_when_setting_absent() -> None:
    scene = _render_scene()  # older saved file / partial stub
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
    assert request.dlss_enabled is True


# --- Worker-config (carb) authoring -----------------------------------------


def test_dlss_carb_overrides_enabled_is_empty() -> None:
    assert rtpt_worker_config.dlss_carb_overrides(True) == {}


def test_dlss_carb_overrides_disabled_authors_execmode() -> None:
    overrides = rtpt_worker_config.dlss_carb_overrides(False)
    assert _nested(overrides, _DLSS_CARB_PATH) == int(DLSS_DISABLED_EXECMODE)


def test_compose_worker_config_disabled_writes_execmode_preserving_rtpt() -> None:
    text = rtpt_worker_config.compose_worker_config(
        '{"log": {"level": "Info"}}', None, dlss_enabled=False
    )
    tree = json.loads(text)
    assert _nested(tree, _DLSS_CARB_PATH) == int(DLSS_DISABLED_EXECMODE)
    # RTPT defaults still authored, other opinions preserved.
    assert tree["rtx"]["rtpt"]["maxBounces"] == 3
    assert tree["log"]["level"] == "Info"


def test_compose_worker_config_enabled_omits_dlss_key() -> None:
    text = rtpt_worker_config.compose_worker_config("{}", None, dlss_enabled=True)
    tree = json.loads(text)
    assert "dlss" not in tree.get("rtx", {}).get("post", {})


def test_compose_worker_config_enabled_removes_stale_execmode() -> None:
    # Toggling back ON must clear a stale execMode this add-on wrote in a
    # previous OFF state so a freshly launched worker returns to the engine
    # default instead of staying stuck in the Performance preset. Empty
    # containers the add-on created are pruned; unrelated opinions survive.
    off = rtpt_worker_config.compose_worker_config(
        '{"log": {"level": "Info"}}', None, dlss_enabled=False
    )
    assert _nested(json.loads(off), _DLSS_CARB_PATH) == int(DLSS_DISABLED_EXECMODE)
    on = json.loads(
        rtpt_worker_config.compose_worker_config(off, None, dlss_enabled=True)
    )
    assert "dlss" not in on.get("rtx", {}).get("post", {})
    assert "post" not in on.get("rtx", {})
    # RTPT opinions and unrelated keys are preserved across the toggle.
    assert on["rtx"]["rtpt"]["maxBounces"] == 3
    assert on["log"]["level"] == "Info"


def test_compose_worker_config_enabled_preserves_unrelated_post_keys() -> None:
    # Removal must prune ONLY the add-on-owned execMode leaf, never sibling
    # opinions living under the same /rtx/post subtree.
    existing = json.dumps(
        {"rtx": {"post": {"dlss": {"execMode": 0}, "aa": {"op": 3}}}}
    )
    on = json.loads(
        rtpt_worker_config.compose_worker_config(existing, None, dlss_enabled=True)
    )
    assert "dlss" not in on["rtx"]["post"]
    assert on["rtx"]["post"]["aa"]["op"] == 3


def test_author_worker_config_writes_execmode_to_package_config(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pkg"
    root.mkdir()
    worker_command = f"worker --package-root {root}"
    result = rtpt_worker_config.author_worker_config(
        worker_command, None, dlss_enabled=False
    )
    assert result["status"] == "written"
    tree = json.loads((root / "ovrtx.config.json").read_text(encoding="utf-8"))
    assert _nested(tree, _DLSS_CARB_PATH) == int(DLSS_DISABLED_EXECMODE)


# --- Composition / digest participation (RenderProduct channel) -------------


def test_disabled_authors_execmode_on_render_product(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR", str(tmp_path / "w"))
    spec = ovrtx_session.build_spec(_live_request(tmp_path, dlss_enabled=False))
    block = _render_product_block(_presentation_text(spec.ovrtx_scene_composition))
    assert _DLSS_OFF_LINE in block, block
    assert block.index(_DLSS_OFF_LINE) < block.index('def RenderVar "LdrColor"')


def test_enabled_leaves_engine_default_no_execmode_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR", str(tmp_path / "w"))
    spec = ovrtx_session.build_spec(_live_request(tmp_path, dlss_enabled=True))
    text = _presentation_text(spec.ovrtx_scene_composition)
    assert DLSS_EXECMODE_ATTRIBUTE not in text


def test_toggle_change_rekeys_the_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The proven no-restart mechanism: a DLSS toggle change yields a distinct
    # composition digest so reuse_decision replaces the session, whose new
    # composition authors the new execMode opinion honored at session creation.
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR", str(tmp_path / "w"))
    on = ovrtx_session.build_spec(_live_request(tmp_path, dlss_enabled=True))
    off = ovrtx_session.build_spec(_live_request(tmp_path, dlss_enabled=False))
    assert (
        on.ovrtx_scene_composition.digest
        != off.ovrtx_scene_composition.digest
    )
    assert (
        ovrtx_session.reuse_decision(on, off).reason == "scene_composition_changed"
    )


def test_direct_usd_route_does_not_author_dlss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The fixture/direct-USD route generates no RenderProduct, so no DLSS line
    # is emitted regardless of the toggle.
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR", str(tmp_path / "w"))
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
        dlss_enabled=False,
    )
    assert DLSS_EXECMODE_ATTRIBUTE not in _presentation_text(composition)


def test_diagnostics_report_dlss_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR", str(tmp_path / "w"))
    request = _live_request(tmp_path, dlss_enabled=False)
    spec = ovrtx_session.build_spec(request)
    evidence = ovrtx_scene_composition.diagnostics(
        spec.ovrtx_scene_composition, request=request
    )
    layer = next(
        item
        for item in evidence["presentation_layers"]
        if item["source"] == "viewport_camera_projection"
    )
    assert layer["dlss_enabled"] is False
    assert layer["dlss_execmode_authored"] is True
    assert evidence["conflict_count"] == 0


# --- Panel row (source-level) -----------------------------------------------


def test_panel_draws_dlss_toggle_after_quality_controls() -> None:
    ui_source = (ROOT / "addon" / "ovrtx_blender_example" / "ui.py").read_text(
        encoding="utf-8"
    )
    draw = ui_source[ui_source.index("def draw(self, context"):]
    assert 'layout.prop(settings, "dlss_enabled")' in draw
    assert draw.index('rtpt_firefly_filter_enabled') < draw.index(
        'layout.prop(settings, "dlss_enabled")'
    )


# --- Headless Blender: registration / default / persistence -----------------

_DRIVER = """
import json, sys, traceback
result = {"errors": [], "steps": []}
output_path = sys.argv[sys.argv.index("--") + 1]
blend_path = sys.argv[sys.argv.index("--") + 2]
try:
    import bpy
    sys.path.insert(0, __ADDON_PATH__)
    from ovrtx_blender_example import properties
    bpy.ops.wm.read_homefile(use_empty=True)
    properties.register()
    result["steps"].append("registered")
    settings = bpy.context.scene.ovrtx_example
    result["default"] = bool(settings.dlss_enabled)
    settings.dlss_enabled = False
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    bpy.ops.wm.read_homefile(use_empty=True)
    bpy.ops.wm.open_mainfile(filepath=blend_path)
    result["reloaded"] = bool(bpy.context.scene.ovrtx_example.dlss_enabled)
    result["steps"].append("reloaded")
    properties.unregister()
    result["steps"].append("unregistered")
except Exception:
    result["errors"].append(traceback.format_exc())
with open(output_path, "w", encoding="utf-8") as stream:
    json.dump(result, stream)
"""


def test_dlss_property_registers_defaults_true_and_persists(tmp_path: Path) -> None:
    blender = blender_executable()
    if blender is None:
        pytest.skip("no Blender executable available for headless regression")
    driver = tmp_path / "dlss_driver.py"
    driver.write_text(
        _DRIVER.replace("__ADDON_PATH__", repr(str(ROOT / "addon"))),
        encoding="utf-8",
    )
    output = tmp_path / "result.json"
    completed = subprocess.run(
        (
            str(blender), "--background", "--factory-startup",
            "--python-exit-code", "1", "--python", str(driver),
            "--", str(output), str(tmp_path / "dlss.blend"),
        ),
        cwd=ROOT, capture_output=True, text=True, check=False, timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["errors"] == []
    assert result["steps"] == ["registered", "reloaded", "unregistered"]
    assert result["default"] is True          # ships DLSS-on (worker default)
    assert result["reloaded"] is False         # toggle survives .blend round-trip

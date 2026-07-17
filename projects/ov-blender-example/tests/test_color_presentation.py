# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import color_presentation, properties  # noqa: E402


def _scene(mode: str | None = None) -> SimpleNamespace:
    scene = SimpleNamespace(
        view_settings=SimpleNamespace(
            view_transform="AgX",
            look="Medium High Contrast",
            exposure=0.5,
            gamma=1.2,
        ),
        display_settings=SimpleNamespace(display_device="sRGB"),
    )
    if mode is not None:
        scene.ovrtx_example = SimpleNamespace(color_presentation_mode=mode)
    return scene


def test_default_presentation_records_current_ldr_rgba8_contract() -> None:
    diagnostics = color_presentation.presentation_from_scene(_scene(), requested_mode="")

    assert diagnostics["requested_mode"] == color_presentation.MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH
    assert diagnostics["active_mode"] == color_presentation.MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH
    assert diagnostics["status"] == color_presentation.STATUS_CURRENT
    assert diagnostics["frame_format"] == color_presentation.FRAME_FORMAT_RGBA8
    assert diagnostics["frame_color_mode"] == color_presentation.FRAME_COLOR_MODE_DISPLAY_LDR
    assert diagnostics["render_var"] == color_presentation.RENDER_VAR_LDR_COLOR
    assert diagnostics["conversion"] == color_presentation.CONVERSION_PASSTHROUGH
    assert diagnostics["blender_display_transform_applied"] is False
    assert diagnostics["authored_values_adjusted"] is False
    assert diagnostics["view_settings"] == {
        "view_transform": "AgX",
        "look": "Medium High Contrast",
        "exposure": 0.5,
        "gamma": 1.2,
        "display_device": "sRGB",
    }


def test_scene_linear_hdr_mode_records_hdr_render_var_contract() -> None:
    diagnostics = color_presentation.presentation_from_scene(
        _scene(),
        requested_mode=color_presentation.MODE_SCENE_LINEAR_HDR,
    )

    assert diagnostics["requested_mode"] == color_presentation.MODE_SCENE_LINEAR_HDR
    assert diagnostics["active_mode"] == color_presentation.MODE_SCENE_LINEAR_HDR
    assert diagnostics["status"] == color_presentation.STATUS_CURRENT
    assert diagnostics["unavailable_reason"] == ""
    assert diagnostics["frame_format"] == color_presentation.FRAME_FORMAT_RGBA16F
    assert diagnostics["frame_color_mode"] == color_presentation.FRAME_COLOR_MODE_SCENE_LINEAR
    assert diagnostics["render_var"] == color_presentation.RENDER_VAR_HDR_COLOR
    assert diagnostics["conversion"] == color_presentation.CONVERSION_SCENE_LINEAR


def test_ocio_baked_mode_fails_closed_to_current_ldr_path() -> None:
    diagnostics = color_presentation.presentation_from_scene(
        _scene(),
        requested_mode="ocio-baked",
    )

    assert diagnostics["requested_mode"] == color_presentation.MODE_OCIO_BAKED_DISPLAY
    assert diagnostics["active_mode"] == color_presentation.MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH
    assert diagnostics["status"] == color_presentation.STATUS_UNAVAILABLE
    assert diagnostics["unavailable_reason"] == "ocio_baked_display_conversion_unavailable"
    assert diagnostics["requested_conversion"] == color_presentation.CONVERSION_OCIO_BAKED


def test_diagnostics_from_request_result_records_result_frame_metadata() -> None:
    request = SimpleNamespace(color_presentation=color_presentation.presentation_from_scene(_scene()))
    result = SimpleNamespace(frame_format="rgba8", frame_color_mode="display_encoded_ldr")

    diagnostics = color_presentation.diagnostics_from_request_result(request, result)

    assert diagnostics["result_frame_format"] == "rgba8"
    assert diagnostics["result_frame_color_mode"] == "display_encoded_ldr"


def _request(mode: str, **kwargs: object) -> SimpleNamespace:
    return SimpleNamespace(
        color_presentation=color_presentation.presentation_from_scene(
            _scene(), requested_mode=mode, **kwargs
        )
    )


def test_display_transform_evidence_bumps_schema_to_version_2() -> None:
    request = _request(color_presentation.MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH)
    result = SimpleNamespace(
        frame_format="rgba8", frame_color_mode="display_encoded_ldr", linear_rgba16f=b""
    )

    diagnostics = color_presentation.diagnostics_from_request_result(request, result)

    assert diagnostics["schema_version"] == color_presentation.PRESENTATION_SCHEMA_VERSION
    assert color_presentation.PRESENTATION_SCHEMA_VERSION == 2


def test_ldr_frame_evidence_ovrtx_owner_count_one() -> None:
    request = _request(color_presentation.MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH)
    result = SimpleNamespace(
        frame_format="rgba8", frame_color_mode="display_encoded_ldr", linear_rgba16f=b""
    )

    diagnostics = color_presentation.diagnostics_from_request_result(request, result)

    assert diagnostics["render_var"] == color_presentation.RENDER_VAR_LDR_COLOR
    assert diagnostics["display_transform_owner"] == color_presentation.DISPLAY_TRANSFORM_OWNER_OVRTX
    assert diagnostics["display_transform_application_count"] == 1
    assert diagnostics["display_transform_applied_by"] == color_presentation.DISPLAY_TRANSFORM_OWNER_OVRTX
    assert diagnostics["display_transform_consistent"] is True


def test_scene_linear_frame_evidence_consumer_owner_count_one() -> None:
    request = _request(
        color_presentation.MODE_SCENE_LINEAR_HDR, hdr_readback_available=True
    )
    result = SimpleNamespace(
        frame_format="rgba16f", frame_color_mode="scene_linear", linear_rgba16f=b"\x00\x00"
    )

    diagnostics = color_presentation.diagnostics_from_request_result(request, result)

    assert diagnostics["render_var"] == color_presentation.RENDER_VAR_HDR_COLOR
    assert diagnostics["display_transform_owner"] == color_presentation.DISPLAY_TRANSFORM_OWNER_CONSUMER
    assert diagnostics["display_transform_application_count"] == 1
    assert diagnostics["display_transform_applied_by"] == color_presentation.DISPLAY_TRANSFORM_OWNER_CONSUMER
    assert diagnostics["display_transform_consistent"] is True


def test_scene_linear_frame_drawn_raw_reports_count_zero_inconsistent() -> None:
    # Scene-linear mode but the frame carries no linear payload: nothing to
    # draw through the display-space shader, so neither stage applied it.
    request = _request(
        color_presentation.MODE_SCENE_LINEAR_HDR, hdr_readback_available=True
    )
    result = SimpleNamespace(
        frame_format="rgba16f", frame_color_mode="scene_linear", linear_rgba16f=b""
    )

    diagnostics = color_presentation.diagnostics_from_request_result(request, result)

    assert diagnostics["display_transform_application_count"] == 0
    assert diagnostics["display_transform_applied_by"] == color_presentation.DISPLAY_TRANSFORM_APPLIED_BY_NONE
    assert diagnostics["display_transform_consistent"] is False


def test_ldr_frame_through_display_space_shader_reports_count_two_inconsistent() -> None:
    # A display-encoded LDR frame that also carries a linear RGBA16F payload
    # (both stages would apply): count 2, never silently normalized to 1.
    request = _request(color_presentation.MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH)
    result = SimpleNamespace(
        frame_format="rgba16f",
        frame_color_mode="display_encoded_ldr",
        linear_rgba16f=b"\x00\x00",
    )

    diagnostics = color_presentation.diagnostics_from_request_result(request, result)

    assert diagnostics["display_transform_application_count"] == 2
    assert diagnostics["display_transform_applied_by"] == color_presentation.DISPLAY_TRANSFORM_APPLIED_BY_NONE
    assert diagnostics["display_transform_consistent"] is False


def test_ldr_frame_in_scene_linear_mode_surfaces_owner_mismatch() -> None:
    # Declared owner is the consumer (scene-linear mode) but the frame arrived
    # display-encoded LDR (OVRTX applied). Raw count is 1, yet the applying
    # stage disagrees with the declared owner, so it must not pass as
    # consistent.
    request = _request(
        color_presentation.MODE_SCENE_LINEAR_HDR, hdr_readback_available=True
    )
    result = SimpleNamespace(
        frame_format="rgba8", frame_color_mode="display_encoded_ldr", linear_rgba16f=b""
    )

    diagnostics = color_presentation.diagnostics_from_request_result(request, result)

    assert diagnostics["display_transform_owner"] == color_presentation.DISPLAY_TRANSFORM_OWNER_CONSUMER
    assert diagnostics["display_transform_application_count"] == 1
    assert diagnostics["display_transform_applied_by"] == color_presentation.DISPLAY_TRANSFORM_OWNER_OVRTX
    assert diagnostics["display_transform_consistent"] is False


def test_display_transform_evidence_without_result_uses_declared_intent() -> None:
    # A pre-first-frame viewport draw (result is None) reports the declared
    # mode intent as consistent evidence rather than a false count-0 alarm.
    request = _request(
        color_presentation.MODE_SCENE_LINEAR_HDR, hdr_readback_available=True
    )

    diagnostics = color_presentation.diagnostics_from_request_result(request, None)

    assert diagnostics["display_transform_application_count"] == 1
    assert diagnostics["display_transform_applied_by"] == color_presentation.DISPLAY_TRANSFORM_OWNER_CONSUMER
    assert diagnostics["display_transform_consistent"] is True


@pytest.fixture(autouse=True)
def _clear_presentation_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolution tests own the env override explicitly."""
    monkeypatch.delenv(color_presentation.ENV_COLOR_PRESENTATION_MODE, raising=False)


def test_default_mode_source_is_default_without_env_or_ui() -> None:
    diagnostics = color_presentation.presentation_from_scene(_scene())

    assert diagnostics["requested_mode"] == color_presentation.MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH
    assert diagnostics["active_mode"] == color_presentation.MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH
    assert diagnostics["mode_source"] == color_presentation.MODE_SOURCE_DEFAULT


def test_ui_ldr_selection_resolves_ldr_from_ui() -> None:
    diagnostics = color_presentation.presentation_from_scene(
        _scene(color_presentation.MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH)
    )

    assert diagnostics["active_mode"] == color_presentation.MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH
    assert diagnostics["mode_source"] == color_presentation.MODE_SOURCE_UI


def test_ui_scene_linear_selection_resolves_hdr_when_bindings_present() -> None:
    diagnostics = color_presentation.presentation_from_scene(
        _scene(color_presentation.MODE_SCENE_LINEAR_HDR),
        hdr_readback_available=True,
    )

    assert diagnostics["requested_mode"] == color_presentation.MODE_SCENE_LINEAR_HDR
    assert diagnostics["active_mode"] == color_presentation.MODE_SCENE_LINEAR_HDR
    assert diagnostics["mode_source"] == color_presentation.MODE_SOURCE_UI
    assert diagnostics["status"] == color_presentation.STATUS_CURRENT
    assert diagnostics["render_var"] == color_presentation.RENDER_VAR_HDR_COLOR
    assert diagnostics["unavailable_reason"] == ""


def test_ui_scene_linear_fails_closed_when_bindings_absent() -> None:
    diagnostics = color_presentation.presentation_from_scene(
        _scene(color_presentation.MODE_SCENE_LINEAR_HDR),
        hdr_readback_available=False,
    )

    assert diagnostics["requested_mode"] == color_presentation.MODE_SCENE_LINEAR_HDR
    assert diagnostics["active_mode"] == color_presentation.MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH
    assert diagnostics["mode_source"] == color_presentation.MODE_SOURCE_UI
    assert diagnostics["status"] == color_presentation.STATUS_UNAVAILABLE
    assert (
        diagnostics["unavailable_reason"]
        == color_presentation.HDR_COLOR_READBACK_UNAVAILABLE_REASON
    )
    assert diagnostics["frame_format"] == color_presentation.FRAME_FORMAT_RGBA8
    assert diagnostics["render_var"] == color_presentation.RENDER_VAR_LDR_COLOR


def test_ui_scene_linear_unknown_bindings_do_not_fail_closed() -> None:
    diagnostics = color_presentation.presentation_from_scene(
        _scene(color_presentation.MODE_SCENE_LINEAR_HDR),
        hdr_readback_available=None,
    )

    assert diagnostics["active_mode"] == color_presentation.MODE_SCENE_LINEAR_HDR
    assert diagnostics["mode_source"] == color_presentation.MODE_SOURCE_UI
    assert diagnostics["status"] == color_presentation.STATUS_CURRENT


def test_env_overrides_ui_scene_linear_over_ldr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        color_presentation.ENV_COLOR_PRESENTATION_MODE,
        color_presentation.MODE_SCENE_LINEAR_HDR,
    )

    diagnostics = color_presentation.presentation_from_scene(
        _scene(color_presentation.MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH),
        hdr_readback_available=True,
    )

    assert diagnostics["requested_mode"] == color_presentation.MODE_SCENE_LINEAR_HDR
    assert diagnostics["active_mode"] == color_presentation.MODE_SCENE_LINEAR_HDR
    assert diagnostics["mode_source"] == color_presentation.MODE_SOURCE_ENV


def test_env_overrides_ui_ldr_over_scene_linear(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(color_presentation.ENV_COLOR_PRESENTATION_MODE, "ldr")

    diagnostics = color_presentation.presentation_from_scene(
        _scene(color_presentation.MODE_SCENE_LINEAR_HDR),
        hdr_readback_available=True,
    )

    assert diagnostics["active_mode"] == color_presentation.MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH
    assert diagnostics["mode_source"] == color_presentation.MODE_SOURCE_ENV


def test_env_scene_linear_fails_closed_when_bindings_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        color_presentation.ENV_COLOR_PRESENTATION_MODE,
        color_presentation.MODE_SCENE_LINEAR_HDR,
    )

    diagnostics = color_presentation.presentation_from_scene(
        _scene(color_presentation.MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH),
        hdr_readback_available=False,
    )

    assert diagnostics["requested_mode"] == color_presentation.MODE_SCENE_LINEAR_HDR
    assert diagnostics["active_mode"] == color_presentation.MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH
    assert diagnostics["mode_source"] == color_presentation.MODE_SOURCE_ENV
    assert diagnostics["status"] == color_presentation.STATUS_UNAVAILABLE
    assert (
        diagnostics["unavailable_reason"]
        == color_presentation.HDR_COLOR_READBACK_UNAVAILABLE_REASON
    )


def test_ocio_stays_env_only_and_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(color_presentation.ENV_COLOR_PRESENTATION_MODE, "ocio")

    diagnostics = color_presentation.presentation_from_scene(
        _scene(color_presentation.MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH)
    )

    assert diagnostics["requested_mode"] == color_presentation.MODE_OCIO_BAKED_DISPLAY
    assert diagnostics["active_mode"] == color_presentation.MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH
    assert diagnostics["mode_source"] == color_presentation.MODE_SOURCE_ENV
    assert diagnostics["status"] == color_presentation.STATUS_UNAVAILABLE
    assert diagnostics["unavailable_reason"] == "ocio_baked_display_conversion_unavailable"


def test_resolve_presentation_mode_returns_mode_and_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(color_presentation.ENV_COLOR_PRESENTATION_MODE, raising=False)

    assert color_presentation.resolve_presentation_mode(_scene()) == (
        color_presentation.MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH,
        color_presentation.MODE_SOURCE_DEFAULT,
    )
    assert color_presentation.resolve_presentation_mode(
        _scene(color_presentation.MODE_SCENE_LINEAR_HDR)
    ) == (color_presentation.MODE_SCENE_LINEAR_HDR, color_presentation.MODE_SOURCE_UI)


def test_blender_selector_exposes_only_implemented_modes_with_ldr_default() -> None:
    modes = tuple(item[0] for item in properties.COLOR_PRESENTATION_ENUM_ITEMS)
    assert modes == (
        color_presentation.MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH,
        color_presentation.MODE_SCENE_LINEAR_HDR,
    )
    assert (
        properties.COLOR_PRESENTATION_DEFAULT
        == color_presentation.MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH
    )


def test_unknown_mode_retains_explicit_unavailable_diagnostic() -> None:
    diagnostics = color_presentation.presentation_from_scene(
        _scene(),
        requested_mode="future-output-mode",
    )

    assert diagnostics["requested_mode"] == "future_output_mode"
    assert diagnostics["status"] == color_presentation.STATUS_UNAVAILABLE
    assert diagnostics["unavailable_reason"] == "unknown_color_presentation_mode"


def test_scene_free_scripts_classify_every_render_request_explicitly() -> None:
    for path in sorted((ROOT / "scripts").glob("*.py")):
        source = path.read_text(encoding="utf-8")
        request_count = source.count("RenderRequest(")
        if request_count:
            assert source.count("color_presentation=") == request_count, path

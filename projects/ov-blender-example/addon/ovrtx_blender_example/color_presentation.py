# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Color presentation classification for OVRTX viewport frames."""

from __future__ import annotations

import os
from typing import Any, Mapping


ENV_COLOR_PRESENTATION_MODE = "OV_BLENDER_EXAMPLE_COLOR_PRESENTATION"
MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH = "ldr_rgba8_display_passthrough"
MODE_SCENE_LINEAR_HDR = "scene_linear_hdr"
MODE_OCIO_BAKED_DISPLAY = "ocio_baked_display"
# Compatibility alias for callers that predate ``resolve_presentation_mode``.
# The default presentation is LDR display passthrough; prefer
# ``resolve_presentation_mode`` (env > UI > default) over reading this directly.
DEFAULT_MODE = MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH
FRAME_FORMAT_RGBA8 = "rgba8"
FRAME_FORMAT_RGBA16F = "rgba16f"
FRAME_COLOR_MODE_DISPLAY_LDR = "display_encoded_ldr"
FRAME_COLOR_MODE_SCENE_LINEAR = "scene_linear"
RENDER_VAR_LDR_COLOR = "LdrColor"
RENDER_VAR_HDR_COLOR = "HdrColor"
STATUS_CURRENT = "current_behavior"
STATUS_UNAVAILABLE = "unavailable"
CONVERSION_PASSTHROUGH = "passthrough"
CONVERSION_SCENE_LINEAR = "scene_linear"
CONVERSION_OCIO_BAKED = "ocio_baked"
DISPLAY_TRANSFORM_OWNER_OVRTX = "ovrtx_render_product"
DISPLAY_TRANSFORM_OWNER_CONSUMER = "consumer"
DISPLAY_TRANSFORM_APPLIED_BY_NONE = "none"
PRESENTATION_SCHEMA_VERSION = 2
SCENE_PRESENTATION_PROPERTY = "color_presentation_mode"
MODE_SOURCE_ENV = "env"
MODE_SOURCE_UI = "ui"
MODE_SOURCE_DEFAULT = "default"
HDR_COLOR_READBACK_UNAVAILABLE_REASON = "hdr_color_readback_unavailable"

_MODE_ALIASES = {
    "": MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH,
    "ldr": MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH,
    "rgba8": MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH,
    "ldr_rgba8": MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH,
    MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH: MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH,
    "hdr": MODE_SCENE_LINEAR_HDR,
    "linear": MODE_SCENE_LINEAR_HDR,
    "scene_linear": MODE_SCENE_LINEAR_HDR,
    MODE_SCENE_LINEAR_HDR: MODE_SCENE_LINEAR_HDR,
    "ocio": MODE_OCIO_BAKED_DISPLAY,
    "ocio_baked": MODE_OCIO_BAKED_DISPLAY,
    MODE_OCIO_BAKED_DISPLAY: MODE_OCIO_BAKED_DISPLAY,
}


def resolve_presentation_mode(
    scene: Any | None,
    *,
    requested_mode: str | None = None,
) -> tuple[str, str]:
    """Resolve the presentation mode and its source.

    Precedence is env var (when set) > UI selection > default LDR. The UI
    selection is either the explicit ``requested_mode`` a caller supplies or,
    when that is ``None``, the ``scene.ovrtx_example.color_presentation_mode``
    enum property. When neither an env override, an explicit request, nor a
    scene property is present, the default LDR passthrough mode is used.
    Returns ``(normalized_mode, mode_source)`` where ``mode_source`` is one of
    ``MODE_SOURCE_ENV`` / ``MODE_SOURCE_UI`` / ``MODE_SOURCE_DEFAULT``.
    """

    env_value = os.environ.get(ENV_COLOR_PRESENTATION_MODE, "")
    if env_value.strip():
        return normalize_mode(env_value), MODE_SOURCE_ENV
    if requested_mode is not None:
        return normalize_mode(requested_mode), MODE_SOURCE_UI
    scene_value = _scene_presentation_mode(scene)
    if scene_value.strip():
        return normalize_mode(scene_value), MODE_SOURCE_UI
    return MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH, MODE_SOURCE_DEFAULT


def presentation_from_scene(
    scene: Any | None,
    *,
    requested_mode: str | None = None,
    hdr_readback_available: bool | None = None,
) -> dict[str, Any]:
    requested, mode_source = resolve_presentation_mode(
        scene, requested_mode=requested_mode
    )
    diagnostics = {
        "schema_version": PRESENTATION_SCHEMA_VERSION,
        "requested_mode": requested,
        "mode_source": mode_source,
        "active_mode": MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH,
        "status": STATUS_CURRENT,
        "unavailable_reason": "",
        "frame_format": FRAME_FORMAT_RGBA8,
        "frame_color_mode": FRAME_COLOR_MODE_DISPLAY_LDR,
        "render_var": RENDER_VAR_LDR_COLOR,
        "conversion": CONVERSION_PASSTHROUGH,
        "display_transform_owner": DISPLAY_TRANSFORM_OWNER_OVRTX,
        "blender_display_transform_applied": False,
        "authored_values_adjusted": False,
        "view_settings": view_settings_from_scene(scene),
    }
    if requested == MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH:
        return diagnostics
    if requested == MODE_SCENE_LINEAR_HDR:
        # Fail closed to LDR passthrough when the native client cannot read
        # back HdrColor. ``None`` means the capability is unknown at
        # classification time (e.g. a pure UI draw) and does not fail closed
        # by itself; only an explicit ``False`` from a request-time probe does.
        if hdr_readback_available is False:
            return {
                **diagnostics,
                "status": STATUS_UNAVAILABLE,
                "unavailable_reason": HDR_COLOR_READBACK_UNAVAILABLE_REASON,
            }
        return {
            **diagnostics,
            "active_mode": MODE_SCENE_LINEAR_HDR,
            "status": STATUS_CURRENT,
            "unavailable_reason": "",
            "frame_format": FRAME_FORMAT_RGBA16F,
            "frame_color_mode": FRAME_COLOR_MODE_SCENE_LINEAR,
            "render_var": RENDER_VAR_HDR_COLOR,
            "conversion": CONVERSION_SCENE_LINEAR,
            "display_transform_owner": DISPLAY_TRANSFORM_OWNER_CONSUMER,
        }
    if requested == MODE_OCIO_BAKED_DISPLAY:
        return {
            **diagnostics,
            "status": STATUS_UNAVAILABLE,
            "unavailable_reason": "ocio_baked_display_conversion_unavailable",
            "requested_conversion": CONVERSION_OCIO_BAKED,
        }
    return {
        **diagnostics,
        "status": STATUS_UNAVAILABLE,
        "unavailable_reason": "unknown_color_presentation_mode",
    }


def diagnostics_from_request_result(request: Any | None, result: Any | None) -> dict[str, Any]:
    source = getattr(request, "color_presentation", None) if request is not None else None
    diagnostics = dict(source) if isinstance(source, Mapping) else presentation_from_scene(None)
    result_frame_format = str(getattr(result, "frame_format", "") or diagnostics.get("frame_format", ""))
    result_frame_color_mode = str(
        getattr(result, "frame_color_mode", "")
        or diagnostics.get("frame_color_mode", "")
    )
    diagnostics["result_frame_format"] = result_frame_format
    diagnostics["result_frame_color_mode"] = result_frame_color_mode
    diagnostics["schema_version"] = PRESENTATION_SCHEMA_VERSION
    diagnostics.update(
        _display_transform_evidence(result, diagnostics, result_frame_format, result_frame_color_mode)
    )
    return diagnostics


def _display_transform_evidence(
    result: Any | None,
    diagnostics: Mapping[str, Any],
    result_frame_format: str,
    result_frame_color_mode: str,
) -> dict[str, Any]:
    """Prove the exactly-once display-transform ownership for one frame.

    The count is derived from the actual result frame — which stage(s)
    transformed it — never hard-coded from the mode label. Two independent
    facts are summed:

    * OVRTX applied the display transform iff the frame is display-encoded
      LDR (baked into the ``RGBA8`` payload).
    * Blender applied it iff a scene-linear ``RGBA16F`` payload was handed
      over to be drawn through the display-space shader (task02-05), which
      is also what gets inserted linear into the F12 result.

    A well-formed frame applies the transform exactly once (count ``1``); a
    frame that skipped both stages (count ``0``) or was hit by both (count
    ``2``) is surfaced with ``display_transform_consistent = False`` rather
    than silently normalized. ``display_transform_consistent`` additionally
    requires the applying stage to match the declared
    ``display_transform_owner`` so an owner/frame mismatch (e.g. an LDR frame
    presented under scene-linear mode) does not pass as consistent just
    because the raw count is ``1``.
    """

    ovrtx_applied = result_frame_color_mode == FRAME_COLOR_MODE_DISPLAY_LDR
    if result is not None:
        # A concrete frame: Blender owns the transform only when a scene-linear
        # RGBA16F payload exists to draw through the display-space shader.
        linear_payload_present = bool(getattr(result, "linear_rgba16f", b"") or b"")
        blender_applied = (
            result_frame_format == FRAME_FORMAT_RGBA16F and linear_payload_present
        )
    else:
        # No presented frame yet (e.g. a pre-first-frame viewport draw): fall
        # back to the declared color mode intent.
        blender_applied = result_frame_color_mode == FRAME_COLOR_MODE_SCENE_LINEAR
    count = int(ovrtx_applied) + int(blender_applied)
    if count == 1:
        applied_by = (
            DISPLAY_TRANSFORM_OWNER_CONSUMER
            if blender_applied
            else DISPLAY_TRANSFORM_OWNER_OVRTX
        )
    else:
        applied_by = DISPLAY_TRANSFORM_APPLIED_BY_NONE
    declared_owner = str(diagnostics.get("display_transform_owner", ""))
    return {
        "display_transform_application_count": count,
        "display_transform_applied_by": applied_by,
        "display_transform_consistent": count == 1 and applied_by == declared_owner,
    }


def normalize_mode(value: str | None) -> str:
    key = str(value or "").strip().lower().replace("-", "_")
    return _MODE_ALIASES.get(key, key)


def _scene_presentation_mode(scene: Any | None) -> str:
    settings = getattr(scene, "ovrtx_example", None)
    value = getattr(settings, SCENE_PRESENTATION_PROPERTY, None)
    return str(value) if value else ""


def view_settings_from_scene(scene: Any | None) -> dict[str, Any]:
    view_settings = getattr(scene, "view_settings", None)
    display_settings = getattr(scene, "display_settings", None)
    return {
        "view_transform": str(getattr(view_settings, "view_transform", "")),
        "look": str(getattr(view_settings, "look", "")),
        "exposure": _float_attr(view_settings, "exposure", 0.0),
        "gamma": _float_attr(view_settings, "gamma", 1.0),
        "display_device": str(getattr(display_settings, "display_device", "")),
    }


def _float_attr(source: Any, name: str, default: float) -> float:
    try:
        return float(getattr(source, name, default))
    except Exception:
        return default

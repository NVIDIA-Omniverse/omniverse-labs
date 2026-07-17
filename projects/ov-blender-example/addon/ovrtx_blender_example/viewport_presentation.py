# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Viewport presentation policy for OVRTX rendered previews."""

from __future__ import annotations

import time
from typing import Any


OVRTX_RENDERED_PRESENTATION = "ovrtx_rendered"
NATIVE_VIEWPORT_FALLBACK = "native_viewport_fallback"
ORTHOGRAPHIC_USER_VIEW = "orthographic_user_view"
ORTHOGRAPHIC_CAMERA_VIEW = "orthographic_camera_view"
FALLBACK_SHADING_TYPE = "SOLID"
OVRTX_SHADING_TYPE = "RENDERED"
_MONITOR_INTERVAL_S = 0.1
_ORIENTATION_OVERLAY_ATTRS = (
    "show_overlays",
    "show_floor",
    "show_ortho_grid",
    "show_axis_x",
    "show_axis_y",
    "show_axis_z",
)

_FALLBACKS: dict[int, dict[str, Any]] = {}
_TIMER_BPY_MODULE: Any | None = None


def viewport_presentation_for_context(context: Any) -> dict[str, Any]:
    """Return the presentation state implied by a Blender viewport context."""

    return _presentation_state(
        fallback_reason_for_context(context),
        owned=False,
        view_perspective=_view_perspective_from_context(context),
    )


def apply_native_fallback_for_context(context: Any) -> dict[str, Any]:
    """Apply native fallback for an unsupported active View3D context."""

    space = getattr(context, "space_data", None)
    if space is None:
        return viewport_presentation_for_context(context)
    return reconcile_space_presentation(
        space,
        getattr(context, "scene", None),
        region_data=getattr(context, "region_data", None),
    )


def reconcile_space_presentation(
    space: Any,
    scene: Any | None,
    *,
    region_data: Any | None = None,
    allow_enter_fallback: bool = True,
) -> dict[str, Any]:
    """Apply or restore addon-owned fallback for one View3D space."""

    region_data = region_data or getattr(space, "region_3d", None)
    view_perspective = str(getattr(region_data, "view_perspective", ""))
    reason = fallback_reason_for_view(scene, region_data)
    key = _space_key(space)
    fallback = _FALLBACKS.get(key)
    shading = getattr(space, "shading", None)
    shading_type = str(getattr(shading, "type", ""))

    if reason:
        if shading_type == OVRTX_SHADING_TYPE and (allow_enter_fallback or fallback is not None):
            fallback = _enter_native_fallback(space, reason, view_perspective)
            return _presentation_state(reason, owned=True, view_perspective=view_perspective, changed=True)
        if fallback is not None and shading_type == FALLBACK_SHADING_TYPE:
            fallback["reason"] = reason
            fallback["view_perspective"] = view_perspective
            return _presentation_state(reason, owned=True, view_perspective=view_perspective)
        if fallback is not None and shading_type != FALLBACK_SHADING_TYPE:
            _FALLBACKS.pop(key, None)
        return _presentation_state(reason, owned=False, view_perspective=view_perspective)

    if fallback is None:
        return _presentation_state("", owned=False, view_perspective=view_perspective)

    if shading_type == FALLBACK_SHADING_TYPE:
        _restore_addon_owned_fallback(space, fallback)
        _FALLBACKS.pop(key, None)
        return _presentation_state("", owned=False, view_perspective=view_perspective, changed=True)

    _FALLBACKS.pop(key, None)
    return _presentation_state("", owned=False, view_perspective=view_perspective)


def fallback_reason_for_context(context: Any) -> str:
    return fallback_reason_for_view(getattr(context, "scene", None), getattr(context, "region_data", None))


def fallback_reason_for_view(scene: Any | None, region_data: Any | None) -> str:
    return ""


def register_viewport_presentation_monitor(bpy_module: Any) -> bool:
    """Register the View3D fallback monitor."""

    global _TIMER_BPY_MODULE
    timers = getattr(getattr(bpy_module, "app", None), "timers", None)
    if timers is None:
        return False
    _TIMER_BPY_MODULE = bpy_module
    try:
        if callable(getattr(timers, "is_registered", None)) and timers.is_registered(_monitor_viewports):
            return False
        timers.register(_monitor_viewports, first_interval=_MONITOR_INTERVAL_S)
        return True
    except Exception:
        _TIMER_BPY_MODULE = None
        return False


def unregister_viewport_presentation_monitor(bpy_module: Any) -> bool:
    """Unregister the View3D fallback monitor and clear addon-owned state."""

    global _TIMER_BPY_MODULE
    timers = getattr(getattr(bpy_module, "app", None), "timers", None)
    unregistered = False
    if timers is not None:
        try:
            if callable(getattr(timers, "is_registered", None)) and timers.is_registered(_monitor_viewports):
                timers.unregister(_monitor_viewports)
                unregistered = True
        except Exception:
            pass
    _TIMER_BPY_MODULE = None
    _FALLBACKS.clear()
    return unregistered


def reset_viewport_presentation_state() -> None:
    """Clear fallback ownership state for tests and shutdown."""

    _FALLBACKS.clear()


def _monitor_viewports() -> float | None:
    bpy_module = _TIMER_BPY_MODULE
    if bpy_module is None:
        return None
    try:
        reconcile_all_viewports(bpy_module)
    except Exception:
        pass
    return _MONITOR_INTERVAL_S


def reconcile_all_viewports(bpy_module: Any) -> list[dict[str, Any]]:
    """Reconcile all visible View3D spaces."""

    states: list[dict[str, Any]] = []
    active_keys: set[int] = set()
    windows = getattr(getattr(getattr(bpy_module, "context", None), "window_manager", None), "windows", ())
    for window in windows:
        screen = getattr(window, "screen", None)
        scene = getattr(window, "scene", getattr(getattr(bpy_module, "context", None), "scene", None))
        for area in getattr(screen, "areas", ()):
            if getattr(area, "type", "") != "VIEW_3D":
                continue
            for space in getattr(area, "spaces", ()):
                if getattr(space, "type", "") != "VIEW_3D":
                    continue
                active_keys.add(_space_key(space))
                states.append(reconcile_space_presentation(space, scene, allow_enter_fallback=False))
                tag_redraw = getattr(area, "tag_redraw", None)
                if states[-1].get("changed") and callable(tag_redraw):
                    tag_redraw()
    for key in tuple(_FALLBACKS):
        if key not in active_keys:
            _FALLBACKS.pop(key, None)
    return states


def _enter_native_fallback(space: Any, reason: str, view_perspective: str) -> dict[str, Any]:
    key = _space_key(space)
    shading = getattr(space, "shading", None)
    overlay = getattr(space, "overlay", None)
    fallback = {
        "reason": reason,
        "view_perspective": view_perspective,
        "previous_shading_type": str(getattr(shading, "type", OVRTX_SHADING_TYPE)),
        "overlay": _capture_overlay_state(overlay),
        "entered_at_ns": time.time_ns(),
    }
    _FALLBACKS[key] = fallback
    if shading is not None:
        shading.type = FALLBACK_SHADING_TYPE
    _apply_orientation_overlays(overlay)
    return fallback


def _restore_addon_owned_fallback(space: Any, fallback: dict[str, Any]) -> None:
    shading = getattr(space, "shading", None)
    if shading is not None:
        shading.type = str(fallback.get("previous_shading_type") or OVRTX_SHADING_TYPE)
    _restore_overlay_state(getattr(space, "overlay", None), fallback.get("overlay", {}))


def _presentation_state(
    fallback_reason: str,
    *,
    owned: bool,
    view_perspective: str,
    changed: bool = False,
) -> dict[str, Any]:
    fallback_active = bool(fallback_reason)
    return {
        "presentation_mode": NATIVE_VIEWPORT_FALLBACK if fallback_active else OVRTX_RENDERED_PRESENTATION,
        "fallback_reason": fallback_reason,
        "fallback_owned_by_addon": bool(owned and fallback_active),
        "view_perspective": view_perspective,
        "changed": bool(changed),
    }


def _capture_overlay_state(overlay: Any | None) -> dict[str, bool]:
    if overlay is None:
        return {}
    return {name: bool(getattr(overlay, name)) for name in _ORIENTATION_OVERLAY_ATTRS if hasattr(overlay, name)}


def _apply_orientation_overlays(overlay: Any | None) -> None:
    if overlay is None:
        return
    for name in _ORIENTATION_OVERLAY_ATTRS:
        if hasattr(overlay, name):
            setattr(overlay, name, True)


def _restore_overlay_state(overlay: Any | None, state: Any) -> None:
    if overlay is None or not isinstance(state, dict):
        return
    for name, value in state.items():
        if hasattr(overlay, name):
            setattr(overlay, name, bool(value))


def _view_perspective_from_context(context: Any) -> str:
    return str(getattr(getattr(context, "region_data", None), "view_perspective", ""))


def _space_key(space: Any) -> int:
    as_pointer = getattr(space, "as_pointer", None)
    if callable(as_pointer):
        try:
            return int(as_pointer())
        except Exception:
            pass
    return id(space)


__all__ = [
    "FALLBACK_SHADING_TYPE",
    "NATIVE_VIEWPORT_FALLBACK",
    "ORTHOGRAPHIC_CAMERA_VIEW",
    "ORTHOGRAPHIC_USER_VIEW",
    "OVRTX_RENDERED_PRESENTATION",
    "OVRTX_SHADING_TYPE",
    "apply_native_fallback_for_context",
    "fallback_reason_for_context",
    "fallback_reason_for_view",
    "reconcile_all_viewports",
    "reconcile_space_presentation",
    "register_viewport_presentation_monitor",
    "reset_viewport_presentation_state",
    "unregister_viewport_presentation_monitor",
    "viewport_presentation_for_context",
]

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Blender UI panels for the OVRTX render example."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from pathlib import Path
import os
import subprocess
import sys
from typing import Any

try:
    import bpy  # type: ignore
except ModuleNotFoundError:
    bpy = None  # type: ignore[assignment]

from .engine import ENGINE_ID
from . import color_presentation
from . import session_lifecycle
from . import user_messages


BLENDER_AVAILABLE = bpy is not None


def _require_blender() -> Any:
    if bpy is None:
        raise RuntimeError("ovrtx_blender_example UI requires Blender's bpy module")
    return bpy


# --- Stock panel COMPAT_ENGINES registration (Cycles pattern) ---------------
#
# At add-on registration the engine ID is added to the ``COMPAT_ENGINES`` of
# stock Blender property panels (spec blender-live-render, task03-01), the
# same way Cycles' ``get_panels()`` does: enumerate registered panel classes
# whose ``COMPAT_ENGINES`` contains ``BLENDER_RENDER``, minus a curated
# exclusion list. Unregistration recomputes the same enumeration and removes
# the ID again. Only ``COMPAT_ENGINES`` membership is modified — a stock
# panel's ``poll`` is never wrapped or replaced.
#
# Documented limitation (matches Cycles): panels registered by other add-ons
# *after* this add-on registers are not touched; they only pick up the engine
# ID if that add-on re-runs our registration (e.g. via add-on reload).

# Curated exclusions from the BLENDER_RENDER membership rule, audited against
# Blender 5.1.2 headless. Criterion: exclude a panel when showing it would imply
# feature support OVRTX does not have and whose edits neither flow to the
# authored converters, nor to live value edits, nor to engine-agnostic output
# settings. Ambiguous cases default to included (a visible-but-inert panel is
# less harmful than a missing one).
#
# Blender 5.1 ships no stock BLENDER_RENDER sampling/raytracing panels — the
# EEVEE/Cycles/Workbench sampling panels carry their own COMPAT_ENGINES — so
# no sampling entries are needed here; OVRTX sampling stays on the add-on's
# own OVRTXEXAMPLE_PT_render_settings panel.
STOCK_PANEL_COMPAT_EXCLUSIONS = frozenset(
    {
        # Freestyle line rendering: strokes are drawn only by Blender's
        # internal Freestyle pass (EEVEE/Cycles pipelines). OVRTX never
        # renders them, and Freestyle settings reach neither the converters
        # nor value edits nor engine-agnostic output settings.
        "RENDER_PT_freestyle",
        "MATERIAL_PT_freestyle_line",
        "VIEWLAYER_PT_freestyle",
        "VIEWLAYER_PT_freestyle_animation",
        "VIEWLAYER_PT_freestyle_edge_detection",
        "VIEWLAYER_PT_freestyle_lineset",
        "VIEWLAYER_PT_freestyle_lineset_collection",
        "VIEWLAYER_PT_freestyle_lineset_edgetype",
        "VIEWLAYER_PT_freestyle_lineset_facemarks",
        "VIEWLAYER_PT_freestyle_lineset_visibilty",  # Blender's own spelling
        "VIEWLAYER_PT_freestyle_linestyle_alpha",
        "VIEWLAYER_PT_freestyle_linestyle_color",
        "VIEWLAYER_PT_freestyle_linestyle_geometry",
        "VIEWLAYER_PT_freestyle_linestyle_strokes",
        "VIEWLAYER_PT_freestyle_linestyle_strokes_chaining",
        "VIEWLAYER_PT_freestyle_linestyle_strokes_dashedline",
        "VIEWLAYER_PT_freestyle_linestyle_strokes_selection",
        "VIEWLAYER_PT_freestyle_linestyle_strokes_sorting",
        "VIEWLAYER_PT_freestyle_linestyle_strokes_splitting",
        "VIEWLAYER_PT_freestyle_linestyle_strokes_splitting_pattern",
        "VIEWLAYER_PT_freestyle_linestyle_texture",
        "VIEWLAYER_PT_freestyle_linestyle_thickness",
        "VIEWLAYER_PT_freestyle_style_modules",
        # Grease Pencil rendering: GP objects are neither converted nor
        # composited by OVRTX; these panels tune Blender's internal GP draw
        # engine (anti-aliasing, simplify) with no OVRTX-visible effect.
        "RENDER_PT_gpencil",
        "RENDER_PT_grease_pencil_render",
        "RENDER_PT_grease_pencil_viewport",
        "RENDER_PT_simplify_greasepencil",
        # EEVEE light probes: GI capture volumes evaluated only by EEVEE;
        # probe settings never reach OVRTX. (DATA_PT_context_lightprobe, the
        # generic data-block selector, stays included per the
        # ambiguous-defaults-to-included rule.)
        "DATA_PT_lightprobe",
        "DATA_PT_lightprobe_display",
        "DATA_PT_lightprobe_parallax",
        "DATA_PT_lightprobe_visibility",
        # EEVEE material raster settings (render method, backface culling,
        # transparency options): engine-specific shading controls with no
        # path into the OVRTX material conversion.
        "EEVEE_MATERIAL_PT_viewport_settings",
        # Light type-button stubs: the BLENDER_RENDER DATA_PT_light panel
        # (and its node-editor sibling) draws only the Point/Sun/Spot/Area
        # type buttons; the full light edit surface lives in the EEVEE-named
        # panels in the extra-inclusion list below. Keeping the stub too
        # would show two "Light" panels, so it is excluded — the same way
        # EEVEE itself hides it.
        "DATA_PT_light",
        "NODE_DATA_PT_light",
        # Light preview render widget: the engine declares
        # ``bl_use_preview = False``, so showing it would imply
        # preview-render support that does not exist; the widget carries no
        # editable data.
        "DATA_PT_preview",
    }
)

# Stock panels the membership rule misses — their COMPAT_ENGINES name
# EEVEE/Workbench explicitly instead of BLENDER_RENDER — but that belong to
# the spec's included groups (same idea as Cycles' extra-include list).
STOCK_PANEL_COMPAT_EXTRA_INCLUSIONS = frozenset(
    {
        # Material slot list / data-block selector: despite the EEVEE_
        # prefix this is the stock material context panel; without it the
        # Material properties tab has no slot or data-block UI (Cycles
        # registers its own copy instead of reusing it).
        "EEVEE_MATERIAL_PT_context_material",
        # Light settings: in Blender 5.1 the full light edit surface —
        # type, color, power, shape and size, all of which flow into the
        # light converters and live value edits — is drawn by the
        # EEVEE-named light panels; the BLENDER_RENDER DATA_PT_light stub
        # shows only the type buttons (see exclusions above). The
        # Shadow/Influence/Custom Distance sub-panels ride along per the
        # ambiguous-defaults-to-included rule.
        "DATA_PT_EEVEE_light",
        "DATA_PT_EEVEE_light_distance",
        "DATA_PT_EEVEE_light_influence",
        "DATA_PT_EEVEE_light_shadow",
        "NODE_DATA_PT_EEVEE_light",
        # Camera depth of field: DOF edits flow into the composed scene
        # camera (task01-03 authors DOF opinions), so these belong to the
        # camera-data edit surface even though their compat sets list
        # EEVEE/Workbench explicitly.
        "DATA_PT_camera_dof",
        "DATA_PT_camera_dof_aperture",
    }
)


def stock_panel_included(name: str, compat_engines: Collection[str] | None) -> bool:
    """Pure inclusion rule for one panel class (spec task03-01).

    Membership-based inclusion (``BLENDER_RENDER`` in ``COMPAT_ENGINES``)
    minus the curated exclusion list, plus the small extra-inclusion list for
    stock panels whose compat sets name EEVEE/Workbench explicitly.
    """

    if name in STOCK_PANEL_COMPAT_EXTRA_INCLUSIONS:
        return True
    if compat_engines is None or "BLENDER_RENDER" not in compat_engines:
        return False
    return name not in STOCK_PANEL_COMPAT_EXCLUSIONS


def stock_compat_panel_classes() -> tuple[type, ...]:
    """Enumerate registered ``bpy.types`` panel classes OVRTX should join.

    Recomputed at both register and unregister time — no stored list of
    "panels we touched" — so double register/unregister (add-on reload, test
    harness) stays idempotent via set add/discard.
    """

    _bpy = _require_blender()
    panel_base = _bpy.types.Panel
    panels: list[type] = []
    for name in dir(_bpy.types):
        try:
            cls = getattr(_bpy.types, name, None)
        except Exception:  # pragma: no cover - defensive against RNA quirks
            continue
        if not (isinstance(cls, type) and issubclass(cls, panel_base)):
            continue
        compat = getattr(cls, "COMPAT_ENGINES", None)
        # Only mutable sets are joined; a non-set COMPAT_ENGINES (possible in
        # third-party panels) cannot be edited in place and is skipped.
        if not isinstance(compat, set):
            continue
        if stock_panel_included(name, compat):
            panels.append(cls)
    return tuple(panels)


def register_stock_panel_compat() -> int:
    """Add the engine ID to included stock panels; returns the panel count."""

    count = 0
    for panel in stock_compat_panel_classes():
        panel.COMPAT_ENGINES.add(ENGINE_ID)
        count += 1
    return count


def unregister_stock_panel_compat() -> int:
    """Remove the engine ID from included stock panels (symmetric discard)."""

    count = 0
    for panel in stock_compat_panel_classes():
        panel.COMPAT_ENGINES.discard(ENGINE_ID)
        count += 1
    return count


def _tag_view3d_redraw() -> None:
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def reconnect_viewport_session_result(
    reconnect: Callable[[], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if reconnect is None:
        from .engine import reconnect_viewport_sessions as reconnect

    return dict(reconnect())


def restart_ovrtx_worker_result(
    restart: Callable[[], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if restart is None:
        from .engine import restart_ovrtx_workers as restart

    return dict(restart())


def _with_wait_cursor(context: Any, action: Callable[[], Any]) -> Any:
    cursor_set = getattr(getattr(context, "window", None), "cursor_set", None)
    try:
        if callable(cursor_set):
            cursor_set("WAIT")
        return action()
    finally:
        if callable(cursor_set):
            cursor_set("DEFAULT")


def _recovery_error(
    result: Mapping[str, Any], expected_status: str, fallback: str
) -> str:
    return "" if result.get("status") == expected_status else str(result.get("error") or fallback)


# Session lifecycle statuses that mean OVRTX is mid-startup — not Live yet and
# not idle/failed. First-run shader compilation surfaces as ``compiling`` while
# the runtime-services owner is already ``ready``, so the pending check reads
# the session status the panel displays, not the owner boot status.
START_PENDING_STATUSES = frozenset(
    {
        session_lifecycle.STATUS_STARTING,
        session_lifecycle.STATUS_LOADING,
        session_lifecycle.STATUS_COMPILING,
        session_lifecycle.STATUS_RESYNCING,
        session_lifecycle.STATUS_RECONNECTING,
        session_lifecycle.STATUS_RESTARTING,
    }
)


def runtime_start_pending(
    status: Callable[[], Mapping[str, Any]] | None = None,
) -> bool:
    """True while an OVRTX viewport session is still coming up.

    Covers first-run shader compilation (which can take several minutes) and
    the load/resync/reconnect/restart transitions. Reconnecting or restarting
    during this window discards startup progress and begins again, so the
    operators confirm before proceeding.
    """

    if status is None:
        status = viewport_session_status
    result = dict(status())
    return any(
        str(session.get("status", "")) in START_PENDING_STATUSES
        for session in (result, *result.get("sessions", ()))
    )


# Confirmation copy shown when a reconnect/restart is requested while startup
# is still in flight, warning before discarding shader-compile progress.
RECONNECT_START_PENDING_WARNING = (
    "OVRTX is still starting up — first-run shader compilation can take "
    "several minutes. Reconnecting now may interrupt that startup. Proceed "
    "anyway?"
)
RESTART_WORKER_START_PENDING_WARNING = (
    "OVRTX is still starting up — first-run shader compilation can take "
    "several minutes. Restarting the worker now discards that progress and "
    "begins again. Proceed anyway?"
)


def viewport_session_status(
    statuses: Callable[[], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if statuses is None:
        from .engine import viewport_session_statuses as statuses

    result = dict(statuses())
    sessions = list(result.get("sessions", ()))
    first = dict(sessions[0]) if sessions else session_lifecycle.derive_status(
        engine_active=False,
        ready=False,
    )
    if "logs" not in first:
        first["logs"] = session_lifecycle.log_diagnostics()
    return {
        "status": first.get("status", "stopped"),
        "label": first.get("label", "Stopped"),
        "hint": first.get("hint", ""),
        "active_session_count": int(result.get("active_session_count", len(sessions))),
        "sessions": sessions,
        "logs": dict(first.get("logs", session_lifecycle.log_diagnostics())),
    }


def open_log_folder_result(
    logs: Mapping[str, Any] | None = None,
    opener: Callable[[Path], object] | None = None,
) -> dict[str, Any]:
    log_diagnostics = dict(logs or session_lifecycle.log_diagnostics())
    log_dir_value = str(log_diagnostics.get("log_dir") or "")
    if not log_dir_value:
        return {
            "status": "failed",
            "log_dir": "",
            "error": (
                "File logging is not configured: worker output goes to the "
                "console/Info panel. Set "
                f"{session_lifecycle.WORKER_LOG_ENV} to capture a log file."
            ),
        }
    path = Path(log_dir_value).expanduser()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return {"status": "failed", "log_dir": str(path), "error": f"{type(exc).__name__}: {exc}"}
    opener = opener or _open_path
    try:
        opener(path)
    except OSError as exc:
        return {"status": "failed", "log_dir": str(path), "error": f"{type(exc).__name__}: {exc}"}
    return {"status": "opened", "log_dir": str(path)}


def _open_path(path: Path) -> None:
    if sys.platform == "win32":
        os.startfile(path)  # type: ignore[attr-defined]
        return
    command = "open" if sys.platform == "darwin" else "xdg-open"
    subprocess.Popen([command, str(path)])


def _wrapped_hint_lines(hint: str, width: int = 55, max_lines: int = 4) -> list[str]:
    """Wrap a status hint for panel labels (labels do not wrap themselves)."""

    words = str(hint).split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
        if len(lines) == max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    if len(lines) == max_lines and len(" ".join(words)) > sum(len(line) + 1 for line in lines):
        lines[-1] = lines[-1][: max(0, width - 1)] + "…"
    return lines


# --- OVRTX color-control gating (spec render-quality-color-controls,
# task03-01) ---------------------------------------------------------------
#
# In LDR display-passthrough mode OVRTX owns the display encoding, so Blender's
# View Transform / Look / Exposure / Gamma controls do not affect the presented
# frame. They stay visible for discoverability but are disabled, with a short
# explanation (the issue #54 disable-with-explanation decision). Gating keys
# off the *resolved* presentation mode from
# ``color_presentation.presentation_from_scene`` — which folds in the env
# override, the UI enum, and fail-closed scene-linear — never the raw property
# value. A fail-closed scene-linear selection (``status == unavailable``) is
# gated as LDR with its ``unavailable_reason`` surfaced so the artist knows why
# the selection is not in effect.

LDR_COLOR_GATING_EXPLANATION = (
    "OVRTX owns the display encoding in LDR passthrough; Blender color "
    "management does not affect the presented frame."
)

# Human-readable copy for the fail-closed reasons a scene-linear selection can
# surface. Unknown reasons fall back to the raw code so nothing is silently
# swallowed.
_GATING_UNAVAILABLE_REASON_TEXT = {
    color_presentation.HDR_COLOR_READBACK_UNAVAILABLE_REASON: (
        "Scene-Linear (Blender Color Management) is unavailable: the OVRTX "
        "runtime cannot read back HdrColor frames."
    ),
}


def _gating_unavailable_reason_text(reason: str) -> str:
    reason = str(reason or "")
    return _GATING_UNAVAILABLE_REASON_TEXT.get(
        reason,
        f"Scene-Linear (Blender Color Management) is unavailable: {reason}.",
    )


def color_control_gating(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve OVRTX view-settings gating from presentation diagnostics.

    Blender's display-transform controls (View Transform / Look / Exposure /
    Gamma) are effective only when the resolved presentation is scene-linear
    HDR and currently available. LDR passthrough disables them (OVRTX owns the
    display encoding); a fail-closed scene-linear selection (``status ==
    unavailable``) is likewise disabled — gated as LDR — with the
    ``unavailable_reason`` surfaced so the artist knows why their selection is
    not in effect.

    Returns ``{"enabled": bool, "explanation": list[str]}``; ``explanation`` is
    empty when the controls are enabled.
    """

    active_mode = diagnostics.get("active_mode")
    status = diagnostics.get("status")
    enabled = (
        active_mode == color_presentation.MODE_SCENE_LINEAR_HDR
        and status == color_presentation.STATUS_CURRENT
    )
    if enabled:
        return {"enabled": True, "explanation": []}
    explanation = [LDR_COLOR_GATING_EXPLANATION]
    reason = str(diagnostics.get("unavailable_reason") or "")
    if status == color_presentation.STATUS_UNAVAILABLE and reason:
        explanation.append(_gating_unavailable_reason_text(reason))
    return {"enabled": False, "explanation": explanation}


if bpy is not None:

    class OVRTXEXAMPLE_OT_reset_viewport_camera(bpy.types.Operator):  # type: ignore[misc]
        bl_idname = "ovrtx_example.reset_viewport_camera"
        bl_label = "Reset Preview Camera"
        bl_description = "Use the USD camera for the ovrtx viewport preview"

        def execute(self, context: Any) -> set[str]:
            context.scene.ovrtx_example.sync_viewport_camera = False
            _tag_view3d_redraw()
            return {"FINISHED"}

    class OVRTXEXAMPLE_OT_reconnect_viewport_session(bpy.types.Operator):  # type: ignore[misc]
        bl_idname = "ovrtx_example.reconnect_viewport_session"
        bl_label = "Reconnect OVRTX Session"
        bl_description = "Reconnect to the running OVRTX worker without restarting it"

        def invoke(self, context: Any, event: Any) -> set[str]:
            if runtime_start_pending():
                return context.window_manager.invoke_confirm(
                    self,
                    event,
                    title="OVRTX Is Still Starting",
                    message=RECONNECT_START_PENDING_WARNING,
                    confirm_text="Reconnect Anyway",
                )
            return self.execute(context)

        def execute(self, context: Any) -> set[str]:
            result = _with_wait_cursor(context, reconnect_viewport_session_result)
            error = _recovery_error(
                result, "requested", "OVRTX viewport teardown was not confirmed"
            )
            if error:
                user_messages.report_for_operator(self, {"ERROR"}, error)
                return {"CANCELLED"}
            reconnected = int(result.get("reconnected_session_count", 0))
            user_messages.report_for_operator(
                self,
                {"INFO"},
                f"Reconnect requested for {reconnected} OVRTX viewport session(s)",
            )
            _tag_view3d_redraw()
            return {"FINISHED"}

    class OVRTXEXAMPLE_OT_restart_ovrtx_worker(bpy.types.Operator):  # type: ignore[misc]
        bl_idname = "ovrtx_example.restart_ovrtx_worker"
        bl_label = "Restart OVRTX Worker"
        bl_description = "Stop and relaunch the OVRTX worker process, then reconnect"

        def invoke(self, context: Any, event: Any) -> set[str]:
            if runtime_start_pending():
                return context.window_manager.invoke_confirm(
                    self,
                    event,
                    title="OVRTX Is Still Starting",
                    message=RESTART_WORKER_START_PENDING_WARNING,
                    confirm_text="Restart Anyway",
                )
            return self.execute(context)

        def execute(self, context: Any) -> set[str]:
            result = _with_wait_cursor(context, restart_ovrtx_worker_result)
            error = _recovery_error(
                result, "restarted", "OVRTX worker teardown was not confirmed"
            )
            if error:
                user_messages.report_for_operator(self, {"ERROR"}, error)
                return {"CANCELLED"}
            restarted = int(result.get("restarted_worker_count", 0))
            user_messages.report_for_operator(
                self,
                {"INFO"},
                f"Worker restarted for {restarted} OVRTX viewport session(s)",
            )
            _tag_view3d_redraw()
            return {"FINISHED"}

    class OVRTXEXAMPLE_OT_open_log_folder(bpy.types.Operator):  # type: ignore[misc]
        bl_idname = "ovrtx_example.open_log_folder"
        bl_label = "Open OVRTX Logs"
        bl_description = "Open the stable OVRTX worker and renderer log folder"

        def execute(self, context: Any) -> set[str]:
            result = open_log_folder_result(viewport_session_status().get("logs", {}))
            if result.get("status") == "opened":
                user_messages.report_for_operator(
                    self, {"INFO"}, f"Opened OVRTX logs: {result['log_dir']}"
                )
                return {"FINISHED"}
            user_messages.report_for_operator(
                self,
                {"ERROR"},
                str(result.get("error", "Could not open OVRTX logs")),
            )
            return {"CANCELLED"}

    class OVRTXEXAMPLE_PT_render_settings(bpy.types.Panel):  # type: ignore[misc]
        bl_label = "ovrtx Example"
        bl_idname = "OVRTXEXAMPLE_PT_render_settings"
        bl_space_type = "PROPERTIES"
        bl_region_type = "WINDOW"
        bl_context = "render"
        COMPAT_ENGINES = {ENGINE_ID}

        @classmethod
        def poll(cls, context: Any) -> bool:
            return context.engine in cls.COMPAT_ENGINES

        def draw(self, context: Any) -> None:
            layout = self.layout
            settings = context.scene.ovrtx_example
            status = viewport_session_status()
            # render_product_path / camera_prim_path are deliberately not
            # shown: the live route generates both prims (mirroring the
            # Blender viewport or scene camera) and ignores the settings;
            # they remain programmatic validation surface for the
            # direct-USD (env-override) route only.
            layout.prop(settings, "min_samples")
            layout.prop(settings, "max_samples")
            # RTPT quality controls: one contiguous quality section directly
            # after the samples rows. Only the four documented RTPT attributes
            # are shown (spec render-quality-color-controls, task01-02) — no
            # render-mode selector and no path-tracer-only control. Drawn in
            # the documented order: Max Bounces, Max Specular and Transmission
            # Bounces, Max Volume Bounces, Firefly Filter.
            layout.prop(settings, "rtpt_max_bounces")
            layout.prop(settings, "rtpt_max_specular_and_transmission_bounces")
            layout.prop(settings, "rtpt_max_volume_bounces")
            layout.prop(settings, "rtpt_firefly_filter_enabled")
            # DLSS Super-Resolution toggle. Honored on the RenderProduct at
            # session creation (real-GPU A/B, runtime measurements), so a change
            # re-keys the session and applies on the next viewport refresh with
            # no worker restart. This worker build exposes no full DLSS off, so
            # unchecking selects the DLSS Performance execution mode; the tooltip
            # states the proven behavior and apply latency.
            layout.prop(settings, "dlss_enabled")
            row = layout.row(align=True)
            row.prop(settings, "sync_viewport_camera")
            row.operator("ovrtx_example.reset_viewport_camera", text="Reset")
            # Color Management section (spec render-quality-color-controls,
            # task02-03): Blender's own ``scene.view_settings`` drawn directly.
            # No add-on-owned copies and no value mirroring — editing here and
            # in Blender's stock Color Management panel edit the same data.
            # These are Blender's display controls applied to scene-linear
            # OVRTX frames, not an OVRTX post-grade and not scene compensation.
            color_box = layout.box()
            color_box.label(text="Color Management")
            color_box.label(
                text="Blender display transform applied to scene-linear OVRTX frames"
            )
            # Presentation-mode selector (task02-01) first; it always applies.
            color_box.prop(settings, "color_presentation_mode")
            # The display-transform controls below are Blender's own
            # ``scene.view_settings``, drawn following the stock
            # RENDER_PT_color_management layout (View Transform, Look, then
            # Exposure, Gamma). They are gated by the *resolved* presentation
            # mode (task03-01): enabled in scene-linear HDR, disabled (grayed)
            # in LDR passthrough and in fail-closed scene-linear.
            # ``presentation_from_scene`` folds in the env override, the UI
            # enum, and fail-closed handling; ``color_control_gating`` maps the
            # resulting diagnostics to the enabled flag and the
            # disable-with-explanation copy. At UI draw time HdrColor readback
            # availability is unknown (the native bindings live in the render
            # worker), so scene-linear is not failed closed here on that basis;
            # an env-selected unavailable mode still surfaces its reason.
            gating = color_control_gating(
                color_presentation.presentation_from_scene(context.scene)
            )
            view = context.scene.view_settings
            view_col = color_box.column()
            view_col.enabled = gating["enabled"]
            view_col.prop(view, "view_transform")
            view_col.prop(view, "look")
            view_col.prop(view, "exposure")
            view_col.prop(view, "gamma")
            # Disable-with-explanation: labels do not wrap themselves, so the
            # gating copy is wrapped to the panel width.
            for line in gating["explanation"]:
                for wrapped in _wrapped_hint_lines(line):
                    color_box.label(text=wrapped)
            box = layout.box()
            box.label(text=f"OVRTX Session: {status['label']}")
            hint = str(status.get("hint", "") or "")
            if hint:
                # The actionable detail (e.g. a blocked-conversion reason)
                # must be visible where the user looks, not only in a
                # transient report.
                for line in _wrapped_hint_lines(hint):
                    box.label(text=line)
            logs = status.get("logs", {})
            if logs.get("status") == "file" and logs.get("log_dir"):
                # File logging only exists when the log env overrides are
                # set (validation lanes); the default routes worker output
                # to the console/Info instead.
                box.label(text=f"Logs: {logs['log_dir']}")
                box.operator("ovrtx_example.open_log_folder")
            row = layout.row(align=True)
            row.operator("ovrtx_example.reconnect_viewport_session")
            row.operator("ovrtx_example.restart_ovrtx_worker")

else:
    OVRTXEXAMPLE_OT_reset_viewport_camera = None  # type: ignore[assignment]
    OVRTXEXAMPLE_OT_reconnect_viewport_session = None  # type: ignore[assignment]
    OVRTXEXAMPLE_OT_restart_ovrtx_worker = None  # type: ignore[assignment]
    OVRTXEXAMPLE_OT_open_log_folder = None  # type: ignore[assignment]
    OVRTXEXAMPLE_PT_render_settings = None  # type: ignore[assignment]


_CLASSES = tuple(
    cls
    for cls in (
        OVRTXEXAMPLE_OT_reset_viewport_camera,
        OVRTXEXAMPLE_OT_reconnect_viewport_session,
        OVRTXEXAMPLE_OT_restart_ovrtx_worker,
        OVRTXEXAMPLE_OT_open_log_folder,
        OVRTXEXAMPLE_PT_render_settings,
    )
    if cls is not None
)


def register() -> None:
    _bpy = _require_blender()
    # Also invoked from the add-on's top-level register(); COMPAT_ENGINES
    # joining is a set add, so repeating it here keeps ui.register() complete
    # on its own and stays idempotent.
    register_stock_panel_compat()
    for cls in _CLASSES:
        if not getattr(cls, "is_registered", False):
            _bpy.utils.register_class(cls)


def unregister() -> None:
    _bpy = _require_blender()
    for cls in reversed(_CLASSES):
        try:
            _bpy.utils.unregister_class(cls)
        except (RuntimeError, ValueError) as exc:
            if "missing bl_rna" not in str(exc) and "not registered" not in str(exc):
                raise
    unregister_stock_panel_compat()


__all__ = [
    "BLENDER_AVAILABLE",
    "LDR_COLOR_GATING_EXPLANATION",
    "color_control_gating",
    "STOCK_PANEL_COMPAT_EXCLUSIONS",
    "STOCK_PANEL_COMPAT_EXTRA_INCLUSIONS",
    "stock_panel_included",
    "stock_compat_panel_classes",
    "register_stock_panel_compat",
    "unregister_stock_panel_compat",
    "OVRTXEXAMPLE_OT_reset_viewport_camera",
    "OVRTXEXAMPLE_OT_reconnect_viewport_session",
    "OVRTXEXAMPLE_OT_restart_ovrtx_worker",
    "OVRTXEXAMPLE_OT_open_log_folder",
    "OVRTXEXAMPLE_PT_render_settings",
    "open_log_folder_result",
    "reconnect_viewport_session_result",
    "restart_ovrtx_worker_result",
    "runtime_start_pending",
    "START_PENDING_STATUSES",
    "RECONNECT_START_PENDING_WARNING",
    "RESTART_WORKER_START_PENDING_WARNING",
    "viewport_session_status",
    "register",
    "unregister",
]

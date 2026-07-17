# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RenderEngine shell for the OVRTX render example."""

from __future__ import annotations

from array import array
from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import struct
import threading
import time
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence
import weakref

try:
    import bpy  # type: ignore
except ModuleNotFoundError:
    bpy = None  # type: ignore[assignment]


if bpy is None:
    def _persistent(function: Any) -> Any:
        return function
else:
    _persistent = getattr(
        getattr(getattr(bpy, "app", None), "handlers", None),
        "persistent",
        lambda function: function,
    )

from . import scene_generation_sessions, interactive_operator_state as operator_state
from .blender_callback_adapters import (
    BlenderEditCallbackAdapter,
    BlenderRenderCallbackAdapter,
    EditTranslatorFactory,
    ExactStageRenderCallbackAdapter,
)
from .blender_interactive_edit_builders import build_interactive_edits_from_depsgraph
from .blender_signals import BlenderRenderIntent, BlenderRenderSignalSource
from .blender_signal_translation import BlenderSignalTranslationError, RenderRequestTranslator
from .interactive_edit_planner import InteractiveEdit
from .interactive_edit_workflow import InteractiveEditWorkflow, EditWorkflowResult
from . import color_presentation
from . import blender_signal_translation
from .ovrtx_runtime_client import (
    RenderClientError,
    RenderResult,
    RuntimeServicesPreparingError,
    SIMULATION_ID_PREFIX,
)
from .ovrtx_session_controller import OvrtxSessionController
from .runtime_scheduler import RuntimeScheduler, RuntimeTickResult, RuntimeTickStatus
from . import viewport_handoff, viewport_render_thread
from . import viewport_profile, render_requests
from .render_requests import RenderRequest
from . import usd_paths as usd_paths
from . import usd_prim_resolver
from . import viewport_artifact_recorder
from . import viewport_presentation
from . import ovrtx_scene_composition, ovrtx_session
from . import session_lifecycle
from . import runtime_services
from . import user_messages
from .shared_stage_composition import BodyPose, write_rgba_png
from .value_edit_conversion import ValueEditConversionPolicies


BLENDER_AVAILABLE = bpy is not None
ENGINE_ID = "OVRTX_EXAMPLE"
FINAL_RENDER_USE_POSTPROCESS = False
VIEWPORT_SESSION_TEARDOWN_TIMEOUT_SECONDS = 600.0
_ACTIVE_VIEWPORT_ENGINES: weakref.WeakSet[Any] = weakref.WeakSet()
_RENDER_CALLBACK_ADAPTERS: dict[
    str,
    BlenderRenderCallbackAdapter | ExactStageRenderCallbackAdapter,
] = {}
_EXACT_STAGE_CONFIGURATION: dict[str, str] | None = None
suppress_interactive_edit_bridge = operator_state.suppress_interactive_edit_bridge
class ViewportSessionEndReason(str, Enum):
    OUTPUT_WRITTEN = "output_written"
    ENGINE_DESTROYED = "engine_destroyed"
    SESSION_REPLACED = "session_replaced"
    RECONNECT_REQUESTED = "reconnect_requested"
    WORKER_RESTART_REQUESTED = "worker_restart_requested"
    NATIVE_FALLBACK = "native_fallback"


def _end_reason_value(reason: ViewportSessionEndReason | str) -> str:
    if isinstance(reason, ViewportSessionEndReason):
        return reason.value
    return str(reason)


def _logs_from_startup_diagnostics(startup: Mapping[str, Any]) -> dict[str, Any]:
    worker = startup.get("render_worker")
    if isinstance(worker, Mapping):
        logs = worker.get("logs")
        if isinstance(logs, Mapping):
            return dict(logs)
    return session_lifecycle.log_diagnostics()


#: Plain-Python sidecar for teardown-critical per-engine handles, keyed by
#: ``id(engine)`` (which stays usable after the RNA dies). On a freed
#: engine wrapper EVERY attribute access raises ``ReferenceError`` —
#: ``bpy_struct`` validates the RNA pointer before any lookup, including
#: plain Python instance attributes — so ``__del__``-time teardown and
#: dead-wrapper pruning must never touch the wrapper itself
#: (2026-07-07: this both stormed the panel draw and silently skipped
#: thread teardown, leaking render threads and orphaning workers).
_ENGINE_RUNTIMES: dict[int, dict[str, Any]] = {}
_DIRECT_VIEWPORT_REUSE_BLOCKED = False


def _sidecar_generation_runtime(runtime: Mapping[str, Any] | None) -> Any | None:
    if not runtime or not runtime.get("authored"):
        return None
    generation_runtime = runtime.get("generation_runtime")
    if generation_runtime is not None:
        return generation_runtime
    scene = runtime.get("scene")
    if scene is None:
        return None
    try:
        return scene_generation_sessions.active_runtime_for_scene(scene)
    except Exception:
        return None


def _fail_closed_runtime_reuse(runtime: Any | None) -> None:
    """Disable only the viewport runtime whose termination is unconfirmed."""

    global _DIRECT_VIEWPORT_REUSE_BLOCKED
    if runtime is None:
        _DIRECT_VIEWPORT_REUSE_BLOCKED = True
    else:
        scene_generation_sessions.fail_closed_runtime_reuse(runtime)
    for sidecar in tuple(_ENGINE_RUNTIMES.values()):
        affected = (
            not sidecar.get("authored")
            if runtime is None
            else _sidecar_generation_runtime(sidecar) is runtime
        )
        if not affected:
            continue
        request_stop = getattr(sidecar.get("render_loop"), "request_stop", None)
        if callable(request_stop):
            request_stop()


def _teardown_engine_runtime(runtime: Mapping[str, Any] | None) -> bool:
    """Stop a dead engine's render loop/thread from its sidecar handles.

    Bounded (the thread's own join timeout); safe to call repeatedly and
    with ``None``. This is the ``__del__``/pruning teardown path for
    wrappers whose RNA is gone — the full artifact-writing session end
    needs the wrapper and only runs while it is alive.
    """

    if not runtime:
        return True
    if runtime.get("stop_confirmed") is False:
        return False
    loop = runtime.get("render_loop")
    request_stop = getattr(loop, "request_stop", None)
    if callable(request_stop):
        try:
            request_stop()
        except Exception:
            pass
    render_thread = runtime.get("render_thread")
    teardown = runtime.get("teardown")
    teardown_state = runtime.get("teardown_state")
    generation_runtime = _sidecar_generation_runtime(runtime)
    stop = getattr(render_thread, "stop", None)
    if callable(teardown) and callable(getattr(render_thread, "submit", None)):
        try:
            render_thread.submit(teardown, label="session-teardown")
        except Exception:
            pass
    joined = render_thread is None
    if callable(stop):
        try:
            outcome = stop()
            joined = bool(
                isinstance(outcome, Mapping)
                and outcome.get("joined", not outcome.get("leaked_thread", False))
            )
            teardown_ran = bool(
                isinstance(teardown_state, Mapping) and teardown_state.get("ran")
            )
            if joined and callable(teardown) and not teardown_ran:
                teardown()
            if not joined:
                _fail_closed_runtime_reuse(generation_runtime)
                message = (
                    "[ovrtx_blender_example] defect: dead render engine's viewport "
                    f"session {runtime.get('signal_id', '?')!r} did not complete "
                    "owner-thread teardown before its configured deadline; "
                    "runtime reuse disabled"
                )
                print(message)
                # stdout already carries the defect; also surface it in the
                # Info window (console channel handled by the print above).
                user_messages.report_warning(message, dedup=False, to_console=False)
            elif generation_runtime is not None:
                scene_generation_sessions.detach_viewport(
                    str(runtime.get("signal_id", "")), runtime=generation_runtime
                )
        except Exception:
            _fail_closed_runtime_reuse(generation_runtime)
            joined = False
    runtime["stop_confirmed"] = joined
    if joined:
        runtime["render_loop"] = None
        runtime["render_thread"] = None
        runtime["teardown"] = None
        runtime["teardown_state"] = None
    return joined


def _forget_tracked_engine(engine: Any) -> None:
    """Remove one wrapper from the tracking set, dead or alive.

    ``WeakSet.discard`` re-hashes/compares the wrapper through
    ``bpy_struct``'s RNA-validating dunders, which can misbehave on a
    freed engine (the 2026-07-07 storm never converged because the dead
    wrapper survived discard). Rebuild the set from the survivors when
    plain discard does not stick.
    """

    try:
        _ACTIVE_VIEWPORT_ENGINES.discard(engine)
        if engine not in list(_ACTIVE_VIEWPORT_ENGINES):
            return
    except Exception:
        pass
    survivors = [
        tracked for tracked in list(_ACTIVE_VIEWPORT_ENGINES) if tracked is not engine
    ]
    try:
        _ACTIVE_VIEWPORT_ENGINES.clear()
        for tracked in survivors:
            _ACTIVE_VIEWPORT_ENGINES.add(tracked)
    except Exception:
        pass


def _visit_active_viewport_engines(
    stop: Callable[[Any], bool] | None = None,
) -> tuple[list[Any], list[bool]]:
    """Visit live engines and aggregate every requested or dead-wrapper stop.

    Blender frees a render engine's StructRNA the moment the engine is
    destroyed (viewport closed, workspace switched, F12 render finished),
    but the Python wrapper lives until garbage collection — timers and
    queued thread commands hold bound-method references — so the WeakSet
    can contain dead wrappers. Probe liveness, prune dead wrappers (via
    set rebuild; their hash/eq is unreliable), and stop their render
    runtime from the sidecar — never through the wrapper.
    """

    engines: list[Any] = []
    stopped: list[bool] = []
    dead: list[Any] = []
    tracked = list(_ACTIVE_VIEWPORT_ENGINES)
    for engine in tracked:
        try:
            engine.as_pointer()
        except ReferenceError:
            dead.append(engine)
            runtime = _ENGINE_RUNTIMES.get(id(engine))
            confirmed = _teardown_engine_runtime(runtime)
            stopped.append(confirmed)
            if confirmed:
                _ENGINE_RUNTIMES.pop(id(engine), None)
            continue
        except Exception:
            pass
        engines.append(engine)
        if stop is None:
            continue
        try:
            stopped.append(bool(stop(engine)))
        except ReferenceError:
            engines.pop()
            dead.append(engine)
            runtime = _ENGINE_RUNTIMES.get(id(engine))
            confirmed = _teardown_engine_runtime(runtime)
            stopped.append(confirmed)
            if confirmed:
                _ENGINE_RUNTIMES.pop(id(engine), None)
        except Exception:
            runtime = _ENGINE_RUNTIMES.get(id(engine))
            _fail_closed_runtime_reuse(_sidecar_generation_runtime(runtime))
            stopped.append(False)
    if dead:
        try:
            _ACTIVE_VIEWPORT_ENGINES.clear()
            for engine in engines:
                _ACTIVE_VIEWPORT_ENGINES.add(engine)
        except Exception:
            pass
    tracked_ids = {id(engine) for engine in tracked}
    stopped.extend(
        False
        for engine_id, runtime in _ENGINE_RUNTIMES.items()
        if engine_id not in tracked_ids and runtime.get("stop_confirmed") is False
    )
    return engines, stopped


def _live_viewport_engines() -> list[Any]:
    return _visit_active_viewport_engines()[0]


def _edit_submission_engines(engines: Iterable[Any]) -> tuple[Any, ...]:
    """Choose one edit submitter per authoring runtime or exact-stage pane."""

    result = []
    seen_runtimes: set[int] = set()
    for engine in engines:
        runtime = getattr(engine, "_viewport_generation_runtime", None)
        if runtime is not None:
            identity = id(runtime)
            if identity in seen_runtimes:
                continue
            seen_runtimes.add(identity)
        result.append(engine)
    return tuple(result)


def write_viewport_session_outputs() -> int:
    """Write viewport session outputs for active render-engine instances."""

    written = 0
    for engine in _live_viewport_engines():
        writer = getattr(engine, "_write_viewport_session_outputs", None)
        if not callable(writer):
            continue
        writer(end_reason=ViewportSessionEndReason.OUTPUT_WRITTEN)
        written += 1
    return written


def _stop_active_viewport_engines(stop: Callable[[Any], bool]) -> list[bool]:
    return _visit_active_viewport_engines(stop)[1]


def _end_active_viewport_sessions(reason: ViewportSessionEndReason) -> bool:
    return all(
        _stop_active_viewport_engines(
            lambda engine: engine._end_viewport_session(reason)
        )
    )


def reconnect_viewport_sessions() -> dict[str, Any]:
    """End active viewport sessions and let the next redraw re-attach to the
    still-warm worker (the session's runtime is preserved)."""

    reconnected = 0

    def _reconnect(engine: Any) -> bool:
        nonlocal reconnected
        had_session, stopped = engine._request_viewport_session_reconnect()
        reconnected += bool(had_session and stopped)
        return stopped

    engines, stopped = _visit_active_viewport_engines(_reconnect)
    return {
        "status": "requested" if all(stopped) else "teardown_unconfirmed",
        "active_session_count": len(engines),
        "reconnected_session_count": reconnected,
        "teardown_confirmed": all(stopped),
        "end_reason": ViewportSessionEndReason.RECONNECT_REQUESTED.value,
    }


def restart_ovrtx_workers() -> dict[str, Any]:
    """Quiesce every pane, then synchronously replace only the OVRTX worker."""

    restarted = 0

    def _restart(engine: Any) -> bool:
        nonlocal restarted
        had_session, stopped = engine._request_viewport_worker_restart()
        restarted += bool(had_session and stopped)
        return stopped

    engines, stopped = _visit_active_viewport_engines(_restart)
    result = {
        "active_session_count": len(engines),
        "restarted_worker_count": restarted,
        "teardown_confirmed": all(stopped),
        "end_reason": ViewportSessionEndReason.WORKER_RESTART_REQUESTED.value,
    }
    if not all(stopped):
        return {**result, "status": "teardown_unconfirmed"}
    if not scene_generation_sessions.pause_preparation():
        scene_generation_sessions.resume_preparation()
        return {**result, "status": "teardown_unconfirmed", "teardown_confirmed": False}
    try:
        if not scene_generation_sessions.deactivate_all_ovrtx():
            return {**result, "status": "teardown_unconfirmed", "teardown_confirmed": False}
        try:
            from . import runtime_bundle_status

            runtime = runtime_bundle_status()
            if runtime.get("state") != "ready":
                raise RuntimeError(str(runtime.get("message") or "runtime is unavailable"))
            runtime_services.owner.restart_ovrtx(Path(str(runtime["current_root"])))
        except Exception as exc:
            return {
                **result,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "restarted_worker_count": 0,
            }
    finally:
        scene_generation_sessions.resume_preparation()
    return {**result, "status": "restarted"}


def viewport_session_statuses() -> dict[str, Any]:
    """Return lifecycle diagnostics for currently tracked viewport engines."""

    sessions: list[dict[str, Any]] = []
    for engine in _live_viewport_engines():
        status_getter = getattr(engine, "_viewport_session_status", None)
        if not callable(status_getter):
            continue
        try:
            sessions.append(dict(status_getter()))
        except ReferenceError:
            continue
    return {
        "status": "available",
        "active_session_count": len(sessions),
        "sessions": sessions,
    }


def submit_interactive_edit_to_active_viewports(edit: InteractiveEdit) -> list[EditWorkflowResult]:
    """Submit a resolved stock-Blender edit to active viewport engines."""

    results: list[EditWorkflowResult] = []
    for engine in _edit_submission_engines(_live_viewport_engines()):
        submitter = getattr(engine, "submit_interactive_edit", None)
        if not callable(submitter):
            continue
        try:
            results.append(submitter(edit))
        except ReferenceError:
            continue
    return results


def submit_depsgraph_interactive_edits_to_active_viewports(
    depsgraph: Any,
    *,
    context: Any | None = None,
    scene: Any | None = None,
) -> list[EditWorkflowResult]:
    """Submit tagged stock-Blender depsgraph edits to active viewport engines."""

    return _edit_callback_adapter(scene=scene).submit_depsgraph_interactive_edits(
        depsgraph,
        context=context,
        scene=scene,
    )


def submit_render_setting_change_to_active_viewports(
    property_name: str,
    value: Any,
) -> list[EditWorkflowResult]:
    """Apply a live RTPT quality change on each active viewport's render thread.

    Engines without an active session (no request) author no runtime write;
    the value reaches their next session through composition authoring
    (render-quality-color-controls task01-04).
    """

    results: list[EditWorkflowResult] = []
    for engine in _live_viewport_engines():
        submitter = getattr(engine, "submit_render_setting_change", None)
        if not callable(submitter):
            continue
        try:
            result = submitter(property_name, value)
        except ReferenceError:
            continue
        if result is not None:
            results.append(result)
    return results


def _track_viewport_engine(engine: Any) -> None:
    _ACTIVE_VIEWPORT_ENGINES.add(engine)
    engine_id = id(engine)
    existing = _ENGINE_RUNTIMES.get(engine_id)
    if existing is not None and existing.get("stop_confirmed") is False:
        # Keep the one sidecar store; a short negative-key scan is
        # enough for the rare case where Python reuses a dead wrapper's id.
        orphan_id = -1
        while orphan_id in _ENGINE_RUNTIMES:
            orphan_id -= 1
        _ENGINE_RUNTIMES[orphan_id] = existing
    # Sidecar registration happens while the RNA is alive: the signal id
    # (pointer-derived) is captured now because it cannot be recomputed
    # from a dead wrapper.
    _ENGINE_RUNTIMES[engine_id] = {
        "signal_id": _engine_signal_id(engine),
        "render_thread": None,
        "render_loop": None,
        "teardown": None,
        "teardown_state": None,
        # One route bit is enough; teardown needs no route type.
        "authored": False,
        "generation_runtime": None,
        "scene": None,
    }


def _untrack_viewport_engine(engine: Any, runtime: Mapping[str, Any] | None = None) -> None:
    _forget_tracked_engine(engine)
    if runtime is None:
        runtime = _ENGINE_RUNTIMES.pop(id(engine), None)
    signal_id = str((runtime or {}).get("signal_id", "") or "")
    if signal_id:
        _RENDER_CALLBACK_ADAPTERS.pop(signal_id, None)


def register_interactive_edit_bridge() -> bool:
    return operator_state.register_interactive_edit_bridge(bpy, _live_interactive_edit_depsgraph_handler)


def unregister_interactive_edit_bridge() -> bool:
    return operator_state.unregister_interactive_edit_bridge(bpy, _live_interactive_edit_depsgraph_handler)


def stop_viewport_render_threads_for_file_load() -> bool:
    """End viewport sessions before scene-generation cleanup."""

    return _end_active_viewport_sessions(ViewportSessionEndReason.ENGINE_DESTROYED)


def stop_viewport_sessions_for_unregister() -> bool:
    """Stop viewport sessions before scene-generation cleanup."""

    return _end_active_viewport_sessions(ViewportSessionEndReason.ENGINE_DESTROYED)


#: Set by the render_init handler when a live viewport session was ended for F12,
#: so the render_complete/render_cancel handlers know to bring it back.
_FINAL_RENDER_RESTORE_VIEWPORT = False

#: Render request authored by the render_init handler on the main thread and consumed
#: by render() on the render job thread (see the handler for why).
_FINAL_RENDER_REQUEST: RenderRequest | None = None
_FINAL_RENDER_INIT_ADAPTER_ID = "__ovrtx_final_render_init__"


@_persistent
def _final_render_init_handler(scene: Any, *_args: Any) -> None:
    """End the active viewport session on the main thread before F12 renders.

    ``render_init`` fires on Blender's main thread before ``render()`` runs on the
    render job thread. A live viewport session holds the main thread and the GPU
    context, so if ``render()`` reaches ``_run_on_main_thread(final_render)`` first
    (engine.py render(), before it ends the viewport at line ~1781) the main thread
    never services the marshalled callback and F12 deadlocks. Freeing the viewport
    here -- earlier, and on the main thread -- breaks that ordering. The viewport is
    brought back by the render_complete/render_cancel handlers.
    """

    if getattr(getattr(scene, "render", None), "engine", "") != ENGINE_ID:
        return
    global _FINAL_RENDER_RESTORE_VIEWPORT, _FINAL_RENDER_REQUEST
    # reconnect_viewport_sessions() ends the live session now (freeing the main thread
    # and GPU for F12) but flags it RECONNECT_REQUESTED, so the next viewport redraw
    # starts a fresh session -- which the render_complete/cancel handler triggers.
    try:
        result = reconnect_viewport_sessions()
        _FINAL_RENDER_RESTORE_VIEWPORT = int(result.get("active_session_count", 0)) > 0
    except Exception:
        _FINAL_RENDER_RESTORE_VIEWPORT = False
    # Build the whole render request here, on the main thread. The stock USD export
    # (bpy.ops.wm.usd_export) needs the window-manager context, and material/camera
    # translation reads bpy.data, which is only safe on the main thread. render() then
    # consumes this request instead of marshalling final_render through bpy.app.timers --
    # that timer callback never runs during an F12 INVOKE render, so the marshalling
    # deadlocks.
    try:
        _FINAL_RENDER_REQUEST = _render_callback_adapter(
            _FINAL_RENDER_INIT_ADAPTER_ID
        ).final_render_from_scene(scene)
    except Exception:
        _FINAL_RENDER_REQUEST = None


@_persistent
def _final_render_end_handler(scene: Any, *_args: Any) -> None:
    """Re-enable the viewport after F12 if it was on before the render.

    Redrawing the 3D viewports restarts a Rendered-shading OVRTX session on the next
    ``view_draw`` (a Solid/Material-Preview viewport just repaints, no session).
    """

    global _FINAL_RENDER_RESTORE_VIEWPORT
    if not _FINAL_RENDER_RESTORE_VIEWPORT:
        return
    _FINAL_RENDER_RESTORE_VIEWPORT = False
    # Redraw from a one-shot main-loop timer, not inline: render_complete fires in the
    # render context, where tag_redraw on the 3D viewports does not reliably stick (the
    # render-result window has focus). A timer runs on the main event loop afterwards.
    timers = getattr(getattr(bpy, "app", None), "timers", None)
    if timers is not None:
        timers.register(_restore_viewport_redraw, first_interval=0.1)


def _restore_viewport_redraw() -> None:
    context = getattr(bpy, "context", None)
    wm = getattr(context, "window_manager", None)
    for window in getattr(wm, "windows", ()) or ():
        screen = getattr(window, "screen", None)
        for area in getattr(screen, "areas", ()) or ():
            if getattr(area, "type", "") == "VIEW_3D":
                try:
                    area.tag_redraw()
                except Exception:
                    pass
    return None


def register_final_render_handlers() -> bool:
    handlers = getattr(getattr(bpy, "app", None), "handlers", None)
    if handlers is None:
        return False
    if _final_render_init_handler not in handlers.render_init:
        handlers.render_init.append(_final_render_init_handler)
    for hook in (handlers.render_complete, handlers.render_cancel):
        if _final_render_end_handler not in hook:
            hook.append(_final_render_end_handler)
    return True


def unregister_final_render_handlers() -> bool:
    handlers = getattr(getattr(bpy, "app", None), "handlers", None)
    if handlers is None:
        return False
    if _final_render_init_handler in handlers.render_init:
        handlers.render_init.remove(_final_render_init_handler)
    for hook in (handlers.render_complete, handlers.render_cancel):
        if _final_render_end_handler in hook:
            hook.remove(_final_render_end_handler)
    return True



def interactive_edit_bridge_diagnostics() -> dict[str, Any]:
    return operator_state.interactive_edit_bridge_diagnostics()


def resolve_blender_selection_to_edit_owners(context: Any | None = None) -> dict[str, Any]:
    return operator_state.resolve_blender_selection_to_edit_owners(context, bpy_module=bpy)


def _request_at_initial_condition(request: RenderRequest) -> bool:
    return operator_state.request_at_initial_condition(request)


def _should_mirror_runtime_poses(*, at_initial_condition: bool, lock_was_active: bool) -> bool:
    return operator_state.should_mirror_runtime_poses(
        at_initial_condition=at_initial_condition,
        lock_was_active=lock_was_active,
    )


@_persistent
def _live_interactive_edit_depsgraph_handler(scene: Any, depsgraph: Any) -> None:
    # ``@_persistent`` is load-bearing: Blender clears non-persistent
    # ``depsgraph_update_post`` handlers on every file load, and without it
    # opening a .blend silently kills the live-edit bridge — no transform,
    # light, camera, or material deltas reach the runtime for the rest of
    # the session while the viewport keeps rendering.
    try:
        affected_ids = scene_generation_sessions.affected_blender_ids(depsgraph)
        identity_changes = scene_generation_sessions.topology_identity_changes(scene)
        results = submit_depsgraph_interactive_edits_to_active_viewports(
            depsgraph,
            context=_selection_context_for_scene(scene, depsgraph),
            scene=scene,
        )
        replacement_requested = any(
            result.plan is not None
            and result.plan.impact.scene_generation_replacement_requested
            for result in results
        )
        scene_update = any(
            str(getattr(getattr(getattr(update, "id", update), "bl_rna", None), "identifier", ""))
            == "Scene"
            for update in getattr(depsgraph, "updates", ())
        )
        if (
            scene_update
            and not replacement_requested
            and not identity_changes
        ):
            scene_generation_sessions.record_ignored_scene_update(scene)
        if (
            not _interactive_edit_bridge_suppressed()
            and not scene_generation_sessions.is_reconciling(scene)
            and (replacement_requested or identity_changes)
        ):
            if identity_changes and not replacement_requested:
                affected_ids = set(identity_changes)
            else:
                affected_ids.update(identity_changes)
            scene_generation_sessions.mark_scene_dirty(
                scene,
                affected_ids,
                defer_world_reconciliation=any(
                    getattr(engine, "_viewport_scene", None) is scene
                    for engine in _live_viewport_engines()
                ),
            )
    except Exception as exc:
        _record_interactive_edit_bridge_diagnostics(
            active_engine_count=len(list(_ACTIVE_VIEWPORT_ENGINES)),
            submitted_edit_count=0,
            result_count=0,
            last_error=f"{type(exc).__name__}: {exc}",
        )


@dataclass(frozen=True)
class _SceneSelectionContext:
    scene: Any
    view_layer: Any | None
    selected_objects: tuple[Any, ...]


def _selection_context_for_scene(scene: Any, depsgraph: Any) -> Any | None:
    if bpy is None or scene is None:
        return None
    context = getattr(bpy, "context", None)
    if context is None:
        return context

    view_layer = getattr(depsgraph, "view_layer", None)
    if view_layer is not None:
        return _SceneSelectionContext(
            scene=scene,
            view_layer=view_layer,
            selected_objects=_selected_objects_for_scene(scene, view_layer),
        )
    if getattr(context, "scene", None) is scene:
        return context

    window_manager = getattr(context, "window_manager", None)
    for window in getattr(window_manager, "windows", ()):
        if getattr(window, "scene", None) is not scene:
            continue
        view_layer = getattr(window, "view_layer", None)
        return _SceneSelectionContext(
            scene=scene,
            view_layer=view_layer,
            selected_objects=_selected_objects_for_scene(scene, view_layer),
        )

    view_layers = getattr(scene, "view_layers", ())
    view_layer = view_layers[0] if view_layers else None
    return _SceneSelectionContext(
        scene=scene,
        view_layer=view_layer,
        selected_objects=_selected_objects_for_scene(scene, view_layer),
    )


def _selected_objects_for_scene(scene: Any, view_layer: Any | None) -> tuple[Any, ...]:
    selected = []
    for obj in getattr(scene, "objects", ()):
        select_get = getattr(obj, "select_get", None)
        if not callable(select_get):
            continue
        try:
            is_selected = select_get(view_layer=view_layer)
        except TypeError:
            is_selected = select_get()
        except Exception:
            continue
        if is_selected:
            selected.append(obj)
    return tuple(selected)


def _depsgraph_updated_id_descriptors(
    depsgraph: Any,
) -> set[tuple[str, str]] | None:
    """(ID type name, name) descriptors for a depsgraph's reported updates.

    Scopes the next reconcile to the sources the depsgraph actually
    reported (ADR 0014 targeted dirty-ID conversion): a settings-slider
    drag reports only the Scene ID, so a heavy scene no longer pays a
    full reconversion per event. Returns ``None`` (everything dirty) when
    the update list is unreadable — never silently narrower than the
    truth. Deletions and membership changes stay covered regardless of
    scope: reachability reconciliation, not conversion, owns removals.
    """

    try:
        updates = tuple(getattr(depsgraph, "updates", ()) or ())
    except Exception:
        return None
    descriptors: set[tuple[str, str]] = set()
    for update in updates:
        id_data = getattr(update, "id", None)
        if id_data is None:
            return None
        original = getattr(id_data, "original", None) or id_data
        name = str(
            getattr(original, "name_full", "") or getattr(original, "name", "")
        )
        if not name:
            return None
        descriptors.add((type(original).__name__, name))
    return descriptors


def _interactive_edit_bridge_suppressed() -> bool:
    return operator_state.interactive_edit_bridge_suppressed()


def _record_interactive_edit_bridge_diagnostics(
    *,
    registered: bool | None = None,
    suppressed: bool | None = None,
    active_engine_count: int | None = None,
    submitted_edit_count: int | None = None,
    result_count: int | None = None,
    last_error: str = "",
) -> None:
    operator_state.record_interactive_edit_bridge_diagnostics(
        registered=registered,
        suppressed=suppressed,
        active_engine_count=active_engine_count,
        submitted_edit_count=submitted_edit_count,
        result_count=result_count,
        last_error=last_error,
    )


@dataclass(frozen=True)
class _ViewportTextureUpload:
    texture: Any
    texture_size: tuple[Any, ...]
    accepts_rgba8: bool
    diagnostics: dict[str, Any]
    #: Frame color mode of the payload the texture was built from. A change
    #: between LDR (``display_encoded_ldr``) and scene-linear
    #: (``scene_linear``) requires a fresh texture because the two use
    #: different GPU formats (RGBA8 vs RGBA16F), so the cached fast paths must
    #: not carry a texture across a mode flip (task02-05).
    color_mode: str = color_presentation.FRAME_COLOR_MODE_DISPLAY_LDR


def _result_artifact_path_from_environment() -> str:
    return os.environ.get(
        "OV_BLENDER_EXAMPLE_RENDER_ARTIFACT",
        os.environ.get("OV_BLENDER_EXAMPLE_FRAME_ARTIFACT", ""),
    )


def _render_request_translator(
    *, include_material_presentation: bool = True
) -> RenderRequestTranslator:
    return RenderRequestTranslator(
        blender_module_provider=lambda: bpy,
        include_material_presentation=include_material_presentation,
    )


def _render_callback_adapter(
    engine_id: str = "",
) -> BlenderRenderCallbackAdapter | ExactStageRenderCallbackAdapter:
    if engine_id and engine_id in _RENDER_CALLBACK_ADAPTERS:
        return _RENDER_CALLBACK_ADAPTERS[engine_id]
    configuration = _EXACT_STAGE_CONFIGURATION
    if configuration is not None:
        adapter: BlenderRenderCallbackAdapter | ExactStageRenderCallbackAdapter = (
            ExactStageRenderCallbackAdapter(
                input_usd_path=configuration["input_usd_path"],
                camera_prim_path=configuration["camera_prim_path"],
                render_product_path=configuration["render_product_path"],
                translator=_render_request_translator(include_material_presentation=False),
                engine_id=engine_id,
            )
        )
    else:
        adapter = BlenderRenderCallbackAdapter(
            generation_for_scene=scene_generation_sessions.generation_for_scene,
            viewport_generation_for_scene=(
                scene_generation_sessions.generation_for_viewport
            ),
            translator=_render_request_translator(),
            engine_id=engine_id,
        )
    if engine_id:
        _RENDER_CALLBACK_ADAPTERS[engine_id] = adapter
    return adapter


def configure_exact_stage(
    *,
    input_usd_path: str,
    camera_prim_path: str,
    render_product_path: str,
) -> None:
    """Configure the concrete exact-stage adapter for an internal harness."""

    global _EXACT_STAGE_CONFIGURATION
    path = Path(input_usd_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"exact-stage input does not exist: {path}")
    if not str(camera_prim_path).startswith("/"):
        raise ValueError("exact-stage camera prim path must be absolute")
    if not str(render_product_path).startswith("/"):
        raise ValueError("exact-stage RenderProduct prim path must be absolute")
    _EXACT_STAGE_CONFIGURATION = {
        "input_usd_path": str(path),
        "camera_prim_path": str(camera_prim_path),
        "render_product_path": str(render_product_path),
    }
    _RENDER_CALLBACK_ADAPTERS.clear()


def clear_exact_stage() -> None:
    global _EXACT_STAGE_CONFIGURATION
    _EXACT_STAGE_CONFIGURATION = None
    _RENDER_CALLBACK_ADAPTERS.clear()


def _edit_callback_adapter(*, scene: Any | None = None) -> BlenderEditCallbackAdapter:
    current_scene = scene is not None and _EXACT_STAGE_CONFIGURATION is None
    return BlenderEditCallbackAdapter(
        active_engines=lambda: () if current_scene else _edit_submission_engines(
            _live_viewport_engines()
        ),
        bridge_suppressed=_interactive_edit_bridge_suppressed,
        selection_resolver=resolve_blender_selection_to_edit_owners,
        bridge_diagnostics_recorder=_record_interactive_edit_bridge_diagnostics,
        edit_builder=build_interactive_edits_from_depsgraph,
        edit_observer=(
            None if current_scene else scene_generation_sessions.retain_interactive_edit
        ),
        edit_observer_context=scene_generation_sessions.current_generation_edit_context,
        edit_group_observer=(
            scene_generation_sessions.submit_current_scene_edit_group
            if current_scene
            else None
        ),
    )


def _engine_signal_id(engine: Any) -> str:
    as_pointer = getattr(engine, "as_pointer", None)
    identity = int(as_pointer()) if callable(as_pointer) else id(engine)
    return f"{ENGINE_ID}:{identity:x}"


class _PublicationRedrawSignaler:
    """Coalesced publication→redraw signaling for one engine (task02-05).

    The render thread announces frame publications; Blender presents on the
    main thread. ``bpy.app.timers.register`` is the documented thread-safe
    entry to main-thread execution and is the only Blender API this class
    touches from the render thread — the single crossing. ``signal()`` sets
    an atomic ``redraw_pending`` flag and registers a one-shot timer only on
    the False→True transition, so any burst of publications before the timer
    fires collapses into one main-thread redraw request (the next draw reads
    the newest frame anyway). The timer callback clears the flag, tags the
    owning viewport's redraw through a weakref to the engine (a dead
    reference makes the callback a no-op), and returns ``None`` so the timer
    unregisters itself. When ``engine.tag_redraw()`` raises (engine freed
    between the reference check and the call), the exception-guarded VIEW_3D
    area scan (``_tag_viewport_redraws``) is the fallback.

    ``redraw_requested_monotonic_ns`` is stamped when the pending flag
    latches (publication time, render-thread side) so the publish→draw span
    in the profile includes timer dispatch latency — that latency is part of
    perceived input latency and must be measured, not hidden.
    """

    def __init__(self, engine: Any) -> None:
        self._engine_ref = weakref.ref(engine)
        self._lock = threading.Lock()
        self._pending = False
        self._requested_monotonic_ns = 0
        self._signal_count = 0
        self._timer_registrations = 0
        self._timer_fires = 0
        self._fallback_redraws = 0
        self._registration_failures = 0

    def signal(self) -> None:
        """Request one coalesced main-thread redraw; safe from any thread."""

        with self._lock:
            self._signal_count += 1
            if self._pending:
                # Absorbed: a redraw is already pending and the next draw
                # reads the newest published frame anyway.
                return
            self._pending = True
            self._requested_monotonic_ns = time.perf_counter_ns()
        try:
            if bpy is None:
                raise RuntimeError("bpy is unavailable")
            bpy.app.timers.register(self._fire_redraw, first_interval=0.0)
        except Exception:
            # Best-effort registration (Blender shutting down, or timers
            # unavailable under test): release the latch so the next
            # publication retries instead of wedging redraws forever.
            with self._lock:
                self._pending = False
                self._registration_failures += 1
            return
        with self._lock:
            self._timer_registrations += 1

    def _fire_redraw(self) -> None:
        """Main-thread timer callback: clear the latch, tag, unregister."""

        with self._lock:
            self._pending = False
            self._timer_fires += 1
        engine = self._engine_ref()
        if engine is None:
            return None
        try:
            engine.tag_redraw()
        except Exception:
            with self._lock:
                self._fallback_redraws += 1
            try:
                engine._tag_viewport_redraws()
            except Exception:
                pass
        return None

    def consume_request_mark(self) -> int | None:
        """Return-and-clear the newest publication-side redraw stamp."""

        with self._lock:
            mark = self._requested_monotonic_ns
            self._requested_monotonic_ns = 0
        return mark or None

    def diagnostics(self) -> dict[str, Any]:
        """Signal/timer counters (task02-09 embeds these in the artifact)."""

        with self._lock:
            return {
                "redraw_pending": self._pending,
                "signals": self._signal_count,
                "timer_registrations": self._timer_registrations,
                "timer_fires": self._timer_fires,
                "fallback_redraws": self._fallback_redraws,
                "registration_failures": self._registration_failures,
            }


def _new_tick_absorb_counters() -> dict[str, int]:
    """Fresh tick-absorb handoff counters (task02-09; 02-07 follow-up)."""

    return {
        "handoffs": 0,
        "idle_skipped": 0,
        "stale_loop_dropped": 0,
        "coalesced": 0,
        "timer_registrations": 0,
        "registration_failures": 0,
        "absorbs_applied": 0,
        "absorbs_empty": 0,
    }


def _rgba8_to_float_array(payload: bytes) -> Any:
    try:
        import numpy as np  # type: ignore

        return np.frombuffer(payload, dtype=np.uint8).astype(np.float32) * np.float32(1.0 / 255.0)
    except Exception:
        return array("f", (value / 255.0 for value in payload))


def _rgba16f_to_float_array(payload: bytes) -> Any:
    try:
        import numpy as np  # type: ignore

        rgba = np.frombuffer(payload, dtype=np.float16).astype(np.float32)
        rgba[3::4] = np.float32(1.0)
        return rgba
    except Exception:
        rgba = array("f", (float(value[0]) for value in struct.iter_unpack("<e", payload)))
        for index in range(3, len(rgba), 4):
            rgba[index] = 1.0
        return rgba


def _upload_viewport_texture(
    gpu: Any,
    render_result: RenderResult,
    *,
    cached_texture: Any | None,
    cached_texture_size: tuple[Any, ...] | None,
    cached_texture_snapshot_index: int,
    snapshot_index: int,
    accepts_rgba8: bool,
    cached_texture_color_mode: str = color_presentation.FRAME_COLOR_MODE_DISPLAY_LDR,
) -> _ViewportTextureUpload:
    texture_size = (
        render_result.width,
        render_result.height,
        render_result.frame_format,
    )
    # Scene-linear (HdrColor RGBA16F) frames must be drawn through Blender's
    # display-space shader (task02-05, contract step 4); they carry linear
    # pixels in ``linear_rgba16f`` and use an RGBA16F GPU texture. LDR frames
    # keep the RGBA8 raw-draw path. A mode flip changes the GPU texture format,
    # so the cached reuse/update fast paths only apply when the cached
    # texture's color mode matches the incoming frame's.
    display_transform = viewport_handoff.frame_applies_blender_display_transform(
        render_result
    )
    color_mode = str(
        render_result.frame_color_mode
        or color_presentation.FRAME_COLOR_MODE_DISPLAY_LDR
    )
    same_color_mode = cached_texture_color_mode == color_mode
    linear = (
        render_result.frame_format == color_presentation.FRAME_FORMAT_RGBA16F
        and bool(render_result.linear_rgba16f)
    )
    update_available = _gpu_texture_update_available(cached_texture)
    payload_bytes = (
        len(render_result.linear_rgba16f)
        if (display_transform or linear)
        else len(render_result.rgba8)
    )
    diagnostics: dict[str, Any] = {
        "texture_path": "new_texture",
        "texture_cache_hit": False,
        "gpu_texture_update_available": update_available,
        "texture_upload_bytes": payload_bytes,
        "texture_update_ms": 0.0,
        "texture_convert_ms": 0.0,
        "texture_buffer_ms": 0.0,
        "texture_create_ms": 0.0,
        "texture_filter_ms": 0.0,
        "display_transform": display_transform,
        "frame_color_mode": color_mode,
    }

    if (
        cached_texture is not None
        and cached_texture_size == texture_size
        and cached_texture_snapshot_index == snapshot_index
        and same_color_mode
    ):
        diagnostics["texture_path"] = "reuse"
        diagnostics["texture_cache_hit"] = True
        diagnostics["texture_upload_bytes"] = 0
        return _ViewportTextureUpload(
            texture=cached_texture,
            texture_size=texture_size,
            accepts_rgba8=accepts_rgba8,
            diagnostics=diagnostics,
            color_mode=color_mode,
        )

    # The in-place UBYTE update fast path is RGBA8/LDR only; a scene-linear
    # frame (or a mode flip) always rebuilds the texture so an RGBA8 update is
    # never applied to an RGBA16F texture (or vice versa).
    if (
        not display_transform
        and same_color_mode
        and cached_texture is not None
        and cached_texture_size == texture_size
    ):
        update = getattr(cached_texture, "update", None)
        if callable(update):
            try:
                update_started_ns = time.perf_counter_ns()
                if linear:
                    convert_started_ns = time.perf_counter_ns()
                    rgba = _rgba16f_to_float_array(render_result.linear_rgba16f)
                    diagnostics["texture_convert_ms"] += (
                        time.perf_counter_ns() - convert_started_ns
                    ) / 1_000_000.0
                    buffer_started_ns = time.perf_counter_ns()
                    buffer = gpu.types.Buffer("FLOAT", len(rgba), rgba)
                    diagnostics["texture_buffer_ms"] += (
                        time.perf_counter_ns() - buffer_started_ns
                    ) / 1_000_000.0
                    update(buffer, format="FLOAT")
                else:
                    buffer_started_ns = time.perf_counter_ns()
                    buffer = gpu.types.Buffer("UBYTE", len(render_result.rgba8), render_result.rgba8)
                    diagnostics["texture_buffer_ms"] += (
                        time.perf_counter_ns() - buffer_started_ns
                    ) / 1_000_000.0
                    update(buffer, format="UBYTE")
                diagnostics["texture_path"] = "update"
                diagnostics["texture_cache_hit"] = True
                diagnostics["texture_update_ms"] = (time.perf_counter_ns() - update_started_ns) / 1_000_000.0
                return _ViewportTextureUpload(
                    texture=cached_texture,
                    texture_size=texture_size,
                    accepts_rgba8=accepts_rgba8,
                    diagnostics=diagnostics,
                    color_mode=color_mode,
                )
            except Exception as exc:
                diagnostics["texture_update_failed"] = True
                diagnostics["texture_update_error"] = str(exc)

    texture, accepts_rgba8, texture_path, upload_bytes = _new_viewport_texture(
        gpu,
        render_result,
        accepts_rgba8,
        diagnostics,
        display_transform=display_transform,
    )
    diagnostics["texture_path"] = texture_path
    diagnostics["texture_cache_hit"] = False
    diagnostics["texture_upload_bytes"] = upload_bytes
    if not update_available:
        diagnostics["gpu_texture_update_available"] = _gpu_texture_update_available(texture)
    return _ViewportTextureUpload(
        texture=texture,
        texture_size=texture_size,
        accepts_rgba8=accepts_rgba8,
        diagnostics=diagnostics,
        color_mode=color_mode,
    )


def _new_viewport_texture(
    gpu: Any,
    render_result: RenderResult,
    accepts_rgba8: bool,
    diagnostics: dict[str, Any],
    *,
    display_transform: bool = False,
) -> tuple[Any, bool, str, int]:
    if display_transform or (
        render_result.frame_format == color_presentation.FRAME_FORMAT_RGBA16F
        and render_result.linear_rgba16f
    ):
        # Scene-linear frame: upload the linear RGBA16F pixels as a float
        # RGBA16F texture. Blender's display-space shader tone maps these on
        # draw (the exactly-once application point) — never an LDR-encoded
        # payload, so no RGBA8 fast path here.
        convert_started_ns = time.perf_counter_ns()
        rgba = _rgba16f_to_float_array(render_result.linear_rgba16f)
        diagnostics["texture_convert_ms"] += (
            time.perf_counter_ns() - convert_started_ns
        ) / 1_000_000.0
        buffer_started_ns = time.perf_counter_ns()
        buffer = gpu.types.Buffer("FLOAT", len(rgba), rgba)
        diagnostics["texture_buffer_ms"] += (
            time.perf_counter_ns() - buffer_started_ns
        ) / 1_000_000.0
        create_started_ns = time.perf_counter_ns()
        texture = gpu.types.GPUTexture(
            (render_result.width, render_result.height),
            format="RGBA16F",
            data=buffer,
        )
        diagnostics["texture_create_ms"] += (
            time.perf_counter_ns() - create_started_ns
        ) / 1_000_000.0
        filter_started_ns = time.perf_counter_ns()
        texture.filter_mode(True)
        diagnostics["texture_filter_ms"] += (
            time.perf_counter_ns() - filter_started_ns
        ) / 1_000_000.0
        return texture, accepts_rgba8, "scene_linear_float", _buffer_upload_bytes(rgba, 4)

    if accepts_rgba8:
        create_started_ns = None
        try:
            buffer_started_ns = time.perf_counter_ns()
            buffer = gpu.types.Buffer("UBYTE", len(render_result.rgba8), render_result.rgba8)
            diagnostics["texture_buffer_ms"] += (
                time.perf_counter_ns() - buffer_started_ns
            ) / 1_000_000.0
            create_started_ns = time.perf_counter_ns()
            texture = gpu.types.GPUTexture((render_result.width, render_result.height), format="RGBA8", data=buffer)
            diagnostics["texture_create_ms"] += (
                time.perf_counter_ns() - create_started_ns
            ) / 1_000_000.0
            create_started_ns = None
            filter_started_ns = time.perf_counter_ns()
            texture.filter_mode(True)
            diagnostics["texture_filter_ms"] += (
                time.perf_counter_ns() - filter_started_ns
            ) / 1_000_000.0
            return texture, accepts_rgba8, "new_texture", len(render_result.rgba8)
        except Exception:
            if create_started_ns is not None:
                diagnostics["texture_create_ms"] += (
                    time.perf_counter_ns() - create_started_ns
                ) / 1_000_000.0
            accepts_rgba8 = False

    convert_started_ns = time.perf_counter_ns()
    rgba = _rgba8_to_float_array(render_result.rgba8)
    diagnostics["texture_convert_ms"] += (
        time.perf_counter_ns() - convert_started_ns
    ) / 1_000_000.0
    buffer_started_ns = time.perf_counter_ns()
    buffer = gpu.types.Buffer("FLOAT", len(rgba), rgba)
    diagnostics["texture_buffer_ms"] += (
        time.perf_counter_ns() - buffer_started_ns
    ) / 1_000_000.0
    create_started_ns = time.perf_counter_ns()
    texture = gpu.types.GPUTexture((render_result.width, render_result.height), format="RGBA8", data=buffer)
    diagnostics["texture_create_ms"] += (
        time.perf_counter_ns() - create_started_ns
    ) / 1_000_000.0
    filter_started_ns = time.perf_counter_ns()
    texture.filter_mode(True)
    diagnostics["texture_filter_ms"] += (
        time.perf_counter_ns() - filter_started_ns
    ) / 1_000_000.0
    return texture, accepts_rgba8, "fallback_float", _buffer_upload_bytes(rgba, 4)


def _gpu_texture_update_available(texture: Any | None) -> bool:
    return texture is not None and callable(getattr(texture, "update", None))


def _buffer_upload_bytes(buffer_data: Any, default_item_size: int) -> int:
    try:
        return int(buffer_data.nbytes)
    except AttributeError:
        return len(buffer_data) * default_item_size


def _viewport_profile_enabled() -> bool:
    return bool(os.environ.get("OV_BLENDER_EXAMPLE_VIEWPORT_PROFILE", ""))


def build_request_from_scene(
    scene: Any,
    context: Any | None = None,
    *,
    source: BlenderRenderSignalSource,
    intent: BlenderRenderIntent,
) -> RenderRequest:
    """Compatibility shim for callers that have not built a render signal."""

    return _render_callback_adapter().translate(source, intent, scene, context)


#: Ceiling for a main-thread scene-preparation round trip during F12.
_MAIN_THREAD_SCENE_PREP_TIMEOUT_S = 600.0


def _run_on_main_thread(
    function: Callable[[], Any],
    *,
    timeout: float = _MAIN_THREAD_SCENE_PREP_TIMEOUT_S,
) -> Any:
    """Run ``function`` on Blender's main thread and return its result.

    The stock USD export drives ``bpy.ops.wm.usd_export`` (and, for particle
    hair, other operators), which require the main thread's window-manager
    context. F12's ``render()`` runs on Blender's render *job* thread with a
    restricted context where those operators fail ``poll()`` with "context is
    incorrect". When no live viewport has authored the scene generation yet
    (e.g. F12 straight from Solid / Material Preview), the first export must
    therefore be marshalled onto the main thread. ``bpy.app.timers.register`` is
    the documented thread-safe entry to main-thread execution (already used here
    for redraws); the render thread blocks on the result while Blender's main
    event loop runs the callback. On the main thread (or with ``bpy`` absent
    under test) the function runs inline.
    """

    if bpy is None or threading.current_thread() is threading.main_thread():
        return function()
    done = threading.Event()
    outcome: dict[str, Any] = {}

    def _invoke() -> None:
        try:
            outcome["result"] = function()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller thread
            outcome["error"] = exc
        finally:
            done.set()
        return None

    bpy.app.timers.register(_invoke, first_interval=0.0)
    if not done.wait(timeout):
        raise BlenderSignalTranslationError(
            "Scene preparation on Blender's main thread timed out"
        )
    error = outcome.get("error")
    if error is not None:
        raise error
    return outcome.get("result")


#: Artifact ``input_source`` vocabulary.
INPUT_SOURCE_ACTIVE_SCENE = "active_scene"
INPUT_SOURCE_ENV_OVERRIDE = "env_override"


def _request_input_source(request: RenderRequest | None) -> str:
    """Scene-input provenance of a translated request."""

    if request is None:
        return ""
    if getattr(request, "authored_generation_digest", ""):
        return INPUT_SOURCE_ACTIVE_SCENE
    if getattr(request, "input_usd_path", ""):
        return INPUT_SOURCE_ENV_OVERRIDE
    return ""


def _scene_generation_provenance(request: RenderRequest | None) -> dict[str, Any]:
    """Viewport-artifact provenance for the current scene generation."""

    input_usd_path = str(getattr(request, "input_usd_path", "") or "")
    return {
        "input_source": _request_input_source(request),
        "authored_generation_digest": str(
            getattr(request, "authored_generation_digest", "") or ""
        ),
        "authored_generation": getattr(request, "authored_generation", None),
        "input_usd_path": input_usd_path,
    }


FINAL_RENDER_BREAK_POLL_SECONDS = 0.1

#: Final-render job outcome statuses (task05-01).
FINAL_RENDER_STATUS_COMPLETED = "completed"
FINAL_RENDER_STATUS_CANCELLED = "cancelled"

FINAL_RENDER_PROGRESS_STARTING = "Starting final render"
FINAL_RENDER_PROGRESS_RENDERING = "Rendering"


def _final_render_progress() -> dict[str, Any]:
    return {"status": FINAL_RENDER_PROGRESS_STARTING, "completed_samples": 0}


def _final_render_simulation_id() -> str:
    """F12 simulation-ID lane on the shared worker (task05-01).

    Keeps the ``ovrtx-blender-<lane>-<pid>`` convention so the worker-attach
    sweep's stale-PID parser (task05-02) covers crashed final renders.
    """

    return f"{SIMULATION_ID_PREFIX}final-{os.getpid()}"


def _viewport_final_render_host(scene: Any) -> tuple[Any | None, bool]:
    """Viewport engine whose running render thread serves ``scene``.

    Scene binding (task05-01): F12 rides a viewport session's RPC thread
    only when that engine's authored scene matches the scene being
    rendered. Everything else — direct-route (env-override) sessions,
    other scenes, missing or stopped threads — takes the standalone
    cold-boot path (the task05-03 seam).
    """

    scene_uid = int(getattr(scene, "session_uid", 0) or 0)
    if scene_uid <= 0:
        return None, True
    engines, stopped = _visit_active_viewport_engines()
    for engine in engines:
        viewport_scene = getattr(engine, "_viewport_scene", None)
        if int(getattr(viewport_scene, "session_uid", 0) or 0) != scene_uid:
            continue
        render_thread = getattr(engine, "_render_thread", None)
        if render_thread is None or getattr(engine, "_render_loop", None) is None:
            continue
        if render_thread.status() != viewport_render_thread.STATUS_RUNNING:
            continue
        return engine, all(stopped)
    return None, all(stopped)


def _run_final_render_job(
    request: RenderRequest,
    cancel_event: threading.Event,
    progress: MutableMapping[str, Any] | None = None,
    suspend_host_session: Callable[[], str] | None = None,
    *,
    controller: OvrtxSessionController | None = None,
    composition: ovrtx_scene_composition.OvrtxSceneComposition | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Final-render job body; runs on a session RPC thread.

    One code path for final rendering: current-scene requests pass their
    prepared authoring controller and composition, while the viewport-hosted
    and standalone exact-stage routes let this job own a dedicated final
    controller. The job owns its progress and requests bounded sample chunks
    that grow toward its fixed endpoint;
    ``cancel_event`` — flagged by the render job thread when
    ``test_break()`` reports a user cancel — is checked between batches.

    ``suspend_host_session`` (viewport-hosted route): the worker loads one
    simulation at a time, so the hosting viewport session's simulation is
    suspended — deleted, controller kept open — before the final lane is
    created. Safe here because this job runs exclusively on that session's
    own RPC thread; after the job returns, the viewport loop's replacement
    probe reports ``no_active_session`` and re-ensures the session before
    viewport sampling resumes.
    """

    def _note_progress(status: str, completed_samples: int = 0) -> None:
        if progress is not None:
            progress["status"] = status
            progress["completed_samples"] = int(completed_samples)

    if (controller is None) != (composition is None):
        raise ValueError(
            "final-render controller and composition must be supplied together"
        )
    owns_controller = controller is None
    if controller is None:
        controller = OvrtxSessionController(simulation_id=_final_render_simulation_id())
        # This job already runs on the session RPC thread; adopt it so the
        # debug thread-confinement guard stays truthful for the F12 lane.
        controller.adopt_owning_thread()

    def _cancelled() -> bool:
        if cancel_requested is not None and cancel_requested():
            cancel_event.set()
        return cancel_event.is_set()

    try:
        if suspend_host_session is not None and suspend_host_session() == "failed":
            raise RenderClientError(
                "Could not suspend the viewport OVRTX simulation for the final render"
            )
        if composition is None:
            composition = controller.ensure(request).composition
        _note_progress(FINAL_RENDER_PROGRESS_RENDERING)
        endpoint = max(1, int(request.max_samples))
        result: RenderResult | None = None
        completed_samples = 0
        cancelled = _cancelled()
        while not cancelled:
            additional_samples = min(
                endpoint - completed_samples,
                max(1, completed_samples),
            )
            acquired = controller.render(
                request,
                additional_samples=additional_samples,
            )
            completed_samples += additional_samples
            result = replace(acquired, completed_samples=completed_samples)
            _note_progress(
                FINAL_RENDER_PROGRESS_RENDERING,
                completed_samples,
            )
            cancelled = _cancelled()
            if cancelled or completed_samples >= endpoint:
                break
        if cancelled:
            return {
                "status": FINAL_RENDER_STATUS_CANCELLED,
                "result": None,
                "composition": composition,
            }
        return {
            "status": FINAL_RENDER_STATUS_COMPLETED,
            "result": result,
            "composition": composition,
        }
    finally:
        # Dedicated exact-stage controllers own their teardown here. A
        # current-scene caller restores its borrowed preparation outside the
        # job, after success, cancellation, or failure.
        if owns_controller:
            try:
                controller.shutdown()
            except Exception:
                pass


def _final_render_color_presentation_from_scene(scene: Any) -> Mapping[str, Any]:
    requested = os.environ.get(color_presentation.ENV_COLOR_PRESENTATION_MODE)
    if requested is None or not requested.strip():
        return color_presentation.presentation_from_scene(
            scene,
            requested_mode=color_presentation.MODE_SCENE_LINEAR_HDR,
        )
    return color_presentation.presentation_from_scene(scene, requested_mode=requested)


if bpy is not None:

    class OvrtxExampleRenderEngine(bpy.types.RenderEngine):  # type: ignore[misc]
        bl_idname = ENGINE_ID
        bl_label = "ovrtx Example"
        bl_use_preview = False
        bl_use_postprocess = FINAL_RENDER_USE_POSTPROCESS
        bl_use_eevee_viewport = False
        bl_use_shading_nodes_custom = False
        bl_use_gpu_context = True

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            _track_viewport_engine(self)
            self._viewport_request: RenderRequest | None = None
            self._ovrtx_session_controller: OvrtxSessionController | None = None
            self._viewport_scene: Any | None = None
            self._viewport_generation_runtime: Any | None = None
            self._current_result: RenderResult | None = None
            self._snapshot_index = 0
            self._viewport_camera_controls_mode = render_requests.CAMERA_CONTROLS_USD
            self._viewport_reported_error = ""
            self._ovrtx_scene_composition: ovrtx_scene_composition.OvrtxSceneComposition | None = None
            self._scene_generation_artifact_request: RenderRequest | None = None
            self._viewport_texture: Any | None = None
            self._viewport_texture_size: tuple[Any, ...] | None = None
            self._texture_snapshot_index = 0
            self._viewport_texture_accepts_rgba8 = True
            self._viewport_texture_color_mode = (
                color_presentation.FRAME_COLOR_MODE_DISPLAY_LDR
            )
            self._texture_upload: dict[str, Any] = {}
            self._image_artifact: dict[str, Any] = {}
            self._viewport_last_operator_view: dict[str, Any] = {}
            self._pose_mirror: dict[str, Any] = {}
            self._applied_pose_mirror: dict[str, Any] = {}
            self._pending_pose_mirror: dict[str, Any] = {}
            self._pose_mirror_timer_registered = False
            # Render-thread → main-thread tick-result handoff (task02-07):
            # the loop's tick_result_sink stores the newest result plus the
            # snapshot-derived at-initial-condition fact here and registers
            # a coalesced one-shot absorb timer (latest-wins: a burst of
            # ticks before the timer fires absorbs only the newest result,
            # which carries the newest complete pose set).
            self._tick_absorb_lock = threading.Lock()
            self._pending_tick_absorb: tuple[RuntimeTickResult, bool] | None = None
            self._tick_absorb_timer_pending = False
            # Absorb-timer counters (task02-09; the 02-07 follow-up): the
            # artifact witnesses the render-thread → main-thread pose-mirror
            # handoff cadence. Written under _tick_absorb_lock.
            self._tick_absorb_counters: dict[str, int] = _new_tick_absorb_counters()
            self._viewport_presentation: dict[str, Any] = {
                "presentation_mode": viewport_presentation.OVRTX_RENDERED_PRESENTATION,
                "fallback_reason": "",
                "fallback_owned_by_addon": False,
                "view_perspective": "",
                "changed": False,
            }
            self._viewport_snapshot_count = 0
            self._viewport_draw_count = 0
            # Main-thread override for the redraw-request stamp; measurement
            # scripts (scripts/run_blender_navigation.py) write it directly
            # when they inject input events. Publication-driven redraws stamp
            # the signaler instead (task02-05).
            self._redraw_requested_monotonic_ns: int | None = None
            self._render_count = 0
            self._viewport_camera_update_count = 0
            self._viewport_session_started_ns = 0
            self._runtime_startup_diagnostics: dict[str, Any] = {"render_worker": {"status": "not_started"}}
            self._viewport_lifecycle_phase = ""
            self._viewport_start_failure_count = 0
            self._viewport_restart_count = 0
            self._viewport_session_outputs_written = False
            self._viewport_crash_marker: dict[str, Any] = {}
            self._viewport_log_diagnostics: dict[str, Any] = session_lifecycle.log_diagnostics()
            self._viewport_cleanup_diagnostics: dict[str, Any] = {"status": "not_requested"}
            self._lifecycle_report_message = ""
            self._usd_prim_resolver = usd_prim_resolver.UsdPrimResolver()
            self._viewport_artifact_recorder = viewport_artifact_recorder.Recorder(
                profile_factory=viewport_profile.new,
                record_profile=viewport_profile.record,
                profile_summary=viewport_profile.summary,
                enabled=_viewport_profile_enabled,
                render_records_summary=viewport_profile.render_thread_summary,
            )
            self._runtime_scheduler: RuntimeScheduler | None = None
            self._runtime_tick_result: RuntimeTickResult | None = None
            self._interactive_edit_workflow: InteractiveEditWorkflow | None = None
            self._physics_playback_lock = operator_state.PhysicsPlaybackLock()
            self._physics_playback_lock_reported_message = ""
            # Async viewport loop handoff (task02-04): the main thread only
            # writes snapshots to the mailbox and peeks the latest published
            # frame; all srtx RPCs run on the per-session render thread.
            self._camera_mailbox = viewport_handoff.CameraRequestMailbox()
            self._frame_slot = viewport_handoff.LatestFrameSlot()
            # Publication → coalesced one-shot redraw timer (task02-05):
            # every publish to the frame slot signals this; the timer tags
            # the redraw on the main thread.
            self._redraw_signaler = _PublicationRedrawSignaler(self)
            self._render_thread: viewport_render_thread.ViewportRenderThread | None = None
            self._render_loop: viewport_render_thread.LatestViewRenderLoop | None = None
            # Final thread/loop diagnostics captured when the render loop
            # stops (task02-09): the session-end artifact is written after
            # _stop_render_loop clears the live references, so the ending
            # session's thread identity, loop counters, and per-iteration
            # records are preserved here until the next session starts.
            self._thread_model_final: dict[str, Any] = {}
            self._presented_publication_index = 0
            self._presented_frame: viewport_handoff.FrameState | None = None
            self._written_snapshot_key: tuple[Any, ...] | None = None

        def __del__(self) -> None:
            runtime = _ENGINE_RUNTIMES.get(id(self))
            try:
                self._end_viewport_session(ViewportSessionEndReason.ENGINE_DESTROYED)
            except Exception:
                # A dead RNA wrapper cannot run the full session end. The
                # sidecar is also the safe fallback for any partial failure.
                _teardown_engine_runtime(runtime)
            finally:
                _untrack_viewport_engine(self, runtime)
                if runtime is None or runtime.get("stop_confirmed") is not False:
                    _ENGINE_RUNTIMES.pop(id(self), None)

        def update_render_passes(self, scene: Any = None, renderlayer: Any = None) -> None:
            self.register_pass(scene, renderlayer, "Combined", 4, "RGBA", "COLOR")

        def submit_interactive_edit(self, edit: InteractiveEdit) -> EditWorkflowResult:
            runtime = self._viewport_generation_runtime
            if runtime is not None:
                result = runtime.submit_edit_group((edit,))[0]
            else:
                locked_result = self._physics_playback_lock.reject_edit(edit)
                if locked_result is not None:
                    if locked_result.reason == "physics_playback_locked":
                        self._report_physics_playback_lock_rejection(edit)
                    return locked_result
                result = self._ensure_interactive_edit_workflow().preview_edit(edit)
            # Once-per-key user-visible classification report (task04-07):
            # the workflow dedupes on (target, source field) for the
            # session; the edit record is written every time regardless.
            user_report = getattr(result, "user_report", "")
            if user_report:
                self._report_final({"INFO"}, user_report)
            if result.accepted:
                self._usd_prim_resolver.record_uv_loop_order_validation(edit)
                self._current_result = None
                self._texture_snapshot_index = -1
                try:
                    self.tag_redraw()
                except Exception:
                    pass
            return result

        def submit_render_setting_change(
            self,
            property_name: str,
            value: Any,
        ) -> EditWorkflowResult | None:
            """Route a live RTPT quality change to the render thread (task01-04).

            The write targets the active render product prim resolved from this
            engine's current request and is applied on the session-owning
            render thread (the edit is queued into the shared
            ``ViewUpdateStream`` here on the main thread; the write itself runs
            in the render loop's ``apply_pending``). With no active session
            (no request) nothing is written — the property update alone stands.
            """

            from . import rtpt_live_change

            loop = self._render_loop
            if loop is not None and not loop.rtpt_live_route_supported:
                # This worker rejected a live render-setting write earlier, so
                # the live route is disabled for the session (task01-04
                # fallback). The property change already tagged a viewport
                # redraw; the new value re-keys the session through the
                # composition digest (task01-03) instead of a would-be-rejected
                # runtime write.
                return None
            edit = rtpt_live_change.render_setting_edit_for_request(
                property_name, value, self._viewport_request
            )
            if edit is None:
                return None
            return self.submit_interactive_edit(edit)

        def record_interactive_selection_resolution(
            self,
            selection_resolution: Mapping[str, Any],
        ) -> dict[str, Any]:
            return self._ensure_interactive_edit_workflow().record_selection_resolution(selection_resolution)

        def build_interactive_edits_from_depsgraph(
            self,
            depsgraph: Any,
            *,
            context: Any | None = None,
            selection_resolution: Mapping[str, Any] | None = None,
            value_edit_conversion_policies: ValueEditConversionPolicies | None = None,
            edit_translator_factory: EditTranslatorFactory | None = None,
        ) -> list[InteractiveEdit]:
            request = self._viewport_request
            resolver = self._usd_prim_resolver
            light_objects = (
                tuple(getattr(getattr(bpy, "data", None), "objects", ()) or ())
                if bpy is not None
                else ()
            )
            if request is None and _EXACT_STAGE_CONFIGURATION is None:
                resolver, light_objects = (
                    scene_generation_sessions.current_generation_edit_context(
                        getattr(depsgraph, "scene", None)
                    )
                )
            else:
                if request is None:
                    request = RenderRequest(
                        input_usd_path=str(
                            _EXACT_STAGE_CONFIGURATION["input_usd_path"]
                        )
                    )
                resolver.scan(request)
            write_target_input_usd_path, ignored_layer_identifiers = (
                self._write_target_resolution_context(request)
                if _EXACT_STAGE_CONFIGURATION is not None
                else ("", ())
            )
            worlds = tuple(getattr(getattr(bpy, "data", None), "worlds", ()) or ()) if bpy is not None else ()
            adapter = BlenderEditCallbackAdapter(
                edit_builder=build_interactive_edits_from_depsgraph,
                value_edit_conversion_policies=value_edit_conversion_policies,
                edit_translator_factory=edit_translator_factory,
            )
            edits = tuple(
                adapter.translate_depsgraph_edits(
                    depsgraph,
                    context=context,
                    selection_resolution=selection_resolution,
                    input_usd_path=write_target_input_usd_path or None,
                    ignored_layer_identifiers=ignored_layer_identifiers,
                    usd_prim_resolver=resolver,
                    light_objects=light_objects,
                    worlds=worlds,
                    value_edit_conversion_policies=value_edit_conversion_policies,
                )
            )
            return list(edits)

        def _write_target_resolution_context(
            self,
            request: RenderRequest | None,
        ) -> tuple[str, tuple[str, ...]]:
            input_usd_path = str(getattr(request, "input_usd_path", "") or "")
            composition = self._ovrtx_scene_composition
            if (
                composition is not None
                and input_usd_path
                and composition.source_scene_path == input_usd_path
            ):
                return (
                    composition.composed_scene_path,
                    tuple(composition.session_layer_identifiers),
                )
            return input_usd_path, ()

        def _report_physics_playback_lock_rejection(self, edit: InteractiveEdit) -> None:
            prim_path = edit.usd_prim_path or "<unknown>"
            message = (
                "Physics owns this playback state. Rewind to frame 1 to edit "
                f"physics inputs for {prim_path}."
            )
            self.update_stats("ovrtx", message)
            if message == self._physics_playback_lock_reported_message:
                return
            try:
                self.report({"INFO"}, message)
            except Exception:
                pass
            user_messages.report_info(
                message, context=f"physics-playback-lock:{id(self)}"
            )
            self._physics_playback_lock_reported_message = message

        def update(self, data: Any, depsgraph: Any) -> None:
            self.update_stats("ovrtx", "Render engine ready")

        def render(self, depsgraph: Any) -> None:
            try:
                adapter = _render_callback_adapter(_engine_signal_id(self))
                # The request is normally authored on the main thread by the
                # render_init handler (bpy.ops export + bpy.data reads need the main
                # thread); render() is on the render job thread. Consume it here. Fall
                # back to inline authoring only when render() itself runs on the main
                # thread (e.g. bpy.ops.render.render() EXEC), where bpy access is safe.
                global _FINAL_RENDER_REQUEST
                request = _FINAL_RENDER_REQUEST
                _FINAL_RENDER_REQUEST = None
                if request is None:
                    request = _run_on_main_thread(lambda: adapter.final_render(depsgraph))
            except BlenderSignalTranslationError as exc:
                self._report_final({"ERROR"}, str(exc))
                return
            sample_endpoint = max(1, int(request.max_samples))
            request = replace(
                request,
                max_samples=sample_endpoint,
                color_presentation=_final_render_color_presentation_from_scene(
                    depsgraph.scene
                ),
            )
            scene = getattr(depsgraph, "scene", None)
            if scene is not None and scene_generation_sessions.owns_request(scene, request):
                # Scene-generation-owned request (generated/physics scenes):
                # render synchronously through the generation runtime's
                # controller after ending any live viewport sessions so the
                # worker is free for this lane.
                if not _end_active_viewport_sessions(
                    ViewportSessionEndReason.SESSION_REPLACED
                ):
                    self._report_final(
                        {"ERROR"},
                        "OVRTX final render cannot start while viewport teardown is unconfirmed",
                    )
                    return
                self.update_stats("ovrtx", "Rendering")
                controller = OvrtxSessionController()
                runtime = None
                restore_status = "stopped"
                render_error = None
                try:
                    runtime = scene_generation_sessions.activate_for_final_render(
                        scene,
                        request,
                        controller=controller,
                    )
                    outcome = _run_final_render_job(
                        request,
                        threading.Event(),
                        controller=runtime.controller,
                        composition=runtime.last_ensure_result.composition,
                        cancel_requested=self.test_break,
                    )
                except (RenderClientError, RuntimeError) as exc:
                    render_error = exc
                finally:
                    try:
                        if runtime is not None:
                            restore_status = runtime.deactivate()
                    finally:
                        controller.shutdown()
                if restore_status == "failed":
                    message = "Could not restore prepared viewport state"
                    if render_error is not None:
                        message = f"{message} after render failure: {render_error}"
                    self.report({"ERROR"}, message)
                    return
                if render_error is not None:
                    self.report({"ERROR"}, str(render_error))
                    return
                if outcome.get("status") != FINAL_RENDER_STATUS_COMPLETED:
                    return
                render_result = outcome["result"]
                composition = outcome.get("composition")
            else:
                self.update_stats("ovrtx", "Rendering")
                host, dead_viewports_stopped = _viewport_final_render_host(scene)
                if not dead_viewports_stopped:
                    self._report_final(
                        {"ERROR"},
                        "OVRTX final render cannot start while viewport teardown is unconfirmed",
                    )
                    return
                if host is not None:
                    # An active viewport session serves this scene: the F12 job
                    # runs on that session's RPC thread.
                    outcome = self._final_render_on_viewport_thread(host, request)
                    if outcome is None:
                        return
                    render_result, composition = outcome
                else:
                    # No viewport session serves this scene, so F12 owns a
                    # short-lived RPC context. Any unrelated viewport ends
                    # first to release the exclusive GPU lease.
                    if not _end_active_viewport_sessions(
                        ViewportSessionEndReason.SESSION_REPLACED
                    ):
                        self._report_final(
                            {"ERROR"},
                            "OVRTX final render cannot start while viewport teardown is unconfirmed",
                        )
                        return
                    outcome = self._final_render_standalone(request)
                    if outcome is None:
                        return
                    render_result, composition = outcome
            self._write_blender_result(render_result)
            self._write_result_artifact(
                render_result,
                request,
                composition,
                scene=depsgraph.scene,
            )
            self.update_stats("ovrtx", "Done")

        def _final_render_on_viewport_thread(
            self,
            host: Any,
            request: RenderRequest,
        ) -> tuple[RenderResult, ovrtx_scene_composition.OvrtxSceneComposition | None] | None:
            """Run the F12 job on the viewport session's RPC thread and wait.

            The job is an exclusive render-loop command
            (:meth:`LatestViewRenderLoop.call`): the viewport loop yields
            between iterations while it is queued/running. The worker loads
            one simulation at a time, so the job suspends the viewport
            simulation (controller kept open) before creating the final
            lane; after the job the loop's replacement probe re-ensures the
            session on the next snapshot. Blender's render job thread
            blocks here on the future, polling ``test_break()`` between
            bounded waits — a user cancel flags the shared event, the job
            observes it between sample batches and deletes the final-render
            simulation, and this method returns ``None`` silently. Errors
            and a viewport thread dying mid-render report through
            ``self.report({'ERROR'})`` and return ``None``.
            """

            loop = getattr(host, "_render_loop", None)
            if loop is None:
                self._report_final(
                    {"ERROR"},
                    "OVRTX final render could not start: the viewport render "
                    "loop is no longer available",
                )
                return None
            host_controller = getattr(host, "_ovrtx_session_controller", None)
            suspend_host_session = getattr(host_controller, "suspend", None)
            cancel_event = threading.Event()
            progress = _final_render_progress()
            try:
                future = loop.call(
                    lambda: _run_final_render_job(
                        request,
                        cancel_event,
                        progress,
                        suspend_host_session=suspend_host_session,
                    ),
                    label="final-render",
                )
            except viewport_render_thread.RenderThreadError as exc:
                self._report_final({"ERROR"}, f"OVRTX final render could not start: {exc}")
                return None
            return self._await_final_render_outcome(
                future, cancel_event, request, progress
            )

        def _final_render_standalone(
            self,
            request: RenderRequest,
        ) -> tuple[RenderResult, ovrtx_scene_composition.OvrtxSceneComposition | None] | None:
            """Standalone F12 (task05-03): a short-lived session RPC thread.

            No viewport session serves this scene, so the render constructs
            its own RPC context: a ``ViewportRenderThread`` owned for the
            render's duration runs the exact job the viewport-hosted route
            submits (``_run_final_render_job`` — worker launch, session
            creation, chunked cancellable batches, readback), keeping the
            single-RPC-thread invariant without a parallel direct-call
            implementation. The job's own ``finally`` is the render-end
            teardown (simulation delete + worker shutdown) and runs on the
            thread that owns the session handles, ahead of the stop
            sentinel by queue order — the 02-08 session-teardown pattern —
            so the bounded stop/join below returns only after the worker
            shutdown finished. A fresh worker launch pays the first-attach
            sweep against an empty worker (task05-02: nothing to list, no
            retry sleeps), so the historical startup cleanup storm is gone.
            Wait/cancel semantics are shared with the
            viewport-hosted route via ``_await_final_render_outcome``.
            """

            render_thread = viewport_render_thread.ViewportRenderThread(
                f"final-{os.getpid()}"
            )
            cancel_event = threading.Event()
            progress = _final_render_progress()
            try:
                try:
                    render_thread.start()
                    future = render_thread.call(
                        lambda: _run_final_render_job(request, cancel_event, progress),
                        label="final-render",
                    )
                except Exception as exc:
                    # Deliberately broader than the viewport route's
                    # RenderThreadError catch: a failing Thread.start()
                    # re-raises the OS-level exception (e.g. RuntimeError)
                    # unchanged, and this brand-new thread has no other
                    # observer to surface it.
                    self._report_final(
                        {"ERROR"}, f"OVRTX final render could not start: {exc}"
                    )
                    return None
                return self._await_final_render_outcome(
                    future, cancel_event, request, progress
                )
            finally:
                # Retire the short-lived thread with the bounded join
                # (never unbounded). After a cancel return the
                # in-flight job drains to its next batch boundary and runs
                # its teardown first (queue order guarantees
                # teardown-before-sentinel); a join timeout abandons the
                # daemon thread — the process-exit daemon rule is the
                # backstop — and is surfaced as a defect diagnostic.
                stop_outcome = render_thread.stop()
                if stop_outcome.get("leaked_thread"):
                    try:
                        message = (
                            "[ovrtx_blender_example] defect: standalone "
                            f"final-render thread {render_thread.name!r} "
                            "leaked (join timeout "
                            f"{stop_outcome.get('join_timeout_seconds')}s); "
                            "daemon thread abandoned"
                        )
                        print(message)
                        # stdout already carries the defect; also surface it in
                        # the Info window.
                        user_messages.report_warning(
                            message, dedup=False, to_console=False
                        )
                    except Exception:
                        pass

        def _await_final_render_outcome(
            self,
            future: viewport_render_thread.RenderThreadResult,
            cancel_event: threading.Event,
            request: RenderRequest,
            progress: Mapping[str, Any] | None = None,
        ) -> tuple[RenderResult, ovrtx_scene_composition.OvrtxSceneComposition | None] | None:
            """Block the render job thread on a final-render job future.

            Shared by the viewport-hosted and standalone routes: polls
            ``test_break()`` between bounded waits (a user cancel flags the
            shared event, the job cleans up at its next batch boundary, and
            this returns ``None`` silently — Blender's cancel semantics)
            and reports errors through ``self.report({'ERROR'})``.
            """
            observed_progress: tuple[str, int] | None = None
            cancelled = False
            while not future.wait(FINAL_RENDER_BREAK_POLL_SECONDS):
                state = (
                    str((progress or {}).get("status", FINAL_RENDER_PROGRESS_STARTING)),
                    int((progress or {}).get("completed_samples", 0) or 0),
                )
                if state != observed_progress:
                    observed_progress = state
                    status, completed_samples = state
                    if completed_samples:
                        status = (
                            f"{status} samples {completed_samples}/"
                            f"{max(1, int(request.max_samples))}"
                        )
                    self.update_stats("ovrtx", status)
                if not cancelled and self.test_break():
                    # Cancel between thread-job milestones: the job checks
                    # the event between sample batches and cleans up.
                    cancel_event.set()
                    cancelled = True
            try:
                outcome = future.result(0)
            except RenderClientError as exc:
                self._report_final({"ERROR"}, str(exc))
                return None
            except Exception as exc:
                # RenderThreadRejectedError lands here when the viewport
                # thread died mid-render: surface it through the render
                # job's error path (task02-01 contract).
                self._report_final(
                    {"ERROR"},
                    f"OVRTX final render failed: {type(exc).__name__}: {exc}",
                )
                return None
            if outcome.get("status") != FINAL_RENDER_STATUS_COMPLETED:
                # User cancel: no result, no error report.
                return None
            return outcome["result"], outcome.get("composition")

        def _write_blender_result(self, render_result: RenderResult) -> None:
            # F12 keeps passing float scene-linear pixels for HdrColor frames
            # (Blender's image pipeline owns the display transform from there);
            # LDR frames insert their display-encoded RGBA8. The shared
            # classifier guarantees only a genuine scene-linear RGBA16F frame
            # takes the linear branch, so LDR-encoded pixels are never inserted
            # as if linear (task02-05, contract step 4).
            if viewport_handoff.frame_applies_blender_display_transform(render_result):
                if len(render_result.linear_rgba16f) != render_result.width * render_result.height * 8:
                    raise RenderClientError("OVRTX render result size does not match its RGBA16F payload")
                rgba = _rgba16f_to_float_array(render_result.linear_rgba16f)
            else:
                if len(render_result.rgba8) != render_result.width * render_result.height * 4:
                    raise RenderClientError("OVRTX render result size does not match its RGBA payload")
                rgba = _rgba8_to_float_array(render_result.rgba8)
            blender_result = self.begin_result(0, 0, render_result.width, render_result.height)
            layer = blender_result.layers[0].passes["Combined"]
            layer.rect.foreach_set(rgba)
            self.end_result(blender_result)

        def _write_result_artifact(
            self,
            render_result: RenderResult,
            request: RenderRequest,
            composition: ovrtx_scene_composition.OvrtxSceneComposition | None = None,
            *,
            scene: Any | None = None,
        ) -> None:
            path = _result_artifact_path_from_environment()
            if not path:
                return
            artifact = {
                "schema_version": 1,
                "artifact_id": "ovrtx-render-result",
                "status": "pass",
                "width": render_result.width,
                "height": render_result.height,
                "min_samples": request.min_samples,
                "max_samples": request.max_samples,
                "completed_samples": render_result.completed_samples,
                "simulation_time_ns": render_result.simulation_time_ns,
                "sensor_paths": list(request.sensor_paths),
                "selected_sensor_paths": list(request.selected_sensor_paths),
                "input_usd_path": request.input_usd_path,
                "scene_generation": (
                    scene_generation_sessions.diagnostics_for_scene(
                        scene,
                        input_usd_path=request.input_usd_path,
                    )
                    if scene is not None
                    else {"status": "unavailable", "scene_uid": 0}
                ),
                # Provenance (task05-04): which scene input fed this render
                # and, on the live route, the exact authored generation
                # identity (ADR 0014 content digest + generation number)
                # stamped by the current scene generation.
                # The viewport artifact records the same keys, making
                # viewport/F12 same-generation parity an artifact-level
                # equality check.
                "input_source": _request_input_source(request),
                "authored_generation_digest": request.authored_generation_digest,
                "authored_generation": request.authored_generation,
                "render_product_path": request.render_product_path,
                "addon_path": str(Path(__file__).resolve().parent),
                "color_presentation": color_presentation.diagnostics_from_request_result(request, render_result),
                "blender_postprocess_enabled": FINAL_RENDER_USE_POSTPROCESS,
                "render_var": render_result.render_var,
                "frame_format": render_result.frame_format,
                "frame_color_mode": render_result.frame_color_mode,
                "written_at_ns": time.time_ns(),
            }
            if composition is not None:
                artifact["ovrtx_scene_composition"] = ovrtx_scene_composition.diagnostics(
                    composition,
                    request=request,
                )
            try:
                if render_result.linear_rgba16f:
                    hdr_path = Path(path).with_suffix(".HdrColor.rgba16f")
                    hdr_path.write_bytes(render_result.linear_rgba16f)
                    artifact["hdr_artifact"] = {
                        "path": str(hdr_path),
                        "size_bytes": len(render_result.linear_rgba16f),
                        "sha256": hashlib.sha256(render_result.linear_rgba16f).hexdigest(),
                        "encoding": "rgba16f",
                        "color_space": "scene_linear",
                    }
                Path(path).write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            except OSError as exc:
                self._report_final({"ERROR"}, f"Could not write render artifact: {exc}")

        def view_update(self, context: Any, depsgraph: Any) -> None:
            """Translate + reconcile on the main thread, hand off, return.

            No service RPCs, no sleeps, no waiting for activation (spec
            main-thread contract). Signal translation and authoring-source
            reconciliation read Blender data and therefore stay here; the
            render thread receives only add-on-owned payloads.
            """

            if (
                _EXACT_STAGE_CONFIGURATION is None
                and scene_generation_sessions.is_authoring(depsgraph.scene)
            ):
                return
            presentation = viewport_presentation.apply_native_fallback_for_context(context)
            if self._use_native_viewport_fallback(presentation):
                self._enter_native_viewport_fallback(presentation)
                return
            self._viewport_presentation = dict(presentation)
            try:
                request = _render_callback_adapter(_engine_signal_id(self)).view_update(context, depsgraph)
                self._begin_async_viewport_session(request, depsgraph.scene, depsgraph)
                self._write_viewport_camera_snapshot(request, {}, {})
            except (BlenderSignalTranslationError, RenderClientError) as exc:
                self._report_viewport_error(exc)
                return
            self.update_stats("ovrtx", self._viewport_status_message())
            self.tag_redraw()

        def view_draw(self, context: Any, depsgraph: Any) -> None:
            """Snapshot to the mailbox, present the latest published frame.

            Remaining main-thread phases (spec success criterion 2): request
            translation, mailbox write, latest-frame read, texture upload,
            draw. Rendering, scheduler ticks, and camera value updates run
            on the per-session render thread (``LatestViewRenderLoop``).
            """

            if (
                _EXACT_STAGE_CONFIGURATION is None
                and scene_generation_sessions.is_authoring(depsgraph.scene)
            ):
                return
            view_draw_started_at_ns = time.time_ns()
            view_draw_started_ns = time.perf_counter_ns()
            # The redraw-request stamp normally comes from the publication
            # signaler (stamped on the render thread at publish time, so the
            # publish→draw span includes timer dispatch latency). Measurement
            # scripts may override it via the attribute for injected input
            # events; both slots are consumed each draw.
            publication_redraw_mark = self._redraw_signaler.consume_request_mark()
            redraw_requested_mark = self._redraw_requested_monotonic_ns
            self._redraw_requested_monotonic_ns = None
            if redraw_requested_mark is None:
                redraw_requested_mark = publication_redraw_mark
            span_boundaries: dict[str, int | None] = {
                "redraw_requested_monotonic_ns": redraw_requested_mark,
                "render_callback_started_monotonic_ns": view_draw_started_ns,
            }
            timings: dict[str, float] = {phase: 0.0 for phase in viewport_profile.TIMING_PHASES}
            presentation_started_ns = time.perf_counter_ns()
            presentation = viewport_presentation.apply_native_fallback_for_context(context)
            timings["presentation_ms"] = (
                time.perf_counter_ns() - presentation_started_ns
            ) / 1_000_000.0
            if self._use_native_viewport_fallback(presentation):
                self._enter_native_viewport_fallback(presentation)
                return
            self._viewport_presentation = dict(presentation)
            request_started_ns = time.perf_counter_ns()
            try:
                callback_adapter = _render_callback_adapter(_engine_signal_id(self))
                request = callback_adapter.view_draw(context, depsgraph)
                timings["request_ms"] = (time.perf_counter_ns() - request_started_ns) / 1_000_000.0
                translation_timings_snapshot = getattr(
                    callback_adapter, "_translation_timings_snapshot", None
                )
                if callable(translation_timings_snapshot):
                    request_timings = translation_timings_snapshot()
                    timings["request_translation_ms"] = float(
                        request_timings.get("total_ms", 0.0)
                    )
                    for target, source in (
                        ("request_scene_inputs_ms", "scene_inputs_ms"),
                        ("request_camera_ms", "camera_ms"),
                        ("request_runtime_inputs_ms", "runtime_inputs_ms"),
                        ("request_runtime_state_ms", "runtime_state_ms"),
                        ("request_runtime_defaults_ms", "runtime_defaults_ms"),
                        (
                            "request_native_client_preflight_ms",
                            "native_client_preflight_ms",
                        ),
                        ("request_material_ms", "material_ms"),
                        ("request_build_ms", "request_build_ms"),
                    ):
                        timings[target] = float(
                            request_timings.get(source, 0.0)
                        )
                    timings["request_adapter_overhead_ms"] = max(
                        0.0,
                        timings["request_ms"] - timings["request_translation_ms"],
                    )
                # Session work stays off the steady-state draw path: only a
                # missing or dead render thread triggers the main-thread
                # reconcile + handoff here (first draw before a
                # view_update). Session ensure/replace failures are retried
                # on the thread by the loop's lifecycle policy (task02-06),
                # never by the draw callback.
                if self._render_loop_needs_start():
                    self._begin_async_viewport_session(request, getattr(depsgraph, "scene", None), depsgraph)
                else:
                    self._note_translated_request(request)
                snapshot = self._write_viewport_camera_snapshot(request, timings, span_boundaries)
            except (BlenderSignalTranslationError, RenderClientError) as exc:
                self._report_viewport_error(exc)
                return

            frame = self._frame_slot.peek_latest()
            if frame is None:
                # First frame: nothing published yet for this session (the
                # publication index resets with the session), so keep the
                # existing background and show the loading lifecycle status
                # instead of stale or garbage content. The first publication
                # schedules the redraw that presents it (task02-05) — no
                # polling from the draw path.
                status_message = self._viewport_status_message()
                self._report_lifecycle_transition(status_message)
                self.update_stats("ovrtx", status_message)
                return
            if frame.status == viewport_handoff.FRAME_STATUS_FAILED:
                self._report_viewport_error(
                    RenderClientError(frame.detail or "Viewport render failed")
                )
                return
            if frame.status == viewport_handoff.FRAME_STATUS_RESYNCING:
                # task02-06 publishes this state around session replacement;
                # keep presenting the last frame we drew. Progress arrives as
                # further publications, each scheduling its own redraw.
                self._present_cached_frame(context)
                self._report_lifecycle_transition(frame.detail or "Re-syncing scene")
                self.update_stats("ovrtx", frame.detail or "Re-syncing scene")
                return

            render_result = frame.render_result
            new_publication = frame.publication_index != self._presented_publication_index
            self._current_result = render_result
            # Publication index is the texture identity: an unchanged index
            # reuses the cached texture, a new one takes the update path.
            self._snapshot_index = frame.publication_index
            self._merge_thread_timing_marks(frame, timings, span_boundaries, rendered=new_publication)
            rgba_available_monotonic_ns = frame.published_monotonic_ns or time.perf_counter_ns()
            texture_started_ns = time.perf_counter_ns()
            span_boundaries["texture_upload_started_monotonic_ns"] = texture_started_ns
            texture = self._upload_viewport_texture(render_result)
            texture_completed_ns = time.perf_counter_ns()
            span_boundaries["texture_upload_completed_monotonic_ns"] = texture_completed_ns
            timings["texture_upload_ms"] = (texture_completed_ns - texture_started_ns) / 1_000_000.0
            self._texture_upload["texture_upload_ms"] = timings["texture_upload_ms"]
            draw_started_ns = time.perf_counter_ns()
            span_boundaries["viewport_draw_started_monotonic_ns"] = draw_started_ns
            self._draw_viewport_texture(
                context,
                texture,
                render_result,
                getattr(depsgraph, "scene", None),
            )
            draw_completed_ns = time.perf_counter_ns()
            span_boundaries["viewport_draw_completed_monotonic_ns"] = draw_completed_ns
            timings["viewport_texture_draw_ms"] = (draw_completed_ns - draw_started_ns) / 1_000_000.0
            # Cross-thread spans (task02-09): render-thread publication →
            # this presenting draw. Only a *new* publication measures
            # presentation latency (re-presenting a cached frame does not);
            # both ends are ``perf_counter_ns`` (the slot stamps
            # ``published_monotonic_ns`` with the same clock).
            published_ns = int(frame.published_monotonic_ns or 0)
            if published_ns:
                span_boundaries["frame_published_monotonic_ns"] = published_ns
                if new_publication:
                    if view_draw_started_ns >= published_ns:
                        timings["publish_to_redraw_ms"] = (
                            view_draw_started_ns - published_ns
                        ) / 1_000_000.0
                    if draw_completed_ns >= published_ns:
                        timings["publish_to_draw_ms"] = (
                            draw_completed_ns - published_ns
                        ) / 1_000_000.0

            loop = self._render_loop
            tick_result = loop.last_tick_result if loop is not None else None
            # No tick absorption here (task02-07): pose mirroring and
            # playback-lock transitions flow render thread →
            # _handoff_runtime_tick_result → main-thread absorb timer,
            # independent of draws. This advisory read only feeds the
            # artifact's "composition running" flag.
            composition_running = (
                bool(tick_result.should_request_redraw) if tick_result is not None else False
            )
            if loop is not None:
                loop_diagnostics = loop.diagnostics()
                self._viewport_camera_update_count = int(
                    loop_diagnostics.get("camera_update_count", self._viewport_camera_update_count)
                )
                self._viewport_camera_controls_mode = str(
                    loop_diagnostics.get("camera_controls_mode", self._viewport_camera_controls_mode)
                )
            if new_publication:
                if self._viewport_crash_marker.get("marker_active"):
                    self._viewport_crash_marker = session_lifecycle.clear_crash_marker()
                self._viewport_lifecycle_phase = ""
                self._viewport_start_failure_count = 0
                # Publication count is the thread-side render count proxy
                # until task02-09 embeds the loop diagnostics directly.
                self._render_count = frame.publication_index
            self._presented_publication_index = frame.publication_index
            self._presented_frame = frame

            timings["viewport_callback_ms"] = (time.perf_counter_ns() - view_draw_started_ns) / 1_000_000.0
            timings["callback_overhead_ms"] = max(
                0.0,
                timings["viewport_callback_ms"]
                - sum(
                    timings[phase]
                    for phase in (
                        "presentation_ms",
                        "request_ms",
                        "ensure_session_ms",
                        "acquire_result_ms",
                        "texture_upload_ms",
                        "viewport_texture_draw_ms",
                    )
                ),
            )
            self._record_profile(
                render_result,
                timings,
                new_publication,
                started_at_ns=view_draw_started_at_ns,
                ended_at_ns=time.time_ns(),
                started_monotonic_ns=view_draw_started_ns,
                rgba_available_monotonic_ns=rgba_available_monotonic_ns,
                ended_monotonic_ns=time.perf_counter_ns(),
                span_boundaries=span_boundaries,
            )
            max_samples = int(request.max_samples)
            completed_samples = int(frame.completed_samples)
            still_refining = render_requests.viewport_sampling_due(
                completed_samples, max_samples
            )
            frame_is_for_written_view = snapshot is None or frame.snapshot_key == snapshot.key
            self._write_viewport_artifact(
                running=still_refining or composition_running or not frame_is_for_written_view
            )
            self._viewport_reported_error = ""
            # One Info-panel milestone when frames start presenting (or
            # resume after a resync); the per-draw sample progress stays in
            # the header stats only.
            self._report_lifecycle_transition("OVRTX viewport live")
            sample_progress = (
                f"{completed_samples}/continuous"
                if max_samples == 0
                else f"{completed_samples}/{max_samples}"
            )
            self.update_stats("ovrtx", f"Viewport samples {sample_progress}")
            # No draw-tail redraw polling: refinement progress is presented
            # by the publication-driven redraw timer (task02-05) — every
            # published step schedules exactly one coalesced redraw.

        def _begin_async_viewport_session(
            self,
            request: RenderRequest,
            scene: Any | None = None,
            depsgraph: Any | None = None,
        ) -> None:
            """Main-thread half of viewport session startup (no service RPCs).

            Reconciles the authoring source (Blender reads stay on this
            thread) and hands the latest-view render loop — with its
            session lifecycle hooks (task02-06: startup ensure, replacement
            triggers, resize debounce, resync presentation, retry policy) —
            to the per-session render thread. Never waits for activation.
            """

            authored = bool(
                scene is not None
                and scene_generation_sessions.owns_request(scene, request)
            )
            if authored:
                if scene_generation_sessions.runtime_reuse_blocked(scene):
                    raise RenderClientError(
                        "Viewport runtime cannot restart after a teardown deadline was exceeded"
                    )
            elif _DIRECT_VIEWPORT_REUSE_BLOCKED:
                raise RenderClientError(
                    "Viewport runtime cannot restart after a teardown deadline was exceeded"
                )
            self._ensure_render_loop(request, scene=scene, authored=authored)

        def _prepare_direct_session(self, request: RenderRequest) -> None:
            if self._ovrtx_session_controller is None:
                self._ovrtx_session_controller = OvrtxSessionController()
            self._viewport_request = request

        def _note_translated_request(self, request: RenderRequest) -> None:
            """Record the newest main-thread translation for the render loop."""

            scene = self._viewport_scene
            authored = bool(
                scene is not None
                and scene_generation_sessions.owns_request(scene, request)
            )
            runtime = _ENGINE_RUNTIMES.get(id(self))
            if runtime is not None and bool(runtime.get("authored")) != authored:
                if not self._end_viewport_session(
                    ViewportSessionEndReason.SESSION_REPLACED
                ):
                    raise RenderClientError(
                        "Viewport route cannot change while teardown is unconfirmed"
                    )
                self._begin_async_viewport_session(request, scene=scene)
                return
            self._viewport_request = request

        def _render_loop_needs_start(self) -> bool:
            render_thread = self._render_thread
            if render_thread is None:
                return True
            return render_thread.status() != viewport_render_thread.STATUS_RUNNING

        def _ensure_render_loop(
            self,
            request: RenderRequest,
            *,
            scene: Any | None,
            authored: bool,
        ) -> None:
            render_thread = self._render_thread
            if render_thread is not None:
                if render_thread.status() == viewport_render_thread.STATUS_RUNNING:
                    # A running loop owns all session lifecycle from here:
                    # generation changes, reuse blockers, resize, and
                    # ensure-failure retries are evaluated on the thread
                    # from mailbox/generation state (task02-06) — the
                    # callbacks restart a healthy thread only when its
                    # authoritative ownership route changes.
                    runtime = _ENGINE_RUNTIMES.get(id(self))
                    if runtime is None or bool(runtime.get("authored")) == authored:
                        self._viewport_scene = scene
                        self._viewport_request = request
                        return
                    if not self._end_viewport_session(
                        ViewportSessionEndReason.SESSION_REPLACED
                    ):
                        raise RenderClientError(
                            "Viewport route cannot change while teardown is unconfirmed"
                        )
                    render_thread = None
                if render_thread is not None:
                    self._stop_render_loop()
            self._viewport_scene = scene
            runtime = _ENGINE_RUNTIMES.get(id(self))
            if runtime is not None:
                # This pre-thread decision stays authoritative for the
                # complete lifetime of the thread created below.
                runtime["authored"] = authored
                runtime["scene"] = self._viewport_scene
                runtime["stop_confirmed"] = None
            generation_runtime = None
            if authored:
                generation_runtime = scene_generation_sessions.runtime_for_viewport(
                    scene,
                    viewport_id=_engine_signal_id(self),
                )
                self._viewport_generation_runtime = generation_runtime
                self._ovrtx_session_controller = generation_runtime.ovrtx.controller
                self._runtime_scheduler = generation_runtime.scheduler
                self._physics_playback_lock = generation_runtime.playback_lock
                if runtime is not None:
                    runtime["generation_runtime"] = generation_runtime
            else:
                self._prepare_direct_session(request)
            controller = self._ovrtx_session_controller
            if controller is None:
                raise RenderClientError("No active OVRTX session controller")
            scheduler = (
                generation_runtime.scheduler
                if generation_runtime is not None
                else self._ensure_runtime_scheduler()
            )
            replacing_session = self._viewport_session_started_ns > 0
            self._camera_mailbox = viewport_handoff.CameraRequestMailbox()
            self._frame_slot = viewport_handoff.LatestFrameSlot()
            self._presented_publication_index = 0
            self._presented_frame = None
            self._written_snapshot_key = None
            self._viewport_session_outputs_written = False
            # Fresh session, fresh teardown ledger: a leak recorded by a
            # previous session on this engine must not bleed into the new
            # session's artifacts (the leaked session already wrote its
            # outputs with the leak recorded).
            self._viewport_cleanup_diagnostics = {"status": "not_requested"}
            # Same rule for the thread-aware diagnostics (task02-09): the
            # previous session's final thread/loop capture and absorb
            # counters were already written with its outputs.
            self._thread_model_final = {}
            with self._tick_absorb_lock:
                self._tick_absorb_counters = _new_tick_absorb_counters()
            self._current_result = None
            self._snapshot_index = 0
            self._viewport_texture = None
            self._viewport_texture_size = None
            self._texture_snapshot_index = 0
            self._texture_upload = {}
            self._viewport_last_operator_view = {}
            # Session-start bookkeeping (started_ns stamp, counter resets)
            # happens on the thread after a successful ensure: a failed first
            # activation must leave the next attempt in the loading (not
            # resyncing) phase.
            self._viewport_lifecycle_phase = (
                session_lifecycle.PHASE_RESYNCING
                if replacing_session
                else session_lifecycle.PHASE_LOADING
            )
            self._viewport_crash_marker = session_lifecycle.write_crash_marker(
                phase=self._viewport_lifecycle_phase,
                scene_name=str(request.input_usd_path or ""),
            )
            # Weak closures: the loop and the queued command must not keep
            # the engine alive (Blender frees engines by dropping the last
            # reference; __del__ is what stops this thread).
            engine_ref = weakref.ref(self)

            def _request_for_snapshot(snapshot: viewport_handoff.ViewSnapshot) -> RenderRequest:
                engine = engine_ref()
                if engine is None:
                    raise RenderClientError("Viewport engine was released")
                return engine._render_request_for_snapshot(snapshot)

            mailbox = self._camera_mailbox
            # Publishers go through the signaling wrapper so every
            # publication (frame, resync, failure state) schedules exactly
            # one coalesced main-thread redraw (task02-05). The engine's
            # read path (peek_latest) keeps using the raw slot.
            frame_slot = viewport_render_thread.RedrawSignalingFrameSlot(
                self._frame_slot,
                self._redraw_signaler.signal,
            )
            # Discard any request stamp left over from a previous session so
            # the new session's first publish→draw span starts clean.
            self._redraw_signaler.consume_request_mark()

            # Session lifecycle hooks (task02-06): the loop performs the
            # session ensure/replace on the render thread, triggered by
            # mailbox/generation state. All hooks resolve the engine weakly
            # (same rule as _request_for_snapshot).
            def _ensure_session(loop_request: RenderRequest) -> RuntimeScheduler:
                engine = engine_ref()
                if engine is None:
                    raise RenderClientError("Viewport engine was released")
                return engine._thread_ensure_session(loop_request)

            def _replacement_reason(loop_request: RenderRequest) -> str:
                engine = engine_ref()
                if engine is None:
                    return ""
                return engine._thread_replacement_reason(loop_request)

            def _retry_allowed() -> bool:
                engine = engine_ref()
                if engine is None:
                    return False
                return session_lifecycle.should_auto_retry(
                    engine._viewport_start_failure_count
                )

            # Pose-mirror handoff (task02-07): every successful tick's
            # result crosses to the main-thread absorb timer. Data-only on
            # this side; a released engine makes the handoff a no-op (the
            # loop is being stopped by __del__ at that point).
            # ``loop`` binds late (assigned below, read at call time): the
            # handoff carries its source loop so a leaked (join-timeout)
            # thread that resumes a tick after teardown cannot mirror a
            # stale pose set into a replacement session (task02-08; the
            # 02-07 leaked-handoff follow-up).
            def _tick_result_sink(result: RuntimeTickResult, loop_request: RenderRequest) -> None:
                engine = engine_ref()
                if engine is None:
                    return
                engine._handoff_runtime_tick_result(
                    result, loop_request, source_loop=loop
                )

            def _controller_provider() -> Any:
                # Resolve the engine's current controller each step: the
                # authored ensure can swap it after the loop starts, and the
                # loop must follow the swap rather than render a torn-down
                # session. None (engine released/cleared) becomes a retryable
                # "No active OVRTX session".
                engine = engine_ref()
                return None if engine is None else engine._ovrtx_session_controller

            loop = viewport_render_thread.LatestViewRenderLoop(
                mailbox=mailbox,
                frame_slot=frame_slot,
                controller_provider=_controller_provider,
                scheduler=scheduler,
                owns_scheduler=not authored,
                request_for_snapshot=_request_for_snapshot,
                lifecycle=viewport_render_thread.SessionLifecycleHooks(
                    ensure_session=_ensure_session,
                    replacement_reason=_replacement_reason,
                    retry_allowed=_retry_allowed,
                ),
                tick_result_sink=_tick_result_sink,
            )
            self._render_loop = loop
            new_thread = viewport_render_thread.ViewportRenderThread(
                _engine_signal_id(self),
                join_timeout_seconds=VIEWPORT_SESSION_TEARDOWN_TIMEOUT_SECONDS,
            )
            self._render_thread = new_thread
            # Mirror the teardown-critical handles into the sidecar: after
            # the RNA dies these are unreachable through the wrapper.
            runtime = _ENGINE_RUNTIMES.get(id(self))
            if runtime is not None:
                runtime["render_loop"] = loop
                runtime["render_thread"] = new_thread
                runtime["scene"] = self._viewport_scene
                teardown, teardown_state = self._runtime_teardown_state()
                runtime["teardown"] = teardown
                runtime["teardown_state"] = teardown_state

            try:
                new_thread.start()
                # The loop is the thread's long-lived submit command; its
                # first adopted snapshot triggers the startup ensure on the
                # thread (session work never runs from view_draw).
                new_thread.submit(loop.run, label="latest-view-loop")
            except Exception as exc:
                self._stop_render_loop()
                raise RenderClientError(
                    f"Viewport render thread could not start: {type(exc).__name__}: {exc}"
                ) from exc

        def _thread_ensure_session(self, request: RenderRequest) -> RuntimeScheduler:
            """Render-thread session ensure/replace (startup and resync).

            Called by the loop's session lifecycle (task02-06) for the
            startup ensure, ensure-failure retries, and background
            replacements. Stamps the lifecycle phase and crash marker on
            the thread with the existing vocabulary
            (``PHASE_LOADING``/``PHASE_RESYNCING``); marker file I/O has a
            single writer once the loop runs — this thread. The ADR 0014
            activation ordering (OVPhysX before OVRTX inside
            ``AuthoringSession.activate``, break-before-make in the
            registry slot and ``controller.ensure``) is unchanged.
            """

            self._ensure_viewport_session(request, scene=self._viewport_scene)
            return self._ensure_runtime_scheduler()

        def _thread_replacement_reason(self, request: RenderRequest) -> str:
            """Evaluate the session replacement triggers (render thread).

            Authored generation change first (topology edits reconciled on
            the main thread produce a generation the thread has not
            activated), then the ``reuse_decision`` blockers via the
            controller's read-only probe (output size, composition digest,
            declared sensors, camera prim, pose-source downgrade).
            """

            controller = self._ovrtx_session_controller
            if controller is None:
                return ""
            return controller.would_replace(request)

        def _render_request_for_snapshot(
            self,
            snapshot: viewport_handoff.ViewSnapshot,
        ) -> RenderRequest:
            """Overlay a mailbox snapshot onto the newest translated request.

            Called on the render thread. The base request carries session
            identity (input path, worker command, sensors); the snapshot
            carries the per-view fields the mailbox transports.
            """

            base = self._viewport_request
            if base is None:
                raise RenderClientError("No translated viewport request is available")
            return viewport_handoff.request_from_snapshot(base, snapshot)

        def _write_viewport_camera_snapshot(
            self,
            request: RenderRequest,
            timings: dict[str, float],
            span_boundaries: Mapping[str, int | None],
        ) -> viewport_handoff.ViewSnapshot:
            """Write the newest camera/request snapshot to the mailbox."""

            marks: dict[str, int] = {}
            redraw_mark = span_boundaries.get("redraw_requested_monotonic_ns")
            if redraw_mark:
                marks["redraw_requested_monotonic_ns"] = int(redraw_mark)
            snapshot = viewport_handoff.snapshot_from_render_request(
                request,
                timing_marks=marks,
            )
            if snapshot.key != self._written_snapshot_key:
                self._written_snapshot_key = snapshot.key
                self._viewport_snapshot_count += 1
                timings["snapshot_changed"] = 1.0
            self._camera_mailbox.write(snapshot)
            return snapshot

        def _merge_thread_timing_marks(
            self,
            frame: viewport_handoff.FrameState,
            timings: dict[str, float],
            span_boundaries: dict[str, int | None],
            *,
            rendered: bool,
        ) -> None:
            """Attribute thread-side spans to the profile record.

            ``render_ms``/``composition_update_ms`` come from the published
            frame's timing marks (the render thread's spans), never from
            main-thread wall time — spec success criterion 2 attribution.
            """

            marks = frame.timing_marks
            for name in (
                "snapshot_written_monotonic_ns",
                "runtime_update_started_monotonic_ns",
                "runtime_update_completed_monotonic_ns",
                "render_call_started_monotonic_ns",
                "render_call_completed_monotonic_ns",
            ):
                value = marks.get(name)
                if value:
                    span_boundaries[name] = int(value)
            if not rendered:
                return
            render_started = marks.get("render_call_started_monotonic_ns")
            render_completed = marks.get("render_call_completed_monotonic_ns")
            if render_started and render_completed:
                timings["render_ms"] = (int(render_completed) - int(render_started)) / 1_000_000.0
            update_started = marks.get("runtime_update_started_monotonic_ns")
            update_completed = marks.get("runtime_update_completed_monotonic_ns")
            if update_started and update_completed:
                timings["composition_update_ms"] = (
                    int(update_completed) - int(update_started)
                ) / 1_000_000.0
            # Cross-thread span (task02-09): main-thread mailbox write →
            # render thread picked the snapshot up and started rendering.
            snapshot_written = marks.get("snapshot_written_monotonic_ns")
            if (
                snapshot_written
                and render_started
                and int(render_started) >= int(snapshot_written)
            ):
                timings["snapshot_to_render_start_ms"] = (
                    int(render_started) - int(snapshot_written)
                ) / 1_000_000.0

        def _present_cached_frame(self, context: Any) -> None:
            """Redraw the last presented frame (resync presentation)."""

            presented = self._presented_frame
            if presented is None or presented.render_result is None:
                return
            self._snapshot_index = presented.publication_index
            texture = self._upload_viewport_texture(presented.render_result)
            self._draw_viewport_texture(context, texture, presented.render_result)

        def _viewport_status_message(self) -> str:
            latest = self._frame_slot.peek_latest()
            if latest is not None and latest.status == viewport_handoff.FRAME_STATUS_FRAME:
                return "Viewport session ready"
            phase = self._viewport_lifecycle_phase
            status = session_lifecycle.derive_status(
                engine_active=True,
                ready=False,
                busy=bool(phase),
                phase=phase,
                failure_count=self._viewport_start_failure_count,
                last_error=self._viewport_reported_error,
            )
            return str(status.get("label", "Starting OVRTX"))

        def _capture_thread_model(self, loop: Any | None, render_thread: Any | None) -> None:
            """Preserve final thread/loop diagnostics for the ending session.

            Called by ``_stop_render_loop`` after the stop/join attempt so
            the session-end artifact (written after the live references are
            cleared) still carries the thread identity, loop counters, and
            per-iteration records (task02-09). After a confirmed join the
            loop's state is final (single writer has exited); for leaked
            threads the snapshot is advisory.
            """

            if loop is None and render_thread is None:
                return
            capture: dict[str, Any] = {}
            if render_thread is not None:
                try:
                    capture["render_thread"] = render_thread.diagnostics()
                except Exception:
                    pass
            if loop is not None:
                try:
                    capture["render_loop"] = loop.diagnostics()
                except Exception:
                    pass
                try:
                    capture["render_loop_records"] = loop.iteration_records()
                except Exception:
                    pass
            self._thread_model_final = capture

        def _thread_model_state(self) -> dict[str, Any]:
            """Thread/loop diagnostics for the artifact: live, else final.

            A running session reads the live thread and loop; after
            ``_stop_render_loop`` the capture taken at stop time is the
            source (the aggregation itself always runs here, on the main
            thread, at artifact-write time).
            """

            state: dict[str, Any] = dict(self._thread_model_final)
            render_thread = self._render_thread
            loop = self._render_loop
            if render_thread is not None:
                try:
                    state["render_thread"] = render_thread.diagnostics()
                except Exception:
                    pass
            if loop is not None:
                try:
                    state["render_loop"] = loop.diagnostics()
                except Exception:
                    pass
                try:
                    state["render_loop_records"] = loop.iteration_records()
                except Exception:
                    pass
            state.setdefault("render_thread", {})
            state.setdefault("render_loop", {})
            state.setdefault("render_loop_records", [])
            return state

        def _stop_render_loop(self, *, teardown: Any | None = None) -> dict[str, Any]:
            """Stop the loop, run owner-thread teardown, and join once.

            ``teardown`` is queued behind the exiting loop and ahead of the
            stop sentinel. The interactive thread owns a single 600-second
            join attempt; exceeding it permanently disables runtime reuse.
            """

            loop = self._render_loop
            render_thread = self._render_thread
            if loop is not None:
                loop.request_stop()
            if render_thread is None:
                runtime = _ENGINE_RUNTIMES.get(id(self))
                if runtime is not None and runtime.get("stop_confirmed") is False:
                    return {
                        "status": "teardown_unconfirmed",
                        "joined": False,
                        "leaked_thread": True,
                        "join_timeout_seconds": VIEWPORT_SESSION_TEARDOWN_TIMEOUT_SECONDS,
                    }
                self._capture_thread_model(loop, render_thread)
                self._render_loop = None
                return {"status": "no_thread", "joined": True, "leaked_thread": False}
            if teardown is not None:
                try:
                    render_thread.submit(teardown, label="session-teardown")
                except Exception:
                    # Dead or already-stopping thread rejects the command;
                    # the caller runs the teardown on the main thread after
                    # the confirmed join hands RPC ownership back.
                    pass
            try:
                outcome = render_thread.stop()
            except Exception:
                outcome = {
                    "status": "stop_failed",
                    "joined": False,
                    "leaked_thread": False,
                    "join_timeout_seconds": VIEWPORT_SESSION_TEARDOWN_TIMEOUT_SECONDS,
                    "failure": "render thread stop raised",
                }
            # Preserve the ending session's thread/loop diagnostics for the
            # artifact written after the references are cleared (task02-09).
            self._capture_thread_model(loop, render_thread)
            # The tick handoff belongs to the loop that produced it: drop
            # any pending absorb once the loop is stopped (its pose set and
            # owning generation are stale for whatever comes next; a
            # still-registered absorb timer no-ops on the empty handoff).
            with self._tick_absorb_lock:
                self._pending_tick_absorb = None
            if not outcome.get("joined", False):
                self._record_unconfirmed_teardown(render_thread, outcome)
            self._render_loop = None
            self._render_thread = None
            runtime = _ENGINE_RUNTIMES.get(id(self))
            if runtime is not None:
                runtime["stop_confirmed"] = bool(outcome.get("joined", False))
                if runtime["stop_confirmed"]:
                    runtime["render_loop"] = None
                    runtime["render_thread"] = None
                    runtime["teardown"] = None
                    runtime["teardown_state"] = None
                    runtime["generation_runtime"] = None
                    runtime["scene"] = None
            return outcome

        def _record_unconfirmed_teardown(
            self,
            render_thread: Any,
            outcome: Mapping[str, Any],
        ) -> None:
            """Report one terminal owner-thread teardown failure."""

            thread_name = str(getattr(render_thread, "name", ""))
            runtime = _sidecar_generation_runtime(_ENGINE_RUNTIMES.get(id(self)))
            _fail_closed_runtime_reuse(runtime)
            # Merged into session-lifecycle diagnostics: the artifact's
            # ``session_lifecycle.cleanup`` block carries this dict. The
            # stop outcome's own ``status`` (the thread status) is kept
            # under ``thread_status``.
            deadline_exceeded = bool(outcome.get("leaked_thread", False))
            self._viewport_cleanup_diagnostics = {
                **outcome,
                "thread_status": str(outcome.get("status", "")),
                "status": (
                    "teardown_deadline_exceeded"
                    if deadline_exceeded
                    else "teardown_stop_failed"
                ),
                "thread_name": thread_name,
            }
            try:
                detail = (
                    f" within {outcome.get('join_timeout_seconds')}s"
                    if deadline_exceeded
                    else ""
                )
                message = (
                    "[ovrtx_blender_example] defect: viewport session "
                    f"{thread_name!r} did not confirm owner-thread teardown{detail}; "
                    "runtime reuse disabled"
                )
                print(message)
                # stdout already carries the defect; also surface it in the
                # Info window.
                user_messages.report_warning(message, dedup=False, to_console=False)
            except Exception:
                pass

        def _runtime_teardown_state(self) -> tuple[Any, dict[str, Any]]:
            """Session teardown RPCs as a render-thread command (task02-08).

            Captures strong references — never the engine — so
            ``__del__``-driven teardown works while the engine object is
            being finalized, and a timed-out thread that later resumes can
            still run the queued command. Every step is exception-guarded:
            the command is ``submit``-shaped and must not fail the thread.
            """

            scheduler = self._runtime_scheduler
            controller = self._ovrtx_session_controller
            prepared = self._viewport_generation_runtime
            scene = self._viewport_scene
            state: dict[str, Any] = {"ran": False, "errors": []}

            def _teardown() -> None:
                state["ran"] = True
                if prepared is None and scheduler is not None:
                    # Normally a no-op: the loop's ``finally`` already shut
                    # the scheduler down on this thread (task02-07). Kept
                    # for loops that never ran; shutdown is idempotent and
                    # non-terminal.
                    try:
                        scheduler.shutdown()
                    except Exception as exc:
                        state["errors"].append(
                            f"scheduler_shutdown: {type(exc).__name__}: {exc}"
                        )
                controller_runtime = (
                    None
                    if scene is None
                    else scene_generation_sessions.active_runtime_for_scene(scene)
                )
                controller_is_generation_owned = bool(
                    controller_runtime is not None
                    and getattr(getattr(controller_runtime, "ovrtx", None), "controller", None)
                    is controller
                )
                if (
                    prepared is None
                    and controller is not None
                    and not controller_is_generation_owned
                ):
                    try:
                        adopt = getattr(controller, "adopt_owning_thread", None)
                        if callable(adopt):
                            adopt()
                        controller.shutdown()
                    except Exception as exc:
                        state["errors"].append(
                            f"controller_shutdown: {type(exc).__name__}: {exc}"
                        )

            return _teardown, state

        def _ensure_viewport_session(
            self,
            request: RenderRequest,
            scene: Any | None = None,
            *,
            timings: dict[str, float] | None = None,
        ) -> None:
            # Direct-route (env-override / scene-less compatibility) session
            # ensure. Runs on the render thread as part of the loop's
            # session lifecycle (task02-06): startup, ensure-failure
            # retries, and reuse-blocker replacements all land here for
            # requests without an authoring identity. The authored route is
            # _activate_prepared_authored_session (via
            # _thread_ensure_session). Scene-generation-owned requests
            # (main-thread callers pass ``scene``) activate their
            # generation runtime instead of a plain controller ensure.
            #
            # Runtime services (re)starting is a transient wait, not a failure:
            # raise the typed deferral before any activation (the authored path
            # would otherwise wrap it and lose the type). The loop holds the
            # loading state and retries.
            if runtime_services.owner.diagnostics().get("status") == "starting":
                raise RuntimeServicesPreparingError(
                    "Runtime services are still preparing"
                )
            controller = self._ovrtx_session_controller
            if controller is None:
                controller = OvrtxSessionController()
                self._ovrtx_session_controller = controller
            replacing_session = self._viewport_session_started_ns > 0
            self._viewport_request = request
            self._viewport_lifecycle_phase = (
                session_lifecycle.PHASE_RESYNCING
                if replacing_session
                else session_lifecycle.PHASE_LOADING
            )
            crash_marker_started_ns = time.perf_counter_ns()
            self._viewport_crash_marker = session_lifecycle.write_crash_marker(
                phase=self._viewport_lifecycle_phase,
                scene_name=str(getattr(request, "input_usd_path", "") or ""),
            )
            if timings is not None:
                timings["ensure_crash_marker_write_ms"] = (
                    time.perf_counter_ns() - crash_marker_started_ns
                ) / 1_000_000.0
            try:
                if (
                    scene is not None
                    and scene_generation_sessions.owns_request(scene, request)
                ):
                    runtime = scene_generation_sessions.activate_for_viewport(
                        scene,
                        request,
                        viewport_id=_engine_signal_id(self),
                        wake_hook=self._camera_mailbox.wake,
                        on_generation_settled=self._redraw_signaler.signal,
                        expected_runtime=self._viewport_generation_runtime,
                    )
                    self._viewport_generation_runtime = runtime
                    self._ovrtx_session_controller = runtime.ovrtx.controller
                    controller = runtime.ovrtx.controller
                    self._runtime_scheduler = runtime.scheduler
                    self._runtime_tick_result = runtime.last_activation_update
                    sidecar = _ENGINE_RUNTIMES.get(id(self))
                    if sidecar is not None:
                        teardown, teardown_state = self._runtime_teardown_state()
                        sidecar["teardown"] = teardown
                        sidecar["teardown_state"] = teardown_state
                        sidecar["generation_runtime"] = runtime
                    ensure_result = runtime.ovrtx.last_ensure_result
                else:
                    ensure_result = controller.ensure(request)
            except Exception as exc:
                self._runtime_startup_diagnostics = dict(controller.diagnostics()["startup"])
                self._viewport_log_diagnostics = _logs_from_startup_diagnostics(self._runtime_startup_diagnostics)
                self._viewport_start_failure_count += 1
                self._viewport_lifecycle_phase = ""
                self._current_result = None
                self._viewport_texture = None
                self._viewport_texture_size = None
                self._texture_snapshot_index = -1
                if self._viewport_crash_marker.get("marker_active"):
                    self._viewport_crash_marker = session_lifecycle.clear_crash_marker()
                self._write_viewport_artifact(running=False)
                if isinstance(exc, RenderClientError):
                    raise
                raise RenderClientError(
                    f"Scene generation activation failed: {type(exc).__name__}: {exc}"
                ) from exc
            if timings is not None:
                ensure_timings = controller._ensure_timings_snapshot()
                timings["ensure_controller_ms"] = float(
                    ensure_timings.get("total_ms", 0.0)
                )
                timings["ensure_build_spec_ms"] = float(
                    ensure_timings.get("build_spec_ms", 0.0)
                )
                timings["ensure_reuse_decision_ms"] = float(
                    ensure_timings.get("reuse_decision_ms", 0.0)
                )
                timings["ensure_controller_other_ms"] = float(
                    ensure_timings.get("other_ms", 0.0)
                )
            self._ovrtx_scene_composition = ensure_result.composition
            self._scene_generation_artifact_request = request
            if not ensure_result.session_started:
                self._viewport_lifecycle_phase = ""
                if self._viewport_crash_marker.get("marker_active"):
                    crash_marker_started_ns = time.perf_counter_ns()
                    self._viewport_crash_marker = session_lifecycle.clear_crash_marker()
                    if timings is not None:
                        timings["ensure_crash_marker_clear_ms"] = (
                            time.perf_counter_ns() - crash_marker_started_ns
                        ) / 1_000_000.0
                return
            diagnostics_started_ns = time.perf_counter_ns()
            diagnostics = controller.diagnostics()
            if timings is not None:
                timings["ensure_diagnostics_ms"] = (
                    time.perf_counter_ns() - diagnostics_started_ns
                ) / 1_000_000.0
            self._runtime_startup_diagnostics = dict(diagnostics["startup"])
            self._viewport_log_diagnostics = _logs_from_startup_diagnostics(self._runtime_startup_diagnostics)
            self._viewport_session_outputs_written = False
            self._current_result = None
            self._snapshot_index = 0
            self._viewport_texture = None
            self._viewport_texture_size = None
            self._texture_snapshot_index = 0
            self._texture_upload = {}
            self._viewport_last_operator_view = {}
            if not replacing_session:
                # A fresh session compiles its non-cacheable MaterialX/pipeline
                # shaders on the first frame (minutes for Junk Shop or a cold
                # run); label that window so the still viewport reads as working
                # rather than hung. Clears to "Live" on the first publication.
                self._viewport_lifecycle_phase = session_lifecycle.PHASE_COMPILING
                self._viewport_snapshot_count = 0
                self._viewport_draw_count = 0
                self._render_count = 0
                self._viewport_camera_update_count = 0
                self._viewport_session_started_ns = time.time_ns()
                self._viewport_artifact_recorder.reset()

        def _use_native_viewport_fallback(self, presentation: Mapping[str, Any]) -> bool:
            return (
                presentation.get("presentation_mode")
                == viewport_presentation.NATIVE_VIEWPORT_FALLBACK
            )

        def _enter_native_viewport_fallback(self, presentation: Mapping[str, Any]) -> None:
            presentation = dict(presentation)
            already_in_native_fallback = self._use_native_viewport_fallback(
                self._viewport_presentation
            )
            ended_viewport_session = (
                self._ovrtx_session_controller is not None
                or self._current_result is not None
                or self._runtime_scheduler is not None
                or self._viewport_session_started_ns > 0
                or self._render_thread is not None
            )
            if ended_viewport_session:
                self._viewport_presentation = presentation
                self._end_viewport_session(ViewportSessionEndReason.NATIVE_FALLBACK)
            reason = str(presentation.get("fallback_reason", ""))
            self._viewport_presentation = presentation
            self._viewport_camera_controls_mode = viewport_presentation.NATIVE_VIEWPORT_FALLBACK
            self._current_result = None
            self._texture_snapshot_index = -1
            self._texture_upload = {}
            self._image_artifact = {}
            self._ovrtx_scene_composition = None
            self._scene_generation_artifact_request = None
            self._viewport_last_operator_view = {
                "view_perspective": str(presentation.get("view_perspective", "")),
                "viewport_presentation": presentation,
            }
            if not ended_viewport_session and not already_in_native_fallback:
                self._write_viewport_artifact(running=False)
            fallback_message = f"Native viewport fallback: {reason}"
            self.update_stats("ovrtx", fallback_message)
            # Overlay-only until now: also surface the fallback to console + Info
            # (once per distinct reason, not per draw).
            user_messages.report_warning(
                fallback_message, context=f"native-fallback:{id(self)}"
            )

        def _handoff_runtime_tick_result(
            self,
            result: RuntimeTickResult,
            request: RenderRequest,
            *,
            source_loop: Any | None = None,
        ) -> None:
            """Render-thread half of the tick-result handoff (task02-07).

            Data-only: derives the at-initial-condition fact from the
            snapshot-derived request (timeline fields ride the snapshot —
            no ``bpy.context`` reads here) and schedules the coalesced
            main-thread absorb timer. Every Blender data read/write —
            the scene object scan in ``prepare_runtime_pose_mirror``, the
            pose apply, and the physics playback-lock transitions — happens
            in the timer callback on the main thread.
            ``bpy.app.timers.register`` is the documented thread-safe
            crossing (same pattern as :class:`_PublicationRedrawSignaler`).

            ``source_loop`` (task02-08): the loop the tick came from. A
            handoff whose loop is no longer the engine's current render
            loop is dropped — a leaked (join-timeout) thread that resumes
            after teardown must not mirror its stale pose set / owning
            generation into whatever session runs by then.
            """

            if source_loop is not None and source_loop is not self._render_loop:
                with self._tick_absorb_lock:
                    self._tick_absorb_counters["stale_loop_dropped"] += 1
                return
            at_initial_condition = _request_at_initial_condition(request)
            if (
                not result.physics_pose_set
                and not at_initial_condition
                and not isinstance(result.update.get("update_result"), Mapping)
            ):
                # Nothing to absorb: no poses to mirror, no lock-clear
                # opportunity, no update result to record. Idle ticks must
                # not spam main-thread timers.
                with self._tick_absorb_lock:
                    self._tick_absorb_counters["idle_skipped"] += 1
                return
            with self._tick_absorb_lock:
                self._pending_tick_absorb = (result, at_initial_condition)
                self._tick_absorb_counters["handoffs"] += 1
                if self._tick_absorb_timer_pending:
                    # Coalesced: the pending timer reads the newest handoff.
                    self._tick_absorb_counters["coalesced"] += 1
                    return
                self._tick_absorb_timer_pending = True
            try:
                if bpy is None:
                    raise RuntimeError("bpy is unavailable")
                bpy.app.timers.register(
                    self._absorb_pending_tick_result, first_interval=0.0
                )
            except Exception:
                # Best-effort registration (Blender shutting down, or
                # timers unavailable under test): release the latch so the
                # next tick retries instead of wedging the handoff forever.
                with self._tick_absorb_lock:
                    self._tick_absorb_timer_pending = False
                    self._tick_absorb_counters["registration_failures"] += 1
                return
            with self._tick_absorb_lock:
                self._tick_absorb_counters["timer_registrations"] += 1

        def _absorb_pending_tick_result(self) -> None:
            """Main-thread absorb timer: apply the newest handed-off tick."""

            with self._tick_absorb_lock:
                self._tick_absorb_timer_pending = False
                pending = self._pending_tick_absorb
                self._pending_tick_absorb = None
                if pending is None:
                    self._tick_absorb_counters["absorbs_empty"] += 1
                else:
                    self._tick_absorb_counters["absorbs_applied"] += 1
            if pending is None:
                return None
            result, at_initial_condition = pending
            self._apply_runtime_tick_result_main(
                result, at_initial_condition=at_initial_condition
            )
            return None

        def _tick_absorb_diagnostics(self) -> dict[str, Any]:
            """Tick-absorb handoff counters + pending state (task02-09)."""

            with self._tick_absorb_lock:
                return {
                    **self._tick_absorb_counters,
                    "timer_pending": self._tick_absorb_timer_pending,
                    "pending_handoff": self._pending_tick_absorb is not None,
                }

        def _apply_runtime_tick_result_main(
            self,
            result: RuntimeTickResult,
            *,
            at_initial_condition: bool,
        ) -> None:
            """Absorb a tick result on the main thread.

            Records the update result for edit diagnostics, performs the
            physics playback-lock transitions (main-thread only: ``clear``
            on the initial-condition frame here, ``lock_object`` during
            mirror application in the pose-mirror timer), and prepares the
            runtime pose mirror (Blender scene reads live here, not on the
            render thread).
            """

            self._runtime_tick_result = result
            update_result = result.update.get("update_result")
            if self._interactive_edit_workflow is not None and isinstance(update_result, Mapping):
                self._interactive_edit_workflow.record_update_result(update_result)
            if result.status == RuntimeTickStatus.FAILED:
                return
            lock_was_active = self._physics_playback_lock.is_active()
            if at_initial_condition and lock_was_active:
                self._physics_playback_lock.clear(reason="initial_condition_frame", frame1_cleared=True)
            if operator_state.should_mirror_runtime_poses(
                at_initial_condition=at_initial_condition,
                lock_was_active=lock_was_active,
            ):
                self._pose_mirror = self._prepare_runtime_pose_mirror(
                    result.physics_pose_set,
                    lock_runtime_owned=not at_initial_condition,
                    owning_generation=result.generation,
                )
            else:
                self._pose_mirror = {
                    "enabled": False,
                    "reason": "initial_condition_editable",
                    "source_authority": "OVPhysX",
                    "last_applied": dict(self._applied_pose_mirror),
                }

        def _prepare_runtime_pose_mirror(
            self,
            physics_pose_set: tuple[BodyPose, ...],
            *,
            lock_runtime_owned: bool = True,
            owning_generation: int | None = None,
        ) -> dict[str, Any]:
            """Prepare pending pose-mirror work (main thread only).

            ``operator_state.prepare_runtime_pose_mirror`` scans
            ``scene.objects`` and reads id-properties — Blender data reads
            that must never run on the render thread (task02-07 audit).
            Both prepare and apply therefore live on the timer side of the
            tick handoff.
            """

            pending, diagnostics = operator_state.prepare_runtime_pose_mirror(
                physics_pose_set,
                bpy,
                None,
                lock_runtime_owned=lock_runtime_owned,
                owning_generation=owning_generation,
                last_applied=self._applied_pose_mirror,
            )
            if pending:
                self._pending_pose_mirror = pending
                self._ensure_pose_mirror_timer()
            return dict(diagnostics)

        def _ensure_pose_mirror_timer(self) -> None:
            if bpy is None or self._pose_mirror_timer_registered:
                return
            try:
                bpy.app.timers.register(self._apply_pending_pose_mirror, first_interval=0.0)
            except Exception as exc:
                self._pose_mirror = {
                    "enabled": False,
                    "status": "failed",
                    "reason": f"timer_registration_failed:{type(exc).__name__}: {exc}",
                }
                return
            self._pose_mirror_timer_registered = True

        def _apply_pending_pose_mirror(self) -> None:
            self._pose_mirror_timer_registered = False
            pending = self._pending_pose_mirror
            self._pending_pose_mirror = {}
            applied_diagnostics = operator_state.apply_pending_runtime_pose_mirror(
                bpy,
                pending,
                self._physics_playback_lock,
            )
            if applied_diagnostics is None:
                return None
            if not applied_diagnostics.get("enabled", False):
                self._pose_mirror = applied_diagnostics
                return None
            self._tag_viewport_redraws()
            self._applied_pose_mirror = dict(applied_diagnostics)
            self._pose_mirror = applied_diagnostics
            return None

        def _tag_viewport_redraws(self) -> None:
            try:
                windows = getattr(bpy.context.window_manager, "windows", ())
            except Exception:
                return
            for window in windows:
                screen = getattr(window, "screen", None)
                for area in getattr(screen, "areas", ()):
                    if getattr(area, "type", "") == "VIEW_3D":
                        area.tag_redraw()

        def _ensure_runtime_scheduler(self) -> RuntimeScheduler:
            if self._runtime_scheduler is None:
                self._runtime_scheduler = RuntimeScheduler()
            return self._runtime_scheduler

        def _ensure_interactive_edit_workflow(self) -> InteractiveEditWorkflow:
            if self._interactive_edit_workflow is None:
                self._interactive_edit_workflow = InteractiveEditWorkflow(
                    runtime_scheduler=self._ensure_runtime_scheduler(),
                )
            return self._interactive_edit_workflow

        def _upload_viewport_texture(self, render_result: RenderResult) -> Any:
            import gpu  # type: ignore

            upload = _upload_viewport_texture(
                gpu,
                render_result,
                cached_texture=self._viewport_texture,
                cached_texture_size=self._viewport_texture_size,
                cached_texture_snapshot_index=self._texture_snapshot_index,
                snapshot_index=self._snapshot_index,
                accepts_rgba8=self._viewport_texture_accepts_rgba8,
                cached_texture_color_mode=self._viewport_texture_color_mode,
            )
            self._viewport_texture = upload.texture
            self._viewport_texture_size = upload.texture_size
            self._texture_snapshot_index = self._snapshot_index
            self._viewport_texture_accepts_rgba8 = upload.accepts_rgba8
            self._viewport_texture_color_mode = upload.color_mode
            self._texture_upload = dict(upload.diagnostics)
            return upload.texture

        def _draw_viewport_texture(
            self,
            context: Any,
            texture: Any,
            render_result: RenderResult,
            scene: Any | None = None,
        ) -> None:
            import gpu  # type: ignore
            from gpu_extras.presets import draw_texture_2d  # type: ignore
            from mathutils import Matrix  # type: ignore

            region_width = max(1, int(context.region.width))
            region_height = max(1, int(context.region.height))
            if scene is None:
                scene = getattr(context, "scene", None)
            region_data = getattr(context, "region_data", None)
            scene_camera = getattr(scene, "camera", None)
            camera_frame_rect = _viewport_camera_frame_rect(context)
            draw_geometry = _viewport_draw_geometry(
                render_result.width,
                render_result.height,
                region_width,
                region_height,
                target_rect=camera_frame_rect,
                target_name="camera_frame" if camera_frame_rect is not None else "region",
            )
            draw_rect = draw_geometry["texture_draw_rect"]
            draw_x = float(draw_rect["x"])
            draw_y = float(draw_rect["y"])
            draw_width = float(draw_rect["width"])
            draw_height = float(draw_rect["height"])
            # Scene-linear frames route through Blender's display-space shader
            # so the scene's View Transform, Look, Exposure, and Gamma are
            # applied exactly once by Blender on draw (task02-05, contract
            # step 4). LDR frames are already display-encoded by OVRTX and draw
            # raw — binding the shader for them would double-transform.
            display_transform = (
                viewport_handoff.frame_applies_blender_display_transform(
                    render_result
                )
            )
            self._viewport_last_operator_view = {
                **draw_geometry,
                "view_perspective": str(getattr(region_data, "view_perspective", "")),
                "scene_camera": str(getattr(scene_camera, "name", "")),
                "viewport_presentation": dict(self._viewport_presentation),
                # Routing decision (scene-linear vs passthrough) vs the actual
                # bind result: they differ only on a rare bind failure, which
                # also records ``display_space_shader_bind_error``.
                "frame_display_transform": display_transform,
                "frame_color_mode": str(render_result.frame_color_mode),
            }
            scene_camera_delta = render_requests.scene_camera_pose_delta(context)
            if scene_camera_delta is not None:
                self._viewport_last_operator_view["scene_camera_pose_delta"] = scene_camera_delta
                self._viewport_last_operator_view["scene_camera_pose_matched"] = scene_camera_delta <= 1.0e-4

            framebuffer = gpu.state.active_framebuffer_get()
            framebuffer.clear(color=(0.0, 0.0, 0.0, 1.0))
            display_shader_bound = False
            if display_transform:
                display_shader_bound = self._bind_display_space_shader(scene)
            self._viewport_last_operator_view[
                "display_transform_applied_by_blender"
            ] = display_shader_bound
            # Premultiplied alpha while the display-space shader is bound
            # matches Blender's custom-engine draw template; LDR keeps the
            # opaque raw blend.
            gpu.state.blend_set(
                "ALPHA_PREMULT" if display_shader_bound else "NONE"
            )
            try:
                with gpu.matrix.push_pop():
                    with gpu.matrix.push_pop_projection():
                        gpu.matrix.load_matrix(Matrix.Identity(4))
                        gpu.matrix.load_projection_matrix(_ortho_2d(region_width, region_height))
                        draw_texture_2d(texture, (draw_x, draw_y), draw_width, draw_height)
            finally:
                if display_shader_bound:
                    self._unbind_display_space_shader()
                    gpu.state.blend_set("NONE")

        def _bind_display_space_shader(self, scene: Any) -> bool:
            """Bind Blender's linear->display GPU shader for the scene.

            Wraps ``RenderEngine.bind_display_space_shader`` (task02-05, the
            exactly-once application point for scene-linear frames). Returns
            ``True`` when the shader was bound so the caller knows to unbind.
            A missing scene or a harness without the base-class method (tests)
            fails closed to no bind and a raw draw.
            """

            bind = getattr(self, "bind_display_space_shader", None)
            if scene is None or not callable(bind):
                return False
            try:
                bind(scene)
                return True
            except Exception as exc:  # pragma: no cover - GPU/driver runtime
                self._viewport_last_operator_view[
                    "display_space_shader_bind_error"
                ] = str(exc)
                return False

        def _unbind_display_space_shader(self) -> None:
            unbind = getattr(self, "unbind_display_space_shader", None)
            if callable(unbind):
                try:
                    unbind()
                except Exception:  # pragma: no cover - GPU/driver runtime
                    pass

        def _record_profile(
            self,
            render_result: RenderResult,
            timings: dict[str, float],
            rendered: bool,
            *,
            started_at_ns: int,
            ended_at_ns: int,
            started_monotonic_ns: int,
            rgba_available_monotonic_ns: int,
            ended_monotonic_ns: int,
            span_boundaries: Mapping[str, int | None],
        ) -> None:
            self._viewport_session_outputs_written = False
            self._viewport_draw_count += 1
            request = self._viewport_request
            # Correlation with the render thread's per-iteration records
            # (task02-09): the presented publication carries the index and
            # snapshot key both streams share.
            presented = self._presented_frame
            record = {
                "thread": "main",
                "draw_index": self._viewport_draw_count,
                "snapshot_index": self._snapshot_index,
                "publication_index": (
                    int(presented.publication_index)
                    if presented is not None
                    else int(self._snapshot_index)
                ),
                "snapshot_key": viewport_profile.snapshot_key_token(
                    presented.snapshot_key if presented is not None else None
                ),
                "started_at_ns": started_at_ns,
                "ended_at_ns": ended_at_ns,
                "started_monotonic_ns": started_monotonic_ns,
                "rgba_available_monotonic_ns": rgba_available_monotonic_ns,
                "ended_monotonic_ns": ended_monotonic_ns,
                "span_boundaries": dict(span_boundaries),
                "snapshot_count": self._viewport_snapshot_count,
                "rendered": rendered,
                "composition_changed": bool(timings.get("composition_changed", 0.0)),
                "camera_changed": bool(timings.get("camera_changed", 0.0)),
                "snapshot_changed": bool(timings.get("snapshot_changed", 0.0)),
                "refinement_reset_reason": str(timings.get("refinement_reset_reason", "")),
                "timeline_controls_enabled": (
                    request.timeline_controls_enabled
                    if request is not None
                    else False
                ),
                "timeline_playing": (
                    request.timeline_playing
                    if request is not None
                    else False
                ),
                "timeline_frame": (
                    request.timeline_frame
                    if request is not None
                    else 0
                ),
                "timeline_reset": bool(timings.get("timeline_reset", 0.0)),
                "requested_additional_samples": int(timings.get("requested_additional_samples", 0.0)),
                "completed_samples": render_result.completed_samples,
                "session_completed_samples": render_result.session_completed_samples,
                "min_samples": request.min_samples if request is not None else 0,
                "max_samples": request.max_samples if request is not None else 0,
                "width": render_result.width,
                "height": render_result.height,
                "timings_ms": {
                    phase: timings.get(phase, 0.0)
                    for phase in viewport_profile.TIMING_PHASES
                },
            }
            record.update(self._texture_upload)
            native_timings = {}
            if rendered and render_result.native_timings:
                native_timings["render_result"] = dict(render_result.native_timings)
            controller = self._ovrtx_session_controller
            if record["camera_changed"] and controller is not None:
                diagnostics_started_ns = time.perf_counter_ns()
                value_timings = controller._value_update_timings_snapshot().get("native_timings", {})
                timings["profile_diagnostics_ms"] = (
                    time.perf_counter_ns() - diagnostics_started_ns
                ) / 1_000_000.0
                record["timings_ms"]["profile_diagnostics_ms"] = timings[
                    "profile_diagnostics_ms"
                ]
                if isinstance(value_timings, Mapping) and value_timings:
                    native_timings["value_update"] = dict(value_timings)
            if native_timings:
                record["native_timings"] = native_timings
            self._viewport_artifact_recorder.record(record)

        def _viewport_artifact(
            self,
            write_latency_ms: float = 0.0,
            *,
            running: bool = False,
            end_reason: ViewportSessionEndReason | str = "",
        ) -> dict[str, Any]:
            request = self._viewport_request
            render_result = self._current_result
            controller_diagnostics = (
                self._ovrtx_session_controller.diagnostics()
                if self._ovrtx_session_controller is not None
                else {
                    "simulation_id": None,
                    "session_reuse": {},
                    "lifecycle_events": (),
                }
            )
            prim_resolution_diagnostics = self._usd_prim_resolver.diagnostics()
            thread_model = self._thread_model_state()
            return self._viewport_artifact_recorder.artifact(
                viewport_artifact_recorder.State(
                    simulation_id=controller_diagnostics["simulation_id"],
                    request=request,
                    result=render_result,
                    snapshot_index=self._snapshot_index,
                    render_count=self._render_count,
                    draw_count=self._viewport_draw_count,
                    snapshot_count=self._viewport_snapshot_count,
                    camera_update_count=self._viewport_camera_update_count,
                    camera_controls_mode=self._viewport_camera_controls_mode,
                    viewport_presentation=self._viewport_presentation,
                    operator_view=self._viewport_last_operator_view,
                    pose_mirror=self._pose_mirror,
                    playback_lock=self._physics_playback_lock.diagnostics(),
                    image_artifact=self._image_artifact,
                    edit_bridge=interactive_edit_bridge_diagnostics(),
                    edit_workflow=(
                        self._interactive_edit_workflow.diagnostics()
                        if self._interactive_edit_workflow is not None
                        else {"event_count": 0}
                    ),
                    usd_prim_resolution=prim_resolution_diagnostics,
                    texture_upload=self._texture_upload,
                    # Scene-input provenance (task05-04): the authored
                    # generation identity this session presents, from the
                    # same request whose composition the artifact records
                    # (the direct env-override route carries no authoring
                    # session and records input_source=env_override).
                    authored_scene_provenance=_scene_generation_provenance(request),
                    ovrtx_scene_composition=ovrtx_scene_composition.diagnostics(
                        self._ovrtx_scene_composition,
                        request=self._scene_generation_artifact_request,
                    ),
                    ovrtx_session_reuse=controller_diagnostics["session_reuse"],
                    ovrtx_lifecycle_events=controller_diagnostics["lifecycle_events"],
                    startup=self._runtime_startup_diagnostics,
                    shared_stage_composition=(
                        self._runtime_scheduler.diagnostics()
                        if self._runtime_scheduler is not None
                        else {"enabled": False}
                    ),
                    session_lifecycle=self._session_lifecycle_diagnostics(
                        running=running,
                        end_reason=end_reason,
                    ),
                    session_started_at_ns=self._viewport_session_started_ns,
                    end_reason=_end_reason_value(end_reason),
                    write_ms=write_latency_ms,
                    running=running,
                    render_thread=thread_model["render_thread"],
                    render_loop=thread_model["render_loop"],
                    redraw_signaling=self._redraw_signaler.diagnostics(),
                    tick_absorb=self._tick_absorb_diagnostics(),
                    render_loop_records=thread_model["render_loop_records"],
                )
            )

        def _session_lifecycle_diagnostics(
            self,
            *,
            running: bool,
            end_reason: ViewportSessionEndReason | str = "",
        ) -> dict[str, Any]:
            startup_worker = self._runtime_startup_diagnostics.get("render_worker", {})
            worker_failed = (
                isinstance(startup_worker, Mapping)
                and str(startup_worker.get("status", "")) == "failed"
            )
            failure_count = (
                max(self._viewport_start_failure_count, session_lifecycle.MAX_AUTO_RETRIES)
                if worker_failed
                else self._viewport_start_failure_count
            )
            status = session_lifecycle.derive_status(
                engine_active=not bool(end_reason),
                ready=bool(not end_reason and (running or self._current_result is not None)),
                busy=bool(self._viewport_lifecycle_phase),
                phase=self._viewport_lifecycle_phase,
                failure_count=failure_count,
                last_error=self._viewport_reported_error,
            )
            marker = dict(self._viewport_crash_marker)
            if not marker:
                marker = session_lifecycle.read_stale_crash_marker()
            runtime_status = (
                self._runtime_tick_result.status.value
                if self._runtime_tick_result is not None
                else ""
            )
            runtime_failure = (
                self._runtime_tick_result.skipped_reason
                if self._runtime_tick_result is not None
                and self._runtime_tick_result.status == RuntimeTickStatus.FAILED
                else ""
            )
            return {
                **status,
                "completed_samples": int(
                    getattr(self._current_result, "completed_samples", 0) or 0
                ),
                "max_samples": int(
                    getattr(self._viewport_request, "max_samples", 0) or 0
                ),
                "active_generation": (
                    self._viewport_generation_runtime.ovrtx.active_generation
                    if self._viewport_generation_runtime is not None
                    else None
                ),
                "viewport_snapshot_count": self._viewport_snapshot_count,
                "viewport_render_count": self._render_count,
                "phase": self._viewport_lifecycle_phase,
                "failure_count": failure_count,
                "max_auto_retries": session_lifecycle.MAX_AUTO_RETRIES,
                "auto_retry_allowed": session_lifecycle.should_auto_retry(failure_count),
                "restart_count": self._viewport_restart_count,
                "end_reason": _end_reason_value(end_reason),
                "logs": dict(self._viewport_log_diagnostics),
                "crash_marker": marker,
                "cleanup": dict(self._viewport_cleanup_diagnostics),
                "runtime_status": runtime_status,
                "runtime_failure": runtime_failure,
            }

        def _write_viewport_artifact(
            self,
            *,
            running: bool,
            end_reason: ViewportSessionEndReason | str = "",
        ) -> float:
            artifact_path = os.environ.get("OV_BLENDER_EXAMPLE_VIEWPORT_ARTIFACT", "")
            if not artifact_path:
                return 0.0
            started_ns = time.perf_counter_ns()
            artifact = self._viewport_artifact(running=running, end_reason=end_reason)
            self._image_artifact = _write_image(
                os.environ.get("OV_BLENDER_EXAMPLE_IMAGE_ARTIFACT", ""),
                self._current_result,
            )
            if self._image_artifact:
                artifact["image_artifact"] = dict(self._image_artifact)
            try:
                Path(artifact_path).write_text(
                    json.dumps(artifact, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            except OSError as exc:
                self._report_viewport_error(RenderClientError(f"Could not write viewport artifact: {exc}"))
                return 0.0
            return (time.perf_counter_ns() - started_ns) / 1_000_000.0

        def _write_viewport_session_outputs(
            self,
            *,
            end_reason: ViewportSessionEndReason | str = "",
        ) -> None:
            if self._viewport_session_outputs_written:
                return
            controller_diagnostics = (
                self._ovrtx_session_controller.diagnostics()
                if self._ovrtx_session_controller is not None
                else {"active": False, "lifecycle_events": ()}
            )
            if (
                not controller_diagnostics["active"]
                and self._current_result is None
                and self._runtime_scheduler is None
                and self._viewport_session_started_ns <= 0
                and not controller_diagnostics["lifecycle_events"]
            ):
                return

            if self._pending_pose_mirror:
                self._apply_pending_pose_mirror()
            if (
                self._pose_mirror.get("status") == "scheduled"
                and self._applied_pose_mirror
            ):
                self._pose_mirror = dict(self._applied_pose_mirror)
            profile_path = os.environ.get("OV_BLENDER_EXAMPLE_VIEWPORT_PROFILE", "")
            artifact_write_ms = self._write_viewport_artifact(
                running=False,
                end_reason=end_reason,
            )
            if not profile_path or profile_path == "1":
                return
            profile = {
                # Version 3 removes the redundant controller refinement ledger;
                # version 2 introduced the thread-aware artifact payload
                # (thread_model / render_thread_profile / latest_view,
                # cross-thread timing phases, per-record thread field).
                "schema_version": 3,
                "artifact_id": "ovrtx-viewport-profile",
                "status": "complete",
                "viewport_artifact": self._viewport_artifact(
                    artifact_write_ms,
                    running=False,
                    end_reason=end_reason,
                ),
            }
            try:
                Path(profile_path).write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            except OSError as exc:
                self._report_viewport_error(RenderClientError(f"Could not write viewport profile: {exc}"))
                return
            self._viewport_session_outputs_written = True

        def _request_viewport_session_reconnect(self) -> tuple[bool, bool]:
            had_session = (
                self._ovrtx_session_controller is not None
                or self._current_result is not None
            )
            self._viewport_restart_count += 1
            self._viewport_start_failure_count = 0
            stopped = self._end_viewport_session(
                ViewportSessionEndReason.RECONNECT_REQUESTED
            )
            # Transient status until the next redraw re-attaches to the warm
            # worker; a reused session settles straight to "Live".
            self._viewport_lifecycle_phase = session_lifecycle.PHASE_RECONNECTING
            try:
                self.tag_redraw()
            except Exception:
                pass
            return had_session, stopped

        def _request_viewport_worker_restart(self) -> tuple[bool, bool]:
            had_session = (
                self._ovrtx_session_controller is not None
                or self._current_result is not None
            )
            self._viewport_restart_count += 1
            self._viewport_start_failure_count = 0
            stopped = self._end_viewport_session(
                ViewportSessionEndReason.WORKER_RESTART_REQUESTED
            )
            self._viewport_lifecycle_phase = session_lifecycle.PHASE_RESTARTING
            try:
                self.tag_redraw()
            except Exception:
                pass
            return had_session, stopped

        def _viewport_session_status(self) -> dict[str, Any]:
            controller = self._ovrtx_session_controller
            return self._session_lifecycle_diagnostics(
                running=bool(controller is not None and controller.diagnostics()["active"]),
            )

        def _report_final(self, level: Any, message: str) -> None:
            """``self.report`` for the final-render / edit paths, also mirrored
            to stdout/stderr.

            In these main-thread contexts ``self.report`` reaches the Info
            window natively, so this only additionally mirrors to the console
            (stdout for INFO, stderr for WARNING/ERROR) — no Info double-post.
            """

            try:
                self.report(level, message)
            except Exception:
                pass
            user_messages.mirror_to_console(level, message)

        def _report_viewport_error(
            self,
            exc: BlenderSignalTranslationError | RenderClientError,
        ) -> None:
            message = str(exc)
            if message != self._viewport_reported_error:
                self.report({"ERROR"}, message)
                # ``self.report`` does not reach the Info window from the
                # viewport-draw / render-thread context this runs in, so the
                # error is fanned out through the central bus as well: stderr
                # immediately, Info window via the main-thread pump. Overlay
                # (update_stats) is untouched below.
                #
                # ``dedup=False``: this method already emits change-only via the
                # ``_viewport_reported_error`` guard, and that guard is reset to
                # "" on the next good frame (see view_draw). The bus's own
                # per-context dedup does not see that reset, so leaving it on
                # would swallow a genuine error recurrence (same text) after an
                # intervening successful frame -- exactly the repeated distinct
                # event that must still reach every channel.
                user_messages.report_error(
                    message, context=f"viewport-error:{id(self)}", dedup=False
                )
                self._viewport_reported_error = message
            self.update_stats("ovrtx", f"Viewport failed: {message}")

        def _report_lifecycle_transition(self, message: str) -> None:
            """Log a session lifecycle transition to the Info panel + stdout.

            The default log routing has no files (worker output inherits
            Blender's stdout): the add-on's own session milestones surface
            here — once per transition, never per draw. ``report({'INFO'})``
            keeps the native Info-panel report (which the interactive
            operator context still forwards), and the central user-message bus
            additionally writes ``[ovrtx] ...`` to stdout (also covering
            headless runs) and posts to the Info editor via the main-thread
            pump, which is the path that survives the viewport-draw /
            render-thread contexts where ``report`` does not reach Info.
            """

            if not message or message == self._lifecycle_report_message:
                return
            self._lifecycle_report_message = message
            try:
                self.report({"INFO"}, message)
            except Exception:
                pass
            user_messages.report_info(message, context=f"lifecycle:{id(self)}")

        def _end_viewport_session(self, reason: ViewportSessionEndReason) -> bool:
            # One linear teardown: stop request → loop exit → exact-stage
            # scheduler/controller teardown → one bounded join. A current-
            # scene loop only detaches from its authoring runtime here.
            teardown, teardown_state = self._runtime_teardown_state()
            outcome = self._stop_render_loop(teardown=teardown)
            joined = bool(outcome.get("joined", False))
            # Drop any tick handoff the stopped loop left behind: its pose
            # set belongs to the session being torn down. A still-pending
            # absorb timer no-ops on the empty handoff.
            with self._tick_absorb_lock:
                self._pending_tick_absorb = None
            if self._viewport_crash_marker.get("marker_active"):
                self._viewport_crash_marker = session_lifecycle.clear_crash_marker()
            if joined and not teardown_state["ran"]:
                # No running thread accepted the teardown command (never
                # started, already stopped, or failed): the confirmed join
                # handed RPC ownership back to this thread, so adopt the
                # controller (RPC thread guard) and tear down here. The
                # defensive ``stop_failed`` outcome (stop() raising) keeps
                # ``joined`` False and lands in neither branch: with the
                # thread state unknown, running RPCs here would race it.
                controller = self._ovrtx_session_controller
                adopt = getattr(controller, "adopt_owning_thread", None)
                if callable(adopt):
                    try:
                        adopt()
                    except Exception:
                        pass
                teardown()
            if teardown_state["errors"]:
                # Merged before the outputs write so the ending session's
                # artifact (session_lifecycle.cleanup) carries them.
                cleanup = dict(self._viewport_cleanup_diagnostics)
                if cleanup.get("status") in (None, "", "not_requested"):
                    cleanup["status"] = "teardown_errors"
                cleanup["teardown_errors"] = list(teardown_state["errors"])
                self._viewport_cleanup_diagnostics = cleanup
            self._write_viewport_session_outputs(end_reason=reason)
            if joined:
                scene_generation_sessions.detach_viewport(_engine_signal_id(self))
            authored_runtime = self._viewport_generation_runtime
            self._viewport_generation_runtime = None
            physics_lock = getattr(self, "_physics_playback_lock", None)
            if physics_lock is not None and authored_runtime is None:
                physics_lock.clear(reason="viewport_shutdown")
            # Exact-stage scheduler shutdown lives behind the thread boundary;
            # a current-scene scheduler remains owned by the authoring runtime.
            self._runtime_scheduler = None
            self._runtime_tick_result = None
            self._interactive_edit_workflow = None
            self._clear_ovrtx_runtime_state()
            self._viewport_presentation = {
                "presentation_mode": viewport_presentation.OVRTX_RENDERED_PRESENTATION,
                "fallback_reason": "",
                "fallback_owned_by_addon": False,
                "view_perspective": "",
                "changed": False,
            }
            self._usd_prim_resolver.reset()
            self._pose_mirror = {}
            self._applied_pose_mirror = {}
            self._pending_pose_mirror = {}
            self._pose_mirror_timer_registered = False
            self._viewport_snapshot_count = 0
            self._viewport_draw_count = 0
            self._render_count = 0
            self._viewport_camera_update_count = 0
            self._viewport_session_started_ns = 0
            self._viewport_artifact_recorder.reset()
            return joined

        def _close_ovrtx_runtime(self) -> None:
            """Synchronous session close: teardown RPCs on the calling thread.

            Retained for thread-less paths (sessions that never started a
            render thread) and as the seam existing tests pin. The
            production teardown path is ``_end_viewport_session``, which
            runs the RPCs on the render thread (task02-08).
            """

            controller = self._ovrtx_session_controller
            prepared = self._viewport_generation_runtime
            scene_generation_sessions.detach_viewport(_engine_signal_id(self))
            self._viewport_generation_runtime = None
            self._clear_ovrtx_runtime_state()
            if controller is not None and prepared is None:
                controller.shutdown()

        def _clear_ovrtx_runtime_state(self) -> None:
            """Main-thread state clearing shared by every session-close path."""

            self._viewport_scene = None
            self._ovrtx_session_controller = None
            self._viewport_request = None
            self._ovrtx_scene_composition = None
            self._scene_generation_artifact_request = None
            self._current_result = None
            self._viewport_texture = None
            self._viewport_texture_size = None
            self._texture_snapshot_index = 0
            self._texture_upload = {}
            self._image_artifact = {}
            self._viewport_last_operator_view = {}
            self._viewport_camera_controls_mode = render_requests.CAMERA_CONTROLS_USD
            self._runtime_startup_diagnostics = {"render_worker": {"status": "not_started"}}
            self._viewport_lifecycle_phase = ""
            # Fresh handoff structures: the publication index resets with
            # the session so a new session can never present a stale frame.
            self._camera_mailbox = viewport_handoff.CameraRequestMailbox()
            self._frame_slot = viewport_handoff.LatestFrameSlot()
            self._presented_publication_index = 0
            self._presented_frame = None
            self._written_snapshot_key = None

else:
    OvrtxExampleRenderEngine = None  # type: ignore[assignment]


def _write_image(path: str, render_result: RenderResult | None) -> dict[str, Any]:
    if not path or render_result is None:
        return {}
    try:
        row_size = render_result.width * 4
        # OVRTX rows are bottom-up for GPU upload; PNG scanlines are top-down.
        rgba8 = b"".join(
            render_result.rgba8[row * row_size : (row + 1) * row_size]
            for row in reversed(range(render_result.height))
        )
        return write_rgba_png(
            Path(path), render_result.width, render_result.height, rgba8
        )
    except Exception as exc:
        return {
            "path": path,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _fit_size(source_width: int, source_height: int, target_width: int, target_height: int) -> tuple[float, float]:
    source_aspect = source_width / max(1, source_height)
    target_aspect = target_width / max(1, target_height)
    if target_aspect > source_aspect:
        height = float(target_height)
        return height * source_aspect, height
    width = float(target_width)
    return width, width / source_aspect


def _viewport_draw_geometry(
    result_width: int,
    result_height: int,
    region_width: int,
    region_height: int,
    *,
    target_rect: Mapping[str, Any] | None = None,
    target_name: str = "region",
) -> dict[str, Any]:
    region_width = max(1, int(region_width))
    region_height = max(1, int(region_height))
    target = _valid_viewport_target_rect(target_rect) or {
        "x": 0.0,
        "y": 0.0,
        "width": float(region_width),
        "height": float(region_height),
    }
    draw_width, draw_height = _fit_size(
        result_width,
        result_height,
        int(round(float(target["width"]))),
        int(round(float(target["height"]))),
    )
    return {
        "render_result": {
            "width": int(result_width),
            "height": int(result_height),
        },
        "region": {
            "width": region_width,
            "height": region_height,
        },
        "draw_target": target_name if target_rect is not None else "region",
        "draw_target_rect": dict(target),
        "texture_draw_rect": {
            "x": float(target["x"]) + (float(target["width"]) - draw_width) * 0.5,
            "y": float(target["y"]) + (float(target["height"]) - draw_height) * 0.5,
            "width": draw_width,
            "height": draw_height,
        },
    }


def _valid_viewport_target_rect(rect: Mapping[str, Any] | None) -> dict[str, float] | None:
    if not isinstance(rect, Mapping):
        return None
    try:
        x = float(rect["x"])
        y = float(rect["y"])
        width = float(rect["width"])
        height = float(rect["height"])
    except (KeyError, TypeError, ValueError):
        return None
    if width <= 0.0 or height <= 0.0:
        return None
    return {"x": x, "y": y, "width": width, "height": height}


def _viewport_camera_frame_rect(context: Any) -> dict[str, float] | None:
    region_data = getattr(context, "region_data", None)
    if str(getattr(region_data, "view_perspective", "")) != "CAMERA":
        return None
    scene = getattr(context, "scene", None)
    camera = getattr(scene, "camera", None)
    region = getattr(context, "region", None)
    if scene is None or camera is None or region is None:
        return None
    try:
        from bpy_extras.view3d_utils import location_3d_to_region_2d  # type: ignore
    except Exception:
        return None
    try:
        update = getattr(region_data, "update", None)
        if callable(update):
            update()
        points = []
        for corner in camera.data.view_frame(scene=scene):
            projected = location_3d_to_region_2d(region, region_data, camera.matrix_world @ corner)
            if projected is None:
                return None
            points.append((float(projected.x), float(projected.y)))
    except Exception:
        return None
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_min = max(0.0, min(xs))
    y_min = max(0.0, min(ys))
    x_max = min(float(getattr(region, "width", 0) or 0), max(xs))
    y_max = min(float(getattr(region, "height", 0) or 0), max(ys))
    if x_max <= x_min or y_max <= y_min:
        return None
    return {
        "x": x_min,
        "y": y_min,
        "width": x_max - x_min,
        "height": y_max - y_min,
    }


def _ortho_2d(width: int, height: int) -> Any:
    from mathutils import Matrix  # type: ignore

    return Matrix(
        (
            (2.0 / max(1, width), 0.0, 0.0, -1.0),
            (0.0, 2.0 / max(1, height), 0.0, -1.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )


# Viewport preview draws the usd ovrtx render product. Blender viewport
# camera controls, tagged stock object edits, and explicitly enabled existing
# material values can drive same-session updates; broad material and
# topology authoring still use resolved write targets.

__all__ = [
    "BLENDER_AVAILABLE",
    "ENGINE_ID",
    "OvrtxExampleRenderEngine",
    "build_request_from_scene",
    "interactive_edit_bridge_diagnostics",
    "register_interactive_edit_bridge",
    "viewport_session_statuses",
    "reconnect_viewport_sessions",
    "resolve_blender_selection_to_edit_owners",
    "submit_depsgraph_interactive_edits_to_active_viewports",
    "suppress_interactive_edit_bridge",
    "unregister_interactive_edit_bridge",
]

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replaceable OVRTX session lifecycle and raw render acquisition."""

from __future__ import annotations

import copy
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
import os
import threading
import time
from typing import Any, Callable, Mapping

from . import color_presentation, ovrtx_session, render_requests, rtpt_worker_config
from .ovrtx_runtime_client import OvrtxRuntimeClient, RenderClientError, RenderResult
from .ovrtx_scene_composition import OvrtxSceneComposition, diagnostics as composition_diagnostics
from .ovrtx_value_updates import OvrtxSessionUpdatePort, OvrtxUpdatePort
from .render_requests import RenderRequest
from .runtime_scheduler import RuntimeTickResult


RPC_THREAD_GUARD_ENV = "OV_BLENDER_EXAMPLE_DEBUG_RPC_THREAD_GUARD"
_PRESENTATION_UNSET = object()


def _serialized(function: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(function)
    def call(self: "OvrtxSessionController", *args: Any, **kwargs: Any) -> Any:
        with self._transport_lock:
            return function(self, *args, **kwargs)

    return call


class OvrtxThreadConfinementError(RenderClientError):
    """An srtx RPC entry point was called off the owning render thread."""


def _rpc_thread_guard_enabled(env: Mapping[str, str] = os.environ) -> bool:
    return env.get(RPC_THREAD_GUARD_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _runtime_client_from_request(request: RenderRequest) -> OvrtxRuntimeClient:
    return OvrtxRuntimeClient(
        worker_command=request.worker_command,
        native_client_module=request.native_client_module,
    )


@dataclass(frozen=True)
class OvrtxEnsureResult:
    composition: OvrtxSceneComposition
    session_started: bool


class OvrtxSessionController:
    """Own one replaceable OVRTX session and its transport diagnostics.

    ``simulation_id`` pins every session this controller starts to one
    simulation ID lane instead of the runtime client's default viewport
    lane. The F12 final render (blender-live-render task05-01) uses it to
    create its own session on the already-running worker without colliding
    with the viewport session's simulation; callers must keep the
    ``ovrtx-blender-<lane>-<pid>`` naming convention so the worker-attach
    sweep's stale-PID parser (task05-02) covers the lane.
    """

    def __init__(self, *, simulation_id: str | None = None) -> None:
        self._requested_simulation_id = str(simulation_id) if simulation_id else None
        self._client: Any | None = None
        self._simulation_id: str | None = None
        self._spec: ovrtx_session.OvrtxSessionSpec | None = None
        self._runtime_binding: tuple[str, str] | None = None
        self._project_complete_pose = False
        self._closed = False
        self._reuse: dict[str, Any] = {}
        # retain recent lifecycle evidence; add counters only if
        # lifetime totals become a diagnostics contract.
        self._events: deque[dict[str, Any]] = deque(maxlen=120)
        self._startup: dict[str, Any] = {"render_worker": {"status": "not_started"}}
        self._rtpt_config: dict[str, Any] = {"status": "not_authored"}
        self._dlss_enabled: bool = True
        self._render_timings: dict[str, Any] = {}
        self._value_update_timings: dict[str, Any] = {}
        self._ensure_timings: dict[str, float] = {}
        self._last_stop_status = "not_found"
        self._owning_thread_ident: int | None = None
        self._thread_guard_active = False
        self._transport_lock = threading.RLock()
        self._exclusive_gate = threading.Condition()
        self._exclusive_pending = 0
        self._transport_waiters: deque[object] = deque()
        self._presentations: dict[
            int, tuple[Callable[[], None] | None, Callable[[], None] | None]
        ] = {}
        self._presentation_key: Any = _PRESENTATION_UNSET
        self._session_revision = 0

    @property
    def worker_owned(self) -> bool | None:
        """Whether the active session's worker process was launched by this process.

        Mirrors :attr:`OvrtxRuntimeClient.worker_owned` for the active client.
        ``True`` means the session teardown (``client.shutdown()`` inside
        ``_deactivate_active``) terminates the worker, so the next
        :meth:`ensure` relaunches it and it re-reads the worker startup config
        (``rtpt_worker_config``) with the current RTPT slider values. ``False``
        means the serving worker is foreign and survives restarts with its old
        launch-time settings. ``None``: no active client / not yet known.
        """

        client = self._client
        if client is None:
            return None
        value = getattr(client, "worker_owned", None)
        return None if value is None else bool(value)

    def adopt_owning_thread(self, thread_ident: int | None = None) -> int:
        """Record the render thread that owns every srtx RPC entry point.

        Spec design decision (one RPC thread per session): correctness comes
        from confinement, not locking around the native client. When the
        ``OV_BLENDER_EXAMPLE_DEBUG_RPC_THREAD_GUARD`` debug env is enabled,
        RPC entry points raise :class:`OvrtxThreadConfinementError` on any
        other thread; release runs skip the check (no hot-path cost).
        """

        ident = int(thread_ident) if thread_ident is not None else threading.get_ident()
        self._owning_thread_ident = ident
        self._thread_guard_active = _rpc_thread_guard_enabled()
        return ident

    def _allow_serialized_threads(self) -> None:
        """Use transport serialization when several presentation threads share us."""

        self._owning_thread_ident = None
        self._thread_guard_active = False

    def _attach_presentation(
        self,
        presentation_key: int,
        wake: Callable[[], None] | None = None,
        restore: Callable[[], None] | None = None,
    ) -> None:
        with self._exclusive_gate:
            self._presentations[presentation_key] = (wake, restore)

    def _detach_presentation(self, presentation_key: int) -> None:
        with self._exclusive_gate:
            self._presentations.pop(presentation_key, None)
            callbacks = next(iter(self._presentations.values()), (None, None))
            restore_single = len(self._presentations) == 1
        if restore_single and callbacks[1] is not None:
            callbacks[1]()

    def _shared_output_shape(self) -> tuple[int, int] | None:
        """Return the active session's canonical shape while panes share it."""

        with self._exclusive_gate:
            shared = len(self._presentations) > 1
        spec = self._spec
        return (spec.width, spec.height) if shared and spec is not None else None

    def _has_shared_presentations(self) -> bool:
        with self._exclusive_gate:
            return len(self._presentations) > 1

    def _advance_session_revision(self) -> None:
        self._session_revision += 1
        with self._exclusive_gate:
            presentation_wakes = tuple(
                callbacks[0]
                for callbacks in self._presentations.values()
                if callbacks[0] is not None
            )
        for wake in presentation_wakes:
            wake()

    @contextmanager
    def _serialized_transport(
        self,
        presentation_key: Any = _PRESENTATION_UNSET,
        exclusive_pending: Callable[[], bool] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ):
        """Keep one presentation's mutation and readback contiguous, FIFO."""

        while True:
            waiter = object()
            with self._exclusive_gate:
                self._transport_waiters.append(waiter)
            while True:
                caller_has_job = bool(exclusive_pending and exclusive_pending())
                with self._exclusive_gate:
                    if cancelled and cancelled():
                        self._transport_waiters.remove(waiter)
                        self._exclusive_gate.notify_all()
                        yield False, True
                        return
                    if caller_has_job or (
                        not self._exclusive_pending
                        and self._transport_waiters[0] is waiter
                    ):
                        caller_owns_exclusive = bool(caller_has_job)
                        if caller_owns_exclusive:
                            self._transport_waiters.remove(waiter)
                        break
                    self._exclusive_gate.wait()
            if caller_owns_exclusive:
                try:
                    with self._transport_lock:
                        yield False, True
                finally:
                    with self._exclusive_gate:
                        self._exclusive_gate.notify_all()
                return
            try:
                with self._transport_lock:
                    with self._exclusive_gate:
                        self._transport_waiters.remove(waiter)
                        if self._exclusive_pending:
                            self._exclusive_gate.notify_all()
                            continue
                    presentation_changed = (
                        presentation_key is not _PRESENTATION_UNSET
                        and self._presentation_key is not _PRESENTATION_UNSET
                        and presentation_key != self._presentation_key
                    )
                    yield presentation_changed, False
                    if presentation_key is not _PRESENTATION_UNSET:
                        self._presentation_key = presentation_key
            finally:
                with self._exclusive_gate:
                    self._exclusive_gate.notify_all()
            return

    def _wake_serialized_transport(self) -> None:
        """Wake waiters so a stopped presentation can leave the FIFO."""

        with self._exclusive_gate:
            self._exclusive_gate.notify_all()

    @contextmanager
    def _validated_session(self, revision: int):
        """Keep publication atomic with respect to session replacement."""

        with self._transport_lock:
            yield self._session_revision == int(revision)

    def _request_exclusive_transport(self) -> None:
        """Prevent new presentation transactions while an exclusive job waits."""

        with self._exclusive_gate:
            self._exclusive_pending += 1
            self._exclusive_gate.notify_all()

    def _release_exclusive_transport(self) -> None:
        with self._exclusive_gate:
            self._exclusive_pending = max(0, self._exclusive_pending - 1)
            if not self._exclusive_pending:
                self._exclusive_gate.notify_all()

    @contextmanager
    def _exclusive_transport(self):
        """Hold the shared runtime across one complete exclusive job."""

        with self._transport_lock:
            yield

    def would_replace(self, request: RenderRequest) -> str:
        """Reason the next :meth:`ensure` would replace the session.

        Read-only reuse probe for the render loop's replacement triggers
        (blender-live-render task02-06): returns ``""`` when the active
        session would be reused, otherwise the ``reuse_decision`` blocker
        (``output_shape_changed``, ``scene_composition_changed``,
        ``declared_sensors_changed``, ``camera_prim_changed``,
        ``camera_pose_override_removed``, ``render_var_changed``),
        ``runtime_binding_changed``, or ``no_active_session``. Performs no
        RPCs, so it is deliberately not
        thread-guarded, and it mutates no controller state.
        """

        if self._closed or self._spec is None or self._simulation_id is None:
            return "no_active_session"
        binding = (request.worker_command, request.native_client_module)
        if self._runtime_binding != binding:
            return "runtime_binding_changed"
        desired = ovrtx_session.build_spec(request)
        decision = ovrtx_session.reuse_decision(self._spec, desired)
        return "" if decision.reuse else decision.reason

    @_serialized
    def ensure(self, request: RenderRequest) -> OvrtxEnsureResult:
        self._guard_rpc_thread("ensure")
        ensure_started_ns = time.perf_counter_ns()
        self._require_open()
        build_started_ns = time.perf_counter_ns()
        desired = ovrtx_session.build_spec(request)
        build_spec_ms = (time.perf_counter_ns() - build_started_ns) / 1_000_000.0
        reuse_decision_ms = 0.0
        binding = (request.worker_command, request.native_client_module)
        reuse = {"reuse": False, "reason": "no_active_session"}
        if self._spec is not None and self._simulation_id is not None:
            if self._runtime_binding == binding:
                reuse_started_ns = time.perf_counter_ns()
                decision = ovrtx_session.reuse_decision(self._spec, desired)
                reuse_decision_ms = (
                    time.perf_counter_ns() - reuse_started_ns
                ) / 1_000_000.0
                reuse = {"reuse": decision.reuse, "reason": decision.reason}
                if decision.reuse:
                    self._spec = desired
                    self._reuse = reuse
                    total_ms = (
                        time.perf_counter_ns() - ensure_started_ns
                    ) / 1_000_000.0
                    self._ensure_timings = {
                        "total_ms": total_ms,
                        "build_spec_ms": build_spec_ms,
                        "reuse_decision_ms": reuse_decision_ms,
                        "other_ms": max(
                            0.0,
                            total_ms - build_spec_ms - reuse_decision_ms,
                        ),
                    }
                    return OvrtxEnsureResult(
                        composition=desired.ovrtx_scene_composition,
                        session_started=False,
                    )
            else:
                reuse = {"reuse": False, "reason": "runtime_binding_changed"}

        active_before_close = self._simulation_id is not None
        replacing = active_before_close or self._project_complete_pose
        replacement_client = self._client if self._runtime_binding == binding else None
        self._close_active(
            preserve_complete_pose=replacing,
            preserve_worker=replacement_client is not None,
        )
        # Author the RTPT quality values into the worker's startup config before
        # a fresh worker can launch. This worker build ignores the RenderProduct
        # omni:rtx:rtpt:* attributes but honors the same values as /rtx/rtpt/*
        # carb settings read at process launch (runtime measurements, real-GPU A/B).
        # Launch-only: a value change reaches the renderer when the worker
        # process (re)starts, not on an in-process session re-key that reuses a
        # running worker. Best-effort; never blocks session startup.
        # The DLSS toggle rides the same authoring call. Unlike the rtpt family,
        # DLSS execMode is ALSO honored on the RenderProduct at session creation
        # (runtime measurements), so it additionally applies via the composition
        # digest re-key with no worker restart; the config write covers freshly
        # launched workers.
        self._dlss_enabled = bool(getattr(request, "dlss_enabled", True))
        self._rtpt_config = rtpt_worker_config.author_worker_config(
            request.worker_command,
            getattr(request, "rtpt_quality", None),
            self._dlss_enabled,
        )
        client = replacement_client or _runtime_client_from_request(request)
        try:
            if self._requested_simulation_id:
                simulation_id = client.start_session(
                    desired, self._requested_simulation_id
                )
            else:
                simulation_id = client.start_session(desired)
        except RenderClientError:
            self._startup = dict(getattr(client, "startup_diagnostics", {}))
            try:
                client.shutdown()
            except Exception:
                pass
            self._project_complete_pose = replacing
            raise

        self._client = client
        self._simulation_id = simulation_id
        self._spec = desired
        self._runtime_binding = binding
        self._project_complete_pose = replacing
        self._reuse = reuse
        self._startup = dict(getattr(client, "startup_diagnostics", {}))
        self._advance_session_revision()
        self._events.append(
            {
                "event": "replaced" if replacing else "created",
                "reason": reuse["reason"],
                "simulation_id": simulation_id,
                "time_ns": time.time_ns(),
            }
        )
        total_ms = (time.perf_counter_ns() - ensure_started_ns) / 1_000_000.0
        self._ensure_timings = {
            "total_ms": total_ms,
            "build_spec_ms": build_spec_ms,
            "reuse_decision_ms": reuse_decision_ms,
            "other_ms": max(0.0, total_ms - build_spec_ms - reuse_decision_ms),
        }
        return OvrtxEnsureResult(
            composition=desired.ovrtx_scene_composition,
            session_started=True,
        )

    @_serialized
    def apply_runtime_updates(
        self,
        operation: Callable[[OvrtxUpdatePort, bool], RuntimeTickResult],
    ) -> RuntimeTickResult:
        self._guard_rpc_thread("apply_runtime_updates")
        client, simulation_id = self._active()
        result = operation(
            OvrtxSessionUpdatePort(client, simulation_id),
            self._project_complete_pose,
        )
        self._value_update_timings = dict(
            getattr(client, "last_value_update_timings", {})
        )
        if not isinstance(result, RuntimeTickResult):
            raise TypeError("OVRTX runtime update operation must return RuntimeTickResult")
        if result.complete_pose_projected is False:
            self._close_active(preserve_complete_pose=True)
        elif result.complete_pose_projected is True:
            self._project_complete_pose = False
        return result

    @_serialized
    def render(
        self,
        request: RenderRequest,
        *,
        additional_samples: int,
    ) -> RenderResult:
        self._guard_rpc_thread("render")
        client, simulation_id = self._active()
        additional_samples = int(additional_samples)
        if additional_samples < 1:
            raise ValueError("additional_samples must be at least 1")
        result = client.render_result(
            simulation_id,
            selected_sensor_paths=request.selected_sensor_paths,
            render_var=str(
                request.color_presentation.get(
                    "render_var",
                    color_presentation.RENDER_VAR_LDR_COLOR,
                )
            ),
            additional_samples=additional_samples,
        )
        self._render_timings = dict(getattr(client, "last_render_timings", {}))
        return result

    def diagnostics(self) -> Mapping[str, Any]:
        snapshot = {
            "active": self._simulation_id is not None,
            "simulation_id": self._simulation_id,
            "ovrtx_scene_composition": composition_diagnostics(
                self._spec.ovrtx_scene_composition if self._spec is not None else None
            ),
            "session_reuse": dict(self._reuse),
            "rtpt_worker_config": dict(self._rtpt_config),
            "dlss_enabled": self._dlss_enabled,
            "worker_owned": self.worker_owned,
            "lifecycle_events": tuple(dict(event) for event in self._events),
            "startup": dict(self._startup),
            "render_timings": dict(self._render_timings),
            "value_update_timings": dict(self._value_update_timings),
            "stop_status": self._last_stop_status,
            "rpc_thread": {
                "owning_thread_ident": self._owning_thread_ident or 0,
                "adopted": self._owning_thread_ident is not None,
                "guard_active": self._thread_guard_active,
            },
        }
        return copy.deepcopy(snapshot)

    def _render_timings_snapshot(self) -> Mapping[str, Any]:
        return copy.deepcopy(self._render_timings)

    def _ensure_timings_snapshot(self) -> Mapping[str, float]:
        return dict(self._ensure_timings)

    def _value_update_timings_snapshot(self) -> Mapping[str, Any]:
        return copy.deepcopy(self._value_update_timings)

    @_serialized
    def deactivate(self) -> str:
        self._guard_rpc_thread("deactivate")
        return self._deactivate_active()

    @_serialized
    def suspend(self) -> str:
        """Delete the active simulation but keep the controller open.

        The worker loads one simulation at a time, so the F12 final render
        must borrow the worker from a live viewport session: the job
        suspends the viewport simulation on the shared RPC thread before
        creating the final-render lane. ``would_replace`` reports
        ``no_active_session`` afterwards, and the next :meth:`ensure`
        recreates the session as a replacement — the complete pose set
        re-projects like any other replacement.
        """

        self._guard_rpc_thread("suspend")
        if self._closed or self._simulation_id is None:
            return "not_found"
        status = self._deactivate_active(preserve_complete_pose=True)
        if status in {"stopped", "not_found"}:
            self._project_complete_pose = True
        return status

    @_serialized
    def shutdown(self) -> str:
        self._guard_rpc_thread("shutdown")
        if self._closed:
            return "not_found"
        status = self._deactivate_active()
        if status == "failed":
            return status
        self._closed = True
        return status

    def _guard_rpc_thread(self, operation: str) -> None:
        if not self._thread_guard_active:
            return
        ident = threading.get_ident()
        if ident != self._owning_thread_ident:
            raise OvrtxThreadConfinementError(
                f"OvrtxSessionController.{operation} was called from thread "
                f"{ident} ({threading.current_thread().name!r}); srtx RPCs are "
                f"confined to the owning render thread {self._owning_thread_ident}"
            )

    def _require_open(self) -> None:
        if self._closed:
            raise RenderClientError("OVRTX session controller is shut down")

    def _active(self) -> tuple[Any, str]:
        self._require_open()
        if self._client is None or self._simulation_id is None:
            raise RenderClientError("No active OVRTX session")
        return self._client, self._simulation_id

    def _deactivate_active(
        self,
        *,
        preserve_complete_pose: bool = False,
        preserve_worker: bool = False,
    ) -> str:
        client = self._client
        simulation_id = self._simulation_id
        project_complete_pose = self._project_complete_pose if preserve_complete_pose else False
        if client is None or simulation_id is None:
            self._project_complete_pose = project_complete_pose
            self._last_stop_status = "not_found"
            return self._last_stop_status
        try:
            status = str(client.delete_simulation(simulation_id) or "failed")
        except Exception as exc:
            self._events.append(
                {
                    "event": "stop_failed",
                    "simulation_id": simulation_id,
                    "error": f"{type(exc).__name__}: {exc}",
                    "time_ns": time.time_ns(),
                }
            )
            self._last_stop_status = "failed"
            return self._last_stop_status
        self._last_stop_status = status
        if status not in {"stopped", "not_found"}:
            return "failed"
        self._client = None
        self._simulation_id = None
        self._spec = None
        self._runtime_binding = None
        self._project_complete_pose = project_complete_pose
        self._advance_session_revision()
        if not preserve_worker:
            try:
                client.shutdown()
            except Exception:
                pass
        self._events.append(
            {
                "event": "stopped",
                "status": status,
                "simulation_id": simulation_id,
                "time_ns": time.time_ns(),
            }
        )
        return status

    def _close_active(
        self,
        *,
        preserve_complete_pose: bool = False,
        preserve_worker: bool = False,
    ) -> None:
        status = self._deactivate_active(
            preserve_complete_pose=preserve_complete_pose,
            preserve_worker=preserve_worker,
        )
        if status == "failed":
            raise RenderClientError("OVRTX simulation deletion was not confirmed")


__all__ = [
    "OvrtxEnsureResult",
    "OvrtxSessionController",
    "OvrtxThreadConfinementError",
    "RPC_THREAD_GUARD_ENV",
]

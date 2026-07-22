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
from .ovrtx_runtime_client import (
    OvrtxRuntimeClient,
    RenderClientError,
    RenderReadTicket,
    RenderResult,
    RenderSubmission,
)
from .ovrtx_scene_composition import OvrtxSceneComposition, diagnostics as composition_diagnostics
from .ovrtx_value_updates import (
    OvrtxSessionUpdatePort,
    OvrtxTransformValue,
    OvrtxUpdatePort,
    OvrtxValueUpdateResult,
)
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


class OvrtxSessionRetirementRequiredError(RenderClientError):
    """A split-render RPC left its exact-time session unsafe to reuse."""

    def __init__(
        self,
        message: str,
        *,
        controller: Any,
        session_revision: int,
        operation: str,
    ) -> None:
        super().__init__(message)
        self.controller = controller
        self.session_revision = int(session_revision)
        self.operation = str(operation)


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


@dataclass(frozen=True)
class OvrtxPreparedRenderSubmission:
    """One transform batch and the sample submitted immediately behind it."""

    submission: RenderSubmission
    update_result: OvrtxValueUpdateResult
    session_revision: int


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
        self._render_session_invalid_revision: int | None = None
        self._exclusive_gate = threading.Condition()
        self._exclusive_pending = 0
        self._transport_waiters: deque[object] = deque()
        self._serialized_presentation_key: Any = _PRESENTATION_UNSET
        self._prefetched_read_reservation: Any = _PRESENTATION_UNSET
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
        with self._exclusive_gate:
            if len(self._presentations) > 1:
                self._owning_thread_ident = None
                self._thread_guard_active = False
            else:
                self._owning_thread_ident = ident
                self._thread_guard_active = _rpc_thread_guard_enabled()
        return ident

    def _allow_serialized_threads(self) -> None:
        """Use transport serialization when several presentation threads share us."""

        with self._exclusive_gate:
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
            if len(self._presentations) > 1:
                # Shared presentations serialize through _transport_lock; no
                # single render thread can remain the controller-wide RPC
                # owner. Handle both attach-before-adopt and adopt-before-attach.
                self._owning_thread_ident = None
                self._thread_guard_active = False

    def _detach_presentation(self, presentation_key: int) -> None:
        with self._exclusive_gate:
            self._presentations.pop(presentation_key, None)
            if self._prefetched_read_reservation == presentation_key:
                # Detach is the last-resort abort boundary. Normal teardown
                # cancels and releases first; if it did not, fail the session
                # closed so another pane cannot reuse potentially live native
                # ownership after this orphan-prevention release.
                self._prefetched_read_reservation = _PRESENTATION_UNSET
                self._render_session_invalid_revision = self._session_revision
                self._exclusive_gate.notify_all()
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

    def _reserve_prefetched_render_read(self, presentation_key: int) -> bool:
        """Reserve transport re-entry for one presentation's live read ticket.

        The check is atomic with presentation attach and global-exclusive
        admission. It is valid only while ``presentation_key`` owns the current
        serialized transaction; callers cannot manufacture a reservation from
        outside that boundary.
        """

        with self._exclusive_gate:
            current = self._prefetched_read_reservation
            if current is not _PRESENTATION_UNSET:
                return current == presentation_key
            if (
                self._serialized_presentation_key != presentation_key
                or presentation_key not in self._presentations
                or len(self._presentations) != 1
                or self._exclusive_pending
            ):
                return False
            self._prefetched_read_reservation = presentation_key
            return True

    def _release_prefetched_render_read(self, presentation_key: int) -> bool:
        """Release ``presentation_key``'s logical transport reservation."""

        with self._exclusive_gate:
            if self._prefetched_read_reservation != presentation_key:
                return False
            self._prefetched_read_reservation = _PRESENTATION_UNSET
            self._exclusive_gate.notify_all()
            return True

    def _supports_split_render(self) -> bool:
        """Whether the active client exposes the bounded split-render API."""

        client = self._client
        return client is not None and all(
            callable(getattr(client, name, None))
            for name in (
                "submit_render_sample",
                "complete_render_sample",
                "discard_render_sample",
            )
        )

    def _supports_async_render_read(self) -> bool:
        """Whether the active client exposes the complete async-read API."""

        client = self._client
        if client is None:
            return False
        methods_present = all(
            callable(getattr(client, name, None))
            for name in (
                "begin_render_sample_read",
                "poll_render_sample_read",
                "cancel_render_sample_read",
            )
        )
        if not methods_present:
            return False
        supports = getattr(client, "supports_async_render_read", None)
        if supports is None:
            return True
        return bool(supports() if callable(supports) else supports)

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
        """Keep one presentation's mutation and readback contiguous, FIFO.

        A presentation with a live prefetched read retains a logical
        reservation after its physical lock scope ends. Other presentations
        remain FIFO-blocked while its owner may bypass the queue to drain or
        cancel that one ticket.
        """

        while True:
            waiter = object()
            with self._exclusive_gate:
                self._transport_waiters.append(waiter)
            cancelled_entry = False
            while True:
                caller_has_job = bool(exclusive_pending and exclusive_pending())
                with self._exclusive_gate:
                    reservation = self._prefetched_read_reservation
                    caller_owns_reservation = (
                        reservation is not _PRESENTATION_UNSET
                        and presentation_key == reservation
                    )
                    if cancelled and cancelled():
                        self._transport_waiters.remove(waiter)
                        self._exclusive_gate.notify_all()
                        cancelled_entry = True
                        reservation_reentry = caller_owns_reservation
                        caller_owns_exclusive = caller_owns_reservation
                        break
                    if caller_owns_reservation:
                        self._transport_waiters.remove(waiter)
                        reservation_reentry = True
                        caller_owns_exclusive = bool(
                            caller_has_job or self._exclusive_pending
                        )
                        break
                    if reservation is not _PRESENTATION_UNSET:
                        self._exclusive_gate.wait()
                        continue
                    if caller_has_job or (
                        not self._exclusive_pending
                        and self._transport_waiters[0] is waiter
                    ):
                        reservation_reentry = False
                        caller_owns_exclusive = bool(caller_has_job)
                        if caller_owns_exclusive:
                            self._transport_waiters.remove(waiter)
                        break
                    self._exclusive_gate.wait()
            if cancelled_entry and not reservation_reentry:
                yield False, True
                return
            if reservation_reentry or caller_owns_exclusive:
                try:
                    with self._transport_lock:
                        with self._exclusive_gate:
                            previous_serialized_key = (
                                self._serialized_presentation_key
                            )
                            self._serialized_presentation_key = presentation_key
                        try:
                            yield False, bool(
                                caller_owns_exclusive or cancelled_entry
                            )
                        finally:
                            with self._exclusive_gate:
                                self._serialized_presentation_key = (
                                    previous_serialized_key
                                )
                finally:
                    with self._exclusive_gate:
                        self._exclusive_gate.notify_all()
                return
            try:
                with self._transport_lock:
                    with self._exclusive_gate:
                        self._transport_waiters.remove(waiter)
                        reservation = self._prefetched_read_reservation
                        blocked_by_reservation = (
                            reservation is not _PRESENTATION_UNSET
                            and presentation_key != reservation
                        )
                        if self._exclusive_pending or blocked_by_reservation:
                            self._exclusive_gate.notify_all()
                            continue
                        previous_serialized_key = self._serialized_presentation_key
                        self._serialized_presentation_key = presentation_key
                    presentation_changed = (
                        presentation_key is not _PRESENTATION_UNSET
                        and self._presentation_key is not _PRESENTATION_UNSET
                        and presentation_key != self._presentation_key
                    )
                    try:
                        yield presentation_changed, False
                        if presentation_key is not _PRESENTATION_UNSET:
                            self._presentation_key = presentation_key
                    finally:
                        with self._exclusive_gate:
                            self._serialized_presentation_key = (
                                previous_serialized_key
                            )
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

        wake_reserved_owner = None
        with self._exclusive_gate:
            self._exclusive_pending += 1
            reservation = self._prefetched_read_reservation
            if reservation is not _PRESENTATION_UNSET:
                wake_reserved_owner = self._presentations.get(
                    reservation,
                    (None, None),
                )[0]
            self._exclusive_gate.notify_all()
        if wake_reserved_owner is not None:
            wake_reserved_owner()

    def _release_exclusive_transport(self) -> None:
        with self._exclusive_gate:
            self._exclusive_pending = max(0, self._exclusive_pending - 1)
            if not self._exclusive_pending:
                self._exclusive_gate.notify_all()

    def _has_pending_exclusive_transport(self) -> bool:
        """Whether any controller-wide exclusive owner is waiting or active."""

        with self._exclusive_gate:
            return bool(self._exclusive_pending)

    @contextmanager
    def _exclusive_transport(self):
        """Hold the shared runtime after any prefetched read is released."""

        while True:
            with self._exclusive_gate:
                while self._prefetched_read_reservation is not _PRESENTATION_UNSET:
                    self._exclusive_gate.wait()
            self._transport_lock.acquire()
            with self._exclusive_gate:
                if self._prefetched_read_reservation is _PRESENTATION_UNSET:
                    break
            self._transport_lock.release()
        try:
            yield
        finally:
            self._transport_lock.release()

    def would_replace(self, request: RenderRequest) -> str:
        """Reason the next :meth:`ensure` would replace the session.

        Read-only reuse probe for the render loop's replacement triggers
        (blender-live-render task02-06): returns ``""`` when the active
        session would be reused, otherwise the ``reuse_decision`` blocker
        (``output_shape_changed``, ``scene_composition_changed``,
        ``declared_sensors_changed``, ``camera_prim_changed``,
        ``camera_pose_override_removed``, ``render_var_changed``),
        ``runtime_binding_changed``, ``render_operation_failed``, or
        ``no_active_session``. Performs no RPCs, so it is deliberately not
        thread-guarded, and it mutates no controller state.
        """

        if self._closed or self._spec is None or self._simulation_id is None:
            return "no_active_session"
        if self._render_session_invalid_revision == self._session_revision:
            return "render_operation_failed"
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
            if self._render_session_invalid_revision == self._session_revision:
                reuse = {"reuse": False, "reason": "render_operation_failed"}
            elif self._runtime_binding == binding:
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
        self._render_session_invalid_revision = None
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
        try:
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
        except Exception as exc:
            raise self._render_retirement_error("render", exc) from exc
        self._render_timings = dict(getattr(client, "last_render_timings", {}))
        return result

    @_serialized
    def submit_render_sample(self, request: RenderRequest) -> RenderSubmission:
        """Submit one sample step while preserving controller confinement."""

        self._guard_rpc_thread("submit_render_sample")
        client, simulation_id = self._active()
        try:
            return self._submit_render_sample(client, simulation_id, request)
        except Exception as exc:
            raise self._render_retirement_error("render submission", exc) from exc

    @_serialized
    def submit_render_sample_after_transforms(
        self,
        request: RenderRequest,
        transforms: tuple[OvrtxTransformValue, ...],
        *,
        expected_session_revision: int,
    ) -> OvrtxPreparedRenderSubmission:
        """Write camera transforms and submit their sample as one transaction.

        This is the producer half of the viewport's bounded camera pipeline.
        It intentionally does not run the runtime scheduler: queued scene and
        simulation edits remain owned by the next normal viewport tick.
        """

        self._guard_rpc_thread("submit_render_sample_after_transforms")
        expected = int(expected_session_revision)
        if self._session_revision != expected:
            raise RenderClientError(
                "OVRTX session changed before preparing a camera successor"
            )
        client, simulation_id = self._active()
        try:
            update_result = OvrtxSessionUpdatePort(
                client,
                simulation_id,
            ).update_transforms(transforms)
            self._value_update_timings = dict(
                getattr(client, "last_value_update_timings", {})
            )
            if self._session_revision != expected:
                raise RenderClientError(
                    "OVRTX session changed while preparing a camera successor"
                )
            submission = self._submit_render_sample(client, simulation_id, request)
        except Exception as exc:
            # The transform RPC may have reached the bridge even when its
            # client-side outcome is ambiguous. Keep the predecessor read
            # alive so its owner can cancel/drain it, but make this session
            # non-reusable until that owner explicitly retires it.
            raise self._render_retirement_error(
                "camera successor mutation/submission",
                exc,
                session_revision=expected,
            ) from exc
        return OvrtxPreparedRenderSubmission(
            submission=submission,
            update_result=update_result,
            session_revision=expected,
        )

    @_serialized
    def retire_render_session(
        self,
        *,
        expected_session_revision: int,
    ) -> str:
        """Retire an ambiguous session after its predecessor read is drained."""

        self._guard_rpc_thread("retire_render_session")
        expected = int(expected_session_revision)
        if self._session_revision != expected:
            return "superseded"
        if self._render_session_invalid_revision != expected:
            return "not_required"
        return self._deactivate_active(preserve_complete_pose=True)

    def _render_retirement_error(
        self,
        operation: str,
        exc: BaseException,
        *,
        session_revision: int | None = None,
    ) -> OvrtxSessionRetirementRequiredError:
        revision = (
            self._session_revision
            if session_revision is None
            else int(session_revision)
        )
        self._render_session_invalid_revision = revision
        return OvrtxSessionRetirementRequiredError(
            f"OVRTX {operation} failed; session retirement is required: {exc}",
            controller=self,
            session_revision=revision,
            operation=operation,
        )

    @staticmethod
    def _submit_render_sample(
        client: Any,
        simulation_id: str,
        request: RenderRequest,
    ) -> RenderSubmission:
        submit = getattr(client, "submit_render_sample", None)
        if not callable(submit):
            raise RenderClientError("Active OVRTX client does not support split render submission")
        return submit(
            simulation_id,
            selected_sensor_paths=request.selected_sensor_paths,
            render_var=str(
                request.color_presentation.get(
                    "render_var",
                    color_presentation.RENDER_VAR_LDR_COLOR,
                )
            ),
        )

    @_serialized
    def complete_render_sample(self, submission: RenderSubmission) -> RenderResult:
        """Complete one exact-time split render submission."""

        self._guard_rpc_thread("complete_render_sample")
        client, _simulation_id = self._active()
        complete = getattr(client, "complete_render_sample", None)
        if not callable(complete):
            raise RenderClientError("Active OVRTX client does not support split render completion")
        try:
            result = complete(submission)
        except Exception as exc:
            raise self._render_retirement_error("render completion", exc) from exc
        self._render_timings = dict(getattr(client, "last_render_timings", {}))
        return result

    @_serialized
    def discard_render_sample(self, submission: RenderSubmission) -> None:
        """Release local ownership of a speculative split render result."""

        self._guard_rpc_thread("discard_render_sample")
        client, _simulation_id = self._active_for_cleanup()
        discard = getattr(client, "discard_render_sample", None)
        if not callable(discard):
            raise RenderClientError("Active OVRTX client does not support split render discard")
        discard(submission)

    @_serialized
    def begin_render_sample_read(
        self,
        submission: RenderSubmission,
    ) -> RenderReadTicket:
        """Begin an exact-time asynchronous read on the confined RPC thread."""

        self._guard_rpc_thread("begin_render_sample_read")
        client, _simulation_id = self._active()
        if not self._supports_async_render_read():
            raise RenderClientError(
                "Active OVRTX client does not support asynchronous render read"
            )
        try:
            return client.begin_render_sample_read(submission)
        except Exception as exc:
            raise self._render_retirement_error("render read begin", exc) from exc

    @_serialized
    def poll_render_sample_read(
        self,
        ticket: RenderReadTicket,
    ) -> RenderResult | None:
        """Poll an asynchronous render read without blocking."""

        self._guard_rpc_thread("poll_render_sample_read")
        client, _simulation_id = self._active()
        if not self._supports_async_render_read():
            raise RenderClientError(
                "Active OVRTX client does not support asynchronous render read"
            )
        try:
            result = client.poll_render_sample_read(ticket)
        except Exception as exc:
            raise self._render_retirement_error("render read poll", exc) from exc
        if result is not None:
            self._render_timings = dict(
                getattr(client, "last_render_timings", {})
            )
        return result

    @_serialized
    def cancel_render_sample_read(self, ticket: RenderReadTicket) -> None:
        """Cancel and consume an asynchronous render read on its owner thread."""

        self._guard_rpc_thread("cancel_render_sample_read")
        client, _simulation_id = self._active_for_cleanup()
        if not self._supports_async_render_read():
            raise RenderClientError(
                "Active OVRTX client does not support asynchronous render read"
            )
        try:
            client.cancel_render_sample_read(ticket)
        except Exception as exc:
            raise self._render_retirement_error("render read cancellation", exc) from exc

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
        with self._exclusive_gate:
            guard_active = self._thread_guard_active
            owning_thread_ident = self._owning_thread_ident
        if not guard_active:
            return
        ident = threading.get_ident()
        if ident != owning_thread_ident:
            raise OvrtxThreadConfinementError(
                f"OvrtxSessionController.{operation} was called from thread "
                f"{ident} ({threading.current_thread().name!r}); srtx RPCs are "
                f"confined to the owning render thread {owning_thread_ident}"
            )

    def _require_open(self) -> None:
        if self._closed:
            raise RenderClientError("OVRTX session controller is shut down")

    def _active(self) -> tuple[Any, str]:
        client, simulation_id = self._active_for_cleanup()
        if self._render_session_invalid_revision == self._session_revision:
            raise RenderClientError(
                "OVRTX session has an ambiguous render operation and must be replaced"
            )
        return client, simulation_id

    def _active_for_cleanup(self) -> tuple[Any, str]:
        """Return an active session even when only cleanup remains legal."""

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
            self._render_session_invalid_revision = None
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
        self._render_session_invalid_revision = None
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
    "OvrtxPreparedRenderSubmission",
    "OvrtxSessionRetirementRequiredError",
    "OvrtxSessionController",
    "OvrtxThreadConfinementError",
    "RPC_THREAD_GUARD_ENV",
]

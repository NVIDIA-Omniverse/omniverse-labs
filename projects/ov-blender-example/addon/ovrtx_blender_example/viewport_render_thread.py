# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-viewport-session render thread that owns all srtx RPCs.

Spec design decision (blender-live-render): native client thread-safety
across threads is unproven, so every srtx RPC is funneled through the one
render thread that owns the session. Other threads (Blender main, Blender
render job) interact only by submitting commands and reading published
state. Stdlib only: ``threading.Thread`` (daemon), ``queue.Queue``, and a
small event-based future wrapper — no asyncio, no concurrent.futures.

:class:`LatestViewRenderLoop` (task02-03) is the loop body that runs *on*
this thread under the latest-view contract (ADR 0013): take the newest
snapshot from the camera mailbox, apply pending value updates and the
scheduler tick, acquire one additional sample per eligible iteration toward
positive ``max_samples`` (or continuously when it is ``0``), and let newer
snapshots supersede in-flight refinement between acquisitions. All readback
polling runs here; when fully refined with no pending work the loop parks on
``mailbox.take(timeout=None)``.
"""

from __future__ import annotations

import hashlib
import queue
import threading
import time
import traceback
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import replace as _dataclass_replace
from typing import Any, Callable

from . import camera_value_conversion
from . import render_requests
from . import usd_paths
from . import user_messages
from . import viewport_profile
from .properties import RTPT_RENDER_SETTINGS
from .interactive_edit_planner import (
    DataAuthority,
    EditMechanism,
    EditShape,
    InteractiveEdit,
    InteractiveEditPlanner,
    edit_location,
)
from .ovrtx_runtime_client import (
    RenderClientError,
    RenderResult,
    RuntimeServicesPreparingError,
)
from .ovrtx_value_updates import OvrtxAttributeValue, OvrtxTransformValue
from .runtime_scheduler import RuntimeTickResult, RuntimeTickStatus
from .shared_stage_errors import SharedStageCompositionError
from .viewport_handoff import (
    FRAME_STATUS_FAILED,
    FRAME_STATUS_FRAME,
    FRAME_STATUS_RESYNCING,
    CameraRequestMailbox,
    FrameState,
    LatestFrameSlot,
    ViewSnapshot,
)

RENDER_THREAD_JOIN_TIMEOUT_SECONDS = 5.0

#: Pacing between retry attempts while the loop is in a failed state and
#: pending view updates keep requesting work. Without it a persistently
#: failing tick that returns before draining pending view updates would
#: busy-retry at ``timeout=0`` (the pending updates keep the park condition
#: false forever) — the task02-03 follow-up this constant closes.
FAILURE_RETRY_BACKOFF_SECONDS = 0.5

#: Pending-ensure reason recorded for the first session of a loop run.
SESSION_STARTUP_REASON = "startup"

#: Newest per-loop-iteration timing records retained for the artifact
#: (task02-09); matches the profile's recent-draw retention policy.
ITERATION_RECORD_LIMIT = viewport_profile.RECENT_DRAW_LIMIT

#: Span-boundary mark pairs mapped onto the existing profile phase names
#: for the render thread's per-iteration records (task02-09). The
#: cross-thread ``snapshot_to_render_start_ms`` span starts at the main
#: thread's mailbox write (``snapshot_written_monotonic_ns``); both ends
#: use ``time.perf_counter_ns()``.
ITERATION_TIMING_SPANS = {
    "ensure_session_ms": (
        "session_ensure_started_monotonic_ns",
        "session_ensure_completed_monotonic_ns",
    ),
    "composition_update_ms": (
        "runtime_update_started_monotonic_ns",
        "runtime_update_completed_monotonic_ns",
    ),
    "render_ms": (
        "render_call_started_monotonic_ns",
        "render_call_completed_monotonic_ns",
    ),
    "snapshot_to_render_start_ms": (
        "snapshot_written_monotonic_ns",
        "render_call_started_monotonic_ns",
    ),
}

#: ``ovrtx_session.reuse_decision`` reason string for an output resize.
#: Resize-triggered replacement is debounced: it starts only when two
#: consecutively taken snapshots agree on the new size (task02-06).
RESIZE_REPLACEMENT_REASON = "output_shape_changed"

#: Pending-ensure reason prefix for a camera value class the live-honor
#: probe just concluded unhonored (task04-05): the applied write was
#: accepted-but-ignored, so the loop schedules the replacement resync
#: immediately — the edit is never a silent no-op. The class's values fold
#: into the composition digest from here on.
CAMERA_VALUES_UNHONORED_REASON = "camera_values_unhonored"

#: Pending-ensure reason recorded when the worker rejects a live RTPT
#: render-setting write (render-quality-color-controls task01-04 fallback). The
#: live route is disabled for this session, the RTPT values fold back into the
#: composition digest, and the session re-keys through the background-resync
#: path so the recomposed layer re-authors the change — a rejected write never
#: kills the viewport. Note: on worker builds that also ignore the
#: ``omni:rtx:rtpt:*`` attributes on the RenderProduct (see runtime measurements,
#: "OVRTX worker ignores omni:rtx:rtpt:* render-quality attributes on the
#: RenderProduct"), the re-key re-authors the value but the render is unchanged.
RENDER_SETTING_UNHONORED_REASON = "render_setting_live_unhonored"


def _render_setting_fallback_message(reason: str, *, worker_owned: bool | None) -> str:
    """The one-per-session RTPT fallback WARNING, per worker-ownership case.

    This worker build ignores the ``omni:rtx:rtpt:*`` attributes on the
    ``RenderProduct`` and honors the same values only as ``/rtx/rtpt/*`` carb
    settings in the worker's startup config, read once at worker-process
    launch (runtime measurements). The add-on authors the current slider values
    into that config on every session ensure, so what the artist must do
    depends on who owns the worker process:

    - Owned (this Blender launched it): session teardown terminates the
      worker and the replacement ensure relaunches it, re-reading the config —
      the fallback's own background resync therefore applies the change
      automatically; the artist does nothing.
    - Attached (pre-existing/foreign worker): the add-on never terminates a
      worker it did not launch, so that worker keeps serving with its old
      launch-time settings; the artist must let it exit (quit Blender and
      make sure the OVRTX worker process has ended) before relaunching.
    """

    prefix = (
        "Live render-quality updates are not supported by this OVRTX worker "
        f"({reason}); re-syncing the session to re-author the change. Note: "
        "this worker build ignores the RTPT render-quality settings (bounce "
        "counts, firefly filter) on the render product and applies them only "
        "from the worker's startup config (/rtx/rtpt/*). "
    )
    if worker_owned:
        return prefix + (
            "Your current values are written there and the session re-sync "
            "restarts the add-on-managed worker, so the change applies "
            "automatically (runtime measurements)."
        )
    return prefix + (
        "Your current values are written there, but the running OVRTX worker "
        "was not launched by this Blender session and keeps its old settings "
        "across session restarts: quit Blender and ensure the OVRTX worker "
        "process has exited before relaunching (runtime measurements)."
    )


def render_result_digest(result: Any) -> str:
    """Stable digest of a rendered frame's presented pixels (task04-05).

    Used by the camera value live-honor probe to compare the pre-edit and
    post-edit ``min_samples`` frames. The dimensions and render var join
    the payload so equal byte-strings of different shapes never collide.
    """

    payload = getattr(result, "rgba8", b"") or b""
    header = (
        f"{int(getattr(result, 'width', 0))}x{int(getattr(result, 'height', 0))}"
        f":{getattr(result, 'render_var', '')}:"
    ).encode("utf-8")
    return hashlib.sha256(header + bytes(payload)).hexdigest()[:16]

STATUS_CREATED = "created"
STATUS_RUNNING = "running"
STATUS_STOPPED = "stopped"
STATUS_FAILED = "failed"


class RenderThreadError(RuntimeError):
    """Render-thread lifecycle misuse (for example starting twice)."""


class RenderThreadRejectedError(RenderThreadError):
    """Typed rejection: command submitted to a stopped or failed thread."""


class RenderThreadTimeoutError(RenderThreadError, TimeoutError):
    """Waiting on a render-thread result exceeded the caller's timeout."""


class RenderThreadResult:
    """Future-like handle for ``call``: completion event + value/exception."""

    __slots__ = ("_event", "_value", "_exception")

    def __init__(self) -> None:
        self._event = threading.Event()
        self._value: Any = None
        self._exception: BaseException | None = None

    def done(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)

    def exception(self, timeout: float | None = None) -> BaseException | None:
        if not self._event.wait(timeout):
            raise RenderThreadTimeoutError(
                "render thread result was not available in time"
            )
        return self._exception

    def result(self, timeout: float | None = None) -> Any:
        if not self._event.wait(timeout):
            raise RenderThreadTimeoutError(
                "render thread result was not available in time"
            )
        if self._exception is not None:
            raise self._exception
        return self._value

    def _resolve(self, value: Any) -> None:
        self._value = value
        self._event.set()

    def _reject(self, exception: BaseException) -> None:
        self._exception = exception
        self._event.set()


class _Command:
    __slots__ = ("fn", "result", "label")

    def __init__(
        self,
        fn: Callable[[], Any],
        result: RenderThreadResult | None,
        label: str,
    ) -> None:
        self.fn = fn
        self.result = result
        self.label = label


_STOP = object()


def _format_error(exception: BaseException) -> str:
    return "".join(traceback.format_exception(exception)).strip()


class ViewportRenderThread:
    """Owns all srtx RPCs for one viewport session via a command funnel.

    Commands run in submission order on one daemon thread named
    ``ovrtx-render-<session-id>``. ``submit`` is fire-and-forget: an
    unhandled exception is thread-fatal (status ``failed``, remaining
    commands rejected). ``call`` returns a :class:`RenderThreadResult`
    whose exception delivery *is* the handler, so a failed ``call`` does
    not take the thread down. ``stop`` performs a bounded join; a join
    timeout publishes a ``leaked_thread`` diagnostic and abandons the
    daemon thread (it exits with the process).
    """

    def __init__(
        self,
        session_id: str,
        *,
        join_timeout_seconds: float = RENDER_THREAD_JOIN_TIMEOUT_SECONDS,
    ) -> None:
        self._session_id = str(session_id)
        self._name = f"ovrtx-render-{self._session_id}"
        self._join_timeout_seconds = float(join_timeout_seconds)
        self._queue: queue.Queue[Any] = queue.Queue()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._status = STATUS_CREATED
        self._stop_requested = False
        self._leaked_thread = False
        self._failure = ""
        self._thread_ident = 0
        self._started_time_ns = 0
        self._started_monotonic_ns = 0
        self._ended_time_ns = 0
        self._commands_submitted = 0
        self._commands_completed = 0
        self._commands_failed = 0
        self._commands_rejected = 0
        self._pending_commands = 0
        self._last_command_label = ""
        self._last_command_ms = 0.0

    @property
    def name(self) -> str:
        return self._name

    @property
    def session_id(self) -> str:
        return self._session_id

    def status(self) -> str:
        with self._lock:
            return self._status

    def failure(self) -> str:
        with self._lock:
            return self._failure

    def is_alive(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def start(self) -> None:
        with self._lock:
            if self._stop_requested or self._status != STATUS_CREATED:
                raise RenderThreadError(
                    f"render thread {self._name} was already started or stopped"
                )
            thread = threading.Thread(target=self._run, name=self._name, daemon=True)
            self._thread = thread
            self._status = STATUS_RUNNING
            self._started_time_ns = time.time_ns()
            self._started_monotonic_ns = time.monotonic_ns()
        try:
            thread.start()
        except BaseException as exc:
            with self._lock:
                self._thread = None
                self._status = STATUS_FAILED
                self._failure = _format_error(exc)
                self._ended_time_ns = time.time_ns()
            # Commands queued before start would otherwise never resolve.
            self._drain_pending("render thread failed to start")
            raise

    def submit(self, fn: Callable[[], Any], *, label: str = "") -> None:
        """Fire-and-forget command; an unhandled exception fails the thread."""

        self._enqueue(fn, None, label)

    def call(self, fn: Callable[[], Any], *, label: str = "") -> RenderThreadResult:
        """Command with a future-like handle for the value or exception."""

        result = RenderThreadResult()
        self._enqueue(fn, result, label)
        return result

    def stop(self, timeout: float | None = None) -> dict[str, Any]:
        """Request stop and join with a bounded timeout.

        Returns the join outcome. On join timeout the thread is abandoned
        (daemon backstop) and ``leaked_thread`` is published in the outcome
        and in :meth:`diagnostics` for session-lifecycle reporting.
        """

        join_timeout = (
            self._join_timeout_seconds if timeout is None else float(timeout)
        )
        with self._lock:
            thread = self._thread
            self._stop_requested = True
            if thread is None:
                if self._status == STATUS_CREATED:
                    self._status = STATUS_STOPPED
                    self._ended_time_ns = time.time_ns()
            else:
                self._queue.put(_STOP)
        if thread is None:
            self._drain_pending("render thread was stopped before it started")
            return self._stop_outcome(joined=True, timeout=join_timeout)
        if thread is threading.current_thread():
            # A command asked its own thread to stop; the loop exits on the
            # sentinel. Joining here would deadlock.
            return self._stop_outcome(joined=False, timeout=join_timeout)
        thread.join(join_timeout)
        if thread.is_alive():
            with self._lock:
                self._leaked_thread = True
                if self._status == STATUS_RUNNING:
                    self._status = STATUS_FAILED
                    self._failure = (
                        "leaked_thread: render thread join did not complete "
                        f"within {join_timeout:.1f}s; daemon thread abandoned"
                    )
                    self._ended_time_ns = time.time_ns()
            return self._stop_outcome(joined=False, timeout=join_timeout)
        self._drain_pending("render thread is stopped")
        return self._stop_outcome(joined=True, timeout=join_timeout)

    def diagnostics(self) -> dict[str, Any]:
        """Thread identity/timing metadata for the viewport artifact recorder."""

        thread = self._thread
        with self._lock:
            return {
                "name": self._name,
                "session_id": self._session_id,
                "status": self._status,
                "alive": bool(thread is not None and thread.is_alive()),
                "daemon": True,
                "thread_ident": self._thread_ident,
                "stop_requested": self._stop_requested,
                "leaked_thread": self._leaked_thread,
                "failure": self._failure,
                "join_timeout_seconds": self._join_timeout_seconds,
                "started_time_ns": self._started_time_ns,
                "started_monotonic_ns": self._started_monotonic_ns,
                "ended_time_ns": self._ended_time_ns,
                "commands_submitted": self._commands_submitted,
                "commands_completed": self._commands_completed,
                "commands_failed": self._commands_failed,
                "commands_rejected": self._commands_rejected,
                "pending_commands": self._pending_commands,
                "last_command_label": self._last_command_label,
                "last_command_ms": self._last_command_ms,
            }

    def _enqueue(
        self,
        fn: Callable[[], Any],
        result: RenderThreadResult | None,
        label: str,
    ) -> None:
        if not callable(fn):
            raise TypeError("render thread commands must be callable")
        with self._lock:
            if self._stop_requested or self._status in (STATUS_STOPPED, STATUS_FAILED):
                self._commands_rejected += 1
                reason = self._failure or "the render thread is stopped"
                raise RenderThreadRejectedError(
                    f"render thread {self._name} rejected command"
                    f"{f' {label!r}' if label else ''}: {reason}"
                )
            self._commands_submitted += 1
            self._pending_commands += 1
            # Enqueue under the lock so no command can land behind the stop
            # sentinel (stop() sets the flag and enqueues the sentinel while
            # holding the same lock).
            self._queue.put(_Command(fn, result, str(label)))

    def _run(self) -> None:
        with self._lock:
            self._thread_ident = threading.get_ident()
        failure = ""
        try:
            while True:
                item = self._queue.get()
                if item is _STOP:
                    break
                command: _Command = item
                with self._lock:
                    self._pending_commands -= 1
                started_monotonic_ns = time.monotonic_ns()
                try:
                    value = command.fn()
                except BaseException as exc:
                    self._record_command(command, started_monotonic_ns, failed=True)
                    if command.result is not None:
                        # The waiting caller is the handler: deliver and
                        # keep serving commands.
                        command.result._reject(exc)
                        continue
                    failure = _format_error(exc)
                    break
                self._record_command(command, started_monotonic_ns, failed=False)
                if command.result is not None:
                    command.result._resolve(value)
        except BaseException as exc:
            failure = _format_error(exc)
        finally:
            with self._lock:
                self._stop_requested = True
                if failure:
                    self._status = STATUS_FAILED
                    self._failure = failure
                elif self._status == STATUS_RUNNING:
                    self._status = STATUS_STOPPED
                self._ended_time_ns = time.time_ns()
            self._drain_pending(
                f"render thread failed: {failure.splitlines()[-1]}"
                if failure
                else "render thread is stopped"
            )

    def _record_command(
        self,
        command: _Command,
        started_monotonic_ns: int,
        *,
        failed: bool,
    ) -> None:
        elapsed_ms = (time.monotonic_ns() - started_monotonic_ns) / 1_000_000.0
        with self._lock:
            if failed:
                self._commands_failed += 1
            else:
                self._commands_completed += 1
            self._last_command_label = command.label
            self._last_command_ms = elapsed_ms

    def _drain_pending(self, reason: str) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            if item is _STOP:
                continue
            with self._lock:
                self._commands_rejected += 1
                self._pending_commands -= 1
            if item.result is not None:
                item.result._reject(RenderThreadRejectedError(reason))

    def _stop_outcome(self, *, joined: bool, timeout: float) -> dict[str, Any]:
        with self._lock:
            return {
                "status": self._status,
                "joined": joined,
                "leaked_thread": self._leaked_thread,
                "join_timeout_seconds": timeout,
                "failure": self._failure,
            }


class RedrawSignalingFrameSlot:
    """:class:`LatestFrameSlot` decorator: publication drives presentation.

    Every publication that lands in the wrapped slot — frames, resyncing
    and failure states alike — invokes ``signal_redraw`` *after* the state
    is stored, so a main-thread redraw request can never observe the slot
    without the publication it announces (task02-05). The signal callable
    owns the coalescing policy (at most one pending redraw per engine);
    this wrapper only guarantees publish-before-signal ordering. All other
    attributes delegate to the wrapped slot.
    """

    def __init__(
        self,
        frame_slot: LatestFrameSlot,
        signal_redraw: Callable[[], None],
    ) -> None:
        self._frame_slot = frame_slot
        self._signal_redraw = signal_redraw

    def publish(self, frame_state: FrameState) -> FrameState:
        published = self._frame_slot.publish(frame_state)
        self._signal_redraw()
        return published

    def __getattr__(self, name: str) -> Any:
        return getattr(self._frame_slot, name)


class SessionLifecycleHooks:
    """Engine-provided callables for on-thread session lifecycle (task02-06).

    Every hook runs on the render thread. The loop owns *when* the session
    lifecycle acts (startup, replacement triggers, retry gating); the hooks
    own *how* (the ADR 0014 activation ordering — OVPhysX before OVRTX,
    break-before-make, no predecessor restoration on failure — is relocated
    into the hook implementations, not redesigned here).

    - ``ensure_session(request)``: create/replace the session for the
      snapshot-derived request. Raises ``RenderClientError`` /
      ``SharedStageCompositionError`` on failure (failure accounting —
      failure counts, startup diagnostics, crash-marker clearing — happens
      inside the hook).
    - ``replacement_reason(request)``: evaluate the replacement triggers
      (authored generation change, ``reuse_decision`` blockers) against the
      current session; return a reason string, or ``""`` for no
      replacement.
    - ``retry_allowed()``: whether a failed ensure may attempt again
      (``session_lifecycle.should_auto_retry`` projection; policy values
      unchanged).
    """

    __slots__ = ("ensure_session", "replacement_reason", "retry_allowed")

    def __init__(
        self,
        *,
        ensure_session: Callable[[Any], Any],
        replacement_reason: Callable[[Any], str],
        retry_allowed: Callable[[], bool],
    ) -> None:
        for name, hook in (
            ("ensure_session", ensure_session),
            ("replacement_reason", replacement_reason),
            ("retry_allowed", retry_allowed),
        ):
            if not callable(hook):
                raise TypeError(f"session lifecycle hook {name!r} must be callable")
        self.ensure_session = ensure_session
        self.replacement_reason = replacement_reason
        self.retry_allowed = retry_allowed


class LatestViewRenderLoop:
    """Latest-view render loop body for the per-session render thread.

    One instance per viewport session; :meth:`run` is designed to be the
    thread's long-lived ``submit`` command (an unexpected exception fails
    the thread per the 02-01 contract, while expected render/composition
    errors publish a failure state and keep the loop alive).

    Contract (ADR 0013 + task02-03 clarifications):

    - ``mailbox.take`` adopts input; a fresh snapshot resets
      refinement and renders ``min_samples`` first (the controller's
      existing snapshot-key reset). Stable viewport refinement uses one
      additional sample per native call so a newer view waits for at most
      one obsolete sample.
    - The in-flight render call completes and publishes visual feedback, then
      the newest pending snapshot is adopted before the next render call.
    - Camera pose changes apply as live ``omni:xform`` value updates inside
      the serialized presentation transaction, after the shared scheduler
      tick and before acquisition.
    - Park condition: snapshot stable AND a positive ``max_samples`` is
      complete AND no pending view updates AND the scheduler reports no due
      work → ``take(timeout=None)``. A zero limit remains continuously
      eligible. Wake sources: mailbox write, value-edit
      submission (``ViewUpdateStream`` wake hook → ``mailbox.wake``), and
      stop requests. While physics playback is active the loop waits one
      scheduler-tick interval instead of parking.
    - Loop errors (``RenderClientError``, ``SharedStageCompositionError``)
      publish :data:`~ovrtx_blender_example.viewport_handoff.FRAME_STATUS_FAILED`
      with detail and never kill the thread; retries follow the task02-06
      policy below.
    - Every successful tick's result hands off through ``tick_result_sink``
      (task02-07): physics pose sets and snapshot-derived timeline facts
      cross to the engine's main-thread pose-mirror timer; this loop never
      reads or writes Blender data. On loop exit an exact-stage scheduler is
      shut down on this thread before ``run`` returns. A current-scene loop
      only borrows the authoring runtime's scheduler and leaves it running.

    Session lifecycle (task02-06, with :class:`SessionLifecycleHooks`):

    - Session ensure/replace runs *on this loop*, triggered by mailbox and
      generation state — never by ``view_draw``. Startup ensure happens on
      the first adopted snapshot; replacement triggers (authored generation
      change, ``reuse_decision`` blockers) are evaluated on every adopted
      snapshot via ``replacement_reason``.
    - A replacement presents as a background resync: the loop publishes
      :data:`~ovrtx_blender_example.viewport_handoff.FRAME_STATUS_RESYNCING`
      before the replace RPCs, the main thread keeps presenting the last
      published frame, and the replacement session presents only after its
      first frame publishes (presentation gating, ADR 0014 readiness
      boundary).
    - Resize replacement is debounced: it starts only when two
      consecutively taken snapshots agree on the new size, so a drag-resize
      never causes a restart storm. Until then the session renders at its
      original size and the draw path scales by fit geometry.
    - Ensure failures publish a failure state and follow the existing
      ``session_lifecycle.should_auto_retry`` failure-count policy: the
      next snapshot retries while allowed, then the loop parks in the
      failed state (a session restart is the recovery path). Failed states
      with pending view updates retry paced by
      :data:`FAILURE_RETRY_BACKOFF_SECONDS`, never busy-polling.
    """

    def __init__(
        self,
        *,
        mailbox: CameraRequestMailbox,
        frame_slot: LatestFrameSlot,
        controller: Any = None,
        scheduler: Any,
        request_for_snapshot: Callable[[ViewSnapshot], Any],
        owns_scheduler: bool = True,
        lifecycle: SessionLifecycleHooks | None = None,
        failure_retry_backoff_seconds: float = FAILURE_RETRY_BACKOFF_SECONDS,
        tick_result_sink: Callable[[RuntimeTickResult, Any], None] | None = None,
        camera_value_probe: camera_value_conversion.CameraValueProbe | None = None,
        controller_provider: Callable[[], Any] | None = None,
    ) -> None:
        self._mailbox = mailbox
        self._frame_slot = frame_slot
        # Resolve the controller via a provider on every access: the authored
        # ensure (activate_for_viewport) can swap the engine's controller
        # mid-session, so a captured reference would keep rendering the
        # torn-down one. ``controller`` is still accepted (wrapped) for
        # direct/test callers that hold a single object.
        if controller_provider is None:
            controller_provider = lambda captured=controller: captured
        self._controller_provider = controller_provider
        self._scheduler = scheduler
        self._owns_scheduler = bool(owns_scheduler)
        self._wake_hook_setter: Callable[[Any], None] | None = None
        self._request_for_snapshot = request_for_snapshot
        self._lifecycle = lifecycle
        self._failure_retry_backoff_seconds = float(failure_retry_backoff_seconds)
        # Pose-mirror handoff seam (task02-07): called on this thread with
        # every successful tick result and the snapshot-derived request.
        # The sink must be data-only (no Blender reads/writes) and must not
        # raise — Blender-data application belongs to the engine's
        # main-thread timer on the other side of the handoff.
        self._tick_result_sink = tick_result_sink
        self._stop = threading.Event()
        self._run_lock = threading.Lock()
        self._running = False
        # Exclusive-job seam (task05-01): ``run`` occupies the render
        # thread's command queue for the whole session lifetime, so the F12
        # final-render job is funneled through the loop instead — jobs run
        # on the loop's thread *between* iterations (the viewport yields
        # while a job is queued/running and resumes afterwards), never
        # interleaved with a render step or an in-flight session
        # ensure/replacement.
        self._job_lock = threading.Lock()
        self._pending_jobs: list[_Command] = []
        self._jobs_closed = False
        self._job_active = False
        self._job_count = 0
        self._job_failure_count = 0
        self._job_rejection_count = 0
        self._last_job_label = ""
        # Loop state below is touched only on the render thread while
        # running; diagnostics reads from other threads are advisory.
        self._snapshot: ViewSnapshot | None = None
        self._restore_output_shape_needed = threading.Event()
        self._request: Any | None = None
        self._snapshot_key: tuple[Any, ...] | None = None
        self._completed_samples = 0
        self._current_result: RenderResult | None = None
        self._failed = False
        self._camera_update_needed = False
        # Camera value live-honor probe state (task04-05). The probe is
        # per viewport render session (this loop's lifetime): it persists
        # across background replacements so an unhonored class does not
        # re-probe after its own replacement resync.
        self._camera_value_probe = (
            camera_value_probe or camera_value_conversion.CameraValueProbe()
        )
        self._pending_camera_probe: dict[str, Any] | None = None
        #: Composed-camera attribute values currently in effect on the
        #: session (name -> value). ``None`` until the first baseline —
        #: the session's composed values, rebased on every ensure.
        self._applied_camera_values: dict[str, Any] | None = None
        #: Newest acquisition reaching ``min_samples`` after a refinement
        #: reset, kept by reference as the probe's pre-edit
        #: comparison frame; digested only when a probe actually runs.
        self._baseline_min_frame: dict[str, Any] | None = None
        self._camera_value_update_count = 0
        self._camera_probe_conclusion_count = 0
        # Live RTPT render-setting write capability (task01-04 fallback). Starts
        # optimistic; flips off the first time the worker rejects a live write.
        # Once off, the RTPT values ride the composition digest (rtpt_value_route
        # disabled) and every change re-keys the session (task01-03 path) instead
        # of attempting a runtime write. Advisory cross-thread read by the engine
        # so it stops dispatching would-be-rejected writes.
        self._rtpt_live_route_supported = True
        self._rtpt_fallback_reported = False
        self._render_setting_rejection_count = 0
        self._presented_scheduler_revision = 0
        self._observed_applied_revision = 0
        self._observed_session_revision = int(
            getattr(self._controller, "_session_revision", 0) or 0
        )
        self._snapshot_changed_pending = False
        self._tick_should_request_redraw = False
        self._last_tick_result: RuntimeTickResult | None = None
        self._last_timeline_reset = False
        self._generation = 0
        self._camera_controls_mode = render_requests.CAMERA_CONTROLS_USD
        self._last_reset_reason = ""
        self._last_failure_detail = ""
        # Session lifecycle state (task02-06; only meaningful with hooks).
        self._ensure_pending = lifecycle is not None
        self._pending_ensure_reason = (
            SESSION_STARTUP_REASON if lifecycle is not None else ""
        )
        self._ensure_failed = False
        # Set when a session start defers on the runtime services still
        # (re)starting (worker restart): a transient wait, not a failure. It
        # holds the loading state and paces the retry (see _take_timeout).
        self._ensure_deferred = False
        self._resync_recovery_pending = False
        self._last_snapshot_size: tuple[int, int] | None = None
        self._resize_debounce_pending = False
        self._session_ensure_count = 0
        self._session_replacement_count = 0
        self._ensure_failure_count = 0
        self._resync_publication_count = 0
        self._last_resync_reason = ""
        self._iterations = 0
        self._snapshots_taken = 0
        self._snapshots_superseded = 0
        self._camera_update_count = 0
        self._publication_count = 0
        self._failure_count = 0
        self._park_count = 0
        self._retry_wait_count = 0
        # Per-loop-iteration timing records (task02-09): appended only on
        # the render thread (list append, no locks on the hot path); the
        # main thread snapshots the list at artifact-write time
        # (single-writer aggregation).
        self._iteration_records: list[dict[str, Any]] = []
        self._iteration_record_count = 0

    def _active_controller(self) -> Any:
        """Resolve the current controller; raise the retryable render error
        (not an ``AttributeError``) if teardown has cleared it."""

        controller = self._controller_provider()
        if controller is None:
            raise RenderClientError("No active OVRTX session")
        return controller

    @property
    def _controller(self) -> Any:
        return self._active_controller()

    @property
    def stop_requested(self) -> bool:
        return self._stop.is_set()

    @property
    def last_tick_result(self) -> RuntimeTickResult | None:
        """Newest scheduler tick result (task02-07 pose-mirror handoff seam)."""

        return self._last_tick_result

    @property
    def rtpt_live_route_supported(self) -> bool:
        """Whether the worker still accepts live RTPT render-setting writes.

        Flips to ``False`` the first time the worker rejects a live write
        (task01-04 fallback). Read advisorily by the engine's main thread so it
        stops dispatching writes that would only be rejected; from then on RTPT
        changes re-key the session through the composition digest instead.
        """

        return self._rtpt_live_route_supported

    def request_stop(self) -> None:
        """Ask the loop to exit; safe from any thread, wakes a parked loop."""

        self._stop.set()
        self._mailbox.wake()
        wake_transport = getattr(self._controller, "_wake_serialized_transport", None)
        if callable(wake_transport):
            wake_transport()

    def call(self, fn: Callable[[], Any], *, label: str = "") -> RenderThreadResult:
        """Run ``fn`` on the loop's thread between iterations (task05-01).

        The exclusive-job funnel for the F12 final render: ``run`` is the
        render thread's long-lived command, so a raw
        :meth:`ViewportRenderThread.call` would queue behind it for the
        session lifetime. Jobs submitted here execute on the same RPC
        thread between loop iterations — atomically with respect to render
        steps and session ensure/replacement (both live inside
        :meth:`_iterate`) — while the viewport loop yields; it resumes
        normally afterwards, its session untouched. A job exception
        delivers to the returned future (the waiting caller is the
        handler) and never takes the loop or the thread down, mirroring
        the thread-level ``call`` contract. Jobs still pending when the
        loop exits are rejected with :class:`RenderThreadRejectedError`;
        submissions after stop/exit raise the same type.
        """

        if not callable(fn):
            raise TypeError("render loop jobs must be callable")
        result = RenderThreadResult()
        with self._job_lock:
            if self._stop.is_set() or self._jobs_closed:
                self._job_rejection_count += 1
                raise RenderThreadRejectedError(
                    "latest-view render loop rejected job"
                    f"{f' {label!r}' if label else ''}: the loop is stopped"
                )
            request_exclusive = getattr(
                self._controller, "_request_exclusive_transport", None
            )
            if callable(request_exclusive):
                request_exclusive()
            self._pending_jobs.append(_Command(fn, result, str(label)))
        # Wake a parked loop; the one-shot latch also covers a job landing
        # between the loop's timeout computation and its mailbox take.
        self._mailbox.wake()
        return result

    def _has_pending_jobs(self) -> bool:
        with self._job_lock:
            return bool(self._pending_jobs)

    def _run_pending_jobs(self) -> bool:
        """Run queued exclusive jobs in submission order (loop thread only).

        Jobs still queued once stop is requested are left for the exit
        rejection: running a long final render inside a session stop would
        risk the engine's bounded thread join. Returns whether a job ran.
        """

        ran_job = False
        while True:
            with self._job_lock:
                if not self._pending_jobs or self._stop.is_set():
                    return ran_job
                command = self._pending_jobs.pop(0)
                self._job_active = True
                ran_job = True
            self._job_count += 1
            self._last_job_label = command.label
            try:
                transaction = getattr(
                    self._controller, "_exclusive_transport", None
                )
                with transaction() if callable(transaction) else nullcontext():
                    try:
                        value = command.fn()
                    except BaseException as exc:
                        self._job_failure_count += 1
                        command.result._reject(exc)
                    else:
                        command.result._resolve(value)
                    if not self._stop.is_set():
                        self._schedule_replacement_after_job()
                    if self._ensure_pending and not self._stop.is_set():
                        self._run_pending_ensure({}, transport_owned=True)
            finally:
                release_exclusive = getattr(
                    self._controller, "_release_exclusive_transport", None
                )
                if callable(release_exclusive):
                    release_exclusive()
                with self._job_lock:
                    self._job_active = False

    def _schedule_replacement_after_job(self) -> None:
        """Re-ensure a viewport session suspended by an exclusive final job."""

        if self._lifecycle is None or self._request is None or self._ensure_pending:
            return
        try:
            reason = str(self._lifecycle.replacement_reason(self._request) or "")
        except (RenderClientError, SharedStageCompositionError):
            return
        if reason:
            self._ensure_pending = True
            self._pending_ensure_reason = reason

    def _reject_pending_jobs(self, reason: str) -> None:
        """Resolve every queued job future with a typed rejection."""

        with self._job_lock:
            self._jobs_closed = True
            pending = self._pending_jobs
            self._pending_jobs = []
            self._job_rejection_count += len(pending)
        for command in pending:
            release_exclusive = getattr(
                self._controller, "_release_exclusive_transport", None
            )
            if callable(release_exclusive):
                release_exclusive()
            command.result._reject(RenderThreadRejectedError(reason))

    def run(self) -> None:
        """Consume the mailbox until stop; designed as a thread ``submit``."""

        with self._run_lock:
            if self._running:
                raise RenderThreadError(
                    "latest-view render loop is already running"
                )
            self._running = True
        try:
            # Everything from adoption onward sits inside the try so a
            # thread-fatal setup failure still uninstalls the wake hook
            # and releases the running latch (the 02-01 contract fails
            # the thread; the loop instance must not lie about running).
            adopt = getattr(self._controller_provider(), "adopt_owning_thread", None)
            if callable(adopt):
                adopt()
            attach = getattr(self._controller, "_attach_presentation", None)
            if callable(attach):
                attach(
                    id(self),
                    self._mailbox.wake,
                    self._restore_native_output_shape,
                )
            hook_setter = (
                getattr(self._scheduler, "set_edit_wake_hook", None)
                if self._owns_scheduler
                else None
            )
            if callable(hook_setter):
                self._wake_hook_setter = hook_setter
                self._wake_hook_setter(self._mailbox.wake)
            while not self._stop.is_set():
                snapshot = self._mailbox.take(self._take_timeout())
                if self._stop.is_set():
                    break
                # Exclusive jobs (task05-01) run here, between iterations:
                # the viewport yields for the queued final-render work and
                # resumes with the snapshot below (mailbox writes made
                # while a job runs coalesce latest-wins as usual).
                ran_jobs = self._run_pending_jobs()
                if self._stop.is_set():
                    break
                if ran_jobs:
                    newest = self._mailbox.take(0.0)
                    if newest is not None:
                        snapshot = newest
                restore_output_shape = self._restore_output_shape_needed.is_set()
                if snapshot is not None:
                    self._adopt_snapshot(snapshot)
                    if restore_output_shape:
                        # Preserve newest-wins input, then confirm the native
                        # size once more for the existing resize debounce.
                        self._mailbox.wake()
                elif restore_output_shape and self._snapshot is not None:
                    self._restore_output_shape_needed.clear()
                    self._adopt_snapshot(self._snapshot)
                if self._request is None:
                    continue
                if snapshot is None and not self._work_due_without_new_snapshot():
                    # Woken without input (stale self-wake from the loop's
                    # own edit submission, or a timeout) and nothing due:
                    # re-evaluate the park condition instead of running a
                    # no-op (or failure-retrying) iteration.
                    continue
                self._iterate()
        finally:
            # Exclusive jobs never outlive the loop: reject queued futures
            # so an F12 job caught by a session stop resolves instead of
            # blocking its render job thread until the wait deadline.
            self._reject_pending_jobs("latest-view render loop exited")
            detach = getattr(self._controller, "_detach_presentation", None)
            if callable(detach):
                detach(id(self))
            if self._owns_scheduler and self._wake_hook_setter is not None:
                self._wake_hook_setter(None)
                self._wake_hook_setter = None
            # The exact-stage loop owns its standalone scheduler and shuts it
            # down on its RPC thread. Current-scene scheduler lifetime belongs
            # to the shared authoring runtime, not this presentation.
            shutdown = (
                getattr(self._scheduler, "shutdown", None)
                if self._owns_scheduler
                else None
            )
            if callable(shutdown):
                try:
                    shutdown()
                except Exception:
                    pass
            with self._run_lock:
                self._running = False

    def diagnostics(self) -> dict[str, Any]:
        snapshot = self._snapshot
        return {
            "running": self._running,
            "stop_requested": self._stop.is_set(),
            "failed_state": self._failed,
            "iterations": self._iterations,
            "snapshots_taken": self._snapshots_taken,
            "snapshots_superseded": self._snapshots_superseded,
            # Latest-view evidence (task02-09, ADR 0013): whether the last
            # adopted view reached full refinement before idle, plus the
            # mailbox's count of distinct pending views replaced before
            # adoption (see the artifact's ``latest_view`` block).
            "final_view_refined": bool(
                snapshot is not None
                and not self._failed
                and int(snapshot.max_samples) > 0
                and self._completed_samples >= int(snapshot.max_samples)
            ),
            "iteration_record_count": self._iteration_record_count,
            "mailbox": self._mailbox.diagnostics(),
            "camera_update_count": self._camera_update_count,
            "camera_value_update_count": self._camera_value_update_count,
            "camera_probe_conclusions": self._camera_probe_conclusion_count,
            "camera_value_probe": self._camera_value_probe.diagnostics(),
            "publications": self._publication_count,
            "failures": self._failure_count,
            "parks": self._park_count,
            "retry_waits": self._retry_wait_count,
            "exclusive_jobs": self._job_count,
            "exclusive_job_failures": self._job_failure_count,
            "exclusive_job_rejections": self._job_rejection_count,
            "exclusive_job_active": self._job_active,
            "exclusive_jobs_pending": len(self._pending_jobs),
            "last_exclusive_job_label": self._last_job_label,
            "session_ensures": self._session_ensure_count,
            "session_replacements": self._session_replacement_count,
            # Live RTPT render-setting fallback evidence (task01-04): whether
            # the worker still honors live writes, and how many were rejected
            # (each folds into the digest and re-keys instead of killing the
            # viewport).
            "rtpt_live_route_supported": self._rtpt_live_route_supported,
            "render_setting_rejections": self._render_setting_rejection_count,
            "presented_scheduler_revision": self._presented_scheduler_revision,
            "observed_applied_revision": self._observed_applied_revision,
            "ensure_failures": self._ensure_failure_count,
            "ensure_pending": self._ensure_pending,
            "resync_publications": self._resync_publication_count,
            "last_resync_reason": self._last_resync_reason,
            "resize_debounce_pending": self._resize_debounce_pending,
            "retry_blocked": bool(self._failed and not self._retry_allowed()),
            "completed_samples": self._completed_samples,
            "max_samples": int(snapshot.max_samples) if snapshot is not None else 0,
            "generation": self._generation,
            "last_timeline_reset": self._last_timeline_reset,
            "camera_controls_mode": self._camera_controls_mode,
            "last_reset_reason": self._last_reset_reason,
            "last_failure_detail": self._last_failure_detail,
        }

    def iteration_records(self) -> list[dict[str, Any]]:
        """Snapshot of the render thread's per-iteration records (task02-09).

        Safe to call from the main thread: the render thread only appends
        fully-built records (never mutated after append), so a list
        snapshot observes complete dicts. Aggregation of these records
        happens on the main thread at artifact-write time.
        """

        return [dict(record) for record in list(self._iteration_records)]

    def _record_iteration(
        self,
        status: str,
        marks: dict[str, int],
        *,
        publication_index: int = 0,
        completed_samples: int = 0,
        tick_result: RuntimeTickResult | None = None,
        detail: str = "",
        reset_reason: str = "",
    ) -> None:
        """Append one render-thread timing record (loop thread only)."""

        timings: dict[str, float] = {}
        for phase, (started_name, completed_name) in ITERATION_TIMING_SPANS.items():
            started_ns = marks.get(started_name)
            completed_ns = marks.get(completed_name)
            if started_ns and completed_ns and int(completed_ns) >= int(started_ns):
                timings[phase] = (int(completed_ns) - int(started_ns)) / 1_000_000.0
        if tick_result is not None:
            update_result = tick_result.update.get("update_result")
            if isinstance(update_result, Mapping):
                try:
                    timings["camera_update_ms"] = float(
                        update_result.get("value_apply_ms", 0.0)
                    )
                except (TypeError, ValueError):
                    pass
        record: dict[str, Any] = {
            "thread": "render",
            "status": str(status),
            "iteration": self._iterations,
            "publication_index": int(publication_index),
            "snapshot_key": viewport_profile.snapshot_key_token(self._snapshot_key),
            "completed_samples": int(completed_samples),
            "generation": self._generation,
            "timings_ms": timings,
            "span_boundaries": {name: int(value) for name, value in marks.items()},
        }
        if reset_reason:
            # Refinement-reset reason for this applied batch (task04-06).
            record["reset_reason"] = str(reset_reason)
        if detail:
            record["detail"] = detail
        self._iteration_record_count += 1
        self._iteration_records.append(record)
        if len(self._iteration_records) > ITERATION_RECORD_LIMIT:
            del self._iteration_records[: len(self._iteration_records) - ITERATION_RECORD_LIMIT]

    def _take_timeout(self) -> float | None:
        """Latest-view park policy: 0 while work is due, else cadence/park."""

        if self._has_pending_jobs():
            # A queued exclusive job (task05-01) must run promptly even on
            # a fully-refined view. Defense-in-depth: the submitter's
            # mailbox wake latch already covers a parked loop; the park
            # policy should agree rather than rely on that latch.
            return 0.0
        if self._request is not None:
            if self._controller_session_due():
                return 0.0
            if self._failed:
                if self._retry_allowed() and bool(
                    getattr(self._scheduler, "has_pending_view_updates", False)
                    or getattr(self._scheduler, "has_pending_sim_updates", False)
                ):
                    # Pending edits may fix the failure, but a tick that
                    # fails without draining them must not busy-loop at 0:
                    # pace retries by the backoff interval (task02-06).
                    self._retry_wait_count += 1
                    return self._failure_retry_backoff_seconds
                # A failed view parks until fresh input (snapshot write or
                # edit wake) — or indefinitely once the auto-retry policy
                # is exhausted (a session restart is the recovery path).
                self._park_count += 1
                return None
            if self._ensure_pending:
                if self._ensure_deferred:
                    # Runtime services (re)starting: pace the retry by the
                    # backoff rather than busy-polling at 0 while they come up.
                    self._retry_wait_count += 1
                    return self._failure_retry_backoff_seconds
                # A pending ensure scheduled mid-iteration (the camera
                # value probe concluding unhonored, task04-05) must run
                # promptly even when refinement is already complete.
                # Defense-in-depth: the probe's own edit submission also
                # leaves the mailbox wake latched, but the park policy
                # should agree with _work_due_without_new_snapshot rather
                # than rely on that latch.
                return 0.0
            if bool(getattr(self._scheduler, "has_pending_view_updates", False)):
                return 0.0
            if self._scheduler_presentation_due():
                return 0.0
            if self._scheduler_applied_due():
                return 0.0
            if render_requests.viewport_sampling_due(
                self._completed_samples, self._snapshot.max_samples
            ):
                # Active refinement: no sleeping between the publication
                # and the next snapshot take.
                return 0.0
        self._park_count += 1
        return None

    def _work_due_without_new_snapshot(self) -> bool:
        """Whether an input-less wake still has work: mirrors the park policy."""

        if self._controller_session_due():
            return True
        if self._failed:
            # Failed views retry on fresh input, or — paced by the retry
            # backoff — while pending edits keep requesting work and the
            # auto-retry policy still allows an attempt.
            return self._retry_allowed() and bool(
                getattr(self._scheduler, "has_pending_view_updates", False)
                or getattr(self._scheduler, "has_pending_sim_updates", False)
            )
        if self._ensure_pending:
            return True
        if bool(getattr(self._scheduler, "has_pending_view_updates", False)):
            return True
        if self._scheduler_presentation_due():
            return True
        if self._scheduler_applied_due():
            return True
        if bool(getattr(self._scheduler, "has_pending_sim_updates", False)):
            # A sim-edit wake runs one tick so the pending initial-condition
            # value applies. Deliberately absent from _take_timeout's
            # timeout=0 poll condition: a tick that cannot drain sim
            # pendings (physics not enabled) must re-park instead of
            # busy-polling; the next edit wake retries.
            return True
        if render_requests.viewport_sampling_due(
            self._completed_samples, self._snapshot.max_samples
        ):
            return True
        return self._tick_should_request_redraw

    def _retry_allowed(self) -> bool:
        """Whether a failed state may attempt again.

        Ensure failures follow the engine's auto-retry policy
        (``session_lifecycle.should_auto_retry``); render/tick failures
        always retry on fresh input, mirroring the historical per-draw
        retry behavior.
        """

        if not self._ensure_failed or self._lifecycle is None:
            return True
        return bool(self._lifecycle.retry_allowed())

    def _scheduler_presentation_revision(self) -> int:
        return int(getattr(self._scheduler, "presentation_revision", 0) or 0)

    def _scheduler_presentation_due(self) -> bool:
        return (
            self._scheduler_presentation_revision()
            > self._presented_scheduler_revision
        )

    def _scheduler_applied_revision(self) -> int:
        return int(getattr(self._scheduler, "applied_revision", 0) or 0)

    def _scheduler_applied_due(self) -> bool:
        return self._scheduler_applied_revision() > self._observed_applied_revision

    def _controller_session_revision(self) -> int:
        return int(getattr(self._controller, "_session_revision", 0) or 0)

    def _controller_session_due(self) -> bool:
        return self._controller_session_revision() > self._observed_session_revision

    def _serialized_viewport_iteration(self):
        transaction = getattr(
            self._controller,
            "_serialized_transport",
            None,
        )
        return (
            transaction(
                presentation_key=id(self),
                exclusive_pending=self._has_pending_jobs,
                cancelled=self._stop.is_set,
            )
            if callable(transaction)
            else nullcontext((False, False))
        )

    def _adopt_snapshot(self, snapshot: ViewSnapshot) -> None:
        previous = self._snapshot
        key = snapshot.key
        key_changed = key != self._snapshot_key
        self._snapshot = snapshot
        self._request = self._with_shared_output_shape(
            self._with_rtpt_digest_route(
                self._with_camera_value_route(self._request_for_snapshot(snapshot))
            )
        )
        self._snapshots_taken += 1
        # Fresh input clears a failure latch when the retry policy allows
        # it: render failures always retry on a new snapshot (the per-draw
        # retry this mirrors); ensure failures follow
        # ``session_lifecycle.should_auto_retry``.
        if self._failed and self._retry_allowed():
            self._failed = False
        self._evaluate_replacement(snapshot)
        if not key_changed:
            # Same view identity (timeline/cursor fields may still differ
            # and flow into the next tick request) — no refinement reset.
            return
        if previous is not None and render_requests.viewport_sampling_due(
            self._completed_samples, previous.max_samples
        ):
            self._snapshots_superseded += 1
        self._cancel_camera_probe(
            camera_value_conversion.PROBE_INCONCLUSIVE_POSE_CHANGED
        )
        self._snapshot_key = key
        self._completed_samples = 0
        self._current_result = None
        self._snapshot_changed_pending = True
        self._refresh_camera_state()

    def _with_shared_output_shape(self, request: Any) -> Any:
        """Use one session shape while several panes borrow the controller."""

        output_shape = getattr(self._controller, "_shared_output_shape", None)
        shape = output_shape() if callable(output_shape) else None
        if shape is None or (request.width, request.height) == shape:
            return request
        try:
            return _dataclass_replace(request, width=shape[0], height=shape[1])
        except TypeError:
            return request

    def _restore_native_output_shape(self) -> None:
        """Re-adopt the pane snapshot when it becomes the sole presentation."""

        self._restore_output_shape_needed.set()
        self._mailbox.wake()

    def _refresh_camera_state(self) -> None:
        """Flag the newest snapshot's pose for (re)application."""

        snapshot = self._snapshot
        self._camera_update_needed = snapshot.camera_matrix is not None and usd_paths.known_usd_path(
            snapshot.camera_prim_path
        )
        self._camera_controls_mode = (
            render_requests.CAMERA_CONTROLS_BLENDER_VIEW
            if self._camera_update_needed
            else render_requests.CAMERA_CONTROLS_USD
        )

    def _with_camera_value_route(self, request: Any) -> Any:
        """Stamp the probe's current value-route classes onto a request.

        The request is what session identity derives from (``build_spec``
        → composition digest), so the route classes must ride it: honored
        and not-yet-probed classes stay out of the digest, unhonored
        classes fold their values in so ``reuse_decision`` forces the
        replacement (task04-05). Requests without the field (injected
        fakes) pass through unchanged.
        """

        route = self._camera_value_probe.value_route_classes()
        current = getattr(request, "camera_value_route_classes", None)
        if current is None or tuple(current) == route:
            return request
        try:
            return _dataclass_replace(request, camera_value_route_classes=route)
        except TypeError:
            return request

    def _with_rtpt_digest_route(self, request: Any) -> Any:
        """Fold RTPT values into the composition digest once the live route is off.

        While the worker honors live render-setting writes the request keeps
        ``rtpt_value_route=True`` (RTPT excluded from the digest, so a change
        does not replace the session). Once a write is rejected the route flips
        off for the session's lifetime, so every rebuilt request carries
        ``rtpt_value_route=False`` — RTPT rejoins the digest and a changed value
        re-keys the session (task01-03). Requests without the field (injected
        fakes) pass through unchanged.
        """

        if self._rtpt_live_route_supported:
            return request
        if getattr(request, "rtpt_value_route", None) in (None, False):
            return request
        try:
            return _dataclass_replace(request, rtpt_value_route=False)
        except TypeError:
            return request

    def _apply_rtpt_values_to_request(self, request: Any, values: Any) -> Any:
        """Patch the rejected live RTPT values into the request's ``rtpt_quality``.

        The re-key composes ``request.rtpt_quality`` (task01-03), so the pending
        change the worker rejected is written into it — keyed by
        ``RTPT_RENDER_SETTINGS`` property name via the authored attribute — so
        the recomposed session authors the new value rather than dropping it.

        The rejected write carries the *wire* value (what was sent to OVRTX);
        ``rtpt_quality`` holds artist-facing UI values, so it is converted back
        with ``spec.from_wire`` (the inverse of the +2 Max Bounces remap). Composition
        then re-applies ``to_wire`` and authors the same wire value the artist set.
        """

        quality = dict(getattr(request, "rtpt_quality", {}) or {})
        property_by_attribute = {
            spec.attribute: (name, spec) for name, spec in RTPT_RENDER_SETTINGS.items()
        }
        patched = False
        for value in values or ():
            if not isinstance(value, Mapping):
                continue
            resolved = property_by_attribute.get(str(value.get("attribute", "")))
            if resolved is None:
                continue
            property_name, spec = resolved
            quality[property_name] = spec.from_wire(value.get("value"))
            patched = True
        if not patched:
            return request
        try:
            return _dataclass_replace(request, rtpt_quality=quality)
        except TypeError:
            return request

    def _handle_render_setting_rejection(self, rejection: Mapping[str, Any]) -> None:
        """Fall back to session re-keying when a live RTPT write is rejected.

        A rejected runtime render-setting write must never kill the viewport
        (task01-04): the scheduler already de-fataled the tick, so here the
        loop disables the live route for the session, patches the pending RTPT
        values into the request so the re-key authors them, folds them into the
        composition digest, and schedules the background-resync replacement.
        Reported once per session through the user-messages bus as a WARNING.
        """

        self._render_setting_rejection_count += 1
        self._report_render_setting_fallback(rejection)
        if not self._rtpt_live_route_supported:
            # Already folded into the digest: subsequent changes re-key through
            # the normal replacement path (task01-03), no extra resync needed.
            return
        self._rtpt_live_route_supported = False
        self._request = self._with_rtpt_digest_route(
            self._apply_rtpt_values_to_request(
                self._request, rejection.get("values", ())
            )
        )
        if self._lifecycle is not None and not self._ensure_pending:
            self._ensure_pending = True
            self._pending_ensure_reason = RENDER_SETTING_UNHONORED_REASON

    def _report_render_setting_fallback(self, rejection: Mapping[str, Any]) -> None:
        """Emit the one-per-session fallback WARNING (overlay/Info/console bus)."""

        if self._rtpt_fallback_reported:
            return
        self._rtpt_fallback_reported = True
        reason = str(rejection.get("skipped_reason", "") or "render_setting_value_update_error")
        message = _render_setting_fallback_message(
            reason,
            worker_owned=getattr(self._controller_provider(), "worker_owned", None),
        )
        try:
            # Once-per-session is already guaranteed by the
            # ``_rtpt_fallback_reported`` instance flag above, so bus-level
            # change-dedup is redundant here and would only add a hazard: the
            # bus keeps per-context dedup state for the process lifetime while
            # ``id(self)`` can be reused after a prior loop is GC'd, which would
            # silently swallow a later/new session's identical warning. Emit
            # unconditionally so every session's fallback is reported.
            user_messages.report_warning(
                message, context=f"rtpt-live-fallback:{id(self)}", dedup=False
            )
        except Exception:
            # The user-message bus must never take down the render loop.
            pass

    def _rebase_camera_values(self) -> None:
        """Adopt the current request's composed camera values as applied."""

        self._cancel_camera_probe(
            camera_value_conversion.PROBE_INCONCLUSIVE_POSE_CHANGED
        )
        self._baseline_min_frame = None
        self._applied_camera_values = {
            attribute.name: attribute.value
            for attribute in camera_value_conversion.usd_attribute_values(
                getattr(self._request, "camera_projection", None)
            )
        }

    def _pending_camera_value_edits(
        self,
    ) -> tuple[dict[str, Any] | None, tuple[OvrtxAttributeValue, ...]]:
        """Build presentation-local camera projection value updates.

        Returns a probe context when the edit doubles as
        the class's live-honor probe (the pre-edit ``min_samples`` frame
        was available at the same snapshot key), otherwise ``None``. The
        viewport retains the context until post-edit acquisition reaches
        the same sample count.
        """

        request = self._request
        attributes = camera_value_conversion.usd_attribute_values(
            getattr(request, "camera_projection", None)
        )
        if self._applied_camera_values is None:
            # First adoption without a lifecycle ensure (hook-less loops):
            # the session was composed outside the loop from these same
            # values — they are the baseline, not edits.
            self._applied_camera_values = {
                attribute.name: attribute.value for attribute in attributes
            }
            return None, ()
        if not attributes:
            return None, ()
        snapshot = self._snapshot
        if not usd_paths.known_usd_path(snapshot.camera_prim_path):
            return None, ()
        changed: dict[str, list[Any]] = {}
        for attribute in attributes:
            probe_class = str(attribute.metadata.get("probe_class", ""))
            if not probe_class:
                continue
            if self._applied_camera_values.get(attribute.name) == attribute.value:
                continue
            changed.setdefault(probe_class, []).append(attribute)
        if not changed:
            return None, ()
        self._cancel_camera_probe(
            camera_value_conversion.PROBE_INCONCLUSIVE_CONCURRENT_EDITS
        )
        probe = self._camera_value_probe
        unknown_classes = [
            probe_class
            for probe_class in sorted(changed)
            if probe.status(probe_class)
            == camera_value_conversion.PROBE_STATUS_UNKNOWN
        ]
        probe_context: dict[str, Any] | None = None
        if unknown_classes:
            if len(changed) > 1:
                # Several classes' values apply in one tick: a digest
                # difference could not be attributed to one class, so the
                # unknown ones retry on their next isolated edit.
                for probe_class in unknown_classes:
                    probe.record_inconclusive(
                        probe_class,
                        camera_value_conversion.PROBE_INCONCLUSIVE_CONCURRENT_EDITS,
                    )
            else:
                probe_context = self._begin_camera_probe(
                    unknown_classes[0], changed[unknown_classes[0]]
                )
        values: list[OvrtxAttributeValue] = []
        for probe_class in sorted(changed):
            if (
                probe.status(probe_class)
                == camera_value_conversion.PROBE_STATUS_UNHONORED
            ):
                # Unhonored class: its values are folded into the
                # composition digest (the route classes exclude it), so
                # the replacement triggers own this change — never a
                # value write, never a silent no-op.
                continue
            for attribute in changed[probe_class]:
                values.append(
                    OvrtxAttributeValue(
                        snapshot.camera_prim_path,
                        attribute.name,
                        attribute.value,
                        attribute.value_type,
                    )
                )
        return probe_context, tuple(values)

    def _cancel_camera_probe(self, reason: str) -> None:
        context = self._pending_camera_probe
        if context is None:
            return
        self._camera_value_probe.record_inconclusive(
            str(context["probe_class"]),
            reason,
        )
        self._pending_camera_probe = None

    def _begin_camera_probe(
        self,
        probe_class: str,
        attributes: list[Any],
    ) -> dict[str, Any] | None:
        """Start the class's live-honor probe if the guards allow it.

        Guards (each records an inconclusive reason and retries on the
        next edit of the class — the edit itself still applies):

        - physics playback advancing content between frames,
        - other pending edits that would apply in the same tick,
        - no pre-edit ``min_samples`` frame, or one captured at a
          different snapshot key (the camera pose changed mid-probe).
        """

        probe = self._camera_value_probe
        if self._tick_should_request_redraw or bool(
            getattr(self._snapshot, "timeline_playing", False)
        ):
            # ``_tick_should_request_redraw`` is the previous tick's
            # verdict; the snapshot's playing flag additionally covers
            # playback starting in this very tick — either way the frame
            # content may change for reasons other than the camera edit.
            probe.record_inconclusive(
                probe_class,
                camera_value_conversion.PROBE_INCONCLUSIVE_PHYSICS_ACTIVE,
            )
            return None
        if bool(getattr(self._scheduler, "has_pending_view_updates", False)) or bool(
            getattr(self._scheduler, "has_pending_sim_updates", False)
        ):
            probe.record_inconclusive(
                probe_class,
                camera_value_conversion.PROBE_INCONCLUSIVE_CONCURRENT_EDITS,
            )
            return None
        baseline = self._baseline_min_frame
        if baseline is None:
            probe.record_inconclusive(
                probe_class,
                camera_value_conversion.PROBE_INCONCLUSIVE_NO_BASELINE,
            )
            return None
        if baseline["key"] != self._snapshot_key:
            probe.record_inconclusive(
                probe_class,
                camera_value_conversion.PROBE_INCONCLUSIVE_POSE_CHANGED,
            )
            return None
        min_samples = int(self._snapshot.min_samples)
        if int(baseline["samples"]) != min_samples:
            probe.record_inconclusive(
                probe_class,
                camera_value_conversion.PROBE_INCONCLUSIVE_SAMPLE_MISMATCH,
            )
            return None
        attempt = probe.begin_attempt(probe_class)
        return {
            "probe_class": probe_class,
            "attempt": attempt,
            "pre_result": baseline["result"],
            "pre_samples": int(baseline["samples"]),
            "snapshot_key": self._snapshot_key,
            "attributes": {
                attribute.name: attribute.value for attribute in attributes
            },
        }

    def _capture_baseline_min_frame(self, result: RenderResult | None) -> None:
        """Remember the newest ``min_samples`` step as the probe baseline.

        An acquisition completing exactly ``min_samples`` is the comparison
        milestone after a refinement reset — the freshest full picture of the session's
        current content. Kept by reference (immutable ``rgba8`` bytes);
        digested only when a probe runs, so sessions that never edit
        camera values never pay for hashing.
        """

        if result is None:
            return
        snapshot = self._snapshot
        if int(result.completed_samples) != int(snapshot.min_samples):
            return
        self._baseline_min_frame = {
            "key": self._snapshot_key,
            "samples": int(result.completed_samples),
            "result": result,
        }

    def _conclude_camera_probe(
        self,
        probe_context: dict[str, Any],
        result: RenderResult | None,
    ) -> None:
        """Compare the post-edit frame against the pre-edit frame.

        Digest-different at equal samples and an unchanged snapshot key →
        the class is honored (value route stays). Digest-equal → unhonored:
        the class folds into the composition digest and a replacement
        resync is scheduled immediately so this very edit still renders
        correctly (acceptance: no camera edit is silently ignored).
        """

        probe = self._camera_value_probe
        probe_class = str(probe_context["probe_class"])
        if (
            result is None
            or int(result.completed_samples) != int(probe_context["pre_samples"])
        ):
            probe.record_inconclusive(
                probe_class,
                camera_value_conversion.PROBE_INCONCLUSIVE_SAMPLE_MISMATCH,
            )
            return
        pre_digest = render_result_digest(probe_context["pre_result"])
        post_digest = render_result_digest(result)
        honored = pre_digest != post_digest
        self._camera_probe_conclusion_count += 1
        probe.record_result(
            probe_class,
            honored=honored,
            evidence={
                "pre_frame_digest": pre_digest,
                "post_frame_digest": post_digest,
                "compared_samples": int(probe_context["pre_samples"]),
                "snapshot_key": viewport_profile.snapshot_key_token(
                    probe_context["snapshot_key"]
                ),
                "attributes": dict(probe_context["attributes"]),
                "attempt": int(probe_context["attempt"]),
                "generation": self._generation,
            },
        )
        if honored:
            return
        # Fold the class into session identity and resync in the
        # background: the accepted-but-ignored write must not present.
        self._request = self._with_camera_value_route(self._request)
        if self._lifecycle is not None:
            self._ensure_pending = True
            self._pending_ensure_reason = (
                f"{CAMERA_VALUES_UNHONORED_REASON}:{probe_class}"
            )

    def _evaluate_replacement(self, snapshot: ViewSnapshot) -> None:
        """Evaluate the task02-06 replacement triggers for a taken snapshot.

        Runs on the thread from snapshot + generation state. A resize
        (``output_shape_changed``) is debounced by two consecutively taken
        same-size snapshots so drag-resizes never cause restart storms; all
        other reasons (generation change, composition digest, declared
        sensors, camera prim, pose-source downgrade) schedule the
        replacement immediately.
        """

        size = (int(snapshot.width), int(snapshot.height))
        size_confirmed = self._last_snapshot_size == size
        self._last_snapshot_size = size
        if self._lifecycle is None or self._ensure_pending:
            return
        try:
            reason = str(self._lifecycle.replacement_reason(self._request) or "")
        except (RenderClientError, SharedStageCompositionError):
            # A broken composition surfaces through the ensure/tick paths
            # with a published failure state; the probe itself must not
            # kill the loop or force a replacement.
            return
        if not reason:
            self._resize_debounce_pending = False
            return
        if reason == RESIZE_REPLACEMENT_REASON and not size_confirmed:
            self._resize_debounce_pending = True
            return
        self._resize_debounce_pending = False
        self._ensure_pending = True
        self._pending_ensure_reason = reason

    def _iterate(self) -> None:
        """One tick + one acquisition + one publication (or failure state)."""

        self._iterations += 1
        snapshot = self._snapshot
        marks: dict[str, int] = dict(snapshot.timing_marks)
        marks["snapshot_written_monotonic_ns"] = snapshot.written_monotonic_ns
        if not self._run_pending_ensure(marks):
            return
        scheduler_revision = self._scheduler_presentation_revision()
        session_changed = False
        camera_changed = False
        applied_revision = self._scheduler_applied_revision()
        refinement_reset = False
        try:
            with self._serialized_viewport_iteration() as (
                presentation_changed,
                exclusive_pending,
            ):
                if exclusive_pending:
                    return
                session_revision = self._controller_session_revision()
                session_changed = session_revision > self._observed_session_revision
                if session_changed:
                    self._observed_session_revision = session_revision
                    self._failed = False
                    self._completed_samples = 0
                    self._current_result = None
                shared_presentations = getattr(
                    self._controller, "_has_shared_presentations", None
                )
                bind_presentation = presentation_changed or bool(
                    session_changed
                    and callable(shared_presentations)
                    and shared_presentations()
                )
                if bind_presentation:
                    self._refresh_camera_state()
                    replacement_reason = (
                        self._lifecycle.replacement_reason
                        if self._lifecycle is not None
                        else getattr(self._controller, "would_replace", None)
                    )
                    if (
                        callable(shared_presentations)
                        and shared_presentations()
                        and self._camera_value_probe.unhonored_findings()
                        and callable(replacement_reason)
                        and replacement_reason(self._request)
                        == "scene_composition_changed"
                    ):
                        # one composed session cannot carry two
                        # unhonored projections; fail closed until the runtime
                        # supports their live value route.
                        self._publish_failure(
                            RenderClientError(
                                "shared viewport projections require live camera "
                                "value updates from the OVRTX runtime"
                            ),
                            marks,
                        )
                        return
                camera_update = self._pending_camera_update(
                    presentation_changed=bind_presentation
                )
                new_camera_probe, changed_camera_values = (
                    self._pending_camera_value_edits()
                )
                takeover_camera_probe, takeover_camera_values = (
                    self._presentation_camera_values(
                        presentation_changed=bind_presentation,
                        allow_probe=new_camera_probe is None,
                    )
                )
                if new_camera_probe is None:
                    new_camera_probe = takeover_camera_probe
                    probe_camera_values = takeover_camera_values
                else:
                    probe_camera_values = changed_camera_values
                camera_values = tuple(
                    {
                        (value.prim_path, value.attribute): value
                        for value in takeover_camera_values + changed_camera_values
                    }.values()
                )
                tick_result = self._apply_runtime_tick(
                    marks,
                    camera_update=camera_update,
                    camera_values=camera_values,
                )
                for value in changed_camera_values:
                    self._applied_camera_values[value.attribute] = value.value
                self._camera_value_update_count += len(changed_camera_values)
                if camera_update is not None:
                    self._camera_update_needed = False
                    self._camera_update_count += 1
                    camera_changed = True
                self._last_tick_result = tick_result
                self._tick_should_request_redraw = bool(
                    tick_result.should_request_redraw
                )
                self._last_timeline_reset = bool(tick_result.timeline_reset)
                self._generation = int(tick_result.generation)
                if self._tick_result_sink is not None:
                    # Pose mirroring must be handed off even if acquisition fails.
                    self._tick_result_sink(tick_result, self._request)
                scheduler_revision = tick_result.presentation_revision
                applied_revision = tick_result.applied_revision
                applied_revision_changed = (
                    applied_revision > self._observed_applied_revision
                )
                if new_camera_probe is not None:
                    if probe_camera_values and not tick_result.values_written:
                        new_camera_probe["applied_revision"] = applied_revision
                        self._pending_camera_probe = new_camera_probe
                    else:
                        self._camera_value_probe.record_inconclusive(
                            str(new_camera_probe["probe_class"]),
                            camera_value_conversion.PROBE_INCONCLUSIVE_CONCURRENT_EDITS,
                        )
                elif (
                    self._pending_camera_probe is not None
                    and applied_revision_changed
                    and applied_revision
                    > int(self._pending_camera_probe["applied_revision"])
                ):
                    self._cancel_camera_probe(
                        camera_value_conversion.PROBE_INCONCLUSIVE_CONCURRENT_EDITS
                    )
                if self._pending_camera_probe is not None and (
                    tick_result.should_request_redraw
                    or bool(getattr(self._snapshot, "timeline_playing", False))
                ):
                    self._cancel_camera_probe(
                        camera_value_conversion.PROBE_INCONCLUSIVE_PHYSICS_ACTIVE
                    )
                refinement_reset = (
                    tick_result.should_reset_refinement
                    or applied_revision_changed
                    or bool(changed_camera_values)
                    or takeover_camera_probe is not None
                    or session_changed
                )
                if refinement_reset:
                    self._completed_samples = 0
                    self._current_result = None
                acquired = self._acquire_sample(marks)
        except (RenderClientError, SharedStageCompositionError) as exc:
            self._publish_failure(exc, marks)
            return
        result = acquired or self._current_result
        snapshot_changed = self._snapshot_changed_pending
        self._snapshot_changed_pending = False
        reset_reason = ""
        if snapshot_changed or refinement_reset:
            # Diagnostics vocabulary: composition_changed / camera_changed /
            # snapshot_changed, plus value_edit (task04-06) when an applied
            # value-update batch reset refinement with no other change.
            # Camera-only changes keep their existing reasons (precedence
            # lives in render_requests.reset_reason).
            reset_reason = render_requests.reset_reason(
                composition_changed=bool(tick_result.stage_changed),
                camera_changed=camera_changed,
                snapshot_changed=snapshot_changed,
                value_edit=bool(tick_result.values_written or refinement_reset),
            )
            self._last_reset_reason = reset_reason
        published = None
        if (
            acquired is not None
            or (self._resync_recovery_pending and result is not None)
            or (
                scheduler_revision > self._presented_scheduler_revision
                and result is not None
            )
        ):
            # The resync-recovery publication covers the reuse edge: a
            # scheduled replacement whose ensure reused the session (for
            # example a generation change materializing to an identical
            # composition digest) resets no refinement, so no rendered
            # step would follow the RESYNCING state and the resync status
            # would wedge. Re-publishing the session's current result
            # lifts the resync presentation with content that is still
            # valid for the reused session.
            validate_session = getattr(
                self._controller, "_validated_session", None
            )
            with (
                validate_session(session_revision)
                if callable(validate_session)
                else nullcontext(True)
            ) as session_current:
                if session_current:
                    published = self._frame_slot.publish(
                        FrameState(
                            status=FRAME_STATUS_FRAME,
                            render_result=result,
                            snapshot_key=snapshot.key,
                            completed_samples=self._completed_samples,
                            generation=self._generation,
                            presentation_revision=scheduler_revision,
                            applied_revision=applied_revision,
                            timing_marks=marks,
                        )
                    )
        if published is not None:
            self._publication_count += 1
            self._presented_scheduler_revision = max(
                self._presented_scheduler_revision,
                scheduler_revision,
            )
            self._observed_applied_revision = max(
                self._observed_applied_revision,
                applied_revision,
            )
            self._resync_recovery_pending = False
            self._record_iteration(
                "published",
                marks,
                publication_index=published.publication_index,
                completed_samples=self._completed_samples,
                tick_result=tick_result,
                reset_reason=reset_reason,
            )
        else:
            self._record_iteration(
                "no_publication",
                marks,
                completed_samples=self._completed_samples,
                tick_result=tick_result,
                reset_reason=reset_reason,
            )
        # Camera value probe bookkeeping (task04-05): a min-samples step is
        # the freshest pre-edit baseline for the next probe; an in-flight
        # probe concludes against this iteration's own post-edit step.
        self._capture_baseline_min_frame(acquired)
        pending_camera_probe = self._pending_camera_probe
        if (
            pending_camera_probe is not None
            and acquired is not None
            and int(acquired.completed_samples)
            == int(pending_camera_probe["pre_samples"])
        ):
            self._pending_camera_probe = None
            self._conclude_camera_probe(pending_camera_probe, acquired)
        # A completed iteration clears a failure latch even without fresh
        # input (the paced pending-edit retry path recovers here).
        self._failed = False

    def _run_pending_ensure(
        self, marks: dict[str, int], *, transport_owned: bool = False
    ) -> bool:
        """Run a pending session ensure/replace; ``False`` aborts the iteration.

        Startup ensure runs silently behind the engine's loading phase (no
        prior frame exists to keep presenting). A replacement first
        publishes :data:`FRAME_STATUS_RESYNCING` so the main thread keeps
        presenting the last published frame until the new session's first
        frame publishes (presentation gating). Ensure failures publish a
        failure state; retries follow ``retry_allowed`` (the
        ``session_lifecycle.should_auto_retry`` policy).
        """

        if self._lifecycle is None or not self._ensure_pending:
            return True
        if self._ensure_failed and not self._retry_allowed():
            # Auto-retry exhausted: hold the failed state without another
            # attempt (existing policy: a session restart resets counts).
            self._failed = True
            return False
        reason = self._pending_ensure_reason
        replacing = reason != SESSION_STARTUP_REASON
        try:
            transaction = getattr(self._controller, "_serialized_transport", None)
            with (
                transaction(
                    exclusive_pending=self._has_pending_jobs,
                    cancelled=self._stop.is_set,
                )
                if callable(transaction) and not transport_owned
                else nullcontext((False, False))
            ) as (_, exclusive_pending):
                if exclusive_pending:
                    return False
                self._request = self._with_shared_output_shape(self._request)
                if replacing:
                    self._publish_resyncing(reason, marks)
                marks["session_ensure_started_monotonic_ns"] = time.perf_counter_ns()
                scheduler = self._lifecycle.ensure_session(self._request)
            if scheduler is not None:
                if self._scheduler is not None and scheduler is not self._scheduler:
                    raise RenderClientError(
                        "session lifecycle attempted to replace the runtime scheduler"
                    )
                if self._scheduler is None:
                    self._scheduler = scheduler
                    hook_setter = (
                        getattr(scheduler, "set_edit_wake_hook", None)
                        if self._owns_scheduler
                        else None
                    )
                    if callable(hook_setter):
                        self._wake_hook_setter = hook_setter
                        self._wake_hook_setter(self._mailbox.wake)
            if self._scheduler is None:
                raise RenderClientError(
                    "session lifecycle did not provide a runtime scheduler"
                )
        except RuntimeServicesPreparingError:
            # Runtime services are (re)starting (worker restart): a transient
            # wait, not a failure. Hold the loading state -- no failure
            # publication -- and retry, paced by the backoff, until they serve.
            self._ensure_deferred = True
            return False
        except (RenderClientError, SharedStageCompositionError) as exc:
            self._ensure_deferred = False
            self._ensure_failed = True
            self._ensure_failure_count += 1
            self._publish_failure(exc, marks)
            return False
        marks["session_ensure_completed_monotonic_ns"] = time.perf_counter_ns()
        self._session_ensure_count += 1
        if replacing:
            self._session_replacement_count += 1
            # Guarantee a frame publication follows the RESYNCING state
            # even if the ensure reused the session (see _iterate).
            self._resync_recovery_pending = True
        self._ensure_pending = False
        self._ensure_failed = False
        self._ensure_deferred = False
        self._failed = False
        self._pending_ensure_reason = ""
        # The new session restarts acquisition and re-applies the
        # newest camera pose — a replaced session lost the live pose value
        # update that was applied to its predecessor.
        self._completed_samples = 0
        self._current_result = None
        self._snapshot_changed_pending = True
        self._refresh_camera_state()
        # The ensured session composed the request's current camera values
        # into its scene: they are the new applied baseline (task04-05),
        # and any in-flight probe/pre-edit frame belongs to the
        # predecessor's content.
        self._rebase_camera_values()
        return True

    def _publish_resyncing(self, reason: str, marks: dict[str, int]) -> None:
        """Present the background replacement via the shared frame slot."""

        self._last_resync_reason = reason
        self._resync_publication_count += 1
        published = self._frame_slot.publish(
            FrameState(
                status=FRAME_STATUS_RESYNCING,
                snapshot_key=self._snapshot_key,
                generation=self._generation,
                detail="Re-syncing scene",
                timing_marks=marks,
            )
        )
        self._record_iteration(
            "resyncing",
            marks,
            publication_index=published.publication_index,
            detail=reason,
        )

    def _pending_camera_update(
        self, *, presentation_changed: bool = False
    ) -> OvrtxTransformValue | None:
        """Build the newest presentation camera's typed live transform."""

        snapshot = self._snapshot
        if not (
            (self._camera_update_needed or presentation_changed)
            and snapshot.camera_matrix is not None
            and usd_paths.known_usd_path(snapshot.camera_prim_path)
        ):
            return None
        edit = InteractiveEdit(
            shape=EditShape.VALUE,
            data_authority=DataAuthority.VIEW,
            **edit_location(
                usd_prim_path=snapshot.camera_prim_path,
                usd_attribute="omni:xform",
                blender_property_path="viewport_camera_matrix",
                provenance={"source": "viewport_camera"},
            ),
            value=snapshot.camera_matrix,
            metadata={},
        )
        plan = InteractiveEditPlanner().plan(edit)
        if plan.mechanism != EditMechanism.UPDATE:
            raise RenderClientError(
                f"Viewport camera update is unsupported: {plan.reason}"
            )
        return OvrtxTransformValue(
            prim_path=snapshot.camera_prim_path,
            matrix=[list(row) for row in snapshot.camera_matrix],
        )

    def _presentation_camera_values(
        self, *, presentation_changed: bool, allow_probe: bool
    ) -> tuple[dict[str, Any] | None, tuple[OvrtxAttributeValue, ...]]:
        """Rebind proven projection values and probe one unknown class."""

        if not presentation_changed:
            return None, ()
        snapshot = self._snapshot
        if not usd_paths.known_usd_path(snapshot.camera_prim_path):
            return None, ()
        by_class: dict[str, list[Any]] = {}
        for attribute in camera_value_conversion.usd_attribute_values(
            getattr(self._request, "camera_projection", None)
        ):
            probe_class = str(attribute.metadata.get("probe_class", ""))
            if probe_class:
                by_class.setdefault(probe_class, []).append(attribute)
        unknown = [
            probe_class
            for probe_class in sorted(by_class)
            if self._camera_value_probe.status(probe_class)
            == camera_value_conversion.PROBE_STATUS_UNKNOWN
        ]
        attempted_class = unknown[0] if allow_probe and unknown else None
        probe_context = (
            self._begin_camera_probe(attempted_class, by_class[attempted_class])
            if attempted_class is not None
            else None
        )
        values: list[OvrtxAttributeValue] = []
        for probe_class in sorted(by_class):
            status = self._camera_value_probe.status(probe_class)
            if status != camera_value_conversion.PROBE_STATUS_HONORED and (
                probe_class != attempted_class
            ):
                continue
            for attribute in by_class[probe_class]:
                values.append(
                    OvrtxAttributeValue(
                        snapshot.camera_prim_path,
                        attribute.name,
                        attribute.value,
                        attribute.value_type,
                    )
                )
        return probe_context, tuple(values)

    def _apply_runtime_tick(
        self,
        marks: dict[str, int],
        *,
        camera_update: OvrtxTransformValue | None = None,
        camera_values: tuple[OvrtxAttributeValue, ...] = (),
    ) -> RuntimeTickResult:
        marks["runtime_update_started_monotonic_ns"] = time.perf_counter_ns()
        tick_request = render_requests.tick(self._request, now_ns=time.monotonic_ns())

        def _tick_and_bind_camera(ovrtx_updates, project_complete_pose):
            result = self._scheduler.tick_viewport(
                tick_request,
                ovrtx_updates=ovrtx_updates,
                project_complete_pose=project_complete_pose,
            )
            if camera_update is not None:
                try:
                    ovrtx_updates.update_transforms((camera_update,))
                except RenderClientError:
                    raise
                except Exception as exc:
                    raise RenderClientError(
                        f"Viewport camera update failed: {exc}"
                    ) from exc
            if camera_values:
                try:
                    ovrtx_updates.update_attribute_values(camera_values)
                except RenderClientError:
                    raise
                except Exception as exc:
                    raise RenderClientError(
                        f"Viewport camera projection update failed: {exc}"
                    ) from exc
            return result

        try:
            result = self._active_controller().apply_runtime_updates(
                _tick_and_bind_camera
            )
        except SharedStageCompositionError as exc:
            raise RenderClientError(f"Shared-stage composition failed: {exc}") from exc
        marks["runtime_update_completed_monotonic_ns"] = time.perf_counter_ns()
        # A rejected live RTPT render-setting write is de-fataled by the
        # scheduler; fall back to session re-keying here (task01-04). Handled
        # before the failure raise so the route flips even if another lane in
        # the same tick genuinely failed.
        if result.render_setting_rejected:
            self._handle_render_setting_rejection(result.render_setting_rejected)
        if result.status == RuntimeTickStatus.FAILED:
            raise RenderClientError(
                "Shared-stage composition failed: "
                f"{result.skipped_reason or result.status.value}"
            )
        return result

    def _acquire_sample(self, marks: dict[str, int]) -> RenderResult | None:
        if not render_requests.viewport_sampling_due(
            self._completed_samples, self._snapshot.max_samples
        ):
            return None
        marks["render_call_started_monotonic_ns"] = time.perf_counter_ns()
        result = self._active_controller().render(
            self._request, additional_samples=1
        )
        marks["render_call_completed_monotonic_ns"] = time.perf_counter_ns()
        self._completed_samples += 1
        self._current_result = _dataclass_replace(
            result,
            completed_samples=self._completed_samples,
        )
        return self._current_result

    def _publish_failure(self, exc: BaseException, marks: dict[str, int]) -> None:
        self._failed = True
        self._failure_count += 1
        detail = f"{type(exc).__name__}: {exc}"
        self._last_failure_detail = detail
        published = self._frame_slot.publish(
            FrameState(
                status=FRAME_STATUS_FAILED,
                snapshot_key=self._snapshot_key,
                generation=self._generation,
                detail=detail,
                timing_marks=marks,
            )
        )
        self._record_iteration(
            "failed",
            marks,
            publication_index=published.publication_index,
            detail=detail,
        )


__all__ = [
    "CAMERA_VALUES_UNHONORED_REASON",
    "RENDER_SETTING_UNHONORED_REASON",
    "FAILURE_RETRY_BACKOFF_SECONDS",
    "ITERATION_RECORD_LIMIT",
    "ITERATION_TIMING_SPANS",
    "RENDER_THREAD_JOIN_TIMEOUT_SECONDS",
    "RESIZE_REPLACEMENT_REASON",
    "SESSION_STARTUP_REASON",
    "STATUS_CREATED",
    "STATUS_FAILED",
    "STATUS_RUNNING",
    "STATUS_STOPPED",
    "LatestViewRenderLoop",
    "RedrawSignalingFrameSlot",
    "RenderThreadError",
    "RenderThreadRejectedError",
    "RenderThreadResult",
    "RenderThreadTimeoutError",
    "SessionLifecycleHooks",
    "ViewportRenderThread",
    "render_result_digest",
]

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Latest-wins handoff structures for the viewport render loop.

Spec design decision (blender-live-render, "Latest-wins handoff, no queues
of stale work"): camera poses and view requests use a single-slot mailbox
where the newest snapshot overwrites — nothing queues — and completed
frames publish to a single latest-frame slot the main thread only ever
reads the newest value from.

Two structures, one ``threading.Condition`` each:

- :class:`CameraRequestMailbox` — the Blender main thread ``write``s the
  newest :class:`ViewSnapshot`; the render thread ``take``s it atomically
  (returns-and-clears, or times out). ``take(timeout)`` doubles as the
  render loop's park/wake mechanism; intermediate snapshots are droppable
  by design. Non-snapshot wake sources (task02-03: value-edit submission,
  stop requests) interrupt a parked ``take`` via :meth:`~CameraRequestMailbox.wake`,
  which makes ``take`` return ``None`` early so the loop re-evaluates its
  pending work.
- :class:`LatestFrameSlot` — the render thread ``publish``es a
  :class:`FrameState` (``RenderResult`` + snapshot key + completed samples
  + generation); publication stamps a strictly monotonic publication index
  so stale reads are impossible. ``peek_latest()`` is non-blocking and
  never clears. Resync/failure states (task02-06) flow through the same
  slot via the ``FRAME_STATUS_*`` variants so presentation has one source.

Locking discipline: each condition's lock is held only for slot swaps and
reads — never across RPCs or texture uploads. No payload copies are made:
``RenderResult.rgba8`` bytes are immutable and pass by reference.

Timing marks (``written_monotonic_ns``, ``published_monotonic_ns``,
``timing_marks``) use ``time.perf_counter_ns()``, matching the engine's
span-boundary vocabulary (``redraw_requested_monotonic_ns``,
``rgba_available_monotonic_ns``, ...) so task02-09 can map the existing
profile names onto the new pipeline.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Mapping

from . import color_presentation
from . import render_requests

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .ovrtx_runtime_client import RenderResult


#: Poll slice for the render loop's *finite* park timeout (the scheduler
#: tick interval). ``threading.Condition.wait`` quantizes to the OS timer
#: tick — on Windows the default ~15.6 ms tick rounds a 33.3 ms (1/30 s)
#: tick park up to ~46 ms (~21.6 Hz measured), so a tick-gated viewport
#: paces ~28% slow and unevenly relative to refinement-gated frames. Python
#: 3.11+ ``time.sleep`` uses a high-resolution waitable timer on Windows and
#: does not quantize, so the finite-timeout park waits in short ``time.sleep``
#: slices, re-checking the slot each slice. The slice bounds wake latency
#: (a mid-park write/wake is observed within one slice) while keeping the
#: park accurate; indefinite (``None``) and non-blocking (``0``) parks keep
#: the plain condition wait — neither quantizes and neither needs polling.
_FINITE_PARK_SLICE_SECONDS = 0.0015

FRAME_STATUS_FRAME = "frame"
FRAME_STATUS_RESYNCING = "resyncing"
FRAME_STATUS_FAILED = "failed"
FRAME_STATUSES = (
    FRAME_STATUS_FRAME,
    FRAME_STATUS_RESYNCING,
    FRAME_STATUS_FAILED,
)

#: How a completed frame's pixels reach the display (task02-05, contract
#: step 4 "exactly once"). ``FRAME_DISPLAY_PASSTHROUGH`` frames are already
#: display-encoded (OVRTX applied the View Transform, Look, Exposure, Gamma
#: into ``LdrColor`` RGBA8) and are drawn raw — Blender must not transform
#: them again. ``FRAME_DISPLAY_BLENDER_TRANSFORM`` frames are scene-linear
#: ``HdrColor`` RGBA16F; the linear pixels are drawn through Blender's
#: display-space shader so Blender's display pipeline performs the transform
#: exactly once.
FRAME_DISPLAY_PASSTHROUGH = "passthrough"
FRAME_DISPLAY_BLENDER_TRANSFORM = "blender_display_transform"
FRAME_DISPLAY_MODES = (
    FRAME_DISPLAY_PASSTHROUGH,
    FRAME_DISPLAY_BLENDER_TRANSFORM,
)


def frame_display_transform(render_result: Any) -> str:
    """Classify who owns the display transform for a completed frame.

    Scene-linear frames (``HdrColor`` RGBA16F with a scene-linear color mode
    and a non-empty linear payload) return
    :data:`FRAME_DISPLAY_BLENDER_TRANSFORM`: their linear pixels must be drawn
    through Blender's display-space shader so the View Transform / Look /
    Exposure / Gamma are applied exactly once by Blender. Every other frame
    (LDR display-encoded ``RGBA8``, or a scene-linear frame missing its linear
    payload) returns :data:`FRAME_DISPLAY_PASSTHROUGH` and is drawn raw with no
    Blender transform on top — OVRTX already display-encoded it.
    """

    frame_format = str(
        getattr(render_result, "frame_format", color_presentation.FRAME_FORMAT_RGBA8)
        or color_presentation.FRAME_FORMAT_RGBA8
    )
    frame_color_mode = str(
        getattr(
            render_result,
            "frame_color_mode",
            color_presentation.FRAME_COLOR_MODE_DISPLAY_LDR,
        )
        or color_presentation.FRAME_COLOR_MODE_DISPLAY_LDR
    )
    linear_payload = getattr(render_result, "linear_rgba16f", b"") or b""
    if (
        frame_format == color_presentation.FRAME_FORMAT_RGBA16F
        and frame_color_mode == color_presentation.FRAME_COLOR_MODE_SCENE_LINEAR
        and linear_payload
    ):
        return FRAME_DISPLAY_BLENDER_TRANSFORM
    return FRAME_DISPLAY_PASSTHROUGH


def frame_applies_blender_display_transform(render_result: Any) -> bool:
    """True when Blender must apply its display transform to ``render_result``."""

    return (
        frame_display_transform(render_result)
        == FRAME_DISPLAY_BLENDER_TRANSFORM
    )


@dataclass(frozen=True)
class ViewSnapshot:
    """Newest camera/request state handed from the main thread to the loop.

    Pure data (no Blender objects — same rule as ``RuntimeTickRequest``).
    The camera matrix is normalized through
    ``render_requests.stable_camera_matrix`` on construction so numerically
    equivalent poses share rendered-view identity.
    """

    camera_matrix: tuple[tuple[float, ...], ...] | None = None
    camera_prim_path: str = ""
    #: Viewport camera projection state (``CameraProjectionState`` — pure
    #: data). Without it, lens/FOV changes (numpad-0 into the scene-camera
    #: view, viewport lens edits) never crossed the mailbox and the render
    #: thread kept the session-start projection (Junk Shop regression,
    #: 2026-07-07).
    camera_projection: Any | None = None
    min_samples: int = 1
    max_samples: int = 128
    selected_sensor_paths: tuple[str, ...] = ()
    render_var: str = color_presentation.RENDER_VAR_LDR_COLOR
    #: Region-derived viewport resolution.
    width: int = 64
    height: int = 64
    timeline_controls_enabled: bool = False
    timeline_playing: bool = False
    timeline_frame: int = 1
    timeline_start: int = 1
    timeline_end: int = 1
    simulation_reset_token: int = 0
    #: ``time.perf_counter_ns()`` at snapshot creation (stamped when 0).
    written_monotonic_ns: int = 0
    #: Extra span-boundary marks (for example
    #: ``redraw_requested_monotonic_ns``) forwarded to profiling.
    timing_marks: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "camera_matrix",
            render_requests.stable_camera_matrix(self.camera_matrix),
        )
        object.__setattr__(
            self, "selected_sensor_paths", tuple(self.selected_sensor_paths)
        )
        object.__setattr__(self, "render_var", str(self.render_var))
        # Own the marks: a frozen snapshot crossing threads must not alias a
        # caller-mutable mapping (the engine's live ``timings`` dict is the
        # natural payload here).
        object.__setattr__(self, "timing_marks", dict(self.timing_marks))
        if not self.written_monotonic_ns:
            object.__setattr__(
                self, "written_monotonic_ns", time.perf_counter_ns()
            )

    @property
    def key(self) -> tuple[Any, ...]:
        """Viewport identity whose changes reset refinement downstream."""

        return (
            self.selected_sensor_paths,
            self.render_var,
            self.min_samples,
            self.camera_matrix,
            render_requests.camera_projection_key(self.camera_projection),
        )


def snapshot_from_render_request(
    request: Any,
    *,
    timing_marks: Mapping[str, int] | None = None,
) -> ViewSnapshot:
    """Build a :class:`ViewSnapshot` from a translated ``RenderRequest``.

    ``RenderRequest`` is already add-on-owned pure data (the Blender-signal
    translation happens on the main thread before this), so the snapshot
    inherits the no-Blender-objects rule for free. ``request.width`` and
    ``request.height`` are the region-derived viewport resolution.
    """

    return ViewSnapshot(
        camera_matrix=request.camera_matrix,
        camera_prim_path=str(request.camera_prim_path),
        camera_projection=request.camera_projection,
        min_samples=int(request.min_samples),
        max_samples=int(request.max_samples),
        selected_sensor_paths=tuple(request.selected_sensor_paths),
        render_var=str(
            request.color_presentation.get(
                "render_var", color_presentation.RENDER_VAR_LDR_COLOR
            )
        ),
        width=int(request.width),
        height=int(request.height),
        timeline_controls_enabled=bool(request.timeline_controls_enabled),
        timeline_playing=bool(request.timeline_playing),
        timeline_frame=int(request.timeline_frame),
        timeline_start=int(request.timeline_start),
        timeline_end=int(request.timeline_end),
        simulation_reset_token=int(request.simulation_reset_token),
        timing_marks=timing_marks or {},
    )


def request_from_snapshot(base_request: Any, snapshot: ViewSnapshot) -> Any:
    """Overlay a snapshot's view state onto a translated base ``RenderRequest``.

    Inverse companion of :func:`snapshot_from_render_request` for the render
    loop (task02-03): the main thread keeps translating full requests (they
    carry session identity — input path, worker command, sensors), while the
    per-view fields the mailbox transports (camera, sample range, resolution,
    render var, timeline cursor) come from the newest snapshot. Returns a new
    frozen request; ``base_request`` is not mutated.
    """

    color_presentation_map = dict(base_request.color_presentation)
    color_presentation_map["render_var"] = snapshot.render_var
    return replace(
        base_request,
        width=int(snapshot.width),
        height=int(snapshot.height),
        min_samples=int(snapshot.min_samples),
        max_samples=int(snapshot.max_samples),
        camera_prim_path=str(snapshot.camera_prim_path),
        camera_matrix=snapshot.camera_matrix,
        # A snapshot without projection state (no viewport context on that
        # draw) keeps the base request's projection rather than clearing a
        # composed one.
        camera_projection=(
            snapshot.camera_projection
            if snapshot.camera_projection is not None
            else base_request.camera_projection
        ),
        selected_sensor_paths=tuple(snapshot.selected_sensor_paths),
        timeline_controls_enabled=bool(snapshot.timeline_controls_enabled),
        timeline_playing=bool(snapshot.timeline_playing),
        timeline_frame=int(snapshot.timeline_frame),
        timeline_start=int(snapshot.timeline_start),
        timeline_end=int(snapshot.timeline_end),
        simulation_reset_token=int(snapshot.simulation_reset_token),
        color_presentation=color_presentation_map,
    )


@dataclass(frozen=True)
class FrameState:
    """One publication of the latest-frame slot.

    ``status`` is :data:`FRAME_STATUS_FRAME` for completed renders (which
    require ``render_result`` and ``snapshot_key``) or a non-frame variant
    (:data:`FRAME_STATUS_RESYNCING` / :data:`FRAME_STATUS_FAILED`, wired by
    task02-06) so presentation reads one source. ``publication_index`` and
    ``published_monotonic_ns`` are always stamped by
    :meth:`LatestFrameSlot.publish`; caller-supplied values are ignored.
    """

    status: str = FRAME_STATUS_FRAME
    render_result: "RenderResult | None" = None
    snapshot_key: tuple[Any, ...] | None = None
    completed_samples: int = 0
    generation: int = 0
    presentation_revision: int = 0
    applied_revision: int = 0
    #: Human-readable context for non-frame statuses (failure text, resync
    #: reason). Empty for ordinary frame publications.
    detail: str = ""
    publication_index: int = 0
    #: ``time.perf_counter_ns()`` at publication (stamped by the slot).
    published_monotonic_ns: int = 0
    #: Extra span-boundary marks (for example
    #: ``rgba_available_monotonic_ns``) forwarded to profiling.
    timing_marks: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in FRAME_STATUSES:
            raise ValueError(
                f"unknown frame status {self.status!r}; expected one of "
                f"{FRAME_STATUSES}"
            )
        if self.status == FRAME_STATUS_FRAME:
            if self.render_result is None:
                raise ValueError(
                    "frame publications require a render_result"
                )
            if self.snapshot_key is None:
                raise ValueError(
                    "frame publications require a snapshot_key"
                )
        # Own the marks (same aliasing rule as ``ViewSnapshot``); the
        # ``RenderResult`` payload itself still passes by reference.
        object.__setattr__(self, "timing_marks", dict(self.timing_marks))


class CameraRequestMailbox:
    """Single-slot, latest-wins mailbox for the newest :class:`ViewSnapshot`.

    The main thread overwrites via :meth:`write`; the render thread empties
    the slot atomically via :meth:`take`. There is deliberately no queue:
    a snapshot that is overwritten before it is taken was stale by
    definition. One condition object; its lock is held only for the slot
    swap.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._snapshot: ViewSnapshot | None = None
        self._woken = False
        self._writes = 0
        self._overwrites = 0
        self._superseded = 0
        self._takes = 0
        self._wakes = 0
        self._last_write_monotonic_ns = 0
        self._last_take_monotonic_ns = 0

    def write(self, snapshot: ViewSnapshot) -> None:
        """Overwrite the slot with the newest snapshot and wake the taker."""

        if not isinstance(snapshot, ViewSnapshot):
            raise TypeError("mailbox writes must be ViewSnapshot instances")
        with self._condition:
            if self._snapshot is not None:
                self._overwrites += 1
                if self._snapshot.key != snapshot.key:
                    # A distinct pending view was replaced before the render
                    # thread ever adopted it — direct latest-wins evidence
                    # (ADR 0013; task02-09 surfaces this in the artifact).
                    # Same-key rewrites are routine draw-path refreshes, not
                    # supersessions.
                    self._superseded += 1
            self._snapshot = snapshot
            self._writes += 1
            self._last_write_monotonic_ns = time.perf_counter_ns()
            self._condition.notify_all()

    def take(self, timeout: float | None = None) -> ViewSnapshot | None:
        """Return-and-clear the newest snapshot, or ``None`` on timeout/wake.

        This is the render loop's park/wake mechanism: with an empty slot
        the caller sleeps on the condition until a write arrives, a
        :meth:`wake` interrupt fires, or ``timeout`` (seconds) elapses.
        ``timeout=0`` polls without blocking; ``timeout=None`` waits
        indefinitely (a parked loop). A :meth:`wake` makes ``take`` return
        ``None`` without consuming any snapshot so the caller re-evaluates
        its pending work (value edits, stop requests).
        """

        # Indefinite park and non-blocking poll keep the plain condition wait:
        # an indefinite wait wakes promptly on ``notify_all`` with no polling,
        # and ``timeout == 0`` returns immediately — neither quantizes. Only a
        # *finite positive* timeout (the render loop's scheduler tick interval)
        # is paced with a high-resolution wakeable sleep-slice loop, because
        # ``Condition.wait`` rounds it up to the OS timer tick on Windows
        # (see ``_FINITE_PARK_SLICE_SECONDS``).
        if timeout is None or timeout <= 0.0:
            with self._condition:
                if self._snapshot is None and not self._woken:
                    self._condition.wait_for(
                        lambda: self._snapshot is not None or self._woken, timeout
                    )
                return self._consume_locked()
        deadline = time.perf_counter() + timeout
        while True:
            with self._condition:
                if self._snapshot is not None or self._woken:
                    return self._consume_locked()
            remaining = deadline - time.perf_counter()
            if remaining <= 0.0:
                return None
            time.sleep(min(_FINITE_PARK_SLICE_SECONDS, remaining))

    def _consume_locked(self) -> ViewSnapshot | None:
        """Return-and-clear the slot; caller holds ``self._condition``."""

        self._woken = False
        snapshot = self._snapshot
        self._snapshot = None
        if snapshot is not None:
            self._takes += 1
            self._last_take_monotonic_ns = time.perf_counter_ns()
        return snapshot

    def wake(self) -> None:
        """Interrupt a parked :meth:`take` without writing a snapshot.

        Non-snapshot wake sources (value-edit submission via the
        ``ViewUpdateStream`` wake hook, render-loop stop requests) use this
        to break the loop out of ``take(timeout=None)``. The wake is
        one-shot latching: if no taker is currently waiting, the next
        ``take`` returns immediately (no lost wakeups).
        """

        with self._condition:
            self._woken = True
            self._wakes += 1
            self._condition.notify_all()

    def peek(self) -> ViewSnapshot | None:
        """Non-blocking, non-clearing read (diagnostics/tests only)."""

        with self._condition:
            return self._snapshot

    def diagnostics(self) -> dict[str, Any]:
        with self._condition:
            return {
                "occupied": self._snapshot is not None,
                "writes": self._writes,
                "overwrites": self._overwrites,
                "superseded_snapshots": self._superseded,
                "takes": self._takes,
                "wakes": self._wakes,
                "last_write_monotonic_ns": self._last_write_monotonic_ns,
                "last_take_monotonic_ns": self._last_take_monotonic_ns,
            }


class LatestFrameSlot:
    """Single latest-frame slot with a strictly monotonic publication index.

    The render thread :meth:`publish`es completed frames (and resync or
    failure states); the main thread :meth:`peek_latest`s without blocking
    and never clears. The slot is the only writer of
    ``publication_index``, and it only ever replaces the slot content under
    the same lock that bumps the index, so a consumer that remembers the
    last index it presented can never observe an older publication (stale
    reads are impossible). One condition object; its lock is held only for
    the stamp-and-swap.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._latest: FrameState | None = None
        self._publication_index = 0

    def publish(self, frame_state: FrameState) -> FrameState:
        """Stamp, store, and notify; returns the stamped :class:`FrameState`.

        ``publication_index`` (monotonic, starting at 1) and
        ``published_monotonic_ns`` are assigned here regardless of the
        caller-supplied values. The payload is stored by reference — no
        copies of ``RenderResult.rgba8``.
        """

        if not isinstance(frame_state, FrameState):
            raise TypeError("publications must be FrameState instances")
        with self._condition:
            self._publication_index += 1
            stamped = replace(
                frame_state,
                publication_index=self._publication_index,
                published_monotonic_ns=time.perf_counter_ns(),
            )
            self._latest = stamped
            self._condition.notify_all()
        return stamped

    def peek_latest(self) -> FrameState | None:
        """Non-blocking read of the newest publication; never clears."""

        with self._condition:
            return self._latest

    def latest_index(self) -> int:
        with self._condition:
            return self._publication_index

    def wait_for_newer(
        self, publication_index: int, timeout: float | None = None
    ) -> FrameState | None:
        """Wait for a publication newer than ``publication_index``.

        Bounded companion to :meth:`peek_latest` for callers that may
        block (tests, first-frame waits). Returns the newest publication
        once its index exceeds ``publication_index``, or ``None`` on
        timeout. Never used on the Blender main thread's draw path.
        """

        with self._condition:
            self._condition.wait_for(
                lambda: self._publication_index > publication_index, timeout
            )
            if self._publication_index > publication_index:
                return self._latest
            return None

    def diagnostics(self) -> dict[str, Any]:
        with self._condition:
            latest = self._latest
            return {
                "publication_index": self._publication_index,
                "occupied": latest is not None,
                "status": latest.status if latest is not None else "",
                "completed_samples": (
                    latest.completed_samples if latest is not None else 0
                ),
                "generation": latest.generation if latest is not None else 0,
                "published_monotonic_ns": (
                    latest.published_monotonic_ns if latest is not None else 0
                ),
            }


__all__ = [
    "FRAME_DISPLAY_BLENDER_TRANSFORM",
    "FRAME_DISPLAY_MODES",
    "FRAME_DISPLAY_PASSTHROUGH",
    "FRAME_STATUSES",
    "FRAME_STATUS_FAILED",
    "FRAME_STATUS_FRAME",
    "FRAME_STATUS_RESYNCING",
    "CameraRequestMailbox",
    "FrameState",
    "LatestFrameSlot",
    "ViewSnapshot",
    "frame_applies_blender_display_transform",
    "frame_display_transform",
    "request_from_snapshot",
    "snapshot_from_render_request",
]

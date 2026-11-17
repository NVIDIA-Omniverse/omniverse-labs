# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native profile/memory diagnostics and OVUI-independent timeline interaction.

``native.profile_snapshot(session, scene_id=0, max_events=200000)`` and
``native.memory_snapshot(...)`` read only the current native event/frame under
one Session mutex/read guard. Owned records remain valid after seek/destruction.
Use read_profile/read_memory below in a worker for a private file Session; the
UI's displayed session must never be sought by an analysis sweep.

Profile identity is (zone_id, event_id, context_id, thread_id). Ticks remain exact
integers; start_ms/end_ms are offsets from snapshot.start_tick and duration_ms
uses the recorded rational conversion. The range includes observed unmatched
start/stop timestamps as in Qt; unmatched pairs never contribute duration.
Counters have NO timestamp. Unmatched
boundaries are reported, not assigned guessed duration. OVD returns unsupported.

Memory frame rows preserve original same-frame allocation/free address matching.
Outstanding entries are separately read from the core's recorded signed32 totals,
with None when absent; full source paths keep equal basenames distinguishable.
Neither this module nor its models parses capture bytes.
"""
from bisect import bisect_right
from concurrent.futures import CancelledError, ThreadPoolExecutor
from dataclasses import dataclass
import math
from pathlib import Path
from .ingestion import retain_capture, submit_capture
import tempfile

import _pvd_native as native
from .analysis import AnalysisJob, UnsupportedAnalysis, _check_cancel, _frame_range, _write_csv


@dataclass(frozen=True, order=True)
class EventKey:
    zone_id: int
    event_id: int
    context_id: int
    thread_id: int

    @classmethod
    def of(cls, record):
        return cls(record.zone_id, record.event_id, record.context_id, record.thread_id)


@dataclass(frozen=True, order=True)
class LaneKey:
    zone_id: int
    context_id: int
    thread_id: int
    depth: int


def thread_name(thread_id):
    return "Cross Thread" if thread_id == native.CROSS_THREAD_ID else f"Thread {thread_id}"


@dataclass(frozen=True)
class DiagnosticsRequest:
    path: Path
    scene_id: int
    frame: int
    max_events: int = 200_000

    @classmethod
    def from_display(cls, path, session, scene_id=None):
        if session is None:
            raise UnsupportedAnalysis("Open a capture before inspecting diagnostics")
        info = session.info()
        if info.ovd:
            raise UnsupportedAnalysis("OVD does not contain legacy profile or memory event streams")
        if info.live and not info.ended:
            raise UnsupportedAnalysis("Detach and save the producer before inspecting its diagnostics")
        if path is None or not Path(path).is_file():
            raise UnsupportedAnalysis("Diagnostics require a saved capture")
        scenes = session.scenes()
        scene = next((s for s in scenes if scene_id is None or s.id == scene_id), None)
        if scene is None:
            raise UnsupportedAnalysis("This capture has no selected scene timeline")
        return cls(Path(path).expanduser().resolve(), scene.id, scene.current_frame)


def _read(request, function, cancel=None, progress=None):
    _check_cancel(cancel)
    if progress:
        progress(0, 1)
    with retain_capture(request.path), tempfile.TemporaryDirectory(prefix="pvd-diagnostics-") as temporary:
        session = native.Session(str(Path(request.path).expanduser().resolve()), temporary)
        try:
            if session.info().ovd:
                raise UnsupportedAnalysis("OVD does not contain legacy profile or memory event streams")
            _frame_range(session, request.scene_id, request.frame, request.frame + 1, 1)
            _check_cancel(cancel)
            if not session.seek(request.scene_id, request.frame):
                raise UnsupportedAnalysis("The requested frame is not published")
            result = function(session, request.scene_id, request.max_events)
            _check_cancel(cancel)
            if not result.supported:
                raise UnsupportedAnalysis(result.reason)
            if progress:
                progress(1, 1)
            return result
        finally:
            del session


def read_profile(request, *, cancel=None, progress=None):
    return _read(request, native.profile_snapshot, cancel, progress)


def read_memory(request, *, cancel=None, progress=None):
    return _read(request, native.memory_snapshot, cancel, progress)


@dataclass(frozen=True)
class ProfileHistoryRequest:
    path: Path
    scene_id: int
    zone_id: int
    event_id: int
    context_id: int
    thread_id: int | None  # None sums this event/context across all threads.
    start: int = 1
    stop: int | None = None
    max_events: int = 200_000


@dataclass(frozen=True)
class ProfileSeries:
    request: ProfileHistoryRequest
    name: str
    frames: tuple[int, ...]
    values: tuple[float | None, ...]
    unit: str = "ms"

    @property
    def bounds(self):
        values = [v for v in self.values if v is not None]
        return (min(values), max(values)) if values else None

    def nearest_frame(self, position):
        if not self.frames or not math.isfinite(position):
            return None
        return min(self.frames[-1], max(self.frames[0], math.floor(position + .5)))


def read_profile_history(request, *, cancel=None, progress=None, max_samples=100_000):
    """Duration sums over raw clip frames; unknown/unmatched measurements are gaps.

A declared event with no occurrences in a complete frame contributes measured
zero, matching the original duration graph. An unmatched boundary for the selected event key makes its sample a gap;
unrelated cross-frame events do not erase valid measurements.
A zone/event not declared by the file is an error rather than an all-zero graph.
"""
    _check_cancel(cancel)
    with retain_capture(request.path), tempfile.TemporaryDirectory(prefix="pvd-profile-history-") as temporary:
        session = native.Session(str(Path(request.path).expanduser().resolve()), temporary)
        try:
            if session.info().ovd:
                raise UnsupportedAnalysis("OVD does not contain legacy profile event streams")
            start, stop = _frame_range(session, request.scene_id, request.start, request.stop, max_samples)
            frames, values, name, found_identity = [], [], None, False
            if progress:
                progress(0, stop - start)
            for frame in range(start, stop):
                _check_cancel(cancel)
                if not session.seek(request.scene_id, frame):
                    raise UnsupportedAnalysis("Requested profile frame is not published")
                data = native.profile_snapshot(session, request.scene_id, request.max_events, include_context_names=False)
                if not data.supported:
                    raise UnsupportedAnalysis(data.reason)
                zone = next((z for z in data.zones if z.id == request.zone_id), None)
                event = next((e for e in zone.events if e.id == request.event_id), None) if zone else None
                if event is None:
                    raise ValueError("Unknown native profile zone/event identity")
                name = event.name.removeprefix(".")
                def matches(record):
                    return (record.zone_id == request.zone_id and record.event_id == request.event_id
                            and record.context_id == request.context_id
                            and (request.thread_id is None or record.thread_id == request.thread_id))
                matched = [s for s in data.summaries if matches(s)]
                incomplete = any(matches(record) for record in data.unmatched)
                found_identity |= bool(matched) or incomplete
                # Only incomplete scopes for this key create a gap. Counters have
                # no duration and cannot establish a graphable timed identity.
                values.append(None if incomplete else sum(row.total_ms for row in matched))
                frames.append(frame)
                if progress:
                    progress(frame - start + 1, stop - start)
            _check_cancel(cancel)
            if not found_identity:
                raise ValueError("No timed scopes match this context/thread/event in the requested range")
            return ProfileSeries(request, name or "Profile duration", tuple(frames), tuple(values))
        finally:
            del session


def export_profile_csv(series, path, *, overwrite=False):
    suffix = "all threads" if series.request.thread_id is None else thread_name(series.request.thread_id)
    return _write_csv(path, ("frame", f"{series.name} ({suffix}, ms)"),
                      zip(series.frames, ("" if v is None else repr(v) for v in series.values)), overwrite=overwrite)


class ProfileModel:
    """Selection, event visibility, search and scoped timeline pan/zoom/markers.

Call begin_capture() at every file/session boundary; load() at each analysis
frame. Event selection/hidden keys persist across frames but refer only to native
zone/context/thread/event IDs. All times for interaction are relative milliseconds.
The model emits no GUI callbacks, avoiding recursive selection/history updates.
"""
    def __init__(self):
        self.begin_capture()

    def begin_capture(self):
        self.snapshot = None
        self.hidden = set()
        self.selected = frozenset()
        self.view_range = None
        self.marked_range = None
        self.marker_ms = None
        self._rows = {}
        self._lanes = {}
        self._starts = {}

    def load(self, snapshot):
        self.snapshot = snapshot
        self._rows = {EventKey.of(row): row for row in snapshot.summaries} if snapshot.supported else {}
        lanes = {}
        for scope in snapshot.scopes if snapshot.supported else ():
            key = LaneKey(scope.zone_id, scope.context_id, scope.thread_id, scope.depth)
            lanes.setdefault(key, []).append(scope)
        self._lanes = {key: tuple(sorted(values, key=lambda s: s.start_ms)) for key, values in lanes.items()}
        self._starts = {key: tuple(s.start_ms for s in values) for key, values in self._lanes.items()}
        self.reset_zoom()
        self.marker_ms = self.marked_range = None

    @property
    def supported(self):
        return self.snapshot is not None and self.snapshot.supported

    @property
    def reason(self):
        return self.snapshot.reason if self.snapshot else "Open a capture with profile events"

    @property
    def lanes(self):
        return tuple(sorted(self._lanes))

    def rows(self, *, include_hidden=False):
        return tuple(row for key, row in sorted(self._rows.items()) if include_hidden or key not in self.hidden)

    def scopes(self, lane=None, *, include_hidden=False):
        source = self._lanes.get(lane, ()) if lane else (s for values in self._lanes.values() for s in values)
        return tuple(s for s in source if include_hidden or EventKey.of(s) not in self.hidden)

    def search(self, expression, by="name"):
        """Original case-sensitive name substring and unsigned numeric ID criteria."""
        if by not in ("name", "context_id", "thread_id", "event_id", "zone_id"):
            raise ValueError("Unknown profile search criterion")
        if by == "name":
            return tuple(row for row in self.rows(include_hidden=True) if expression in row.name)
        if not isinstance(expression, str) or not expression.isascii() or not expression.isdecimal():
            return ()
        value = int(expression)
        maximum = {"context_id": 2**64 - 1, "thread_id": 2**32 - 1, "event_id": 2**16 - 1, "zone_id": 2**31 - 1}[by]
        if value > maximum:
            return ()
        return tuple(row for row in self.rows(include_hidden=True) if getattr(row, by) == value)

    def select(self, keys, *, toggle=False):
        keys = frozenset(key for key in keys if key in self._rows)
        self.selected = self.selected.symmetric_difference(keys) if toggle else keys

    def hide(self, keys, hidden=True):
        keys = set(keys)
        if hidden:
            self.hidden.update(keys)
        else:
            self.hidden.difference_update(keys)

    def set_selected_visibility(self, visible, *, inverse=False):
        keys = set(self._rows) - self.selected if inverse else self.selected
        self.hide(keys, not visible)

    def reset_zoom(self):
        duration = self.snapshot.duration_ms if self.supported else 0
        self.view_range = (0.0, duration) if duration > 0 else None

    def _time(self, value):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("Timeline time must be finite")
        if not self.supported or self.snapshot.duration_ms <= 0:
            return None
        return min(self.snapshot.duration_ms, max(0.0, value))

    def mark(self, start, stop):
        start, stop = self._time(start), self._time(stop)
        self.marked_range = tuple(sorted((start, stop))) if start is not None else None
        return self.marked_range

    def set_marker(self, time_ms):
        self.marker_ms = self._time(time_ms)

    def cancel_mark(self):
        self.marked_range = None

    def zoom_to_mark(self):
        if self.marked_range and self.marked_range[1] > self.marked_range[0]:
            self.view_range = self.marked_range
            self.marked_range = None
            return True
        return False

    def zoom(self, factor, anchor=None):
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError("Zoom factor must be positive and finite")
        if self.view_range is None:
            return
        low, high = self.view_range
        anchor = (low + high) / 2 if anchor is None else self._time(anchor)
        fraction = min(1, max(0, (anchor - low) / (high - low)))
        span = min(self.snapshot.duration_ms, max(self.snapshot.duration_ms / 1_000_000, (high - low) * factor))
        low = min(self.snapshot.duration_ms - span, max(0, anchor - fraction * span))
        self.view_range = low, low + span

    def pan(self, delta_ms):
        if not math.isfinite(delta_ms):
            raise ValueError("Pan distance must be finite")
        if self.view_range:
            low, high = self.view_range
            span = high - low
            low = min(self.snapshot.duration_ms - span, max(0, low + delta_ms))
            self.view_range = low, low + span

    def hit_test(self, lane, time_ms):
        """Logarithmic lane hit test; hidden events cannot receive picks."""
        if not math.isfinite(time_ms):
            return None
        scopes = self._lanes.get(lane, ())
        index = bisect_right(self._starts.get(lane, ()), time_ms) - 1
        if index < 0:
            return None
        scope = scopes[index]
        return scope if scope.start_ms <= time_ms <= scope.end_ms and EventKey.of(scope) not in self.hidden else None


class MemoryModel:
    """Frame totals and captured outstanding totals remain distinct measurements."""
    def __init__(self, snapshot):
        self.snapshot = snapshot

    def rows(self, text="", *, sort="allocated_bytes", reverse=True):
        if sort not in ("file", "type", "line", "allocated_bytes", "deallocated_bytes", "allocation_count", "deallocation_count", "churn_bytes"):
            raise ValueError("Unknown memory sort column")
        term = text.casefold()
        rows = (r for r in self.snapshot.rows if not term or term in f"{r.file} {r.line} {r.type}".casefold())
        return tuple(sorted(rows, key=lambda r: (getattr(r, sort), r.file, r.line, r.type), reverse=reverse))

    def outstanding(self, text="", *, sort="bytes", reverse=True):
        if sort not in ("file", "type", "line", "index", "bytes"):
            raise ValueError("Unknown outstanding sort column")
        term = text.casefold()
        rows = [r for r in self.snapshot.outstanding if not term or term in f"{r.file} {r.line} {r.type}".casefold()]
        # Unknown captured values remain distinct and sort last in both directions.
        known = [r for r in rows if sort != "bytes" or r.bytes is not None]
        missing = [r for r in rows if sort == "bytes" and r.bytes is None]
        return tuple(sorted(known, key=lambda r: (getattr(r, sort), r.file, r.line, r.type), reverse=reverse) + missing)

    def export_csv(self, path, *, text="", sort="file", reverse=False, outstanding=False, overwrite=False):
        if not self.snapshot.supported:
            raise UnsupportedAnalysis(self.snapshot.reason)
        if outstanding:
            rows = ((r.file, r.line, r.type, r.index, r.bytes) for r in self.outstanding(text, sort=sort, reverse=reverse))
            header = ("file", "line", "type", "recorded_index", "recorded_outstanding_bytes")
        else:
            rows = ((r.file, r.line, r.type, r.allocated_bytes, r.allocation_count,
                     r.deallocated_bytes, r.deallocation_count, r.churn_bytes) for r in self.rows(text, sort=sort, reverse=reverse))
            header = ("file", "line", "type", "allocated_bytes", "allocations", "deallocated_bytes", "deallocations", "churn_bytes")
        return _write_csv(path, header, rows, overwrite=overwrite)


class DiagnosticsTasks:
    """Cancellable private-session work; the UI polls Future/progress as with AnalysisTasks."""
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pvd-diagnostics")
        self.jobs = []

    def _submit(self, function, request):
        job = AnalysisJob()
        job.request = request
        self.jobs = [j for j in self.jobs if not j.future.done()]
        self.jobs.append(job)
        job.future = submit_capture(self.executor, request.path, function, request, cancel=job.cancel_event, progress=job._update)
        return job

    def submit_profile(self, request):
        return self._submit(read_profile, request)

    def submit_memory(self, request):
        return self._submit(read_memory, request)

    def submit_history(self, request):
        return self._submit(read_profile_history, request)

    def close(self):
        for job in self.jobs:
            job.cancel()
        self.executor.shutdown(wait=True, cancel_futures=True)

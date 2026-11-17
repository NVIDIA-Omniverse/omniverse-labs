# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture analysis over owned native data, independent of OVUI widgets.

Property history follows PVDNext/mainwindow.cpp::onLoadValues: every raw frame
in a clip's [start, stop) interval is sampled, including both OVD simulation
phases. A private file Session preserves the displayed session's playhead and
does not block a live producer. The native scalar batch restores its own frame,
owns its data, and rejects nonnumeric/array properties. Missing values are gaps,
never invented zeros. Native graph values use doubles, as the legacy Qwt graph
did; integer properties above 2**53 may lose precision in plotted/CSV values.

CaptureMessages follows loadOvdMessages/classifyOvdMessage and exposes owned
native records, severity filtering and frame targets. Frame timing is available
only when the native PXD2 stream recorded it; OVD zero timings are not presented
as measurements. Neither this module nor the frontend parses capture bytes.
"""
from concurrent.futures import CancelledError, ThreadPoolExecutor
from dataclasses import dataclass
from enum import IntEnum
import csv
import math
import os
from pathlib import Path
from .ingestion import retain_capture, submit_capture
import tempfile
from threading import Event, Lock

from _pvd_native import Session


class UnsupportedAnalysis(ValueError):
    """The selected capture/property cannot supply this measurement."""


SCALAR_TYPES = frozenset(prefix + suffix for prefix in ("Px", "Pvd")
                         for suffix in ("F32", "F64", "I8", "I16", "I32", "I64",
                                        "U8", "U16", "U32", "U64", "Bool"))


def can_plot_property(prop):
    return (not prop.array and prop.type in SCALAR_TYPES
            and getattr(prop, "field_count", -1) in (-1, 1))


@dataclass(frozen=True)
class HistoryRequest:
    path: Path
    scene_id: int
    instance_id: int
    property_id: int
    start: int = 1
    stop: int | None = None

    @classmethod
    def from_display(cls, path, session, instance_id, property_id, start=1, stop=None, scene_id=None):
        if session is None:
            raise UnsupportedAnalysis("Open a capture before plotting a property")
        info = session.info()
        if info.live and not info.ended:
            raise UnsupportedAnalysis("Detach and save the live capture before plotting its property history")
        if path is None or not Path(path).is_file():
            raise UnsupportedAnalysis("Open a saved recording before plotting its property history")
        scenes = session.scenes()
        scene_id = scene_id if scene_id is not None else scenes[0].id if scenes else None
        if scene_id is None:
            raise UnsupportedAnalysis("This capture has no scene timeline")
        return cls(Path(path).expanduser().resolve(), int(scene_id), int(instance_id), int(property_id), start, stop)


@dataclass(frozen=True)
class ScalarSeries:
    request: HistoryRequest
    name: str
    type: str
    frames: tuple[int, ...]
    values: tuple[float | None, ...]
    unit: str = ""

    @property
    def missing(self):
        return sum(value is None for value in self.values)

    @property
    def bounds(self):
        finite = [value for value in self.values if value is not None and math.isfinite(value)]
        return (min(finite), max(finite)) if finite else None

    def nearest_frame(self, position):
        """Round within sampled frames for plot double-click→timeline navigation."""
        if not self.frames or not math.isfinite(float(position)):
            return None
        target = int(math.floor(float(position) + .5))
        return max(self.frames[0], min(target, self.frames[-1]))


def _check_cancel(cancel):
    if cancel is not None and cancel.is_set():
        raise CancelledError("Analysis cancelled")


def _frame_range(session, scene_id, start, stop, max_samples):
    scene = next((scene for scene in session.scenes() if scene.id == scene_id), None)
    if scene is None:
        raise ValueError("The requested scene is not in this capture")
    stop = scene.frame_count if stop is None else stop
    if (not isinstance(start, int) or not isinstance(stop, int)
            or not 0 <= start < stop <= scene.frame_count):
        raise ValueError("Analysis range must fit the scene and use an exclusive stop frame")
    if not isinstance(max_samples, int) or max_samples <= 0 or stop - start > max_samples:
        raise ValueError(f"Analysis range exceeds the {max_samples:,}-sample limit")
    return start, stop


def read_scalar_history(request, *, cancel=None, progress=None, batch_size=512, max_samples=1_000_000):
    """Load and sample in the calling worker; no displayed Session is mutated.

    ``progress(done,total)`` is called on that worker, after each native batch.
    Cancellation checks occur before/after file loading and between bounded
    batches. The result is returned only if complete; cancellation never returns
    a deceptively complete partial series.
    """
    if not isinstance(batch_size, int) or not 1 <= batch_size <= 4096:
        raise ValueError("Analysis batch size must be between 1 and 4096")
    _check_cancel(cancel)
    path = Path(request.path).expanduser().resolve()
    with retain_capture(path), tempfile.TemporaryDirectory(prefix="pvd-analysis-") as temporary:
        before = path.stat()
        session = Session(str(path), temporary)
        try:
            _check_cancel(cancel)
            start, stop = _frame_range(session, request.scene_id, request.start, request.stop, max_samples)
            frames, values = [], []
            name = type_name = ""
            if progress:
                progress(0, stop - start)
            for first in range(start, stop, batch_size):
                _check_cancel(cancel)
                batch = session.scalar_history(request.scene_id, request.instance_id, request.property_id,
                                               first, min(stop, first + batch_size))
                name, type_name = batch.property_name, batch.type
                frames.extend(int(frame) for frame in batch.frames)
                values.extend(float(value) if valid else None for value, valid in zip(batch.values, batch.valid))
                if progress:
                    progress(len(frames), stop - start)
            _check_cancel(cancel)
            after = path.stat()
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                    after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
                raise UnsupportedAnalysis("Capture changed during analysis; open a completed recording and try again")
            return ScalarSeries(request, name, type_name, tuple(frames), tuple(values))
        finally:
            # Release native file/cache handles before deleting its temporary files.
            del session


@dataclass(frozen=True)
class FrameTimingSeries:
    path: Path
    scene_id: int
    frames: tuple[int, ...]
    inner_ms: tuple[float, ...]
    total_ms: tuple[float, ...]


def read_frame_timing(path, scene_id, start=1, stop=None, *, cancel=None, progress=None, max_samples=1_000_000):
    """Two legacy frame graph curves, measured from native timestamp conversion."""
    _check_cancel(cancel)
    path = Path(path).expanduser().resolve()
    with retain_capture(path), tempfile.TemporaryDirectory(prefix="pvd-timing-") as temporary:
        session = Session(str(path), temporary)
        try:
            if session.info().ovd:
                raise UnsupportedAnalysis("OVD captures do not contain legacy PVD frame timing measurements")
            start, stop = _frame_range(session, scene_id, start, stop, max_samples)
            frames, inner, total = [], [], []
            if progress:
                progress(0, stop - start)
            for frame in range(start, stop):
                _check_cancel(cancel)
                session.seek(scene_id, frame)
                timing = session.frame_timing()
                if not timing.supported:
                    raise UnsupportedAnalysis("This capture has no supported frame timing measurements")
                frames.append(frame)
                inner.append(timing.inner_ms)
                total.append(timing.total_ms)
                if progress and ((frame - start + 1) % 128 == 0 or frame == stop - 1):
                    progress(frame - start + 1, stop - start)
            _check_cancel(cancel)
            return FrameTimingSeries(path, scene_id, tuple(frames), tuple(inner), tuple(total))
        finally:
            del session


def _write_csv(path, header, rows, *, overwrite=False):
    path = Path(path).expanduser().absolute()
    if path.suffix.lower() != ".csv":
        raise ValueError("Analysis exports must use a .csv extension")
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError("CSV destination must be a regular file")
    if path.exists() and not overwrite:
        raise FileExistsError(f"CSV already exists: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=".pvd-csv-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path, follow_symlinks=False)
        return path
    finally:
        Path(temporary).unlink(missing_ok=True)


def export_frame_timing_csv(series,path,*,overwrite=False):
    return _write_csv(path,("frame","physx_ms","total_ms"),
                      zip(series.frames,series.inner_ms,series.total_ms),overwrite=overwrite)


def export_series_csv(series, path, *, overwrite=False):
    """UTF-8 CSV with explicit frame column, named curves and blank missing values."""
    if isinstance(series, ScalarSeries):
        header = ("frame", series.name)
        rows = zip(series.frames, ("" if value is None else repr(value) for value in series.values))
    elif isinstance(series, FrameTimingSeries):
        header = ("frame", "PhysX frame (ms)", "Total frame (ms)")
        rows = zip(series.frames, series.inner_ms, series.total_ms)
    else:
        raise TypeError("Unsupported series type")
    return _write_csv(path, header, rows, overwrite=overwrite)


class MessageSeverity(IntEnum):
    INFO = 0
    WARNING = 1
    ERROR = 2


def classify_message(type_name):
    if not type_name:
        return MessageSeverity.INFO
    if "WARNING" in type_name:
        return MessageSeverity.WARNING
    if "INFO" in type_name:
        return MessageSeverity.INFO
    return MessageSeverity.ERROR


@dataclass(frozen=True)
class CaptureMessage:
    index: int
    frame: int
    type_name: str
    text: str
    file: str
    line: int
    severity: MessageSeverity
    type: int
    handle: int

    @property
    def location(self):
        return f"{self.file}:{self.line}"

    def target_frame(self, frame_count):
        return min(max(1, self.frame), frame_count - 1) if self.frame > 0 and frame_count > 1 else None


class CaptureMessages:
    """Append-only message adapter; refresh owns copies and reports new records.

    A fresh adapter belongs to each displayed Session. Native paging observes an
    initial bounded count so a producer appending messages cannot prolong one UI
    refresh indefinitely. ``limit`` bounds retained rows independently of native
    input resource limits; ``truncated`` exposes any omitted rows to the UI.
    """
    def __init__(self, session, *, limit=100_000):
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("Message limit must be a positive integer")
        self.session = session
        self.limit = limit
        self.records = ()
        self.total = 0
        self.truncated = False

    def refresh(self):
        total = self.session.message_count()
        target = min(total, self.limit)
        if total == self.total and target == len(self.records):
            return False
        previous_count = len(self.records)
        previous_total = self.total
        if target < len(self.records):
            self.records = ()
        records = list(self.records)
        for offset in range(len(records), target, 1000):
            native = self.session.messages(offset, min(1000, target - offset))
            records.extend(CaptureMessage(offset + index, item.frame_id, item.type_name, item.message,
                                          item.file, item.line, classify_message(item.type_name), item.type, item.handle)
                           for index, item in enumerate(native))
        changed = len(records) != previous_count or total != previous_total
        self.records = tuple(records)
        self.total, self.truncated = total, total > self.limit
        return changed

    @property
    def counts(self):
        return {severity: sum(record.severity == severity for record in self.records) for severity in MessageSeverity}

    def filtered(self, severity=None, text="", *, sort="frame", reverse=False):
        if sort not in ("frame", "severity", "type_name", "text", "file", "line", "location", "index"):
            raise ValueError("Unsupported message sort column")
        term = text.casefold()
        records = (record for record in self.records
                   if (severity is None or record.severity == MessageSeverity(severity))
                   and (not term or term in f"{record.type_name} {record.text} {record.location}".casefold()))
        def key(record):
            value=getattr(record,sort)
            if sort=="type_name":value=value or "(unknown)"
            return value
        return tuple(sorted(records, key=key, reverse=reverse))

    def export_csv(self, path, *, severity=None, text="", sort="frame", reverse=False, overwrite=False):
        return export_messages_csv(self.filtered(severity,text,sort=sort,reverse=reverse),path,overwrite=overwrite)


def export_messages_csv(records,path,*,overwrite=False):
    rows=((record.frame,record.severity.name.title(),record.type_name,record.text,record.file,record.line)
          for record in records)
    return _write_csv(path,("frame","severity","type","message","file","line"),rows,overwrite=overwrite)


class AnalysisJob:
    def __init__(self):
        self.cancel_event = Event()
        self._lock = Lock()
        self._progress = (0, 0)
        self.future = None

    def _update(self, done, total):
        with self._lock:
            self._progress = (done, total)

    @property
    def progress(self):
        with self._lock:
            return self._progress

    def cancel(self):
        self.cancel_event.set()
        if self.future:
            self.future.cancel()


class AnalysisTasks:
    """A cancellable worker for history sweeps; widget callbacks poll the Future."""
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pvd-analysis")
        self.jobs = []

    def submit_history(self, request, **options):
        job = AnalysisJob()
        self.jobs = [existing for existing in self.jobs if not existing.future.done()]
        self.jobs.append(job)
        job.future = submit_capture(self.executor, request.path, read_scalar_history, request, cancel=job.cancel_event,
                                          progress=job._update, **options)
        return job

    def submit_frame_timing(self, path, scene_id, start=1, stop=None, **options):
        job = AnalysisJob()
        self.jobs = [existing for existing in self.jobs if not existing.future.done()]
        self.jobs.append(job)
        job.future = submit_capture(self.executor, path, read_frame_timing, path, scene_id, start, stop,
                                          cancel=job.cancel_event, progress=job._update, **options)
        return job

    def close(self):
        for job in self.jobs:
            job.cancel()
        self.executor.shutdown(wait=True, cancel_futures=True)

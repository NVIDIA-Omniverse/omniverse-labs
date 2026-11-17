# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Legacy Error Stream over the first UserCustomedEvent.Events native array.

PVDNext/mainwindow.cpp::loadErrorLog displays Code, Message and File:Line from
PVDModel::getErrorLog(0). These records have no captured frame/timestamp field;
current_frame identifies the snapshot, never a made-up per-message timestamp.
OVD's separate CaptureMessages API is intentionally not adapted here.
"""
from dataclasses import dataclass
from pathlib import Path
from .ingestion import retain_capture
import tempfile

import _pvd_native as native
from .analysis import UnsupportedAnalysis, _check_cancel, _frame_range, _write_csv
from .profiling import DiagnosticsTasks


@dataclass(frozen=True)
class ErrorRequest:
    path: Path
    scene_id: int
    frame: int
    max_records: int = 10_000

    @classmethod
    def from_display(cls, path, session, scene_id=None):
        if session is None:
            raise UnsupportedAnalysis("Open a legacy capture to inspect its Error Stream")
        info = session.info()
        if info.ovd:
            raise UnsupportedAnalysis("OVD uses Capture messages; it has no legacy Error Stream")
        if info.live and not info.ended:
            raise UnsupportedAnalysis("Detach and save the capture before reading its Error Stream")
        if path is None or not Path(path).is_file():
            raise UnsupportedAnalysis("Open a saved capture to inspect its Error Stream")
        scenes = session.scenes()
        scene = next((s for s in scenes if scene_id is None or s.id == scene_id), None)
        if scene is None:
            raise UnsupportedAnalysis("The selected capture has no scene timeline")
        return cls(Path(path).expanduser().resolve(), scene.id, scene.current_frame)


def read_errors(request, *, cancel=None, progress=None):
    """Owned snapshot from a private file engine; never seeks the display engine."""
    _check_cancel(cancel)
    if progress:
        progress(0, 1)
    with retain_capture(request.path), tempfile.TemporaryDirectory(prefix="pvd-errors-") as temporary:
        session = native.Session(str(Path(request.path).expanduser().resolve()), temporary)
        try:
            if session.info().ovd:
                raise UnsupportedAnalysis("OVD uses Capture messages; it has no legacy Error Stream")
            _frame_range(session, request.scene_id, request.frame, request.frame + 1, 1)
            _check_cancel(cancel)
            if not session.seek(request.scene_id, request.frame):
                raise UnsupportedAnalysis("The requested error frame is not published")
            result = native.error_snapshot(session, request.scene_id, request.max_records)
            _check_cancel(cancel)
            if not result.supported:
                raise UnsupportedAnalysis(result.reason)
            if progress:
                progress(1, 1)
            return result
        finally:
            del session


class ErrorStreamModel:
    def __init__(self, snapshot):
        self.snapshot = snapshot

    def rows(self, text="", *, sort="index", reverse=False):
        if sort not in ("index", "code", "message", "file", "line"):
            raise ValueError("Unknown error sort column")
        term = text.casefold()
        rows = (r for r in self.snapshot.records if not term or term in f"{r.code} {r.message} {r.file}:{r.line}".casefold())
        return tuple(sorted(rows, key=lambda r: (getattr(r, sort), r.index), reverse=reverse))

    def export_csv(self, path, *, text="", sort="index", reverse=False, overwrite=False):
        if not self.snapshot.supported:
            raise UnsupportedAnalysis(self.snapshot.reason)
        rows = ((r.index, r.code, r.message, r.file, r.line) for r in self.rows(text, sort=sort, reverse=reverse))
        return _write_csv(path, ("index", "code", "message", "file", "line"), rows, overwrite=overwrite)


class ErrorTasks(DiagnosticsTasks):
    def submit_errors(self, request):
        return self._submit(read_errors, request)

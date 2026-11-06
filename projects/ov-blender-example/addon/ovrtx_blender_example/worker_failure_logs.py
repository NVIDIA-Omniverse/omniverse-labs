# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native log locations and error lines of a failed renderer worker.

Each worker epoch writes ``worker.log`` (its stdout/stderr) and ``ovrtx.log``
(the SDK's own log) into ``<output root>/epoch-N``. A worker failure names that
``worker.log`` in the user-visible message as ``Details: <path>``, so the path
survives every later rewording of the error text. The matched error lines and
any GPU crash dump go to the console and the viewport artifact, never into UI
text. bpy-free.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
import re
import threading
import time
from typing import Any, Mapping

from .direct_runtime_session import is_fatal_device_error

DETAILS_PREFIX = "Details: "
WORKER_LOG = "worker.log"
OVRTX_LOG = "ovrtx.log"
GPU_DUMP_SUFFIX = ".nv-gpudmp"
MAX_LINES = 20
# Only the end of a log is read: the failure is at its end, and a long session
# can leave a large log.
_TAIL_BYTES = 1024 * 1024
_ERROR_WORDS = re.compile(r"\b(error|fatal)\b", re.IGNORECASE)
_DETAILS = re.compile(re.escape(DETAILS_PREFIX) + r"(.+?" + re.escape(WORKER_LOG) + r")")
# A dump path is either absolute (spaces allowed, as in a Windows user folder)
# or a single token.
_GPU_DUMP = re.compile(
    r"((?:(?<![\w:])[A-Za-z]:[\\/]|(?<![\w.\\/])(?:\\\\|/))[^\r\n\"'<>|*?]*?" + re.escape(GPU_DUMP_SUFFIX)
    + r"|[^\s\"'<>|*?]+" + re.escape(GPU_DUMP_SUFFIX) + r")",
    re.IGNORECASE,
)
_RECENT_LIMIT = 8
_recent: deque[dict[str, Any]] = deque(maxlen=_RECENT_LIMIT)
_recent_lock = threading.Lock()


def epoch_logs(output_root: str | Path, epoch: int) -> dict[str, str]:
    """The ``worker.log`` and ``ovrtx.log`` paths of one worker epoch."""
    root = Path(output_root) / f"epoch-{int(epoch)}"
    return {"worker_log": str(root / WORKER_LOG), "ovrtx_log": str(root / OVRTX_LOG)}


def log_diagnostics(output_root: str | Path, epoch: int) -> dict[str, Any]:
    """Startup log diagnostics for a running worker (session_lifecycle's keys)."""
    logs = epoch_logs(output_root, epoch)
    return {
        "status": "file",
        "log_dir": str(Path(logs["worker_log"]).parent),
        "worker_log": logs["worker_log"],
        "renderer_log": logs["ovrtx_log"],
        "epoch": int(epoch),
    }


def _tail_lines(path: Path) -> list[str]:
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - _TAIL_BYTES))
            data = handle.read()
    except OSError:
        return []
    lines = data.decode("utf-8", errors="replace").splitlines()
    # A cut first line is partial unless the read started at the file start.
    return lines[1:] if size > _TAIL_BYTES else lines


def matching_line(line: str) -> bool:
    """A fatal device pattern (n-06-02) or the word ``error``/``fatal``."""
    return is_fatal_device_error(line) or bool(_ERROR_WORDS.search(line))


def collect(worker_log: str | Path, *, limit: int = MAX_LINES) -> dict[str, Any]:
    """Both logs of a worker epoch, its last matching lines and GPU dump paths."""
    worker_path = Path(worker_log)
    ovrtx_path = worker_path.with_name(OVRTX_LOG)
    lines: list[str] = []
    dumps: list[str] = []
    for path in (worker_path, ovrtx_path):
        for line in _tail_lines(path):
            text = line.strip()
            if not text:
                continue
            for match in _GPU_DUMP.findall(text):
                if match not in dumps:
                    dumps.append(match)
            if matching_line(text):
                lines.append(f"{path.name}: {text[:500]}")
    try:
        # A dump written next to the logs is reported even if no log names it.
        for path in sorted(worker_path.parent.glob("*" + GPU_DUMP_SUFFIX)):
            if str(path) not in dumps:
                dumps.append(str(path))
    except OSError:
        pass
    return {
        "worker_log": str(worker_path),
        "ovrtx_log": str(ovrtx_path),
        "worker_log_exists": worker_path.is_file(),
        "ovrtx_log_exists": ovrtx_path.is_file(),
        "lines": lines[-max(0, int(limit)):] if limit else [],
        "gpu_dumps": dumps,
    }


def with_details(message: str, worker_log: str | Path) -> str:
    """``message`` naming the worker log once, as ``Details: <path>``."""
    text = str(message).rstrip()
    if details_path(text):
        return text
    separator = "" if not text or text.endswith((".", "!", "?")) else "."
    return f"{text}{separator} {DETAILS_PREFIX}{worker_log}"


def details_path(message: Any) -> str:
    """The worker log a message names with ``Details: <path>``, or ``""``."""
    match = _DETAILS.search(str(message))
    return match.group(1) if match else ""


def failure_of(error: BaseException | None) -> dict[str, Any] | None:
    """The ``worker_failure`` an error or one of its causes carries, if any."""
    seen: set[int] = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        failure = getattr(error, "worker_failure", None)
        if isinstance(failure, Mapping) and failure.get("worker_log"):
            return dict(failure)
        error = error.__cause__ or error.__context__
    return None


def describe(error: BaseException, message: str | None = None) -> str:
    """``message`` (default ``str(error)``) naming the worker log behind ``error``."""
    text = str(error) if message is None else str(message)
    failure = failure_of(error)
    return with_details(text, failure["worker_log"]) if failure else text


def remember(summary: Mapping[str, Any], **context: Any) -> dict[str, Any]:
    """Keep a failure summary (newest last) for artifacts and later lookups."""
    record = {**dict(summary), **context, "observed_at_ns": time.time_ns()}
    with _recent_lock:
        for index, existing in enumerate(_recent):
            if existing.get("worker_log") == record.get("worker_log"):
                record = {**existing, **record}
                del _recent[index]
                break
        _recent.append(record)
    return dict(record)


def recent(since_ns: int = 0) -> list[dict[str, Any]]:
    """Remembered failure summaries observed at or after ``since_ns``."""
    with _recent_lock:
        return [dict(record) for record in _recent if int(record.get("observed_at_ns", 0)) >= since_ns]


def lookup(worker_log: str | Path) -> dict[str, Any]:
    """The remembered summary for ``worker_log``, collected now if unknown."""
    key = str(worker_log)
    with _recent_lock:
        for record in reversed(_recent):
            if record.get("worker_log") == key:
                return dict(record)
    return remember(collect(key))


def console_lines(summary: Mapping[str, Any]) -> list[str]:
    """Console mirror of a failure: log locations, matched lines, GPU dumps."""
    rows = [f"Native worker log: {summary.get('worker_log', '')}",
            f"Native OVRTX log: {summary.get('ovrtx_log', '')}"]
    rows.extend(f"  {line}" for line in summary.get("lines", ()) or ())
    rows.extend(f"GPU crash dump: {path}" for path in summary.get("gpu_dumps", ()) or ())
    return rows


def mirror_to_console(summary: Mapping[str, Any], level: Any = "WARNING", *, emit=None) -> list[str]:
    """Write :func:`console_lines` to the console only (never the Info window)."""
    if emit is None:
        from . import user_messages
        emit = user_messages.mirror_to_console
    rows = console_lines(summary)
    for row in rows:
        emit(level, row)
    return rows


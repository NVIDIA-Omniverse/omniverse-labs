# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in pyinstrument captures and exact Blender-side wall timers.

Set :data:`OUTPUT_DIR_ENV` before Blender starts.  The add-on then writes one
pyinstrument call tree for every profiled thread and a process-wide JSON timer
summary.  pyinstrument is deliberately an optional development dependency: a
normal extension install has no import or packaging dependency on it, and an
enabled capture reports a clear error while retaining the exact timers when it
is unavailable.

Timer scopes are inclusive wall-clock spans.  Nested scopes are useful for
attribution but must not be summed.  Each thread owns its pyinstrument
``Profiler`` because pyinstrument requires ``start()`` and ``stop()`` to run on
the same thread.
"""

from __future__ import annotations

import atexit
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any, ParamSpec, TypeVar


OUTPUT_DIR_ENV = "OV_BLENDER_EXAMPLE_PYINSTRUMENT_DIR"
INTERVAL_ENV = "OV_BLENDER_EXAMPLE_PYINSTRUMENT_INTERVAL"
PYINSTRUMENT_PATH_ENV = "OV_BLENDER_EXAMPLE_PYINSTRUMENT_PATH"
DEFAULT_INTERVAL_SECONDS = 0.001
TIMER_SAMPLE_LIMIT = 4096
SUMMARY_NAME = "ovrtx-blender-performance.json"

P = ParamSpec("P")
R = TypeVar("R")


@dataclass
class _TimerStat:
    count: int = 0
    total_ns: int = 0
    min_ns: int | None = None
    max_ns: int | None = None
    samples_ns: deque[int] = field(
        default_factory=lambda: deque(maxlen=TIMER_SAMPLE_LIMIT)
    )
    threads: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class _Capture:
    profiler: Any
    label: str
    thread_ident: int
    thread_name: str
    sequence: int


_lock = threading.RLock()
_timers: dict[str, _TimerStat] = {}
_captures: list[dict[str, Any]] = []
_errors: list[str] = []
_active_by_thread: dict[int, _Capture] = {}
_main_capture: _Capture | None = None
_sequence = 0
_atexit_installed = False
_missing_dependency_reported = False
_output_directory = (
    Path(os.environ[OUTPUT_DIR_ENV]).expanduser().resolve()
    if os.environ.get(OUTPUT_DIR_ENV, "").strip()
    else None
)


def enabled() -> bool:
    """Return whether capture was requested before the add-on was imported."""

    return _output_directory is not None


def output_directory() -> Path | None:
    return _output_directory


def _interval_seconds() -> float:
    value = os.environ.get(INTERVAL_ENV, "").strip()
    if not value:
        return DEFAULT_INTERVAL_SECONDS
    try:
        interval = float(value)
    except ValueError:
        _note_error(f"{INTERVAL_ENV} must be a positive number; got {value!r}")
        return DEFAULT_INTERVAL_SECONDS
    if interval <= 0.0:
        _note_error(f"{INTERVAL_ENV} must be positive; got {value!r}")
        return DEFAULT_INTERVAL_SECONDS
    return interval


def _note_error(message: str) -> None:
    message = str(message)
    with _lock:
        if message not in _errors:
            _errors.append(message)


def _record(scope: str, elapsed_ns: int) -> None:
    elapsed_ns = max(0, int(elapsed_ns))
    thread_name = threading.current_thread().name
    with _lock:
        stat = _timers.setdefault(str(scope), _TimerStat())
        stat.count += 1
        stat.total_ns += elapsed_ns
        stat.min_ns = elapsed_ns if stat.min_ns is None else min(stat.min_ns, elapsed_ns)
        stat.max_ns = elapsed_ns if stat.max_ns is None else max(stat.max_ns, elapsed_ns)
        stat.samples_ns.append(elapsed_ns)
        stat.threads[thread_name] = stat.threads.get(thread_name, 0) + 1


class _DisabledTimer:
    __slots__ = ()

    def __enter__(self) -> None:
        return None

    def __exit__(self, *_args: Any) -> None:
        return None


class _ActiveTimer:
    __slots__ = ("scope", "started_ns")

    def __init__(self, scope: str) -> None:
        self.scope = scope
        self.started_ns = 0

    def __enter__(self) -> None:
        self.started_ns = time.perf_counter_ns()
        return None

    def __exit__(self, *_args: Any) -> None:
        _record(self.scope, time.perf_counter_ns() - self.started_ns)


_DISABLED_TIMER = _DisabledTimer()


def timer(scope: str) -> _ActiveTimer | _DisabledTimer:
    """Measure one inclusive wall-clock scope when profiling is enabled."""

    return _ActiveTimer(str(scope)) if enabled() else _DISABLED_TIMER


def timed(scope: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorate an add-on boundary with an exact inclusive wall timer."""

    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        # OUTPUT_DIR_ENV is a launch-time switch. Returning the original
        # function keeps production callback and RPC paths wrapper-free.
        if not enabled():
            return function

        @wraps(function)
        def measured(*args: P.args, **kwargs: P.kwargs) -> R:
            with timer(scope):
                return function(*args, **kwargs)

        return measured

    return decorate


def _next_sequence() -> int:
    global _sequence
    with _lock:
        _sequence += 1
        return _sequence


def _start(label: str) -> _Capture | None:
    global _missing_dependency_reported
    if not enabled():
        return None
    thread_ident = threading.get_ident()
    with _lock:
        # A profiled entry point can legitimately run under the long-lived
        # main capture.  pyinstrument does not support nested profilers on one
        # thread, so the outer capture owns the call tree.
        if thread_ident in _active_by_thread:
            return None
    dependency_path = os.environ.get(PYINSTRUMENT_PATH_ENV, "").strip()
    if dependency_path and dependency_path not in sys.path:
        sys.path.insert(0, dependency_path)
    try:
        from pyinstrument import Profiler
    except (ImportError, OSError) as exc:
        message = (
            "pyinstrument capture requested but pyinstrument could not be imported: "
            f"{type(exc).__name__}: {exc}"
        )
        _note_error(message)
        if not _missing_dependency_reported:
            print(f"[ovrtx-profile] {message}", flush=True)
            _missing_dependency_reported = True
        _write_summary()
        return None
    profiler = Profiler(interval=_interval_seconds(), async_mode="disabled")
    thread_name = threading.current_thread().name
    capture = _Capture(
        profiler=profiler,
        label=str(label),
        thread_ident=thread_ident,
        thread_name=thread_name,
        sequence=_next_sequence(),
    )
    try:
        profiler.start(target_description=f"OVRTX Blender add-on: {label}")
    except Exception as exc:
        _note_error(
            f"could not start pyinstrument for {label!r}: {type(exc).__name__}: {exc}"
        )
        _write_summary()
        return None
    with _lock:
        _active_by_thread[thread_ident] = capture
    return capture


def _safe_file_stem(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    return value or "thread"


def _stop(capture: _Capture | None) -> None:
    if capture is None:
        return
    if threading.get_ident() != capture.thread_ident:
        _note_error(
            f"pyinstrument capture {capture.label!r} was stopped from a different thread"
        )
        _write_summary()
        return
    with _lock:
        active = _active_by_thread.get(capture.thread_ident)
    if active is not capture:
        return
    record: dict[str, Any] = {
        "label": capture.label,
        "thread_name": capture.thread_name,
        "thread_ident": capture.thread_ident,
        "sequence": capture.sequence,
        "files": {},
    }
    try:
        session = capture.profiler.stop()
        record.update(
            duration_seconds=float(getattr(session, "duration", 0.0)),
            cpu_seconds=float(getattr(session, "cpu_time", 0.0)),
            sample_count=int(getattr(session, "sample_count", 0)),
        )
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        _note_error(f"could not stop pyinstrument for {capture.label!r}: {record['error']}")
    finally:
        with _lock:
            _active_by_thread.pop(capture.thread_ident, None)

    directory = output_directory()
    if directory is not None and "error" not in record:
        directory.mkdir(parents=True, exist_ok=True)
        stem = (
            f"{os.getpid()}-{capture.sequence:02d}-"
            f"{_safe_file_stem(capture.label)}-{_safe_file_stem(capture.thread_name)}"
        )
        outputs = {
            "html": directory / f"{stem}.html",
            "text": directory / f"{stem}.txt",
            "json": directory / f"{stem}.json",
            "session": directory / f"{stem}.pyisession",
        }
        try:
            outputs["html"].write_text(
                capture.profiler.output_html(), encoding="utf-8"
            )
            outputs["text"].write_text(
                capture.profiler.output_text(
                    unicode=False,
                    color=False,
                    show_all=True,
                    processor_options={"filter_threshold": 0.0},
                ),
                encoding="utf-8",
            )
            from pyinstrument.renderers import JSONRenderer

            outputs["json"].write_text(
                capture.profiler.output(
                    JSONRenderer(
                        show_all=True,
                        processor_options={"filter_threshold": 0.0},
                    )
                ),
                encoding="utf-8",
            )
            capture.profiler.last_session.save(outputs["session"])
            record["files"] = {
                key: str(path) for key, path in outputs.items()
            }
        except Exception as exc:
            record["output_error"] = f"{type(exc).__name__}: {exc}"
            _note_error(
                f"could not write pyinstrument output for {capture.label!r}: "
                f"{record['output_error']}"
            )
    with _lock:
        _captures.append(record)
    _write_summary()


@contextmanager
def capture(label: str) -> Iterator[None]:
    """Profile an entry point on its current thread, unless already profiled."""

    token = _start(label)
    try:
        yield
    finally:
        _stop(token)


def profiled_thread(label: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorate a complete add-on thread/job entry point with pyinstrument."""

    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        if not enabled():
            return function

        @wraps(function)
        def profiled(*args: P.args, **kwargs: P.kwargs) -> R:
            with capture(label):
                return function(*args, **kwargs)

        return profiled

    return decorate


def start_main() -> None:
    """Start the Blender main-thread capture once."""

    global _main_capture, _atexit_installed
    if not enabled() or _main_capture is not None:
        return
    _main_capture = _start("blender-main")
    if not _atexit_installed:
        atexit.register(stop_main)
        _atexit_installed = True
    _write_summary()


def stop_main() -> None:
    """Stop and persist the Blender main-thread capture and timer summary."""

    global _main_capture
    capture_to_stop = _main_capture
    _main_capture = None
    _stop(capture_to_stop)
    _write_summary()


def _percentile_ns(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, int(len(ordered) * fraction + 0.999999999))
    return ordered[min(len(ordered) - 1, rank - 1)]


def _ms(value_ns: int | float | None) -> float | None:
    return None if value_ns is None else value_ns / 1_000_000.0


def summary() -> dict[str, Any]:
    """Return a JSON-safe snapshot of captures and exact timer statistics."""

    with _lock:
        timer_items = list(_timers.items())
        captures = [dict(item) for item in _captures]
        errors = list(_errors)
        active = [
            {
                "label": item.label,
                "thread_name": item.thread_name,
                "thread_ident": item.thread_ident,
                "sequence": item.sequence,
            }
            for item in _active_by_thread.values()
        ]
    timers: dict[str, Any] = {}
    for scope, stat in sorted(timer_items):
        samples = list(stat.samples_ns)
        timers[scope] = {
            "count": stat.count,
            "total_ms": _ms(stat.total_ns),
            "mean_ms": _ms(stat.total_ns / stat.count) if stat.count else 0.0,
            "min_ms": _ms(stat.min_ns),
            "p50_ms": _ms(_percentile_ns(samples, 0.50)),
            "p95_ms": _ms(_percentile_ns(samples, 0.95)),
            "max_ms": _ms(stat.max_ns),
            "retained_sample_count": len(samples),
            "threads": dict(sorted(stat.threads.items())),
        }
    return {
        "schema_version": 1,
        "artifact_id": "ovrtx-blender-performance",
        "process_id": os.getpid(),
        "clock": "time.perf_counter_ns",
        "timer_semantics": "inclusive_wall_time_do_not_sum_nested_scopes",
        "pyinstrument_interval_seconds": _interval_seconds(),
        "timer_sample_limit_per_scope": TIMER_SAMPLE_LIMIT,
        "timers": timers,
        "captures": captures,
        "active_captures": active,
        "errors": errors,
        "written_at_ns": time.time_ns(),
    }


def _write_summary() -> None:
    directory = output_directory()
    if directory is None:
        return
    try:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / SUMMARY_NAME
        temporary = directory / (
            f".{SUMMARY_NAME}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        temporary.write_text(
            json.dumps(summary(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except OSError as exc:
        _note_error(f"could not write performance summary: {type(exc).__name__}: {exc}")


__all__ = [
    "DEFAULT_INTERVAL_SECONDS",
    "INTERVAL_ENV",
    "OUTPUT_DIR_ENV",
    "PYINSTRUMENT_PATH_ENV",
    "SUMMARY_NAME",
    "capture",
    "enabled",
    "output_directory",
    "profiled_thread",
    "start_main",
    "stop_main",
    "summary",
    "timed",
    "timer",
]

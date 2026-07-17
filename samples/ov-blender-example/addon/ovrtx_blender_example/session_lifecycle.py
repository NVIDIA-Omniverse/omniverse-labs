# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Viewport session lifecycle status, logs, and recovery diagnostics."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, MutableMapping
from pathlib import Path
import json
import os
import tempfile
import time
from typing import Any


MAX_AUTO_RETRIES = 3

STATUS_STOPPED = "stopped"
STATUS_STARTING = "starting"
STATUS_ERROR = "error"
STATUS_LOADING = "loading"
STATUS_COMPILING = "compiling"
STATUS_RESYNCING = "resyncing"
STATUS_RECONNECTING = "reconnecting"
STATUS_RESTARTING = "restarting"
STATUS_LIVE = "live"
STATUS_CRASHED = "crashed"
STATUS_FAILED = "failed"
STATUS_CLEANUP_IN_PROGRESS = "cleanup_in_progress"

PHASE_LOADING = "loading"
PHASE_COMPILING = "compiling"
PHASE_RESYNCING = "resyncing"
PHASE_RECONNECTING = "reconnecting"
PHASE_RESTARTING = "restarting"

WORKER_LOG_ENV = "OV_BLENDER_EXAMPLE_WORKER_LOG"
NATIVE_WORKER_LOG_ENV = "OVRTX_EXAMPLE_WORKER_LOG"
RENDERER_LOG_ENV = "OVRTX_WORKER_RENDERER_LOG"


def derive_status(
    *,
    engine_active: bool,
    ready: bool,
    busy: bool = False,
    phase: str = "",
    failure_count: int = 0,
    worker_exit_code: int | None = None,
    cleanup_in_progress: bool = False,
    max_auto_retries: int = MAX_AUTO_RETRIES,
    last_error: str = "",
) -> dict[str, Any]:
    """Project runtime/session inputs into artist-facing lifecycle status."""

    if cleanup_in_progress:
        return _status(STATUS_CLEANUP_IN_PROGRESS, "Cleaning up OVRTX processes")
    if busy:
        if phase == PHASE_RESTARTING:
            return _status(
                STATUS_RESTARTING,
                "Restarting OVRTX worker",
                "Recycling the OVRTX worker process.",
            )
        if phase == PHASE_RECONNECTING:
            return _status(STATUS_RECONNECTING, "Reconnecting to OVRTX")
        if phase == PHASE_LOADING:
            return _status(
                STATUS_LOADING,
                "Loading scene in OVRTX",
                "Blender may be unresponsive while OVRTX loads the scene.",
            )
        if phase == PHASE_COMPILING:
            return _status(
                STATUS_COMPILING,
                "Compiling OVRTX shaders",
                "First run compiles materials and pipelines; this can take a few minutes.",
            )
        if phase == PHASE_RESYNCING:
            return _status(STATUS_RESYNCING, "Re-syncing scene")
        return _status(STATUS_STARTING, "Starting OVRTX")
    if worker_exit_code is not None:
        return _status(
            STATUS_CRASHED,
            "OVRTX renderer stopped unexpectedly",
            f"Renderer process exited with code {worker_exit_code}.",
        )
    if ready:
        return _status(STATUS_LIVE, "Live")
    if not engine_active:
        return _status(STATUS_STOPPED, "Stopped")
    if last_error:
        # A not-ready engine with a recorded viewport error must say so:
        # an eternal "Starting OVRTX" while (for example) scene conversion
        # is blocked hides the actionable failure (Junk Shop regression,
        # 2026-07-07). The error clears on the next successful frame.
        return _status(STATUS_ERROR, "Viewport error", last_error)
    if failure_count >= max(0, int(max_auto_retries)):
        return _status(
            STATUS_FAILED,
            "OVRTX could not start",
            f"Gave up after {int(failure_count)} attempt(s).",
        )
    return _status(STATUS_STARTING, "Starting OVRTX", "Waiting for the first frame.")


def should_auto_retry(failure_count: int, *, max_auto_retries: int = MAX_AUTO_RETRIES) -> bool:
    return int(failure_count) < max(0, int(max_auto_retries))


def base_dir() -> Path:
    return Path(tempfile.gettempdir()) / "ov-blender-example"


def prepare_logs(
    env: MutableMapping[str, str] = os.environ,
) -> dict[str, Any]:
    """Resolve worker/renderer log routing before worker startup.

    Default: no log files. With the log env overrides unset, the worker
    child inherits Blender's stdout/stderr, so worker/renderer output
    lands in the console alongside the add-on's own messages (session
    lifecycle transitions additionally surface as Info-panel reports).
    Explicitly set overrides (validation lanes that must capture logs)
    keep file logging: a set worker-log value is mirrored across both
    worker-log keys so the native client and the worker agree.
    """

    worker_log = env.get(WORKER_LOG_ENV) or env.get(NATIVE_WORKER_LOG_ENV) or ""
    if worker_log:
        env.setdefault(WORKER_LOG_ENV, worker_log)
        env.setdefault(NATIVE_WORKER_LOG_ENV, worker_log)
    return log_diagnostics(env)


def log_diagnostics(env: Mapping[str, str] = os.environ) -> dict[str, Any]:
    worker_log = str(env.get(WORKER_LOG_ENV, "") or env.get(NATIVE_WORKER_LOG_ENV, ""))
    renderer_log = str(env.get(RENDERER_LOG_ENV, ""))
    if worker_log:
        directory = str(Path(worker_log).expanduser().parent)
    elif renderer_log:
        directory = str(Path(renderer_log).expanduser().parent)
    else:
        directory = ""
    return {
        "status": "file" if (worker_log or renderer_log) else "stdout",
        "log_dir": directory,
        "worker_log": worker_log,
        "renderer_log": renderer_log,
        "worker_log_env": WORKER_LOG_ENV,
        "native_worker_log_env": NATIVE_WORKER_LOG_ENV,
        "renderer_log_env": RENDERER_LOG_ENV,
    }


def marker_path(directory: str | Path | None = None) -> Path:
    return base_dir() / "startup-crash-marker.json" if directory is None else Path(directory) / "startup-crash-marker.json"


def write_crash_marker(
    *,
    phase: str,
    scene_name: str = "",
    pid: int | None = None,
    directory: str | Path | None = None,
    now: Callable[[], float] = time.time,
) -> dict[str, Any]:
    path = marker_path(directory)
    payload = {
        "phase": str(phase or ""),
        "scene": str(scene_name or ""),
        "pid": int(pid if pid is not None else os.getpid()),
        "written_at": float(now()),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError as exc:
        return {
            "status": "failed",
            "path": str(path),
            "error": f"{type(exc).__name__}: {exc}",
            "marker_active": False,
        }
    return {"status": "written", "path": str(path), "marker_active": True, **payload}


def clear_crash_marker(directory: str | Path | None = None) -> dict[str, Any]:
    path = marker_path(directory)
    try:
        existed = path.exists()
        path.unlink(missing_ok=True)
    except OSError as exc:
        return {
            "status": "failed",
            "path": str(path),
            "error": f"{type(exc).__name__}: {exc}",
            "marker_active": True,
        }
    return {"status": "cleared", "path": str(path), "existed": existed, "marker_active": False}


def clear_crash_marker_if_mine(
    *,
    current_pid: int | None = None,
    directory: str | Path | None = None,
) -> dict[str, Any]:
    marker = read_crash_marker(directory=directory)
    pid = marker.get("pid")
    if pid != int(current_pid if current_pid is not None else os.getpid()):
        return {"status": "not_mine", "path": str(marker_path(directory)), "marker_active": bool(marker)}
    return clear_crash_marker(directory)


def read_crash_marker(directory: str | Path | None = None) -> dict[str, Any]:
    path = marker_path(directory)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "invalid", "path": str(path), "error": f"{type(exc).__name__}: {exc}"}
    if not isinstance(payload, Mapping):
        return {"status": "invalid", "path": str(path), "error": "marker payload is not an object"}
    return {"status": "recorded", "path": str(path), **dict(payload)}


def read_stale_crash_marker(
    *,
    current_pid: int | None = None,
    directory: str | Path | None = None,
    pid_running: Callable[[int], bool] | None = None,
) -> dict[str, Any]:
    marker = read_crash_marker(directory=directory)
    if not marker or marker.get("status") == "invalid":
        return marker if marker.get("status") == "invalid" else {}
    pid = marker.get("pid")
    me = int(current_pid if current_pid is not None else os.getpid())
    if not isinstance(pid, int) or pid == me:
        return {}
    checker = pid_running or _pid_running
    if checker(pid):
        return {}
    return {**marker, "status": "stale", "stale": True}


def select_cleanup_targets(
    processes: Iterable[Mapping[str, Any]],
    *,
    own_pids: Iterable[int] = (),
    include_non_orphans: bool = False,
) -> tuple[int, ...]:
    own = {int(pid) for pid in own_pids}
    selected: list[int] = []
    for process in processes:
        try:
            pid = int(process["pid"])
        except (KeyError, TypeError, ValueError):
            continue
        if pid in own:
            continue
        parent_alive = bool(process.get("parent_alive", True))
        if include_non_orphans or not parent_alive:
            selected.append(pid)
    return tuple(selected)


def cleanup_diagnostics(
    *,
    processes: Iterable[Mapping[str, Any]],
    own_pids: Iterable[int] = (),
    selected_pids: Iterable[int] = (),
    killed_count: int = 0,
    include_non_orphans: bool = False,
    status: str = "planned",
) -> dict[str, Any]:
    process_list = [dict(process) for process in processes]
    own = [int(pid) for pid in own_pids]
    selected = [int(pid) for pid in selected_pids]
    return {
        "status": str(status),
        "process_count": len(process_list),
        "own_pids": own,
        "selected_pids": selected,
        "selected_count": len(selected),
        "killed_count": int(killed_count),
        "include_non_orphans": bool(include_non_orphans),
    }


def _status(state: str, label: str, hint: str = "") -> dict[str, Any]:
    return {"status": state, "label": label, "hint": hint}


_WINDOWS_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_WINDOWS_STILL_ACTIVE = 259
_WINDOWS_ERROR_ACCESS_DENIED = 5


def pid_running(pid: int) -> bool:
    """Best-effort liveness check for a local process ID.

    On Windows ``os.kill(pid, 0)`` is *not* a liveness probe: CPython
    implements any non-console-event signal as ``TerminateProcess``, so
    probing would kill the target. The Windows branch uses
    ``OpenProcess`` + ``GetExitCodeProcess`` instead. Uncertain states
    (access denied, unreadable exit code) report running so callers
    never treat a possibly-live owner as dead.
    """

    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if os.name == "nt":
        return _windows_pid_running(pid)
    return _posix_pid_running(pid)


def _posix_pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _windows_pid_running(pid: int) -> bool:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)

    handle = kernel32.OpenProcess(_WINDOWS_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return ctypes.get_last_error() == _WINDOWS_ERROR_ACCESS_DENIED
    try:
        exit_code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return True
        return exit_code.value == _WINDOWS_STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


_pid_running = pid_running

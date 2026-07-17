# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Best-effort backstop that terminates orphaned runtime worker processes.

DISCUSSION ARTIFACT (dev/jomiller/fix-ovphysx-server-orphan). On Blender close
``ovphysx-bridge-server`` orphans while ``ovrtx-bridge-server`` does not: Blender
does not reliably run add-on ``unregister()`` on application quit, there is no
``atexit`` hook, and the native ``start_worker`` result exposes no PID/handle to
Python (only ``worker_process_alive``/``address``). So Python cannot cleanly
target its own workers — it has to scrape the process table by executable name.

That awkwardness is the argument for moving the fix **down into the native
client**, which owns the spawn handle and can place the worker in a Windows Job
Object (``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``) — covering crashes and hard
kills this backstop cannot. This module is the narrow, low-risk Python-side
mitigation for the graceful-quit case, and the seam (``register_worker_pid``)
for the native fix if the client is taught to report its PID.

Why this is safe to run at interpreter finalization:

- No ``bpy``/RNA access, no gRPC, no thread joins — only OS process calls.
- It only ever targets a process that BOTH matches a known worker executable
  basename AND is a direct child of this process; never an unrelated PID.
- ``TerminateProcess`` (not the native gRPC ``shutdown``) is used deliberately:
  a hard OS kill needs no gRPC channel or thread coordination, both of which
  are unstable while the interpreter is being torn down.
- Every operation is guarded and swallows exceptions; a failure to reap never
  raises out of the ``atexit`` handler.
- The handler is registered exactly once and removed on ``unregister()`` so an
  add-on reload/disable can neither leak nor double-register it.

Coverage limits (further arguments for the native fix): only fires on graceful
interpreter exit (not crashes/``TerminateProcess``/Task-Manager kill), and only
matches workers that are *direct* children of this process.
"""

from __future__ import annotations

import atexit
import os
import threading


#: Fixed worker executables this add-on spawns. Matches ``bundled_runtime``:
#: ``ovrtx-bridge-server{suffix}`` and ``ovphysx-bridge-server{suffix}``. Compared
#: case-insensitively by basename. ``register_worker_executable`` can widen this
#: if a native client ever reports a different path.
_DEFAULT_WORKER_BASENAMES = frozenset(
    {
        "ovrtx-bridge-server",
        "ovrtx-bridge-server.exe",
        "ovphysx-bridge-server",
        "ovphysx-bridge-server.exe",
    }
)

_lock = threading.Lock()
_installed = False
_worker_basenames: set[str] = set(_DEFAULT_WORKER_BASENAMES)
_explicit_pids: set[int] = set()


def install() -> None:
    """Register the exit-time reap exactly once (call from ``register()``)."""

    global _installed
    with _lock:
        if _installed:
            return
        atexit.register(reap)
        _installed = True


def uninstall() -> None:
    """Remove the exit-time reap (call from ``unregister()``).

    Load-bearing: ``atexit`` registrations are not tied to add-on lifetime, so
    without this a reload/disable would leak a handler closing over stale state
    and a re-enable would double-register.
    """

    global _installed
    with _lock:
        if not _installed:
            return
        try:
            atexit.unregister(reap)
        except Exception:
            pass
        _installed = False


def register_worker_executable(name_or_path: str) -> None:
    """Also treat this executable basename as one of our workers (optional).

    The fixed defaults already cover both shipped workers; this exists so a
    future native client that reports its worker path can widen the match.
    """

    base = _basename_lower(name_or_path)
    if base:
        with _lock:
            _worker_basenames.add(base)


def register_worker_pid(pid: int) -> None:
    """Record a worker PID to terminate at exit.

    The native ``start_worker`` result exposes no PID today, so nothing calls
    this yet — it is the seam for the native-client fix under discussion. If the
    client hands Python the spawned PID, the exit backstop becomes exact and the
    process-table scan is no longer needed.
    """

    value = _coerce_pid(pid)
    if value is not None:
        with _lock:
            _explicit_pids.add(value)


def forget_worker_pid(pid: int) -> None:
    """Drop a PID once its worker has been shut down cleanly."""

    value = _coerce_pid(pid)
    if value is not None:
        with _lock:
            _explicit_pids.discard(value)


def reap() -> int:
    """Terminate our still-running worker processes; return the count killed.

    Idempotent and best-effort. Safe to call from ``atexit`` and from
    ``unregister()`` (the disable path, which has the same orphan gap as quit).
    """

    try:
        own_pid = os.getpid()
        with _lock:
            basenames = frozenset(_worker_basenames)
            pids = set(_explicit_pids)
        if os.name == "nt":
            try:
                pids |= _windows_child_worker_pids(basenames, own_pid)
            except Exception:
                pass
        killed = 0
        for pid in pids:
            if pid == own_pid or pid <= 0:
                continue
            if _terminate_pid(pid):
                killed += 1
        return killed
    except Exception:
        return 0


def _basename_lower(value: str) -> str:
    try:
        # Normalize backslashes so a Windows-style worker path resolves to its
        # basename even on a POSIX host, where ``os.path.basename`` treats "\"
        # as an ordinary character rather than a separator.
        text = str(value).strip().strip('"').replace("\\", "/")
        return os.path.basename(text).lower()
    except Exception:
        return ""


def _coerce_pid(pid: object) -> int | None:
    try:
        value = int(pid)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _terminate_pid(pid: int) -> bool:
    try:
        if os.name == "nt":
            return _windows_terminate(pid)
        import signal

        os.kill(pid, signal.SIGKILL)
        return True
    except Exception:
        return False


def _windows_terminate(pid: int) -> bool:
    import ctypes
    from ctypes import wintypes

    _PROCESS_TERMINATE = 0x0001
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.TerminateProcess.argtypes = (wintypes.HANDLE, wintypes.UINT)
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = kernel32.OpenProcess(_PROCESS_TERMINATE, False, int(pid))
    if not handle:
        return False
    try:
        return bool(kernel32.TerminateProcess(handle, 1))
    finally:
        kernel32.CloseHandle(handle)


def _windows_child_worker_pids(basenames: frozenset[str], parent_pid: int) -> set[int]:
    """PIDs of live worker processes that are direct children of ``parent_pid``."""

    import ctypes
    from ctypes import wintypes

    _TH32CS_SNAPPROCESS = 0x00000002
    _INVALID_HANDLE = wintypes.HANDLE(-1).value

    class _PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.CreateToolhelp32Snapshot.argtypes = (wintypes.DWORD, wintypes.DWORD)
    kernel32.Process32FirstW.argtypes = (wintypes.HANDLE, ctypes.POINTER(_PROCESSENTRY32W))
    kernel32.Process32NextW.argtypes = (wintypes.HANDLE, ctypes.POINTER(_PROCESSENTRY32W))
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)

    snapshot = kernel32.CreateToolhelp32Snapshot(_TH32CS_SNAPPROCESS, 0)
    if not snapshot or snapshot == _INVALID_HANDLE:
        return set()
    pids: set[int] = set()
    try:
        entry = _PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(_PROCESSENTRY32W)
        ok = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while ok:
            if (
                int(entry.th32ParentProcessID) == parent_pid
                and str(entry.szExeFile).lower() in basenames
            ):
                pids.add(int(entry.th32ProcessID))
            ok = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    return pids


__all__ = [
    "install",
    "uninstall",
    "reap",
    "register_worker_executable",
    "register_worker_pid",
    "forget_worker_pid",
]

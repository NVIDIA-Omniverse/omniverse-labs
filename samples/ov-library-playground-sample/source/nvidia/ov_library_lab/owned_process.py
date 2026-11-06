# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Windows venv redirectors spawn a child. Track and close that owned tree."""
import ctypes
from ctypes import wintypes as w
import subprocess
import threading


def children(parent):
    class Entry(ctypes.Structure):
        _fields_ = [('dwSize', w.DWORD), ('cntUsage', w.DWORD), ('pid', w.DWORD),
                    ('heap', ctypes.c_size_t), ('module', w.DWORD), ('threads', w.DWORD),
                    ('parent', w.DWORD), ('priority', w.LONG), ('flags', w.DWORD), ('exe', w.WCHAR*260)]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateToolhelp32Snapshot.restype = w.HANDLE
    kernel.Process32FirstW.argtypes = [w.HANDLE, ctypes.POINTER(Entry)]
    kernel.Process32NextW.argtypes = [w.HANDLE, ctypes.POINTER(Entry)]
    kernel.CloseHandle.argtypes = [w.HANDLE]
    snapshot = kernel.CreateToolhelp32Snapshot(2, 0)
    if snapshot == w.HANDLE(-1).value:
        return []
    found = []
    try:
        entry = Entry()
        entry.dwSize = ctypes.sizeof(entry)
        ok = kernel.Process32FirstW(snapshot, ctypes.byref(entry))
        while ok:
            if entry.parent == parent:
                found.append(entry.pid)
            ok = kernel.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel.CloseHandle(snapshot)
    return found


class OwnedProcess:
    def __init__(self, *args, **kwargs):
        self.process = subprocess.Popen(*args, **kwargs)
        self.pid = self.process.pid
        self.child_handles = {}
        self._lock = threading.RLock()
        self._released = False
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        self.kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
        self.kernel.OpenProcess.restype = w.HANDLE
        self.kernel.TerminateProcess.argtypes = [w.HANDLE, w.UINT]
        self.kernel.WaitForSingleObject.argtypes = [w.HANDLE, w.DWORD]
        self.kernel.CloseHandle.argtypes = [w.HANDLE]

    def owns(self, pid):
        with self._lock:
            return self._owns(pid)

    def _owns(self, pid):
        if self._released:
            return False
        if pid == self.pid:
            return True
        # Keep real handles, not stale PIDs, so shutdown cannot hit a reused PID.
        if pid not in self.child_handles and self.poll() is None and pid in children(self.pid):
            handle = self.kernel.OpenProcess(0x100001, False, pid)  # synchronize + terminate
            if handle:
                self.child_handles[pid] = handle
        return pid in self.child_handles

    def poll(self):
        return self.process.poll()

    def wait(self, timeout):
        return self.process.wait(timeout=timeout)

    def terminate(self):
        with self._lock:
            self._terminate()

    def _terminate(self):
        if self._released:
            return
        if self.poll() is None:
            for pid in children(self.pid):
                self.owns(pid)
        for handle in self.child_handles.values():
            if self.kernel.WaitForSingleObject(handle, 0) == 258:
                self.kernel.TerminateProcess(handle, 1)
                self.kernel.WaitForSingleObject(handle, 1000)
        if self.poll() is None:
            self.process.terminate()

    kill = terminate

    def release(self):
        # The UI's final shutdown and the session worker can finish together.
        # Serialize handle access so no handle is closed while termination uses it.
        with self._lock:
            for handle in self.child_handles.values():
                self.kernel.CloseHandle(handle)
            self.child_handles.clear()
            self._released = True

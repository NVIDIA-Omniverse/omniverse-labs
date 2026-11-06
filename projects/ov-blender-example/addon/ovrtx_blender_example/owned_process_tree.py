# SPDX-License-Identifier: Apache-2.0
"""OS ownership independent of the Python launcher/producer PID distinction."""
from __future__ import annotations
import os
import signal
import subprocess
import time


class OwnedProcessTree:
    def __init__(self):
        self.windows = os.name == 'nt'
        self.process = None
        self.job = None
        self.worker_handle = None
        self.worker_pid = None
        self.worker_returncode = None
        self.member_handles = []
        if os.name == 'nt':
            import ctypes
            from ctypes import wintypes as w
            self.ctypes = ctypes
            self.api = ctypes.WinDLL('kernel32', use_last_error=True)
            signatures = {
                'CreateJobObjectW': ([w.LPVOID, w.LPCWSTR], w.HANDLE),
                'AssignProcessToJobObject': ([w.HANDLE, w.HANDLE], w.BOOL),
                'IsProcessInJob': ([w.HANDLE, w.HANDLE, ctypes.POINTER(w.BOOL)], w.BOOL),
                'OpenProcess': ([w.DWORD, w.BOOL, w.DWORD], w.HANDLE),
                'WaitForSingleObject': ([w.HANDLE, w.DWORD], w.DWORD),
                'GetExitCodeProcess': ([w.HANDLE, ctypes.POINTER(w.DWORD)], w.BOOL),
                'CloseHandle': ([w.HANDLE], w.BOOL),
                'TerminateJobObject': ([w.HANDLE, w.UINT], w.BOOL),
                'QueryInformationJobObject': ([w.HANDLE, ctypes.c_int, w.LPVOID, w.DWORD, w.LPVOID], w.BOOL),
                'SetInformationJobObject': ([w.HANDLE, ctypes.c_int, w.LPVOID, w.DWORD], w.BOOL),
            }
            for name, (args, result) in signatures.items():
                fn = getattr(self.api, name); fn.argtypes = args; fn.restype = result
            self.job = self.api.CreateJobObjectW(None, None)
            if not self.job:
                raise ctypes.WinError(ctypes.get_last_error())
            # JOBOBJECT_EXTENDED_LIMIT_INFORMATION; pointer-sized fields on x64.
            class Basic(ctypes.Structure):
                _fields_ = [('per_process', ctypes.c_int64), ('per_job', ctypes.c_int64),
                    ('flags', w.DWORD), ('min_ws', ctypes.c_size_t), ('max_ws', ctypes.c_size_t),
                    ('active', w.DWORD), ('affinity', ctypes.c_size_t), ('priority', w.DWORD), ('scheduling', w.DWORD)]
            class Extended(ctypes.Structure):
                _fields_ = [('basic', Basic), ('io', ctypes.c_uint64 * 6),
                    ('process_memory', ctypes.c_size_t), ('job_memory', ctypes.c_size_t),
                    ('peak_process', ctypes.c_size_t), ('peak_job', ctypes.c_size_t)]
            limits = Extended(); limits.basic.flags = 0x2000  # KILL_ON_JOB_CLOSE
            if not self.api.SetInformationJobObject(self.job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
                self.api.CloseHandle(self.job); self.job = None
                raise ctypes.WinError(ctypes.get_last_error())

    def spawn(self, command, **options):
        if self.job:
            options['creationflags'] = options.get('creationflags', 0) | 4  # CREATE_SUSPENDED
        self.process = subprocess.Popen(command, **options)
        if self.job:
            try:
                if not self.api.AssignProcessToJobObject(self.job, int(self.process._handle)):
                    raise self.ctypes.WinError(self.ctypes.get_last_error())
                resume = self.ctypes.WinDLL('ntdll').NtResumeProcess
                resume.argtypes = [self.ctypes.c_void_p]; resume.restype = self.ctypes.c_long
                if resume(int(self.process._handle)) < 0:
                    raise OSError('Could not resume owned launcher')
            except BaseException:
                self.process.kill(); self.process.wait(timeout=2)
                self.api.CloseHandle(self.job); self.job = None
                raise
        return self.process

    def contains(self, pid):
        if type(pid) is not int or pid <= 0:
            return False
        if self.job:
            from ctypes import wintypes as w
            handle = self.api.OpenProcess(0x1000, False, pid)
            if not handle:
                return False
            try:
                result = w.BOOL()
                return bool(self.api.IsProcessInJob(handle, self.job, self.ctypes.byref(result)) and result.value)
            finally:
                self.api.CloseHandle(handle)
        try:
            return os.getpgid(pid) == self.process.pid
        except ProcessLookupError:
            return False

    def authenticate(self, pid):
        if not self.contains(pid):
            raise ValueError('Worker is outside the owned process tree')
        if self.job:
            from ctypes import wintypes as w
            handle = self.api.OpenProcess(0x1000 | 0x100000, False, pid)
            if not handle:
                raise self.ctypes.WinError(self.ctypes.get_last_error())
            result = w.BOOL()
            if not self.api.IsProcessInJob(handle, self.job, self.ctypes.byref(result)) or not result.value:
                self.api.CloseHandle(handle)
                raise ValueError('Worker changed during authentication')
            self.worker_handle = handle
        self.worker_pid = pid

    def poll_worker(self):
        if self.worker_handle:
            from ctypes import wintypes as w
            if self.api.WaitForSingleObject(self.worker_handle, 0) == 0x102:
                return None
            code = w.DWORD()
            if not self.api.GetExitCodeProcess(self.worker_handle, self.ctypes.byref(code)):
                raise self.ctypes.WinError(self.ctypes.get_last_error())
            self.worker_returncode = code.value
            return code.value
        if self.worker_returncode is not None:
            return self.worker_returncode
        # POSIX exec preserves the launcher PID and its waitable ownership.
        return self.process.poll() if self.process is not None else None

    def stop(self):
        """Reap the launcher AND all owned descendants, even if launcher exited."""
        if self.job:
            # Job accounting can reach zero before process handles signal.
            # Pin members before termination so PID reuse cannot turn cleanup
            # confirmation into an unrelated-process liveness probe.
            capacity = 64
            while True:
                data = self.ctypes.create_string_buffer(8 + capacity * self.ctypes.sizeof(self.ctypes.c_size_t))
                if self.api.QueryInformationJobObject(self.job, 3, data, len(data), None):
                    break
                if self.ctypes.get_last_error() != 234 or capacity >= 65536:
                    raise self.ctypes.WinError(self.ctypes.get_last_error())
                capacity *= 2
            count = int.from_bytes(data.raw[4:8], 'little')
            width = self.ctypes.sizeof(self.ctypes.c_size_t)
            for index in range(count):
                pid = int.from_bytes(data.raw[8 + index * width:8 + (index + 1) * width], 'little')
                handle = self.api.OpenProcess(0x100000, False, pid)
                if handle:
                    self.member_handles.append(handle)
                elif self.ctypes.get_last_error() != 87:
                    raise self.ctypes.WinError(self.ctypes.get_last_error())
            if not self.api.TerminateJobObject(self.job, 1):
                raise self.ctypes.WinError(self.ctypes.get_last_error())
            deadline = time.monotonic() + 3
            while True:
                # Basic accounting ends with TotalProcesses, ActiveProcesses, TerminatedProcesses.
                data = self.ctypes.create_string_buffer(48)
                if not self.api.QueryInformationJobObject(self.job, 1, data, len(data), None):
                    raise self.ctypes.WinError(self.ctypes.get_last_error())
                if int.from_bytes(data.raw[40:44], 'little') == 0:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError('Owned process tree teardown not confirmed')
                time.sleep(.01)
            for handle in self.member_handles:
                timeout_ms = max(0, int((deadline - time.monotonic()) * 1000))
                if self.api.WaitForSingleObject(handle, timeout_ms) != 0:
                    raise RuntimeError('Owned descendant process death not confirmed')
            while self.member_handles:
                if not self.api.CloseHandle(self.member_handles[-1]):
                    raise self.ctypes.WinError(self.ctypes.get_last_error())
                self.member_handles.pop()
            if not self.api.CloseHandle(self.job):
                raise self.ctypes.WinError(self.ctypes.get_last_error())
            self.job = None
            if self.worker_handle:
                self.poll_worker()
                if not self.api.CloseHandle(self.worker_handle):
                    raise self.ctypes.WinError(self.ctypes.get_last_error())
                self.worker_handle = None
        elif self.process is not None and not self.windows:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if self.process is not None:
            self.process.wait(timeout=3)
            if not self.windows:
                # Sending SIGKILL is not a receipt of descendant death. Reap
                # our child then confirm the dedicated group no longer exists.
                deadline = time.monotonic() + 3
                while True:
                    try:
                        os.killpg(self.process.pid, 0)
                    except ProcessLookupError:
                        break
                    if time.monotonic() >= deadline:
                        raise RuntimeError('Owned process group teardown not confirmed')
                    time.sleep(.01)

# SPDX-License-Identifier: Apache-2.0
"""Original stdlib Win32 bindings for the public OVStream SHM v2 contract.

Wire: https://github.com/NVIDIA-Omniverse/ovstream/blob/v0.5.7/docs/ARCHITECTURE.md
(protocolVersion 2, unchanged since v0.4.5). No vendor imports. Local mappings
are kernel-reference-counted; closing our handles never unlinks another
process's resources. Wake events are unused: the ordered control response and
producer ACK hold provide readiness. Producers from v0.5 no longer create a wake
event, so ``existing_resources`` reports it absent.
"""
from __future__ import annotations

import ctypes as C
import math
import re
import struct
import time

_DWORD = C.c_uint32
_BOOL = C.c_int32
_HANDLE = C.c_void_p
_INVALID_HANDLE = C.c_void_p(-1).value
_FILE_MAP_READ = 4
_API = None


def _api():
    global _API
    if _API is None:
        dll = C.WinDLL('kernel32', use_last_error=True)
        signatures = {
            'OpenFileMappingW': ([_DWORD, _BOOL, C.c_wchar_p], _HANDLE),
            'MapViewOfFile': ([_HANDLE, _DWORD, _DWORD, _DWORD, C.c_size_t], C.c_void_p),
            'UnmapViewOfFile': ([C.c_void_p], _BOOL),
            'CloseHandle': ([_HANDLE], _BOOL),
            'CreateFileW': ([C.c_wchar_p, _DWORD, _DWORD, C.c_void_p, _DWORD, _DWORD, _HANDLE], _HANDLE),
            'WaitNamedPipeW': ([C.c_wchar_p, _DWORD], _BOOL),
            'SetNamedPipeHandleState': ([_HANDLE, C.POINTER(_DWORD), C.c_void_p, C.c_void_p], _BOOL),
            'ReadFile': ([_HANDLE, C.c_void_p, _DWORD, C.POINTER(_DWORD), C.c_void_p], _BOOL),
            'WriteFile': ([_HANDLE, C.c_void_p, _DWORD, C.POINTER(_DWORD), C.c_void_p], _BOOL),
            'OpenProcess': ([_DWORD, _BOOL, _DWORD], _HANDLE),
            'OpenEventW': ([_DWORD, _BOOL, C.c_wchar_p], _HANDLE),
            'WaitForSingleObject': ([_HANDLE, _DWORD], _DWORD),
            'GlobalMemoryStatusEx': ([C.c_void_p], _BOOL),
        }
        for name, (args, result) in signatures.items():
            function = getattr(dll, name)
            function.argtypes, function.restype = args, result
        _API = dll
    return _API


def _error(operation):
    code = C.get_last_error()
    return OSError(code, f'{operation} failed (Win32 error {code})')


def _name(stream):
    if not isinstance(stream, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,63}', stream):
        raise ValueError('Invalid SHM stream name')
    return 'ovstream-' + stream


def process_exists(pid):
    """Read-only liveness probe; os.kill(pid, 0) is destructive on Windows."""
    if not isinstance(pid, int) or not 0 < pid <= 0xffffffff:
        raise ValueError('Invalid producer PID')
    api = _api()
    handle = api.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE only.
    if not handle:
        code = C.get_last_error()
        if code == 87:  # ERROR_INVALID_PARAMETER: PID is absent.
            return False
        if code == 5:  # Access denied is never proof of death.
            return True
        raise _error('OpenProcess')
    try:
        state = api.WaitForSingleObject(handle, 0)
        if state == 0:
            return False
        if state == 258:
            return True
        raise _error('WaitForSingleObject(process)')
    finally:
        api.CloseHandle(handle)


class _MemoryStatus(C.Structure):
    _fields_ = [('length', _DWORD), ('load', _DWORD)] + [
        (name, C.c_uint64) for name in ('total_phys', 'avail_phys', 'total_page',
            'avail_page', 'total_virtual', 'avail_virtual', 'avail_extended')]


def available_memory():
    """Conservative capacity for a pagefile-backed CPU mapping, not disk space."""
    status = _MemoryStatus()
    status.length = C.sizeof(status)
    if not _api().GlobalMemoryStatusEx(C.byref(status)):
        raise _error('GlobalMemoryStatusEx')
    return min(status.avail_phys, status.avail_page, status.avail_virtual)


def existing_resources(stream_name):
    """Probe exact existing kernel names without creating or connecting them."""
    base = _name(stream_name)
    api = _api()
    result = {}
    for name, opener, access in [('Local\\' + base, api.OpenFileMappingW, _FILE_MAP_READ),
            ('Local\\' + base + '-wake', api.OpenEventW, 0x00100000)]:
        handle = opener(access, False, name)
        if handle:
            api.CloseHandle(handle)
            result[name] = True
        elif C.get_last_error() == 2:
            result[name] = False
        else:
            raise _error('Open existing resource')
    pipe = '\\\\.\\pipe\\' + base
    if api.WaitNamedPipeW(pipe, 1):
        result[pipe] = True
    else:
        code = C.get_last_error()
        if code not in (2, 121, 231):
            raise _error('WaitNamedPipeW(resource probe)')
        result[pipe] = code != 2
    return result


class ReadOnlyMapping:
    """Open an EXISTING kernel mapping and expose bounded immutable byte slices."""
    def __init__(self, stream_name, *, max_bytes=1024**3):
        self._handle = self._address = None
        self._size = 0
        api = _api()
        try:
            self._handle = api.OpenFileMappingW(_FILE_MAP_READ, False, 'Local\\' + _name(stream_name))
            if not self._handle:
                raise _error('OpenFileMappingW')
            self._address = api.MapViewOfFile(self._handle, _FILE_MAP_READ, 0, 0, 128)
            if not self._address:
                raise _error('MapViewOfFile(header)')
            header = struct.unpack('<8I2Q3I', C.string_at(self._address, 60))
            if header[:3] != (0x4D53564F, 2, 1):
                raise ValueError('Unsupported SHM mapping header')
            slots, stride, offset = header[3], header[7], header[8]
            size = offset + slots * stride
            if not 2 <= slots <= 8 or offset < 128 or stride < 64 or not 128 <= size <= max_bytes:
                raise ValueError('SHM mapping exceeds supported bounds')
            if not api.UnmapViewOfFile(self._address):
                raise _error('UnmapViewOfFile(header)')
            self._address = None
            # Request the exact validated extent: an undersized existing mapping
            # must fail here; never trust an unchecked pointer from the header.
            self._address = api.MapViewOfFile(self._handle, _FILE_MAP_READ, 0, 0, size)
            if not self._address:
                raise _error('MapViewOfFile(region)')
            self._size = size
        except BaseException:
            self.close()
            raise

    def __len__(self):
        return self._size

    def __getitem__(self, item):
        if not self._address:
            raise ValueError('SHM mapping is closed')
        if not isinstance(item, slice) or item.step not in (None, 1):
            raise TypeError('SHM mapping requires contiguous slices')
        start, stop, _ = item.indices(self._size)
        return C.string_at(self._address + start, max(0, stop - start))

    def copy_into(self, destination, *, destination_offset, source_offset, count):
        """Copy checked mapped bytes into owned storage; never expose a pointer."""
        if not self._address:
            raise ValueError('SHM mapping is closed')
        if type(destination) is not bytearray:
            raise TypeError('SHM destination must be an owned bytearray')
        if any(type(value) is not int or value < 0
               for value in (destination_offset, source_offset, count)):
            raise ValueError('SHM copy offsets and count must be nonnegative integers')
        if source_offset + count > self._size or destination_offset + count > len(destination):
            raise ValueError('SHM copy exceeds source or destination bounds')
        if count:
            # The exported destination pins its size while ctypes releases the
            # GIL for memmove. Neither address survives this method.
            target = C.c_char.from_buffer(destination, destination_offset)
            C.memmove(C.addressof(target), self._address + source_offset, count)

    def close(self):
        api = _api()
        if self._address:
            if not api.UnmapViewOfFile(self._address):
                raise _error('UnmapViewOfFile')
            self._address = None
        if self._handle:
            if not api.CloseHandle(self._handle):
                raise _error('CloseHandle(mapping)')
            self._handle = None
        self._size = 0


class ControlPipe:
    """Tiny synchronous protocol with nonblocking Win32 I/O and one deadline.

    PIPE_NOWAIT is intentionally used for this bounded handshake, not for
    background/overlapped I/O. No queued operation can outlive a Python buffer.
    Microsoft documents immediate reads/writes and partial byte-pipe writes:
    https://learn.microsoft.com/windows/win32/ipc/named-pipe-type-read-and-wait-modes
    """
    def __init__(self, stream_name, timeout_seconds):
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError('Finite positive pipe timeout required')
        self._handle = None
        self._deadline = time.monotonic() + timeout_seconds
        path = '\\\\.\\pipe\\' + _name(stream_name)
        api = _api()
        try:
            while True:
                self._remaining()
                handle = api.CreateFileW(path, 0xc0000000, 0, None, 3, 0, None)
                if handle and handle != _INVALID_HANDLE:
                    self._handle = handle
                    break
                code = C.get_last_error()
                if code not in (2, 231):
                    raise _error('CreateFileW(pipe)')
                if code == 231:
                    api.WaitNamedPipeW(path, min(50, max(1, int(self._remaining()*1000))))
                else:
                    self._pause()
            mode = _DWORD(1)  # PIPE_READMODE_BYTE | PIPE_NOWAIT
            if not api.SetNamedPipeHandleState(self._handle, C.byref(mode), None, None):
                raise _error('SetNamedPipeHandleState')
        except BaseException:
            self.close()
            raise

    def _remaining(self):
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('OVStream named-pipe handshake timed out')
        return remaining

    def _pause(self):
        time.sleep(min(.005, self._remaining()))

    def sendall(self, data):
        if not self._handle:
            raise OSError('OVStream pipe is closed')
        offset = 0
        while offset < len(data):
            self._remaining()
            buffer = C.create_string_buffer(data[offset:])
            count = _DWORD()
            if not _api().WriteFile(self._handle, buffer, len(data)-offset, C.byref(count), None):
                raise _error('WriteFile(pipe)')
            offset += count.value
            if not count.value:
                self._pause()

    def recv(self, size):
        if not self._handle:
            raise OSError('OVStream pipe is closed')
        if not 1 <= size <= 256:
            raise ValueError('Invalid control read size')
        buffer = C.create_string_buffer(size)
        while True:
            self._remaining()
            count = _DWORD()
            if _api().ReadFile(self._handle, buffer, size, C.byref(count), None):
                return buffer.raw[:count.value]
            if C.get_last_error() != 232:  # ERROR_NO_DATA for empty nonblocking pipe.
                raise _error('ReadFile(pipe)')
            self._pause()

    def close(self):
        if self._handle:
            if not _api().CloseHandle(self._handle):
                raise _error('CloseHandle(pipe)')
            self._handle = None

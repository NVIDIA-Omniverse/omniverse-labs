# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise Unicode ownership and failure cleanup against simulated Win32 calls."""
import ctypes
from types import SimpleNamespace

import pytest

from ovd_ui import clipboard


def api(*, failure=None):
    calls = []
    memory = ctypes.create_string_buffer(4096)
    handle, window = 0x123456789, 0x234567890
    def operation(name, result):
        def call(*args):
            calls.append((name, args))
            return 0 if failure == name else result
        return call
    def owner(hwnd, process):
        ctypes.cast(process, ctypes.POINTER(ctypes.c_ulong))[0] = 42
        return 7
    user = SimpleNamespace(**{name: operation(name, value) for name, value in {
        'GetActiveWindow': window, 'IsWindow': 1, 'GetForegroundWindow': window,
        'OpenClipboard': 1, 'EmptyClipboard': 1, 'SetClipboardData': handle,
        'GetClipboardData': handle, 'CloseClipboard': 1}.items()})
    user.GetWindowThreadProcessId = owner
    kernel = SimpleNamespace(**{name: operation(name, value) for name, value in {
        'GetCurrentProcessId': 42, 'GetCurrentThreadId': 7, 'GlobalAlloc': handle,
        'GlobalLock': ctypes.addressof(memory), 'GlobalUnlock': 1, 'GlobalFree': 0}.items()})
    return user, kernel, calls, memory


@pytest.mark.parametrize('text', ['', '物理 μ 🙂\nline\t '])
def test_unicode_publication_uses_owned_foreground_window_and_transfers_memory(text):
    user, kernel, calls, memory = api()
    backend = clipboard._Win32Clipboard(user, kernel, lambda: (640, 480))
    backend.copy(text.encode('utf-8'))
    names = [name for name, _ in calls]
    assert ('OpenClipboard', (0x234567890,)) in calls
    assert names.index('GlobalAlloc') < names.index('EmptyClipboard') < names.index('SetClipboardData')
    assert names[-1] == 'CloseClipboard' and 'GlobalFree' not in names
    expected = text.encode('utf-16-le') + b'\0\0'
    assert memory.raw[:len(expected)] == expected
    assert user.SetClipboardData.argtypes[1] is ctypes.c_void_p
    assert kernel.GlobalAlloc.restype is ctypes.c_void_p


@pytest.mark.parametrize('failure', ['GetActiveWindow', 'IsWindow', 'GetForegroundWindow', 'GlobalAlloc', 'OpenClipboard', 'EmptyClipboard', 'SetClipboardData', 'GetClipboardData', 'GlobalLock'])
def test_failures_report_error_and_release_only_owned_resources(failure):
    user, kernel, calls, memory = api(failure=failure)
    backend = clipboard._Win32Clipboard(user, kernel, lambda: (640, 480))
    with pytest.raises(clipboard.ClipboardError):
        backend.copy(b'value')
    names = [name for name, _ in calls]
    if failure in ('GetActiveWindow', 'IsWindow', 'GetForegroundWindow'):
        assert 'EmptyClipboard' not in names and 'OpenClipboard' not in names
    if 'OpenClipboard' in names and failure != 'OpenClipboard':
        assert names[-1] == 'CloseClipboard'
    if failure in ('EmptyClipboard', 'SetClipboardData', 'OpenClipboard', 'GlobalLock'):
        assert 'GlobalFree' in names
    if failure == 'GetClipboardData':
        assert 'GlobalFree' not in names


@pytest.mark.parametrize('process,thread', [(99, 7), (42, 99)])
def test_foreign_process_or_thread_is_rejected_before_clipboard_mutation(process, thread):
    user, kernel, calls, memory = api()
    def owner(hwnd, result):
        ctypes.cast(result, ctypes.POINTER(ctypes.c_ulong))[0] = process
        return thread
    user.GetWindowThreadProcessId = owner
    backend = clipboard._Win32Clipboard(user, kernel, lambda: (640, 480))
    with pytest.raises(clipboard.ClipboardError):
        backend.copy(b'value')
    assert 'OpenClipboard' not in [name for name, _ in calls]


def test_uninitialized_ui_rejected_before_any_os_call():
    user, kernel, calls, memory = api()
    backend = clipboard._Win32Clipboard(user, kernel, lambda: (0, 0))
    with pytest.raises(clipboard.ClipboardError, match='active OVUI desktop'):
        backend.copy(b'value')
    assert not calls

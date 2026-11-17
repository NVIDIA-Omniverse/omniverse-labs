# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit opt-in for hidden Windows layout tests; never manipulate other windows."""
import atexit
import ctypes
import os
import subprocess
import sys
import tempfile


HIDDEN_WINDOWS = sys.platform == 'win32' and os.environ.get('OVD_UI_HIDDEN_GUI_TESTS') == '1'
GUI_AVAILABLE = bool(os.environ.get('DISPLAY')) or HIDDEN_WINDOWS
_directory = None


def select_test_display():
    """Honor the harness's X11 DISPLAY instead of an inherited WSLg session.

    Otherwise GLFW can choose Wayland even under xvfb-run, letting concurrent
    test windows steal desktop focus and clear injected keys/buttons.
    """
    if sys.platform == 'linux' and os.environ.get('DISPLAY'):
        os.environ['XDG_SESSION_TYPE'] = 'x11'
        os.environ.pop('WAYLAND_DISPLAY', None)


def run_gui_process(*args, **kwargs):
    """Retain a hung child's progress and faulthandler output in CI failures."""
    try:
        return subprocess.run(*args, **kwargs)
    except subprocess.TimeoutExpired as error:
        def text(value):
            return value.decode('utf-8', errors='replace') if isinstance(value, bytes) else value or ''
        raise AssertionError(f'GUI timeout after {error.timeout}s\nstdout:\n{text(error.stdout)}'
                             f'\nstderr:\n{text(error.stderr)}') from error


def isolate_gui_output():
    global _directory
    select_test_display()
    # ImGui writes imgui.ini to CWD, independently of APPDATA/XDG settings.
    _directory = tempfile.TemporaryDirectory(prefix='ovui-gui-output-')
    previous = os.getcwd()
    os.chdir(_directory.name)
    def cleanup():
        os.chdir(previous)
        _directory.cleanup()
    atexit.register(cleanup)


def hide_owned_window():
    if not HIDDEN_WINDOWS:
        return
    user = ctypes.WinDLL('user32', use_last_error=True)
    user.GetWindowThreadProcessId.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    user.ShowWindow.argtypes = [ctypes.c_void_p, ctypes.c_int]
    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def hide(hwnd, _):
        process = ctypes.c_ulong()
        user.GetWindowThreadProcessId(hwnd, ctypes.byref(process))
        if process.value == os.getpid():
            user.ShowWindow(hwnd, 0)
        return True
    user.EnumWindows(hide, 0)


def size_hidden_client(width, height, *, title='PVD on OVUI'):
    """Give only this hidden test window its requested logical client size.

    Windows CI's small desktop clamps GLFW's initial window size. Use the
    same no-activation resize as the Display regression, not desktop settings.
    """
    if not HIDDEN_WINDOWS:
        return
    from ctypes import wintypes
    import omni.ui as ui
    user = ctypes.WinDLL('user32', use_last_error=True)
    user.GetWindowThreadProcessId.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    user.GetWindowTextW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int]
    user.GetClientRect.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.RECT)]
    user.GetWindowRect.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.RECT)]
    user.IsWindowVisible.argtypes = [ctypes.c_void_p]
    user.SetWindowPos.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                 ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
    owned = []
    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def find(hwnd, _):
        process = ctypes.c_ulong()
        user.GetWindowThreadProcessId(hwnd, ctypes.byref(process))
        if process.value == os.getpid():
            text = ctypes.create_unicode_buffer(256)
            user.GetWindowTextW(hwnd, text, len(text))
            if text.value == title:
                owned.append(hwnd)
        return True
    user.EnumWindows(find, 0)
    assert len(owned) == 1, ('Expected one owned GUI window', title, owned)
    hwnd = owned[0]
    assert not user.IsWindowVisible(hwnd), 'Resize is restricted to hidden test windows'
    dpi = ui.Workspace.get_dpi_scale()
    target = (round(width * dpi), round(height * dpi))
    outer, client = wintypes.RECT(), wintypes.RECT()
    assert user.GetClientRect(hwnd, ctypes.byref(client))
    if (client.right, client.bottom) != target:
        assert user.GetWindowRect(hwnd, ctypes.byref(outer))
        width = target[0] + (outer.right - outer.left) - client.right
        height = target[1] + (outer.bottom - outer.top) - client.bottom
        # NOSENDCHANGING | NOACTIVATE | NOZORDER | NOMOVE.
        assert user.SetWindowPos(hwnd, None, 0, 0, width, height, 0x0400 | 0x0010 | 0x0004 | 0x0002)
        assert user.GetClientRect(hwnd, ctypes.byref(client))
    assert (client.right, client.bottom) == target, (target, client.right, client.bottom)

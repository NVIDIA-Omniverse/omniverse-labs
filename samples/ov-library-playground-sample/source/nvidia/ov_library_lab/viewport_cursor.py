# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Open/closed grab cursors, scoped to the Windows desktop viewport.

ovui does not yet expose custom native cursors. Use documented Win32 cursor and
window-subclass APIs, without changing system cursors, mouse capture or position.
The widget's explicit hover keeps floating controls on their normal UI cursor.
Headless captures and unsupported platforms retain ovui's own cursor.
"""
import ctypes as c
from ctypes import wintypes as w
import os
import struct
import warnings

from .cursor_art import grab_cursor


def create_cursor():
    if os.name != 'nt' or os.environ.get('OMNIUI_HEADLESS') == '1':
        return None
    try:
        return _WindowsCursor()
    except (OSError, RuntimeError) as exc:
        warnings.warn(f'Navigation cursor unavailable: {exc}', RuntimeWarning)
        return None


class _WindowsCursor:
    def __init__(self):
        self.user = c.WinDLL('user32', use_last_error=True)
        self.common = c.WinDLL('comctl32', use_last_error=True)
        self.handles = {}
        self.active = self.dragging = False
        self.hwnd = None
        self.callback = None
        self.selected = None
        self._bind(self.user, 'GetCursorPos', [c.POINTER(w.POINT)], w.BOOL)
        self._bind(self.user, 'WindowFromPoint', [w.POINT], w.HWND)
        self._bind(self.user, 'GetForegroundWindow', [], w.HWND)
        self._bind(self.user, 'GetCursor', [], w.HANDLE)
        self._bind(self.user, 'SetCursor', [w.HANDLE], w.HANDLE)
        self._bind(self.user, 'GetDpiForWindow', [w.HWND], w.UINT)
        self._bind(self.user, 'LoadCursorW', [w.HINSTANCE, c.c_void_p], w.HANDLE)
        self._bind(self.user, 'DestroyCursor', [w.HANDLE], w.BOOL)
        self._bind(self.user, 'CreateIconFromResourceEx',
                   [c.c_void_p, w.DWORD, w.BOOL, w.DWORD, c.c_int, c.c_int, w.UINT], w.HANDLE)
        self.arrow = self.user.LoadCursorW(None, c.c_void_p(32512))
        enum_type = c.WINFUNCTYPE(w.BOOL, w.HWND, w.LPARAM)
        self._bind(self.user, 'EnumThreadWindows', [w.DWORD, enum_type, w.LPARAM], w.BOOL)
        self._bind(self.user, 'GetClassNameW', [w.HWND, w.LPWSTR, c.c_int], c.c_int)
        @enum_type
        def find_window(hwnd, _):
            name = c.create_unicode_buffer(128)
            self.user.GetClassNameW(hwnd, name, len(name))
            if name.value == 'GLFW30':
                self.hwnd = hwnd
                return False
            return True
        self.user.EnumThreadWindows(c.WinDLL('kernel32').GetCurrentThreadId(), find_window, 0)
        if not self.hwnd:
            raise RuntimeError('No standalone desktop window on this UI thread')
        # Common-controls subclassing preserves the backend's WndProc chain.
        callback_type = c.WINFUNCTYPE(c.c_ssize_t, w.HWND, w.UINT, w.WPARAM,
                                     w.LPARAM, c.c_size_t, c.c_size_t)
        self._bind(self.common, 'SetWindowSubclass',
                   [w.HWND, callback_type, c.c_size_t, c.c_size_t], w.BOOL)
        self._bind(self.common, 'RemoveWindowSubclass',
                   [w.HWND, callback_type, c.c_size_t], w.BOOL)
        self._bind(self.common, 'DefSubclassProc',
                   [w.HWND, w.UINT, w.WPARAM, w.LPARAM], c.c_ssize_t)
        self.callback = callback_type(self._message)
        if not self.common.SetWindowSubclass(self.hwnd, self.callback, id(self), 0):
            raise c.WinError(c.get_last_error())

    @staticmethod
    def _bind(library, name, args, result):
        fn = getattr(library, name)
        fn.argtypes, fn.restype = args, result

    def _owns_pointer(self):
        point = w.POINT()
        return (self.hwnd and self.user.GetForegroundWindow() == self.hwnd
                and self.user.GetCursorPos(c.byref(point))
                and self.user.WindowFromPoint(point) == self.hwnd)

    def _cursor(self):
        scale = max(1., self.user.GetDpiForWindow(self.hwnd)/96.)
        key = (scale, self.dragging)
        if key not in self.handles:
            size = round(24*scale)
            bgra, hotspot = grab_cursor(size, self.dragging)
            mask = bytes(((size+31)//32)*4*size)
            dib = struct.pack('<IIIHHIIIIII', 40, size, size*2, 1, 32, 0,
                              size*size*4, 0, 0, 0, 0)
            data = struct.pack('<HH', *hotspot)+dib+bgra[::-1].tobytes()+mask
            buffer = c.create_string_buffer(data)
            handle = self.user.CreateIconFromResourceEx(buffer, len(data), False,
                                                       0x00030000, size, size, 0)
            if not handle:
                raise c.WinError(c.get_last_error())
            self.handles[key] = handle
        return self.handles[key]

    def _message(self, hwnd, message, wp, lp, *_):
        # WM_SETCURSOR / HTCLIENT: retain the navigation cursor during motion.
        if (message == 0x20 and lp & 0xffff == 1 and self.active
                and self.selected and self._owns_pointer()):
            self.user.SetCursor(self.selected)
            return 1
        if message == 0x82:  # WM_NCDESTROY: remove before the native window dies.
            self.common.RemoveWindowSubclass(hwnd, self.callback, id(self))
            self.hwnd = None
        return self.common.DefSubclassProc(hwnd, message, wp, lp)

    def update(self, hovered, dragging):
        if not self.hwnd:
            return
        self.active, self.dragging = bool(hovered), bool(dragging)
        if self.active and self._owns_pointer():
            try:
                self.selected = self._cursor()
            except OSError as exc:
                # Cursor decoration must never interrupt navigation/rendering.
                self.close()
                warnings.warn(f'Navigation cursor unavailable: {exc}', RuntimeWarning)
                return
            self.user.SetCursor(self.selected)
        elif self.user.GetCursor() in self.handles.values():
            self.user.SetCursor(self.arrow)

    def close(self):
        self.active = False
        if self.hwnd:
            self.common.RemoveWindowSubclass(self.hwnd, self.callback, id(self))
            self.hwnd = None
        if self.user.GetCursor() in self.handles.values():
            self.user.SetCursor(self.arrow)
        for handle in self.handles.values():
            self.user.DestroyCursor(handle)
        self.handles.clear()

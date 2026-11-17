# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Copy property text to the desktop clipboard through OVUI's public GLFW API.

Call ``copy_text(str)`` on the UI main thread after ``ui.init``. It returns None
on success and raises ClipboardError when the running backend cannot publish a
desktop clipboard. Invalid text raises TypeError/ValueError before any OS call.
No test clipboard callbacks, second GLFW runtime, window/context creation or
background clipboard process is involved.

GLFW copies the UTF-8 bytes before returning. On X11 the active OVUI event loop
serves selection requests, so callers must not block that loop while waiting for
another application to paste. Another application's Copy replaces ownership;
we never restore old contents. After normal shutdown, persistence depends on a
desktop clipboard manager. A bare X server loses the selection when GLFW exits.
Windows publishes immediate Unicode text through the current UI thread's
active foreground window. Headless/null platforms fail explicitly.

The private ``omni.ui._ui._set_clipboard_text`` function is intentionally unused:
the pinned binding replaces ImGui's platform callbacks with an in-memory test
clipboard, which is not the system clipboard.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import sys
import threading


class ClipboardError(RuntimeError):
    """The active OVUI desktop could not copy text to the operating system."""


# Public GLFW 3.4 platform constants, not OVUI private state.
_DESKTOP_PLATFORMS = frozenset((0x00060001, 0x00060002, 0x00060003, 0x00060004))
_NOT_INITIALIZED = 0x00010001
_backend = None


class _Win32Clipboard:
    """Immediate CF_UNICODETEXT publication; transferred memory belongs to Windows."""
    def __init__(self, user32, kernel32, window_size):
        self.user, self.kernel, self.window_size = user32, kernel32, window_size
        handle, word, size = ctypes.c_void_p, ctypes.c_ulong, ctypes.c_size_t
        signatures = {
            'GetActiveWindow': (handle, []), 'IsWindow': (ctypes.c_int, [handle]),
            'GetForegroundWindow': (handle, []),
            'GetWindowThreadProcessId': (word, [handle, ctypes.POINTER(word)]),
            'OpenClipboard': (ctypes.c_int, [handle]), 'EmptyClipboard': (ctypes.c_int, []),
            'SetClipboardData': (handle, [ctypes.c_uint, handle]),
            'GetClipboardData': (handle, [ctypes.c_uint]), 'CloseClipboard': (ctypes.c_int, [])}
        for name, (result, arguments) in signatures.items():
            function = getattr(user32, name)
            function.restype, function.argtypes = result, arguments
        signatures = {'GetCurrentProcessId': (word, []), 'GetCurrentThreadId': (word, []),
            'GlobalAlloc': (handle, [ctypes.c_uint, size]), 'GlobalLock': (handle, [handle]),
            'GlobalUnlock': (ctypes.c_int, [handle]), 'GlobalFree': (handle, [handle])}
        for name, (result, arguments) in signatures.items():
            function = getattr(kernel32, name)
            function.restype, function.argtypes = result, arguments

    def copy(self, payload):
        text = payload.decode('utf-8', errors='strict')
        encoded = text.encode('utf-16-le', errors='strict') + b'\0\0'
        try:
            width, height = self.window_size()
        except Exception as error:
            raise ClipboardError('Copy requires an active OVUI desktop window.') from error
        if width <= 0 or height <= 0:
            raise ClipboardError('Copy requires an active OVUI desktop window.')
        user, kernel = self.user, self.kernel
        window = user.GetActiveWindow()
        process = ctypes.c_ulong()
        if (not window or not user.IsWindow(window)
                or user.GetWindowThreadProcessId(window, ctypes.byref(process)) != kernel.GetCurrentThreadId()
                or process.value != kernel.GetCurrentProcessId()
                or user.GetForegroundWindow() != window):
            raise ClipboardError('Copy requires this UI thread\'s active foreground window.')
        allocation = kernel.GlobalAlloc(0x0002, len(encoded))  # GMEM_MOVEABLE
        if not allocation:
            raise ClipboardError('The operating system could not allocate clipboard text.')
        opened = False
        try:
            pointer = kernel.GlobalLock(allocation)
            if not pointer:
                raise ClipboardError('The operating system could not lock clipboard text.')
            try:
                ctypes.memmove(pointer, encoded, len(encoded))
            finally:
                kernel.GlobalUnlock(allocation)
            if not user.OpenClipboard(window):
                raise ClipboardError('The desktop clipboard is unavailable.')
            opened = True
            if not user.EmptyClipboard() or not user.SetClipboardData(13, allocation):
                raise ClipboardError('The operating system could not publish clipboard text.')
            allocation = None  # Ownership transferred only after SetClipboardData succeeds.
            published = user.GetClipboardData(13)
            pointer = kernel.GlobalLock(published) if published else None
            if not pointer:
                raise ClipboardError('The published clipboard text could not be verified.')
            try:
                if ctypes.string_at(pointer, len(encoded)) != encoded:
                    raise ClipboardError('The desktop clipboard did not retain the copied text.')
            finally:
                kernel.GlobalUnlock(published)
        finally:
            if allocation:
                kernel.GlobalFree(allocation)
            if opened:
                user.CloseClipboard()


class _GlfwClipboard:
    def __init__(self, library):
        self.library = library  # Keep the loaded OVUI library alive with its C functions.
        try:
            self.get_error = library.glfwGetError
            self.get_error.argtypes = [ctypes.POINTER(ctypes.c_char_p)]
            self.get_error.restype = ctypes.c_int
            self.get_platform = library.glfwGetPlatform
            self.get_platform.argtypes = []
            self.get_platform.restype = ctypes.c_int
            self.set_text = library.glfwSetClipboardString
            self.set_text.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            self.set_text.restype = None
            self.get_text = library.glfwGetClipboardString
            self.get_text.argtypes = [ctypes.c_void_p]
            self.get_text.restype = ctypes.c_char_p
        except AttributeError as error:
            raise ClipboardError("This OVUI runtime does not export the public desktop clipboard API.") from error

    def _check_error(self):
        description = ctypes.c_char_p()
        code = self.get_error(ctypes.byref(description))
        if code == _NOT_INITIALIZED:
            raise ClipboardError("Copy requires an active OVUI desktop window.")
        if code:
            # Do not echo clipboard text or arbitrary native diagnostics into UI/errors.
            raise ClipboardError(f"The operating system could not copy text (GLFW error 0x{code:08x}).")

    def copy(self, payload):
        self.get_error(None)  # Attribute subsequent errors to this operation.
        platform = self.get_platform()
        self._check_error()
        if platform not in _DESKTOP_PLATFORMS:
            raise ClipboardError("Copy is unavailable in this headless OVUI backend.")
        # GLFW documents the deprecated window argument as accepting NULL. No
        # OpenGL context handle is needed (and it would be wrong for Vulkan).
        self.set_text(None, payload)
        self._check_error()
        accepted = self.get_text(None)
        self._check_error()
        if accepted != payload:
            raise ClipboardError("The desktop clipboard did not retain the copied text.")


def _load_backend():
    try:
        import omni.ui as ui
    except (ImportError, OSError) as error:
        raise ClipboardError("The OVUI desktop runtime is unavailable.") from error
    if sys.platform == "linux":
        directory = Path(ui.__file__).resolve().parent
        # Wheels can contain distinct files for the unversioned name and SONAME.
        # Only open an already loaded one: dlopen on the other copy would create
        # another GLFW state even though the code and package directory match.
        candidates = [directory / "libomniui_standalone.so.3", directory / "libomniui_standalone.so"]
        candidates.extend(sorted(directory.glob("libomniui_standalone.so.*")))
        library = None
        for path in dict.fromkeys(candidates):
            if path.is_file():
                try:
                    library = ctypes.CDLL(str(path), mode=os.RTLD_NOW | os.RTLD_NOLOAD)
                    break
                except OSError:
                    continue
        if library is None:
            raise ClipboardError("The active OVUI desktop clipboard library could not be located.")
        return _GlfwClipboard(library)
    elif sys.platform == "win32":
        from omni.ui import standalone
        return _Win32Clipboard(ctypes.WinDLL('user32', use_last_error=True),
            ctypes.WinDLL('kernel32', use_last_error=True), standalone.get_window_size)
    else:
        raise ClipboardError("Desktop clipboard support is unavailable on this platform.")


def copy_text(text: str) -> None:
    """Publish exact Unicode text, or raise without pretending an internal copy succeeded."""
    if not isinstance(text, str):
        raise TypeError("Clipboard text must be a string.")
    if "\0" in text:
        raise ValueError("Clipboard text cannot contain a NUL character.")
    try:
        payload = text.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise ValueError("Clipboard text must contain valid Unicode characters.") from error
    if threading.current_thread() is not threading.main_thread():
        raise ClipboardError("Copy must run on the UI main thread.")
    global _backend
    if _backend is None:
        _backend = _load_backend()
    _backend.copy(payload)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scope Windows timer precision to the live app, restoring it on shutdown."""
import os
import asyncio
import sys
import time
from contextlib import contextmanager


def configure_desktop_presentation():
    """Remove the OpenGL swap wait from ovui's cooperative input/render loop.

    The application's existing UI pacing remains active. This is per-context, not a driver
    preference; closing the app releases it. Headless/Vulkan contexts are left
    alone. Keeping vsync here blocks both UI polling and CUDA texture handoff.
    """
    if os.name != 'nt':return False
    import ctypes
    gl = ctypes.WinDLL('opengl32')
    gl.wglGetCurrentContext.argtypes = []
    gl.wglGetCurrentContext.restype = ctypes.c_void_p
    if not gl.wglGetCurrentContext():return False
    gl.wglGetProcAddress.argtypes = [ctypes.c_char_p]
    gl.wglGetProcAddress.restype = ctypes.c_void_p
    address = gl.wglGetProcAddress(b'wglSwapIntervalEXT')
    if address in (None,0,1,2,3,ctypes.c_void_p(-1).value):return False
    swap_interval = ctypes.WINFUNCTYPE(ctypes.c_int,ctypes.c_int)(address)
    return bool(swap_interval(0))


def high_resolution_clock():
    return (os.name == 'nt' and time.get_clock_info('monotonic').resolution > .001
            and os.environ.get('OVLAB_LEGACY_CLOCK') != '1')


def run_event_loop(coroutine):
    """Use QPC for this app's asyncio deadlines on coarse-clock Python builds."""
    def create_loop():
        loop = asyncio.new_event_loop()
        if high_resolution_clock():
            # CPython 3.12's Windows loop otherwise treats deadlines within
            # 15.625 ms as ready, defeating short UI frame caps. This loop is
            # owned and closed by Runner; no system/event-loop policy changes.
            loop.time = time.perf_counter
            loop._clock_resolution = time.get_clock_info('perf_counter').resolution
        return loop
    with asyncio.Runner(loop_factory=create_loop) as runner:
        return runner.run(coroutine)


@contextmanager
def precise_timers():
    # Windows' default wait granularity can turn a 16.7 ms UI tick into 31 ms.
    # This pairs the documented multimedia timer request with its release;
    # it neither spins nor changes persistent system configuration.
    timer = None
    standalone = sys.modules.get('omni.ui.standalone')
    old_clock = None
    if high_resolution_clock() and standalone is not None:
        # The pinned ovui loop also reads time.monotonic for dt and its cap.
        # Replace only that module's clock reference, never Python's time module
        # or the installed SDK. Restore it after the app's runner has closed.
        old_clock = standalone.time
        class Clock:
            monotonic = staticmethod(time.perf_counter)
            def __getattr__(self, name):
                return getattr(old_clock, name)
        standalone.time = Clock()
    if os.name == 'nt':
        import ctypes
        timer = ctypes.WinDLL('winmm')
        if timer.timeBeginPeriod(1) != 0:
            timer = None
    try:
        yield
    finally:
        if old_clock is not None:
            standalone.time = old_clock
        if timer:
            timer.timeEndPeriod(1)

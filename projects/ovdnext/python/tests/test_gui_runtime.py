# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GUI test isolation and diagnostics from real subprocesses."""
import sys

import pytest
import gui_runtime


@pytest.mark.parametrize("text", [False, True])
def test_gui_timeout_retains_child_stdout_and_stderr(text):
    code = (
        "import sys, time; print('last GUI progress', flush=True); "
        "print('child stack diagnostic', file=sys.stderr, flush=True); time.sleep(60)"
    )
    with pytest.raises(AssertionError) as caught:
        gui_runtime.run_gui_process([sys.executable, "-B", "-c", code],
                                    capture_output=True, text=text, timeout=2)
    message = str(caught.value)
    assert "last GUI progress" in message and "child stack diagnostic" in message
    assert "2" in message and "timeout" in message.lower()


def test_linux_display_wins_over_inherited_wayland(monkeypatch):
    monkeypatch.setattr(gui_runtime.sys, 'platform', 'linux')
    monkeypatch.setenv('DISPLAY', ':123')
    monkeypatch.setenv('WAYLAND_DISPLAY', 'wayland-0')
    monkeypatch.setenv('XDG_SESSION_TYPE', 'wayland')
    gui_runtime.select_test_display()
    assert gui_runtime.os.environ['DISPLAY'] == ':123'
    assert gui_runtime.os.environ['XDG_SESSION_TYPE'] == 'x11'
    assert 'WAYLAND_DISPLAY' not in gui_runtime.os.environ


@pytest.mark.parametrize('platform,display', [('win32', ':123'), ('linux', '')])
def test_without_a_linux_display_backend_is_unchanged(monkeypatch, platform, display):
    monkeypatch.setattr(gui_runtime.sys, 'platform', platform)
    monkeypatch.setenv('DISPLAY', display)
    monkeypatch.setenv('WAYLAND_DISPLAY', 'wayland-0')
    monkeypatch.setenv('XDG_SESSION_TYPE', 'wayland')
    gui_runtime.select_test_display()
    assert gui_runtime.os.environ['WAYLAND_DISPLAY'] == 'wayland-0'
    assert gui_runtime.os.environ['XDG_SESSION_TYPE'] == 'wayland'


def test_exact_sizing_never_touches_windows_without_hidden_opt_in(monkeypatch):
    monkeypatch.setattr(gui_runtime, 'HIDDEN_WINDOWS', False)
    def forbidden(*args, **kwargs):
        pytest.fail('No window API may be used without hidden-test opt-in')
    monkeypatch.setattr(gui_runtime.ctypes, 'WinDLL', forbidden, raising=False)
    gui_runtime.size_hidden_client(1280, 800)

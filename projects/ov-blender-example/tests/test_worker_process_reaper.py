# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the exit-time worker-process reaper backstop."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import worker_process_reaper as wpr


def test_install_registers_once_and_uninstall_removes(monkeypatch: pytest.MonkeyPatch) -> None:
    registered: list = []
    unregistered: list = []
    monkeypatch.setattr(wpr, "_installed", False)
    monkeypatch.setattr(wpr.atexit, "register", lambda fn: registered.append(fn))
    monkeypatch.setattr(wpr.atexit, "unregister", lambda fn: unregistered.append(fn))

    wpr.install()
    wpr.install()  # idempotent — must not double-register
    assert registered == [wpr.reap]

    wpr.uninstall()
    wpr.uninstall()  # idempotent
    assert unregistered == [wpr.reap]


def test_pid_seam_register_and_forget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wpr, "_explicit_pids", set())
    wpr.forget_worker_pid(4242)  # forget-before-register is a no-op
    wpr.register_worker_pid(4242)
    assert 4242 in wpr._explicit_pids
    wpr.register_worker_pid("not-a-pid")  # invalid inputs ignored
    wpr.register_worker_pid(0)
    wpr.register_worker_pid(-1)
    assert wpr._explicit_pids == {4242}
    wpr.forget_worker_pid(4242)
    assert wpr._explicit_pids == set()


def test_basename_and_pid_helpers() -> None:
    assert wpr._basename_lower(r"C:\bundle\bin\Ovphysx-Bridge-Server.exe") == "ovphysx-bridge-server.exe"
    assert wpr._basename_lower('  "ovrtx-bridge-server"  ') == "ovrtx-bridge-server"
    assert wpr._basename_lower("") == ""
    assert wpr._coerce_pid("7") == 7
    assert wpr._coerce_pid(0) is None
    assert wpr._coerce_pid(-3) is None
    assert wpr._coerce_pid("x") is None


def test_reap_with_no_match_is_a_safe_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    # No explicit PIDs and a basename that cannot match any real process.
    monkeypatch.setattr(wpr, "_explicit_pids", set())
    monkeypatch.setattr(wpr, "_worker_basenames", {"ovrtx-nonexistent-worker-xyz.exe"})
    assert wpr.reap() == 0


def test_reap_never_targets_own_pid(monkeypatch: pytest.MonkeyPatch) -> None:
    # Even if our own PID is registered, reap must skip it (never self-terminate).
    killed_pids: list = []
    monkeypatch.setattr(wpr, "_explicit_pids", {os.getpid()})
    monkeypatch.setattr(wpr, "_worker_basenames", set())
    monkeypatch.setattr(wpr, "_terminate_pid", lambda pid: killed_pids.append(pid) or True)
    wpr.reap()
    assert os.getpid() not in killed_pids


@pytest.mark.skipif(os.name != "nt", reason="Windows toolhelp enumeration + TerminateProcess")
def test_windows_reaps_named_child_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wpr, "_explicit_pids", set())
    monkeypatch.setattr(wpr, "_worker_basenames", {"ping.exe"})
    child = subprocess.Popen(
        ["ping", "-n", "60", "127.0.0.1"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        # The child is a direct child of this process, so enumeration finds it.
        for _ in range(20):
            if child.pid in wpr._windows_child_worker_pids(frozenset({"ping.exe"}), os.getpid()):
                break
            time.sleep(0.05)
        assert child.pid in wpr._windows_child_worker_pids(frozenset({"ping.exe"}), os.getpid())
        assert wpr.reap() >= 1
        for _ in range(40):
            if child.poll() is not None:
                break
            time.sleep(0.05)
        assert child.poll() is not None  # terminated
    finally:
        if child.poll() is None:
            child.kill()

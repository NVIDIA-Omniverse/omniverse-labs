# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import os
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import session_lifecycle  # noqa: E402


def test_derive_status_truth_table_and_retry_budget() -> None:
    def status(**overrides: object) -> str:
        inputs = {
            "engine_active": True,
            "ready": False,
            "busy": False,
            "phase": "",
            "failure_count": 0,
            "worker_exit_code": None,
        }
        inputs.update(overrides)
        return session_lifecycle.derive_status(**inputs)["status"]

    assert status(ready=True) == session_lifecycle.STATUS_LIVE
    assert status(busy=True, phase=session_lifecycle.PHASE_LOADING) == session_lifecycle.STATUS_LOADING
    assert status(busy=True, phase=session_lifecycle.PHASE_RESYNCING) == session_lifecycle.STATUS_RESYNCING
    assert status(busy=True, worker_exit_code=1) == session_lifecycle.STATUS_STARTING
    assert status(worker_exit_code=1) == session_lifecycle.STATUS_CRASHED
    assert status(engine_active=False) == session_lifecycle.STATUS_STOPPED
    assert status(failure_count=session_lifecycle.MAX_AUTO_RETRIES) == session_lifecycle.STATUS_FAILED
    assert status(cleanup_in_progress=True) == session_lifecycle.STATUS_CLEANUP_IN_PROGRESS

    assert session_lifecycle.should_auto_retry(0) is True
    assert session_lifecycle.should_auto_retry(session_lifecycle.MAX_AUTO_RETRIES - 1) is True
    assert session_lifecycle.should_auto_retry(session_lifecycle.MAX_AUTO_RETRIES) is False


def test_prepare_logs_defaults_to_stdout_with_no_files(tmp_path: Path) -> None:
    """Default log routing is stdout inheritance: no env defaults are
    written and no log files or directories are created — worker output
    lands in Blender's console; lifecycle messages go to the Info panel."""

    env: dict[str, str] = {}
    result = session_lifecycle.prepare_logs(env)

    assert result["status"] == "stdout"
    assert result["worker_log"] == ""
    assert result["renderer_log"] == ""
    assert result["log_dir"] == ""
    assert env == {}


def test_prepare_logs_preserves_explicit_file_overrides(tmp_path: Path) -> None:
    overrides = {
        session_lifecycle.WORKER_LOG_ENV: str(tmp_path / "custom-worker.log"),
        session_lifecycle.RENDERER_LOG_ENV: str(tmp_path / "custom-renderer.log"),
    }
    preserved = session_lifecycle.prepare_logs(overrides)

    assert preserved["status"] == "file"
    assert preserved["worker_log"] == str(tmp_path / "custom-worker.log")
    assert preserved["renderer_log"] == str(tmp_path / "custom-renderer.log")
    assert preserved["log_dir"] == str(tmp_path)
    # The set worker-log value is mirrored across both worker-log keys so
    # the native client and worker agree.
    assert overrides[session_lifecycle.NATIVE_WORKER_LOG_ENV] == str(tmp_path / "custom-worker.log")


def test_crash_marker_round_trip_and_stale_detection(tmp_path: Path) -> None:
    own_marker = session_lifecycle.write_crash_marker(
        phase=session_lifecycle.PHASE_LOADING,
        scene_name="scene.blend",
        pid=123,
        directory=tmp_path,
        now=lambda: 10.0,
    )

    assert own_marker["status"] == "written"
    assert session_lifecycle.read_stale_crash_marker(current_pid=123, directory=tmp_path) == {}

    stale = session_lifecycle.read_stale_crash_marker(
        current_pid=456,
        directory=tmp_path,
        pid_running=lambda _pid: False,
    )

    assert stale["status"] == "stale"
    assert stale["scene"] == "scene.blend"
    assert stale["phase"] == session_lifecycle.PHASE_LOADING

    assert session_lifecycle.clear_crash_marker_if_mine(current_pid=456, directory=tmp_path)["status"] == "not_mine"
    assert session_lifecycle.clear_crash_marker_if_mine(current_pid=123, directory=tmp_path)["status"] == "cleared"
    assert session_lifecycle.read_crash_marker(directory=tmp_path) == {}


def test_stale_crash_marker_treats_permission_error_as_live(monkeypatch, tmp_path: Path) -> None:
    session_lifecycle.write_crash_marker(
        phase=session_lifecycle.PHASE_LOADING,
        scene_name="scene.usda",
        pid=123,
        directory=tmp_path,
    )

    def deny_signal(_pid: int, _signal: int) -> None:
        raise PermissionError("not mine")

    monkeypatch.setattr(session_lifecycle.os, "kill", deny_signal)
    # Pin the access-denied-means-live semantics through the default
    # checker on every platform: the Windows probe never uses os.kill
    # (which is TerminateProcess there), so route it through the POSIX
    # signal-0 branch for this structural pin.
    monkeypatch.setattr(session_lifecycle, "_windows_pid_running", session_lifecycle._posix_pid_running)

    assert session_lifecycle.read_stale_crash_marker(current_pid=456, directory=tmp_path) == {}


def test_pid_running_reports_current_process_and_rejects_dead_or_invalid_pids() -> None:
    import subprocess
    import sys as _sys

    assert session_lifecycle.pid_running(os.getpid()) is True

    exited = subprocess.Popen([_sys.executable, "-c", "pass"])
    exited.wait()
    assert session_lifecycle.pid_running(exited.pid) is False

    assert session_lifecycle.pid_running(0) is False
    assert session_lifecycle.pid_running(-4) is False
    assert session_lifecycle.pid_running("not-a-pid") is False


@pytest.mark.skipif(os.name != "nt", reason="Windows-only pid probe pin")
def test_pid_running_never_signals_on_windows(monkeypatch) -> None:
    # os.kill(pid, 0) is TerminateProcess on Windows: the liveness probe
    # must never route through it (blender-live-render task05-02).
    def forbidden_kill(_pid: int, _signal: int) -> None:
        raise AssertionError("pid_running must not call os.kill on Windows")

    monkeypatch.setattr(session_lifecycle.os, "kill", forbidden_kill)

    assert session_lifecycle.pid_running(os.getpid()) is True


def test_cleanup_target_selection_and_result() -> None:
    processes = [
        {"pid": 100, "parent_alive": False},
        {"pid": 200, "parent_alive": True},
        {"pid": 300, "parent_alive": True},
    ]

    selected = session_lifecycle.select_cleanup_targets(processes, own_pids={300})
    broad_selected = session_lifecycle.select_cleanup_targets(
        processes,
        own_pids={300},
        include_non_orphans=True,
    )

    assert selected == (100,)
    assert broad_selected == (100, 200)

    result = session_lifecycle.cleanup_diagnostics(
        processes=processes,
        own_pids={300},
        selected_pids=selected,
        killed_count=1,
    )

    assert result["process_count"] == 3
    assert result["own_pids"] == [300]
    assert result["selected_pids"] == [100]
    assert result["killed_count"] == 1


def test_derive_status_surfaces_a_recorded_viewport_error() -> None:
    """A not-ready engine with a recorded error must say so instead of an
    eternal "Starting OVRTX" (Junk Shop regression, 2026-07-07: blocked
    scene conversion was invisible in the panel)."""

    status = session_lifecycle.derive_status(
        engine_active=True,
        ready=False,
        last_error="Authored scene activation failed: unsupported world",
    )
    assert status["status"] == session_lifecycle.STATUS_ERROR
    assert status["label"] == "Viewport error"
    assert "unsupported world" in status["hint"]

    # The error clears on success: a ready engine is Live regardless.
    live = session_lifecycle.derive_status(
        engine_active=True,
        ready=True,
        last_error="stale",
    )
    assert live["status"] == session_lifecycle.STATUS_LIVE

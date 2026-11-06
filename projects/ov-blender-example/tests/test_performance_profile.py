# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact timer and optional pyinstrument integration coverage."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
import threading
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import performance_profile as _performance_profile  # noqa: E402


class _FakeSession:
    duration = 1.25
    cpu_time = 0.5
    sample_count = 17

    def save(self, path: Path) -> None:
        Path(path).write_text('{"session": true}\n', encoding="utf-8")


class _FakeProfiler:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.last_session = None
        self.target_description = ""

    def start(self, *, target_description: str) -> None:
        self.target_description = target_description

    def stop(self) -> _FakeSession:
        self.last_session = _FakeSession()
        return self.last_session

    def output_html(self) -> str:
        return "<html>profile</html>"

    def output_text(self, **_kwargs: object) -> str:
        return "profile text\n"

    def output(self, _renderer: object) -> str:
        return '{"root_frame": {}}\n'


class _FakeJSONRenderer:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


def _reload(monkeypatch, tmp_path: Path):
    pyinstrument = ModuleType("pyinstrument")
    pyinstrument.Profiler = _FakeProfiler
    renderers = ModuleType("pyinstrument.renderers")
    renderers.JSONRenderer = _FakeJSONRenderer
    monkeypatch.setitem(sys.modules, "pyinstrument", pyinstrument)
    monkeypatch.setitem(sys.modules, "pyinstrument.renderers", renderers)
    monkeypatch.setenv(
        _performance_profile.OUTPUT_DIR_ENV,
        str(tmp_path),
    )
    return importlib.reload(_performance_profile)


def test_exact_timers_and_main_capture_are_persisted(monkeypatch, tmp_path: Path) -> None:
    profile = _reload(monkeypatch, tmp_path)
    readings = iter((100, 300, 1_000, 1_600))
    monkeypatch.setattr(profile.time, "perf_counter_ns", lambda: next(readings))

    @profile.timed("main.example")
    def measured(value: int) -> int:
        return value + 1

    profile.start_main()
    assert measured(1) == 2
    assert measured(2) == 3
    profile.stop_main()

    artifact = json.loads((tmp_path / profile.SUMMARY_NAME).read_text(encoding="utf-8"))
    timer = artifact["timers"]["main.example"]
    assert timer == {
        "count": 2,
        "max_ms": 0.0006,
        "mean_ms": 0.0004,
        "min_ms": 0.0002,
        "p50_ms": 0.0002,
        "p95_ms": 0.0006,
        "retained_sample_count": 2,
        "threads": {threading.current_thread().name: 2},
        "total_ms": 0.0008,
    }
    assert artifact["timer_semantics"] == "inclusive_wall_time_do_not_sum_nested_scopes"
    assert artifact["active_captures"] == []
    assert len(artifact["captures"]) == 1
    capture = artifact["captures"][0]
    assert capture["label"] == "blender-main"
    assert capture["sample_count"] == 17
    assert set(capture["files"]) == {"html", "json", "session", "text"}
    for path in capture["files"].values():
        assert Path(path).is_file()


def test_profiled_background_thread_gets_independent_capture(
    monkeypatch, tmp_path: Path
) -> None:
    profile = _reload(monkeypatch, tmp_path)
    completed: list[str] = []

    @profile.profiled_thread("worker")
    @profile.timed("background.worker")
    def worker() -> None:
        completed.append(threading.current_thread().name)

    thread = threading.Thread(target=worker, name="profile-test-worker")
    thread.start()
    thread.join()

    artifact = json.loads((tmp_path / profile.SUMMARY_NAME).read_text(encoding="utf-8"))
    assert completed == ["profile-test-worker"]
    assert artifact["timers"]["background.worker"]["threads"] == {
        "profile-test-worker": 1
    }
    assert artifact["captures"][0]["label"] == "worker"
    assert artifact["captures"][0]["thread_name"] == "profile-test-worker"


def test_disabled_decorator_has_no_timer_side_effect(monkeypatch) -> None:
    monkeypatch.delenv(_performance_profile.OUTPUT_DIR_ENV, raising=False)
    profile = importlib.reload(_performance_profile)

    def operation() -> str:
        return "ok"

    measured = profile.timed("disabled")(operation)
    profiled = profile.profiled_thread("disabled-thread")(operation)

    assert measured is operation
    assert profiled is operation
    assert measured() == "ok"
    assert profile.summary()["timers"] == {}

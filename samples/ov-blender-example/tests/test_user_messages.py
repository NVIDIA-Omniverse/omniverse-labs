# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Central user-facing message reporting (``user_messages``).

Every user-facing status/error the add-on produces must reach the console
(stdout for INFO, stderr for WARNING/ERROR) immediately and the Blender Info
window via a thread-safe queue drained on the main thread. The bpy-free core
(:class:`UserMessageBus`) is exercised here in the plain pytest lane: fan-out,
level->stream mapping, change-only de-duplication, and thread-safe enqueue from
a worker thread. A headless Blender driver additionally verifies the
operator + timer pump path executes when an executable is available; otherwise
that test skips.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from blender_test_support import blender_executable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import user_messages  # noqa: E402
from ovrtx_blender_example.user_messages import (  # noqa: E402
    ERROR,
    INFO,
    PREFIX,
    WARNING,
    UserMessageBus,
    normalize_level,
)


# --- Level normalisation --------------------------------------------------

def test_normalize_level_accepts_strings_case_insensitively() -> None:
    assert normalize_level("info") == INFO
    assert normalize_level("Warning") == WARNING
    assert normalize_level("ERROR") == ERROR


def test_normalize_level_folds_report_style_sets_to_most_severe() -> None:
    assert normalize_level({"ERROR"}) == ERROR
    assert normalize_level({"WARNING"}) == WARNING
    assert normalize_level({"INFO"}) == INFO
    # Blender may pass multiple flags; the most severe recognised one wins.
    assert normalize_level({"INFO", "WARNING"}) == WARNING
    assert normalize_level({"WARNING", "ERROR"}) == ERROR


def test_normalize_level_degrades_unknown_to_info() -> None:
    assert normalize_level("bogus") == INFO
    assert normalize_level({"OPERATOR"}) == INFO
    assert normalize_level(None) == INFO
    assert normalize_level(object()) == INFO


# --- Fan-out + level -> stream mapping ------------------------------------

def _bus_with_capture() -> tuple[UserMessageBus, io.StringIO, io.StringIO]:
    out, err = io.StringIO(), io.StringIO()
    return UserMessageBus(stdout=out, stderr=err), out, err


def test_emit_fans_out_to_console_and_info_queue() -> None:
    bus, out, err = _bus_with_capture()
    assert bus.emit(INFO, "viewport ready") is True
    assert out.getvalue() == f"{PREFIX} viewport ready\n"
    assert err.getvalue() == ""
    pending = bus.take_pending()
    assert [(m.level, m.text) for m in pending] == [(INFO, "viewport ready")]
    # take_pending drains the queue.
    assert bus.pending_count() == 0


def test_info_goes_to_stdout_warning_and_error_go_to_stderr() -> None:
    bus, out, err = _bus_with_capture()
    bus.emit(INFO, "status", context="a")
    bus.emit(WARNING, "careful", context="b")
    bus.emit(ERROR, "boom", context="c")
    assert out.getvalue() == f"{PREFIX} status\n"
    assert err.getvalue() == f"{PREFIX} careful\n{PREFIX} boom\n"


def test_report_style_level_set_maps_to_stderr_for_errors() -> None:
    bus, out, err = _bus_with_capture()
    bus.emit({"ERROR"}, "translation failed")
    assert out.getvalue() == ""
    assert err.getvalue() == f"{PREFIX} translation failed\n"


def test_empty_text_is_dropped() -> None:
    bus, out, err = _bus_with_capture()
    assert bus.emit(INFO, "") is False
    assert bus.emit(INFO, None) is False
    assert out.getvalue() == ""
    assert bus.pending_count() == 0


# --- Change-only de-duplication (frame-rate spam guard) -------------------

def test_repeated_identical_message_emits_once_until_it_changes() -> None:
    bus, out, err = _bus_with_capture()
    context = "viewport-status:1"
    # update_stats-style: same string every draw tick.
    for _ in range(5):
        bus.emit(INFO, "Viewport session ready", context=context)
    assert out.getvalue() == f"{PREFIX} Viewport session ready\n"
    assert bus.pending_count() == 1

    # A changed message emits again (both channels).
    bus.emit(INFO, "Re-syncing scene", context=context)
    assert out.getvalue().count(PREFIX) == 2
    assert bus.pending_count() == 2

    # Returning to the earlier text emits again (dedup tracks only the last).
    bus.emit(INFO, "Viewport session ready", context=context)
    assert out.getvalue().count(PREFIX) == 3


def test_dedup_is_per_context() -> None:
    bus, out, err = _bus_with_capture()
    bus.emit(INFO, "same", context="engine-a")
    bus.emit(INFO, "same", context="engine-b")
    # Distinct contexts each emit even with identical text.
    assert out.getvalue() == f"{PREFIX} same\n{PREFIX} same\n"
    assert bus.pending_count() == 2


def test_dedup_distinguishes_level_change_with_same_text() -> None:
    bus, out, err = _bus_with_capture()
    bus.emit(INFO, "state", context="x")
    bus.emit(ERROR, "state", context="x")
    assert out.getvalue() == f"{PREFIX} state\n"
    assert err.getvalue() == f"{PREFIX} state\n"


def test_dedup_can_be_disabled_for_discrete_events() -> None:
    bus, out, err = _bus_with_capture()
    bus.emit(ERROR, "render failed", context="f12", dedup=False)
    bus.emit(ERROR, "render failed", context="f12", dedup=False)
    assert err.getvalue().count("render failed") == 2
    assert bus.pending_count() == 2


# --- Channel selection ----------------------------------------------------

def test_to_info_false_mirrors_console_only() -> None:
    bus, out, err = _bus_with_capture()
    bus.emit(ERROR, "final render timed out", to_info=False)
    assert err.getvalue() == f"{PREFIX} final render timed out\n"
    assert bus.pending_count() == 0


def test_to_console_false_queues_info_only() -> None:
    bus, out, err = _bus_with_capture()
    bus.emit(WARNING, "thread leaked", to_console=False)
    assert out.getvalue() == ""
    assert err.getvalue() == ""
    assert bus.pending_count() == 1


# --- Drain ----------------------------------------------------------------

def test_drain_hands_each_message_to_sink_and_clears_queue() -> None:
    bus, out, err = _bus_with_capture()
    bus.emit(INFO, "one", context="a")
    bus.emit(ERROR, "two", context="b")
    seen: list[tuple[str, str]] = []
    count = bus.drain(lambda level, text: seen.append((level, text)))
    assert count == 2
    assert seen == [(INFO, "one"), (ERROR, "two")]
    assert bus.pending_count() == 0


def test_drain_survives_a_failing_sink() -> None:
    bus, out, err = _bus_with_capture()
    bus.emit(INFO, "a", context="1")
    bus.emit(INFO, "b", context="2")

    def _boom(level: str, text: str) -> None:
        raise RuntimeError("sink exploded")

    # A sink exception must not strand remaining messages or propagate.
    assert bus.drain(_boom) == 2
    assert bus.pending_count() == 0


# --- Thread-safe enqueue from a worker thread -----------------------------

def test_enqueue_is_thread_safe_from_worker_threads() -> None:
    bus = UserMessageBus(stdout=io.StringIO(), stderr=io.StringIO())
    thread_count = 8
    per_thread = 50
    start = threading.Barrier(thread_count)

    def _worker(index: int) -> None:
        start.wait()
        for i in range(per_thread):
            # A unique context per (thread, i) defeats dedup so every emit
            # enqueues, exercising concurrent appends.
            bus.emit(ERROR, f"failure {index}-{i}", context=f"{index}-{i}")

    threads = [threading.Thread(target=_worker, args=(n,)) for n in range(thread_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    pending = bus.take_pending()
    assert len(pending) == thread_count * per_thread
    # No corruption: every message is well-formed and unique.
    texts = {message.text for message in pending}
    assert len(texts) == thread_count * per_thread


def test_module_level_helpers_use_the_default_bus() -> None:
    bus = user_messages.default_bus()
    bus.reset()
    try:
        user_messages.report(ERROR, "centralized boom", context="unit", to_console=False)
        pending = bus.take_pending()
        assert [(m.level, m.text) for m in pending] == [(ERROR, "centralized boom")]
    finally:
        bus.reset()


def test_mirror_to_console_never_touches_the_info_queue() -> None:
    bus = user_messages.default_bus()
    bus.reset()
    try:
        user_messages.mirror_to_console(INFO, "console only")
        assert bus.pending_count() == 0
    finally:
        bus.reset()


# --- Operator reports: Info natively + console mirror ---------------------


class _FakeOperator:
    def __init__(self) -> None:
        self.reports: list[tuple[set[str], str]] = []

    def report(self, levels: set[str], message: str) -> None:
        self.reports.append((set(levels), message))


class _FailingOperator:
    def report(self, levels: set[str], message: str) -> None:
        raise ReferenceError("StructRNA of type Operator has been removed")


def test_report_for_operator_reports_natively_and_mirrors_to_console(
    capsys: pytest.CaptureFixture,
) -> None:
    bus = user_messages.default_bus()
    bus.reset()
    operator = _FakeOperator()
    try:
        user_messages.report_for_operator(operator, {"INFO"}, "Runtime installed: /x")
        user_messages.report_for_operator(operator, {"ERROR"}, "Runtime install failed")
        # Native operator report carries the original level set + message.
        assert operator.reports == [
            ({"INFO"}, "Runtime installed: /x"),
            ({"ERROR"}, "Runtime install failed"),
        ]
        captured = capsys.readouterr()
        # Console mirror: INFO on stdout, ERROR on stderr, [ovrtx] prefix.
        assert captured.out == f"{PREFIX} Runtime installed: /x\n"
        assert captured.err == f"{PREFIX} Runtime install failed\n"
        # No Info double-post: the operator report reaches Info natively, so
        # nothing is queued for the pump.
        assert bus.pending_count() == 0
    finally:
        bus.reset()


def test_report_for_operator_accepts_plain_string_levels(
    capsys: pytest.CaptureFixture,
) -> None:
    bus = user_messages.default_bus()
    bus.reset()
    operator = _FakeOperator()
    try:
        user_messages.report_for_operator(operator, "WARNING", "runtime not ready")
        assert operator.reports == [({"WARNING"}, "runtime not ready")]
        assert capsys.readouterr().err == f"{PREFIX} runtime not ready\n"
        assert bus.pending_count() == 0
    finally:
        bus.reset()


def test_report_for_operator_repeats_are_not_suppressed(
    capsys: pytest.CaptureFixture,
) -> None:
    # Discrete user-triggered events: pressing the same button twice must
    # produce two console lines and two native reports.
    bus = user_messages.default_bus()
    bus.reset()
    operator = _FakeOperator()
    try:
        user_messages.report_for_operator(operator, {"INFO"}, "Runtime removed.")
        user_messages.report_for_operator(operator, {"INFO"}, "Runtime removed.")
        assert len(operator.reports) == 2
        assert capsys.readouterr().out == (
            f"{PREFIX} Runtime removed.\n{PREFIX} Runtime removed.\n"
        )
    finally:
        bus.reset()


def test_report_for_operator_still_mirrors_when_native_report_fails(
    capsys: pytest.CaptureFixture,
) -> None:
    bus = user_messages.default_bus()
    bus.reset()
    try:
        # A stale operator wrapper must not swallow the message entirely.
        user_messages.report_for_operator(_FailingOperator(), {"ERROR"}, "still visible")
        assert capsys.readouterr().err == f"{PREFIX} still visible\n"
    finally:
        bus.reset()


# --- Headless Blender: operator + pump path -------------------------------
_DRIVER = """
import json
import sys
import traceback

result = {"errors": [], "steps": []}
output_path = sys.argv[sys.argv.index("--") + 1]

try:
    import bpy

    sys.path.insert(0, __ADDON_PATH__)
    from ovrtx_blender_example import user_messages

    # Registration installs the report operator + main-thread pump timer.
    assert user_messages.register(bpy) is True
    result["steps"].append("registered")

    # Directly invoke the operator (the pump's delivery mechanism): its
    # execute() must call self.report(level, message) and finish. INFO reports
    # do not raise through bpy.ops (only ERROR does, by Blender design).
    op_result = bpy.ops.ovrtx_example.report(level="INFO", message="headless status")
    result["operator_result"] = list(op_result)
    result["steps"].append("operator_reported")

    # Enqueue an ERROR from a "worker" (console channel off) and drain it via
    # the exact sink the timer pump uses. An ERROR report makes bpy.ops raise a
    # RuntimeError; the pump's sink must swallow it so the queue still drains.
    user_messages.report(
        user_messages.ERROR, "queued failure", context="headless", to_console=False
    )
    result["pending_before_drain"] = user_messages.default_bus().pending_count()
    user_messages._pump()
    result["pending_after_drain"] = user_messages.default_bus().pending_count()
    result["steps"].append("drained")

    # The pump reschedules itself (returns its interval) even after draining an
    # ERROR message.
    result["pump_interval"] = user_messages._pump()
    result["steps"].append("pumped")

    assert user_messages.unregister(bpy) is True
    result["steps"].append("unregistered")
except Exception:
    result["errors"].append(traceback.format_exc())

with open(output_path, "w", encoding="utf-8") as stream:
    json.dump(result, stream)
"""


def test_report_operator_and_pump_execute_headless(tmp_path: Path) -> None:
    blender = blender_executable()
    if blender is None:
        pytest.skip("no Blender executable available for headless user_messages regression")

    driver = tmp_path / "user_messages_driver.py"
    driver.write_text(
        _DRIVER.replace("__ADDON_PATH__", repr(str(ROOT / "addon"))),
        encoding="utf-8",
    )
    output = tmp_path / "result.json"

    completed = subprocess.run(
        (
            str(blender),
            "--background",
            "--factory-startup",
            "--python-exit-code",
            "1",
            "--python",
            str(driver),
            "--",
            str(output),
        ),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert output.is_file(), completed.stdout + completed.stderr
    result = json.loads(output.read_text(encoding="utf-8"))

    assert result["errors"] == []
    assert result["steps"] == [
        "registered",
        "operator_reported",
        "drained",
        "pumped",
        "unregistered",
    ]
    assert result["operator_result"] == ["FINISHED"]
    assert result["pending_before_drain"] == 1
    # The pump drained the ERROR message despite bpy.ops raising on ERROR reports.
    assert result["pending_after_drain"] == 0
    assert result["pump_interval"] == user_messages._PUMP_INTERVAL_S

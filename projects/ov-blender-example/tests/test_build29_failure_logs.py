# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Worker failures name the native log and keep its error lines (build 29 review).

A real ``RuntimeSession`` drives a fake worker that prints native-looking error
lines to its stdout (``worker.log``) and its ``ovrtx.log`` before it dies.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))
from ovrtx_blender_example import direct_sdk_client, user_messages, worker_failure_logs  # noqa: E402
from ovrtx_blender_example.direct_runtime_session import RuntimeSession  # noqa: E402
from ovrtx_blender_example.ovrtx_runtime_client import RenderClientError  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "runtime_session_worker_for_failure_logs", Path(__file__).with_name("test_direct_runtime_session.py"))
_session_tests = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_session_tests)

# The shared fake worker, plus two native-log hooks that run before any crash.
WORKER = _session_tests.WORKER.replace(
    "    if params.get('crash_once'):",
    "    if params.get('log'):\n"
    "        print(params['log'], flush=True)\n"
    "    if params.get('ovrtx_log'):\n"
    "        with open(pathlib.Path(a.output_root) / 'ovrtx.log', 'a') as native:\n"
    "            native.write(params['ovrtx_log'] + '\\n')\n"
    "    if params.get('crash_once'):",
    1,
)
CUDA_LINE = "[Error] [carb.cudainterop] CUDA error 700: cudaErrorIllegalAddress - an illegal memory access"
DUMP = r"C:\Users\Some User\AppData\Local\Temp\nv crash\ovrtx-0001.nv-gpudmp"


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    user_messages.default_bus().reset()
    monkeypatch.setattr(worker_failure_logs, "_recent", worker_failure_logs.deque(maxlen=8))
    yield
    user_messages.default_bus().reset()


def make_client(tmp_path, monkeypatch, *, max_restarts=2):
    worker = tmp_path / "worker.py"
    worker.write_text(WORKER)
    runtime = RuntimeSession(sys.executable, worker, "renderer", tmp_path / "renderer",
                             timeout_s=5, max_restarts=max_restarts)
    runtime.restart_backoff_s = (0.0, 0.0)
    monkeypatch.setattr(direct_sdk_client, "_session", lambda kind: runtime)
    client = direct_sdk_client.PublicSdkRenderClient()
    # Scene capture and USD metadata reads are not under test.
    monkeypatch.setattr(client, "_load_params", lambda spec: {"scene_path": "scene.usda"})
    monkeypatch.setattr(client, "_snapshot_source", lambda path, flatten=True: path)
    return client, runtime


def epoch_log(runtime, epoch, name="worker.log"):
    return Path(runtime.output_root) / f"epoch-{epoch}" / name


def crash_render(client):
    client._call("load", {"scene_path": "scene.usda"})
    with pytest.raises(RenderClientError) as caught:
        client._call("render", {"log": CUDA_LINE, "ovrtx_log": f"2026-10-21 [Fatal] device lost; dump {DUMP}",
                                "crash_always": True})
    return caught.value


class _FakeRenderEngine:
    def __init__(self, *args, **kwargs):
        self.stats = []
        self.reports = []

    def update_stats(self, engine, message):
        self.stats.append((engine, message))

    def tag_redraw(self):
        pass

    def report(self, levels, message):
        self.reports.append((set(levels), message))


def load_engine(monkeypatch):
    name = "ovrtx_blender_example._engine_failure_logs_test"
    fake_bpy = SimpleNamespace(
        types=SimpleNamespace(RenderEngine=_FakeRenderEngine),
        app=SimpleNamespace(timers=SimpleNamespace()),
        context=SimpleNamespace(window_manager=SimpleNamespace(windows=())),
    )
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    spec = importlib.util.spec_from_file_location(name, ROOT / "addon/ovrtx_blender_example/engine.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


class _FailedFuture:
    def __init__(self, error):
        self.error = error

    def wait(self, timeout):
        return True

    def result(self, timeout):
        raise self.error


def test_final_error_names_worker_log(tmp_path, monkeypatch, capsys):
    client, runtime = make_client(tmp_path, monkeypatch, max_restarts=0)
    try:
        error = crash_render(client)
    finally:
        runtime.close()
    log = epoch_log(runtime, 1)
    assert str(error).endswith(f"Details: {log}")
    assert worker_failure_logs.details_path(str(error)) == str(log)
    assert error.worker_failure["ovrtx_log"] == str(epoch_log(runtime, 1, "ovrtx.log"))
    # The full matched lines go to the console, not into the message.
    console = capsys.readouterr().err
    assert CUDA_LINE in console and f"Native worker log: {log}" in console
    assert CUDA_LINE not in str(error)

    module = load_engine(monkeypatch)
    engine = module.OvrtxExampleRenderEngine()
    request = SimpleNamespace(max_samples=8)
    # Thread-job routes: the job's error reaches _report_final.
    assert engine._await_final_render_outcome(_FailedFuture(error), threading.Event(), request, {}) is None
    # Scene-generation route: activation rewords the error, its cause keeps the log.
    try:
        raise RuntimeError("Scene generation activation failed; predecessor_restore=failed") from error
    except RuntimeError as wrapped:
        assert worker_failure_logs.describe(wrapped).endswith(f"Details: {log}")
        assert engine._await_final_render_outcome(_FailedFuture(wrapped), threading.Event(), request, {}) is None
    errors = [message for levels, message in engine.reports if "ERROR" in levels]
    assert len(errors) == 2 and all(f"Details: {log}" in message for message in errors)
    assert all(len(message) < 400 for message in errors)


def test_viewport_error_names_worker_log(tmp_path, monkeypatch):
    client, runtime = make_client(tmp_path, monkeypatch, max_restarts=0)
    try:
        error = crash_render(client)
    finally:
        runtime.close()
    log = epoch_log(runtime, 1)
    module = load_engine(monkeypatch)
    engine = module.OvrtxExampleRenderEngine()
    user_messages.default_bus().reset()
    # The render loop publishes only the text (frame.detail); no exception chain.
    engine._report_viewport_error(RenderClientError(str(error)))
    assert engine.reports[-1][0] == {"ERROR"} and f"Details: {log}" in engine.reports[-1][1]
    assert f"Details: {log}" in engine.stats[-1][1] and engine.stats[-1][1].startswith("Viewport failed:")
    bus = user_messages.default_bus().take_pending()
    assert [m.level for m in bus] == ["ERROR"] and f"Details: {log}" in bus[0].text
    # An error raised with its cause attached names the log as well.
    engine._report_viewport_error(error)
    assert f"Details: {log}" in engine.reports[-1][1]


def test_matched_error_lines_in_artifact(tmp_path, monkeypatch):
    client, runtime = make_client(tmp_path, monkeypatch, max_restarts=0)
    try:
        client._call("load", {"scene_path": "scene.usda"})
        noise = [f"[Error] [rtx] shader {n} failed" for n in range(25)]
        client._call("render", {"log": "\n".join(["[Info] startup ok", *noise, "plain progress line"])})
        error = crash_render(client)
    finally:
        runtime.close()
    lines = error.worker_failure["lines"]
    # The last 20 matching lines of both logs, oldest first; other lines never.
    assert len(lines) == worker_failure_logs.MAX_LINES
    assert lines[-2] == f"worker.log: {CUDA_LINE}" and lines[-1].startswith("ovrtx.log: 2026-10-21 [Fatal]")
    assert not any("startup ok" in line or "progress" in line for line in lines)

    module = load_engine(monkeypatch)
    engine = module.OvrtxExampleRenderEngine()
    engine._report_viewport_error(RenderClientError(str(error)))
    artifact = engine._viewport_artifact(running=False)
    failures = artifact["worker_failures"]
    assert [failure["worker_log"] for failure in failures] == [str(epoch_log(runtime, 1))]
    assert failures[0]["lines"] == lines and failures[0]["reported_error"] == str(error)


def test_startup_diagnostics_reports_log_paths(tmp_path, monkeypatch):
    client, runtime = make_client(tmp_path, monkeypatch)
    try:
        client.start_session(SimpleNamespace(sensor_paths=(), render_var="LdrColor"), "sim")
        logs = client.startup_diagnostics["render_worker"]["logs"]
        assert logs["status"] == "file"
        assert logs["worker_log"] == str(epoch_log(runtime, 1)) and Path(logs["worker_log"]).is_file()
        assert logs["renderer_log"] == str(epoch_log(runtime, 1, "ovrtx.log"))
        assert logs["log_dir"] == str(epoch_log(runtime, 1).parent)
        module = load_engine(monkeypatch)
        assert module._logs_from_startup_diagnostics(client.startup_diagnostics) == logs

        # A recovered worker is a new epoch with its own logs; the failure is
        # still announced, naming the failed worker's log.
        user_messages.default_bus().reset()
        client._call("render", {"log": CUDA_LINE, "crash_once": str(tmp_path / "crashed")})
        assert client.startup_diagnostics["render_worker"]["logs"]["worker_log"] == str(epoch_log(runtime, 2))
        warnings = [m for m in user_messages.default_bus().take_pending() if m.level == "WARNING"]
        assert len(warnings) == 1 and warnings[0].text.endswith(f"Details: {epoch_log(runtime, 1)}")
        assert worker_failure_logs.recent()[-1]["recovered"] is True
    finally:
        runtime.close()


def test_gpu_dump_path_detected(tmp_path, monkeypatch, capsys):
    client, runtime = make_client(tmp_path, monkeypatch, max_restarts=0)
    try:
        client._call("load", {"scene_path": "scene.usda"})
        client._call("render", {"log": "GPU crash dump written to /var/tmp/nv/core-17.NV-GPUDMP (unrelated text)"})
        (epoch_log(runtime, 1).parent / "beside.nv-gpudmp").write_bytes(b"dump")
        error = crash_render(client)
    finally:
        runtime.close()
    dumps = error.worker_failure["gpu_dumps"]
    assert dumps == ["/var/tmp/nv/core-17.NV-GPUDMP", DUMP, str(epoch_log(runtime, 1).parent / "beside.nv-gpudmp")]
    console = capsys.readouterr().err
    assert all(f"GPU crash dump: {path}" in console for path in dumps)
    assert not worker_failure_logs.collect(tmp_path / "missing" / "worker.log")["gpu_dumps"]

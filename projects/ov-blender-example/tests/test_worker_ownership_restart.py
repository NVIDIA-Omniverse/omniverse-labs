# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Worker-process ownership tracking and the RTPT restart-applies channel.

Real-GPU evidence (runtime measurements, "How to make the RTPT quality sliders take
effect"): the worker honors RTPT quality values only via ``/rtx/rtpt/*`` carb
settings in its startup config, read once at worker-process launch. The native
client terminates only a worker process it spawned itself (verified: an
attached/foreign worker survives ``shutdown()``; a self-launched one dies), so:

- owned worker: the session teardown inside a restart/re-key kills the worker
  and the next ensure relaunches it with the freshly authored config — slider
  changes apply on an in-app session restart (proven by
  ``out/artifacts/ovrtx-rtpt-render-product-honor/probe_restart_applies_sliders.py``:
  luma 86.322 -> 0.0 across ``deactivate()`` + ``ensure()``).
- attached worker: never killed by the add-on; it keeps its old launch-time
  settings, so the user guidance stays "quit Blender and let it exit".

These tests pin the ownership signal (pre-launch endpoint probe + native
``worker_process_alive``), its propagation client -> controller -> fallback
message, and that the controller's ensure authors the startup config.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import socket
import sys
import threading

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import ovrtx_runtime_client as client_module  # noqa: E402
from ovrtx_blender_example import ovrtx_session_controller as controller_module  # noqa: E402
from ovrtx_blender_example.ovrtx_runtime_client import OvrtxRuntimeClient  # noqa: E402
from ovrtx_blender_example.render_requests import RenderRequest  # noqa: E402
from ovrtx_blender_example.viewport_render_thread import (  # noqa: E402
    _render_setting_fallback_message,
)


# --- _endpoint_listening -----------------------------------------------------


def test_endpoint_listening_true_for_bound_port() -> None:
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        assert client_module._endpoint_listening(f"127.0.0.1:{port}") is True


def test_endpoint_listening_false_for_free_port() -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        free_port = probe.getsockname()[1]
    assert client_module._endpoint_listening(f"127.0.0.1:{free_port}") is False


def test_endpoint_listening_false_for_invalid_endpoint() -> None:
    assert client_module._endpoint_listening("") is False
    assert client_module._endpoint_listening("127.0.0.1:notaport") is False
    assert client_module._endpoint_listening("50051") is False


# --- ownership evaluation ----------------------------------------------------


class _Native:
    def __init__(self, health: object = None) -> None:
        self._health = health

    def check_health(self) -> object:
        if isinstance(self._health, Exception):
            raise self._health
        return self._health


def _client() -> OvrtxRuntimeClient:
    return OvrtxRuntimeClient(
        worker_command="worker --address 127.0.0.1 --port 50051",
        native_client_module="fake",
    )


def test_ownership_true_when_our_previous_worker_is_alive(monkeypatch: pytest.MonkeyPatch) -> None:
    # A live worker this process spawned earlier (check_health says so) is
    # ours — the endpoint probe must not even run.
    def _boom(_endpoint: str, timeout_seconds: float = 0) -> bool:
        raise AssertionError("endpoint probe must be skipped")

    monkeypatch.setattr(client_module, "_endpoint_listening", _boom)
    client = _client()
    assert client._evaluate_worker_ownership(_Native({"worker_process_alive": True})) is True


def test_ownership_true_when_port_free(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client_module, "_endpoint_listening", lambda endpoint, **_: False)
    client = _client()
    assert client._evaluate_worker_ownership(_Native({"worker_process_alive": False})) is True


def test_ownership_false_when_foreign_worker_serves_the_port(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}

    def _listening(endpoint: str, **_: object) -> bool:
        seen["endpoint"] = endpoint
        return True

    monkeypatch.setattr(client_module, "_endpoint_listening", _listening)
    client = _client()
    assert client._evaluate_worker_ownership(_Native({"worker_process_alive": False})) is False
    assert seen["endpoint"] == "127.0.0.1:50051"


def test_ownership_probes_port_when_check_health_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client_module, "_endpoint_listening", lambda endpoint, **_: True)
    client = _client()
    assert client._evaluate_worker_ownership(_Native(RuntimeError("no channel"))) is False


def test_worker_owned_none_before_any_session() -> None:
    assert _client().worker_owned is None


# --- controller propagation --------------------------------------------------


class _FakeClient:
    def __init__(self, *, worker_owned: object = True) -> None:
        self.worker_owned = worker_owned
        self.startup_diagnostics = {"render_worker": {"status": "ready"}}
        self.deletes = 0

    def start_session(self, spec: object, simulation_id: str | None = None) -> str:
        return simulation_id or "sim"

    def delete_simulation(self, simulation_id: str) -> str:
        self.deletes += 1
        return "stopped"

    def shutdown(self) -> None:
        pass


def _request(tmp_path: Path, **changes: object) -> RenderRequest:
    return replace(
        RenderRequest(
            input_usd_path=str(tmp_path / "scene.usda"),
            sensor_paths=("/Render/Product",),
            selected_sensor_paths=("/Render/Product",),
            width=1,
            height=1,
            camera_prim_path="/World/Camera",
            worker_command="worker",
            native_client_module="client",
        ),
        **changes,
    )


def test_controller_worker_owned_none_without_client() -> None:
    controller = controller_module.OvrtxSessionController()
    assert controller.worker_owned is None


def test_controller_worker_owned_mirrors_active_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "scene.usda").write_text("#usda 1.0\n", encoding="utf-8")
    for owned, expected in ((True, True), (False, False), (None, None)):
        client = _FakeClient(worker_owned=owned)
        monkeypatch.setattr(
            controller_module, "_runtime_client_from_request", lambda request, c=client: c
        )
        controller = controller_module.OvrtxSessionController()
        controller.ensure(_request(tmp_path))
        assert controller.worker_owned is expected
        assert controller.diagnostics()["worker_owned"] is expected


def test_controller_ensure_authors_worker_config_with_request_quality(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "scene.usda").write_text("#usda 1.0\n", encoding="utf-8")
    authored: list[tuple[str, object]] = []

    def _author(
        worker_command: str, rtpt_quality: object, dlss_enabled: bool = True
    ) -> dict[str, object]:
        authored.append((worker_command, rtpt_quality))
        return {"status": "written", "path": "config"}

    monkeypatch.setattr(
        controller_module.rtpt_worker_config, "author_worker_config", _author
    )
    client = _FakeClient()
    monkeypatch.setattr(
        controller_module, "_runtime_client_from_request", lambda request: client
    )
    controller = controller_module.OvrtxSessionController()
    quality = {"rtpt_max_bounces": 0}
    controller.ensure(_request(tmp_path, rtpt_quality=quality))
    assert authored == [("worker", quality)]
    assert controller.diagnostics()["rtpt_worker_config"]["status"] == "written"


# --- fallback messaging per ownership case -----------------------------------


def test_fallback_message_owned_worker_says_automatic() -> None:
    message = _render_setting_fallback_message("rejected", worker_owned=True)
    assert "applies automatically" in message
    assert "quit Blender" not in message
    assert "/rtx/rtpt/*" in message


def test_fallback_message_attached_worker_says_quit_blender() -> None:
    for owned in (False, None):
        message = _render_setting_fallback_message("rejected", worker_owned=owned)
        assert "quit Blender" in message
        assert "applies automatically" not in message
        assert "was not launched by this Blender session" in message


def test_fallback_message_carries_the_rejection_reason() -> None:
    message = _render_setting_fallback_message(
        "render_setting_value_update_error", worker_owned=True
    )
    assert "render_setting_value_update_error" in message

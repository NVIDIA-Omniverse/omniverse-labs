# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import runtime_services


def test_health_retries_non_serving_on_fixed_schedule() -> None:
    now = [10.0]
    statuses = iter(("NOT_SERVING", "UNKNOWN", "SERVING"))
    timeouts = []

    result = runtime_services.wait_for_serving(
        "OVRTX",
        "127.0.0.1:50051",
        lambda timeout: (timeouts.append(timeout), next(statuses))[1],
        monotonic=lambda: now[0],
        sleep=lambda seconds: now.__setitem__(0, now[0] + seconds),
    )

    assert result.status == "SERVING"
    assert result.attempts == 3
    assert len(timeouts) == 3
    assert now[0] == 12.0


def test_health_fails_fast_for_unsupported_status() -> None:
    with pytest.raises(runtime_services.RuntimeServiceError, match="unsupported"):
        runtime_services.wait_for_serving(
            "OVPhysX", "127.0.0.1:50094", lambda _timeout: "UNIMPLEMENTED"
        )


def test_health_fails_fast_when_managed_process_exits() -> None:
    with pytest.raises(runtime_services.RuntimeServiceError, match="exited"):
        runtime_services.wait_for_serving(
            "OVRTX",
            "127.0.0.1:50051",
            lambda _timeout: "SERVING",
            process_alive=lambda: False,
        )


def test_health_rejects_serving_after_the_shared_deadline() -> None:
    now = [10.0]

    def late_serving(_timeout: float) -> str:
        now[0] = 11.0
        return "SERVING"

    with pytest.raises(runtime_services.RuntimeServiceError, match="health deadline"):
        runtime_services.wait_for_serving(
            "OVRTX",
            "127.0.0.1:50051",
            late_serving,
            deadline=10.5,
            monotonic=lambda: now[0],
        )


def test_health_cancellation_stops_before_another_probe() -> None:
    probes = []

    with pytest.raises(runtime_services.RuntimeServiceError, match="cancelled"):
        runtime_services.wait_for_serving(
            "OVRTX",
            "127.0.0.1:50051",
            lambda timeout: probes.append(timeout) or "SERVING",
            cancelled=lambda: True,
        )

    assert probes == []


def test_owner_rejects_external_cancellation_inside_startup() -> None:
    owner = runtime_services.RuntimeServiceOwner()

    with pytest.raises(runtime_services.RuntimeServiceError, match="cancelled"):
        owner.start(Path("/unused"), cancelled=lambda: True)


def test_owner_stops_every_launched_and_serving_worker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Both workers are launched via start_worker and become healthy. ovphysx's
    # start_worker reports worker_process_alive=False (the flag is unreliable),
    # but the owner must still shut it down on close — otherwise the worker
    # orphans (ovphysx-bridge-server surviving Blender close).
    events = []

    class Client:
        def __init__(self, endpoint: str) -> None:
            events.append(("client", endpoint))

        def health(self, service: str, timeout: float) -> str:
            events.append(("health", service, timeout))
            return "SERVING"

        def close(self) -> None:
            events.append(("close",))

    def module(name: str, managed: bool) -> object:
        return SimpleNamespace(
            start_worker=lambda request: (
                events.append(("start", name, request)),
                {"address": name, "worker_process_alive": managed},
            )[1],
            Client=Client,
            check_health=lambda: {
                "serving": True,
                "worker_process_alive": managed,
            },
            shutdown=lambda: events.append(("shutdown", name)),
        )

    modules = {
        "ovrtx_bridge_client": module("ovrtx", True),
        "ovphysx_bridge_client": module("ovphysx", False),
    }
    defaults = SimpleNamespace(
        worker_command="ovrtx-worker",
        ovphysx_worker_command="ovphysx-worker",
        native_client_path=str(tmp_path),
        ovphysx_root="",
        ovphysx_bridge_runtime_root="",
        ovruntime_root="",
    )
    lease = SimpleNamespace(close=lambda: events.append(("lease-close",)))
    monkeypatch.setattr(runtime_services.bundled_runtime, "defaults", lambda **_kwargs: defaults)
    monkeypatch.setattr(runtime_services.importlib, "import_module", modules.__getitem__)
    monkeypatch.setattr(runtime_services, "acquire_gpu_lease", lambda **_kwargs: lease)
    owner = runtime_services.RuntimeServiceOwner()

    serving = []
    first = owner.start(tmp_path, on_serving=lambda health: serving.append(health.name))
    second = owner.start(tmp_path)
    owner.close()

    assert first == second
    assert serving == ["OVRTX", "OVPhysX"]
    assert ("shutdown", "ovrtx") in events
    # Fixed: a launched-and-serving worker is adopted for shutdown regardless of
    # its (unreliable) worker_process_alive flag.
    assert ("shutdown", "ovphysx") in events


def test_owner_restart_ovrtx_preserves_ovphysx_and_gpu_lease(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = []

    class Client:
        def __init__(self, endpoint: str) -> None:
            self.endpoint = endpoint
            events.append(("client", endpoint))

        def health(self, _service: str, _timeout: float) -> str:
            return "SERVING"

        def close(self) -> None:
            events.append(("close", self.endpoint))

    def module(name: str) -> object:
        return SimpleNamespace(
            start_worker=lambda request: (
                events.append(("start", name)),
                {"address": name, "worker_process_alive": True},
            )[1],
            Client=Client,
            check_health=lambda: {"serving": True, "worker_process_alive": True},
            shutdown=lambda: events.append(("shutdown", name)),
        )

    modules = {
        "ovrtx_bridge_client": module("ovrtx"),
        "ovphysx_bridge_client": module("ovphysx"),
    }
    defaults = SimpleNamespace(
        worker_command="ovrtx-worker",
        ovphysx_worker_command="ovphysx-worker",
        native_client_path=str(tmp_path),
        ovphysx_root="",
        ovphysx_bridge_runtime_root="",
        ovruntime_root="",
    )
    lease = SimpleNamespace(close=lambda: events.append(("lease-close",)))
    monkeypatch.setattr(runtime_services.bundled_runtime, "defaults", lambda **_kwargs: defaults)
    monkeypatch.setattr(runtime_services.importlib, "import_module", modules.__getitem__)
    monkeypatch.setattr(runtime_services, "acquire_gpu_lease", lambda **_kwargs: lease)
    owner = runtime_services.RuntimeServiceOwner()

    owner.start(tmp_path)
    owner.restart_ovrtx(tmp_path)

    assert events.count(("start", "ovrtx")) == 2
    assert events.count(("start", "ovphysx")) == 1
    assert ("shutdown", "ovrtx") in events
    assert ("shutdown", "ovphysx") not in events
    assert ("close", "ovphysx") not in events
    assert ("lease-close",) not in events


def test_owner_restart_observed_during_startup_stays_aggregate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    owner = runtime_services.RuntimeServiceOwner()
    owner._status = "starting"
    events = []

    def cancel() -> None:
        events.append("cancel")
        # Reproduce startup publishing ready between the initial status read
        # and cancellation. The restart decision must remain aggregate.
        owner._status = "ready"
        owner._health["ovphysx"] = runtime_services.ServiceHealth(
            "OVPhysX", "127.0.0.1:50094", "SERVING", 1
        )

    health = runtime_services.ServiceHealth(
        "OVRTX", "127.0.0.1:50051", "SERVING", 1
    )
    monkeypatch.setattr(owner, "cancel", cancel)
    monkeypatch.setattr(
        owner,
        "start",
        lambda root, *, force=False: (
            events.append((Path(root), force)),
            {"ovrtx": health},
        )[1],
    )

    assert owner.restart_ovrtx(tmp_path) is health
    assert events == ["cancel", (tmp_path.resolve(), True)]


def test_owner_does_not_stop_external_connect_service(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # ovphysx with no worker command takes the connect path: the server is
    # externally owned, so the owner must NOT shut it down.
    events = []

    class Client:
        def __init__(self, endpoint: str) -> None:
            events.append(("client", endpoint))

        def health(self, service: str, timeout: float) -> str:
            return "SERVING"

        def close(self) -> None:
            events.append(("close",))

    ovrtx = SimpleNamespace(
        start_worker=lambda request: (
            events.append(("start", "ovrtx", request)),
            {"address": "ovrtx", "worker_process_alive": True},
        )[1],
        Client=Client,
        check_health=lambda: {"serving": True, "worker_process_alive": True},
        shutdown=lambda: events.append(("shutdown", "ovrtx")),
    )
    ovphysx = SimpleNamespace(
        connect=lambda request: (
            events.append(("connect", "ovphysx", request)),
            {"address": "ovphysx", "worker_process_alive": False},
        )[1],
        Client=Client,
        check_health=lambda: {"serving": True, "worker_process_alive": False},
        shutdown=lambda: events.append(("shutdown", "ovphysx")),
    )
    modules = {"ovrtx_bridge_client": ovrtx, "ovphysx_bridge_client": ovphysx}
    defaults = SimpleNamespace(
        worker_command="ovrtx-worker",
        ovphysx_worker_command="",  # no command -> connect path
        native_client_path=str(tmp_path),
        ovphysx_root="",
        ovphysx_bridge_runtime_root="",
        ovruntime_root="",
    )
    lease = SimpleNamespace(close=lambda: events.append(("lease-close",)))
    monkeypatch.delenv("OV_BLENDER_EXAMPLE_OVPHYSX_WORKER_COMMAND", raising=False)
    monkeypatch.setattr(runtime_services.bundled_runtime, "defaults", lambda **_kwargs: defaults)
    monkeypatch.setattr(runtime_services.importlib, "import_module", modules.__getitem__)
    monkeypatch.setattr(runtime_services, "acquire_gpu_lease", lambda **_kwargs: lease)
    owner = runtime_services.RuntimeServiceOwner()

    owner.start(tmp_path)
    owner.close()

    assert ("connect", "ovphysx", {"address": runtime_services.bundled_runtime.DEFAULT_OVPHYSX_ADDRESS}) in events
    assert ("shutdown", "ovrtx") in events
    assert ("shutdown", "ovphysx") not in events

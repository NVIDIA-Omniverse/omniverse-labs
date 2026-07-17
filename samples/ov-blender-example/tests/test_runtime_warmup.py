# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import ovrtx_session_controller, runtime_warmup  # noqa: E402


@pytest.mark.parametrize("stop_status", ["stopped", "not_found"])
def test_warm_shader_cache_discards_one_render(
    tmp_path: Path,
    monkeypatch,
    stop_status: str,
) -> None:
    native = tmp_path / "runtime/native"
    native.mkdir(parents=True)
    defaults = SimpleNamespace(
        native_client_path=str(native),
        worker_command="worker --package-root runtime",
    )
    events: list[object] = []
    progress_updated = threading.Event()

    class Controller:
        def __init__(self, *, simulation_id: str) -> None:
            events.append(("controller", simulation_id))

        def ensure(self, request) -> None:
            events.append(("ensure", request))

        def render(self, request, *, additional_samples: int) -> None:
            assert additional_samples == 1
            assert Path(request.input_usd_path).is_file()
            assert progress_updated.wait(timeout=0.2)
            events.append(("render", request))

        def shutdown(self) -> str:
            events.append("shutdown")
            return stop_status

    monkeypatch.setattr(runtime_warmup.bundled_runtime, "defaults", lambda **_kwargs: defaults)
    monkeypatch.setattr(ovrtx_session_controller, "OvrtxSessionController", Controller)
    monkeypatch.setattr(runtime_warmup, "_PROGRESS_INTERVAL_SECONDS", 0.01)

    messages: list[str] = []

    def progress(message: str) -> None:
        messages.append(message)
        if len(messages) > 1:
            progress_updated.set()

    runtime_warmup.warm_shader_cache(
        tmp_path / "runtime",
        tmp_path,
        progress=progress,
    )

    request = events[1][1]
    assert messages[0] == "Warming shader cache (can take several minutes) — 0:00 elapsed"
    assert len(messages) >= 2
    assert runtime_warmup._progress_message(125).endswith("2:05 elapsed")
    assert (request.width, request.height) == (1, 1)
    assert (request.min_samples, request.max_samples) == (1, 1)
    assert [event[0] if isinstance(event, tuple) else event for event in events] == [
        "controller",
        "ensure",
        "render",
        "shutdown",
    ]
    assert not list(tmp_path.glob("ovrtx-warmup-*"))


@pytest.mark.parametrize(
    ("render_error", "shutdown_error", "expected_error", "message"),
    [
        (None, None, RuntimeError, "cleanup was not confirmed.*'failed'"),
        (ValueError("render failed"), None, ValueError, "render failed"),
        (None, RuntimeError("shutdown failed"), RuntimeError, "shutdown failed"),
        (
            ValueError("render failed"),
            RuntimeError("shutdown failed"),
            ValueError,
            "render failed",
        ),
        (
            ValueError("render failed"),
            KeyboardInterrupt("cancelled"),
            KeyboardInterrupt,
            "cancelled",
        ),
    ],
)
def test_warm_shader_cache_cleanup_error_precedence(
    tmp_path: Path,
    monkeypatch,
    render_error: Exception | None,
    shutdown_error: BaseException | None,
    expected_error: type[BaseException],
    message: str,
) -> None:
    native = tmp_path / "runtime/native"
    native.mkdir(parents=True)
    defaults = SimpleNamespace(native_client_path=str(native), worker_command="worker")

    class Controller:
        def __init__(self, **_kwargs) -> None:
            pass

        def ensure(self, _request) -> None:
            pass

        def render(self, _request, *, additional_samples: int) -> None:
            assert additional_samples == 1
            if render_error is not None:
                raise render_error

        def shutdown(self) -> str:
            if shutdown_error is not None:
                raise shutdown_error
            return "failed"

    monkeypatch.setattr(
        runtime_warmup.bundled_runtime,
        "defaults",
        lambda **_kwargs: defaults,
    )
    monkeypatch.setattr(
        ovrtx_session_controller,
        "OvrtxSessionController",
        Controller,
    )

    with pytest.raises(expected_error, match=message):
        runtime_warmup.warm_shader_cache(tmp_path / "runtime", tmp_path)
    assert not list(tmp_path.glob("ovrtx-warmup-*"))

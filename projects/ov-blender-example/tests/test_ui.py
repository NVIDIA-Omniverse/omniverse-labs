# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import ui  # noqa: E402


def test_wait_cursor_is_restored_when_action_fails() -> None:
    cursors: list[str] = []
    context = SimpleNamespace(window=SimpleNamespace(cursor_set=cursors.append))

    def fail() -> None:
        assert cursors == ["WAIT"]
        raise RuntimeError("shutdown failed")

    with pytest.raises(RuntimeError, match="shutdown failed"):
        ui._with_wait_cursor(context, fail)

    assert cursors == ["WAIT", "DEFAULT"]


def test_reconnect_viewport_session_result_delegates_to_engine_owned_action() -> None:
    calls = 0

    def reconnect() -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {
            "status": "requested",
            "active_session_count": 2,
            "reconnected_session_count": 1,
            "end_reason": "reconnect_requested",
        }

    result = ui.reconnect_viewport_session_result(reconnect)

    assert calls == 1
    assert result == {
        "status": "requested",
        "active_session_count": 2,
        "reconnected_session_count": 1,
        "end_reason": "reconnect_requested",
    }


def test_restart_ovrtx_worker_result_delegates_to_engine_owned_action() -> None:
    calls = 0

    def restart() -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {
            "status": "restarted",
            "active_session_count": 1,
            "restarted_worker_count": 1,
            "end_reason": "worker_restart_requested",
        }

    result = ui.restart_ovrtx_worker_result(restart)

    assert calls == 1
    assert result == {
        "status": "restarted",
        "active_session_count": 1,
        "restarted_worker_count": 1,
        "end_reason": "worker_restart_requested",
    }


def test_recovery_error_rejects_unconfirmed_teardown() -> None:
    assert ui._recovery_error(
        {"status": "teardown_unconfirmed"}, "requested", "teardown failed"
    ) == "teardown failed"
    assert ui._recovery_error(
        {"status": "requested"}, "requested", "teardown failed"
    ) == ""


def test_runtime_start_pending_true_while_session_coming_up() -> None:
    # First-run shader compile surfaces as the "compiling" session status.
    assert ui.runtime_start_pending(lambda: {"status": "compiling"}) is True
    assert ui.runtime_start_pending(lambda: {"status": "starting"}) is True
    assert ui.runtime_start_pending(lambda: {"status": "loading"}) is True


def test_runtime_start_pending_false_when_live_stopped_or_failed() -> None:
    assert ui.runtime_start_pending(lambda: {"status": "live"}) is False
    assert ui.runtime_start_pending(lambda: {"status": "stopped"}) is False
    assert ui.runtime_start_pending(lambda: {"status": "failed"}) is False
    assert ui.runtime_start_pending(lambda: {}) is False


def test_runtime_start_pending_checks_every_active_session() -> None:
    assert ui.runtime_start_pending(
        lambda: {
            "status": "live",
            "sessions": ({"status": "live"}, {"status": "compiling"}),
        }
    ) is True


def test_viewport_session_status_projects_first_active_session() -> None:
    result = ui.viewport_session_status(
        lambda: {
            "status": "available",
            "active_session_count": 1,
            "sessions": [
                {
                    "status": "live",
                    "label": "Live",
                    "hint": "",
                    "logs": {"log_dir": "/tmp/ov-blender-example/logs"},
                }
            ],
        }
    )

    assert result["status"] == "live"
    assert result["label"] == "Live"
    assert result["active_session_count"] == 1
    assert result["logs"] == {"log_dir": "/tmp/ov-blender-example/logs"}


def test_viewport_session_status_falls_back_to_stopped() -> None:
    result = ui.viewport_session_status(
        lambda: {"status": "available", "active_session_count": 0, "sessions": []}
    )

    assert result["status"] == "stopped"
    assert result["label"] == "Stopped"
    assert result["active_session_count"] == 0
    # Default log routing is stdout: no file logging, no log directory
    # (env overrides opt back into files for validation lanes).
    assert result["logs"]["status"] in {"stdout", "file"}


def test_open_log_folder_result_creates_directory_and_uses_injected_opener(tmp_path: Path) -> None:
    opened: list[Path] = []

    result = ui.open_log_folder_result(
        {"log_dir": str(tmp_path / "logs")},
        opener=opened.append,
    )

    assert result == {"status": "opened", "log_dir": str(tmp_path / "logs")}
    assert opened == [tmp_path / "logs"]
    assert (tmp_path / "logs").is_dir()


def test_open_log_folder_result_fails_cleanly_without_file_logging() -> None:
    """Stdout routing has no log folder: the operator explains how to opt
    into file logging instead of opening/creating a stale directory."""

    result = ui.open_log_folder_result({"status": "stdout", "log_dir": ""})

    assert result["status"] == "failed"
    assert result["log_dir"] == ""
    assert "OV_BLENDER_EXAMPLE_WORKER_LOG" in result["error"]


def _stub_version_sources(monkeypatch, *, bl_info, build_label: str) -> None:
    """Point `addon_version_label` at controlled `bl_info` / build metadata.

    `addon_version_label` reads `bl_info` off the already-imported
    `ovrtx_blender_example` package module and late-imports
    `addon_build_label` from the same module, so both are patched there
    rather than on `ui`.
    """

    package = sys.modules["ovrtx_blender_example"]
    monkeypatch.setattr(package, "bl_info", bl_info, raising=False)
    monkeypatch.setattr(package, "addon_build_label", lambda *_a, **_k: build_label, raising=False)


def test_addon_version_label_renders_name_and_version(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_version_sources(
        monkeypatch,
        bl_info={"name": "ovrtx Blender Example", "version": (0, 1, 0)},
        build_label="",
    )

    assert ui.addon_version_label() == "ovrtx Blender Example v0.1.0"


def test_addon_version_label_appends_build_label(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_version_sources(
        monkeypatch,
        bl_info={"name": "ovrtx Blender Example", "version": (0, 2, 0)},
        build_label="Build 1842 (abc1234)",
    )

    assert ui.addon_version_label() == "ovrtx Blender Example v0.2.0 · Build 1842 (abc1234)"


@pytest.mark.parametrize("version", [None, (), ""], ids=["missing", "empty-tuple", "empty-string"])
def test_addon_version_label_omits_dangling_v_without_a_version(
    monkeypatch: pytest.MonkeyPatch, version: object
) -> None:
    """A missing/empty `bl_info["version"]` must not render a trailing "v"."""

    bl_info: dict[str, object] = {"name": "ovrtx Blender Example"}
    if version is not None:
        bl_info["version"] = version
    _stub_version_sources(monkeypatch, bl_info=bl_info, build_label="")

    assert ui.addon_version_label() == "ovrtx Blender Example"


def test_addon_version_label_keeps_build_label_without_a_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_version_sources(
        monkeypatch,
        bl_info={"name": "ovrtx Blender Example", "version": ()},
        build_label="Build 1842 (abc1234)",
    )

    assert ui.addon_version_label() == "ovrtx Blender Example · Build 1842 (abc1234)"


def test_addon_version_label_falls_back_to_product_name_without_bl_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_version_sources(monkeypatch, bl_info={}, build_label="")

    assert ui.addon_version_label() == "ovrtx"


def test_addon_version_label_falls_back_to_product_name_without_an_addon_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_version_sources(monkeypatch, bl_info={"version": (0, 1, 0)}, build_label="")

    assert ui.addon_version_label() == "ovrtx v0.1.0"

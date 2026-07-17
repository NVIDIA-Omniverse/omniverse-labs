# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest

import blender_test_support


def test_explicit_blender_command_takes_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured = "/opt/Blender 5.1/blender"
    calls = []
    monkeypatch.setenv("BLENDER_COMMAND", configured)
    monkeypatch.setattr(
        blender_test_support.shutil,
        "which",
        lambda command: calls.append(command) or configured,
    )

    assert blender_test_support.blender_executable() == Path(configured).absolute()
    assert calls == [configured]


def test_invalid_explicit_blender_command_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BLENDER_COMMAND", "missing-blender")
    monkeypatch.setattr(blender_test_support.shutil, "which", lambda _command: None)

    with pytest.raises(ValueError, match="BLENDER_COMMAND.*missing-blender"):
        blender_test_support.blender_executable()


def test_blender_is_discovered_from_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BLENDER_COMMAND", raising=False)
    monkeypatch.setattr(
        blender_test_support.shutil,
        "which",
        lambda command: "/usr/local/bin/blender" if command == "blender" else None,
    )

    assert blender_test_support.blender_executable() == Path(
        "/usr/local/bin/blender"
    ).absolute()


def test_native_install_is_used_after_path(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    fallback = "/snap/blender/current/blender"
    monkeypatch.delenv("BLENDER_COMMAND", raising=False)
    monkeypatch.setattr(blender_test_support.sys, "platform", "linux")
    monkeypatch.setattr(
        blender_test_support.shutil,
        "which",
        lambda command: calls.append(command) or (fallback if command == fallback else None),
    )

    assert blender_test_support.blender_executable() == Path(fallback).absolute()
    assert calls == ["blender", fallback]


def test_absent_blender_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BLENDER_COMMAND", raising=False)
    monkeypatch.setattr(blender_test_support.sys, "platform", "unknown")
    monkeypatch.setattr(blender_test_support.shutil, "which", lambda _command: None)

    assert blender_test_support.blender_executable() is None


def test_relative_path_result_is_made_absolute(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BLENDER_COMMAND", raising=False)
    monkeypatch.setattr(
        blender_test_support.shutil,
        "which",
        lambda command: "relative/blender" if command == "blender" else None,
    )

    assert blender_test_support.blender_executable() == (
        Path.cwd() / "relative/blender"
    )

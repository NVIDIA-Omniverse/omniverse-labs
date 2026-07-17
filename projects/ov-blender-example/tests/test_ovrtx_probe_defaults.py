# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import ovrtx_probe_support  # noqa: E402


PROBES = tuple(
    importlib.import_module(name)
    for name in (
        "run_blender_orthographic_view_parity_probe",
        "run_ovrtx_color_presentation_probe",
        "run_ovrtx_light_value_probe",
        "run_ovrtx_live_transform_probe",
        "run_ovrtx_material_value_probe",
        "run_ovrtx_operator_seam_probe",
        "run_ovrtx_orthographic_camera_probe",
        "run_ovrtx_primvars_st_probe",
        "run_ovrtx_world_dome_probe",
    )
)
RUNTIME_PROBES = PROBES[2:]
BLENDER_ARGUMENT_PROBES = tuple(
    probe for index, probe in enumerate(PROBES) if index not in {3, 5}
)


class _ParserCaptured(Exception):
    pass


def _defaults(module: object) -> dict[str, object]:
    captured: dict[str, object] = {}

    def capture(parser: argparse.ArgumentParser, *_args, **_kwargs) -> None:
        captured.update({action.dest: action.default for action in parser._actions})
        raise _ParserCaptured

    with mock.patch.object(argparse.ArgumentParser, "parse_args", capture):
        try:
            parser = getattr(module, "_parse_args", None)
            (parser or getattr(module, "main"))([])
        except _ParserCaptured:
            pass
    return captured


def test_probe_defaults_use_shared_owners(monkeypatch) -> None:
    monkeypatch.delenv("OV_BLENDER_EXAMPLE_WORKER_COMMAND", raising=False)
    monkeypatch.delenv("OV_BLENDER_EXAMPLE_NATIVE_CLIENT_PATH", raising=False)
    monkeypatch.setattr(
        ovrtx_probe_support.bundled_runtime,
        "defaults",
        lambda: SimpleNamespace(
            worker_command="installed-worker",
            native_client_path="installed-native",
        ),
    )

    for probe in PROBES:
        assert probe.BLENDER_COMMAND == ovrtx_probe_support.BLENDER_COMMAND
    for probe in BLENDER_ARGUMENT_PROBES:
        assert "blender_command" in _defaults(probe)
    for probe in RUNTIME_PROBES:
        defaults = _defaults(probe)
        assert defaults["worker_command"] == "installed-worker"
        assert str(defaults["native_client_path"]) == "installed-native"

    live = importlib.import_module("run_ovrtx_live_transform_probe")
    operator = importlib.import_module("run_ovrtx_operator_seam_probe")
    assert _defaults(live)["manifest"] == ovrtx_probe_support.DEFAULT_FIXTURE_MANIFEST
    assert _defaults(operator)["manifest"] == ovrtx_probe_support.DEFAULT_FIXTURE_MANIFEST


def test_probe_environment_overrides_runtime_defaults(monkeypatch) -> None:
    monkeypatch.setenv("BLENDER_COMMAND", "configured-blender")
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_WORKER_COMMAND", "configured-worker")
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_NATIVE_CLIENT_PATH", "configured-native")

    for probe in BLENDER_ARGUMENT_PROBES:
        assert _defaults(probe)["blender_command"] == "configured-blender"
    for probe in RUNTIME_PROBES:
        defaults = _defaults(probe)
        assert defaults["worker_command"] == "configured-worker"
        assert str(defaults["native_client_path"]) == "configured-native"

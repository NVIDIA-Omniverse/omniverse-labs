# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Portable launcher preflights host graphics-driver dependencies."""
import importlib.util
import locale
import mimetypes
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "ovui_package/portable_launch.py"


def load_launcher():
    spec = importlib.util.spec_from_file_location("portable_launch", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "platform,loader_name,library,expected_options",
    [
        ("win32", "WinDLL", "vulkan-1.dll", {"winmode": 0x00000800}),
        ("linux", "CDLL", "libcuda.so.1", {}),
    ],
)
def test_driver_preflight_loads_only_the_declared_host_library(
        monkeypatch, platform, loader_name, library, expected_options):
    launcher = load_launcher()
    calls = []

    def load(name, **options):
        calls.append((name, options))

    monkeypatch.setattr(launcher.sys, "platform", platform)
    monkeypatch.setattr(launcher.ctypes, loader_name, load, raising=False)

    launcher.preflight_graphics_driver()

    assert calls == [(library, expected_options)]


@pytest.mark.parametrize(
    "platform,loader_name,library,driver",
    [
        ("win32", "WinDLL", "vulkan-1.dll", "graphics driver"),
        ("linux", "CDLL", "libcuda.so.1", "NVIDIA driver"),
    ],
)
def test_driver_preflight_reports_an_actionable_missing_dependency(
        monkeypatch, platform, loader_name, library, driver):
    launcher = load_launcher()
    missing = OSError(f"could not load {library}")

    def fail(*_args, **_kwargs):
        raise missing

    monkeypatch.setattr(launcher.sys, "platform", platform)
    monkeypatch.setattr(launcher.ctypes, loader_name, fail, raising=False)

    with pytest.raises(RuntimeError) as raised:
        launcher.preflight_graphics_driver()

    message = str(raised.value)
    assert library in message
    assert driver in message
    assert "Install or update" in message
    assert raised.value.__cause__ is missing


@pytest.mark.parametrize("platform", ["linux", pytest.param("win32", marks=pytest.mark.skipif(
    sys.platform != "win32", reason="Windows UTF-8 CRT locale"))])
def test_main_configures_character_locale_before_native_imports(monkeypatch, tmp_path, platform):
    launcher = load_launcher()
    monkeypatch.syspath_prepend(str(SCRIPT.parents[2] / "python"))
    for name in ("knownfiles", "_db", "inited", "suffix_map", "types_map",
                 "encodings_map", "common_types"):
        monkeypatch.setattr(mimetypes, name, getattr(mimetypes, name))
    bundle = tmp_path / "bundle"
    (bundle / "python/Lib").mkdir(parents=True)
    events = []

    monkeypatch.setattr(launcher, "__file__", str(bundle / "portable_launch.py"))
    monkeypatch.setattr(launcher, "sys", SimpleNamespace(executable=str(bundle / "python/python"),
        flags=SimpleNamespace(isolated=True), path=[str(bundle / "python/Lib")],
        platform=platform, argv=["portable_launch.py"]))
    monkeypatch.setattr(launcher.os, "environ", dict(launcher.os.environ))
    categories = (locale.LC_NUMERIC, locale.LC_TIME, locale.LC_COLLATE, locale.LC_MONETARY)
    unchanged = [locale.setlocale(category) for category in categories]

    def preflight():
        character_locale = locale.setlocale(locale.LC_CTYPE)
        assert character_locale.lower().endswith(".utf8") if platform == "win32" else character_locale == "C"
        assert [locale.setlocale(category) for category in categories] == unchanged
        events.append("preflight")

    monkeypatch.setattr(launcher, "preflight_graphics_driver", preflight)
    monkeypatch.setattr(
        launcher.runpy,
        "run_module",
        lambda name, **_kwargs: events.append("import") if name == "ovd_ui" else None,
    )

    original = locale.setlocale(locale.LC_CTYPE)
    try:
        locale.setlocale(locale.LC_CTYPE, "C")
        launcher.main()
    finally:
        locale.setlocale(locale.LC_CTYPE, original)

    assert events == ["preflight", "import"]

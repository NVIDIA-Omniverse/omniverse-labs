# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The CI-only software renderer must never become a package dependency."""
import importlib.util
import os
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("smoke_ovui_gui", Path(__file__).resolve().parents[1] / "smoke_ovui_gui.py")
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


def test_ci_graphics_override_loads_only_explicit_library(monkeypatch, tmp_path):
    assert hasattr(smoke, "ci_graphics"), "CI renderer setup is missing"
    library = tmp_path / "CI Mesa/opengl32.dll"
    library.parent.mkdir()
    library.touch()
    loaded = object()
    def load(path, *, winmode):
        assert path == str(library.resolve()) and winmode == 8
        return loaded
    monkeypatch.setattr(smoke.ctypes, "WinDLL", load, raising=False)
    monkeypatch.setenv("GLCONTEXT_WIN_LIBGL", "old developer override")
    assert smoke.ci_graphics(library) is loaded
    assert os.environ["GLCONTEXT_WIN_LIBGL"] == str(library.resolve())


def test_missing_ci_renderer_fails_instead_of_silently_using_host(tmp_path):
    assert hasattr(smoke, "ci_graphics"), "CI renderer setup is missing"
    with pytest.raises(FileNotFoundError):
        smoke.ci_graphics(tmp_path / "missing.dll")

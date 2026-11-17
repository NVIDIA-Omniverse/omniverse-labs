# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit fixture roots resolve the same native writers on every platform."""
from pathlib import Path

import pytest

from native_fixtures import executable


def test_fixture_root_accepts_platform_suffix_and_individual_override(tmp_path, monkeypatch):
    root = tmp_path / "fixtures"; root.mkdir()
    geometry = root / "pvd_geometry_fixture.exe"; geometry.touch()
    override = tmp_path / "custom-writer"; override.touch()
    monkeypatch.delenv("PVD_GEOMETRY_FIXTURE", raising=False)
    monkeypatch.setenv("PVD_FIXTURE_ROOT", str(root))
    assert executable("pvd_geometry_fixture", "PVD_GEOMETRY_FIXTURE") == geometry
    monkeypatch.setenv("PVD_GEOMETRY_FIXTURE", str(override))
    assert executable("pvd_geometry_fixture", "PVD_GEOMETRY_FIXTURE") == override


def test_missing_configured_fixture_fails_without_building_or_guessing(tmp_path, monkeypatch):
    monkeypatch.delenv("PVD_INGESTION_FIXTURE", raising=False)
    monkeypatch.setenv("PVD_FIXTURE_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="pvd_ingestion_fixture"):
        executable("pvd_ingestion_fixture", "PVD_INGESTION_FIXTURE")

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Packed-image materialization for external-runtime texture reads."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import texture_materialization  # noqa: E402


def _packed_image(name: str, data: bytes, filepath: str = "//textures/tex.png"):
    return SimpleNamespace(
        name=name,
        filepath=filepath,
        library=None,
        packed_file=SimpleNamespace(data=data),
    )


def test_packed_image_materializes_once_content_addressed(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv(
        "OV_BLENDER_EXAMPLE_AUTHORING_WORK_DIR", str(tmp_path / "authored-scenes")
    )
    image = _packed_image("Robot_BaseColor.png", b"png-bytes")

    first = texture_materialization.materialized_image_path(image)
    second = texture_materialization.materialized_image_path(image)

    assert first and first == second
    path = Path(first)
    assert path.is_file()
    assert path.read_bytes() == b"png-bytes"
    assert path.suffix == ".png"
    assert path.parent == tmp_path / "texture-cache"
    # Content-addressed: changed bytes land in a new file (no staleness).
    changed = texture_materialization.materialized_image_path(
        _packed_image("Robot_BaseColor.png", b"other-bytes")
    )
    assert changed != first


def test_images_without_readable_content_degrade(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(
        "OV_BLENDER_EXAMPLE_AUTHORING_WORK_DIR", str(tmp_path / "authored-scenes")
    )
    no_data = SimpleNamespace(
        name="Generated",
        filepath="",
        library=None,
        packed_file=SimpleNamespace(data=b""),
    )
    missing_on_disk = SimpleNamespace(
        name="Missing",
        filepath=str(tmp_path / "nope.png"),
        library=None,
        packed_file=None,
    )
    assert texture_materialization.materialized_image_path(None) == ""
    assert texture_materialization.materialized_image_path(no_data) == ""
    assert texture_materialization.materialized_image_path(missing_on_disk) == ""


def test_on_disk_image_resolves_in_place(tmp_path: Path) -> None:
    texture = tmp_path / "wood.jpg"
    texture.write_bytes(b"jpg")
    image = SimpleNamespace(
        name="Wood",
        filepath=str(texture),
        library=None,
        packed_file=None,
    )
    resolved = texture_materialization.materialized_image_path(image)
    assert Path(resolved) == texture.resolve()

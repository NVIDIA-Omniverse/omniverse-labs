# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Packed-image materialization for external-runtime texture reads."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
import pytest

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


def test_packed_exr_named_png_keeps_hdr_encoding(tmp_path, monkeypatch):
    from ovrtx_blender_example import materialx_openpbr_conversion
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_AUTHORING_WORK_DIR", str(tmp_path / "authored-scenes"))
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR", str(tmp_path / "viewport"))
    payload = b"\x76\x2f\x31\x01" + b"exr-payload"
    image = _packed_image("Sky.png", payload, "//misleading.png")
    for export in (texture_materialization.materialized_image_path, materialx_openpbr_conversion._materialized_image_path):
        path = Path(export(image))
        assert path.suffix == ".exr"
        assert path.read_bytes() == payload
        assert not path.with_suffix('.exr.tmp').exists()


def test_stock_packed_filename_repair_uses_only_exact_exported_bytes(tmp_path):
    from pxr import Sdf, Usd
    from ovrtx_blender_example.scene_generation import _repair_packed_texture_assets
    root = tmp_path / 'stock.usda'
    stage = Usd.Stage.CreateNew(str(root))
    attr = stage.DefinePrim('/Texture', 'Shader').CreateAttribute('inputs:file', Sdf.ValueTypeNames.Asset)
    attr.Set(Sdf.AssetPath('./textures/original.png'))
    textures = tmp_path / 'textures'
    textures.mkdir()
    (textures / 'Generated Image.png').write_bytes(b'packed-content')
    (textures / 'other.png').write_bytes(b'different-content')
    image = _packed_image('Generated Image', b'packed-content', '//original.png')
    _repair_packed_texture_assets(stage, root, [image])
    assert attr.Get().path == './textures/Generated Image.png'
    assert Path(attr.Get().resolvedPath).read_bytes() == b'packed-content'
    assert image.filepath == '//original.png'


@pytest.mark.parametrize('payloads', [(b'missing',), (b'one', b'two'), (b'',)])
def test_stock_packed_filename_repair_rejects_missing_or_ambiguous_bytes(tmp_path, payloads):
    from pxr import Sdf, Usd
    from ovrtx_blender_example.scene_generation import _repair_packed_texture_assets, SceneGenerationError
    root = tmp_path / 'stock.usda'
    stage = Usd.Stage.CreateNew(str(root))
    attr = stage.DefinePrim('/Texture', 'Shader').CreateAttribute('inputs:file', Sdf.ValueTypeNames.Asset)
    attr.Set(Sdf.AssetPath('./textures/original.png'))
    (tmp_path / 'textures').mkdir()
    (tmp_path / 'textures' / 'one.png').write_bytes(b'one')
    images = [_packed_image('Image', value, '//original.png') for value in payloads]
    with pytest.raises(SceneGenerationError, match='unambiguous'):
        _repair_packed_texture_assets(stage, root, images)
    assert attr.Get().path == './textures/original.png'


def test_stock_packed_repair_preserves_resolver_tiles_and_unrelated_missing_assets(tmp_path):
    from pxr import Sdf, Usd
    from ovrtx_blender_example.scene_generation import _repair_packed_texture_assets
    root = tmp_path / 'stock.usda'
    stage = Usd.Stage.CreateNew(str(root))
    prim = stage.DefinePrim('/Texture', 'Shader')
    paths = ['./textures/diffuse.<UDIM>.png', './textures/unrelated.png', '../outside.png']
    attrs = [prim.CreateAttribute('asset' + str(i), Sdf.ValueTypeNames.Asset) for i in range(len(paths))]
    for attr, value in zip(attrs, paths):
        attr.Set(Sdf.AssetPath(value))
    _repair_packed_texture_assets(stage, root, [_packed_image('Tiled', b'packed', '//diffuse.<UDIM>.png')])
    assert [attr.Get().path for attr in attrs] == paths

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import hashlib
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "fixtures"))

from fixture_manifest import (  # noqa: E402
    fixture_runtime_content_sha256,
    load_catalog,
    load_manifest,
    render_fixture,
    shared_stage_runtime_defaults,
)


def test_load_catalog_rejects_duplicate_fixture_ids(tmp_path: Path) -> None:
    for name in ("one", "two"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "spec.json").write_text('{"id": "duplicate"}', encoding="utf-8")

    with pytest.raises(ValueError, match="declared more than once"):
        load_catalog(tmp_path)


def _manifest(fixture: dict[str, object]) -> dict[str, object]:
    return {"_manifest_base_path": str(ROOT / "tests"), "fixtures": [fixture]}


def _fixture(**overrides: object) -> dict[str, object]:
    fixture: dict[str, object] = {
        "id": "demo_stair_drop_1280x720",
        "capabilities": ["ovrtx", "ovphysx"],
        "fixture_usd_path": "fixtures/data/demo_stair_drop_1280x720/fixture/stair_drop_ovrtx_ovphysx.usda",
        "fixture_usd_sha256": "abc123",
        "fixture_content_sha256": "content123",
        "camera_prim_path": "/World/Camera",
        "render_product_prim_path": "/Render/OmniverseKit/HydraTextures/ViewportTexture0",
    }
    fixture.update(overrides)
    return fixture


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _committed_fixture(
    tmp_path: Path,
    *,
    direct_content: bytes = b"#usda 1.0\n",
    nested_content: bytes = b"texture",
) -> tuple[dict[str, object], Path, Path]:
    direct = tmp_path / "tests" / "fixtures" / "data" / "demo" / "stage.usda"
    nested = direct.parent / "textures" / "albedo.png"
    direct.parent.mkdir(parents=True)
    nested.parent.mkdir(parents=True)
    direct.write_bytes(direct_content)
    nested.write_bytes(nested_content)
    direct_path = "fixtures/data/demo/stage.usda"
    nested_path = "fixtures/data/demo/textures/albedo.png"
    runtime_files = [
        {"path": direct_path, "sha256": _sha256(direct_content)},
        {"path": nested_path, "sha256": _sha256(nested_content)},
    ]
    fixture = _fixture(
        fixture_usd_path=direct_path,
        fixture_usd_sha256=_sha256(direct_content),
        fixture_content_sha256=fixture_runtime_content_sha256(runtime_files),
        runtime_files=runtime_files,
    )
    manifest = {
        "_manifest_base_path": str(tmp_path / "tests"),
        "fixtures": [fixture],
    }
    return manifest, direct, nested


def _refresh_runtime_identity(manifest: dict[str, object]) -> None:
    fixture = manifest["fixtures"][0]
    runtime_files = fixture["runtime_files"]
    for item in runtime_files:
        item["sha256"] = _sha256(
            (Path(manifest["_manifest_base_path"]) / item["path"]).read_bytes()
        )
    fixture["fixture_usd_sha256"] = runtime_files[0]["sha256"]
    fixture["fixture_content_sha256"] = fixture_runtime_content_sha256(runtime_files)


def test_render_fixture_requires_explicit_fixture_id() -> None:
    with pytest.raises(ValueError, match="--fixture-id is required"):
        render_fixture(_manifest(_fixture()), "")


def test_render_fixture_requires_capabilities() -> None:
    fixture = _fixture()
    fixture.pop("capabilities")

    with pytest.raises(ValueError, match="capabilities must be a non-empty list"):
        render_fixture(_manifest(fixture), "demo_stair_drop_1280x720")


def test_render_fixture_requires_ovrtx_capability() -> None:
    with pytest.raises(ValueError, match="does not declare ovrtx capability"):
        render_fixture(_manifest(_fixture(capabilities=["ovphysx"])), "demo_stair_drop_1280x720")


def test_render_fixture_returns_capability_record() -> None:
    record = render_fixture(_manifest(_fixture()), "demo_stair_drop_1280x720")

    assert record["id"] == "demo_stair_drop_1280x720"
    assert record["capabilities"] == ("ovrtx", "ovphysx")
    assert record["fixture_usd_sha256"] == "abc123"
    assert record["fixture_content_sha256"] == "content123"
    assert record["camera_prim_path"] == "/World/Camera"
    assert record["resolution"] == {"width": 1280, "height": 720}


def test_load_manifest_resolves_assets_from_tests_fixture_root(tmp_path: Path) -> None:
    manifest_path = tmp_path / "tests" / "fixtures" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        '{"fixtures": [{"id": "demo_stair_drop_1280x720", "capabilities": ["ovrtx"],'
        ' "fixture_usd_path": "fixtures/data/demo/fixture/stage.usda"}]}\n',
        encoding="utf-8",
    )

    record = render_fixture(load_manifest(manifest_path), "demo_stair_drop_1280x720")

    expected = tmp_path / "tests" / "fixtures" / "data" / "demo" / "fixture" / "stage.usda"
    assert record["fixture_usd_path"] == str(expected)


def test_render_fixture_preserves_target_resolution() -> None:
    record = render_fixture(
        _manifest(_fixture(target_resolution={"width": 1200, "height": 800})),
        "demo_stair_drop_1280x720",
    )

    assert record["resolution"] == {"width": 1200, "height": 800}


def test_render_fixture_rejects_removed_usd_fields() -> None:
    fixture = _fixture()
    fixture["usd_path"] = fixture.pop("fixture_usd_path")
    fixture["usd_sha256"] = fixture.pop("fixture_usd_sha256")

    with pytest.raises(ValueError, match="removed USD schema fields"):
        render_fixture(_manifest(fixture), "demo_stair_drop_1280x720")


def test_render_fixture_rejects_invalid_target_resolution() -> None:
    with pytest.raises(ValueError, match="target_resolution width and height must be positive integers"):
        render_fixture(
            _manifest(_fixture(target_resolution={"width": 0, "height": 800})),
            "demo_stair_drop_1280x720",
        )


def test_render_fixture_preserves_runtime_defaults() -> None:
    record = render_fixture(
        _manifest(
            _fixture(
                runtime_defaults={
                    "shared_stage_composition": {
                        "enabled": True,
                        "composition_max_steps": 300,
                    }
                }
            )
        ),
        "demo_stair_drop_1280x720",
    )

    assert record["runtime_defaults"]["shared_stage_composition"]["enabled"] is True
    assert record["runtime_defaults"]["shared_stage_composition"]["composition_max_steps"] == 300


def test_shared_stage_runtime_defaults_returns_manifest_defaults() -> None:
    fixture = {
        "runtime_defaults": {
            "shared_stage_composition": {
                "enabled": True,
                "composition_max_steps": 300,
            }
        }
    }

    defaults = shared_stage_runtime_defaults(fixture)

    assert defaults == {"enabled": True, "composition_max_steps": 300}


def test_render_fixture_verifies_complete_runtime_file_identity(tmp_path: Path) -> None:
    manifest, direct, nested = _committed_fixture(tmp_path)

    record = render_fixture(manifest, "demo_stair_drop_1280x720")

    assert record["runtime_files"] == (
        {
            "path": str(direct),
            "manifest_path": "fixtures/data/demo/stage.usda",
            "sha256": _sha256(direct.read_bytes()),
        },
        {
            "path": str(nested),
            "manifest_path": "fixtures/data/demo/textures/albedo.png",
            "sha256": _sha256(nested.read_bytes()),
        },
    )


def test_render_fixture_rejects_missing_nested_usda_reference(tmp_path: Path) -> None:
    manifest, direct, _nested = _committed_fixture(
        tmp_path,
        direct_content=b'#usda 1.0\nsubLayers = [@layers/nested.usda@]\n',
    )
    nested_usda = direct.parent / "layers" / "nested.usda"
    nested_usda.parent.mkdir()
    nested_usda.write_bytes(
        b'#usda 1.0\ndef Scope "Nested" { asset texture = @missing.png@ }\n'
    )
    fixture = manifest["fixtures"][0]
    fixture["runtime_files"].append(
        {
            "path": "fixtures/data/demo/layers/nested.usda",
            "sha256": _sha256(nested_usda.read_bytes()),
        }
    )
    _refresh_runtime_identity(manifest)

    with pytest.raises(FileNotFoundError, match="fixture asset reference is missing"):
        render_fixture(manifest, "demo_stair_drop_1280x720")


def test_render_fixture_rejects_missing_usda_reference(tmp_path: Path) -> None:
    manifest, direct, _nested = _committed_fixture(
        tmp_path,
        direct_content=b'#usda 1.0\nsubLayers = [@layers/missing.usda@]\n',
    )
    _refresh_runtime_identity(manifest)

    with pytest.raises(FileNotFoundError, match="fixture asset reference is missing"):
        render_fixture(manifest, "demo_stair_drop_1280x720")


def test_render_fixture_rejects_unlisted_usda_reference(tmp_path: Path) -> None:
    manifest, direct, _nested = _committed_fixture(
        tmp_path,
        direct_content=b'#usda 1.0\nsubLayers = [@layers/unlisted.usda@]\n',
    )
    unlisted = direct.parent / "layers" / "unlisted.usda"
    unlisted.parent.mkdir()
    unlisted.write_bytes(b"#usda 1.0\n")
    _refresh_runtime_identity(manifest)

    with pytest.raises(ValueError, match="not declared in runtime_files"):
        render_fixture(manifest, "demo_stair_drop_1280x720")


def test_render_fixture_rejects_usda_reference_outside_runtime_root(tmp_path: Path) -> None:
    manifest, direct, _nested = _committed_fixture(
        tmp_path,
        direct_content=b'#usda 1.0\nsubLayers = [@../../outside.usda@]\n',
    )
    outside = direct.parent.parent / "outside.usda"
    outside.write_bytes(b"#usda 1.0\n")
    _refresh_runtime_identity(manifest)

    with pytest.raises(ValueError, match="escapes runtime root"):
        render_fixture(manifest, "demo_stair_drop_1280x720")


def test_render_fixture_does_not_parse_binary_usdc_references(tmp_path: Path) -> None:
    manifest, direct, _nested = _committed_fixture(
        tmp_path,
        direct_content=b'#usda 1.0\nsubLayers = [@layers/binary.usdc@]\n',
    )
    binary = direct.parent / "layers" / "binary.usdc"
    binary.parent.mkdir()
    binary.write_bytes(b"PXR-USDC\x00@../../outside.png@")
    fixture = manifest["fixtures"][0]
    fixture["runtime_files"].append(
        {
            "path": "fixtures/data/demo/layers/binary.usdc",
            "sha256": _sha256(binary.read_bytes()),
        }
    )
    _refresh_runtime_identity(manifest)

    render_fixture(manifest, "demo_stair_drop_1280x720")


@pytest.mark.parametrize("missing", ["direct", "nested"])
def test_render_fixture_rejects_missing_runtime_file(
    tmp_path: Path,
    missing: str,
) -> None:
    manifest, direct, nested = _committed_fixture(tmp_path)
    {"direct": direct, "nested": nested}[missing].unlink()

    with pytest.raises(FileNotFoundError, match="fixture runtime file is missing"):
        render_fixture(manifest, "demo_stair_drop_1280x720")


@pytest.mark.parametrize("pointer", ["direct", "nested"])
def test_render_fixture_rejects_unmaterialized_lfs_pointer(
    tmp_path: Path,
    pointer: str,
) -> None:
    manifest, direct, nested = _committed_fixture(tmp_path)
    lfs_pointer = (
        b"version https://git-lfs.github.com/spec/v1\n"
        b"oid sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\n"
        b"size 123\n"
    )
    {"direct": direct, "nested": nested}[pointer].write_bytes(lfs_pointer)

    with pytest.raises(ValueError, match="Git LFS pointer"):
        render_fixture(manifest, "demo_stair_drop_1280x720")


def test_render_fixture_rejects_runtime_file_digest_mismatch(tmp_path: Path) -> None:
    manifest, _direct, nested = _committed_fixture(tmp_path)
    nested.write_bytes(b"changed")

    with pytest.raises(ValueError, match="fixture runtime file digest mismatch"):
        render_fixture(manifest, "demo_stair_drop_1280x720")


def test_render_fixture_requires_direct_file_in_runtime_identity(tmp_path: Path) -> None:
    manifest, _direct, _nested = _committed_fixture(tmp_path)
    fixture = manifest["fixtures"][0]
    fixture["runtime_files"] = fixture["runtime_files"][1:]

    with pytest.raises(ValueError, match="direct USD must appear in runtime_files"):
        render_fixture(manifest, "demo_stair_drop_1280x720")


def test_render_fixture_requires_matching_direct_file_digest(tmp_path: Path) -> None:
    manifest, _direct, _nested = _committed_fixture(tmp_path)
    fixture = manifest["fixtures"][0]
    fixture["fixture_usd_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="direct USD digest must match"):
        render_fixture(manifest, "demo_stair_drop_1280x720")


def test_render_fixture_selects_exact_platform_identity(
    tmp_path: Path, monkeypatch
) -> None:
    manifest, direct, _nested = _committed_fixture(tmp_path)
    fixture = manifest["fixtures"][0]
    direct.write_bytes(b"#usda 1.0\nplatform")
    runtime_files = [dict(item) for item in fixture["runtime_files"]]
    runtime_files[0]["sha256"] = _sha256(direct.read_bytes())
    fixture["platform_identities"] = {
        "test-platform": {
            "fixture_usd_sha256": runtime_files[0]["sha256"],
            "fixture_content_sha256": fixture_runtime_content_sha256(runtime_files),
        }
    }
    monkeypatch.setattr("fixture_manifest.sys.platform", "test-platform")

    record = render_fixture(manifest, "demo_stair_drop_1280x720")

    assert record["fixture_usd_sha256"] == runtime_files[0]["sha256"]
    assert record["fixture_content_sha256"] == fixture_runtime_content_sha256(runtime_files)


def test_runtime_content_identity_is_stable_across_manifest_order() -> None:
    runtime_files = [
        {"path": "fixtures/data/demo/stage.usda", "sha256": "a" * 64},
        {"path": "fixtures/data/demo/textures/albedo.png", "sha256": "b" * 64},
    ]

    expected = "8dadb3c28ab7299adbefe13c9b930d04f3bfff70b54fb346d2e2fcaa15ffe158"
    assert fixture_runtime_content_sha256(runtime_files) == expected
    assert fixture_runtime_content_sha256(list(reversed(runtime_files))) == expected


def test_runtime_content_identity_binds_manifest_relative_path() -> None:
    original = [{"path": "fixtures/data/demo/stage.usda", "sha256": "a" * 64}]
    renamed = [{"path": "fixtures/data/demo/renamed.usda", "sha256": "a" * 64}]

    assert fixture_runtime_content_sha256(original) != fixture_runtime_content_sha256(renamed)


def test_render_fixture_rejects_runtime_content_identity_mismatch(tmp_path: Path) -> None:
    manifest, _direct, _nested = _committed_fixture(tmp_path)
    manifest["fixtures"][0]["fixture_content_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="fixture_content_sha256 mismatch"):
        render_fixture(manifest, "demo_stair_drop_1280x720")


def test_render_fixture_requires_declared_runtime_content_identity(tmp_path: Path) -> None:
    manifest, _direct, _nested = _committed_fixture(tmp_path)
    manifest["fixtures"][0].pop("fixture_content_sha256")

    with pytest.raises(ValueError, match="fixture_content_sha256 must be 64 lowercase"):
        render_fixture(manifest, "demo_stair_drop_1280x720")

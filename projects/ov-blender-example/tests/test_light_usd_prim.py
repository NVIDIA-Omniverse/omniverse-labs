# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from types import SimpleNamespace
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import light_usd_prim as light  # noqa: E402
from ovrtx_blender_example import usd_paths as usd_paths  # noqa: E402


class _FakePrim:
    def __init__(self, path: str, type_name: str, *, attributes=None) -> None:
        self.path = path
        self.type_name = type_name
        if attributes is not None:
            self.attributes = attributes


class _FakeObject(dict):
    def __init__(self, name: str, *, parent=None) -> None:
        super().__init__()
        self.name = self.name_full = name
        self.type = "LIGHT"
        self.parent = parent


def test_light_index_freezes_family_and_authored_form() -> None:
    index = light._light_prim_index_from_prims(
        (
            _FakePrim("/World/Point", "SphereLight", attributes=()),
            _FakePrim("/World/Spot", "SphereLight", attributes=("inputs:shaping:cone:angle",)),
            _FakePrim("/World/Rect", "RectLight"),
            _FakePrim("/World/Env", "DomeLight"),
        )
    )

    assert index["prim_paths"] == ("/World/Point", "/World/Spot", "/World/Rect")
    assert [candidate["authored_light_form"] for candidate in index["candidates"]] == [
        "POINT", "SPOT", "AREA_RECT"
    ]


def test_light_resolution_returns_same_value_for_source_and_hierarchy_matches() -> None:
    index = light._light_prim_index_from_prims(
        (_FakePrim("/World/Lighting/Key/KeyLight", "RectLight"),)
    )
    source = _FakeObject("Renamed")
    source[usd_paths.SOURCE_USD_PATH_PROP] = "/World/Lighting/Key"
    world = SimpleNamespace(name="World", parent=None)
    lighting = SimpleNamespace(name="Lighting", parent=world)
    hierarchy = _FakeObject("Key", parent=lighting)

    source_result = light.resolve_light_usd_prim(source, index)
    hierarchy_result = light.resolve_light_usd_prim(hierarchy, index)

    expected = light.LightUsdPrim("/World/Lighting/Key/KeyLight", "RectLight", "AREA_RECT")
    assert source_result.value == expected
    assert hierarchy_result.value == expected
    assert source_result.diagnostics["match_source"] == light.MATCH_SOURCE_USD_PATH
    assert hierarchy_result.diagnostics["match_source"] == light.MATCH_HIERARCHY_PATH


def test_light_resolution_fails_closed_on_ambiguous_name() -> None:
    index = light._light_prim_index_from_prims(
        tuple(_FakePrim(f"/World/{scope}/Key/KeyLight", "RectLight") for scope in ("A", "B"))
    )
    result = light.resolve_light_usd_prim(_FakeObject("Key"), index)

    assert result.value is None
    assert result.error_reason == light.ERROR_AMBIGUOUS
    assert result.diagnostics["candidate_count"] == 2


def _with_authoring_identity(obj: _FakeObject, prim_path: str) -> _FakeObject:
    obj.ov = SimpleNamespace(usd=SimpleNamespace(prim_path=prim_path))
    return obj


def test_light_resolution_prefers_authoring_identity_over_sanitized_leaf_name() -> None:
    # Authored generations sanitize display names ("Key Light" ->
    # ``Key_Light``), so leaf-name matching misses; the authoring identity
    # must resolve first (task04-03, follow-up from 04-01).
    index = light._light_prim_index_from_prims(
        (
            _FakePrim(
                "/World/Lights/Key_Light",
                "SphereLight",
                attributes=("inputs:shaping:cone:angle",),
            ),
            _FakePrim("/World/Lights/Fill_Light", "RectLight"),
        )
    )
    key = _with_authoring_identity(_FakeObject("Key Light"), "/World/Lights/Key_Light")

    result = light.resolve_light_usd_prim(key, index)

    assert result.value == light.LightUsdPrim(
        "/World/Lights/Key_Light", "SphereLight", "SPOT"
    )
    assert result.diagnostics["match_source"] == light.MATCH_AUTHORING_PRIM_PATH
    assert result.diagnostics["authoring_prim_path"] == "/World/Lights/Key_Light"


def test_light_resolution_reads_authoring_identity_through_the_original_datablock() -> None:
    # Evaluated depsgraph copies of some ID types drop add-on
    # PointerProperty data in Blender 5.1 (task04-02 discovery): the
    # identity must be read through ``usd_paths.authoring_prim_path``,
    # which falls back to ``id.original``.
    index = light._light_prim_index_from_prims(
        (_FakePrim("/World/Lights/Key_Light", "RectLight"),)
    )
    original = _with_authoring_identity(_FakeObject("Key Light"), "/World/Lights/Key_Light")
    evaluated = _FakeObject("Key Light")
    evaluated.original = original

    result = light.resolve_light_usd_prim(evaluated, index)

    assert result.value == light.LightUsdPrim(
        "/World/Lights/Key_Light", "RectLight", "AREA_RECT"
    )
    assert result.diagnostics["match_source"] == light.MATCH_AUTHORING_PRIM_PATH


def test_light_resolution_fails_closed_when_authoring_identity_is_not_in_scene() -> None:
    index = light._light_prim_index_from_prims(
        (_FakePrim("/World/Lights/Other_Light", "RectLight"),)
    )
    key = _with_authoring_identity(_FakeObject("Key Light"), "/World/Lights/Key_Light")

    result = light.resolve_light_usd_prim(key, index)

    assert result.value is None
    assert result.error_reason == light.ERROR_AUTHORING_PATH_NOT_IN_SCENE


def test_light_resolution_keeps_source_usd_path_fallback_for_direct_usd_stages() -> None:
    # A stale authoring identity must not break a direct-USD stage whose
    # sourceUsdPath still matches (04-02 fallback-preservation precedent).
    index = light._light_prim_index_from_prims(
        (_FakePrim("/World/Lighting/Key/KeyLight", "RectLight"),)
    )
    source = _with_authoring_identity(_FakeObject("Renamed"), "/World/Lights/Gone")
    source[usd_paths.SOURCE_USD_PATH_PROP] = "/World/Lighting/Key"

    result = light.resolve_light_usd_prim(source, index)

    assert result.value == light.LightUsdPrim(
        "/World/Lighting/Key/KeyLight", "RectLight", "AREA_RECT"
    )
    assert result.diagnostics["match_source"] == light.MATCH_SOURCE_USD_PATH

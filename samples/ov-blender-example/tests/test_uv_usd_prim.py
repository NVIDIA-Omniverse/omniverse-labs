# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import usd_paths as usd_paths  # noqa: E402
from ovrtx_blender_example import uv_usd_prim as uv  # noqa: E402


class _FakeAttr:
    def __init__(self, value: object, *, interpolation: str = "faceVarying") -> None:
        self._value = value
        self._interpolation = interpolation
        self.get_count = 0

    def Get(self) -> object:
        self.get_count += 1
        return self._value

    def GetMetadata(self, name: str) -> object:
        return self._interpolation if name == "interpolation" else None


class _FakePrim:
    def __init__(self, path: str, type_name: str = "Mesh", attrs=None) -> None:
        self._path = path
        self._type_name = type_name
        self._attrs = attrs or {}

    def GetPath(self) -> str:
        return self._path

    def GetTypeName(self) -> str:
        return self._type_name

    def GetAttribute(self, name: str):
        return self._attrs.get(name)


class _FakeUvValue:
    def __init__(self, value) -> None:
        self.uv = value


class _SizedIndexable:
    def __init__(self, values) -> None:
        self._values = tuple(values)

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, index: int):
        return self._values[index]


class _FakeLayer:
    def __init__(self, values) -> None:
        self.name = "UVMap"
        self.data = [_FakeUvValue(value) for value in values]


class _FakeMesh(dict):
    def __init__(self, values=((0.0, 0.0), (1.0, 0.0)), source_path="/World/Quad") -> None:
        super().__init__()
        self.name = self.name_full = "Quad"
        self.type = "MESH"
        self.uv_layers = type("Layers", (), {"active": _FakeLayer(values)})()
        self.loops = tuple(range(len(values)))
        self.polygons = ()
        self.vertices = tuple(range(len(values)))
        self.edges = ()
        if source_path:
            self[usd_paths.SOURCE_USD_PATH_PROP] = source_path


def _index(values=((0.0, 0.0), (1.0, 0.0))):
    attr = _FakeAttr(values)
    return uv._uv_prim_index_from_prims(
        (_FakePrim("/World/Quad", attrs={uv.TARGET_USD_ATTRIBUTE: attr}),)
    ), attr


def test_uv_resolution_freezes_primvar_facts_without_live_prim() -> None:
    index, attr = _index()
    result = uv.resolve_uv_usd_prim(_FakeMesh(), index)

    assert result.value == uv.UvUsdPrim(
        "/World/Quad",
        "primvars:st",
        "Float2Array",
        "faceVarying",
        2,
        ((0.0, 0.0), (1.0, 0.0)),
        uv.uv_digest(((0.0, 0.0), (1.0, 0.0))),
    )
    assert attr.get_count == 1
    assert "_prim" not in repr(index)
    assert "source_uv_values" not in result.diagnostics["candidates"][0]


def test_uv_prim_copies_mutable_input_values() -> None:
    values = [[0.0, 0.0]]
    prim = uv.UvUsdPrim(
        "/World/Quad", "primvars:st", "Float2Array", "faceVarying", 1, values, uv.uv_digest(values)
    )

    values[0][0] = 9.0
    values.append([1.0, 1.0])

    assert prim.source_uv_values == ((0.0, 0.0),)


def test_uv_prim_rejects_digest_mismatch() -> None:
    with pytest.raises(ValueError, match="digest must match"):
        uv.UvUsdPrim("/World/Quad", "primvars:st", "Float2Array", "faceVarying", 1, ((0.0, 0.0),), "bogus")


def test_uv_loop_validation_uses_frozen_value_and_cache_digest() -> None:
    index, attr = _index()
    prim = uv.resolve_uv_usd_prim(_FakeMesh(), index).value
    assert prim is not None
    snapshot = uv.active_uv_snapshot(_FakeMesh())
    validation = uv.validate_loop_order(snapshot, prim)

    assert validation["status"] == uv.RESOLVED
    assert validation["source_uv_digest"] == prim.source_uv_digest
    assert uv.cached_loop_order_validation_is_valid(validation, snapshot, prim) is True
    assert attr.get_count == 1


def test_uv_resolution_accepts_sized_indexable_values() -> None:
    values = _SizedIndexable(((0.0, 0.0), (1.0, 0.0)))
    index, _ = _index(values)

    result = uv.resolve_uv_usd_prim(_FakeMesh(), index)

    assert result.value is not None
    assert result.value.source_uv_values == ((0.0, 0.0), (1.0, 0.0))


def test_uv_resolution_reads_only_target_primvar() -> None:
    target = _FakeAttr(((0.0, 0.0), (1.0, 0.0)))
    unrelated = _FakeAttr(((9.0, 9.0),))
    index = uv._uv_prim_index_from_prims(
        (
            _FakePrim(
                "/World/Quad",
                attrs={uv.TARGET_USD_ATTRIBUTE: target, "primvars:other": unrelated},
            ),
        )
    )

    result = uv.resolve_uv_usd_prim(_FakeMesh(), index)

    assert result.value is not None
    assert target.get_count == 1
    assert unrelated.get_count == 0


def test_uv_resolution_fails_closed_for_indexed_interpolation_and_inferred_path() -> None:
    indexed = uv._uv_prim_index_from_prims(
        (
            _FakePrim(
                "/World/Quad",
                attrs={
                    uv.TARGET_USD_ATTRIBUTE: _FakeAttr(((0.0, 0.0),)),
                    uv.TARGET_USD_ATTRIBUTE + ":indices": object(),
                },
            ),
        )
    )
    unsupported = uv._uv_prim_index_from_prims(
        (
            _FakePrim(
                "/World/Quad",
                attrs={uv.TARGET_USD_ATTRIBUTE: _FakeAttr(((0.0, 0.0),), interpolation="vertex")},
            ),
        )
    )
    inferred_index, _ = _index()

    assert uv.resolve_uv_usd_prim(_FakeMesh(), indexed).error_reason == uv.ERROR_INDEXED_PRIMVAR
    assert (
        uv.resolve_uv_usd_prim(_FakeMesh(), unsupported).error_reason
        == uv.ERROR_UNSUPPORTED_INTERPOLATION
    )
    assert (
        uv.resolve_uv_usd_prim(_FakeMesh(source_path=""), inferred_index).error_reason
        == uv.ERROR_PRIM_INFERRED_ONLY
    )


def test_uv_resolution_reports_missing_and_malformed_primvar() -> None:
    missing = uv._uv_prim_index_from_prims((_FakePrim("/World/Quad"),))
    malformed = uv._uv_prim_index_from_prims(
        (
            _FakePrim(
                "/World/Quad",
                attrs={uv.TARGET_USD_ATTRIBUTE: _FakeAttr(object())},
            ),
        )
    )

    assert uv.resolve_uv_usd_prim(_FakeMesh(), missing).error_reason == uv.ERROR_MISSING_PRIMVAR
    assert uv.resolve_uv_usd_prim(_FakeMesh(), malformed).error_reason == uv.ERROR_MALFORMED_PRIMVAR


def test_active_uv_snapshot_reports_missing_layer() -> None:
    mesh = type("Mesh", (), {"uv_layers": type("Layers", (), {"active": None})()})()

    snapshot = uv.active_uv_snapshot(mesh)

    assert snapshot["status"] == uv.ERROR_MISSING_UV_LAYER
    assert snapshot["uv_values"] == ()


def test_uv_loop_validation_rejects_count_and_value_mismatch() -> None:
    index, _ = _index()
    prim = uv.resolve_uv_usd_prim(_FakeMesh(), index).value
    assert prim is not None

    assert uv.validate_loop_order(uv.active_uv_snapshot(_FakeMesh(((0.0, 0.0),))), prim)["status"] == uv.ERROR_COUNT_MISMATCH
    assert uv.validate_loop_order(uv.active_uv_snapshot(_FakeMesh(((0.25, 0.0), (1.0, 0.0)))), prim)["status"] == uv.ERROR_LOOP_ORDER_UNPROVEN

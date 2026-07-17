# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import material_usd_prim as material  # noqa: E402
from ovrtx_blender_example import usd_paths as usd_paths  # noqa: E402
from ovrtx_blender_example.usd_prim_resolution import UsdPrimResolutionStatus  # noqa: E402


class _FakeAttribute:
    def __init__(self, valid: bool = True, value: object = None) -> None:
        self._valid = valid
        self._value = value

    def IsValid(self) -> bool:
        return self._valid

    def Get(self) -> object:
        return self._value


class _FakePrim:
    def __init__(
        self,
        path: str,
        *,
        type_name: str = "Xform",
        attributes=(),
        info_id="",
        connected_attributes=(),
    ) -> None:
        self.path = path
        self.type_name = type_name
        self.attributes = attributes
        self.info_id = info_id
        self.connected_attributes = connected_attributes

    def GetPath(self) -> str:
        return self.path

    def GetTypeName(self) -> str:
        return self.type_name

    def GetAttribute(self, name: str):
        if name == "info:id" and self.info_id:
            return _FakeAttribute(value=self.info_id)
        return _FakeAttribute() if name in self.attributes else None


class _FakeMaterial(dict):
    def __init__(self, name: str, source_path: str = "", authoring_prim_path: str = "") -> None:
        super().__init__()
        self.name = name
        if source_path:
            self[usd_paths.SOURCE_USD_PATH_PROP] = source_path
        if authoring_prim_path:
            self.ov = SimpleNamespace(usd=SimpleNamespace(prim_path=authoring_prim_path))


def _index():
    return material._material_prim_index_from_prims(
        [
            _FakePrim("/World/Looks/Paint", type_name="Material"),
            _FakePrim(
                "/World/Looks/Paint/Shader",
                type_name="Shader",
                attributes=("inputs:diffuseColor",),
                info_id="UsdPreviewSurface",
            ),
            _FakePrim("/World/Looks/NoColor", type_name="Material"),
        ]
    )


def test_material_resolution_returns_typed_prim_and_local_match_diagnostics() -> None:
    result = material.resolve_material_usd_prim(_FakeMaterial("Paint"), _index())

    assert result.status is UsdPrimResolutionStatus.OK
    assert result.value == material.MaterialUsdPrim(
        "/World/Looks/Paint", "/World/Looks/Paint/Shader", "inputs:diffuseColor"
    )
    assert result.diagnostics["match_source"] == material.MATCH_HIERARCHY_PATH
    assert result.diagnostics["candidate_count"] == 1


def test_material_resolution_prefers_source_path_without_leaking_match_into_value() -> None:
    result = material.resolve_material_usd_prim(
        _FakeMaterial("Renamed", "/World/Looks/Paint/Shader"), _index()
    )

    assert result.value == material.MaterialUsdPrim(
        "/World/Looks/Paint", "/World/Looks/Paint/Shader", "inputs:diffuseColor"
    )
    assert result.diagnostics["match_source"] == material.MATCH_SOURCE_USD_PATH


def test_material_resolution_accepts_source_material_path() -> None:
    result = material.resolve_material_usd_prim(
        _FakeMaterial("Renamed", "/World/Looks/Paint"), _index()
    )

    assert result.value == material.MaterialUsdPrim(
        "/World/Looks/Paint", "/World/Looks/Paint/Shader", "inputs:diffuseColor"
    )


def test_material_resolution_ignores_non_preview_surface_shader() -> None:
    index = material._material_prim_index_from_prims(
        [
            _FakePrim("/World/Looks/Paint", type_name="Material"),
            _FakePrim(
                "/World/Looks/Paint/Shader",
                type_name="Shader",
                attributes=("inputs:diffuseColor",),
                info_id="OtherShader",
            ),
        ]
    )

    result = material.resolve_material_usd_prim(_FakeMaterial("Paint"), index)

    assert result.value is None
    assert result.error_reason == material.ERROR_MISSING_PRIM_ATTRIBUTE


def test_material_resolution_accepts_openpbr_surface_shader() -> None:
    index = material._material_prim_index_from_prims(
        [
            _FakePrim("/World/Looks/Paint", type_name="Material"),
            _FakePrim(
                "/World/Looks/Paint/OpenPBR",
                type_name="Shader",
                attributes=("inputs:geometry_opacity",),
                info_id="ND_open_pbr_surface_surfaceshader",
            ),
        ],
        usd_attribute="inputs:geometry_opacity",
    )

    result = material.resolve_material_usd_prim(
        _FakeMaterial("Paint"),
        index,
        usd_attribute="inputs:geometry_opacity",
    )

    assert result.value == material.MaterialUsdPrim(
        "/World/Looks/Paint", "/World/Looks/Paint/OpenPBR", "inputs:geometry_opacity"
    )


def test_material_resolution_reports_missing_name_and_no_path_match() -> None:
    missing_name = material.resolve_material_usd_prim(_FakeMaterial(""), _index())
    no_match = material.resolve_material_usd_prim(_FakeMaterial("Unknown"), _index())

    assert missing_name.error_reason == material.ERROR_MISSING_MATERIAL_NAME
    assert no_match.error_reason == material.ERROR_NO_PATH_MATCH


# --- Authoring-identity resolution over authored generations (task04-02) ----
#
# The topology orchestrator assigns each visual material's root prim path to
# ``material.ov.usd.prim_path`` and the converter authors the Material prim
# plus its ``PreviewSurface`` shader at exactly that path. Material names can
# sanitize differently from their USD leaf names, so the authoring identity
# is the primary resolution source.


def _authored_index(*, connected: bool = False, with_shader: bool = True):
    prims = [_FakePrim("/World/Materials/Board_Wood", type_name="Material")]
    if with_shader:
        prims.append(
            _FakePrim(
                "/World/Materials/Board_Wood/PreviewSurface",
                type_name="Shader",
                attributes=("inputs:diffuseColor",),
                info_id="UsdPreviewSurface",
                connected_attributes=("inputs:diffuseColor",) if connected else (),
            )
        )
    return material._material_prim_index_from_prims(prims)


def test_material_resolution_uses_authoring_identity_over_leaf_name() -> None:
    # "Board Wood" sanitizes to Board_Wood: a leaf-name match would miss.
    result = material.resolve_material_usd_prim(
        _FakeMaterial("Board Wood", authoring_prim_path="/World/Materials/Board_Wood"),
        _authored_index(),
    )

    assert result.status is UsdPrimResolutionStatus.OK
    assert result.value == material.MaterialUsdPrim(
        "/World/Materials/Board_Wood",
        "/World/Materials/Board_Wood/PreviewSurface",
        "inputs:diffuseColor",
    )
    assert result.diagnostics["match_source"] == material.MATCH_AUTHORING_PRIM_PATH
    assert result.diagnostics["authoring_prim_path"] == "/World/Materials/Board_Wood"


def test_material_resolution_fails_closed_for_unscanned_authoring_identity() -> None:
    # The material claims an authored identity the scanned composition does
    # not contain (stale reconcile): fail closed with the precise reason,
    # not a generic name miss (04-01 precedent).
    result = material.resolve_material_usd_prim(
        _FakeMaterial("Board Wood", authoring_prim_path="/World/Materials/Removed"),
        _authored_index(),
    )

    assert result.value is None
    assert result.error_reason == material.ERROR_AUTHORING_PATH_NOT_IN_SCENE


def test_material_resolution_reports_missing_attribute_for_authored_material() -> None:
    result = material.resolve_material_usd_prim(
        _FakeMaterial("Board Wood", authoring_prim_path="/World/Materials/Board_Wood"),
        _authored_index(with_shader=False),
    )

    assert result.value is None
    assert result.error_reason == material.ERROR_MISSING_PRIM_ATTRIBUTE


def test_material_resolution_reports_authored_texture_connection_state() -> None:
    connected = material.resolve_material_usd_prim(
        _FakeMaterial("Board Wood", authoring_prim_path="/World/Materials/Board_Wood"),
        _authored_index(connected=True),
    )
    unconnected = material.resolve_material_usd_prim(
        _FakeMaterial("Board Wood", authoring_prim_path="/World/Materials/Board_Wood"),
        _authored_index(),
    )

    assert connected.value is not None and connected.value.connected is True
    assert unconnected.value is not None and unconnected.value.connected is False


def test_material_resolution_fails_closed_for_ambiguity_missing_attribute_and_stage() -> None:
    ambiguous = material._material_prim_index_from_prims(
        [
            _FakePrim(f"/World/{scope}/Paint", type_name="Material")
            for scope in ("A", "B")
        ]
        + [
            _FakePrim(
                f"/World/{scope}/Paint/Shader",
                type_name="Shader",
                attributes=("inputs:diffuseColor",),
                info_id="UsdPreviewSurface",
            )
            for scope in ("A", "B")
        ]
    )
    ambiguous_result = material.resolve_material_usd_prim(_FakeMaterial("Paint"), ambiguous)
    missing_result = material.resolve_material_usd_prim(_FakeMaterial("NoColor"), _index())
    unavailable_result = material.resolve_material_usd_prim(
        _FakeMaterial("Paint"), {"available": False, "reason": "stage_open_failed"}
    )

    assert ambiguous_result.error_reason == material.ERROR_AMBIGUOUS
    assert ambiguous_result.diagnostics["candidate_count"] == 2
    assert missing_result.error_reason == material.ERROR_MISSING_PRIM_ATTRIBUTE
    assert unavailable_result.error_reason == material.ERROR_USD_STAGE_UNAVAILABLE
    assert unavailable_result.diagnostics["stage_reason"] == "stage_open_failed"

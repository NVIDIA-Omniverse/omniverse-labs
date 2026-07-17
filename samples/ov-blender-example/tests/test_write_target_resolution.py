# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import write_target_resolution as ownership  # noqa: E402


class _FakeLayer:
    def __init__(
        self,
        identifier: str,
        *,
        real_path: str = "",
        resolved_path: str = "",
        display_name: str = "",
        anonymous: bool = False,
    ) -> None:
        self.identifier = identifier
        self.realPath = real_path
        self.resolvedPath = resolved_path
        self._display_name = display_name
        self.anonymous = anonymous

    def GetDisplayName(self) -> str:
        return self._display_name


class _FakeSpec:
    def __init__(
        self,
        layer: _FakeLayer,
        *,
        specifier: str = "def",
        type_name: str = "",
        path: str = "/World/Thing",
        name: str = "",
        references: tuple[str, ...] = (),
    ) -> None:
        self.layer = layer
        self.specifier = specifier
        self.typeName = type_name
        self.path = path
        self.name = name
        self.referenceList = references


class _FakeProperty:
    def __init__(self, stack: tuple[_FakeSpec, ...], *, valid: bool = True) -> None:
        self._stack = stack
        self._valid = valid

    def IsValid(self) -> bool:
        return self._valid

    def GetPropertyStack(self) -> tuple[_FakeSpec, ...]:
        return self._stack


class _FakePrim:
    def __init__(
        self,
        *,
        prim_stack: tuple[_FakeSpec, ...] = (),
        attributes: dict[str, _FakeProperty] | None = None,
        relationships: dict[str, _FakeProperty] | None = None,
        valid: bool = True,
    ) -> None:
        self._prim_stack = prim_stack
        self._attributes = dict(attributes or {})
        self._relationships = dict(relationships or {})
        self._valid = valid
        self.attribute_calls: list[str] = []
        self.relationship_calls: list[str] = []

    def IsValid(self) -> bool:
        return self._valid

    def GetPrimStack(self) -> tuple[_FakeSpec, ...]:
        return self._prim_stack

    def GetAttribute(self, name: str) -> _FakeProperty:
        self.attribute_calls.append(name)
        return self._attributes.get(name, _FakeProperty((), valid=False))

    def GetRelationship(self, name: str) -> _FakeProperty:
        self.relationship_calls.append(name)
        return self._relationships.get(name, _FakeProperty((), valid=False))


class _FakeStage:
    def __init__(self, prims: dict[str, _FakePrim]) -> None:
        self._prims = prims

    def GetPrimAtPath(self, path: str) -> _FakePrim:
        return self._prims.get(path, _FakePrim(valid=False))


def _usd_module(stage: _FakeStage | None) -> SimpleNamespace:
    class _StageFactory:
        @staticmethod
        def Open(path: str) -> _FakeStage | None:
            assert path == "/fixtures/composed.usda"
            return stage

    return SimpleNamespace(Stage=_StageFactory)


def test_write_target_resolution_result_enforces_binary_invariants() -> None:
    ok = ownership.WriteTargetResolutionResult(
        ownership.WriteTargetResolutionStatus.OK,
        usd_layer_id="/layers/scene.usda",
    )
    error = ownership.WriteTargetResolutionResult(
        ownership.WriteTargetResolutionStatus.ERROR,
        error_reason=ownership.REASON_MISSING_WRITE_TARGET,
    )

    assert ok.error_reason is None
    assert error.usd_layer_id is None
    with pytest.raises(ValueError, match="requires an identifier"):
        ownership.WriteTargetResolutionResult(ownership.WriteTargetResolutionStatus.OK)
    with pytest.raises(ValueError, match="cannot have an identifier"):
        ownership.WriteTargetResolutionResult(
            ownership.WriteTargetResolutionStatus.ERROR,
            usd_layer_id="/layers/scene.usda",
            error_reason=ownership.REASON_MISSING_WRITE_TARGET,
        )
    with pytest.raises(ValueError, match="requires an error reason"):
        ownership.WriteTargetResolutionResult(ownership.WriteTargetResolutionStatus.ERROR)


def test_attribute_target_uses_attribute_property_stack() -> None:
    layer = _FakeLayer(
        "asset:looks.usda",
        real_path="/resolved/looks.usda",
        resolved_path="/resolved/looks.usda",
        display_name="looks.usda",
    )
    spec = _FakeSpec(layer, name="inputs:diffuseColor")
    attr = _FakeProperty((spec,))
    prim = _FakePrim(
        attributes={"inputs:diffuseColor": attr},
        relationships={"inputs:diffuseColor": _FakeProperty((_FakeSpec(_FakeLayer("wrong.usda")),))},
    )

    result = ownership.resolve_write_target(
        "/fixtures/composed.usda",
        usd_prim_path="/World/Looks/Paint/Shader",
        target_kind=ownership.TARGET_KIND_ATTRIBUTE,
        usd_property_name="inputs:diffuseColor",
        usd_module=_usd_module(_FakeStage({"/World/Looks/Paint/Shader": prim})),
    )

    assert result.usd_layer_id == "asset:looks.usda"
    assert result.status is ownership.WriteTargetResolutionStatus.OK
    assert result.error_reason is None
    assert result.diagnostics["target_kind"] == ownership.TARGET_KIND_ATTRIBUTE
    assert result.diagnostics["usd_property_path"] == "/World/Looks/Paint/Shader.inputs:diffuseColor"
    assert result.diagnostics["winning_spec"]["layer_identifier"] == "asset:looks.usda"
    assert result.diagnostics["winning_spec"]["layer_real_path"] == "/resolved/looks.usda"
    assert prim.attribute_calls == ["inputs:diffuseColor"]
    assert prim.relationship_calls == []


def test_relationship_target_uses_relationship_property_stack() -> None:
    layer = _FakeLayer("/layers/bindings.usda")
    relationship = _FakeProperty((_FakeSpec(layer, name="material:binding"),))
    prim = _FakePrim(
        attributes={"material:binding": _FakeProperty((_FakeSpec(_FakeLayer("wrong.usda")),))},
        relationships={"material:binding": relationship},
    )

    result = ownership.resolve_write_target(
        "/fixtures/composed.usda",
        usd_prim_path="/World/Geom/Cube",
        target_kind=ownership.TARGET_KIND_RELATIONSHIP,
        usd_property_name="material:binding",
        usd_module=_usd_module(_FakeStage({"/World/Geom/Cube": prim})),
    )

    assert result.usd_layer_id == "/layers/bindings.usda"
    assert prim.attribute_calls == []
    assert prim.relationship_calls == ["material:binding"]


def test_prim_definition_skips_stronger_over_and_uses_first_durable_definition() -> None:
    over_layer = _FakeLayer("/layers/stronger-over.usda")
    def_layer = _FakeLayer("/layers/layout.usda")
    prim = _FakePrim(
        prim_stack=(
            _FakeSpec(over_layer, specifier="over", type_name=""),
            _FakeSpec(def_layer, specifier="def", type_name="Xform"),
            _FakeSpec(_FakeLayer("/layers/weaker.usda"), specifier="def", type_name="Xform"),
        )
    )

    result = ownership.resolve_write_target(
        "/fixtures/composed.usda",
        usd_prim_path="/World/Key",
        target_kind=ownership.TARGET_KIND_PRIM,
        usd_module=_usd_module(_FakeStage({"/World/Key": prim})),
    )

    assert result.usd_layer_id == "/layers/layout.usda"
    assert result.diagnostics["candidate_specs"][0]["candidate_status"] == "not_prim_definition"
    assert result.diagnostics["winning_spec"]["index"] == 1


def test_ignored_and_anonymous_layers_do_not_win_ownership() -> None:
    session_layer = _FakeLayer("/tmp/session-layer.usda")
    anonymous_layer = _FakeLayer("anon:0001", anonymous=True)
    durable_layer = _FakeLayer("omniverse://server/project/model.usd")
    prim = _FakePrim(
        prim_stack=(
            _FakeSpec(session_layer, specifier="def", type_name="Xform"),
            _FakeSpec(anonymous_layer, specifier="def", type_name="Xform"),
            _FakeSpec(durable_layer, specifier="def", type_name="Xform"),
        )
    )

    result = ownership.resolve_write_target(
        "/fixtures/composed.usda",
        usd_prim_path="/World/Key",
        target_kind=ownership.TARGET_KIND_PRIM,
        ignored_layer_identifiers=("/tmp/session-layer.usda",),
        usd_module=_usd_module(_FakeStage({"/World/Key": prim})),
    )

    assert result.usd_layer_id == "omniverse://server/project/model.usd"
    assert result.diagnostics["candidate_specs"][0]["layer_status"] == "ignored_layer"
    assert result.diagnostics["candidate_specs"][1]["layer_status"] == "anonymous_layer"


def test_session_layer_only_fails_closed() -> None:
    prim = _FakePrim(
        prim_stack=(
            _FakeSpec(_FakeLayer("/tmp/session-layer.usda"), specifier="def", type_name="Xform"),
        )
    )

    result = ownership.resolve_write_target(
        "/fixtures/composed.usda",
        usd_prim_path="/World/Key",
        target_kind=ownership.TARGET_KIND_PRIM,
        ignored_layer_identifiers=("/tmp/session-layer.usda",),
        usd_module=_usd_module(_FakeStage({"/World/Key": prim})),
    )

    assert result.status is ownership.WriteTargetResolutionStatus.ERROR
    assert result.usd_layer_id is None
    assert result.error_reason == ownership.REASON_SESSION_LAYER_ONLY


def test_explicit_metadata_is_verified_or_rejected_against_stack() -> None:
    prim = _FakePrim(
        attributes={
            "inputs:intensity": _FakeProperty(
                (_FakeSpec(_FakeLayer("/layers/lights.usda"), name="inputs:intensity"),)
            )
        }
    )

    stack_only = ownership.resolve_write_target(
        "/fixtures/composed.usda",
        usd_prim_path="/World/Key",
        target_kind=ownership.TARGET_KIND_ATTRIBUTE,
        usd_property_name="inputs:intensity",
        usd_module=_usd_module(_FakeStage({"/World/Key": prim})),
    )
    verified = ownership.resolve_write_target(
        "/fixtures/composed.usda",
        usd_prim_path="/World/Key",
        target_kind=ownership.TARGET_KIND_ATTRIBUTE,
        usd_property_name="inputs:intensity",
        explicit_usd_layer_id="/layers/lights.usda",
        usd_module=_usd_module(_FakeStage({"/World/Key": prim})),
    )
    mismatch = ownership.resolve_write_target(
        "/fixtures/composed.usda",
        usd_prim_path="/World/Key",
        target_kind=ownership.TARGET_KIND_ATTRIBUTE,
        usd_property_name="inputs:intensity",
        explicit_usd_layer_id="/layers/wrong.usda",
        usd_module=_usd_module(_FakeStage({"/World/Key": prim})),
    )

    assert verified.usd_layer_id == "/layers/lights.usda"
    assert verified.status is ownership.WriteTargetResolutionStatus.OK
    assert verified.error_reason is None
    assert (
        stack_only.status,
        stack_only.usd_layer_id,
        stack_only.error_reason,
    ) == (
        verified.status,
        verified.usd_layer_id,
        verified.error_reason,
    )
    assert mismatch.status is ownership.WriteTargetResolutionStatus.ERROR
    assert mismatch.usd_layer_id is None
    assert mismatch.error_reason == ownership.REASON_WRITE_TARGET_MISMATCH
    assert mismatch.diagnostics["explicit_usd_layer_id"] == "/layers/wrong.usda"
    assert mismatch.diagnostics["stack_resolved_identifier"] == "/layers/lights.usda"


def test_pxr_unavailable_rejects_explicit_metadata(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "pxr", None)

    result = ownership.resolve_write_target(
        "/fixtures/composed.usda",
        usd_prim_path="/World/Key",
        target_kind=ownership.TARGET_KIND_PRIM,
        explicit_usd_layer_id="/layers/lights.usda",
    )

    assert result.status is ownership.WriteTargetResolutionStatus.ERROR
    assert result.usd_layer_id is None
    assert result.error_reason == ownership.REASON_PXR_UNAVAILABLE
    assert result.diagnostics["explicit_usd_layer_id"] == "/layers/lights.usda"


def test_stage_open_failure_rejects_explicit_metadata() -> None:
    result = ownership.resolve_write_target(
        "/fixtures/composed.usda",
        usd_prim_path="/World/Key",
        target_kind=ownership.TARGET_KIND_PRIM,
        explicit_usd_layer_id="/layers/lights.usda",
        usd_module=_usd_module(None),
    )

    assert result.status is ownership.WriteTargetResolutionStatus.ERROR
    assert result.usd_layer_id is None
    assert result.error_reason == ownership.REASON_STAGE_OPEN_FAILED


def test_missing_target_and_unsupported_target_kind_fail_closed() -> None:
    missing = ownership.resolve_write_target(
        "/fixtures/composed.usda",
        usd_prim_path="/World/Missing",
        target_kind=ownership.TARGET_KIND_PRIM,
        usd_module=_usd_module(_FakeStage({})),
    )
    unsupported = ownership.resolve_write_target(
        "/fixtures/composed.usda",
        usd_prim_path="/World/Missing",
        target_kind="path_string_guess",
        usd_module=_usd_module(_FakeStage({})),
    )

    assert missing.usd_layer_id is None
    assert missing.error_reason == ownership.REASON_MISSING_TARGET
    assert unsupported.usd_layer_id is None
    assert unsupported.error_reason == ownership.REASON_UNSUPPORTED_TARGET_KIND


def test_missing_property_unavailable_stack_and_missing_durable_layer_fail_closed() -> None:
    stage = _FakeStage({"/World/Key": _FakePrim()})
    missing_property = ownership.resolve_write_target(
        "/fixtures/composed.usda",
        usd_prim_path="/World/Key",
        target_kind=ownership.TARGET_KIND_ATTRIBUTE,
        usd_module=_usd_module(stage),
    )
    unavailable_stack = ownership.resolve_write_target(
        "/fixtures/composed.usda",
        usd_prim_path="/World/Unavailable",
        target_kind=ownership.TARGET_KIND_PRIM,
        usd_module=_usd_module(
            _FakeStage(
                {
                    "/World/Unavailable": SimpleNamespace(
                        IsValid=lambda: True,
                        GetPrimStack=None,
                    )
                }
            )
        ),
    )
    missing_durable = ownership.resolve_write_target(
        "/fixtures/composed.usda",
        usd_prim_path="/World/MissingDurable",
        target_kind=ownership.TARGET_KIND_PRIM,
        usd_module=_usd_module(
            _FakeStage(
                {
                    "/World/MissingDurable": _FakePrim(
                        prim_stack=(_FakeSpec(_FakeLayer(""), specifier="def"),)
                    )
                }
            )
        ),
    )

    assert missing_property.error_reason == ownership.REASON_MISSING_PROPERTY
    assert missing_property.diagnostics["failure_detail"] == "missing_usd_property_name"
    assert unavailable_stack.error_reason == ownership.REASON_TARGET_STACK_UNAVAILABLE
    assert unavailable_stack.diagnostics["failure_detail"] == "missing_GetPrimStack"
    assert missing_durable.error_reason == ownership.REASON_MISSING_WRITE_TARGET
    assert missing_durable.diagnostics["candidate_specs"][0]["layer_status"] == (
        "missing_layer_identifier"
    )


def test_unsupported_prim_topology_fields_fail_closed() -> None:
    prim = _FakePrim(
        prim_stack=(
            _FakeSpec(
                _FakeLayer("/layers/asset-reference.usda"),
                specifier="over",
                references=("@asset.usd@</Root>",),
            ),
        )
    )

    result = ownership.resolve_write_target(
        "/fixtures/composed.usda",
        usd_prim_path="/World/Referenced",
        target_kind=ownership.TARGET_KIND_PRIM,
        usd_module=_usd_module(_FakeStage({"/World/Referenced": prim})),
    )

    assert result.usd_layer_id is None
    assert result.error_reason == ownership.REASON_UNSUPPORTED_TOPOLOGY_FIELDS
    assert result.diagnostics["candidate_specs"][0]["unsupported_topology_fields"] == ["references"]

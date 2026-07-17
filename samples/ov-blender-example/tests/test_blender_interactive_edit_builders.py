# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import math
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example.blender_interactive_edit_builders import (  # noqa: E402
    USD_ATTRIBUTE_PROP,
    light_value_edit,
    light_value_edits_from_prim,
    build_interactive_edits_from_depsgraph,
    material_value_edit,
    material_value_edit_from_prim,
    object_transform_edit,
    property_edit,
    edit_location_from_blender_id,
    uv_value_edit_from_prim,
    uv_value_edits_from_resolver,
    world_value_edits_from_prim,
)
from ovrtx_blender_example import blender_interactive_edit_builders as builders  # noqa: E402
from ovrtx_blender_example import usd_paths as usd_paths  # noqa: E402
from ovrtx_blender_example.blender_signal_translation import InteractiveEditTranslator  # noqa: E402
from ovrtx_blender_example.blender_signals import (  # noqa: E402
    BlenderEditSignal,
    BlenderEditSignalSource,
)
from ovrtx_blender_example.interactive_edit_planner import (  # noqa: E402
    EditMechanism,
    EditPersistence,
    DataAuthority,
    EditShape,
    InteractiveEditPlanner,
)
from ovrtx_blender_example import write_target_resolution as ownership  # noqa: E402
from ovrtx_blender_example import material_usd_prim  # noqa: E402
from ovrtx_blender_example import material_value_conversion as material_conversion  # noqa: E402
from ovrtx_blender_example import light_value_conversion as light_conversion  # noqa: E402
from ovrtx_blender_example import light_usd_prim  # noqa: E402
from ovrtx_blender_example import world_dome_conversion as world_conversion  # noqa: E402
from ovrtx_blender_example import world_dome_usd_prim  # noqa: E402
from ovrtx_blender_example import uv_usd_prim  # noqa: E402
from ovrtx_blender_example import usd_prim_resolver  # noqa: E402
from ovrtx_blender_example import usd_value_edit_support  # noqa: E402
from ovrtx_blender_example import interactive_operator_state as operator_state  # noqa: E402
from ovrtx_blender_example import ovphysx_to_ovrtx  # noqa: E402
from ovrtx_blender_example import render_requests  # noqa: E402
from ovrtx_blender_example.ovrtx_value_updates import (  # noqa: E402
    OvrtxAttributeValue,
    OvrtxTransformValue,
    OvrtxValueUpdateResult,
)
from ovrtx_blender_example.view_update_stream import ViewUpdateStream  # noqa: E402
from ovrtx_blender_example.value_edit_conversion import (  # noqa: E402
    UsdAttributeValue,
    ValueEditConversionPolicies,
    default_value_edit_conversion_policies,
)


class _FakeBlenderId(dict):
    def __init__(self, *, name: str = "Cube", blender_type: str = "MESH", **attrs: object) -> None:
        super().__init__()
        self.name = name
        self.name_full = name
        self.type = blender_type
        for key, value in attrs.items():
            setattr(self, key, value)


class _FakeDepsgraphUpdate:
    def __init__(self, id_data: object, *, is_updated_geometry: bool | None = None) -> None:
        self.id = id_data
        if is_updated_geometry is not None:
            self.is_updated_geometry = is_updated_geometry


class _FakeDepsgraph:
    def __init__(self, updates: list[object]) -> None:
        self.updates = updates


class _FakeResolver:
    def __init__(
        self,
        *,
        material_indexes=None,
        light_index=None,
        world_dome_index=None,
        uv_index=None,
        uv_validations=None,
        mesh_topology_change=None,
        object_path=None,
    ) -> None:
        self.material_indexes = material_indexes or {}
        self.light_index = light_index or {"available": False, "reason": "not_loaded"}
        self.world_dome_index = world_dome_index or {"available": False, "stage_reason": "not_loaded"}
        self.uv_index = uv_index or {"available": False, "reason": "not_loaded"}
        self.uv_validations = uv_validations or {}
        self._mesh_topology_change = mesh_topology_change
        self.object_path = object_path

    def resolve_blender_object(self, _obj):
        return SimpleNamespace(value=self.object_path, diagnostics_dict=lambda: {})

    def resolve_material(self, material, *, usd_attribute, property_name):
        return material_usd_prim.resolve_material_usd_prim(
            material,
            self.material_indexes.get(usd_attribute, {"available": False, "reason": "not_loaded"}),
            usd_attribute=usd_attribute,
            property_name=property_name,
        )

    def resolve_light(self, light):
        return light_usd_prim.resolve_light_usd_prim(light, self.light_index)

    def resolve_world_dome(self):
        return world_dome_usd_prim.resolve_world_dome_usd_prim(self.world_dome_index)

    def resolve_uv(self, mesh):
        return uv_usd_prim.resolve_uv_usd_prim(mesh, self.uv_index)

    def uv_loop_order_validation(self, prim_path):
        return self.uv_validations.get(prim_path)

    def mesh_topology_change(self, _mesh):
        return self._mesh_topology_change


class _FakeSocket:
    def __init__(self, value: object, *, linked: bool = False) -> None:
        self.default_value = value
        self.is_linked = linked


class _FakeNode:
    def __init__(self) -> None:
        self.type = "BSDF_PRINCIPLED"
        self.name = "Principled BSDF"
        self.inputs = {
            "Base Color": _FakeSocket((0.1, 0.2, 0.3, 1.0)),
            "Roughness": _FakeSocket(0.45),
            "Metallic": _FakeSocket(0.2),
            "IOR": _FakeSocket(1.45),
            "Emission Color": _FakeSocket((0.1, 0.2, 0.5, 1.0)),
            "Emission Strength": _FakeSocket(2.0),
            "Alpha": _FakeSocket(0.75),
        }


class _FakeNodeTree:
    def __init__(self) -> None:
        self.nodes = [_FakeNode()]


class _FakeUsdAttr:
    def __init__(self, value: object, *, interpolation: str = "faceVarying") -> None:
        self._value = value
        self._interpolation = interpolation

    def Get(self) -> object:
        return self._value

    def GetMetadata(self, name: str) -> object:
        return self._interpolation if name == "interpolation" else None


class _FakeUsdPrim:
    def __init__(self, path: str, attrs: dict[str, object]) -> None:
        self._path = path
        self._attrs = attrs

    def GetPath(self) -> str:
        return self._path

    def GetTypeName(self) -> str:
        return "Mesh"

    def GetAttribute(self, name: str) -> object | None:
        return self._attrs.get(name)


class _FakeLoop:
    def __init__(self, vertex_index: int) -> None:
        self.vertex_index = vertex_index


class _FakeUvItem:
    def __init__(self, uv: tuple[float, float]) -> None:
        self.uv = uv


class _FakeUvLayer:
    def __init__(self, values: tuple[tuple[float, float], ...], *, name: str = "UVMap") -> None:
        self.name = name
        self.data = [_FakeUvItem(value) for value in values]


class _FakeUvLayers:
    def __init__(self, active: _FakeUvLayer | None) -> None:
        self.active = active


class _FakeMesh(_FakeBlenderId):
    def __init__(
        self,
        *,
        name: str = "Quad",
        values: tuple[tuple[float, float], ...] = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        loop_vertex_indices: tuple[int, ...] | None = None,
        source_path: str = "/World/Quad",
    ) -> None:
        super().__init__(name=name, blender_type="MESH")
        self.uv_layers = _FakeUvLayers(_FakeUvLayer(values))
        self.loops = [
            _FakeLoop(index)
            for index in (loop_vertex_indices or tuple(range(len(values))))
        ]
        self.polygons = ()
        self.vertices = tuple(range(len(values)))
        self.edges = ()
        if source_path:
            self[usd_paths.SOURCE_USD_PATH_PROP] = source_path


def _identity_props(usd_layer_id: str = "/layers/scene.usda") -> dict[str, str]:
    return {
        usd_paths.USD_LAYER_ID_PROP: usd_layer_id,
        usd_paths.USD_PRIM_PATH_PROP: "/World/TestScene/Cube",
        USD_ATTRIBUTE_PROP: "xformOp:transform",
        usd_paths.BLENDER_PROPERTY_PATH_PROP: "matrix_world",
        usd_paths.DATA_AUTHORITY_PROP: "view",
    }


def test_target_from_blender_custom_properties() -> None:
    obj = _FakeBlenderId(name="Cube")
    obj.update(_identity_props())

    target = edit_location_from_blender_id(obj)

    assert target["usd_layer_id"] == "/layers/scene.usda"
    assert target["usd_prim_path"] == "/World/TestScene/Cube"
    assert target["usd_property_path"] == "/World/TestScene/Cube.xformOp:transform"
    assert target["blender_property_path"] == "matrix_world"
    assert target["provenance"] == {"source": "blender", "name": "Cube", "type": "MESH"}


def test_interactive_edit_translator_returns_tuple_for_blender_edit_signal() -> None:
    obj = _FakeBlenderId(
        name="Cube",
        matrix_world=((1, 0, 0, 2), (0, 1, 0, 3), (0, 0, 1, 4), (0, 0, 0, 1)),
    )
    obj.update(_identity_props())
    signal = BlenderEditSignal(
        source=BlenderEditSignalSource.DEPSGRAPH,
        id_items=(obj,),
    )

    edits = InteractiveEditTranslator().translate(signal)

    assert isinstance(edits, tuple)
    assert len(edits) == 1
    assert edits[0].usd_prim_path == "/World/TestScene/Cube"


def test_interactive_edit_translator_preserves_write_target_context(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def _resolve(input_usd_path: str, **kwargs: object) -> ownership.WriteTargetResolutionResult:
        calls.append((input_usd_path, dict(kwargs)))
        return ownership.WriteTargetResolutionResult(
            ownership.WriteTargetResolutionStatus.OK,
            usd_layer_id="/layers/scene.usda",
            diagnostics={
                "ignored_layer_identifiers": list(kwargs["ignored_layer_identifiers"]),
            },
        )

    monkeypatch.setattr(builders.write_target_resolution, "resolve_write_target", _resolve)
    obj = _FakeBlenderId(
        name="Cube",
        matrix_world=((1, 0, 0, 2), (0, 1, 0, 3), (0, 0, 1, 4), (0, 0, 0, 1)),
    )
    props = _identity_props(usd_layer_id="")
    obj.update(props)
    signal = BlenderEditSignal(
        source=BlenderEditSignalSource.DEPSGRAPH,
        id_items=(obj,),
        input_usd_path="/fixtures/composed.usda",
        ignored_layer_identifiers=("/tmp/session-layer.usda",),
    )

    edit = InteractiveEditTranslator().translate(signal)[0]

    assert calls[0][0] == "/fixtures/composed.usda"
    assert calls[0][1]["ignored_layer_identifiers"] == ("/tmp/session-layer.usda",)
    assert edit.usd_layer_id == "/layers/scene.usda"
    assert "write_target_resolution" not in edit.provenance
    assert "write_target_error_reason" not in edit.provenance


def test_failed_write_target_resolution_keeps_runtime_update_and_compact_error(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        builders.write_target_resolution,
        "resolve_write_target",
        lambda *args, **kwargs: ownership.WriteTargetResolutionResult(
            ownership.WriteTargetResolutionStatus.ERROR,
            error_reason=ownership.REASON_PXR_UNAVAILABLE,
            diagnostics={
                "candidate_specs": [{"layer_identifier": "/layers/scene.usda"}],
                "winning_spec": {"layer_identifier": "/layers/scene.usda"},
                "stack_resolved_identifier": "/layers/scene.usda",
            },
        ),
    )
    obj = _FakeBlenderId(
        name="Cube",
        matrix_world=((1, 0, 0, 2), (0, 1, 0, 3), (0, 0, 1, 4), (0, 0, 0, 1)),
    )
    obj.update(_identity_props(usd_layer_id="/layers/scene.usda"))
    edit = InteractiveEditTranslator().translate(
        BlenderEditSignal(
            source=BlenderEditSignalSource.DEPSGRAPH,
            id_items=(obj,),
            input_usd_path="/fixtures/composed.usda",
        )
    )[0]

    plan = InteractiveEditPlanner().plan(edit)

    assert edit.usd_layer_id == ""
    assert edit.provenance["write_target_error_reason"] == ownership.REASON_PXR_UNAVAILABLE
    assert "write_target_resolution" not in edit.provenance
    assert "candidate_specs" not in edit.provenance
    assert "winning_spec" not in edit.provenance
    assert "stack_resolved_identifier" not in edit.provenance
    assert plan.mechanism is EditMechanism.UPDATE
    assert plan.persistence is EditPersistence.NONE


def test_uninspectable_write_targets_fail_closed(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def _resolve(input_usd_path: str, **kwargs: object) -> ownership.WriteTargetResolutionResult:
        target_kind = str(kwargs["target_kind"])
        calls.append((input_usd_path, target_kind))
        reason = (
            ownership.REASON_STAGE_OPEN_FAILED
            if not input_usd_path
            else ownership.REASON_UNSUPPORTED_TARGET_KIND
        )
        return ownership.WriteTargetResolutionResult(
            ownership.WriteTargetResolutionStatus.ERROR,
            error_reason=reason,
        )

    monkeypatch.setattr(builders.write_target_resolution, "resolve_write_target", _resolve)
    target = builders.edit_location(
        usd_prim_path="/World/Cube",
        usd_attribute="xformOp:transform",
        usd_layer_id="/layers/unverified.usda",
        blender_property_path="matrix_world",
    )
    edit = property_edit(object(), property_name="matrix_world", usd_attribute="xformOp:transform", location=target)

    missing_stage = builders._with_write_target_resolution(
        edit, input_usd_path=None, ignored_layer_identifiers=()
    )
    unknown_target = builders._with_write_target_resolution(
        property_edit(
            object(),
            property_name="custom_property",
            usd_attribute="",
            location={**target, "usd_property_path": "", "blender_property_path": "custom_property"},
        ),
        input_usd_path="/stage.usda",
        ignored_layer_identifiers=(),
    )

    assert calls == [("", ownership.TARGET_KIND_ATTRIBUTE), ("/stage.usda", "")]
    for failed in (missing_stage, unknown_target):
        assert failed.usd_layer_id == ""
        assert failed.provenance["write_target_error_reason"]
        assert InteractiveEditPlanner().plan(failed).persistence is EditPersistence.NONE


def test_unresolved_topology_target_requests_scene_generation_replacement() -> None:
    edit = builders.InteractiveEdit(
        shape=EditShape.TOPOLOGY,
        data_authority=DataAuthority.VIEW,
        **builders.edit_location(
            usd_prim_path="/World/Key",
            usd_layer_id="/layers/unverified.usda",
            blender_property_path="object.data",
            provenance={"write_target_error_reason": "stale"},
        ),
        value="new light",
    )

    resolved = builders._with_write_target_resolution(
        edit, input_usd_path=None, ignored_layer_identifiers=()
    )
    plan = InteractiveEditPlanner().plan(resolved)

    assert resolved.usd_layer_id == ""
    assert resolved.provenance["write_target_error_reason"] == "pxr_unavailable"
    assert plan.impact.scene_generation_replacement_requested is True


def test_interactive_edit_translator_resolves_selection_from_signal_context() -> None:
    context = object()
    selection_resolution = {
        "changed": False,
        "group_rejected": False,
        "sources": [
            {
                "source_name": "Cube",
                "owner_name": "Cube",
                "owner_usd_path": "/World/TestScene/Cube",
                "source_session_uid": 101,
                "status": "resolved",
            }
        ],
    }
    contexts: list[object] = []

    def _resolve(received_context: object | None) -> dict[str, object]:
        contexts.append(received_context)
        return selection_resolution

    obj = _FakeBlenderId(
        name="Cube",
        session_uid=101,
        matrix_world=((1, 0, 0, 2), (0, 1, 0, 3), (0, 0, 1, 4), (0, 0, 0, 1)),
    )
    obj.update(_identity_props())
    signal = BlenderEditSignal(
        source=BlenderEditSignalSource.SELECTION,
        id_items=(obj,),
        context=context,
    )
    translator = InteractiveEditTranslator(selection_resolver=_resolve)

    edit = translator.translate(signal)[0]

    assert contexts == [context]
    assert translator.selection_resolution == selection_resolution
    assert edit.provenance["selection_resolution"]["source_name"] == "Cube"


def test_interactive_edit_translator_skips_edits_when_selection_changes() -> None:
    builder_calls: list[object] = []

    def _builder(depsgraph: object, **kwargs: object) -> list[object]:
        builder_calls.append(depsgraph)
        return [object()]

    translator = InteractiveEditTranslator(
        edit_builder=_builder,
        selection_resolver=lambda context: {"changed": True, "group_rejected": False},
    )
    signal = BlenderEditSignal(
        source=BlenderEditSignalSource.SELECTION,
        id_items=(object(),),
        context=object(),
    )

    assert translator.translate(signal) == ()
    assert builder_calls == []
    assert translator.selection_resolution == {"changed": True, "group_rejected": False}


def test_target_uses_imported_usd_source_path_from_object_or_data() -> None:
    obj = _FakeBlenderId(name="Body")
    obj[usd_paths.SOURCE_USD_PATH_PROP] = "/World/PhysicsIsland/DynamicBodies/Body"

    target = edit_location_from_blender_id(obj)

    assert usd_paths.source_usd_path_from_blender_id(obj) == "/World/PhysicsIsland/DynamicBodies/Body"
    assert target["usd_prim_path"] == "/World/PhysicsIsland/DynamicBodies/Body"

    mesh_data = _FakeBlenderId(name="MeshData", blender_type="MESH")
    mesh_data[usd_paths.SOURCE_USD_PATH_PROP] = "/World/PhysicsIsland/DynamicBodies/Body/Geom"
    mesh_obj = _FakeBlenderId(name="Geom", data=mesh_data)

    assert usd_paths.source_usd_path_from_blender_id(mesh_obj) == "/World/PhysicsIsland/DynamicBodies/Body/Geom"


def test_object_transform_edit_uses_stock_matrix_world() -> None:
    matrix = (
        (1, 0, 0, 2),
        (0, 1, 0, 3),
        (0, 0, 1, 4),
        (0, 0, 0, 1),
    )
    obj = _FakeBlenderId(name="Cube", matrix_world=matrix)
    obj.update(_identity_props())

    edit = object_transform_edit(obj)
    plan = InteractiveEditPlanner().plan(edit)

    assert (edit.shape, edit.data_authority) == (EditShape.VALUE, DataAuthority.VIEW)
    assert edit.value == [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [2.0, 3.0, 4.0, 1.0],
    ]
    assert edit.usd_prim_path == "/World/TestScene/Cube"
    assert plan.mechanism == EditMechanism.UPDATE


def test_object_transform_edit_uses_explicit_data_authority_for_sim_value() -> None:
    matrix = (
        (1, 0, 0, 2),
        (0, 1, 0, 3),
        (0, 0, 1, 4),
        (0, 0, 0, 1),
    )
    obj = _FakeBlenderId(name="RigidBody", matrix_world=matrix)
    obj.update(
        {
            **_identity_props("/layers/physics.usda"),
            usd_paths.DATA_AUTHORITY_PROP: "sim",
            USD_ATTRIBUTE_PROP: "xformOp:transform",
            usd_paths.BLENDER_PROPERTY_PATH_PROP: "physics.drop_pose",
        }
    )

    edit = object_transform_edit(obj)

    assert (edit.shape, edit.data_authority) == (EditShape.VALUE, DataAuthority.SIM)
    assert InteractiveEditPlanner().plan(edit).mechanism == EditMechanism.UPDATE


def test_object_transform_edit_requires_explicit_data_authority_metadata() -> None:
    obj = _FakeBlenderId(
        name="Cube",
        matrix_world=((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
    )
    props = _identity_props()
    props.pop(usd_paths.DATA_AUTHORITY_PROP)
    obj.update(props)

    try:
        object_transform_edit(obj)
    except ValueError as exc:
        assert usd_paths.DATA_AUTHORITY_PROP in str(exc)
    else:
        raise AssertionError("object transform edits must require explicit classification")


def test_selection_source_owner_metadata_does_not_create_transform_edit() -> None:
    obj = _FakeBlenderId(
        name="OrangeMesh",
        matrix_world=((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
    )
    obj[usd_paths.SELECTION_OWNER_OBJECT_PROP] = "Orange_00"

    assert usd_paths.selection_owner_object_name(obj) == "Orange_00"
    assert build_interactive_edits_from_depsgraph(_FakeDepsgraph([_FakeDepsgraphUpdate(obj)])) == []


def test_depsgraph_object_resolution_uses_original_blender_identity() -> None:
    original = _FakeBlenderId(
        name="Cube",
        matrix_world=((1, 0, 0, 2), (0, 1, 0, 3), (0, 0, 1, 4), (0, 0, 0, 1)),
    )
    evaluated = SimpleNamespace(original=original, matrix_world=original.matrix_world)

    edits = build_interactive_edits_from_depsgraph(
        _FakeDepsgraph([_FakeDepsgraphUpdate(evaluated)]),
        usd_prim_resolver=_FakeResolver(object_path="/World/Cube"),
    )

    assert len(edits) == 1
    assert edits[0].usd_prim_path == "/World/Cube"


def _uv_index(values: tuple[tuple[float, float], ...] = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))):
    return uv_usd_prim._uv_prim_index_from_prims(
        [
            _FakeUsdPrim(
                "/World/Quad",
                {uv_usd_prim.TARGET_USD_ATTRIBUTE: _FakeUsdAttr(values)},
            )
        ]
    )


def test_uv_value_edit_from_prim_extracts_active_uv_values_and_diagnostics() -> None:
    mesh = _FakeMesh()
    prim = uv_usd_prim.resolve_uv_usd_prim(mesh, _uv_index()).value
    assert prim is not None

    edit = uv_value_edit_from_prim(mesh, prim)
    assert edit is not None
    plan = InteractiveEditPlanner().plan(edit)

    assert (edit.shape, edit.data_authority) == (EditShape.VALUE, DataAuthority.VIEW)
    assert edit.usd_prim_path == "/World/Quad"
    assert edit.usd_attribute == "primvars:st"
    assert edit.blender_property_path == "uv_layers.active"
    assert edit.value == ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    assert "usd_value_type" not in edit.metadata
    assert edit.metadata["element_count"] == 4
    assert edit.metadata["loop_order_validation"]["status"] == uv_usd_prim.RESOLVED
    assert plan.mechanism == EditMechanism.UPDATE
    assert plan.impact.physics_generation_reset_expected is False


def test_uv_value_edit_uses_cached_loop_order_validation_for_changed_values() -> None:
    mesh = _FakeMesh()
    prim = uv_usd_prim.resolve_uv_usd_prim(mesh, _uv_index()).value
    assert prim is not None
    baseline = uv_usd_prim.active_uv_snapshot(mesh)
    validation = uv_usd_prim.validation_record(uv_usd_prim.validate_loop_order(baseline, prim))
    changed_mesh = _FakeMesh(values=((0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)))

    edit = uv_value_edit_from_prim(changed_mesh, prim, loop_order_validation=validation)

    assert edit is not None
    assert edit.value == ((0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75))
    assert edit.metadata["loop_order_validation"]["topology_fingerprint"] == validation["topology_fingerprint"]


def test_uv_builder_classifies_changed_mesh_topology_before_value_routing() -> None:
    baseline = _FakeMesh()
    prim = uv_usd_prim.resolve_uv_usd_prim(baseline, _uv_index()).value
    assert prim is not None
    validation = uv_usd_prim.validation_record(
        uv_usd_prim.validate_loop_order(uv_usd_prim.active_uv_snapshot(baseline), prim)
    )
    changed = _FakeMesh(loop_vertex_indices=(1, 0, 2, 3))

    edit = uv_value_edits_from_resolver(
        changed,
        _FakeResolver(
            mesh_topology_change={
                "usd_prim_path": "/World/Quad",
                "previous_fingerprint": validation["topology_fingerprint"],
                "current_fingerprint": uv_usd_prim.mesh_topology_fingerprint(changed),
            },
        ),
    )[0]
    plan = InteractiveEditPlanner().plan(edit)

    assert edit.shape == EditShape.TOPOLOGY
    assert plan.impact.scene_generation_replacement_requested is True


def test_uv_value_edit_rejects_changed_values_without_cached_loop_order_validation() -> None:
    changed_mesh = _FakeMesh(values=((0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)))
    prim = uv_usd_prim.resolve_uv_usd_prim(changed_mesh, _uv_index()).value
    assert prim is not None

    assert uv_value_edit_from_prim(changed_mesh, prim) is None


def test_interactive_depsgraph_builds_uv_value_edit_from_resolver() -> None:
    mesh = _FakeMesh()
    edits = build_interactive_edits_from_depsgraph(
        _FakeDepsgraph([_FakeDepsgraphUpdate(mesh)]),
        usd_prim_resolver=_FakeResolver(uv_index=_uv_index()),
    )

    assert [(edit.shape, edit.data_authority) for edit in edits] == [(EditShape.VALUE, DataAuthority.VIEW)]
    assert edits[0].usd_attribute == "primvars:st"
    assert edits[0].metadata["uv_layer_name"] == "UVMap"


def test_mesh_transform_callback_does_not_fingerprint_geometry() -> None:
    mesh = _FakeMesh()
    edits = build_interactive_edits_from_depsgraph(
        _FakeDepsgraph(
            [_FakeDepsgraphUpdate(mesh, is_updated_geometry=False)]
        ),
        usd_prim_resolver=_FakeResolver(uv_index=_uv_index()),
    )

    assert edits == []


def test_uv_value_edits_from_resolver_fails_closed_for_unproven_loop_order() -> None:
    mesh = _FakeMesh(values=((0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)))

    assert uv_value_edits_from_resolver(mesh, _FakeResolver(uv_index=_uv_index())) == []


def test_material_value_edit_uses_stock_diffuse_color() -> None:
    material = _FakeBlenderId(
        name="Paint",
        blender_type="MATERIAL",
        diffuse_color=(0.1, 0.2, 0.3, 1.0),
        node_tree=_FakeNodeTree(),
    )
    material.update(
        {
            **_identity_props("/layers/look.usda"),
            usd_paths.USD_PRIM_PATH_PROP: "/World/Asset/Looks/Paint",
            USD_ATTRIBUTE_PROP: "inputs:diffuseColor",
            usd_paths.BLENDER_PROPERTY_PATH_PROP: "diffuse_color",
        }
    )

    edit = material_value_edit(material)
    plan = InteractiveEditPlanner().plan(edit)

    assert (edit.shape, edit.data_authority) == (EditShape.VALUE, DataAuthority.VIEW)
    assert edit.value == (0.1, 0.2, 0.3, 1.0)
    assert edit.usd_layer_id == "/layers/look.usda"
    assert plan.mechanism == EditMechanism.UPDATE
    assert plan.persistence == EditPersistence.WRITE


def test_material_value_edit_from_prim_needs_no_material_metadata() -> None:
    material = _FakeBlenderId(
        name="Paint",
        blender_type="MATERIAL",
        diffuse_color=(0.1, 0.2, 0.3, 1.0),
        node_tree=_FakeNodeTree(),
    )
    prim = material_usd_prim.MaterialUsdPrim(
        "/World/Looks/Paint", "/World/Looks/Paint/Shader", "inputs:diffuseColor"
    )

    edit = material_value_edit_from_prim(material, prim)
    plan = InteractiveEditPlanner().plan(edit)

    assert (edit.shape, edit.data_authority) == (EditShape.VALUE, DataAuthority.VIEW)
    assert edit.value == (0.1, 0.2, 0.3, 1.0)
    assert edit.usd_layer_id == ""
    assert edit.usd_prim_path == "/World/Looks/Paint/Shader"
    assert edit.usd_attribute == "inputs:diffuseColor"
    assert "match_source" not in edit.provenance
    assert plan.mechanism == EditMechanism.UPDATE
    assert plan.persistence == EditPersistence.NONE


def test_light_value_edit_uses_stock_energy() -> None:
    light = _FakeBlenderId(name="Key", blender_type="LIGHT", energy=900.0)
    light.update(
        {
            **_identity_props("/layers/lights.usda"),
            usd_paths.USD_PRIM_PATH_PROP: "/World/TestScene/Key",
            USD_ATTRIBUTE_PROP: "inputs:intensity",
            usd_paths.BLENDER_PROPERTY_PATH_PROP: "energy",
        }
    )

    edit = light_value_edit(light)
    plan = InteractiveEditPlanner().plan(edit)

    assert (edit.shape, edit.data_authority) == (EditShape.VALUE, DataAuthority.VIEW)
    assert edit.value == 900.0
    assert plan.mechanism == EditMechanism.UPDATE


def test_light_value_edits_from_prim_emit_topology_for_rect_to_disk_family_crossing() -> None:
    light_data = _FakeBlenderId(
        name="KeyLightData",
        blender_type="AREA",
        shape="DISK",
        energy=90.0,
        color=(1.0, 0.5, 0.25),
        use_temperature=False,
        temperature_color=(9.0, 9.0, 9.0),
        size=2.0,
        size_y=3.0,
    )
    light = _FakeBlenderId(
        name="Key",
        blender_type="LIGHT",
        data=light_data,
        scale=(2.0, 1.0, 1.0),
    )
    light[usd_paths.USD_LAYER_ID_PROP] = "/layers/lights.usda"
    prim = light_usd_prim.LightUsdPrim("/World/Key/KeyLightData", "RectLight", "AREA_RECT")

    edits = light_value_edits_from_prim(light, prim)

    assert len(edits) == 1
    assert (edits[0].shape, edits[0].data_authority) == (EditShape.TOPOLOGY, DataAuthority.VIEW)
    assert edits[0].usd_prim_path == "/World/Key/KeyLightData"
    assert edits[0].usd_attribute == ""
    assert edits[0].usd_layer_id == "/layers/lights.usda"
    assert edits[0].metadata["topology_change_kinds"] == ("light_form",)
    assert edits[0].metadata["previous_authored_light_form"] == "AREA_RECT"
    assert edits[0].metadata["current_authored_light_form"] == "AREA_DISK"
    assert edits[0].metadata["previous_usd_family"] == "RectLight"
    assert edits[0].metadata["current_usd_family"] == "DiskLight"
    assert edits[0].previous_value == "AREA_RECT"
    assert edits[0].value == "AREA_DISK"
    plan = InteractiveEditPlanner().plan(edits[0])
    assert plan.persistence == EditPersistence.WRITE
    assert plan.mechanism == EditMechanism.COMPOSE


def test_light_value_edits_from_prim_emit_topology_for_point_to_spot_form_crossing() -> None:
    light_data = _FakeBlenderId(
        name="KeyLightData",
        blender_type="SPOT",
        energy=90.0,
        color=(1.0, 0.5, 0.25),
        use_temperature=False,
        temperature_color=(9.0, 9.0, 9.0),
        shadow_soft_size=0.25,
        spot_size=math.radians(60.0),
        spot_blend=0.25,
    )
    light = _FakeBlenderId(
        name="Key",
        blender_type="LIGHT",
        data=light_data,
        scale=(1.0, 1.0, 1.0),
    )
    light[usd_paths.USD_LAYER_ID_PROP] = "/layers/lights.usda"
    prim = light_usd_prim.LightUsdPrim("/World/Key/KeyLightData", "SphereLight", "POINT")

    edits = light_value_edits_from_prim(light, prim)
    plan = InteractiveEditPlanner().plan(edits[0])

    assert len(edits) == 1
    assert (edits[0].shape, edits[0].data_authority) == (EditShape.TOPOLOGY, DataAuthority.VIEW)
    assert edits[0].usd_prim_path == "/World/Key/KeyLightData"
    assert edits[0].usd_attribute == ""
    assert edits[0].blender_property_path == "data.type"
    assert edits[0].metadata["topology_change_kinds"] == ("light_form",)
    assert edits[0].metadata["previous_authored_light_form"] == "POINT"
    assert edits[0].metadata["current_authored_light_form"] == "SPOT"
    assert edits[0].metadata["previous_usd_family"] == "SphereLight"
    assert edits[0].metadata["current_usd_family"] == "SphereLight"
    topology_attributes = {
        attribute["name"]: attribute
        for attribute in edits[0].metadata["topology_attribute_values"]
    }
    assert math.isclose(topology_attributes["inputs:shaping:cone:angle"]["value"], 30.0)
    assert topology_attributes["inputs:shaping:cone:softness"]["value"] == 0.25
    assert plan.persistence == EditPersistence.WRITE
    assert plan.mechanism == EditMechanism.COMPOSE


def test_sphere_light_without_previous_authored_form_fails_closed() -> None:
    light_data = _FakeBlenderId(
        name="KeyLightData",
        blender_type="SPOT",
        energy=90.0,
        color=(1.0, 0.5, 0.25),
        use_temperature=False,
        temperature_color=(9.0, 9.0, 9.0),
        shadow_soft_size=0.25,
        spot_size=math.radians(60.0),
        spot_blend=0.25,
    )
    light = _FakeBlenderId(
        name="Key",
        blender_type="LIGHT",
        data=light_data,
        scale=(1.0, 1.0, 1.0),
    )
    light[usd_paths.USD_LAYER_ID_PROP] = "/layers/lights.usda"
    prim = light_usd_prim.LightUsdPrim("/World/Key/KeyLightData", "SphereLight", "")

    edits = light_value_edits_from_prim(light, prim)
    plan = InteractiveEditPlanner().plan(edits[0])

    assert len(edits) == 1
    assert (edits[0].shape, edits[0].data_authority) == (EditShape.VALUE, DataAuthority.VIEW)
    assert edits[0].usd_attribute == ""
    assert edits[0].metadata["unsupported_reason"] == "missing_previous_authored_light_form"
    assert edits[0].metadata["current_authored_light_form"] == "SPOT"
    assert plan.mechanism == EditMechanism.NONE and plan.persistence == EditPersistence.NONE
    assert plan.unsupported_reason == "missing_previous_authored_light_form"


def test_light_form_crossing_without_write_target_fails_closed() -> None:
    light_data = _FakeBlenderId(
        name="KeyLightData",
        blender_type="AREA",
        shape="DISK",
        energy=90.0,
        color=(1.0, 0.5, 0.25),
        use_temperature=False,
        temperature_color=(9.0, 9.0, 9.0),
        size=2.0,
        size_y=3.0,
    )
    light = _FakeBlenderId(
        name="Key",
        blender_type="LIGHT",
        data=light_data,
        scale=(2.0, 1.0, 1.0),
    )
    prim = light_usd_prim.LightUsdPrim("/World/Key/KeyLightData", "RectLight", "AREA_RECT")

    edits = light_value_edits_from_prim(light, prim)
    plan = InteractiveEditPlanner().plan(edits[0])

    assert len(edits) == 1
    assert (edits[0].shape, edits[0].data_authority) == (EditShape.TOPOLOGY, DataAuthority.VIEW)
    assert edits[0].usd_layer_id == ""
    assert edits[0].usd_attribute == ""
    assert edits[0].metadata["topology_change_kinds"] == ("light_form",)
    assert plan.mechanism == EditMechanism.COMPOSE
    assert plan.persistence == EditPersistence.NONE
    assert plan.impact.scene_generation_replacement_requested is True


def test_depsgraph_extraction_resolves_untagged_existing_light_values() -> None:
    light_data = _FakeBlenderId(
        name="SpotData",
        blender_type="SPOT",
        session_uid=22,
        bl_rna=SimpleNamespace(identifier="Light"),
        energy=2.0,
        color=(1.0, 1.0, 1.0),
        use_temperature=True,
        temperature_color=(1.0, 0.8, 0.6),
        shadow_soft_size=0.25,
        spot_size=math.radians(60.0),
        spot_blend=0.25,
    )
    light = _FakeBlenderId(name="Spot", blender_type="LIGHT", data=light_data, scale=(1.0, 1.0, 1.0))
    index = light_usd_prim._light_prim_index_from_prims(
        (
            _FakeBlenderId(
                path="/World/Spot/SpotData",
                type_name="SphereLight",
                attributes=("inputs:shaping:cone:angle",),
            ),
        )
    )

    edits = build_interactive_edits_from_depsgraph(
        _FakeDepsgraph([_FakeDepsgraphUpdate(light_data)]),
        usd_prim_resolver=_FakeResolver(light_index=index),
        light_objects=(light,),
    )
    by_attribute = {edit.usd_attribute: edit for edit in edits}

    assert "inputs:intensity" in by_attribute
    assert by_attribute["inputs:color"].value == (1.0, 0.8, 0.6)
    assert math.isclose(by_attribute["inputs:shaping:cone:angle"].value, 30.0)
    assert by_attribute["inputs:shaping:cone:softness"].value == 0.25
    assert all(edit.usd_prim_path == "/World/Spot/SpotData" for edit in edits)
    assert all(
        edit.provenance["blender_id_kind"] == "LIGHT"
        and edit.provenance["blender_session_uid"] == 22
        for edit in edits
    )


def test_depsgraph_extraction_matches_evaluated_light_data_by_session_identity() -> None:
    light_data = _FakeBlenderId(
        name="PointData",
        blender_type="POINT",
        session_uid=22,
        energy=2.0,
        color=(1.0, 1.0, 1.0),
        use_temperature=False,
        shadow_soft_size=0.25,
    )
    evaluated_light_data = _FakeBlenderId(
        name="PointData",
        blender_type="POINT",
        session_uid=22,
    )
    light = _FakeBlenderId(
        name="Point",
        blender_type="LIGHT",
        data=light_data,
        scale=(1.0, 1.0, 1.0),
    )
    index = light_usd_prim._light_prim_index_from_prims(
        (
            _FakeBlenderId(
                path="/World/Point/PointData",
                type_name="SphereLight",
                attributes=(),
            ),
        )
    )

    edits = build_interactive_edits_from_depsgraph(
        _FakeDepsgraph([_FakeDepsgraphUpdate(evaluated_light_data)]),
        usd_prim_resolver=_FakeResolver(light_index=index),
        light_objects=(light,),
    )

    assert any(edit.usd_attribute == "inputs:intensity" for edit in edits)


def test_world_value_edits_from_prim_map_flat_world_to_studio_dome() -> None:
    world = _FakeBlenderId(
        name="World",
        blender_type="WORLD",
        use_nodes=False,
        color=(0.25, 0.5, 0.0),
    )
    index = world_dome_usd_prim._world_dome_prim_index_from_prims(
        (
            _FakeBlenderId(
                path="/World/StudioDome",
                type_name="DomeLight",
                attributes=("inputs:intensity", "inputs:color"),
            ),
        )
    )

    prim = world_dome_usd_prim.resolve_world_dome_usd_prim(index).value
    assert prim is not None
    edits = world_value_edits_from_prim(world, prim)
    by_attribute = {edit.usd_attribute: edit for edit in edits}

    assert [(edit.shape, edit.data_authority) for edit in edits] == [(EditShape.VALUE, DataAuthority.VIEW), (EditShape.VALUE, DataAuthority.VIEW)]
    assert math.isclose(by_attribute["inputs:intensity"].value, 0.5 * world_conversion.DOME_LIGHT_SCALE)
    assert by_attribute["inputs:color"].value == (0.5, 1.0, 0.0)
    assert by_attribute["inputs:intensity"].usd_prim_path == "/World/StudioDome"
    assert by_attribute["inputs:intensity"].usd_layer_id == ""
    assert by_attribute["inputs:intensity"].metadata["dome_light_scale"] == world_conversion.DOME_LIGHT_SCALE
    assert by_attribute["inputs:intensity"].provenance["world_dome_conversion"]["peak"] == 0.5


def test_world_prim_resolution_fails_closed_without_configured_dome() -> None:
    world = _FakeBlenderId(
        name="World",
        blender_type="WORLD",
        use_nodes=False,
        color=(1.0, 1.0, 1.0),
    )
    index = world_dome_usd_prim._world_dome_prim_index_from_prims(
        (_FakeBlenderId(path="/World/OtherDome", type_name="DomeLight"),)
    )

    assert world_dome_usd_prim.resolve_world_dome_usd_prim(index).value is None


def _environment_texture_world() -> _FakeBlenderId:
    return _FakeBlenderId(
        name="World",
        blender_type="WORLD",
        use_nodes=True,
        node_tree=_FakeBlenderId(
            name="WorldNodes",
            nodes=[
                _FakeBlenderId(
                    name="Background",
                    blender_type="BACKGROUND",
                    inputs={
                        "Color": _FakeSocket((1.0, 1.0, 1.0, 1.0)),
                        "Strength": _FakeSocket(1.0),
                    },
                ),
                _FakeBlenderId(name="Environment", blender_type="TEX_ENVIRONMENT"),
            ],
        ),
    )


def _studio_dome_index() -> dict:
    return world_dome_usd_prim._world_dome_prim_index_from_prims(
        (
            _FakeBlenderId(
                path="/World/StudioDome",
                type_name="DomeLight",
                attributes=("inputs:intensity", "inputs:color"),
            ),
        )
    )


def test_world_environment_texture_builds_topology_edit_on_generation_route() -> None:
    world = _environment_texture_world()
    prim = world_dome_usd_prim.resolve_world_dome_usd_prim(_studio_dome_index()).value
    assert prim is not None

    edits = world_value_edits_from_prim(world, prim)

    assert len(edits) == 1
    edit = edits[0]
    assert (edit.shape, edit.data_authority) == (EditShape.TOPOLOGY, DataAuthority.VIEW)
    assert edit.usd_prim_path == "/World/StudioDome"
    assert edit.blender_property_path == "node_tree"
    assert edit.metadata["topology_change_kinds"] == ("environment_texture",)
    assert edit.metadata["world_topology_reason"] == world_conversion.ENVIRONMENT_TEXTURE_CHANGED

    plan = InteractiveEditPlanner().plan(edit)
    assert plan.mechanism == EditMechanism.COMPOSE
    assert plan.impact.authoring_reconciliation_requested is True
    assert plan.impact.render_session_reuse_expected is False
    assert plan.impact.refinement_reset_expected is True
    assert world_conversion.ENVIRONMENT_TEXTURE_CHANGED in plan.impact.topology_reasons


def test_node_graph_world_builds_world_node_graph_topology_edit() -> None:
    # Sky-texture-shaped world: a node drives the Background color input,
    # so the world is node-based -> topology for the live-edit route.
    world = _FakeBlenderId(
        name="World",
        blender_type="WORLD",
        use_nodes=True,
        node_tree=_FakeBlenderId(
            name="WorldNodes",
            nodes=[
                _FakeBlenderId(
                    name="Background",
                    blender_type="BACKGROUND",
                    inputs={
                        "Color": _FakeSocket((1.0, 1.0, 1.0, 1.0), linked=True),
                        "Strength": _FakeSocket(1.0),
                    },
                ),
                _FakeBlenderId(name="Sky", blender_type="TEX_SKY"),
            ],
        ),
    )

    edits = build_interactive_edits_from_depsgraph(
        _FakeDepsgraph([_FakeDepsgraphUpdate(world)]),
        usd_prim_resolver=_FakeResolver(world_dome_index=_studio_dome_index()),
    )

    assert len(edits) == 1
    edit = edits[0]
    assert edit.shape == EditShape.TOPOLOGY
    assert edit.metadata["topology_change_kinds"] == ("world_node_graph",)
    assert edit.metadata["world_topology_reason"] == world_conversion.WORLD_NODE_GRAPH_CHANGED
    plan = InteractiveEditPlanner().plan(edit)
    assert plan.mechanism == EditMechanism.COMPOSE
    assert plan.impact.authoring_reconciliation_requested is True
    assert world_conversion.WORLD_NODE_GRAPH_CHANGED in plan.impact.topology_reasons


def test_world_added_where_none_authored_is_world_assignment_topology() -> None:
    world = _FakeBlenderId(
        name="Fresh World",
        blender_type="WORLD",
        use_nodes=False,
        color=(0.2, 0.4, 0.1),
    )
    # Authored generation without a dome prim: the scene had no world when
    # the generation was composed.
    empty_index = world_dome_usd_prim._world_dome_prim_index_from_prims(())

    edits = build_interactive_edits_from_depsgraph(
        _FakeDepsgraph([_FakeDepsgraphUpdate(world)]),
        usd_prim_resolver=_FakeResolver(world_dome_index=empty_index),
    )

    assert len(edits) == 1
    edit = edits[0]
    assert (edit.shape, edit.data_authority) == (EditShape.TOPOLOGY, DataAuthority.VIEW)
    assert edit.usd_prim_path == world_conversion.DEFAULT_DOME_OWNER_PATH
    assert edit.blender_property_path == "world"
    assert (edit.value, edit.previous_value) == ("world", "none")
    assert edit.metadata["topology_change_kinds"] == ("world_assignment",)
    assert edit.metadata["world_present"] is True
    assert edit.metadata["authored_dome_present"] is False

    plan = InteractiveEditPlanner().plan(edit)
    assert plan.mechanism == EditMechanism.COMPOSE
    assert plan.impact.authoring_reconciliation_requested is True
    assert plan.impact.whole_scene_export_requested is True
    assert plan.impact.whole_scene_export_avoided is False
    assert "world_datablock_assignment_is_topology" in plan.impact.topology_reasons


def test_scene_update_with_removed_world_builds_no_world_edit() -> None:
    scene = _FakeBlenderId(
        name="Scene",
        blender_type="SCENE",
        world=None,
        render=SimpleNamespace(engine="OVRTX"),
    )

    edits = build_interactive_edits_from_depsgraph(
        _FakeDepsgraph([_FakeDepsgraphUpdate(scene)]),
        usd_prim_resolver=_FakeResolver(world_dome_index=_studio_dome_index()),
    )

    assert edits == []


def test_scene_and_world_updates_for_one_change_dedupe_the_world_lane() -> None:
    world = _FakeBlenderId(
        name="Fresh World",
        blender_type="WORLD",
        use_nodes=False,
        color=(0.2, 0.4, 0.1),
    )
    scene = _FakeBlenderId(
        name="Scene",
        blender_type="SCENE",
        world=world,
        render=SimpleNamespace(engine="OVRTX"),
    )
    empty_index = world_dome_usd_prim._world_dome_prim_index_from_prims(())

    # World added: both the Scene and the World ID report -> one edit set.
    added_edits = build_interactive_edits_from_depsgraph(
        _FakeDepsgraph([_FakeDepsgraphUpdate(scene), _FakeDepsgraphUpdate(world)]),
        usd_prim_resolver=_FakeResolver(world_dome_index=empty_index),
    )
    assert len(added_edits) == 1
    assert added_edits[0].metadata["topology_change_kinds"] == ("world_assignment",)

    # Value edit: duplicate World updates in one event -> one edit set.
    value_edits = build_interactive_edits_from_depsgraph(
        _FakeDepsgraph([_FakeDepsgraphUpdate(world), _FakeDepsgraphUpdate(world)]),
        usd_prim_resolver=_FakeResolver(world_dome_index=_studio_dome_index()),
    )
    assert sorted(edit.usd_attribute for edit in value_edits) == [
        "inputs:color",
        "inputs:intensity",
    ]


def test_scene_update_without_world_divergence_builds_no_world_edits() -> None:
    # Unrelated scene-level changes must not re-emit dome values (that
    # would reset refinement on every scene edit): world value edits ride
    # World ID updates only.
    world = _FakeBlenderId(
        name="World",
        blender_type="WORLD",
        use_nodes=False,
        color=(0.2, 0.4, 0.1),
    )
    scene = _FakeBlenderId(
        name="Scene",
        blender_type="SCENE",
        world=world,
        render=SimpleNamespace(engine="OVRTX"),
    )

    edits = build_interactive_edits_from_depsgraph(
        _FakeDepsgraph([_FakeDepsgraphUpdate(scene)]),
        usd_prim_resolver=_FakeResolver(world_dome_index=_studio_dome_index()),
    )

    assert edits == []


def test_scene_update_with_missing_dome_builds_no_world_edits() -> None:
    world = _FakeBlenderId(
        name="World",
        blender_type="WORLD",
        use_nodes=False,
        color=(0.2, 0.4, 0.1),
    )
    scene = _FakeBlenderId(
        name="Scene",
        blender_type="SCENE",
        world=world,
        render=SimpleNamespace(engine="OVRTX"),
    )

    edits = build_interactive_edits_from_depsgraph(
        _FakeDepsgraph([_FakeDepsgraphUpdate(scene)]),
        usd_prim_resolver=_FakeResolver(
            world_dome_index=world_dome_usd_prim._world_dome_prim_index_from_prims(())
        ),
    )

    assert edits == []


def test_light_data_update_with_authored_dome_builds_no_world_edits() -> None:
    # bpy.types.Light has ``color`` and ``use_nodes`` like a World; a light
    # data edit must stay in the light lane and never rewrite the dome
    # (task04-04 review regression).
    light_data = _FakeBlenderId(
        name="KeyData",
        blender_type="AREA",
        shape="SQUARE",
        energy=120.0,
        color=(1.0, 0.8, 0.6),
        use_nodes=False,
        size=2.0,
        size_y=2.0,
    )
    light = _FakeBlenderId(name="Key", blender_type="LIGHT", data=light_data)
    light_index = light_usd_prim._light_prim_index_from_prims(
        (
            _FakeBlenderId(
                path="/World/Lights/Key",
                type_name="RectLight",
                attributes=("inputs:intensity",),
            ),
        )
    )

    edits = build_interactive_edits_from_depsgraph(
        _FakeDepsgraph([_FakeDepsgraphUpdate(light_data)]),
        usd_prim_resolver=_FakeResolver(
            light_index=light_index,
            world_dome_index=_studio_dome_index(),
        ),
        light_objects=(light,),
    )

    assert edits, "the light lane still emits its value edits"
    assert all(edit.usd_prim_path == "/World/Lights/Key" for edit in edits)
    assert all("dome_owner_path" not in edit.provenance for edit in edits)


def test_zero_strength_black_world_stays_an_ordinary_value_update() -> None:
    # Pure black / zero strength are ordinary values (task04-04): normal
    # UPDATE mechanism, applied through the view stream with the parity
    # values world_dome_conversion computes — no visibility special-case.
    world = _FakeBlenderId(
        name="World",
        blender_type="WORLD",
        use_nodes=False,
        color=(0.0, 0.0, 0.0),
    )

    edits = build_interactive_edits_from_depsgraph(
        _FakeDepsgraph([_FakeDepsgraphUpdate(world)]),
        usd_prim_resolver=_FakeResolver(world_dome_index=_studio_dome_index()),
    )

    policy_values = {
        attribute.name: attribute.value
        for attribute in world_conversion.usd_attribute_values(world)
    }
    assert policy_values == {"inputs:intensity": 0.0, "inputs:color": (0.0, 0.0, 0.0)}
    assert {edit.usd_attribute for edit in edits} == set(policy_values)
    for edit in edits:
        assert edit.shape == EditShape.VALUE
        assert edit.value == policy_values[edit.usd_attribute]
        plan = InteractiveEditPlanner().plan(edit)
        assert plan.mechanism == EditMechanism.UPDATE
        assert plan.impact.render_session_reuse_expected is True
        stream = ViewUpdateStream()
        assert stream.queue(plan.to_intent())["queued"] is True
        port = _AttributeRecordingPort()
        result = stream.apply_pending(port)
        assert result["values_written"] is True
        assert len(port.batches) == 1 and len(port.batches[0]) == 1
        applied = port.batches[0][0]
        assert applied.prim_path == "/World/StudioDome"
        assert applied.attribute == edit.usd_attribute
        expected = policy_values[edit.usd_attribute]
        assert applied.value == (list(expected) if isinstance(expected, tuple) else expected)


def test_world_value_edits_use_injected_policy_without_default_policy_veto() -> None:
    world = _FakeBlenderId(
        name="World",
        blender_type="WORLD",
        use_nodes=True,
        node_tree=_FakeBlenderId(
            name="WorldNodes",
            nodes=[_FakeBlenderId(name="Environment", blender_type="TEX_ENVIRONMENT")],
        ),
    )
    index = world_dome_usd_prim._world_dome_prim_index_from_prims(
        (
            _FakeBlenderId(
                path="/World/StudioDome",
                type_name="DomeLight",
                attributes=("inputs:intensity", "inputs:color"),
            ),
        )
    )

    class _CustomWorldPolicy:
        SUPPORTED_USD_ATTRIBUTES = {"inputs:intensity": "Float"}

        @staticmethod
        def classify_field(_world: object, _property_name: str) -> object:
            raise AssertionError("classification is not part of edit construction")

        @staticmethod
        def usd_attribute_values(_world: object) -> tuple[UsdAttributeValue, ...]:
            return (
                UsdAttributeValue(
                    "inputs:intensity",
                    42.0,
                    "Float",
                    "custom_environment",
                    {"conversion_policy": "custom_world", "custom": True},
                ),
            )

    defaults = default_value_edit_conversion_policies()
    policies = ValueEditConversionPolicies(
        material=defaults.material,
        light=defaults.light,
        world=_CustomWorldPolicy(),
    )

    prim = world_dome_usd_prim.resolve_world_dome_usd_prim(index).value
    assert prim is not None
    edits = world_value_edits_from_prim(
        world,
        prim,
        value_edit_conversion_policies=policies,
    )

    assert len(edits) == 1
    assert edits[0].value == 42.0
    assert edits[0].blender_property_path == "custom_environment"
    assert edits[0].provenance["world_dome_conversion"]["custom"] is True


def test_depsgraph_extraction_resolves_untagged_world_dome_values() -> None:
    world = _FakeBlenderId(
        name="World",
        blender_type="WORLD",
        use_nodes=False,
        color=(0.2, 0.4, 0.1),
    )
    index = world_dome_usd_prim._world_dome_prim_index_from_prims(
        (
            _FakeBlenderId(
                path="/World/StudioDome",
                type_name="DomeLight",
                attributes=("inputs:intensity", "inputs:color"),
            ),
        )
    )

    edits = build_interactive_edits_from_depsgraph(
        _FakeDepsgraph([_FakeDepsgraphUpdate(world)]),
        usd_prim_resolver=_FakeResolver(world_dome_index=index),
    )
    by_attribute = {edit.usd_attribute: edit for edit in edits}

    assert [(edit.shape, edit.data_authority) for edit in edits] == [(EditShape.VALUE, DataAuthority.VIEW), (EditShape.VALUE, DataAuthority.VIEW)]
    assert by_attribute["inputs:color"].value == (0.5, 1.0, 0.25)
    assert all(edit.usd_prim_path == "/World/StudioDome" for edit in edits)


def test_generic_property_edit_supports_physics_and_semantic_metadata() -> None:
    obj = _FakeBlenderId(name="Cube", mass=2.5)
    obj.update(
        {
            **_identity_props("/layers/physics.usda"),
            USD_ATTRIBUTE_PROP: "physics:mass",
            usd_paths.BLENDER_PROPERTY_PATH_PROP: "mass",
        }
    )

    edit = property_edit(
        obj,
        data_authority=DataAuthority.SIM,
        property_name="mass",
        usd_attribute="physics:mass",
    )

    assert (edit.shape, edit.data_authority) == (EditShape.VALUE, DataAuthority.SIM)
    assert edit.value == 2.5
    assert edit.usd_attribute == "physics:mass"
    plan = InteractiveEditPlanner().plan(edit)
    assert plan.mechanism == EditMechanism.NONE
    assert plan.persistence == EditPersistence.WRITE


def test_missing_stock_identity_becomes_unsupported_planner_input() -> None:
    obj = _FakeBlenderId(name="LooseCube", matrix_world=((1, 0, 0, 0),) * 4)
    obj[usd_paths.DATA_AUTHORITY_PROP] = "view"

    edit = object_transform_edit(obj)
    plan = InteractiveEditPlanner().plan(edit)

    assert edit.usd_layer_id == ""
    assert edit.usd_prim_path == ""
    assert plan.mechanism == EditMechanism.NONE and plan.persistence == EditPersistence.NONE
    assert plan.unsupported_reason == "missing_edit_identity"


def test_depsgraph_extraction_builds_interactive_edits_from_stock_updates() -> None:
    obj = _FakeBlenderId(
        name="Cube",
        matrix_world=((1, 0, 0, 2), (0, 1, 0, 3), (0, 0, 1, 4), (0, 0, 0, 1)),
    )
    obj.update(_identity_props())
    material = _FakeBlenderId(
        name="Paint",
        blender_type="MATERIAL",
        diffuse_color=(0.1, 0.2, 0.3, 1.0),
    )
    material.update(
        {
            **_identity_props("/layers/look.usda"),
            usd_paths.DATA_AUTHORITY_PROP: "view",
            usd_paths.USD_PRIM_PATH_PROP: "/World/Asset/Looks/Paint",
            USD_ATTRIBUTE_PROP: "inputs:diffuseColor",
            usd_paths.BLENDER_PROPERTY_PATH_PROP: "diffuse_color",
        }
    )

    edits = build_interactive_edits_from_depsgraph(
        _FakeDepsgraph([_FakeDepsgraphUpdate(obj), _FakeDepsgraphUpdate(material)])
    )

    assert [(edit.shape, edit.data_authority) for edit in edits] == [(EditShape.VALUE, DataAuthority.VIEW), (EditShape.VALUE, DataAuthority.VIEW)]
    assert edits[0].usd_prim_path == "/World/TestScene/Cube"
    assert edits[1].usd_prim_path == "/World/Asset/Looks/Paint"


def test_depsgraph_extraction_resolves_untagged_material_from_resolver() -> None:
    material = _FakeBlenderId(
        name="Paint",
        blender_type="MATERIAL",
        diffuse_color=(0.1, 0.2, 0.3, 1.0),
        node_tree=_FakeNodeTree(),
    )
    index = material_usd_prim._material_prim_index_from_prims(
        (
            _FakeBlenderId(path="/World/Looks/Paint", type_name="Material"),
            _FakeBlenderId(
                path="/World/Looks/Paint/Shader",
                type_name="Shader",
                attributes=("inputs:diffuseColor",),
                info_id="UsdPreviewSurface",
            ),
        )
    )

    edits = build_interactive_edits_from_depsgraph(
        _FakeDepsgraph([_FakeDepsgraphUpdate(material)]),
        usd_prim_resolver=_FakeResolver(
            material_indexes={material_usd_prim.DEFAULT_USD_MATERIAL_ATTRIBUTE: index}
        ),
    )

    assert [(edit.shape, edit.data_authority) for edit in edits] == [(EditShape.VALUE, DataAuthority.VIEW)]
    assert edits[0].usd_prim_path == "/World/Looks/Paint/Shader"
    assert edits[0].usd_layer_id == ""


def test_direct_material_edit_ignores_unrelated_same_name_selection() -> None:
    material = _FakeBlenderId(
        name="Paint",
        blender_type="MATERIAL",
        session_uid=303,
        diffuse_color=(0.1, 0.2, 0.3, 1.0),
        node_tree=_FakeNodeTree(),
    )
    index = material_usd_prim._material_prim_index_from_prims(
        (
            _FakeBlenderId(path="/World/Looks/Paint", type_name="Material"),
            _FakeBlenderId(
                path="/World/Looks/Paint/Shader",
                type_name="Shader",
                attributes=("inputs:diffuseColor",),
                info_id="UsdPreviewSurface",
            ),
        )
    )
    selection_resolution = {
        "sources": [
            {
                "source_name": "Paint",
                "owner_name": "Paint",
                "owner_usd_path": "/World/Looks/Paint/Shader",
                "source_session_uid": 101,
                "status": "resolved",
            }
        ]
    }

    edits = build_interactive_edits_from_depsgraph(
        _FakeDepsgraph([_FakeDepsgraphUpdate(material)]),
        usd_prim_resolver=_FakeResolver(
            material_indexes={material_usd_prim.DEFAULT_USD_MATERIAL_ATTRIBUTE: index}
        ),
        selection_resolution=selection_resolution,
    )

    assert len(edits) == 1
    assert "selection_resolution" not in edits[0].provenance


def test_depsgraph_extraction_resolves_supported_existing_material_values() -> None:
    material = _FakeBlenderId(
        name="Paint",
        blender_type="MATERIAL",
        diffuse_color=(0.1, 0.2, 0.3, 1.0),
        node_tree=_FakeNodeTree(),
    )
    preview_attributes = {
        "inputs:diffuseColor",
        "inputs:roughness",
        "inputs:metallic",
        "inputs:ior",
        "inputs:emissiveColor",
    }
    indexes = {}
    for usd_attribute in material_conversion.SUPPORTED_USD_ATTRIBUTES:
        shader_info_id = (
            "UsdPreviewSurface"
            if usd_attribute in preview_attributes
            else "ND_open_pbr_surface_surfaceshader"
        )
        indexes[usd_attribute] = material_usd_prim._material_prim_index_from_prims(
            (
                _FakeBlenderId(path="/World/Looks/Paint", type_name="Material"),
                _FakeBlenderId(
                    path="/World/Looks/Paint/Shader",
                    type_name="Shader",
                    attributes=(usd_attribute,),
                    info_id=shader_info_id,
                ),
            ),
            usd_attribute=usd_attribute,
        )

    edits = build_interactive_edits_from_depsgraph(
        _FakeDepsgraph([_FakeDepsgraphUpdate(material)]),
        usd_prim_resolver=_FakeResolver(material_indexes=indexes),
    )

    assert [edit.usd_attribute for edit in edits] == [
        "inputs:diffuseColor",
        "inputs:base_color",
        "inputs:roughness",
        "inputs:specular_roughness",
        "inputs:metallic",
        "inputs:base_metalness",
        "inputs:ior",
        "inputs:specular_ior",
        "inputs:geometry_opacity",
        "inputs:emissiveColor",
        "inputs:emission_color",
        "inputs:emission_luminance",
    ]
    assert [edit.value for edit in edits] == [
        (0.1, 0.2, 0.3),
        (0.1, 0.2, 0.3),
        0.45,
        0.45,
        0.2,
        0.2,
        1.45,
        1.45,
        0.75,
        (0.2, 0.4, 1.0),
        (0.1, 0.2, 0.5),
        2.0 * 120.0 * math.pi * math.pi,
    ]


def test_depsgraph_extraction_routes_linked_material_input_as_graph_topology() -> None:
    material = _FakeBlenderId(
        name="Paint",
        blender_type="MATERIAL",
        node_tree=_FakeNodeTree(),
    )
    material[usd_paths.SOURCE_USD_PATH_PROP] = "/World/Looks/Paint"
    material.node_tree.nodes[0].inputs["Base Color"] = _FakeSocket((0.1, 0.2, 0.3, 1.0), linked=True)

    edits = build_interactive_edits_from_depsgraph(
        _FakeDepsgraph([_FakeDepsgraphUpdate(material)]),
        usd_prim_resolver=_FakeResolver(),
    )
    plan = InteractiveEditPlanner().plan(edits[0])

    assert [(edit.shape, edit.data_authority) for edit in edits] == [(EditShape.TOPOLOGY, DataAuthority.VIEW)]
    assert edits[0].usd_prim_path == "/World/Looks/Paint"
    assert edits[0].blender_property_path == "node_tree"
    assert edits[0].metadata["topology_change_kinds"] == ("material_graph",)
    assert plan.mechanism == EditMechanism.COMPOSE


def test_depsgraph_material_prim_records_write_target_resolution(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def _resolve(input_usd_path: str, **kwargs: object) -> ownership.WriteTargetResolutionResult:
        calls.append((input_usd_path, dict(kwargs)))
        return ownership.WriteTargetResolutionResult(
            ownership.WriteTargetResolutionStatus.OK,
            usd_layer_id="/layers/look.usda",
            diagnostics={
                "target_kind": kwargs["target_kind"],
                "ignored_layer_identifiers": list(kwargs["ignored_layer_identifiers"]),
                "stack_resolved_identifier": "/layers/look.usda",
            },
        )

    monkeypatch.setattr(builders.write_target_resolution, "resolve_write_target", _resolve)
    material = _FakeBlenderId(
        name="Paint",
        blender_type="MATERIAL",
        diffuse_color=(0.1, 0.2, 0.3, 1.0),
        node_tree=_FakeNodeTree(),
    )
    indexes = {
        usd_attribute: material_usd_prim._material_prim_index_from_prims(
            (
                _FakeBlenderId(path="/World/Looks/Paint", type_name="Material"),
                _FakeBlenderId(
                    path="/World/Looks/Paint/Shader",
                    type_name="Shader",
                    attributes=(usd_attribute,),
                    info_id="UsdPreviewSurface",
                ),
            ),
            usd_attribute=usd_attribute,
        )
        for usd_attribute in material_conversion.SUPPORTED_USD_ATTRIBUTES
    }

    edits = build_interactive_edits_from_depsgraph(
        _FakeDepsgraph([_FakeDepsgraphUpdate(material)]),
        usd_prim_resolver=_FakeResolver(material_indexes=indexes),
        write_target_input_usd_path="/fixtures/composed.usda",
        write_target_ignored_layer_identifiers=("/tmp/session-layer.usda",),
    )
    diffuse = next(edit for edit in edits if edit.usd_attribute == "inputs:diffuseColor")

    assert calls
    assert {call[0] for call in calls} == {"/fixtures/composed.usda"}
    assert {call[1]["target_kind"] for call in calls} == {ownership.TARGET_KIND_ATTRIBUTE}
    assert diffuse.usd_layer_id == "/layers/look.usda"
    assert "write_target_resolution" not in diffuse.provenance
    assert "write_target_error_reason" not in diffuse.provenance


def test_depsgraph_existing_light_topology_records_prim_write_target_resolution(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def _resolve(input_usd_path: str, **kwargs: object) -> ownership.WriteTargetResolutionResult:
        calls.append((input_usd_path, dict(kwargs)))
        return ownership.WriteTargetResolutionResult(
            ownership.WriteTargetResolutionStatus.OK,
            usd_layer_id="/layers/lights.usda",
            diagnostics={
                "target_kind": kwargs["target_kind"],
                "usd_prim_path": kwargs["usd_prim_path"],
                "stack_resolved_identifier": "/layers/lights.usda",
            },
        )

    monkeypatch.setattr(builders.write_target_resolution, "resolve_write_target", _resolve)
    light_data = _FakeBlenderId(
        name="KeyLightData",
        blender_type="AREA",
        shape="DISK",
        energy=90.0,
        color=(1.0, 1.0, 1.0),
        size=2.0,
        size_y=3.0,
    )
    light = _FakeBlenderId(name="Key", blender_type="LIGHT", data=light_data)
    light[usd_paths.SOURCE_USD_PATH_PROP] = "/World/Key/KeyLightData"
    index = light_usd_prim._light_prim_index_from_prims(
        (_FakeBlenderId(path="/World/Key/KeyLightData", type_name="RectLight"),)
    )

    edits = build_interactive_edits_from_depsgraph(
        _FakeDepsgraph([_FakeDepsgraphUpdate(light)]),
        usd_prim_resolver=_FakeResolver(light_index=index),
        light_objects=(light,),
        write_target_input_usd_path="/fixtures/composed.usda",
    )

    assert len(edits) == 1
    assert (edits[0].shape, edits[0].data_authority) == (EditShape.TOPOLOGY, DataAuthority.VIEW)
    assert edits[0].usd_layer_id == "/layers/lights.usda"
    assert calls[0][1]["target_kind"] == ownership.TARGET_KIND_PRIM
    assert "write_target_resolution" not in edits[0].provenance
    assert "write_target_error_reason" not in edits[0].provenance


def test_depsgraph_extraction_skips_updates_without_explicit_data_authority() -> None:
    obj = _FakeBlenderId(
        name="Cube",
        matrix_world=((1, 0, 0, 2), (0, 1, 0, 3), (0, 0, 1, 4), (0, 0, 0, 1)),
    )
    props = _identity_props()
    props.pop(usd_paths.DATA_AUTHORITY_PROP)
    obj.update(props)

    assert build_interactive_edits_from_depsgraph(_FakeDepsgraph([_FakeDepsgraphUpdate(obj)])) == []


def test_depsgraph_extraction_uses_unique_exported_blender_object_identity(monkeypatch) -> None:
    obj = _FakeBlenderId(
        name="Object With Spaces",
        matrix_world=((1, 0, 0, 2), (0, 1, 0, 3), (0, 0, 1, 4), (0, 0, 0, 1)),
    )
    prim = _FakeUsdPrim(
        "/Object_With_Spaces",
        {"userProperties:blender:object_name": _FakeUsdAttr("Object With Spaces")},
    )
    prim.GetTypeName = lambda: "Xform"
    resolver = builders.UsdPrimResolver()
    monkeypatch.setattr(
        usd_prim_resolver,
        "_open_stage_prims",
        lambda path: (object(), (prim,), None),
    )
    resolver.scan(type("Request", (), {"input_usd_path": "/tmp/generated.usdc"})())
    monkeypatch.setattr(
        builders.write_target_resolution,
        "resolve_write_target",
        lambda *args, **kwargs: ownership.WriteTargetResolutionResult(
            ownership.WriteTargetResolutionStatus.OK,
            usd_layer_id="/tmp/generated.usdc",
        ),
    )

    edits = build_interactive_edits_from_depsgraph(
        _FakeDepsgraph([_FakeDepsgraphUpdate(obj)]),
        usd_prim_resolver=resolver,
        write_target_input_usd_path="/tmp/generated.usdc",
    )

    assert len(edits) == 1
    assert edits[0].usd_prim_path == "/Object_With_Spaces"
    assert edits[0].data_authority is DataAuthority.VIEW
    assert edits[0].provenance["match_source"] == "blender_object_name"


# --- Transform value edits as live overs (task04-01) -------------------------
#
# Depsgraph object-transform edits over the authored scene composition:
# meshes and lights resolve to their converter-authored object roots via the
# ``ov.usd.prim_path`` authoring identity, plan as UPDATE with view
# authority, and apply as ``OvrtxTransformValue`` batches (``omni:xform`` +
# xform-stack reset in the client) in the USD row convention the composed
# scene and the physics pose lane already use.


_BLENDER_ROTZ90_T123 = (
    (0.0, -1.0, 0.0, 1.0),
    (1.0, 0.0, 0.0, 2.0),
    (0.0, 0.0, 1.0, 3.0),
    (0.0, 0.0, 0.0, 1.0),
)
_USD_ROWS_ROTZ90_T123 = [
    [0.0, 1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [1.0, 2.0, 3.0, 1.0],
]


class _AuthoredScenePrim:
    """Prim fake shaped like the authored generation output (no name attrs)."""

    def __init__(
        self,
        path: str,
        type_name: str,
        *,
        attributes: tuple[str, ...] = (),
        info_id: str = "",
        connected_attributes: tuple[str, ...] = (),
    ) -> None:
        self.path = path
        self.type_name = type_name
        self.attributes = attributes
        self.info_id = info_id
        self.connected_attributes = connected_attributes

    def GetAttribute(self, name: str) -> object | None:
        return None


def _authored_generation_prims() -> tuple[_AuthoredScenePrim, ...]:
    return (
        _AuthoredScenePrim("/World/Cube", "Xform"),
        _AuthoredScenePrim("/World/Cube/Mesh", "Mesh"),
        _AuthoredScenePrim("/World/Lights/Key", "RectLight"),
    )


def _authored_scan_resolver(monkeypatch) -> usd_prim_resolver.UsdPrimResolver:
    monkeypatch.setattr(
        usd_prim_resolver,
        "_open_stage_prims",
        lambda path: (object(), _authored_generation_prims(), None),
    )
    monkeypatch.setattr(
        builders.write_target_resolution,
        "resolve_write_target",
        lambda *args, **kwargs: ownership.WriteTargetResolutionResult(
            ownership.WriteTargetResolutionStatus.ERROR,
            error_reason=ownership.REASON_STAGE_OPEN_FAILED,
        ),
    )
    resolver = builders.UsdPrimResolver()
    resolver.scan(type("Request", (), {"input_usd_path": "/tmp/generation.usdc"})())
    return resolver


def _authored_blender_object(
    name: str,
    blender_type: str,
    prim_path: str,
    **attrs: object,
) -> _FakeBlenderId:
    obj = _FakeBlenderId(name=name, blender_type=blender_type, **attrs)
    obj.ov = SimpleNamespace(usd=SimpleNamespace(prim_path=prim_path))
    return obj


class _TransformRecordingPort:
    def __init__(self) -> None:
        self.batches: list[list[OvrtxTransformValue]] = []

    def update_transforms(self, values) -> OvrtxValueUpdateResult:
        batch = list(values)
        self.batches.append(batch)
        return OvrtxValueUpdateResult(len(batch), 1 if batch else None)


def _apply_transform_intent_through_view_stream(plan) -> tuple[dict, _TransformRecordingPort]:
    """Fake-client application half: planner intent -> OvrtxTransformValue."""

    assert plan.mechanism == EditMechanism.UPDATE
    stream = ViewUpdateStream()
    queued = stream.queue(plan.to_intent())
    assert queued["queued"] is True
    port = _TransformRecordingPort()
    return stream.apply_pending(port), port


def test_depsgraph_mesh_object_transform_is_a_view_update_on_the_authored_root(monkeypatch) -> None:
    resolver = _authored_scan_resolver(monkeypatch)
    obj = _authored_blender_object("Cube", "MESH", "/World/Cube", matrix_world=_BLENDER_ROTZ90_T123)

    edits = build_interactive_edits_from_depsgraph(
        _FakeDepsgraph([_FakeDepsgraphUpdate(obj)]),
        usd_prim_resolver=resolver,
    )

    assert len(edits) == 1
    edit = edits[0]
    assert (edit.shape, edit.data_authority) == (EditShape.VALUE, DataAuthority.VIEW)
    assert edit.usd_prim_path == "/World/Cube"
    assert edit.usd_attribute == "xformOp:transform"
    assert edit.blender_property_path == "matrix_world"
    assert edit.value == _USD_ROWS_ROTZ90_T123
    assert edit.provenance["match_source"] == "authoring_prim_path"

    plan = InteractiveEditPlanner().plan(edit)
    assert plan.mechanism == EditMechanism.UPDATE
    assert plan.impact.render_session_reuse_expected is True
    result, port = _apply_transform_intent_through_view_stream(plan)
    assert result["values_written"] is True
    assert port.batches == [[OvrtxTransformValue("/World/Cube", _USD_ROWS_ROTZ90_T123)]]


def test_depsgraph_light_object_transform_targets_the_light_root_prim(monkeypatch) -> None:
    resolver = _authored_scan_resolver(monkeypatch)
    light_data = _FakeBlenderId(
        name="KeyData",
        blender_type="AREA",
        shape="SQUARE",
        energy=90.0,
        color=(1.0, 1.0, 1.0),
        size=2.0,
        size_y=2.0,
    )
    light = _authored_blender_object(
        "Key",
        "LIGHT",
        "/World/Lights/Key",
        matrix_world=_BLENDER_ROTZ90_T123,
        data=light_data,
    )

    edits = build_interactive_edits_from_depsgraph(
        _FakeDepsgraph([_FakeDepsgraphUpdate(light)]),
        usd_prim_resolver=resolver,
        light_objects=(light,),
    )

    transform_edits = [edit for edit in edits if edit.blender_property_path == "matrix_world"]
    assert len(transform_edits) == 1
    edit = transform_edits[0]
    # The converter authors the UsdLux prim *as* the object root; the
    # transform edit targets that root, not a child light prim.
    assert edit.usd_prim_path == "/World/Lights/Key"
    assert (edit.shape, edit.data_authority) == (EditShape.VALUE, DataAuthority.VIEW)
    assert edit.value == _USD_ROWS_ROTZ90_T123
    assert edit.provenance["match_source"] == "authoring_prim_path"
    # Light data value edits (if any) ride along; none may be a topology
    # edit for a pure move (no session replacement on light motion).
    assert all(other.shape == EditShape.VALUE for other in edits)

    plan = InteractiveEditPlanner().plan(edit)
    result, port = _apply_transform_intent_through_view_stream(plan)
    assert result["values_written"] is True
    assert port.batches == [[OvrtxTransformValue("/World/Lights/Key", _USD_ROWS_ROTZ90_T123)]]


def test_depsgraph_camera_object_without_authored_prim_builds_no_transform_edit(monkeypatch) -> None:
    resolver = _authored_scan_resolver(monkeypatch)
    camera = _authored_blender_object("Camera", "CAMERA", "", matrix_world=_BLENDER_ROTZ90_T123)

    edits = build_interactive_edits_from_depsgraph(
        _FakeDepsgraph([_FakeDepsgraphUpdate(camera)]),
        usd_prim_resolver=resolver,
    )

    # The converters emit no camera prims: a camera object move produces no
    # (mis)targeted transform edit. Rendered camera motion routes through
    # the viewport-camera lane (camera mailbox -> omni:xform on the render
    # camera prim) when the viewport looks through the camera, and through
    # the composed scene-camera pose on final render (task01-03).
    assert edits == []


def test_object_transform_rows_match_composed_scene_and_physics_pose_lane() -> None:
    """Cross-lane matrix convention pinning (task01-03 follow-up).

    The worker's ``omni:xform`` write consumes USD row-vector matrices
    (translation in the fourth row): the physics pose lane that renders
    motion today authors exactly that shape. The edit-builder lane and the
    composed-scene converters must agree with it.
    """

    usd_rows = builders._matrix_rows(_BLENDER_ROTZ90_T123)
    assert usd_rows == _USD_ROWS_ROTZ90_T123
    assert usd_rows == [
        list(row) for row in render_requests.matrix_to_usd_rows(_BLENDER_ROTZ90_T123)
    ]
    half_sqrt2 = math.sin(math.pi / 4.0)
    physics_rows = ovphysx_to_ovrtx._matrix4d_rows(
        (1.0, 2.0, 3.0),
        (0.0, 0.0, half_sqrt2, half_sqrt2),
        1.0,
    )
    for physics_row, usd_row in zip(physics_rows, usd_rows):
        assert physics_row == pytest.approx(usd_row, abs=1e-9)


# --- Material value edits as live overs (task04-02) --------------------------
#
# Depsgraph material value edits over the authored scene composition: the
# material resolves to its converter-authored Material/PreviewSurface prims
# via the ``ov.usd.prim_path`` authoring identity (material names can
# sanitize differently from their USD leaf names), each supported field
# routes to its authored ``inputs:*`` attribute with the shared value type,
# plans as UPDATE, and applies as an ``OvrtxAttributeValue`` batch. Texture
# connects/disconnects on converter-wired sockets are material-graph
# topology and take the generation route.


_AUTHORED_MATERIAL_ROOT = "/World/Materials/Edit_Paint"
_AUTHORED_MATERIAL_SHADER = _AUTHORED_MATERIAL_ROOT + "/PreviewSurface"


def _authored_material_prims(
    connected_attributes: tuple[str, ...] = (),
) -> tuple[_AuthoredScenePrim, ...]:
    return (
        _AuthoredScenePrim("/World/Cube", "Xform"),
        _AuthoredScenePrim("/World/Cube/Mesh", "Mesh"),
        _AuthoredScenePrim(_AUTHORED_MATERIAL_ROOT, "Material"),
        _AuthoredScenePrim(
            _AUTHORED_MATERIAL_SHADER,
            "Shader",
            # The authored UsdPreviewSurface shader carries only the
            # preview-surface value inputs; the OpenPBR entries in
            # SUPPORTED_USD_ATTRIBUTES belong to the MaterialX shader family.
            attributes=tuple(
                "inputs:" + spec[0]
                for spec in usd_value_edit_support.PRINCIPLED_PREVIEW_SURFACE_VALUE_SPECS.values()
            ),
            info_id="UsdPreviewSurface",
            connected_attributes=connected_attributes,
        ),
    )


def _authored_material_resolver(
    monkeypatch,
    connected_attributes: tuple[str, ...] = (),
) -> usd_prim_resolver.UsdPrimResolver:
    monkeypatch.setattr(
        usd_prim_resolver,
        "_open_stage_prims",
        lambda path: (object(), _authored_material_prims(connected_attributes), None),
    )
    monkeypatch.setattr(
        builders.write_target_resolution,
        "resolve_write_target",
        lambda *args, **kwargs: ownership.WriteTargetResolutionResult(
            ownership.WriteTargetResolutionStatus.ERROR,
            error_reason=ownership.REASON_STAGE_OPEN_FAILED,
        ),
    )
    resolver = builders.UsdPrimResolver()
    resolver.scan(type("Request", (), {"input_usd_path": "/tmp/generation.usdc"})())
    return resolver


def _authored_material(node_tree: object | None = None) -> _FakeBlenderId:
    # "Edit Paint" sanitizes to Edit_Paint: leaf-name matching would miss,
    # so resolution must go through the authoring identity.
    material = _FakeBlenderId(
        name="Edit Paint",
        blender_type="MATERIAL",
        diffuse_color=(0.1, 0.2, 0.3, 1.0),
        node_tree=node_tree if node_tree is not None else _FakeNodeTree(),
    )
    material.ov = SimpleNamespace(usd=SimpleNamespace(prim_path=_AUTHORED_MATERIAL_ROOT))
    return material


def _linked_base_color_tree() -> _FakeNodeTree:
    tree = _FakeNodeTree()
    tree.nodes[0].inputs["Base Color"] = _FakeSocket((0.1, 0.2, 0.3, 1.0), linked=True)
    return tree


class _AttributeRecordingPort:
    def __init__(self) -> None:
        self.batches: list[list[OvrtxAttributeValue]] = []

    def update_attribute_values(self, values) -> OvrtxValueUpdateResult:
        batch = list(values)
        self.batches.append(batch)
        return OvrtxValueUpdateResult(len(batch), 1 if batch else None)


def test_depsgraph_material_value_edits_route_per_field_over_the_authored_material(monkeypatch) -> None:
    resolver = _authored_material_resolver(monkeypatch)
    material = _authored_material()

    edits = build_interactive_edits_from_depsgraph(
        _FakeDepsgraph([_FakeDepsgraphUpdate(material)]),
        usd_prim_resolver=resolver,
    )

    # field -> (edit value, applied client value, value type)
    expected = {
        "inputs:diffuseColor": ((0.1, 0.2, 0.3), [0.1, 0.2, 0.3], "Color3f"),
        "inputs:roughness": (0.45, 0.45, "Float"),
        "inputs:metallic": (0.2, 0.2, "Float"),
        "inputs:ior": (1.45, 1.45, "Float"),
        "inputs:emissiveColor": ((0.2, 0.4, 1.0), [0.2, 0.4, 1.0], "Color3f"),
    }
    assert [edit.usd_attribute for edit in edits] == list(expected)
    # A multi-user material still targets its one authored material prim.
    assert {edit.usd_prim_path for edit in edits} == {_AUTHORED_MATERIAL_SHADER}
    for edit in edits:
        edit_value, applied_value, value_type = expected[edit.usd_attribute]
        assert (edit.shape, edit.data_authority) == (EditShape.VALUE, DataAuthority.VIEW)
        assert edit.provenance["material_path"] == _AUTHORED_MATERIAL_ROOT
        assert edit.provenance["match_source"] == "authoring_prim_path"
        assert edit.value == edit_value

        plan = InteractiveEditPlanner().plan(edit)
        assert plan.mechanism == EditMechanism.UPDATE
        assert plan.impact.render_session_reuse_expected is True
        stream = ViewUpdateStream()
        queued = stream.queue(plan.to_intent())
        assert queued["queued"] is True
        port = _AttributeRecordingPort()
        result = stream.apply_pending(port)
        assert result["values_written"] is True
        assert port.batches == [
            [
                OvrtxAttributeValue(
                    _AUTHORED_MATERIAL_SHADER,
                    edit.usd_attribute,
                    applied_value,
                    value_type,
                )
            ]
        ]


def test_depsgraph_texture_connect_is_material_graph_topology(monkeypatch) -> None:
    # Blender socket linked, authored shader input unconnected: the graph
    # was rewired since the scanned generation — TOPOLOGY, generation route,
    # with an edit-record reason (never a silent value skip).
    resolver = _authored_material_resolver(monkeypatch)
    material = _authored_material(node_tree=_linked_base_color_tree())

    edits = build_interactive_edits_from_depsgraph(
        _FakeDepsgraph([_FakeDepsgraphUpdate(material)]),
        usd_prim_resolver=resolver,
    )

    assert len(edits) == 1
    edit = edits[0]
    assert edit.shape == EditShape.TOPOLOGY
    assert edit.data_authority == DataAuthority.VIEW
    assert edit.usd_prim_path == _AUTHORED_MATERIAL_ROOT
    assert edit.blender_property_path == "node_tree"
    assert edit.metadata["topology_change_kinds"] == ("material_graph",)
    assert edit.metadata["diverged_texture_inputs"] == ("inputs:diffuseColor",)

    plan = InteractiveEditPlanner().plan(edit)
    assert plan.mechanism == EditMechanism.COMPOSE
    assert plan.impact.authoring_reconciliation_requested is True
    assert plan.impact.render_session_reuse_expected is False
    assert material_conversion.MATERIAL_GRAPH_CHANGED in plan.impact.topology_reasons


def test_depsgraph_texture_disconnect_is_material_graph_topology(monkeypatch) -> None:
    # Authored shader input connected, Blender socket unlinked: also a
    # rewire; no value edit may target the still-connected authored input.
    resolver = _authored_material_resolver(
        monkeypatch,
        connected_attributes=("inputs:diffuseColor",),
    )
    material = _authored_material()

    edits = build_interactive_edits_from_depsgraph(
        _FakeDepsgraph([_FakeDepsgraphUpdate(material)]),
        usd_prim_resolver=resolver,
    )

    assert [edit.shape for edit in edits] == [EditShape.TOPOLOGY]
    assert edits[0].metadata["diverged_texture_inputs"] == ("inputs:diffuseColor",)
    assert edits[0].metadata["texture_link_divergence"]["inputs:diffuseColor"] == {
        "blender_linked": False,
        "authored_connected": True,
    }


def test_depsgraph_texture_connected_input_is_not_a_value_edit(monkeypatch) -> None:
    # Blender socket linked and authored input connected (states agree): no
    # topology edit; the connected input is unsupported-for-value-edit
    # (task04-07 reporting) while the other fields stay live.
    resolver = _authored_material_resolver(
        monkeypatch,
        connected_attributes=("inputs:diffuseColor",),
    )
    material = _authored_material(node_tree=_linked_base_color_tree())

    edits = build_interactive_edits_from_depsgraph(
        _FakeDepsgraph([_FakeDepsgraphUpdate(material)]),
        usd_prim_resolver=resolver,
    )

    assert all(edit.shape == EditShape.VALUE for edit in edits)
    value_edits = [edit for edit in edits if not edit.metadata.get("unsupported_reason")]
    assert [edit.usd_attribute for edit in value_edits] == [
        "inputs:roughness",
        "inputs:metallic",
        "inputs:ior",
        "inputs:emissiveColor",
    ]
    classification = material_conversion.classify_field(material, "principled:Base Color")
    assert classification.status == material_conversion.STATUS_UNSUPPORTED
    assert classification.reason == material_conversion.TEXTURE_CONNECTED_INPUT
    # task04-07: the silent skip is now a classification-originated
    # report-only record (no value; plans UNSUPPORTED, so no RPC).
    report_edits = [edit for edit in edits if edit.metadata.get("unsupported_reason")]
    assert len(report_edits) == 1
    report_edit = report_edits[0]
    assert report_edit.usd_attribute == "inputs:diffuseColor"
    assert report_edit.blender_property_path == "principled:Base Color"
    assert report_edit.value is None
    assert report_edit.metadata["unsupported_reason"] == material_conversion.TEXTURE_CONNECTED_INPUT
    assert report_edit.metadata["classification"] == "unsupported"


def test_depsgraph_material_with_unscanned_authoring_identity_builds_no_edits(monkeypatch) -> None:
    # The scanned composition (04-01 shape: no material prims) does not
    # contain the claimed authored material: resolution fails closed and no
    # value or topology edit is built.
    resolver = _authored_scan_resolver(monkeypatch)
    material = _authored_material()

    edits = build_interactive_edits_from_depsgraph(
        _FakeDepsgraph([_FakeDepsgraphUpdate(material)]),
        usd_prim_resolver=resolver,
    )

    assert edits == []
    resolution = resolver.resolve_material(
        material,
        usd_attribute="inputs:diffuseColor",
        property_name="diffuse_color",
    )
    assert resolution.error_reason == material_usd_prim.ERROR_AUTHORING_PATH_NOT_IN_SCENE


def test_depsgraph_direct_usd_material_keeps_value_edit_behavior(monkeypatch) -> None:
    # No authoring identity (direct-USD stage): the graph-divergence check
    # does not apply; a linked socket emits no value edit for that
    # attribute (fixture behavior preserved) — only the task04-07
    # report-only classification record.
    index = material_usd_prim._material_prim_index_from_prims(
        (
            _FakeBlenderId(path="/World/Looks/Paint", type_name="Material"),
            _FakeBlenderId(
                path="/World/Looks/Paint/Shader",
                type_name="Shader",
                attributes=("inputs:diffuseColor",),
                info_id="UsdPreviewSurface",
            ),
        )
    )
    material = _FakeBlenderId(
        name="Paint",
        blender_type="MATERIAL",
        diffuse_color=(0.1, 0.2, 0.3, 1.0),
        node_tree=_linked_base_color_tree(),
    )

    edits = build_interactive_edits_from_depsgraph(
        _FakeDepsgraph([_FakeDepsgraphUpdate(material)]),
        usd_prim_resolver=_FakeResolver(
            material_indexes={material_usd_prim.DEFAULT_USD_MATERIAL_ATTRIBUTE: index}
        ),
    )

    assert all(edit.shape == EditShape.VALUE for edit in edits)
    value_edits = [edit for edit in edits if not edit.metadata.get("unsupported_reason")]
    assert "inputs:diffuseColor" not in {edit.usd_attribute for edit in value_edits}
    report_edits = [edit for edit in edits if edit.metadata.get("unsupported_reason")]
    assert [edit.blender_property_path for edit in report_edits] == ["principled:Base Color"]


def test_depsgraph_transform_edit_for_physics_locked_object_keeps_rejection(monkeypatch) -> None:
    resolver = _authored_scan_resolver(monkeypatch)
    identity = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    obj = _authored_blender_object("Cube", "MESH", "/World/Cube", matrix_world=identity)
    lock = operator_state.PhysicsPlaybackLock()
    lock.lock_object("/World/Cube", obj, generation=2)

    obj.matrix_world = _BLENDER_ROTZ90_T123
    edits = build_interactive_edits_from_depsgraph(
        _FakeDepsgraph([_FakeDepsgraphUpdate(obj)]),
        usd_prim_resolver=resolver,
    )
    assert len(edits) == 1

    rejected = lock.reject_edit(edits[0])

    assert rejected is not None
    assert rejected.reason == "physics_playback_locked"
    # The lock restores the pre-edit pose; the attempted value is discarded
    # (behavior asserted, not modified — task04-01 clarification).
    assert obj.matrix_world == identity
    assert lock.diagnostics()["rejected_edit_count"] == 1


# --- Light value edits as live overs (task04-03) ------------------------------
#
# Depsgraph light data edits over the authored scene composition: the light
# resolves to its converter-authored UsdLux prim via the ``ov.usd.prim_path``
# authoring identity (light names can sanitize differently from their USD
# leaf names, and depsgraph updates carry evaluated ID copies), each
# supported value field routes through ``light_value_conversion`` — the same
# functions the topology converter authors from — plans as UPDATE, and
# applies as an ``OvrtxAttributeValue`` batch. Authored-light-form changes
# are topology and take the generation route.


_AUTHORED_SPOT_ROOT = "/World/Lights/Key_Light"
_AUTHORED_AREA_ROOT = "/World/Lights/Soft_Box"


def _authored_light_prims() -> tuple[_AuthoredScenePrim, ...]:
    return (
        _AuthoredScenePrim("/World/Cube", "Xform"),
        _AuthoredScenePrim(
            _AUTHORED_SPOT_ROOT,
            "SphereLight",
            attributes=("inputs:shaping:cone:angle", "inputs:shaping:cone:softness"),
        ),
        _AuthoredScenePrim(_AUTHORED_AREA_ROOT, "RectLight"),
    )


def _authored_light_resolver(monkeypatch) -> usd_prim_resolver.UsdPrimResolver:
    monkeypatch.setattr(
        usd_prim_resolver,
        "_open_stage_prims",
        lambda path: (object(), _authored_light_prims(), None),
    )
    monkeypatch.setattr(
        builders.write_target_resolution,
        "resolve_write_target",
        lambda *args, **kwargs: ownership.WriteTargetResolutionResult(
            ownership.WriteTargetResolutionStatus.ERROR,
            error_reason=ownership.REASON_STAGE_OPEN_FAILED,
        ),
    )
    resolver = builders.UsdPrimResolver()
    resolver.scan(type("Request", (), {"input_usd_path": "/tmp/generation.usdc"})())
    return resolver


def _authored_spot_light() -> tuple[_FakeBlenderId, _FakeBlenderId]:
    # "Key Light" sanitizes to Key_Light: leaf-name matching would miss, so
    # resolution must go through the authoring identity (04-01 follow-up).
    data = _FakeBlenderId(
        name="Key Light Data",
        blender_type="SPOT",
        energy=100.0,
        color=(1.0, 0.5, 0.25),
        use_temperature=False,
        shadow_soft_size=0.25,
        spot_size=math.radians(60.0),
        spot_blend=0.3,
    )
    light = _authored_blender_object(
        "Key Light",
        "LIGHT",
        _AUTHORED_SPOT_ROOT,
        data=data,
        scale=(1.0, 1.0, 1.0),
    )
    return light, data


def _evaluated_copy_of(id_data: _FakeBlenderId) -> _FakeBlenderId:
    """Depsgraph-update shape: same field values, different Python identity."""

    copy = _FakeBlenderId(name=id_data.name, blender_type=id_data.type)
    for key, value in vars(id_data).items():
        setattr(copy, key, value)
    copy.original = id_data
    return copy


def test_depsgraph_light_data_edit_maps_evaluated_copy_by_data_name(monkeypatch) -> None:
    # An evaluated depsgraph copy without a usable ``original`` reference
    # still maps back to its owning light object through the light-data name
    # (evaluation keeps ID names).
    resolver = _authored_light_resolver(monkeypatch)
    light, data = _authored_spot_light()
    evaluated = _evaluated_copy_of(data)
    del evaluated.original

    edits = build_interactive_edits_from_depsgraph(
        _FakeDepsgraph([_FakeDepsgraphUpdate(evaluated)]),
        usd_prim_resolver=resolver,
        light_objects=(light,),
    )

    assert edits, "expected the evaluated light data update to map to its object"
    assert all(edit.usd_prim_path == _AUTHORED_SPOT_ROOT for edit in edits)


def test_depsgraph_area_shape_change_is_light_form_topology(monkeypatch) -> None:
    # The scanned generation authored a RectLight (AREA_RECT); Blender now
    # says DISK: the authored light form changed — TOPOLOGY with the
    # light-form edit-record reason, generation route (task04-03
    # clarification: Rect<->Disk crosses USD families).
    resolver = _authored_light_resolver(monkeypatch)
    data = _FakeBlenderId(
        name="Soft Box Data",
        blender_type="AREA",
        shape="DISK",
        energy=90.0,
        color=(1.0, 1.0, 1.0),
        use_temperature=False,
        size=2.0,
        size_y=3.0,
    )
    light = _authored_blender_object(
        "Soft Box",
        "LIGHT",
        _AUTHORED_AREA_ROOT,
        data=data,
        scale=(1.0, 1.0, 1.0),
    )

    edits = build_interactive_edits_from_depsgraph(
        _FakeDepsgraph([_FakeDepsgraphUpdate(_evaluated_copy_of(data))]),
        usd_prim_resolver=resolver,
        light_objects=(light,),
    )

    assert len(edits) == 1
    edit = edits[0]
    assert (edit.shape, edit.data_authority) == (EditShape.TOPOLOGY, DataAuthority.VIEW)
    assert edit.usd_prim_path == _AUTHORED_AREA_ROOT
    assert edit.blender_property_path == "data.shape"
    assert edit.metadata["topology_change_kinds"] == ("light_form",)
    assert edit.metadata["previous_authored_light_form"] == "AREA_RECT"
    assert edit.metadata["current_authored_light_form"] == "AREA_DISK"
    assert edit.provenance["match_source"] == "authoring_prim_path"

    plan = InteractiveEditPlanner().plan(edit)
    assert plan.mechanism == EditMechanism.COMPOSE
    assert plan.impact.authoring_reconciliation_requested is True
    assert plan.impact.render_session_reuse_expected is False
    assert light_conversion.LIGHT_FORM_CHANGED in plan.impact.topology_reasons


def test_depsgraph_light_type_change_is_light_form_topology(monkeypatch) -> None:
    # The scanned generation authored a plain SphereLight (POINT: no spot
    # shaping attributes); Blender's light.type now says SPOT — the authored
    # light form changed within one USD family: still topology.
    monkeypatch.setattr(
        usd_prim_resolver,
        "_open_stage_prims",
        lambda path: (
            object(),
            (_AuthoredScenePrim(_AUTHORED_SPOT_ROOT, "SphereLight", attributes=()),),
            None,
        ),
    )
    monkeypatch.setattr(
        builders.write_target_resolution,
        "resolve_write_target",
        lambda *args, **kwargs: ownership.WriteTargetResolutionResult(
            ownership.WriteTargetResolutionStatus.ERROR,
            error_reason=ownership.REASON_STAGE_OPEN_FAILED,
        ),
    )
    resolver = builders.UsdPrimResolver()
    resolver.scan(type("Request", (), {"input_usd_path": "/tmp/generation.usdc"})())
    light, data = _authored_spot_light()

    edits = build_interactive_edits_from_depsgraph(
        _FakeDepsgraph([_FakeDepsgraphUpdate(_evaluated_copy_of(data))]),
        usd_prim_resolver=resolver,
        light_objects=(light,),
    )

    assert [edit.shape for edit in edits] == [EditShape.TOPOLOGY]
    assert edits[0].blender_property_path == "data.type"
    assert edits[0].metadata["previous_authored_light_form"] == "POINT"
    assert edits[0].metadata["current_authored_light_form"] == "SPOT"
    plan = InteractiveEditPlanner().plan(edits[0])
    assert plan.mechanism == EditMechanism.COMPOSE
    assert light_conversion.LIGHT_FORM_CHANGED in plan.impact.topology_reasons


def test_depsgraph_light_with_unscanned_authoring_identity_builds_no_edits(monkeypatch) -> None:
    # The scanned composition does not contain the claimed authored light
    # root: resolution fails closed and no value or topology edit is built.
    resolver = _authored_scan_resolver(monkeypatch)
    light, data = _authored_spot_light()

    edits = build_interactive_edits_from_depsgraph(
        _FakeDepsgraph([_FakeDepsgraphUpdate(_evaluated_copy_of(data))]),
        usd_prim_resolver=resolver,
        light_objects=(light,),
    )

    assert edits == []
    resolution = resolver.resolve_light(light)
    assert resolution.error_reason == light_usd_prim.ERROR_AUTHORING_PATH_NOT_IN_SCENE

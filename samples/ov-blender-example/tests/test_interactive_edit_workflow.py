# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example.interactive_edit_planner import (  # noqa: E402
    EditMechanism,
    EditPersistence,
    EditStatus,
    DataAuthority,
    EditShape,
    edit_location,
    InteractiveEdit,
)
from ovrtx_blender_example.interactive_edit_workflow import (  # noqa: E402
    InteractiveEditWorkflow,
    WorkflowAction,
)
from ovrtx_blender_example.edit_persistence import WriteRequest, WriteResult  # noqa: E402
from ovrtx_blender_example.runtime_scheduler import RuntimeScheduler  # noqa: E402
from ovrtx_blender_example.topology_edit_fallback import SCENE_TOPOLOGY_CHANGED  # noqa: E402
from ovrtx_blender_example.usd_opinion_write import UsdOpinionWriter  # noqa: E402


class _FakeUsdAttribute:
    def __init__(self, name: str, type_name: str, custom: bool) -> None:
        self.name = name
        self.type_name = type_name
        self.custom = custom
        self.value: object = None

    def Set(self, value: object) -> None:
        self.value = value


class _FakeUsdPrim:
    def __init__(self, path: str, type_name: str = "") -> None:
        self.path = path
        self.type_name = type_name
        self.typeName = type_name
        self.attributes: dict[str, _FakeUsdAttribute] = {}

    def CreateAttribute(self, name: str, type_name: str, custom: bool = False) -> _FakeUsdAttribute:
        attribute = _FakeUsdAttribute(name, type_name, custom)
        self.attributes[name] = attribute
        return attribute


class _FakeUsdStage:
    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.prims: dict[str, _FakeUsdPrim] = {}
        self.saved = False

    def OverridePrim(self, path: str) -> _FakeUsdPrim:
        prim = self.prims.setdefault(path, _FakeUsdPrim(path))
        return prim

    def DefinePrim(self, path: str, type_name: str) -> _FakeUsdPrim:
        prim = self.prims.setdefault(path, _FakeUsdPrim(path))
        prim.type_name = type_name
        prim.typeName = type_name
        return prim

    def GetRootLayer(self) -> "_FakeUsdStage":
        return self

    def Save(self) -> None:
        self.saved = True


class _FakeUsdStageFactory:
    def __init__(self, owner: "_FakeUsd") -> None:
        self.owner = owner

    def CreateNew(self, filepath: str) -> _FakeUsdStage:
        stage = _FakeUsdStage(filepath)
        self.owner.created_stages.append(stage)
        return stage


class _FakeUsd:
    def __init__(self) -> None:
        self.created_stages: list[_FakeUsdStage] = []
        self.Stage = _FakeUsdStageFactory(self)


class _FakeValueTypeNames:
    Bool = "Bool"
    Color3f = "Color3f"
    Color4f = "Color4f"
    Double = "Double"
    Float = "Float"
    Float2 = "Float2"
    Float3 = "Float3"
    Float4 = "Float4"
    Int = "Int"
    Matrix4d = "Matrix4d"
    String = "String"


class _FakeSdf:
    ValueTypeNames = _FakeValueTypeNames


class _FakeMatrix4d:
    def __init__(self, _value: float) -> None:
        self.rows: list[tuple[float, float, float, float] | None] = [None, None, None, None]

    def SetRow(self, index: int, value: tuple[float, float, float, float]) -> None:
        self.rows[index] = value


class _FakeGf:
    @staticmethod
    def Vec2f(*values: float) -> tuple[str, tuple[float, ...]]:
        return ("Vec2f", tuple(values))

    @staticmethod
    def Vec3f(*values: float) -> tuple[str, tuple[float, ...]]:
        return ("Vec3f", tuple(values))

    @staticmethod
    def Vec4f(*values: float) -> tuple[str, tuple[float, ...]]:
        return ("Vec4f", tuple(values))

    @staticmethod
    def Vec4d(*values: float) -> tuple[float, float, float, float]:
        return tuple(values)  # type: ignore[return-value]

    Matrix4d = _FakeMatrix4d


def _target(
    *,
    usd_layer_id: str = "/layers/scene.usda",
    usd_prim_path: str = "/World/TestScene/Cube",
    usd_attribute: str = "xformOp:transform",
    blender_property_path: str = "location",
) -> dict[str, object]:
    return edit_location(
        usd_prim_path=usd_prim_path,
        usd_attribute=usd_attribute,
        usd_layer_id=usd_layer_id,
        blender_property_path=blender_property_path,
        provenance={"source": "test"},
    )


def _identity_matrix() -> tuple[tuple[float, float, float, float], ...]:
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _session_topology_edit(
    *,
    usd_prim_path: str = "/World/SessionLights/Key",
    intensity: float = 9000.0,
) -> InteractiveEdit:
    return InteractiveEdit(
        shape=EditShape.TOPOLOGY,

        data_authority=DataAuthority.VIEW,
        **_target(
            usd_layer_id="",
            usd_prim_path=usd_prim_path,
            usd_attribute="",
            blender_property_path="object.data",
        ),
        value={"usd_prim_path": usd_prim_path, "intensity": intensity},
    )


def test_workflow_routes_update_to_scheduler_without_export() -> None:
    scheduler = RuntimeScheduler()
    workflow = InteractiveEditWorkflow(runtime_scheduler=scheduler)
    edit = InteractiveEdit(
        shape=EditShape.VALUE,

        data_authority=DataAuthority.VIEW,
        **_target(),
        value=((1.0, 0.0, 0.0, 0.0),) * 4,
    )

    result = workflow.preview_edit(edit)

    assert result.action == WorkflowAction.UPDATE
    assert result.status == EditStatus.QUEUED
    assert result.accepted is True
    assert result.submission_result is not None
    assert result.write_results == {}
    assert scheduler.diagnostics()["last_edit_update"]["queued"] is True
    assert result.diagnostics["target"]["usd_layer_id"] == "/layers/scene.usda"
    assert result.diagnostics["target"]["usd_property_path"] == "/World/TestScene/Cube.xformOp:transform"
    diagnostics = workflow.diagnostics()
    record = diagnostics["edit_records"][0]
    assert record["action"] == "update"
    assert record["data_authority"] == "view"
    assert record["usd_layer_id"] == "/layers/scene.usda"
    assert record["accepted"] is True
    assert record["result"] == "queued"
    assert record["values_written"] is False

    matched = workflow.record_update_result(
        {
            "values_written": True,
            "rendered_effect_observed": True,
            "value_paths": ["/World/TestScene/Cube"],
        }
    )

    assert matched == 1
    updated = workflow.diagnostics()["edit_records"][0]
    assert updated["result"] == "applied"
    assert updated["values_written"] is True
    assert updated["rendered_effect_observed"] is True


def test_workflow_reports_update_when_scheduler_is_unavailable() -> None:
    workflow = InteractiveEditWorkflow()
    edit = InteractiveEdit(
        shape=EditShape.VALUE,

        data_authority=DataAuthority.VIEW,
        **_target(),
        value=((1.0, 0.0, 0.0, 0.0),) * 4,
    )

    result = workflow.preview_edit(edit)

    assert result.action == WorkflowAction.UNSUPPORTED
    assert result.status == EditStatus.UNSUPPORTED
    assert result.accepted is False
    assert result.reason == "update_scheduler_unavailable"
    assert result.write_results == {}
    record = workflow.diagnostics()["edit_records"][0]
    assert record["action"] == "unsupported"
    assert record["fail_reason"] == "update_scheduler_unavailable"
    assert record["accepted"] is False

def test_workflow_topology_fallback_request_does_not_require_write_path() -> None:
    def layer_writer(request: WriteRequest) -> WriteResult:
        return WriteResult(
            requested=True,
            completed=False,
            reason="write_target_write_requested_without_path",
            path="",
            usd_layer_id=request.usd_layer_id,
            diagnostics={"edit_count": len(request.edits)},
        )

    workflow = InteractiveEditWorkflow(writer=layer_writer)
    result = workflow.preview_edit(
        InteractiveEdit(
            shape=EditShape.TOPOLOGY,

            data_authority=DataAuthority.VIEW,
            **_target(
                usd_layer_id="/layers/scene.usda",
                usd_prim_path="/World/TestScene/Light",
                usd_attribute="",
                blender_property_path="light.type",
            ),
            value="new light",
        )
    )

    assert result.action == WorkflowAction.WRITE
    assert result.status == EditStatus.FAILED
    assert result.accepted is False
    assert result.diagnostics["topology_fallback"]["write_requested"] is True
    assert result.diagnostics["topology_fallback"]["requested_write_path"] == ""
    assert result.diagnostics["topology_fallback"]["session_rekey_status"] == "requested"


def test_workflow_routes_session_topology_to_scene_generation_replacement() -> None:
    workflow = InteractiveEditWorkflow()
    edit = _session_topology_edit()

    result = workflow.preview_edit(edit)
    selected = workflow.select_for_write(edit)

    assert result.action == WorkflowAction.COMPOSE
    assert result.status == EditStatus.QUEUED
    assert result.reason == "scene_generation_dirty"
    assert result.diagnostics["scene_generation_replacement_requested"] is True
    assert result.diagnostics["persistence"] == "none"
    assert selected.action == WorkflowAction.UNSUPPORTED
    assert selected.reason == "edit_not_writeable"


def test_workflow_update_material_preview_needs_scheduler_and_does_not_write() -> None:
    write_calls: list[WriteRequest] = []
    workflow = InteractiveEditWorkflow(
        writer=lambda request: write_calls.append(request) or WriteResult(
            requested=True,
            completed=True,
            reason="written",
        )
    )
    edit = InteractiveEdit(
        shape=EditShape.VALUE,

        data_authority=DataAuthority.VIEW,
        **_target(
            usd_layer_id="/layers/look.usda",
            usd_prim_path="/World/Asset/Looks/Paint",
            usd_attribute="inputs:diffuseColor",
            blender_property_path="diffuse_color",
        ),
        value=(1.0, 0.0, 0.0, 1.0),
    )

    result = workflow.preview_edit(edit)

    assert result.action == WorkflowAction.UNSUPPORTED
    assert result.status == EditStatus.UNSUPPORTED
    assert result.accepted is False
    assert result.reason == "update_scheduler_unavailable"
    assert write_calls == []

def test_workflow_selected_write_reports_missing_writer() -> None:
    workflow = InteractiveEditWorkflow()
    edit = InteractiveEdit(
        shape=EditShape.VALUE,

        data_authority=DataAuthority.VIEW,
        **_target(
            usd_layer_id="/layers/look.usda",
            usd_prim_path="/World/Asset/Looks/Paint",
            usd_attribute="inputs:diffuseColor",
            blender_property_path="diffuse_color",
        ),
        value=(1.0, 0.0, 0.0, 1.0),
    )

    workflow.select_for_write(edit)
    result = workflow.write_selected_edits()

    assert result.action == WorkflowAction.UNSUPPORTED
    assert result.accepted is False
    assert result.reason == "writer_unavailable"
    assert result.write_results == {}
    assert workflow.diagnostics()["pending_selected_write_count"] == 1


def test_workflow_selected_write_rejects_non_writeable_edit() -> None:
    workflow = InteractiveEditWorkflow()
    edit = InteractiveEdit(
        shape=EditShape.VALUE,
        data_authority=DataAuthority.VIEW,
        **_target(
            usd_layer_id="",
            usd_prim_path="/World/Camera",
            usd_attribute="omni:xform",
            blender_property_path="viewport_camera_matrix",
        ),
        value=_identity_matrix(),
    )

    result = workflow.select_for_write(edit)

    assert result.action == WorkflowAction.UNSUPPORTED
    assert result.reason == "edit_not_writeable"
    assert workflow.diagnostics()["pending_selected_write_count"] == 0


def test_workflow_selected_write_rejects_unresolved_persistence_identity() -> None:
    calls: list[WriteRequest] = []

    def writer(request: WriteRequest) -> WriteResult:
        calls.append(request)
        return WriteResult(requested=True, completed=True, reason="written")

    workflow = InteractiveEditWorkflow(writer=writer)
    edit = InteractiveEdit(
        shape=EditShape.TOPOLOGY,
        data_authority=DataAuthority.VIEW,
        **_target(
            usd_layer_id="/layers/look.usda",
            usd_prim_path="",
            usd_attribute="material:binding",
            blender_property_path="node_tree",
        ),
        value="new shader",
    )

    result = workflow.select_for_write(edit)

    assert result.action == WorkflowAction.UNSUPPORTED
    assert result.reason == "edit_not_writeable"
    assert workflow.diagnostics()["pending_selected_write_count"] == 0
    assert calls == []


def test_workflow_selected_write_reports_empty_selection() -> None:
    result = InteractiveEditWorkflow().write_selected_edits()

    assert result.action == WorkflowAction.UNSUPPORTED
    assert result.reason == "no_selected_write_edits"
    assert result.write_results == {}


def test_workflow_selected_write_keeps_edits_after_writer_failure() -> None:
    def writer(request: WriteRequest) -> WriteResult:
        return WriteResult(
            requested=True,
            completed=False,
            reason="selected_write_failed",
            usd_layer_id=request.usd_layer_id,
        )

    workflow = InteractiveEditWorkflow(writer=writer)
    edit = InteractiveEdit(
        shape=EditShape.VALUE,
        data_authority=DataAuthority.VIEW,
        **_target(
            usd_layer_id="/layers/look.usda",
            usd_prim_path="/World/Asset/Looks/Paint",
            usd_attribute="inputs:diffuseColor",
            blender_property_path="diffuse_color",
        ),
        value=(1.0, 0.0, 0.0, 1.0),
    )
    workflow.select_for_write(edit)

    result = workflow.write_selected_edits()

    assert result.action == WorkflowAction.WRITE
    assert result.status == EditStatus.FAILED
    assert workflow.diagnostics()["pending_selected_write_count"] == 1

def test_workflow_fixed_target_writer_writes_only_its_selected_target(tmp_path: Path) -> None:
    fake_usd = _FakeUsd()
    writer = UsdOpinionWriter(
        filepath=str(tmp_path / "look-opinions.usda"),
        usd_layer_id="/layers/look.usda",
        usd_module=fake_usd,
        sdf_module=_FakeSdf,
        gf_module=_FakeGf,
    )
    workflow = InteractiveEditWorkflow(writer=writer)
    look_edit = InteractiveEdit(
        shape=EditShape.VALUE,
        data_authority=DataAuthority.VIEW,
        **_target(
            usd_layer_id="/layers/look.usda",
            usd_prim_path="/World/Looks/Paint",
            usd_attribute="inputs:diffuseColor",
            blender_property_path="diffuse_color",
        ),
        value=(1.0, 0.0, 0.0),
    )
    light_edit = InteractiveEdit(
        shape=EditShape.VALUE,
        data_authority=DataAuthority.VIEW,
        **_target(
            usd_layer_id="/layers/lights.usda",
            usd_prim_path="/World/Key",
            usd_attribute="inputs:intensity",
            blender_property_path="data.energy",
        ),
        value=1000.0,
    )
    workflow.select_for_write(look_edit)
    workflow.select_for_write(light_edit)

    result = workflow.write_selected_edits()

    assert result.reason == "selected_write_partial_failure"
    assert len(fake_usd.created_stages) == 1
    assert "/World/Looks/Paint" in fake_usd.created_stages[0].prims
    assert "/World/Key" not in fake_usd.created_stages[0].prims
    assert workflow.diagnostics()["pending_selected_write_count"] == 1


def test_workflow_selected_write_reports_all_targets_unsupported() -> None:
    def writer(request: WriteRequest) -> WriteResult:
        return WriteResult(
            requested=False,
            completed=False,
            reason="unsupported_target",
            usd_layer_id=request.usd_layer_id,
        )

    workflow = InteractiveEditWorkflow(writer=writer)
    for usd_layer_id, prim_path in (
        ("/layers/look.usda", "/World/Looks/Paint"),
        ("/layers/lights.usda", "/World/Key"),
    ):
        workflow.select_for_write(
            InteractiveEdit(
                shape=EditShape.VALUE,
                data_authority=DataAuthority.VIEW,
                **_target(
                    usd_layer_id=usd_layer_id,
                    usd_prim_path=prim_path,
                    usd_attribute="inputs:intensity",
                    blender_property_path="data.energy",
                ),
                value=1.0,
            )
        )

    result = workflow.write_selected_edits()

    assert result.action == WorkflowAction.UNSUPPORTED
    assert result.status == EditStatus.UNSUPPORTED
    assert result.reason == "selected_write_unsupported"
    assert workflow.diagnostics()["pending_selected_write_count"] == 2


def test_workflow_records_observation_only_selection_diagnostics() -> None:
    workflow = InteractiveEditWorkflow()
    selection_resolution = {
        "status": "unsupported_selection_group",
        "group_rejected": True,
        "unresolved_reasons": ["preview_only_selection_source"],
        "sources": [{"status": "preview_only", "owner_category": "inspection_only"}],
    }

    record = workflow.record_selection_resolution(selection_resolution)

    diagnostics = workflow.diagnostics()
    assert record["action"] == "observation"
    assert record["accepted"] is False
    assert diagnostics["event_count"] == 1
    assert diagnostics["edit_record_count"] == 1
    assert diagnostics["events"][0]["edit_record_ids"] == ["edit-000001"]
    assert diagnostics["edit_records"][0]["fail_reason"] == "preview_only_selection_source"


def test_workflow_diagnostic_history_is_bounded_without_reusing_record_ids() -> None:
    workflow = InteractiveEditWorkflow()
    selection_resolution = {
        "status": "unsupported_selection_group",
        "group_rejected": True,
        "unresolved_reasons": ["preview_only_selection_source"],
        "sources": [{"status": "preview_only", "owner_category": "inspection_only"}],
    }

    for _index in range(300):
        workflow.record_selection_resolution(selection_resolution)

    diagnostics = workflow.diagnostics()
    assert diagnostics["event_count"] == 300
    assert diagnostics["edit_record_count"] == 300
    assert diagnostics["retained_event_count"] == len(diagnostics["events"])
    assert diagnostics["retained_edit_record_count"] == len(diagnostics["edit_records"])
    assert diagnostics["retained_event_count"] < diagnostics["event_count"]
    assert diagnostics["retained_edit_record_count"] < diagnostics["edit_record_count"]
    assert diagnostics["events"][-1]["edit_record_ids"] == ["edit-000300"]


def test_usd_opinion_writer_writes_write_target_value_opinions(tmp_path: Path) -> None:
    fake_usd = _FakeUsd()
    filepath = tmp_path / "look-opinions.usda"
    port = UsdOpinionWriter(
        filepath=str(filepath),
        usd_layer_id="/layers/lights.usda",
        usd_module=fake_usd,
        sdf_module=_FakeSdf,
        gf_module=_FakeGf,
    )
    intensity = InteractiveEdit(
        shape=EditShape.VALUE,

        data_authority=DataAuthority.VIEW,
        **_target(
            usd_layer_id="/layers/lights.usda",
            usd_prim_path="/World/TestScene/Key",
            usd_attribute="inputs:intensity",
            blender_property_path="energy",
        ),
        value=900.0,
    )
    light_color = InteractiveEdit(
        shape=EditShape.VALUE,

        data_authority=DataAuthority.VIEW,
        **_target(
            usd_layer_id="/layers/lights.usda",
            usd_prim_path="/World/TestScene/Key",
            usd_attribute="inputs:color",
            blender_property_path="color",
        ),
        value=(0.2, 0.3, 0.4),
    )
    normalize = InteractiveEdit(
        shape=EditShape.VALUE,

        data_authority=DataAuthority.VIEW,
        **_target(
            usd_layer_id="/layers/lights.usda",
            usd_prim_path="/World/TestScene/Key",
            usd_attribute="inputs:normalize",
            blender_property_path="normalize",
        ),
        value=True,
    )
    color = InteractiveEdit(
        shape=EditShape.VALUE,

        data_authority=DataAuthority.VIEW,
        **_target(
            usd_layer_id="/layers/lights.usda",
            usd_prim_path="/World/TestScene/Looks/Paint",
            usd_attribute="inputs:diffuseColor",
            blender_property_path="diffuse_color",
        ),
        value=(0.1, 0.2, 0.3, 1.0),
    )

    result = port(
        WriteRequest(
            edits=(intensity, light_color, normalize, color),
            reason="test",
        )
    )

    assert result.requested is True
    assert result.completed is True
    assert result.reason == "usd_opinion_write_completed"
    assert result.diagnostics["whole_scene_export_requested"] is False
    assert result.diagnostics["mutated_source_stage"] is False
    assert result.diagnostics["usd_layer_path"] == str(filepath)
    assert len(fake_usd.created_stages) == 1
    stage = fake_usd.created_stages[0]
    assert stage.saved is True
    assert stage.prims["/World/TestScene/Key"].attributes["inputs:intensity"].type_name == "Float"
    assert stage.prims["/World/TestScene/Key"].attributes["inputs:intensity"].value == 900.0
    light_color_attr = stage.prims["/World/TestScene/Key"].attributes["inputs:color"]
    assert light_color_attr.type_name == "Color3f"
    assert light_color_attr.value == ("Vec3f", (0.2, 0.3, 0.4))
    assert stage.prims["/World/TestScene/Key"].attributes["inputs:normalize"].type_name == "Bool"
    assert stage.prims["/World/TestScene/Key"].attributes["inputs:normalize"].value is True
    color_attr = stage.prims["/World/TestScene/Looks/Paint"].attributes["inputs:diffuseColor"]
    assert color_attr.type_name == "Color3f"
    assert color_attr.value == ("Vec3f", (0.1, 0.2, 0.3))
    assert result.diagnostics["opinions"][0]["usd_layer_id"] == "/layers/lights.usda"
    assert result.diagnostics["opinions"][1]["usd_value_type"] == "Color3f"
    assert result.diagnostics["opinions"][2]["usd_value_type"] == "Bool"
    assert result.diagnostics["opinions"][3]["usd_value_type"] == "Color3f"


def test_usd_opinion_writer_creates_output_parent_directory(tmp_path: Path) -> None:
    fake_usd = _FakeUsd()
    filepath = tmp_path / "nested" / "scene-opinions.usda"
    port = UsdOpinionWriter(
        filepath=str(filepath),
        usd_layer_id="/layers/lights.usda",
        usd_module=fake_usd,
        sdf_module=_FakeSdf,
        gf_module=_FakeGf,
    )
    edit = InteractiveEdit(
        shape=EditShape.VALUE,

        data_authority=DataAuthority.VIEW,
        **_target(
            usd_layer_id="/layers/lights.usda",
            usd_prim_path="/World/TestScene/Key",
            usd_attribute="inputs:intensity",
            blender_property_path="energy",
        ),
        value=900.0,
    )

    result = port(
        WriteRequest(
            edits=(edit,),
            reason="test",
            usd_layer_id="/layers/lights.usda",
        )
    )

    assert result.completed is True
    assert filepath.parent.exists()
    assert fake_usd.created_stages[0].filepath == str(filepath)


def test_usd_opinion_writer_writes_light_form_topology_opinion(tmp_path: Path) -> None:
    fake_usd = _FakeUsd()
    filepath = tmp_path / "light-form-opinions.usda"
    port = UsdOpinionWriter(
        filepath=str(filepath),
        usd_layer_id="/layers/lights.usda",
        usd_module=fake_usd,
        sdf_module=_FakeSdf,
        gf_module=_FakeGf,
    )
    edit = InteractiveEdit(
        shape=EditShape.TOPOLOGY,
        data_authority=DataAuthority.VIEW,
        **_target(
            usd_layer_id="/layers/lights.usda",
            usd_prim_path="/World/TestScene/Key/KeyLightData",
            usd_attribute="",
            blender_property_path="data.shape",
        ),
        value="AREA_DISK",
        previous_value="AREA_RECT",
        metadata={
            "topology_change_kinds": ("light_form",),
            "previous_authored_light_form": "AREA_RECT",
            "current_authored_light_form": "AREA_DISK",
            "previous_usd_family": "RectLight",
            "current_usd_family": "DiskLight",
            "topology_attribute_values": (
                {
                    "name": "inputs:radius",
                    "value": 1.25,
                    "blender_property_path": "size",
                },
            ),
        },
    )

    result = port(
        WriteRequest(
            edits=(edit,),
            reason="topology_edit_requires_compose_write",
            usd_layer_id="/layers/lights.usda",
        )
    )

    assert result.completed is True
    assert result.reason == "usd_opinion_write_completed"
    stage = fake_usd.created_stages[0]
    assert stage.saved is True
    prim = stage.prims["/World/TestScene/Key/KeyLightData"]
    assert prim.typeName == "DiskLight"
    assert prim.attributes["inputs:radius"].value == 1.25
    opinion = result.diagnostics["opinions"][0]
    assert opinion["topology_type"] == "DiskLight"
    assert opinion["usd_attribute"] == ""
    assert [item["usd_attribute"] for item in opinion["topology_attributes"]] == [
        "inputs:radius",
    ]


def test_usd_opinion_writer_writes_light_form_topology_attribute_values(tmp_path: Path) -> None:
    fake_usd = _FakeUsd()
    filepath = tmp_path / "spot-form-opinions.usda"
    port = UsdOpinionWriter(
        filepath=str(filepath),
        usd_layer_id="/layers/lights.usda",
        usd_module=fake_usd,
        sdf_module=_FakeSdf,
        gf_module=_FakeGf,
    )
    edit = InteractiveEdit(
        shape=EditShape.TOPOLOGY,
        data_authority=DataAuthority.VIEW,
        **_target(
            usd_layer_id="/layers/lights.usda",
            usd_prim_path="/World/TestScene/Key/KeyLightData",
            usd_attribute="",
            blender_property_path="data.type",
        ),
        value="SPOT",
        previous_value="POINT",
        metadata={
            "topology_change_kinds": ("light_form",),
            "previous_authored_light_form": "POINT",
            "current_authored_light_form": "SPOT",
            "previous_usd_family": "SphereLight",
            "current_usd_family": "SphereLight",
            "topology_attribute_values": (
                {
                    "name": "inputs:shaping:cone:angle",
                    "value": 30.0,
                    "blender_property_path": "spot_size",
                },
                {
                    "name": "inputs:shaping:cone:softness",
                    "value": 0.25,
                    "blender_property_path": "spot_blend",
                },
            ),
        },
    )

    result = port(
        WriteRequest(
            edits=(edit,),
            reason="topology_edit_requires_compose_write",
            usd_layer_id="/layers/lights.usda",
        )
    )

    assert result.completed is True
    prim = fake_usd.created_stages[0].prims["/World/TestScene/Key/KeyLightData"]
    assert prim.typeName == "SphereLight"
    assert prim.attributes["inputs:shaping:cone:angle"].value == 30.0
    assert prim.attributes["inputs:shaping:cone:softness"].value == 0.25
    opinion = result.diagnostics["opinions"][0]
    assert [item["usd_attribute"] for item in opinion["topology_attributes"]] == [
        "inputs:shaping:cone:angle",
        "inputs:shaping:cone:softness",
    ]


def test_usd_opinion_writer_rejects_same_family_light_form_without_attribute_values(tmp_path: Path) -> None:
    fake_usd = _FakeUsd()
    port = UsdOpinionWriter(
        filepath=str(tmp_path / "spot-form-opinions.usda"),
        usd_layer_id="/layers/lights.usda",
        usd_module=fake_usd,
        sdf_module=_FakeSdf,
        gf_module=_FakeGf,
    )
    edit = InteractiveEdit(
        shape=EditShape.TOPOLOGY,
        data_authority=DataAuthority.VIEW,
        **_target(
            usd_layer_id="/layers/lights.usda",
            usd_prim_path="/World/TestScene/Key/KeyLightData",
            usd_attribute="",
            blender_property_path="data.type",
        ),
        value="SPOT",
        previous_value="POINT",
        metadata={
            "topology_change_kinds": ("light_form",),
            "previous_usd_family": "SphereLight",
            "current_usd_family": "SphereLight",
        },
    )

    result = port(
        WriteRequest(
            edits=(edit,),
            reason="topology_edit_requires_compose_write",
            usd_layer_id="/layers/lights.usda",
        )
    )

    assert result.completed is False
    assert fake_usd.created_stages == []
    assert result.diagnostics["unsupported_edits"][0]["reasons"] == [
        "topology_edit_requires_attribute_values"
    ]


def test_usd_opinion_writer_rejects_compound_topology_kinds(tmp_path: Path) -> None:
    fake_usd = _FakeUsd()
    port = UsdOpinionWriter(
        filepath=str(tmp_path / "compound-topology-opinions.usda"),
        usd_layer_id="/layers/lights.usda",
        usd_module=fake_usd,
        sdf_module=_FakeSdf,
        gf_module=_FakeGf,
    )
    edit = InteractiveEdit(
        shape=EditShape.TOPOLOGY,
        data_authority=DataAuthority.VIEW,
        **_target(
            usd_layer_id="/layers/lights.usda",
            usd_prim_path="/World/TestScene/Key/KeyLightData",
            usd_attribute="",
            blender_property_path="data.shape",
        ),
        value="AREA_DISK",
        previous_value="AREA_RECT",
        metadata={
            "topology_change_kinds": ("light_form", "material_binding"),
            "previous_usd_family": "RectLight",
            "current_usd_family": "DiskLight",
        },
    )

    result = port(
        WriteRequest(
            edits=(edit,),
            reason="topology_edit_requires_compose_write",
            usd_layer_id="/layers/lights.usda",
        )
    )

    assert result.completed is False
    assert fake_usd.created_stages == []
    assert "topology_edit_requires_writer" in result.diagnostics["unsupported_edits"][0]["reasons"]


def test_usd_opinion_writer_rejects_topology_and_unknown_write_target_without_writing() -> None:
    fake_usd = _FakeUsd()
    port = UsdOpinionWriter(
        filepath="/tmp/layer-opinions.usda",
        usd_layer_id="/layers/scene.usda",
        usd_module=fake_usd,
        sdf_module=_FakeSdf,
        gf_module=_FakeGf,
    )
    unknown_layer_edit = InteractiveEdit(
        shape=EditShape.VALUE,

        data_authority=DataAuthority.VIEW,
        **_target(
            usd_layer_id="",
            usd_prim_path="/World/Asset/Looks/Paint",
            usd_attribute="inputs:diffuseColor",
            blender_property_path="diffuse_color",
        ),
        value=(1.0, 0.0, 0.0),
    )
    topology_edit = InteractiveEdit(
        shape=EditShape.TOPOLOGY,

        data_authority=DataAuthority.VIEW,
        **_target(
            usd_layer_id="/layers/scene.usda",
            usd_prim_path="/World/TestScene/NewCollider",
            usd_attribute="",
            blender_property_path="validation.collider_shape",
        ),
        value="new collider",
    )

    result = port(
        WriteRequest(
            edits=(unknown_layer_edit, topology_edit),
            reason="test",
            usd_layer_id="/layers/scene.usda",
        )
    )

    assert result.requested is False
    assert result.completed is False
    assert result.reason == "usd_opinion_write_unsupported_edits"
    assert result.diagnostics["whole_scene_export_requested"] is False
    assert result.diagnostics["mutated_source_stage"] is False
    assert fake_usd.created_stages == []
    unsupported = result.diagnostics["unsupported_edits"]
    assert "missing_write_target_identity" in unsupported[0]["reasons"]
    assert "topology_edit_requires_writer" in unsupported[1]["reasons"]


def test_usd_opinion_writer_rejects_mixed_write_targets_without_writing() -> None:
    fake_usd = _FakeUsd()
    port = UsdOpinionWriter(
        filepath="/tmp/mixed-opinions.usda",
        usd_layer_id="/layers/look.usda",
        usd_module=fake_usd,
        sdf_module=_FakeSdf,
        gf_module=_FakeGf,
    )
    material_edit = InteractiveEdit(
        shape=EditShape.VALUE,
        data_authority=DataAuthority.VIEW,
        **_target(
            usd_layer_id="/layers/look.usda",
            usd_prim_path="/World/Looks/Paint",
            usd_attribute="inputs:diffuseColor",
            blender_property_path="diffuse_color",
        ),
        value=(1.0, 0.0, 0.0),
    )
    light_edit = InteractiveEdit(
        shape=EditShape.VALUE,
        data_authority=DataAuthority.VIEW,
        **_target(
            usd_layer_id="/layers/lights.usda",
            usd_prim_path="/World/Key",
            usd_attribute="inputs:intensity",
            blender_property_path="data.energy",
        ),
        value=1000.0,
    )

    result = port(
        WriteRequest(
            edits=(material_edit, light_edit),
            reason="selected_write",
        )
    )

    assert result.requested is False
    assert result.completed is False
    assert result.reason == "usd_opinion_write_unsupported_edits"
    assert result.diagnostics["rejected_edit_count"] == 2
    assert result.diagnostics["unsupported_edits"][0] == {
        "reason": "mixed_write_targets",
        "usd_layer_ids": ["/layers/lights.usda", "/layers/look.usda"],
    }
    assert fake_usd.created_stages == []


def test_usd_opinion_writer_rejects_metadata_type_override_for_unlisted_attributes() -> None:
    fake_usd = _FakeUsd()
    port = UsdOpinionWriter(
        filepath="/tmp/layer-opinions.usda",
        usd_layer_id="/layers/physics.usda",
        usd_module=fake_usd,
        sdf_module=_FakeSdf,
        gf_module=_FakeGf,
    )
    edit = InteractiveEdit(
        shape=EditShape.VALUE,

        data_authority=DataAuthority.SIM,
        **_target(
            usd_layer_id="/layers/physics.usda",
            usd_prim_path="/World/TestScene/Cube",
            usd_attribute="physics:mass",
            blender_property_path="mass",
        ),
        value=2.5,
    )
    explicit = InteractiveEdit(
        shape=edit.shape,

        data_authority=edit.data_authority,
        **edit_location(
            blender_property_path=edit.blender_property_path,
            usd_prim_path=edit.usd_prim_path,
            usd_property_path=edit.usd_property_path,
            usd_layer_id=edit.usd_layer_id,
            provenance=edit.provenance,
        ),
        value=edit.value,
        metadata={"usd_value_type": "Double"},
    )

    rejected = port(
        WriteRequest(
            edits=(edit,),
            reason="test",
            usd_layer_id="/layers/physics.usda",
        )
    )
    rejected_with_metadata = port(
        WriteRequest(
            edits=(explicit,),
            reason="test",
            usd_layer_id="/layers/physics.usda",
        )
    )

    assert rejected.requested is False
    assert rejected.diagnostics["unsupported_edits"][0]["reasons"] == ["unsupported_usd_value_type"]
    assert rejected_with_metadata.requested is False
    assert rejected_with_metadata.diagnostics["unsupported_edits"][0]["reasons"] == ["unsupported_usd_value_type"]
    assert fake_usd.created_stages == []

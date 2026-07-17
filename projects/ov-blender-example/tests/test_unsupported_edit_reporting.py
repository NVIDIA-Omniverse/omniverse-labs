# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unsupported edit classification and reporting (blender-live-render task04-07).

Covers the four-value diagnostic classification vocabulary normalized
across the material/light/world/camera policies, the once-per-key
user-visible report with per-event edit records, the free path for
unsupported/non-rendering classifications (no RPC, no refinement reset,
no session churn), and the classification-originated reports for silent
lanes (texture-connected material sockets, node-tree-only world updates).
"""

from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import blender_interactive_edit_builders as builders  # noqa: E402
from ovrtx_blender_example import camera_value_conversion  # noqa: E402
from ovrtx_blender_example import light_value_conversion  # noqa: E402
from ovrtx_blender_example import material_usd_prim  # noqa: E402
from ovrtx_blender_example import material_value_conversion  # noqa: E402
from ovrtx_blender_example import world_dome_conversion  # noqa: E402
from ovrtx_blender_example import world_dome_usd_prim  # noqa: E402
from ovrtx_blender_example.blender_interactive_edit_builders import (  # noqa: E402
    build_interactive_edits_from_depsgraph,
)
from ovrtx_blender_example.interactive_edit_planner import (  # noqa: E402
    DataAuthority,
    EditMechanism,
    EditShape,
    EditStatus,
    InteractiveEdit,
    InteractiveEditPlanner,
    edit_location,
)
from ovrtx_blender_example.interactive_edit_workflow import (  # noqa: E402
    InteractiveEditWorkflow,
    WorkflowAction,
)
from ovrtx_blender_example.runtime_scheduler import EditSubmissionResult  # noqa: E402
from ovrtx_blender_example.value_edit_conversion import (  # noqa: E402
    BLENDER_DATABLOCK_NON_RENDER_FIELD_REASONS,
    CLASSIFICATION_NON_RENDERING,
    CLASSIFICATION_SUPPORTED,
    CLASSIFICATION_TOPOLOGY,
    CLASSIFICATION_UNSUPPORTED,
    DIAGNOSTIC_CLASSIFICATIONS,
    STATUS_NON_RENDER,
    STATUS_SUPPORTED,
    STATUS_TOPOLOGY,
    STATUS_UNSUPPORTED,
    classification_for_unsupported_reason,
    classification_report_message,
    display_field_name,
    normalized_classification,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeBlenderId(dict):
    def __init__(self, *, name: str = "Item", blender_type: str = "", **attrs: object) -> None:
        super().__init__()
        self.name = name
        self.name_full = name
        if blender_type:
            self.type = blender_type
        for key, value in attrs.items():
            setattr(self, key, value)


class _FakeDepsgraph:
    def __init__(self, updates: list[object]) -> None:
        self.updates = [SimpleNamespace(id=item) for item in updates]


class _FakeSocket:
    def __init__(self, value: object, *, linked: bool = False) -> None:
        self.default_value = value
        self.is_linked = linked


class _RecordingScheduler:
    """Scheduler fake: every accepted submission would be an RPC-bound wake."""

    def __init__(self) -> None:
        self.submitted: list[object] = []

    def submit_edit(self, intent: object) -> EditSubmissionResult:
        self.submitted.append(intent)
        return EditSubmissionResult(
            status=EditStatus.QUEUED,
            reason="queued",
            diagnostics={"queued": True},
        )


def _unsupported_edit(
    *,
    prim_path: str = "/World/Lights/Key",
    field: str = "spread",
    reason: str = "unsupported_area_spread",
    classification: str = "",
) -> InteractiveEdit:
    metadata: dict[str, object] = {"unsupported_reason": reason}
    if classification:
        metadata["classification"] = classification
    return InteractiveEdit(
        shape=EditShape.VALUE,
        data_authority=DataAuthority.VIEW,
        **edit_location(
            usd_prim_path=prim_path,
            blender_property_path=field,
            provenance={"source": "blender"},
        ),
        value=None,
        metadata=metadata,
    )


def _family_topology_edit(
    *,
    prim_path: str = "/World/Looks/Paint",
    field: str = "node_tree",
) -> InteractiveEdit:
    return InteractiveEdit(
        shape=EditShape.TOPOLOGY,
        data_authority=DataAuthority.VIEW,
        **edit_location(
            usd_prim_path=prim_path,
            blender_property_path=field,
            provenance={"source": "blender"},
        ),
        value=("inputs:diffuseColor",),
        previous_value=(),
        metadata={
            "topology_change_kinds": ("material_graph",),
            "diverged_texture_inputs": ("inputs:diffuseColor",),
        },
    )


# ---------------------------------------------------------------------------
# Vocabulary normalization
# ---------------------------------------------------------------------------


def test_normalized_classification_maps_every_policy_status() -> None:
    assert normalized_classification(STATUS_SUPPORTED) == CLASSIFICATION_SUPPORTED
    assert normalized_classification(STATUS_TOPOLOGY) == CLASSIFICATION_TOPOLOGY
    assert normalized_classification(STATUS_UNSUPPORTED) == CLASSIFICATION_UNSUPPORTED
    assert normalized_classification(STATUS_NON_RENDER) == CLASSIFICATION_NON_RENDERING
    # Idempotent on already-normalized values.
    for classification in DIAGNOSTIC_CLASSIFICATIONS:
        assert normalized_classification(classification) == classification
    # Unknown statuses fail closed to unsupported; empty stays empty.
    assert normalized_classification("mystery_status") == CLASSIFICATION_UNSUPPORTED
    assert normalized_classification("") == ""
    assert normalized_classification(None) == ""
    assert set(DIAGNOSTIC_CLASSIFICATIONS) == {
        "supported",
        "unsupported",
        "non_rendering",
        "topology",
    }


def test_all_policy_classifications_normalize_to_the_four_value_vocabulary() -> None:
    """Consistency requirement: material/light/world/camera policies share
    exactly the four-value diagnostic vocabulary."""

    material = _FakeBlenderId(
        name="Paint",
        blender_type="MATERIAL",
        node_tree=SimpleNamespace(
            nodes=[
                SimpleNamespace(
                    type="BSDF_PRINCIPLED",
                    name="Principled BSDF",
                    inputs={
                        "Base Color": _FakeSocket((0.1, 0.2, 0.3, 1.0), linked=True),
                        "Roughness": _FakeSocket(0.5),
                    },
                )
            ]
        ),
    )
    spot_light = SimpleNamespace(type="SPOT", shape="")
    solid_world = _FakeBlenderId(
        name="World", blender_type="WORLD", use_nodes=False, color=(1.0, 1.0, 1.0)
    )
    camera = SimpleNamespace(type="PERSP")

    probes = [
        (material_value_conversion.classify_field, material, "principled:Roughness"),
        (material_value_conversion.classify_field, material, "principled:Base Color"),
        (material_value_conversion.classify_field, material, "node_tree"),
        (material_value_conversion.classify_field, material, "use_fake_user"),
        (material_value_conversion.classify_field, material, "unknown_material_thing"),
        (light_value_conversion.classify_field, spot_light, "energy"),
        (light_value_conversion.classify_field, spot_light, "type"),
        (light_value_conversion.classify_field, spot_light, "spread"),
        (light_value_conversion.classify_field, spot_light, "show_cone"),
        (light_value_conversion.classify_field, spot_light, "unknown_light_thing"),
        (world_dome_conversion.classify_field, solid_world, "color"),
        (world_dome_conversion.classify_field, solid_world, "node_tree"),
        (world_dome_conversion.classify_field, solid_world, "lightgroup"),
        (world_dome_conversion.classify_field, solid_world, "name"),
        (camera_value_conversion.classify_field, camera, "lens"),
        (camera_value_conversion.classify_field, camera, "type"),
        (camera_value_conversion.classify_field, camera, "lens_unit"),
        (camera_value_conversion.classify_field, camera, "show_limits"),
        (camera_value_conversion.classify_field, camera, "unknown_camera_thing"),
    ]
    internal_statuses = {
        STATUS_SUPPORTED,
        STATUS_TOPOLOGY,
        STATUS_UNSUPPORTED,
        STATUS_NON_RENDER,
    }
    for classify, target, field in probes:
        classification = classify(target, field)
        assert classification.status in internal_statuses, (field, classification)
        assert (
            normalized_classification(classification.status) in DIAGNOSTIC_CLASSIFICATIONS
        ), (field, classification)


def test_non_rendering_reasons_share_the_non_runtime_prefix_across_policies() -> None:
    reasons = set(BLENDER_DATABLOCK_NON_RENDER_FIELD_REASONS.values())
    reasons.update(light_value_conversion._NON_RENDER_FIELD_REASONS.values())
    reasons.update(material_value_conversion._NON_RENDER_FIELD_REASONS.values())
    reasons.update(world_dome_conversion._NON_RENDER_FIELD_REASONS.values())
    reasons.update(camera_value_conversion._NON_RENDER_FIELD_REASONS.values())
    for reason in reasons:
        assert classification_for_unsupported_reason(reason) == CLASSIFICATION_NON_RENDERING, reason
    assert classification_for_unsupported_reason("unsupported_light_field") == (
        CLASSIFICATION_UNSUPPORTED
    )


def test_report_message_phrasing_per_classification() -> None:
    unsupported = classification_report_message(
        CLASSIFICATION_UNSUPPORTED, field="Base Color"
    )
    assert "Base Color" in unsupported
    assert "not supported by OVRTX value updates" in unsupported
    topology = classification_report_message(CLASSIFICATION_TOPOLOGY, field="node_tree")
    assert "node_tree" in topology
    assert "applies on next scene update" in topology
    non_rendering = classification_report_message(
        CLASSIFICATION_NON_RENDERING, field="use_fake_user"
    )
    assert "use_fake_user" in non_rendering
    assert "does not affect rendering" in non_rendering
    assert classification_report_message(CLASSIFICATION_SUPPORTED, field="energy") == ""
    assert display_field_name("principled:Base Color") == "Base Color"
    assert display_field_name("data.type") == "data.type"


# ---------------------------------------------------------------------------
# Workflow: once-per-key reports, record-every-time, free unsupported path
# ---------------------------------------------------------------------------


def test_unsupported_edit_reports_once_per_key_and_records_every_time() -> None:
    scheduler = _RecordingScheduler()
    workflow = InteractiveEditWorkflow(runtime_scheduler=scheduler)

    first = workflow.preview_edit(_unsupported_edit())
    second = workflow.preview_edit(_unsupported_edit())
    third = workflow.preview_edit(_unsupported_edit())

    assert first.action == WorkflowAction.UNSUPPORTED
    assert first.classification == CLASSIFICATION_UNSUPPORTED
    assert "spread" in first.user_report
    assert "not supported by OVRTX value updates" in first.user_report
    # Drag spam / repeated depsgraph callbacks share the key: the record is
    # written every time, the user-visible report only once.
    assert second.user_report == ""
    assert third.user_report == ""
    assert second.classification == CLASSIFICATION_UNSUPPORTED
    diagnostics = workflow.diagnostics()
    assert diagnostics["edit_record_count"] == 3
    assert diagnostics["user_report_count"] == 1
    assert diagnostics["user_reported_keys"] == [("/World/Lights/Key", "spread")]
    for record in diagnostics["edit_records"]:
        assert record["classification"] == CLASSIFICATION_UNSUPPORTED
        assert record["accepted"] is False
    # Free path: no scheduler submission (no RPC, no wake), no reset.
    assert scheduler.submitted == []


def test_distinct_fields_and_targets_report_independently() -> None:
    workflow = InteractiveEditWorkflow(runtime_scheduler=_RecordingScheduler())

    first = workflow.preview_edit(_unsupported_edit(field="spread"))
    other_field = workflow.preview_edit(
        _unsupported_edit(field="use_shadow", reason="unsupported_light_shadow_toggle")
    )
    other_target = workflow.preview_edit(
        _unsupported_edit(prim_path="/World/Lights/Fill", field="spread")
    )

    assert first.user_report != ""
    assert other_field.user_report != ""
    assert "use_shadow" in other_field.user_report
    assert other_target.user_report != ""
    assert workflow.diagnostics()["user_report_count"] == 3


def test_non_rendering_reason_classifies_and_reports_non_rendering() -> None:
    workflow = InteractiveEditWorkflow(runtime_scheduler=_RecordingScheduler())

    result = workflow.preview_edit(
        _unsupported_edit(field="use_fake_user", reason="non_runtime_blender_datablock_field")
    )

    assert result.classification == CLASSIFICATION_NON_RENDERING
    assert "does not affect rendering" in result.user_report
    record = workflow.diagnostics()["edit_records"][0]
    assert record["classification"] == CLASSIFICATION_NON_RENDERING


def test_unsupported_plan_is_free_no_reset_no_rekey_no_scheduler() -> None:
    plan = InteractiveEditPlanner().plan(_unsupported_edit())

    assert plan.mechanism == EditMechanism.NONE
    assert plan.unsupported_reason == "unsupported_area_spread"
    assert plan.impact.refinement_reset_expected is False
    assert plan.impact.session_rekey_expected is False
    assert plan.impact.update_requested is False
    assert plan.impact.authoring_reconciliation_requested is False


def test_family_topology_edit_reports_next_scene_update_once() -> None:
    workflow = InteractiveEditWorkflow(runtime_scheduler=_RecordingScheduler())
    edit = _family_topology_edit()

    first = workflow.preview_edit(edit)
    second = workflow.preview_edit(edit)

    assert first.action == WorkflowAction.COMPOSE
    assert first.accepted is True
    assert first.classification == CLASSIFICATION_TOPOLOGY
    assert "applies on next scene update" in first.user_report
    assert "node_tree" in first.user_report
    assert second.user_report == ""
    records = workflow.diagnostics()["edit_records"]
    assert len(records) == 2
    assert all(record["classification"] == CLASSIFICATION_TOPOLOGY for record in records)


def test_generic_scene_topology_edit_classifies_without_user_report() -> None:
    # Ordinary scene topology (object adds/removes) reports no user line —
    # only the four families' topology edits carry topology_change_kinds.
    workflow = InteractiveEditWorkflow(runtime_scheduler=_RecordingScheduler())
    edit = InteractiveEdit(
        shape=EditShape.TOPOLOGY,
        data_authority=DataAuthority.VIEW,
        **edit_location(
            usd_prim_path="/World/SessionLights/Key",
            blender_property_path="object.data",
            provenance={"source": "blender"},
        ),
        value={"usd_prim_path": "/World/SessionLights/Key"},
    )

    result = workflow.preview_edit(edit)

    assert result.classification == CLASSIFICATION_TOPOLOGY
    assert result.user_report == ""
    assert workflow.diagnostics()["edit_records"][0]["classification"] == (
        CLASSIFICATION_TOPOLOGY
    )


def test_supported_update_classifies_supported_and_reports_nothing() -> None:
    scheduler = _RecordingScheduler()
    workflow = InteractiveEditWorkflow(runtime_scheduler=scheduler)
    edit = InteractiveEdit(
        shape=EditShape.VALUE,
        data_authority=DataAuthority.VIEW,
        **edit_location(
            usd_prim_path="/World/Cube",
            usd_attribute="xformOp:transform",
            blender_property_path="matrix_world",
            provenance={"source": "blender"},
        ),
        value=((1.0, 0.0, 0.0, 0.0),) * 4,
    )

    result = workflow.preview_edit(edit)

    assert result.accepted is True
    assert result.classification == CLASSIFICATION_SUPPORTED
    assert result.user_report == ""
    assert len(scheduler.submitted) == 1
    assert workflow.diagnostics()["user_report_count"] == 0
    assert workflow.diagnostics()["edit_records"][0]["classification"] == (
        CLASSIFICATION_SUPPORTED
    )


def test_new_workflow_instance_resets_the_report_dedupe() -> None:
    # The engine clears the workflow at viewport session end
    # (_end_viewport_session sets _interactive_edit_workflow = None), so a
    # fresh instance is the session-reset semantics.
    first_session = InteractiveEditWorkflow(runtime_scheduler=_RecordingScheduler())
    assert first_session.preview_edit(_unsupported_edit()).user_report != ""
    assert first_session.preview_edit(_unsupported_edit()).user_report == ""

    second_session = InteractiveEditWorkflow(runtime_scheduler=_RecordingScheduler())
    assert second_session.preview_edit(_unsupported_edit()).user_report != ""


def test_infrastructure_unsupported_results_do_not_user_report() -> None:
    # No scheduler: a supported value edit degrades with
    # update_scheduler_unavailable — an infrastructure outcome, not a field
    # classification; no user-visible classification report.
    workflow = InteractiveEditWorkflow()
    edit = InteractiveEdit(
        shape=EditShape.VALUE,
        data_authority=DataAuthority.VIEW,
        **edit_location(
            usd_prim_path="/World/Cube",
            usd_attribute="xformOp:transform",
            blender_property_path="matrix_world",
            provenance={"source": "blender"},
        ),
        value=((1.0, 0.0, 0.0, 0.0),) * 4,
    )

    result = workflow.preview_edit(edit)

    assert result.action == WorkflowAction.UNSUPPORTED
    assert result.reason == "update_scheduler_unavailable"
    assert result.classification == CLASSIFICATION_SUPPORTED
    assert result.user_report == ""


# ---------------------------------------------------------------------------
# Material lane: texture-connected socket reports originate from classification
# ---------------------------------------------------------------------------


def _connected_material_and_resolver() -> tuple[_FakeBlenderId, object]:
    index = material_usd_prim._material_prim_index_from_prims(
        (
            _FakeBlenderId(name="Paint", path="/World/Looks/Paint", type_name="Material"),
            _FakeBlenderId(
                name="Shader",
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
        node_tree=SimpleNamespace(
            nodes=[
                SimpleNamespace(
                    type="BSDF_PRINCIPLED",
                    name="Principled BSDF",
                    inputs={
                        "Base Color": _FakeSocket((0.1, 0.2, 0.3, 1.0), linked=True),
                    },
                )
            ]
        ),
    )

    class _Resolver:
        def resolve_material(self, mat, *, usd_attribute, property_name):
            return material_usd_prim.resolve_material_usd_prim(
                mat,
                index if usd_attribute == "inputs:diffuseColor" else {
                    "available": False,
                    "reason": "not_loaded",
                },
                usd_attribute=usd_attribute,
                property_name=property_name,
            )

        def resolve_light(self, light):
            raise AssertionError("not a light lane test")

        def resolve_world_dome(self):
            return world_dome_usd_prim.resolve_world_dome_usd_prim(
                {"available": False, "stage_reason": "not_loaded"}
            )

        def resolve_uv(self, mesh):
            raise AssertionError("not a uv lane test")

    return material, _Resolver()


def test_connected_socket_edit_reports_once_with_zero_rpc_submissions() -> None:
    material, resolver = _connected_material_and_resolver()
    scheduler = _RecordingScheduler()
    workflow = InteractiveEditWorkflow(runtime_scheduler=scheduler)

    def _drag_event() -> list[str]:
        reports = []
        edits = build_interactive_edits_from_depsgraph(
            _FakeDepsgraph([material]),
            usd_prim_resolver=resolver,
        )
        report_edits = [
            edit for edit in edits if edit.metadata.get("unsupported_reason")
        ]
        assert len(report_edits) == 1
        for edit in report_edits:
            result = workflow.preview_edit(edit)
            assert result.classification == CLASSIFICATION_UNSUPPORTED
            if result.user_report:
                reports.append(result.user_report)
        return reports

    first_reports = _drag_event()
    second_reports = _drag_event()
    third_reports = _drag_event()

    assert len(first_reports) == 1
    assert "Base Color" in first_reports[0]
    assert "not supported by OVRTX value updates" in first_reports[0]
    assert second_reports == []
    assert third_reports == []
    # The classification records exist every time; nothing reached the
    # scheduler (no RPC, no refinement reset, no thread wake).
    assert scheduler.submitted == []
    diagnostics = workflow.diagnostics()
    assert diagnostics["edit_record_count"] == 3
    assert diagnostics["user_report_count"] == 1
    record = diagnostics["edit_records"][0]
    assert record["reason"] == material_value_conversion.TEXTURE_CONNECTED_INPUT
    assert record["classification"] == CLASSIFICATION_UNSUPPORTED


def test_unresolvable_connected_material_stays_fail_closed_with_no_records() -> None:
    # Fail-closed resolution (unscanned identity / missing index) builds no
    # edits at all — including report-only classification records: a scene
    # mismatch must not masquerade as an unsupported-field report.
    material, _ = _connected_material_and_resolver()

    class _EmptyResolver:
        def resolve_material(self, mat, *, usd_attribute, property_name):
            return material_usd_prim.resolve_material_usd_prim(
                mat,
                {"available": False, "reason": "not_loaded"},
                usd_attribute=usd_attribute,
                property_name=property_name,
            )

        def resolve_world_dome(self):
            return world_dome_usd_prim.resolve_world_dome_usd_prim(
                {"available": False, "stage_reason": "not_loaded"}
            )

    edits = build_interactive_edits_from_depsgraph(
        _FakeDepsgraph([material]),
        usd_prim_resolver=_EmptyResolver(),
    )

    assert edits == []


# ---------------------------------------------------------------------------
# World lane: node-tree-only updates report from classification
# ---------------------------------------------------------------------------


def _studio_dome_index() -> dict:
    return world_dome_usd_prim._world_dome_prim_index_from_prims(
        (
            _FakeBlenderId(
                name="StudioDome",
                path="/World/StudioDome",
                type_name="DomeLight",
                attributes=("inputs:intensity", "inputs:color"),
            ),
        )
    )


class _WorldResolver:
    def __init__(self, index: dict) -> None:
        self._index = index

    def resolve_material(self, material, *, usd_attribute, property_name):
        return material_usd_prim.resolve_material_usd_prim(
            material,
            {"available": False, "reason": "not_loaded"},
            usd_attribute=usd_attribute,
            property_name=property_name,
        )

    def resolve_world_dome(self):
        return world_dome_usd_prim.resolve_world_dome_usd_prim(self._index)


def _env_texture_world() -> _FakeBlenderId:
    node_tree = _FakeBlenderId(
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
        links=[],
    )
    return _FakeBlenderId(
        name="World",
        blender_type="WORLD",
        use_nodes=True,
        node_tree=node_tree,
    )


def test_world_node_tree_only_update_reports_topology_from_classification() -> None:
    # Blender 5.1 reports only the embedded ShaderNodeTree ID for some
    # intermediate node-graph states (task04-04 gap): the report originates
    # from classification, not a submitted edit — planner-free, record +
    # once-per-key "applies on next scene update".
    world = _env_texture_world()
    resolver = _WorldResolver(_studio_dome_index())
    scheduler = _RecordingScheduler()
    workflow = InteractiveEditWorkflow(runtime_scheduler=scheduler)

    edits = build_interactive_edits_from_depsgraph(
        _FakeDepsgraph([world.node_tree]),
        usd_prim_resolver=resolver,
        worlds=(world,),
    )

    assert len(edits) == 1
    edit = edits[0]
    assert edit.shape == EditShape.VALUE
    assert edit.usd_prim_path == "/World/StudioDome"
    assert edit.blender_property_path == "node_tree"
    assert edit.metadata["classification"] == CLASSIFICATION_TOPOLOGY
    assert edit.metadata["unsupported_reason"] == (
        world_dome_conversion.ENVIRONMENT_TEXTURE_CHANGED
    )
    assert edit.metadata["topology_change_kinds"] == ("environment_texture",)

    plan = InteractiveEditPlanner().plan(edit)
    assert plan.mechanism == EditMechanism.NONE
    assert plan.impact.refinement_reset_expected is False
    assert plan.impact.authoring_reconciliation_requested is False

    first = workflow.preview_edit(edit)
    second = workflow.preview_edit(edit)

    assert first.classification == CLASSIFICATION_TOPOLOGY
    assert "applies on next scene update" in first.user_report
    assert second.user_report == ""
    assert scheduler.submitted == []


def test_world_event_with_world_id_keeps_the_real_topology_edit_only() -> None:
    # When the event also reports the World, the world lane's real TOPOLOGY
    # edit (generation route) wins and no extra classification record is
    # emitted for the node tree.
    world = _env_texture_world()
    resolver = _WorldResolver(_studio_dome_index())

    edits = build_interactive_edits_from_depsgraph(
        _FakeDepsgraph([world, world.node_tree]),
        usd_prim_resolver=resolver,
        worlds=(world,),
    )

    assert len(edits) == 1
    assert edits[0].shape == EditShape.TOPOLOGY
    assert edits[0].metadata["topology_change_kinds"] == ("environment_texture",)
    assert "unsupported_reason" not in edits[0].metadata


def test_world_node_tree_only_update_on_supported_world_builds_nothing() -> None:
    # An unlinked utility node next to the bare Background keeps the world
    # supported: nothing render-relevant changed, no record, no report.
    node_tree = _FakeBlenderId(
        name="WorldNodes",
        nodes=[
            _FakeBlenderId(
                name="Background",
                blender_type="BACKGROUND",
                inputs={
                    "Color": _FakeSocket((1.0, 0.5, 0.25, 1.0)),
                    "Strength": _FakeSocket(2.0),
                },
            ),
            _FakeBlenderId(name="Noise", blender_type="TEX_NOISE"),
        ],
        links=[],
    )
    world = _FakeBlenderId(
        name="World",
        blender_type="WORLD",
        use_nodes=True,
        node_tree=node_tree,
    )

    edits = build_interactive_edits_from_depsgraph(
        _FakeDepsgraph([node_tree]),
        usd_prim_resolver=_WorldResolver(_studio_dome_index()),
        worlds=(world,),
    )

    assert edits == []


def test_unmatched_node_tree_update_builds_nothing() -> None:
    stray_tree = _FakeBlenderId(name="MaterialNodes", nodes=[], links=[])

    edits = build_interactive_edits_from_depsgraph(
        _FakeDepsgraph([stray_tree]),
        usd_prim_resolver=_WorldResolver(_studio_dome_index()),
        worlds=(_env_texture_world(),),
    )

    assert edits == []


# ---------------------------------------------------------------------------
# Light lane: the existing classification-record edit reports the field
# ---------------------------------------------------------------------------


def test_light_form_gap_unsupported_edit_reports_field_once() -> None:
    workflow = InteractiveEditWorkflow(runtime_scheduler=_RecordingScheduler())
    edit = builders._unsupported_light_form_edit(
        location=edit_location(
            usd_prim_path="/World/Lights/Key",
            blender_property_path="data.type",
            provenance={"source": "blender"},
        ),
        current_authored_light_form="",
    )

    first = workflow.preview_edit(edit)
    second = workflow.preview_edit(edit)

    assert first.classification == CLASSIFICATION_UNSUPPORTED
    assert "data.type" in first.user_report
    assert "not supported by OVRTX value updates" in first.user_report
    assert second.user_report == ""

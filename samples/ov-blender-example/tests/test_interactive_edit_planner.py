# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example.interactive_edit_planner import (  # noqa: E402
    DataAuthority,
    EditIntent,
    EditMechanism,
    EditPersistence,
    EditShape,
    edit_location,
    InteractiveEdit,
    InteractiveEditPlanner,
)
from ovrtx_blender_example.topology_edit_fallback import (  # noqa: E402
    COLLIDER_TOPOLOGY_CHANGED,
    LIGHT_FORM_CHANGED,
    MATERIAL_BINDING_CHANGED,
    MATERIAL_GRAPH_CHANGED,
)


def _target(
    *,
    usd_layer_id: str = "/layers/scene.usda",
    usd_prim_path: str = "/World/TestScene/Cube",
    usd_attribute: str = "xformOp:transform",
    usd_property_path: str = "",
    blender_property_path: str = "matrix_world",
    provenance: dict[str, object] | None = None,
) -> dict[str, object]:
    return edit_location(
        usd_prim_path=usd_prim_path,
        usd_attribute=usd_attribute,
        usd_property_path=usd_property_path,
        usd_layer_id=usd_layer_id,
        blender_property_path=blender_property_path,
        provenance={"source": "test"} if provenance is None else provenance,
    )


def _value_edit(
    *,
    data_authority: DataAuthority = DataAuthority.VIEW,
    target: dict[str, object] | None = None,
    value: object = ((1.0, 0.0, 0.0, 0.0),),
    metadata: dict[str, object] | None = None,
) -> InteractiveEdit:
    return InteractiveEdit(
        shape=EditShape.VALUE,
        data_authority=data_authority,
        **(target or _target()),
        value=value,
        metadata={} if metadata is None else metadata,
    )


def _topology_edit(
    *,
    target: dict[str, object] | None = None,
    value: object = "new prim",
    metadata: dict[str, object] | None = None,
) -> InteractiveEdit:
    return InteractiveEdit(
        shape=EditShape.TOPOLOGY,
        data_authority=DataAuthority.VIEW,
        **(target or _target(blender_property_path="collection.children")),
        value=value,
        metadata={} if metadata is None else metadata,
    )


def test_viewport_camera_values_without_write_persistence() -> None:
    plan = InteractiveEditPlanner().plan(
        _value_edit(
            target=_target(
                usd_layer_id="",
                usd_prim_path="/World/Camera",
                usd_attribute="omni:xform",
                blender_property_path="viewport_camera_matrix",
                provenance={"source": "viewport_camera"},
            ),
        )
    )

    assert plan.mechanism == EditMechanism.UPDATE
    assert plan.persistence == EditPersistence.NONE
    assert plan.shape == EditShape.VALUE
    assert plan.data_authority == DataAuthority.VIEW
    assert plan.impact.physics_generation_reset_expected is False
    assert plan.usd_layer_id == ""
    assert plan.impact.render_session_reuse_expected is True
    assert plan.impact.whole_scene_export_avoided is True


def test_viewport_camera_without_edit_identity_is_unsupported() -> None:
    plan = InteractiveEditPlanner().plan(
        _value_edit(
            target=_target(
                usd_layer_id="",
                usd_prim_path="",
                usd_attribute="omni:xform",
                blender_property_path="viewport_camera_matrix",
                provenance={"source": "viewport_camera"},
            ),
        )
    )

    assert plan.mechanism == EditMechanism.NONE
    assert plan.persistence == EditPersistence.NONE
    assert plan.unsupported_reason == "missing_edit_identity"
    assert plan.impact.whole_scene_export_requested is False


def test_existing_transform_with_write_target_uses_update_and_write_persistence() -> None:
    plan = InteractiveEditPlanner().plan(_value_edit(target=_target(usd_layer_id="/layers/scene.usda")))

    assert plan.mechanism == EditMechanism.UPDATE
    assert plan.persistence == EditPersistence.WRITE
    assert plan.shape == EditShape.VALUE
    assert plan.data_authority == DataAuthority.VIEW
    assert plan.impact.physics_generation_reset_expected is False
    assert plan.usd_layer_id == "/layers/scene.usda"
    assert plan.impact.update_requested is True
    assert plan.impact.write_requested is True


def test_sim_body_velocity_routes_to_live_update() -> None:
    plan = InteractiveEditPlanner().plan(
        _value_edit(
            data_authority=DataAuthority.SIM,
            target=_target(
                usd_layer_id="",
                usd_attribute="physics:velocity",
                blender_property_path="",
            ),
            value=(4.0, 0.0, 0.0),
        )
    )

    assert plan.mechanism == EditMechanism.UPDATE
    assert plan.persistence == EditPersistence.NONE
    assert plan.impact.physics_generation_reset_expected is False


def test_update_without_layer_identity_remains_preview_only() -> None:
    plan = InteractiveEditPlanner().plan(
        _value_edit(
            target=_target(
                usd_layer_id="",
            )
        )
    )

    assert plan.mechanism == EditMechanism.UPDATE
    assert plan.persistence == EditPersistence.NONE
    assert plan.impact.update_requested is True
    assert plan.impact.write_requested is False


def test_sim_value_transform_uses_initial_condition_update() -> None:
    plan = InteractiveEditPlanner().plan(_value_edit(data_authority=DataAuthority.SIM))

    assert plan.mechanism == EditMechanism.UPDATE
    assert plan.persistence == EditPersistence.WRITE
    assert plan.data_authority == DataAuthority.SIM
    assert plan.impact.physics_generation_reset_expected is True


def test_sim_component_transforms_require_write_until_pose_merge_exists() -> None:
    planner = InteractiveEditPlanner()
    for usd_attribute, value in (
        ("xformOp:translate", (1.0, 2.0, 3.0)),
        ("xformOp:orient", (0.0, 0.0, 0.0, 1.0)),
    ):
        plan = planner.plan(
            _value_edit(
                data_authority=DataAuthority.SIM,
                target=_target(
                    usd_attribute=usd_attribute,
                    blender_property_path=usd_attribute,
                ),
                value=value,
            )
        )

        assert plan.mechanism == EditMechanism.NONE
        assert plan.persistence == EditPersistence.WRITE
        assert plan.impact.update_requested is False
        assert plan.impact.physics_generation_reset_expected is False


def test_look_only_value_edits_can_update_without_write_target() -> None:
    cases = (
        (
            "material",
            "/World/Looks/Paint/Shader",
            "inputs:diffuseColor",
            "diffuse_color",
            (0.8, 0.5, 0.2, 1.0),
        ),
        ("light", "/World/Key/KeyLight", "inputs:intensity", "energy", 900.0),
        ("world", "/World/StudioDome", "inputs:color", "world_dome", (1.0, 0.25, 0.0)),
        ("uv", "/World/Quad", "primvars:st", "uv_layers.active", ((0.0, 0.0), (1.0, 0.0))),
    )
    for kind, usd_prim_path, usd_attribute, blender_property_path, value in cases:
        plan = InteractiveEditPlanner().plan(
            _value_edit(
                target=_target(
                    usd_layer_id="",
                    usd_prim_path=usd_prim_path,
                    usd_attribute=usd_attribute,
                    blender_property_path=blender_property_path,
                ),
                value=value,
            )
        )

        assert plan.mechanism == EditMechanism.UPDATE
        assert plan.persistence == EditPersistence.NONE
        assert plan.reason == "update", kind
        assert plan.impact.physics_generation_reset_expected is False
        assert plan.usd_layer_id == ""


def test_unresolved_value_edit_persists_to_write_target() -> None:
    plan = InteractiveEditPlanner().plan(
        _value_edit(
            target=_target(
                usd_layer_id="/layers/look.usda",
                usd_prim_path="/World/Looks/Paint/Shader",
                usd_attribute="custom:unsupported",
                blender_property_path="custom_property",
            ),
            value=(0.8, 0.5, 0.2),
        )
    )

    assert plan.mechanism == EditMechanism.NONE
    assert plan.persistence == EditPersistence.WRITE
    assert plan.impact.physics_generation_reset_expected is False
    assert plan.reason == "update_unavailable"
    assert plan.usd_layer_id == "/layers/look.usda"
    assert plan.impact.write_requested is True
    assert plan.impact.update_requested is False
    assert plan.impact.whole_scene_export_requested is False


def test_value_edit_without_write_target_or_update_identity_is_unsupported() -> None:
    plan = InteractiveEditPlanner().plan(
        _value_edit(
            target=_target(
                usd_layer_id="",
                usd_prim_path="",
                usd_attribute="inputs:diffuseColor",
                blender_property_path="diffuse_color",
            ),
            value=(0.8, 0.5, 0.2),
        )
    )

    assert plan.mechanism == EditMechanism.NONE
    assert plan.persistence == EditPersistence.NONE
    assert plan.unsupported_reason == "missing_edit_identity"


def test_existing_light_and_world_values_update_by_target_kind() -> None:
    light_plan = InteractiveEditPlanner().plan(
        _value_edit(
            target=_target(
                usd_layer_id="/layers/lights.usda",
                usd_prim_path="/World/Key/KeyLight",
                usd_attribute="inputs:intensity",
                blender_property_path="energy",
            ),
            value=900.0,
        )
    )
    world_plan = InteractiveEditPlanner().plan(
        _value_edit(
            target=_target(
                usd_layer_id="/layers/lights.usda",
                usd_prim_path="/World/StudioDome",
                usd_attribute="inputs:color",
                blender_property_path="world_dome",
            ),
            value=(1.0, 0.25, 0.0),
        )
    )

    assert light_plan.mechanism == EditMechanism.UPDATE
    assert light_plan.impact.physics_generation_reset_expected is False
    assert world_plan.mechanism == EditMechanism.UPDATE
    assert world_plan.impact.physics_generation_reset_expected is False


def test_layer_resolved_uv_value_without_edit_identity_is_unsupported() -> None:
    plan = InteractiveEditPlanner().plan(
        _value_edit(
            target=_target(
                usd_layer_id="/layers/mesh.usda",
                usd_prim_path="",
                usd_attribute="primvars:st",
                blender_property_path="uv_layers.active",
            ),
            value=((0.0, 0.0),),
        )
    )

    assert plan.mechanism == EditMechanism.NONE
    assert plan.persistence == EditPersistence.NONE
    assert plan.unsupported_reason == "missing_edit_identity"


def test_world_value_without_update_identity_is_unsupported() -> None:
    plan = InteractiveEditPlanner().plan(
        _value_edit(
            target=_target(
                usd_layer_id="",
                usd_prim_path="",
                usd_attribute="inputs:color",
                blender_property_path="world_dome",
            ),
            value=(1.0, 0.25, 0.0),
        )
    )

    assert plan.mechanism == EditMechanism.NONE
    assert plan.persistence == EditPersistence.NONE
    assert plan.unsupported_reason == "missing_edit_identity"


def test_material_topology_uses_compose_write_not_value_update() -> None:
    plan = InteractiveEditPlanner().plan(
        _topology_edit(
            target=_target(
                usd_layer_id="/layers/look.usda",
                usd_attribute="material:binding",
                blender_property_path="node_tree",
            ),
            value="new shader node",
        )
    )

    assert plan.persistence == EditPersistence.WRITE
    assert plan.mechanism == EditMechanism.COMPOSE
    assert plan.shape == EditShape.TOPOLOGY
    assert plan.data_authority == DataAuthority.VIEW
    assert plan.impact.update_requested is False
    assert plan.impact.update_stream_rejected is True
    assert plan.impact.topology_reasons == (MATERIAL_GRAPH_CHANGED,)
    assert plan.impact.session_rekey_expected is True
    assert plan.impact.refinement_reset_expected is True
    assert plan.impact.render_session_reuse_expected is False


def test_topology_write_target_without_persistence_identity_is_unsupported() -> None:
    plan = InteractiveEditPlanner().plan(
        _topology_edit(
            target=_target(
                usd_layer_id="/layers/look.usda",
                usd_prim_path="",
                usd_attribute="material:binding",
                blender_property_path="node_tree",
            )
        )
    )

    assert plan.mechanism == EditMechanism.NONE
    assert plan.persistence == EditPersistence.NONE
    assert plan.unsupported_reason == "missing_persistence_identity"


def test_topology_without_write_target_requests_scene_generation_replacement() -> None:
    plan = InteractiveEditPlanner().plan(
        _topology_edit(
            target=_target(
                usd_layer_id="",
                usd_attribute="",
                blender_property_path="validation.collider_shape",
            ),
            value="new ramp collider",
        )
    )

    assert plan.mechanism == EditMechanism.COMPOSE
    assert plan.persistence == EditPersistence.NONE
    assert plan.reason == "scene_generation_replacement"
    assert plan.impact.topology_reasons == (COLLIDER_TOPOLOGY_CHANGED,)
    assert plan.impact.update_stream_rejected is True
    assert plan.impact.session_rekey_expected is True


def test_session_topology_requests_scene_generation_replacement() -> None:
    plan = InteractiveEditPlanner().plan(
        _topology_edit(
            target=_target(
                usd_layer_id="",
                usd_prim_path="/World/SessionLights/Key",
                usd_attribute="",
                blender_property_path="object.data",
            ),
            value="new light",
        )
    )

    assert plan.mechanism == EditMechanism.COMPOSE
    assert plan.persistence == EditPersistence.NONE
    assert plan.impact.physics_generation_reset_expected is False
    assert plan.impact.scene_generation_replacement_requested is True
    assert plan.impact.update_stream_rejected is True
    assert plan.impact.render_session_reuse_expected is False
    assert plan.impact.session_rekey_expected is True
    assert plan.impact.refinement_reset_expected is True


def test_topology_reason_metadata_overrides_default_reason() -> None:
    plan = InteractiveEditPlanner().plan(
        _topology_edit(
            target=_target(
                usd_layer_id="/layers/look.usda",
                usd_attribute="material:binding",
                blender_property_path="material_slots",
            ),
            value="new material binding",
            metadata={"topology_reasons": [MATERIAL_BINDING_CHANGED, MATERIAL_GRAPH_CHANGED]},
        )
    )

    assert plan.persistence == EditPersistence.WRITE
    assert plan.impact.topology_reasons == (MATERIAL_GRAPH_CHANGED, MATERIAL_BINDING_CHANGED)
    assert plan.impact.update_stream_rejected is True


def test_light_form_crossing_is_topology_not_light_value_update() -> None:
    plan = InteractiveEditPlanner().plan(
        _topology_edit(
            target=_target(
                usd_layer_id="/layers/lights.usda",
                usd_prim_path="/World/Key/KeyLight",
                usd_attribute="",
                blender_property_path="data.shape",
            ),
            value="AREA_DISK",
            metadata={"topology_change_kinds": ["light_form"]},
        )
    )

    assert plan.persistence == EditPersistence.WRITE
    assert plan.impact.update_requested is False
    assert plan.impact.topology_reasons == (LIGHT_FORM_CHANGED,)


def test_many_layer_value_and_topology_targets_choose_their_own_write_targets() -> None:
    value_plan = InteractiveEditPlanner().plan(
        _value_edit(
            target=_target(
                usd_layer_id="/layers/look.usda",
                usd_prim_path="/World/Looks/Paint/Shader",
                usd_attribute="inputs:roughness",
                blender_property_path="principled:Roughness",
            ),
            value=0.4,
        )
    )
    topology_plan = InteractiveEditPlanner().plan(
        _topology_edit(
            target=_target(
                usd_layer_id="/layers/layout.usda",
                usd_prim_path="/World/Set",
                usd_attribute="",
                blender_property_path="collection.children",
            ),
            value="new child prim",
        )
    )

    assert value_plan.mechanism == EditMechanism.UPDATE
    assert value_plan.persistence == EditPersistence.WRITE
    assert value_plan.usd_layer_id == "/layers/look.usda"
    assert topology_plan.persistence == EditPersistence.WRITE
    assert topology_plan.usd_layer_id == "/layers/layout.usda"


def test_edit_intent_is_available_only_for_update_plans() -> None:
    update_plan = InteractiveEditPlanner().plan(_value_edit())
    persistence_plan = InteractiveEditPlanner().plan(_topology_edit())

    intent = update_plan.to_intent()

    assert intent.shape == EditShape.VALUE
    assert intent.data_authority == DataAuthority.VIEW
    assert isinstance(intent, EditIntent)
    assert intent.impact.physics_generation_reset_expected is False
    assert intent.usd_attribute == "xformOp:transform"
    try:
        persistence_plan.to_intent()
    except ValueError as exc:
        assert "not an update" in str(exc)
    else:
        raise AssertionError("write plans must not produce edit intent")

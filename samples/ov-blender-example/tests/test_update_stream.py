# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import sys
from typing import Mapping, Sequence

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example.interactive_edit_planner import (  # noqa: E402
    DataAuthority,
    EditShape,
    edit_location,
    InteractiveEdit,
    InteractiveEditPlanner,
)
from ovrtx_blender_example.sim_update_stream import SimUpdateStream  # noqa: E402
from ovrtx_blender_example.ovphysx_stage import (  # noqa: E402
    OvphysxStageResult,
    OvphysxStageStatus,
)
from ovrtx_blender_example.shared_stage_composition import BodyPose, BodyVelocity  # noqa: E402
from ovrtx_blender_example.view_update_stream import ViewUpdateStream  # noqa: E402
from ovrtx_blender_example import uv_usd_prim as uv_identity  # noqa: E402
from ovrtx_blender_example.ovrtx_value_updates import (  # noqa: E402
    OvrtxAttributeValue,
    OvrtxTransformValue,
    OvrtxValueUpdateResult,
)


class _RenderPort:
    def __init__(self) -> None:
        self.updates: list[list[OvrtxTransformValue]] = []
        self.material_updates: list[list[OvrtxAttributeValue]] = []
        self.attribute_updates: list[list[OvrtxAttributeValue]] = []

    def update_transforms(
        self,
        values: Sequence[OvrtxTransformValue],
    ) -> OvrtxValueUpdateResult:
        self.updates.append(list(values))
        return OvrtxValueUpdateResult(len(values), 7 if values else None)

    def update_attribute_values(
        self,
        values: Sequence[OvrtxAttributeValue],
    ) -> OvrtxValueUpdateResult:
        self.attribute_updates.append(list(values))
        return OvrtxValueUpdateResult(len(values), 7 if values else None)


class _NoValueRenderPort:
    pass


class _WrongCountAttributeRenderPort(_RenderPort):
    def update_attribute_values(
        self,
        values: Sequence[OvrtxAttributeValue],
    ) -> OvrtxValueUpdateResult:
        self.attribute_updates.append(list(values))
        updated_count = max(0, len(values) - 1)
        return OvrtxValueUpdateResult(updated_count, 7 if updated_count else None)


class _Controller:
    def __init__(self) -> None:
        self.pose_updates: list[tuple[BodyPose, ...]] = []
        self.initial_overrides: tuple[BodyPose, ...] = ()
        self.started = False
        self.velocity_updates: list[tuple[BodyVelocity, ...]] = []
        self.calls: list[str] = []

    def apply_initial_condition_values(
        self,
        poses: Sequence[BodyPose],
        *,
        reset: bool = False,
    ) -> OvphysxStageResult:
        pose_tuple = tuple(poses)
        self.calls.append("pose")
        self.pose_updates.append(pose_tuple)
        return OvphysxStageResult(
            OvphysxStageStatus.OK,
            "initial_condition_value_edit",
            pose_tuple,
            tuple(pose.prim_path for pose in pose_tuple),
            0,
            7,
            1,
        )

    def apply_body_velocity_edits(
        self,
        velocities: Sequence[BodyVelocity],
        *,
        reset: bool = False,
    ) -> OvphysxStageResult:
        del reset
        values = tuple(velocities)
        self.calls.append("velocity")
        self.velocity_updates.append(values)
        return OvphysxStageResult(
            OvphysxStageStatus.OK,
            "body_velocity_edit",
            (),
            (),
            1,
            8,
            2,
        )


class _FailingController(_Controller):
    def apply_initial_condition_values(
        self,
        poses: Sequence[BodyPose],
        *,
        reset: bool = False,
    ) -> OvphysxStageResult:
        del poses, reset
        raise RuntimeError("sim write failed")


def _target(path: str) -> dict[str, object]:
    return edit_location(
        usd_layer_id="/layers/scene.usda",
        usd_prim_path=path,
        usd_attribute="xformOp:transform",
        blender_property_path="matrix_world",
        provenance={"source": "test"},
    )


def _edit_intent(data_authority: DataAuthority, path: str, value: object):
    plan = InteractiveEditPlanner().plan(
        InteractiveEdit(
            shape=EditShape.VALUE,
            data_authority=data_authority,
            **_target(path),
            value=value,
        )
    )
    return plan.to_intent()


def _velocity_intent(path: str, attribute: str, value: object):
    plan = InteractiveEditPlanner().plan(
        InteractiveEdit(
            shape=EditShape.VALUE,
            data_authority=DataAuthority.SIM,
            **edit_location(
                usd_prim_path=path,
                usd_attribute=attribute,
                provenance={"source": "test"},
            ),
            value=value,
        )
    )
    return plan.to_intent()


def _camera_edit_intent(path: str, value: object):
    plan = InteractiveEditPlanner().plan(
        InteractiveEdit(
            shape=EditShape.VALUE,
            data_authority=DataAuthority.VIEW,
            **edit_location(
                usd_prim_path=path,
                usd_attribute="omni:xform",
                blender_property_path="viewport_camera_matrix",
                provenance={"source": "viewport_camera"},
            ),
            value=value,
        )
    )
    return plan.to_intent()


def _camera_value_edit_intent(
    path: str,
    value: object,
    *,
    usd_attribute: str = "focalLength",
    blender_property_path: str = "data.lens",
):
    plan = InteractiveEditPlanner().plan(
        InteractiveEdit(
            shape=EditShape.VALUE,
            data_authority=DataAuthority.VIEW,
            **edit_location(
                usd_prim_path=path,
                usd_attribute=usd_attribute,
                blender_property_path=blender_property_path,
                provenance={
                    "source": "viewport_camera_projection",
                    "probe_class": "projection",
                },
            ),
            value=value,
        )
    )
    return plan.to_intent()


def _material_edit_intent(
    path: str,
    value: object,
    *,
    usd_attribute: str = "inputs:diffuseColor",
    blender_property_path: str = "diffuse_color",
):
    plan = InteractiveEditPlanner().plan(
        InteractiveEdit(
            shape=EditShape.VALUE,

            data_authority=DataAuthority.VIEW,
            **edit_location(
                usd_prim_path=path,
                usd_attribute=usd_attribute,
                blender_property_path=blender_property_path,
                provenance={"source": "test"},
            ),
            value=value,
        )
    )
    return plan.to_intent()


def _light_edit_intent(
    path: str,
    value: object,
    *,
    usd_attribute: str = "inputs:intensity",
    blender_property_path: str = "energy",
):
    plan = InteractiveEditPlanner().plan(
        InteractiveEdit(
            shape=EditShape.VALUE,

            data_authority=DataAuthority.VIEW,
            **edit_location(
                usd_prim_path=path,
                usd_attribute=usd_attribute,
                blender_property_path=blender_property_path,
                provenance={
                    "source": "test",
                    "light_path": "/World/Key/KeyLight",
                    "usd_family": "SphereLight",
                },
            ),
            value=value,
        )
    )
    return plan.to_intent()


def _world_edit_intent(
    path: str,
    value: object,
    *,
    usd_attribute: str = "inputs:intensity",
    blender_property_path: str = "world_dome",
):
    plan = InteractiveEditPlanner().plan(
        InteractiveEdit(
            shape=EditShape.VALUE,

            data_authority=DataAuthority.VIEW,
            **edit_location(
                usd_prim_path=path,
                usd_attribute=usd_attribute,
                blender_property_path=blender_property_path,
                provenance={
                    "source": "test",
                    "world_dome_conversion": {
                        "dome_light_scale": 360.0 * 3.141592653589793,
                        "formula": "effective_rgb -> color=effective_rgb/peak; intensity=peak*DOME_LIGHT_SCALE",
                    },
                },
            ),
            value=value,
        )
    )
    return plan.to_intent()


def _uv_edit_intent(
    path: str,
    value: object,
    *,
    usd_attribute: str = "primvars:st",
    blender_property_path: str = "uv_layers.active",
    validation_overrides: Mapping[str, object] | None = None,
):
    validation = {
        "status": uv_identity.RESOLVED,
        "validation_kind": uv_identity.VALIDATION_KIND,
        "mesh_prim_path": path,
        "target_attribute": uv_identity.TARGET_USD_ATTRIBUTE,
        "value_type": uv_identity.VALUE_TYPE,
        "uv_layer_name": "UVMap",
        "interpolation": "faceVarying",
        "indexed": False,
        "primvar_shape_status": uv_identity.RESOLVED,
        "element_count": len(value) if isinstance(value, Sequence) else 0,
        "topology_fingerprint": "test-topology",
        "blender_uv_digest": "test-blender-digest",
        "source_uv_digest": "test-usd-digest",
        "tolerance": 1.0e-6,
    }
    if validation_overrides:
        validation.update(validation_overrides)
    provenance = {"source": "test"}
    if validation_overrides is None or validation_overrides.get("status") is not None:
        provenance["uv_loop_order_validation"] = validation
    plan = InteractiveEditPlanner().plan(
        InteractiveEdit(
            shape=EditShape.VALUE,

            data_authority=DataAuthority.VIEW,
            **edit_location(
                usd_prim_path=path,
                usd_attribute=usd_attribute,
                blender_property_path=blender_property_path,
                provenance=provenance,
            ),
            value=value,
            metadata={"usd_value_type": "Float2Array"},
        )
    )
    return plan.to_intent()


def test_update_stream_batches_view_value_updates() -> None:
    stream = ViewUpdateStream()
    ovrtx_updates = _RenderPort()
    matrix_a = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (2.0, 3.0, 4.0, 1.0),
    )
    matrix_b = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (5.0, 6.0, 7.0, 1.0),
    )

    stream.queue(_edit_intent(DataAuthority.VIEW, "/World/Cube_A", matrix_a))
    stream.queue(_edit_intent(DataAuthority.VIEW, "/World/Cube_B", matrix_b))
    diagnostics = stream.apply_pending(ovrtx_updates)

    assert len(ovrtx_updates.updates) == 1
    assert [value.prim_path for value in ovrtx_updates.updates[0]] == ["/World/Cube_A", "/World/Cube_B"]
    assert diagnostics["values_written"] is True
    assert diagnostics["value_count"] == 2
    assert diagnostics["value_paths"] == ["/World/Cube_A", "/World/Cube_B"]
    assert stream.last_result["values_written"] is True


def test_update_stream_coalesces_duplicate_transform_paths_latest_wins() -> None:
    # task04-01: a gizmo drag queues several depsgraph transform edits for
    # the same object between ticks; the client rejects duplicate prim paths
    # in one update_transforms batch, so the batch applies latest-wins per
    # prim (first-seen order kept). Cross-batch coalescing is task04-06.
    stream = ViewUpdateStream()
    ovrtx_updates = _RenderPort()
    stale = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (2.0, 3.0, 4.0, 1.0),
    )
    other = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (5.0, 6.0, 7.0, 1.0),
    )
    newest = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (8.0, 9.0, 10.0, 1.0),
    )

    stream.queue(_edit_intent(DataAuthority.VIEW, "/World/Cube_A", stale))
    stream.queue(_edit_intent(DataAuthority.VIEW, "/World/Cube_B", other))
    stream.queue(_edit_intent(DataAuthority.VIEW, "/World/Cube_A", newest))
    diagnostics = stream.apply_pending(ovrtx_updates)

    assert len(ovrtx_updates.updates) == 1
    applied = ovrtx_updates.updates[0]
    assert [value.prim_path for value in applied] == ["/World/Cube_A", "/World/Cube_B"]
    assert applied[0].matrix == [list(row) for row in newest]
    assert applied[1].matrix == [list(row) for row in other]
    assert diagnostics["values_written"] is True
    assert diagnostics["value_requested_count"] == 2
    assert diagnostics["value_count"] == 2
    assert diagnostics["value_paths"] == ["/World/Cube_A", "/World/Cube_B"]


def test_update_stream_coalesces_duplicate_attribute_targets_latest_wins() -> None:
    # task04-06: a slider/color-picker drag queues several edits for one
    # (usd_prim_path, usd_attribute) target between ticks; the client
    # rejects duplicate attribute targets in one update_attribute_values
    # batch, so the batch applies latest-wins per target. Distinct targets
    # (same prim different attribute, different prim) all apply, in
    # first-submission order.
    stream = ViewUpdateStream()
    ovrtx_updates = _RenderPort()
    shader = "/World/Looks/Paint/Shader"
    other_shader = "/World/Looks/Trim/Shader"

    stream.queue(_material_edit_intent(shader, (0.9, 0.0, 0.0, 1.0)))
    stream.queue(
        _material_edit_intent(
            shader,
            0.75,
            usd_attribute="inputs:roughness",
            blender_property_path="principled:Roughness",
        )
    )
    stream.queue(_material_edit_intent(other_shader, (0.0, 0.9, 0.0, 1.0)))
    stream.queue(_material_edit_intent(shader, (0.2, 0.4, 0.6, 1.0)))
    stream.queue(_material_edit_intent(shader, (0.1, 0.2, 0.3, 1.0)))
    diagnostics = stream.apply_pending(ovrtx_updates)

    # One RPC, one value per distinct target, newest value per target,
    # first-submission order across distinct targets.
    assert len(ovrtx_updates.attribute_updates) == 1
    assert ovrtx_updates.attribute_updates[0] == [
        OvrtxAttributeValue(shader, "inputs:diffuseColor", [0.1, 0.2, 0.3], "Color3f"),
        OvrtxAttributeValue(shader, "inputs:roughness", 0.75, "Float"),
        OvrtxAttributeValue(other_shader, "inputs:diffuseColor", [0.0, 0.9, 0.0], "Color3f"),
    ]
    assert diagnostics["values_written"] is True
    assert diagnostics["value_requested_count"] == 3
    assert diagnostics["value_count"] == 3
    # Target evidence describes current desired state, not raw drag history.
    assert len(diagnostics["targets"]) == 3


def test_update_stream_coalesces_camera_value_targets_latest_wins() -> None:
    # task04-06 (04-05 follow-up): cross-iteration queued camera projection
    # value intents for one attribute previously collided in one tick's
    # update_attribute_values batch; the camera_value lane coalesces
    # latest-wins per (prim, attribute) like every attribute lane.
    stream = ViewUpdateStream()
    ovrtx_updates = _RenderPort()
    camera = "/World/RenderCamera"

    stream.queue(_camera_value_edit_intent(camera, 35.0))
    stream.queue(_camera_value_edit_intent(camera, 50.0))
    diagnostics = stream.apply_pending(ovrtx_updates)

    assert ovrtx_updates.attribute_updates == [[
        OvrtxAttributeValue(camera, "focalLength", 50.0, "Float")
    ]]
    assert diagnostics["values_written"] is True
    assert diagnostics["value_requested_count"] == 1
    assert diagnostics["value_count"] == 1
    assert diagnostics["camera_value_probe_class"] == "projection"


def test_update_stream_coalesces_uv_float2_array_updates() -> None:
    # task04-06: UV Float2Array payloads coalesce too — large arrays
    # benefit most from dropping a drag's intermediate values.
    stream = ViewUpdateStream()
    ovrtx_updates = _RenderPort()
    stale = ((0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5))
    newest = ((0.1, 0.1), (0.6, 0.1), (0.6, 0.6), (0.1, 0.6))

    stream.queue(_uv_edit_intent("/World/Quad", stale))
    stream.queue(_uv_edit_intent("/World/Quad", newest))
    diagnostics = stream.apply_pending(ovrtx_updates)

    assert ovrtx_updates.attribute_updates == [[
        OvrtxAttributeValue(
            "/World/Quad",
            "primvars:st",
            [(0.1, 0.1), (0.6, 0.1), (0.6, 0.6), (0.1, 0.6)],
            "Float2Array",
        )
    ]]
    assert diagnostics["values_written"] is True
    assert diagnostics["value_requested_count"] == 1
    assert diagnostics["value_count"] == 1


def test_update_stream_combines_current_targets_across_lanes() -> None:
    stream = ViewUpdateStream()
    ovrtx_updates = _RenderPort()
    stale = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (2.0, 3.0, 4.0, 1.0),
    )
    newest = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (8.0, 9.0, 10.0, 1.0),
    )

    stream.queue(_edit_intent(DataAuthority.VIEW, "/World/Cube_A", stale))
    stream.queue(_edit_intent(DataAuthority.VIEW, "/World/Cube_A", newest))
    stream.queue(_light_edit_intent("/World/Key/KeyLight", 100.0))
    stream.queue(_light_edit_intent("/World/Key/KeyLight", 900.0))
    stream.queue(_material_edit_intent("/World/Looks/Paint/Shader", (0.1, 0.2, 0.3, 1.0)))
    diagnostics = stream.apply_pending(ovrtx_updates)

    # Transforms in one RPC; each attribute lane applies its own batch.
    assert len(ovrtx_updates.updates) == 1
    assert [value.prim_path for value in ovrtx_updates.updates[0]] == ["/World/Cube_A"]
    assert ovrtx_updates.updates[0][0].matrix == [list(row) for row in newest]
    light_batches = [
        batch
        for batch in ovrtx_updates.attribute_updates
        if batch and batch[0].attribute == "inputs:intensity"
    ]
    assert light_batches == [[
        OvrtxAttributeValue("/World/Key/KeyLight", "inputs:intensity", 900.0, "Float")
    ]]
    assert diagnostics["values_written"] is True
    assert diagnostics["value_requested_count"] == 3
    assert diagnostics["value_count"] == 3


def test_update_stream_retains_edit_queued_during_apply_pending() -> None:
    # apply_pending swaps the pending list atomically (task02-03: the main
    # thread queues while the render thread applies); an edit that lands
    # mid-application must survive for the next apply, never be dropped.
    stream = ViewUpdateStream()
    matrix = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (2.0, 3.0, 4.0, 1.0),
    )
    late_intent = _edit_intent(DataAuthority.VIEW, "/World/Cube_Late", matrix)

    class _QueueDuringApplyPort(_RenderPort):
        queued_late = False

        def update_transforms(
            self,
            values: Sequence[OvrtxTransformValue],
        ) -> OvrtxValueUpdateResult:
            if not self.queued_late:
                self.queued_late = True
                stream.queue(late_intent)
            return super().update_transforms(values)

    ovrtx_updates = _QueueDuringApplyPort()
    stream.queue(_edit_intent(DataAuthority.VIEW, "/World/Cube_A", matrix))

    first = stream.apply_pending(ovrtx_updates)
    assert first["value_paths"] == ["/World/Cube_A"]
    assert stream.has_pending is True

    second = stream.apply_pending(ovrtx_updates)
    assert second["value_paths"] == ["/World/Cube_Late"]
    assert stream.has_pending is False
    assert [
        [value.prim_path for value in batch] for batch in ovrtx_updates.updates
    ] == [["/World/Cube_A"], ["/World/Cube_Late"]]


def test_update_stream_applies_camera_target_as_view_value_update() -> None:
    retained = []
    stream = ViewUpdateStream(transform_sink=lambda values: retained.extend(values))
    ovrtx_updates = _RenderPort()
    matrix = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (2.0, 3.0, 4.0, 1.0),
    )
    intent = _camera_edit_intent("/World/Camera", matrix)

    assert stream.supports(intent) is True

    stream.queue(intent)
    diagnostics = stream.apply_pending(ovrtx_updates)

    assert ovrtx_updates.updates == [[OvrtxTransformValue("/World/Camera", [list(row) for row in matrix])]]
    assert diagnostics["values_written"] is True
    assert diagnostics["data_authority"] == "view"
    assert diagnostics["physics_generation_reset"] is False
    assert diagnostics["value_paths"] == ["/World/Camera"]
    assert retained == [
        OvrtxTransformValue("/World/Camera", [list(row) for row in matrix])
    ]


def test_update_stream_retains_scene_transform_when_queued() -> None:
    retained = []
    stream = ViewUpdateStream(transform_sink=lambda values: retained.extend(values))
    ovrtx_updates = _RenderPort()
    matrix = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (2.0, 3.0, 4.0, 1.0),
    )

    stream.queue(_edit_intent(DataAuthority.VIEW, "/World/Cube", matrix))

    assert retained == [
        OvrtxTransformValue("/World/Cube", [list(row) for row in matrix])
    ]

    stream.apply_pending(ovrtx_updates)

    assert retained == [
        OvrtxTransformValue("/World/Cube", [list(row) for row in matrix])
    ]


def test_update_stream_applies_material_value_updates() -> None:
    retained = []
    stream = ViewUpdateStream(attribute_sink=lambda values: retained.extend(values))
    ovrtx_updates = _RenderPort()

    stream.queue(_material_edit_intent("/World/Looks/Paint/Shader", (0.1, 0.2, 0.3, 1.0)))
    diagnostics = stream.apply_pending(ovrtx_updates)

    assert ovrtx_updates.material_updates == []
    assert len(ovrtx_updates.attribute_updates) == 1
    assert ovrtx_updates.attribute_updates[0] == [
        OvrtxAttributeValue(
            "/World/Looks/Paint/Shader",
            "inputs:diffuseColor",
            [0.1, 0.2, 0.3],
            "Color3f",
        )
    ]
    assert retained == ovrtx_updates.attribute_updates[0]
    assert diagnostics["values_written"] is True
    assert diagnostics["value_count"] == 1
    assert diagnostics["value_paths"] == ["/World/Looks/Paint/Shader"]
    assert diagnostics["value_attributes"] == ["inputs:diffuseColor"]
    assert diagnostics["value_requested_count"] == 1
    assert diagnostics["supported_by_client"] is True
    assert diagnostics["accepted_by_worker"] is True


def test_update_stream_keeps_transform_diagnostics_in_mixed_batches() -> None:
    stream = ViewUpdateStream()
    ovrtx_updates = _RenderPort()
    matrix = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (2.0, 3.0, 4.0, 1.0),
    )

    stream.queue(_edit_intent(DataAuthority.VIEW, "/World/Cube_A", matrix))
    stream.queue(_material_edit_intent("/World/Looks/Paint/Shader", (0.1, 0.2, 0.3, 1.0)))
    diagnostics = stream.apply_pending(ovrtx_updates)

    assert len(ovrtx_updates.updates) == 1
    assert len(ovrtx_updates.attribute_updates) == 1
    assert diagnostics["values_written"] is True
    assert diagnostics["value_count"] == 2
    assert diagnostics["value_paths"] == ["/World/Cube_A", "/World/Looks/Paint/Shader"]
    assert diagnostics["physics_generation_reset"] is False
    assert len(diagnostics["updates"]) == 2


def test_update_stream_applies_uv_value_updates_as_float2_arrays() -> None:
    stream = ViewUpdateStream()
    ovrtx_updates = _RenderPort()
    uv_values = ((0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5))

    stream.queue(_uv_edit_intent("/World/Quad", uv_values))
    diagnostics = stream.apply_pending(ovrtx_updates)

    assert ovrtx_updates.attribute_updates == [[
        OvrtxAttributeValue(
            "/World/Quad",
            "primvars:st",
            [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)],
            "Float2Array",
        )
    ]]
    assert diagnostics["values_written"] is True
    assert diagnostics["physics_generation_reset"] is False
    assert diagnostics["value_paths"] == ["/World/Quad"]
    assert diagnostics["value_attributes"] == ["primvars:st"]
    assert diagnostics["value_types"] == ["Float2Array"]
    assert diagnostics["value_element_counts"] == [4]
    assert diagnostics["accepted_by_worker"] is True
    assert diagnostics["values_written"] is True


def test_update_stream_rejects_unsupported_uv_attribute() -> None:
    stream = ViewUpdateStream()
    ovrtx_updates = _RenderPort()

    stream.queue(_uv_edit_intent("/World/Quad", ((0.0, 0.0),), usd_attribute="primvars:displayColor"))
    diagnostics = stream.apply_pending(ovrtx_updates)

    assert ovrtx_updates.attribute_updates == []
    assert diagnostics["values_written"] is False
    assert diagnostics["failed"] is True
    assert diagnostics["skipped_reason"] == "unsupported_uv_value_attribute"
    assert diagnostics["supported_attribute"] is False


def test_update_stream_rejects_uv_value_with_minimal_claimed_validation() -> None:
    stream = ViewUpdateStream()
    ovrtx_updates = _RenderPort()

    stream.queue(
        _uv_edit_intent(
            "/World/Quad",
            ((0.0, 0.0),),
            validation_overrides={
                "uv_layer_name": "",
                "topology_fingerprint": "",
                "blender_uv_digest": "",
                "source_uv_digest": "",
            },
        )
    )
    diagnostics = stream.apply_pending(ovrtx_updates)

    assert ovrtx_updates.attribute_updates == []
    assert diagnostics["values_written"] is False
    assert diagnostics["failed"] is True
    assert diagnostics["skipped_reason"] == "unsupported_uv_value_attribute"


def test_update_stream_rejects_uv_value_without_loop_order_validation() -> None:
    stream = ViewUpdateStream()
    ovrtx_updates = _RenderPort()

    stream.queue(
        _uv_edit_intent(
            "/World/Quad",
            ((0.0, 0.0),),
            validation_overrides={"status": None},
        )
    )
    diagnostics = stream.apply_pending(ovrtx_updates)

    assert ovrtx_updates.attribute_updates == []
    assert diagnostics["values_written"] is False
    assert diagnostics["failed"] is True
    assert diagnostics["skipped_reason"] == "unsupported_uv_value_attribute"


def test_update_stream_rejects_uv_value_with_stale_loop_order_count() -> None:
    stream = ViewUpdateStream()
    ovrtx_updates = _RenderPort()

    stream.queue(
        _uv_edit_intent(
            "/World/Quad",
            ((0.0, 0.0), (1.0, 0.0)),
            validation_overrides={"element_count": 1},
        )
    )
    diagnostics = stream.apply_pending(ovrtx_updates)

    assert ovrtx_updates.attribute_updates == []
    assert diagnostics["values_written"] is False
    assert diagnostics["failed"] is True
    assert diagnostics["skipped_reason"] == "unsupported_uv_value_attribute"


def test_update_stream_rejects_uv_value_with_indexed_primvar_validation() -> None:
    stream = ViewUpdateStream()
    ovrtx_updates = _RenderPort()

    stream.queue(
        _uv_edit_intent(
            "/World/Quad",
            ((0.0, 0.0),),
            validation_overrides={
                "indexed": True,
                "primvar_shape_status": uv_identity.ERROR_INDEXED_PRIMVAR,
            },
        )
    )
    diagnostics = stream.apply_pending(ovrtx_updates)

    assert ovrtx_updates.attribute_updates == []
    assert diagnostics["values_written"] is False
    assert diagnostics["failed"] is True
    assert diagnostics["skipped_reason"] == "unsupported_uv_value_attribute"


def test_update_stream_applies_float_material_value_updates() -> None:
    stream = ViewUpdateStream()
    ovrtx_updates = _RenderPort()
    path = "/World/Looks/Paint/Shader"

    stream.queue(
        _material_edit_intent(
            path,
            0.75,
            usd_attribute="inputs:roughness",
            blender_property_path="principled:Roughness",
        )
    )
    stream.queue(
        _material_edit_intent(
            path,
            0.2,
            usd_attribute="inputs:metallic",
            blender_property_path="principled:Metallic",
        )
    )
    stream.queue(
        _material_edit_intent(
            path,
            1.45,
            usd_attribute="inputs:ior",
            blender_property_path="principled:IOR",
        )
    )
    stream.queue(
        _material_edit_intent(
            path,
            (0.2, 0.4, 1.0),
            usd_attribute="inputs:emissiveColor",
            blender_property_path="principled:Emission",
        )
    )
    diagnostics = stream.apply_pending(ovrtx_updates)

    assert ovrtx_updates.material_updates == []
    assert ovrtx_updates.attribute_updates[0] == [
        OvrtxAttributeValue(path, "inputs:roughness", 0.75, "Float"),
        OvrtxAttributeValue(path, "inputs:metallic", 0.2, "Float"),
        OvrtxAttributeValue(path, "inputs:ior", 1.45, "Float"),
        OvrtxAttributeValue(path, "inputs:emissiveColor", [0.2, 0.4, 1.0], "Color3f"),
    ]
    assert diagnostics["values_written"] is True
    assert diagnostics["value_count"] == 4
    assert diagnostics["value_attributes"] == [
        "inputs:roughness",
        "inputs:metallic",
        "inputs:ior",
        "inputs:emissiveColor",
    ]
    assert diagnostics["value_requested_count"] == 4
    assert diagnostics["accepted_by_worker"] is True


def test_update_stream_fails_closed_for_unproven_material_attribute() -> None:
    stream = ViewUpdateStream()
    ovrtx_updates = _RenderPort()

    stream.queue(
        _material_edit_intent(
            "/World/Looks/Paint/Shader",
            0.55,
            usd_attribute="inputs:opacity",
            blender_property_path="principled:Alpha",
        )
    )
    diagnostics = stream.apply_pending(ovrtx_updates)

    assert ovrtx_updates.material_updates == []
    assert ovrtx_updates.attribute_updates == []
    assert diagnostics["values_written"] is False
    assert diagnostics["failed"] is True
    assert diagnostics["skipped_reason"] == "unsupported_material_value_attribute"
    assert diagnostics["supported_attribute"] is False
    assert diagnostics["accepted_by_worker"] is False
    assert diagnostics["value_requested_count"] == 1
    assert diagnostics["value_count"] == 0
    assert "unsupported material value attribute" in diagnostics["result"]["error"]


def test_update_stream_rejects_wrong_material_result_count() -> None:
    stream = ViewUpdateStream()
    ovrtx_updates = _WrongCountAttributeRenderPort()

    stream.queue(_material_edit_intent("/World/Looks/Paint/Shader", (0.1, 0.2, 0.3, 1.0)))
    stream.queue(
        _material_edit_intent(
            "/World/Looks/Paint/Shader",
            0.75,
            usd_attribute="inputs:roughness",
            blender_property_path="principled:Roughness",
        )
    )
    diagnostics = stream.apply_pending(ovrtx_updates)

    assert diagnostics["values_written"] is False
    assert diagnostics["failed"] is True
    assert diagnostics["skipped_reason"] == "material_value_update_error"
    assert diagnostics["value_requested_count"] == 2
    assert diagnostics["accepted_by_worker"] is False


def test_update_stream_fails_closed_without_material_value_port() -> None:
    stream = ViewUpdateStream()

    stream.queue(_material_edit_intent("/World/Looks/Paint/Shader", (0.1, 0.2, 0.3, 1.0)))
    diagnostics = stream.apply_pending(_NoValueRenderPort())

    assert diagnostics["values_written"] is False
    assert diagnostics["failed"] is True
    assert diagnostics["skipped_reason"] == "value_update_unavailable"
    assert diagnostics["supported_by_client"] is False
    assert diagnostics["accepted_by_worker"] is False


def test_update_stream_applies_light_value_updates_through_attribute_port() -> None:
    stream = ViewUpdateStream()
    ovrtx_updates = _RenderPort()
    path = "/World/Key/KeyLight"

    stream.queue(_light_edit_intent(path, 900.0))
    stream.queue(
        _light_edit_intent(
            path,
            (1.0, 0.8, 0.6),
            usd_attribute="inputs:color",
            blender_property_path="color",
        )
    )
    stream.queue(
        _light_edit_intent(
            path,
            False,
            usd_attribute="inputs:normalize",
            blender_property_path="normalize",
        )
    )
    diagnostics = stream.apply_pending(ovrtx_updates)

    assert ovrtx_updates.attribute_updates[0] == [
        OvrtxAttributeValue(path, "inputs:intensity", 900.0, "Float"),
        OvrtxAttributeValue(path, "inputs:color", [1.0, 0.8, 0.6], "Color3f"),
        OvrtxAttributeValue(path, "inputs:normalize", False, "Bool"),
    ]
    assert diagnostics["values_written"] is True
    assert diagnostics["data_authority"] == "view"
    assert diagnostics["value_requested_count"] == 3
    assert diagnostics["value_attributes"] == [
        "inputs:intensity",
        "inputs:color",
        "inputs:normalize",
    ]
    assert diagnostics["accepted_by_worker"] is True


def test_update_stream_classifies_all_public_light_policy_properties_as_light() -> None:
    stream = ViewUpdateStream()
    ovrtx_updates = _RenderPort()
    path = "/World/Key/KeyLight"
    attributes = (
        ("inputs:enableColorTemperature", "use_temperature", False),
        ("inputs:radius", "shadow_soft_size", 0.25),
        ("inputs:angle", "angle", 0.5),
        ("inputs:width", "size", 2.0),
        ("inputs:height", "size_y", 3.0),
        ("inputs:shaping:cone:angle", "spot_size", 30.0),
        ("inputs:shaping:cone:softness", "spot_blend", 0.25),
    )
    for usd_attribute, blender_property_path, value in attributes:
        stream.queue(
            _light_edit_intent(
                path,
                value,
                usd_attribute=usd_attribute,
                blender_property_path=blender_property_path,
            )
        )

    diagnostics = stream.apply_pending(ovrtx_updates)

    assert diagnostics["failed"] is False
    assert diagnostics["value_requested_count"] == len(attributes)
    assert [value.attribute for value in ovrtx_updates.attribute_updates[0]] == [
        usd_attribute for usd_attribute, _blender_property_path, _value in attributes
    ]


def test_update_stream_fails_closed_for_unproven_light_attribute() -> None:
    stream = ViewUpdateStream()
    ovrtx_updates = _RenderPort()

    stream.queue(
        _light_edit_intent(
            "/World/Key/KeyLight",
            2.0,
            usd_attribute="inputs:exposure",
            blender_property_path="exposure",
        )
    )
    diagnostics = stream.apply_pending(ovrtx_updates)

    assert ovrtx_updates.attribute_updates == []
    assert diagnostics["values_written"] is False
    assert diagnostics["failed"] is True
    assert diagnostics["skipped_reason"] == "unsupported_light_value_attribute"
    assert diagnostics["supported_attribute"] is False
    assert diagnostics["accepted_by_worker"] is False
    assert diagnostics["value_requested_count"] == 1
    assert diagnostics["value_count"] == 0


def test_update_stream_rejects_wrong_light_result_count() -> None:
    stream = ViewUpdateStream()
    ovrtx_updates = _WrongCountAttributeRenderPort()
    path = "/World/Key/KeyLight"

    stream.queue(_light_edit_intent(path, 900.0))
    stream.queue(
        _light_edit_intent(
            path,
            2.0,
            usd_attribute="inputs:width",
            blender_property_path="size",
        )
    )
    diagnostics = stream.apply_pending(ovrtx_updates)

    assert diagnostics["values_written"] is False
    assert diagnostics["failed"] is True
    assert diagnostics["skipped_reason"] == "light_value_update_error"
    assert diagnostics["value_requested_count"] == 2
    assert diagnostics["accepted_by_worker"] is False


def test_update_stream_applies_world_dome_values_through_attribute_port() -> None:
    stream = ViewUpdateStream()
    ovrtx_updates = _RenderPort()
    path = "/World/StudioDome"

    stream.queue(_world_edit_intent(path, 1133.8))
    stream.queue(
        _world_edit_intent(
            path,
            (1.0, 0.25, 0.0),
            usd_attribute="inputs:color",
        )
    )
    diagnostics = stream.apply_pending(ovrtx_updates)

    assert ovrtx_updates.attribute_updates[0] == [
        OvrtxAttributeValue(path, "inputs:intensity", 1133.8, "Float"),
        OvrtxAttributeValue(path, "inputs:color", [1.0, 0.25, 0.0], "Color3f"),
    ]
    assert diagnostics["values_written"] is True
    assert diagnostics["data_authority"] == "view"
    assert diagnostics["value_requested_count"] == 2
    assert diagnostics["value_attributes"] == ["inputs:intensity", "inputs:color"]
    assert diagnostics["world_dome_owner_path"] == path
    assert diagnostics["world_dome_conversion"]["formula"].endswith("DOME_LIGHT_SCALE")
    assert diagnostics["accepted_by_worker"] is True


def test_update_stream_fails_closed_for_unproven_world_attribute() -> None:
    stream = ViewUpdateStream()
    ovrtx_updates = _RenderPort()

    stream.queue(
        _world_edit_intent(
            "/World/StudioDome",
            "@looks/env.hdr@",
            usd_attribute="inputs:texture:file",
        )
    )
    diagnostics = stream.apply_pending(ovrtx_updates)

    assert ovrtx_updates.attribute_updates == []
    assert diagnostics["values_written"] is False
    assert diagnostics["failed"] is True
    assert diagnostics["skipped_reason"] == "unsupported_world_value_attribute"
    assert diagnostics["supported_attribute"] is False
    assert diagnostics["accepted_by_worker"] is False
    assert diagnostics["value_requested_count"] == 1
    assert diagnostics["value_count"] == 0


def test_update_stream_rejects_wrong_world_result_count() -> None:
    stream = ViewUpdateStream()
    ovrtx_updates = _WrongCountAttributeRenderPort()
    path = "/World/StudioDome"

    stream.queue(_world_edit_intent(path, 1133.8))
    stream.queue(
        _world_edit_intent(
            path,
            (1.0, 0.25, 0.0),
            usd_attribute="inputs:color",
        )
    )
    diagnostics = stream.apply_pending(ovrtx_updates)

    assert diagnostics["values_written"] is False
    assert diagnostics["failed"] is True
    assert diagnostics["skipped_reason"] == "world_value_update_error"
    assert diagnostics["value_requested_count"] == 2
    assert diagnostics["accepted_by_worker"] is False


def test_sim_update_stream_applies_without_ovrtx_updates_and_retains_value() -> None:
    stream = SimUpdateStream()
    controller = _Controller()
    value = {"translate": (2.0, 7.0, 4.0), "orient": (0.0, 0.0, 0.0, 1.0)}

    stream.queue(_edit_intent(DataAuthority.SIM, "/World/Cube_A", value))
    applied = stream.apply_pending(controller)

    assert applied["values_written"] is True
    assert applied["physics_generation_reset"] is True
    assert applied["transform_updated"] is True
    assert applied["sim_value_paths"] == ["/World/Cube_A"]
    assert stream.values_for_controller_start(controller_started=False) == (
        BodyPose("/World/Cube_A", (2.0, 7.0, 4.0), (0.0, 0.0, 0.0, 1.0)),
    )
    assert len(controller.pose_updates) == 1


def test_sim_update_stream_routes_and_retains_body_velocity() -> None:
    stream = SimUpdateStream()
    controller = _Controller()
    stream.queue(_velocity_intent("/World/Cube_A", "physics:velocity", (5.0, 0.0, 0.0)))
    stream.queue(_velocity_intent("/World/Cube_A", "physics:angularVelocity", (0.0, 0.0, 2.0)))

    applied = stream.apply_pending(controller)

    velocity = BodyVelocity("/World/Cube_A", (5.0, 0.0, 0.0), (0.0, 0.0, 2.0))
    assert applied["values_written"] is True
    assert applied["physics_generation_reset"] is False
    assert applied["physics_generation_invalidated"] is True
    assert applied["transform_updated"] is False
    assert applied["render_value_write_applied"] is False
    assert controller.velocity_updates == [(velocity,)]
    assert stream.diagnostics()["velocity_paths"] == ["/World/Cube_A"]

    replacement = _Controller()
    started = OvphysxStageResult(OvphysxStageStatus.OK, "initial", (), (), 0, 0, 0)
    replay = stream.record_controller_start(started, replacement)
    assert replay["reason"] == "body_velocity_values"
    assert replacement.velocity_updates == [(velocity,)]


@pytest.mark.parametrize("velocity_first", [False, True])
def test_sim_update_stream_partitions_mixed_pose_and_velocity_in_kind_order(
    velocity_first: bool,
) -> None:
    stream = SimUpdateStream()
    controller = _Controller()
    pose = _edit_intent(
        DataAuthority.SIM,
        "/World/Cube_A",
        {"translate": (2.0, 7.0, 4.0), "orient": (0.0, 0.0, 0.0, 1.0)},
    )
    velocity = _velocity_intent("/World/Cube_A", "physics:velocity", (5.0, 0.0, 0.0))
    for intent in ((velocity, pose) if velocity_first else (pose, velocity)):
        stream.queue(intent)

    result = stream.apply_pending(controller)

    assert result["failed"] is False
    assert result["value_requested_count"] == 2
    assert controller.calls == (["velocity", "pose"] if velocity_first else ["pose", "velocity"])
    assert stream.values_for_controller_start(controller_started=False) == (
        BodyPose("/World/Cube_A", (2.0, 7.0, 4.0), (0.0, 0.0, 0.0, 1.0)),
    )
    assert stream.diagnostics()["velocity_paths"] == ["/World/Cube_A"]
    assert result["physics_generation_reset"] is True
    velocity_result = next(update for update in result["updates"] if update["reason"] == "body_velocity_update")
    assert velocity_result["physics_generation_reset"] is False
    assert velocity_result["transform_updated"] is False


def test_sim_update_stream_rejects_invalid_pose_before_controller_application() -> None:
    stream = SimUpdateStream()
    controller = _Controller()
    stream.queue(_edit_intent(
        DataAuthority.SIM,
        "/World/Cube_A",
        {"translate": (2.0, 7.0, 4.0), "orient": (0.0, 0.0, 0.0, 0.0)},
    ))

    result = stream.apply_pending(controller)

    assert result["failed"] is True
    assert controller.pose_updates == []


@pytest.mark.parametrize(
    "value",
    [
        {
            "matrix": (
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (2.0, 7.0, 4.0, 1.0),
            )
        },
        {
            "omni:xform": (
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                2.0, 7.0, 4.0, 1.0,
            )
        },
    ],
)
def test_sim_update_stream_preserves_supported_matrix_shapes(value: object) -> None:
    stream = SimUpdateStream()
    controller = _Controller()

    stream.queue(_edit_intent(DataAuthority.SIM, "/World/Cube_A", value))
    applied = stream.apply_pending(controller)

    assert applied["failed"] is False
    assert stream.values_for_controller_start(controller_started=False) == (
        BodyPose("/World/Cube_A", (2.0, 7.0, 4.0), (0.0, 0.0, 0.0, 1.0)),
    )


def test_sim_update_stream_retains_edit_queued_during_apply_pending() -> None:
    # apply_pending swaps the pending list atomically (task02-07: the main
    # thread queues sim edits while the render thread applies them inside
    # the scheduler tick); an edit that lands mid-application must survive
    # for the next apply, never be dropped. Same contract ViewUpdateStream
    # received in the task02-03 review.
    stream = SimUpdateStream()
    late_value = {"translate": (9.0, 9.0, 9.0), "orient": (0.0, 0.0, 0.0, 1.0)}
    late_intent = _edit_intent(DataAuthority.SIM, "/World/Cube_Late", late_value)

    class _QueueDuringApplyController(_Controller):
        queued_late = False

        def apply_initial_condition_values(
            self,
            poses: Sequence[BodyPose],
            *,
            reset: bool = False,
        ) -> OvphysxStageResult:
            if not self.queued_late:
                self.queued_late = True
                stream.queue(late_intent)
            return super().apply_initial_condition_values(poses, reset=reset)

    controller = _QueueDuringApplyController()
    stream.queue(_edit_intent(
        DataAuthority.SIM,
        "/World/Cube_A",
        {"translate": (2.0, 7.0, 4.0), "orient": (0.0, 0.0, 0.0, 1.0)},
    ))

    first = stream.apply_pending(controller)
    assert first["sim_value_paths"] == ["/World/Cube_A"]
    assert stream.has_pending is True

    second = stream.apply_pending(controller)
    assert second["sim_value_paths"] == ["/World/Cube_Late"]
    assert stream.has_pending is False
    assert [
        [pose.prim_path for pose in batch] for batch in controller.pose_updates
    ] == [["/World/Cube_A"], ["/World/Cube_Late"]]


def test_sim_update_stream_wake_hook_fires_after_queue() -> None:
    # Edit-submission wake source (task02-07): a sim edit queued from the
    # main thread must wake the parked render loop so the next tick applies
    # the pending initial-condition value.
    stream = SimUpdateStream()
    wakes: list[int] = []
    stream.set_wake_hook(lambda: wakes.append(1))

    stream.queue(_edit_intent(
        DataAuthority.SIM,
        "/World/Cube_A",
        {"translate": (2.0, 7.0, 4.0), "orient": (0.0, 0.0, 0.0, 1.0)},
    ))
    assert wakes == [1]

    stream.set_wake_hook(None)
    stream.queue(_edit_intent(
        DataAuthority.SIM,
        "/World/Cube_A",
        {"translate": (3.0, 7.0, 4.0), "orient": (0.0, 0.0, 0.0, 1.0)},
    ))
    assert wakes == [1]


def test_view_update_stream_rejects_sim_authority() -> None:
    stream = ViewUpdateStream()
    intent = _edit_intent(
        DataAuthority.SIM,
        "/World/Cube_A",
        {"translate": (0.0, 0.0, 0.0), "orient": (0.0, 0.0, 0.0, 1.0)},
    )

    assert stream.supports(intent) is False

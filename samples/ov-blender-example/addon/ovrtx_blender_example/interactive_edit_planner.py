# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Interactive Blender edit planning for ADR 0009."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from . import topology_edit_fallback


class EditShape(str, Enum):
    VALUE = "value"
    TOPOLOGY = "topology"


class DataAuthority(str, Enum):
    VIEW = "view"
    SIM = "sim"


class EditMechanism(str, Enum):
    UPDATE = "update"
    COMPOSE = "compose"
    NONE = "none"


class EditPersistence(str, Enum):
    WRITE = "write"
    NONE = "none"


class EditStatus(str, Enum):
    APPLIED = "applied"
    QUEUED = "queued"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"


_SIM_INITIAL_CONDITION_ATTRIBUTES = frozenset(
    {
        "omni:xform",
        "xformOp:transform",
    }
)
_SIM_VELOCITY_ATTRIBUTES = frozenset({"physics:velocity", "physics:angularVelocity"})

#: Provenance ``source`` marking a live RTPT render-setting value edit
#: (render-quality-color-controls task01-04). Kept in sync with the same
#: constant in ``view_update_stream`` and ``rtpt_live_change``.
RENDER_SETTING_VALUE_SOURCE = "rtpt_render_setting"


def edit_location(
    *,
    blender_property_path: str = "",
    usd_prim_path: str = "",
    usd_attribute: str = "",
    usd_property_path: str = "",
    usd_layer_id: str = "",
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return constructor fields for a directly located interactive edit."""

    return {
        "blender_property_path": str(blender_property_path),
        "usd_prim_path": str(usd_prim_path),
        "usd_property_path": str(
            usd_property_path
            or (f"{usd_prim_path}.{usd_attribute}" if usd_prim_path and usd_attribute else "")
        ),
        "usd_layer_id": str(usd_layer_id),
        "provenance": dict(provenance or {}),
    }


class _EditLocation:
    blender_property_path: str
    usd_prim_path: str = ""
    usd_property_path: str = ""
    usd_layer_id: str = ""
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def has_edit_identity(self) -> bool:
        return bool(self.usd_prim_path)

    @property
    def usd_attribute(self) -> str:
        prefix = f"{self.usd_prim_path}."
        if self.usd_property_path.startswith(prefix):
            return self.usd_property_path[len(prefix):]
        return ""

    def has_write_target(self) -> bool:
        return bool(self.usd_layer_id)

    def has_persistence_identity(self) -> bool:
        return bool(self.usd_layer_id and self.usd_prim_path)


@dataclass(frozen=True)
class InteractiveEdit(_EditLocation):
    shape: EditShape
    data_authority: DataAuthority
    blender_property_path: str = ""
    usd_prim_path: str = ""
    usd_property_path: str = ""
    usd_layer_id: str = ""
    provenance: Mapping[str, Any] = field(default_factory=dict)
    value: Any = None
    previous_value: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EditPlanImpact:
    update_requested: bool = False
    write_requested: bool = False
    whole_scene_export_requested: bool = False
    whole_scene_export_avoided: bool = True
    render_composition_identity_required: bool = False
    render_session_reuse_expected: bool | None = None
    physics_generation_reset_expected: bool = False
    target_identity_preserved: bool = False
    provenance_preserved: bool = False
    topology_reasons: tuple[str, ...] = ()
    update_stream_rejected: bool = False
    session_rekey_expected: bool = False
    refinement_reset_expected: bool = False
    scene_generation_replacement_requested: bool = False
    # Pre-rename alias kept for the authoring-session route (blender-live-
    # render): a COMPOSE plan asks the authoring session to reconcile the
    # authored generation. Set together with
    # ``scene_generation_replacement_requested``.
    authoring_reconciliation_requested: bool = False


@dataclass(frozen=True)
class EditIntent(_EditLocation):
    shape: EditShape
    data_authority: DataAuthority
    blender_property_path: str = ""
    usd_prim_path: str = ""
    usd_property_path: str = ""
    usd_layer_id: str = ""
    provenance: Mapping[str, Any] = field(default_factory=dict)
    value: Any = None
    impact: EditPlanImpact = field(default_factory=EditPlanImpact)


@dataclass(frozen=True)
class EditPlan:
    edit: InteractiveEdit
    mechanism: EditMechanism
    persistence: EditPersistence
    reason: str
    usd_layer_id: str = ""
    unsupported_reason: str = ""
    impact: EditPlanImpact = field(default_factory=EditPlanImpact)

    @property
    def shape(self) -> EditShape:
        return self.edit.shape

    @property
    def data_authority(self) -> DataAuthority:
        return self.edit.data_authority

    def to_intent(self) -> EditIntent:
        if self.mechanism != EditMechanism.UPDATE:
            raise ValueError(f"edit plan is not an update: {self.mechanism.value}")
        return EditIntent(
            shape=self.edit.shape,
            data_authority=self.edit.data_authority,
            blender_property_path=self.edit.blender_property_path,
            usd_prim_path=self.edit.usd_prim_path,
            usd_property_path=self.edit.usd_property_path,
            usd_layer_id=self.edit.usd_layer_id,
            provenance=self.edit.provenance,
            value=self.edit.value,
            impact=self.impact,
        )


class InteractiveEditPlanner:
    """Routes Blender edit intent to mechanism and persistence decisions."""

    def plan(self, edit: InteractiveEdit) -> EditPlan:
        if reason := _metadata_unsupported_reason(edit.metadata):
            return self._unsupported(edit, reason)

        if edit.shape == EditShape.TOPOLOGY:
            return self._plan_topology_edit(edit)

        if edit.shape != EditShape.VALUE:
            return self._unsupported(edit, "unsupported_edit_shape")

        value_update_kind = _value_update_kind(edit)
        if value_update_kind is None:
            if edit.has_write_target():
                return self._write(edit, "update_unavailable")
            return self._unsupported(edit, "value_update_unresolved")

        if not edit.has_write_target():
            if _supported_live_value_update_kind(value_update_kind):
                if not edit.has_edit_identity():
                    return self._unsupported(edit, "missing_edit_identity")
                return self._update(edit)
            return self._unsupported(edit, "missing_write_target")

        if _supported_live_value_update_kind(value_update_kind) and edit.has_edit_identity():
            return self._update(edit)
        if value_update_kind == "uv":
            if not edit.has_edit_identity():
                return self._unsupported(edit, "missing_edit_identity")
            return self._unsupported(edit, "update_unavailable")

        return self._write(edit, "update_unavailable")

    def _plan_topology_edit(self, edit: InteractiveEdit) -> EditPlan:
        topology_reasons = topology_edit_fallback.topology_reasons_for_edit(
            _topology_default_kind(edit),
            edit.metadata,
        )
        if not edit.has_write_target():
            if not edit.has_edit_identity():
                return self._unsupported(
                    edit,
                    "missing_edit_identity",
                    topology_reasons=topology_reasons,
                )
            return self._reconcile_authoring_source(
                edit,
                topology_reasons=topology_reasons,
            )
        return self._compose_write(
            edit,
            "topology_edit_requires_compose_write",
            topology_reasons=topology_reasons,
        )

    def _update(self, edit: InteractiveEdit) -> EditPlan:
        physics_generation_reset_expected = _value_update_kind(edit) == "initial_condition"
        persistence_ready = _persistence_ready(edit)
        return EditPlan(
            edit=edit,
            mechanism=EditMechanism.UPDATE,
            persistence=EditPersistence.WRITE
            if persistence_ready
            else EditPersistence.NONE,
            reason="update",
            usd_layer_id=edit.usd_layer_id,
            impact=EditPlanImpact(
                update_requested=True,
                write_requested=persistence_ready,
                whole_scene_export_requested=False,
                whole_scene_export_avoided=True,
                render_composition_identity_required=True,
                render_session_reuse_expected=True,
                physics_generation_reset_expected=physics_generation_reset_expected,
                target_identity_preserved=True,
                provenance_preserved=bool(edit.usd_layer_id or edit.provenance),
            ),
        )

    def _write(self, edit: InteractiveEdit, reason: str) -> EditPlan:
        if not _persistence_ready(edit):
            return self._unsupported(edit, "missing_persistence_identity")
        return EditPlan(
            edit=edit,
            mechanism=EditMechanism.NONE,
            persistence=EditPersistence.WRITE,
            reason=reason,
            usd_layer_id=edit.usd_layer_id,
            impact=EditPlanImpact(
                write_requested=True,
                whole_scene_export_requested=False,
                whole_scene_export_avoided=True,
                target_identity_preserved=_persistence_ready(edit),
                provenance_preserved=bool(edit.usd_layer_id or edit.provenance),
            ),
        )

    def _compose_write(
        self,
        edit: InteractiveEdit,
        reason: str,
        *,
        topology_reasons: tuple[str, ...] = (),
    ) -> EditPlan:
        if not _persistence_ready(edit):
            return self._unsupported(
                edit,
                "missing_persistence_identity",
                topology_reasons=topology_reasons,
            )
        return EditPlan(
            edit=edit,
            mechanism=EditMechanism.COMPOSE,
            persistence=EditPersistence.WRITE,
            reason=reason,
            usd_layer_id=edit.usd_layer_id,
            impact=EditPlanImpact(
                write_requested=True,
                whole_scene_export_requested=False,
                whole_scene_export_avoided=True,
                render_composition_identity_required=True,
                render_session_reuse_expected=False,
                target_identity_preserved=_persistence_ready(edit),
                provenance_preserved=bool(edit.usd_layer_id or edit.provenance),
                topology_reasons=topology_reasons,
                update_stream_rejected=True,
                session_rekey_expected=True,
                refinement_reset_expected=True,
            ),
        )

    def _reconcile_authoring_source(
        self,
        edit: InteractiveEdit,
        *,
        topology_reasons: tuple[str, ...] = (),
    ) -> EditPlan:
        full_scene_export = bool(
            {
                topology_edit_fallback.WORLD_ASSIGNMENT_CHANGED,
                topology_edit_fallback.WORLD_NODE_GRAPH_CHANGED,
                topology_edit_fallback.ENVIRONMENT_TEXTURE_CHANGED,
            }.intersection(topology_reasons)
        )
        return EditPlan(
            edit=edit,
            mechanism=EditMechanism.COMPOSE,
            persistence=EditPersistence.NONE,
            reason="scene_generation_replacement",
            impact=EditPlanImpact(
                whole_scene_export_requested=full_scene_export,
                whole_scene_export_avoided=not full_scene_export,
                render_composition_identity_required=True,
                render_session_reuse_expected=False,
                physics_generation_reset_expected=False,
                target_identity_preserved=True,
                provenance_preserved=bool(edit.provenance),
                topology_reasons=topology_reasons,
                update_stream_rejected=True,
                session_rekey_expected=True,
                refinement_reset_expected=True,
                scene_generation_replacement_requested=True,
                authoring_reconciliation_requested=True,
            ),
        )

    def _unsupported(
        self,
        edit: InteractiveEdit,
        reason: str,
        *,
        topology_reasons: tuple[str, ...] = (),
    ) -> EditPlan:
        topology = bool(topology_reasons)
        return EditPlan(
            edit=edit,
            mechanism=EditMechanism.NONE,
            persistence=EditPersistence.NONE,
            reason="unsupported",
            unsupported_reason=reason,
            usd_layer_id=edit.usd_layer_id,
            impact=EditPlanImpact(
                whole_scene_export_requested=False,
                whole_scene_export_avoided=True,
                target_identity_preserved=edit.has_edit_identity(),
                provenance_preserved=bool(edit.usd_layer_id or edit.provenance),
                topology_reasons=topology_reasons,
                update_stream_rejected=topology,
                session_rekey_expected=False,
                refinement_reset_expected=False,
            ),
        )


def _persistence_ready(edit: InteractiveEdit) -> bool:
    return _value_update_kind(edit) != "uv" and edit.has_persistence_identity()


def _value_update_kind(edit: InteractiveEdit) -> str | None:
    target = edit
    attribute = target.usd_attribute
    blender_property_path = target.blender_property_path
    if edit.data_authority == DataAuthority.SIM:
        if attribute in _SIM_VELOCITY_ATTRIBUTES:
            return "body_velocity"
        if attribute in _SIM_INITIAL_CONDITION_ATTRIBUTES or blender_property_path == "matrix_world":
            return "initial_condition"
        return "physics_property"
    if _is_render_setting_value(edit):
        return "render_setting"
    if _is_viewport_camera_value(edit):
        return "camera"
    if _is_viewport_camera_projection_value(edit):
        return "camera_value"
    if attribute in {"omni:xform", "xformOp:transform"} or blender_property_path == "matrix_world":
        return "transform"
    if attribute == "primvars:st" or blender_property_path == "uv_layers.active":
        return "uv"
    if blender_property_path == "world_dome" or "world_dome_conversion" in target.provenance:
        return "world"
    if "light_path" in target.provenance:
        return "light"
    if "material_path" in target.provenance:
        return "material"
    if blender_property_path in {
        "energy",
        "data.energy",
        "color",
        "data.color",
        "normalize",
        "exposure",
        "size",
        "size_y",
        "spot_size",
        "spot_blend",
        "data.type",
        "data.shape",
    }:
        return "light"
    if (
        attribute.startswith("inputs:")
        or blender_property_path.startswith("material.")
        or blender_property_path.startswith("principled:")
        or blender_property_path in {
        "diffuse_color",
        "roughness",
        "metallic",
        "alpha",
        "base_color",
        }
    ):
        return "material"
    return None


def _is_render_setting_value(edit: InteractiveEdit) -> bool:
    """Live RTPT render-setting attribute value edit (task01-04).

    Targets the active ``RenderProduct`` prim's RTPT quality attributes as
    runtime attribute value updates; distinguished from the other value
    lanes by its provenance ``source`` marker.
    """

    return str(edit.provenance.get("source", "")) == RENDER_SETTING_VALUE_SOURCE


def _is_viewport_camera_value(edit: InteractiveEdit) -> bool:
    target = edit
    return (
        target.blender_property_path == "viewport_camera_matrix"
        or str(target.provenance.get("source", "")) == "viewport_camera"
    )


def _is_viewport_camera_projection_value(edit: InteractiveEdit) -> bool:
    """Camera projection/framing attribute value edit (task04-05).

    Distinct from the ``viewport_camera`` pose lane: these target the
    composed camera's projection attributes (``focalLength``, apertures,
    ``clippingRange``) as attribute value updates, not ``omni:xform``.
    """

    return (
        str(edit.provenance.get("source", "")) == "viewport_camera_projection"
    )


def _supported_live_value_update_kind(kind: str | None) -> bool:
    return kind in {
        "camera",
        "camera_value",
        "render_setting",
        "transform",
        "initial_condition",
        "body_velocity",
        "material",
        "light",
        "world",
        "uv",
    }


def _topology_default_kind(edit: InteractiveEdit) -> str:
    raw = edit.metadata.get("topology_default", "")
    if raw:
        return str(raw)
    property_name = edit.blender_property_path
    if property_name in {"node_tree", "material_slots"}:
        return "material_topology"
    if property_name.startswith("validation.collider") or property_name.startswith("collider"):
        return "collider_topology"
    return "scene_topology"


def _metadata_unsupported_reason(metadata: Mapping[str, Any]) -> str:
    reason = metadata.get("unsupported_reason", "")
    return str(reason).strip()

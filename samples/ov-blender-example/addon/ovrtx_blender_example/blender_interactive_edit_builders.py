# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Builders from stock Blender data to interactive edit planner inputs."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable, Mapping

from .interactive_edit_planner import DataAuthority, EditShape, InteractiveEdit, edit_location
from . import usd_paths
from . import write_target_resolution
from . import material_usd_prim
from . import light_value_conversion as light_conversion
from . import light_usd_prim
from . import world_dome_conversion
from . import world_dome_usd_prim
from . import uv_usd_prim
from . import topology_edit_fallback
from . import usd_value_edit_support
from .usd_prim_resolver import UsdPrimResolver
from .value_edit_conversion import (
    CLASSIFICATION_TOPOLOGY,
    CLASSIFICATION_UNSUPPORTED,
    STATUS_TOPOLOGY,
    ValueEditConversionPolicies,
    default_value_edit_conversion_policies,
    normalized_classification,
)


USD_ATTRIBUTE_PROP = "ovrtx.usd_attribute"
BODY_VISUAL_OFFSET_MATRIX_PROP = "ovrtx.body_visual_offset_matrix"


def edit_location_from_blender_id(
    id_data: Any,
    *,
    usd_layer_id: str | None = None,
    usd_prim_path: str | None = None,
    usd_attribute: str | None = None,
    usd_property_path: str | None = None,
    blender_property_path: str | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build direct edit location fields from a Blender ID or Blender-like object."""

    resolved_layer = _string_value(
        usd_layer_id
        if usd_layer_id is not None
        else usd_paths.id_property(id_data, usd_paths.USD_LAYER_ID_PROP, "")
    )
    resolved_usd_prim_path = _string_value(
        usd_prim_path
        if usd_prim_path is not None
        else usd_paths.id_property(id_data, usd_paths.USD_PRIM_PATH_PROP, "")
    )
    if not resolved_usd_prim_path:
        resolved_usd_prim_path = usd_paths.source_usd_path_from_blender_id(id_data)
    resolved_usd_attribute = _string_value(
        usd_attribute
        if usd_attribute is not None
        else usd_paths.id_property(id_data, USD_ATTRIBUTE_PROP, "")
    )
    resolved_usd_property_path = _string_value(
        usd_property_path
        if usd_property_path is not None
        else usd_paths.id_property(id_data, usd_paths.USD_PROPERTY_PATH_PROP, "")
    )
    resolved_blender_property_path = _string_value(
        blender_property_path
        if blender_property_path is not None
        else usd_paths.id_property(id_data, usd_paths.BLENDER_PROPERTY_PATH_PROP, "")
    )
    if not resolved_usd_property_path and resolved_usd_prim_path and resolved_usd_attribute:
        resolved_usd_property_path = f"{resolved_usd_prim_path}.{resolved_usd_attribute}"
    return {
        "blender_property_path": resolved_blender_property_path,
        "usd_prim_path": resolved_usd_prim_path,
        "usd_property_path": resolved_usd_property_path,
        "usd_layer_id": resolved_layer,
        "provenance": dict(provenance or _provenance_from_blender_id(id_data)),
    }


def object_transform_edit(
    obj: Any,
    *,
    data_authority: DataAuthority | str | None = None,
    location: Mapping[str, Any] | None = None,
) -> InteractiveEdit:
    raw_authority = (
        data_authority
        if data_authority is not None
        else usd_paths.id_property(obj, usd_paths.DATA_AUTHORITY_PROP, "")
    )
    if raw_authority in {None, ""}:
        raise ValueError(f"{usd_paths.DATA_AUTHORITY_PROP} is required")
    authority = _parse_data_authority(raw_authority)
    return InteractiveEdit(
        shape=EditShape.VALUE,
        data_authority=authority,
        **dict(location or edit_location_from_blender_id(
            obj, usd_attribute="xformOp:transform", blender_property_path="matrix_world"
        )),
        value=_matrix_rows(_runtime_body_matrix_world(obj)),
        metadata={},
    )


def material_value_edit(
    material: Any,
    *,
    property_name: str = "diffuse_color",
    usd_attribute: str = "inputs:diffuseColor",
    value: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
    location: Mapping[str, Any] | None = None,
) -> InteractiveEdit:
    return _value_edit(
        DataAuthority.VIEW,
        material,
        property_name=property_name,
        usd_attribute=usd_attribute,
        value=value,
        metadata=metadata,
        location=location,
    )


def material_value_edit_from_prim(
    material: Any,
    prim: material_usd_prim.MaterialUsdPrim,
    *,
    property_name: str = material_usd_prim.DEFAULT_BLENDER_MATERIAL_PROPERTY,
    value: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
    usd_layer_id: str | None = None,
) -> InteractiveEdit:
    resolved_layer = _usd_layer_id(material, override=usd_layer_id)
    target = edit_location(
        usd_prim_path=prim.prim_path,
        usd_attribute=prim.usd_attribute,
        usd_layer_id=resolved_layer,
        blender_property_path=property_name,
        provenance={
            **_provenance_from_blender_id(material),
            "material_path": prim.material_prim_path,
        },
    )
    return material_value_edit(
        material,
        property_name=property_name,
        usd_attribute=prim.usd_attribute,
        value=value,
        metadata=metadata,
        location=target,
    )


def material_value_edits_from_resolver(
    material: Any,
    resolver: UsdPrimResolver,
    *,
    value_edit_conversion_policies: ValueEditConversionPolicies | None = None,
    usd_layer_id: str | None = None,
) -> list[InteractiveEdit]:
    policies = _value_edit_conversion_policies(value_edit_conversion_policies)
    topology_edit = _material_graph_topology_edit(
        material,
        resolver,
        policies=policies,
        usd_layer_id=usd_layer_id,
    )
    if topology_edit is not None:
        return [topology_edit]
    edits: list[InteractiveEdit] = []
    for attribute in policies.material.usd_attribute_values(material):
        resolution = resolver.resolve_material(
            material,
            usd_attribute=attribute.name,
            property_name=attribute.blender_property_path,
        )
        if resolution.value is None:
            continue
        edit = material_value_edit_from_prim(
            material,
            resolution.value,
            property_name=attribute.blender_property_path,
            value=attribute.value,
            metadata=dict(attribute.metadata),
            usd_layer_id=usd_layer_id,
        )
        match_source = str(resolution.diagnostics.get("match_source", ""))
        if match_source:
            edit = replace(
                edit,
                provenance={**dict(edit.provenance), "match_source": match_source},
                metadata=dict(edit.metadata),
            )
        edits.append(edit)
    edits.extend(
        _texture_connected_classification_edits(
            material,
            resolver,
            policies=policies,
            usd_layer_id=usd_layer_id,
        )
    )
    return edits


# Reverse of the converter's texture-wiring table: authored value input
# attribute -> Principled socket name, for naming the field in
# classification records (task04-07).
_TEXTURE_INPUT_SOCKETS = {
    "inputs:" + input_name: socket_name
    for socket_name, input_name in (
        usd_value_edit_support.PRINCIPLED_PREVIEW_SURFACE_TEXTURE_INPUTS.items()
    )
}


def _texture_connected_classification_edits(
    material: Any,
    resolver: UsdPrimResolver,
    *,
    policies: ValueEditConversionPolicies,
    usd_layer_id: str | None,
) -> list[InteractiveEdit]:
    """Classification records for texture-connected value sockets (task04-07).

    A value change on a texture-connected Principled socket has no rendered
    effect (the connection wins) and emits no value edit — a silent skip
    since task04-02. The depsgraph carries no field granularity, so the
    once-per-key report originates from classification: whenever the
    material lane runs and a texture-wirable value socket is connected,
    emit one report-only edit per connected socket. The
    ``unsupported_reason`` metadata routes the planner to an UNSUPPORTED
    plan (mechanism NONE): no RPC, no refinement reset, no thread wake.
    The workflow writes the record every time and reports once per
    (target, field) per session.
    """

    link_states_fn = getattr(policies.material, "texture_wired_input_links", None)
    classify = getattr(policies.material, "classify_field", None)
    if not callable(link_states_fn) or not callable(classify):
        return []
    link_states = link_states_fn(material) or {}
    edits: list[InteractiveEdit] = []
    for usd_attribute, blender_linked in link_states.items():
        if not blender_linked:
            continue
        socket_name = _TEXTURE_INPUT_SOCKETS.get(usd_attribute, "")
        field_name = f"principled:{socket_name}" if socket_name else usd_attribute
        classification = classify(material, field_name)
        status = _string_value(getattr(classification, "status", ""))
        if normalized_classification(status) != CLASSIFICATION_UNSUPPORTED:
            continue
        reason = _string_value(getattr(classification, "reason", "")) or (
            "unsupported_material_texture_connected_input"
        )
        resolution = resolver.resolve_material(
            material,
            usd_attribute=usd_attribute,
            property_name=field_name,
        )
        prim = resolution.value
        if prim is None:
            # Fail-closed resolution (e.g. unscanned authoring identity)
            # builds no edits at all — including report-only records; a
            # scene-mismatch failure must not masquerade as an
            # unsupported-field classification (04-02 contract).
            continue
        location = edit_location(
            usd_prim_path=prim.prim_path,
            usd_attribute=usd_attribute,
            usd_layer_id=_usd_layer_id(material, override=usd_layer_id),
            blender_property_path=field_name,
            provenance={
                **_provenance_from_blender_id(material),
                "classification_origin": "material_texture_connected_input",
            },
        )
        edits.append(
            InteractiveEdit(
                shape=EditShape.VALUE,
                data_authority=DataAuthority.VIEW,
                **dict(location),
                value=None,
                metadata={
                    "unsupported_reason": reason,
                    "classification": CLASSIFICATION_UNSUPPORTED,
                    "classification_status": status,
                },
            )
        )
    return edits


def material_graph_topology_edit_from_resolver(
    material: Any,
    resolver: UsdPrimResolver,
    *,
    usd_layer_id: str | None = None,
) -> InteractiveEdit | None:
    material_path = usd_paths.source_usd_path_from_blender_id(material)
    if not material_path:
        resolution = resolver.resolve_material(
            material,
            usd_attribute=material_usd_prim.DEFAULT_USD_MATERIAL_ATTRIBUTE,
            property_name="node_tree",
        )
        if resolution.value is not None:
            material_path = resolution.value.material_prim_path
    if not material_path:
        return None
    return InteractiveEdit(
        shape=EditShape.TOPOLOGY,
        data_authority=DataAuthority.VIEW,
        **edit_location(
            usd_prim_path=material_path,
            usd_attribute="",
            usd_layer_id=_usd_layer_id(material, override=usd_layer_id),
            blender_property_path="node_tree",
            provenance={
                **_provenance_from_blender_id(material),
                "material_path": material_path,
            },
        ),
        value=_string_value(getattr(material, "name_full", getattr(material, "name", ""))),
        metadata={
            "topology_default": "material_topology",
            "topology_change_kinds": ("material_graph",),
            "edit_support_concept": "material.graph",
        },
    )


def _material_graph_topology_edit(
    material: Any,
    resolver: UsdPrimResolver,
    *,
    policies: ValueEditConversionPolicies,
    usd_layer_id: str | None,
) -> InteractiveEdit | None:
    """Detect authored-vs-Blender texture wiring divergence (task04-02).

    Connecting or disconnecting a texture on a converter-wired Principled
    socket rewires the shader graph: that change is TOPOLOGY
    (``material_graph``) and takes the authored-generation route, never a
    value update. Detection compares each texture-wirable value input's
    Blender link state against the authored shader's connection state and
    applies only to authoring-identity-resolved materials (the converter
    contract); direct-USD stages keep the value-edit behavior.
    """

    link_states_fn = getattr(policies.material, "texture_wired_input_links", None)
    if not callable(link_states_fn):
        return None
    link_states = link_states_fn(material)
    if not link_states:
        return None
    diverged: dict[str, dict[str, bool]] = {}
    resolved_prim: material_usd_prim.MaterialUsdPrim | None = None
    for usd_attribute, blender_linked in link_states.items():
        resolution = resolver.resolve_material(
            material,
            usd_attribute=usd_attribute,
            property_name="node_tree",
        )
        prim = resolution.value
        if prim is None:
            continue
        if str(resolution.diagnostics.get("match_source", "")) != material_usd_prim.MATCH_AUTHORING_PRIM_PATH:
            return None
        resolved_prim = resolved_prim or prim
        if bool(prim.connected) != bool(blender_linked):
            diverged[usd_attribute] = {
                "blender_linked": bool(blender_linked),
                "authored_connected": bool(prim.connected),
            }
    if not diverged or resolved_prim is None:
        return None
    resolved_layer = _usd_layer_id(material, override=usd_layer_id)
    location = edit_location(
        usd_prim_path=resolved_prim.material_prim_path,
        usd_layer_id=resolved_layer,
        blender_property_path="node_tree",
        provenance={
            **_provenance_from_blender_id(material),
            "material_path": resolved_prim.material_prim_path,
            "match_source": material_usd_prim.MATCH_AUTHORING_PRIM_PATH,
        },
    )
    diverged_attributes = tuple(sorted(diverged))
    return InteractiveEdit(
        shape=EditShape.TOPOLOGY,
        data_authority=DataAuthority.VIEW,
        **dict(location),
        value=diverged_attributes,
        previous_value=(),
        metadata={
            "topology_change_kinds": ("material_graph",),
            "diverged_texture_inputs": diverged_attributes,
            "texture_link_divergence": diverged,
        },
    )


def light_value_edit(
    light: Any,
    *,
    property_name: str = "energy",
    usd_attribute: str = "inputs:intensity",
    value: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
    location: Mapping[str, Any] | None = None,
) -> InteractiveEdit:
    return _value_edit(
        DataAuthority.VIEW,
        light,
        property_name=property_name,
        usd_attribute=usd_attribute,
        value=value,
        metadata=metadata,
        location=location,
    )


def light_value_edits_from_prim(
    light_object: Any,
    prim: light_usd_prim.LightUsdPrim,
    *,
    value_edit_conversion_policies: ValueEditConversionPolicies | None = None,
    usd_layer_id: str | None = None,
    match_source: str = "",
) -> list[InteractiveEdit]:
    policies = _value_edit_conversion_policies(value_edit_conversion_policies)
    resolved_layer = _usd_layer_id(light_object, override=usd_layer_id)
    provenance = {
        **_provenance_from_blender_id(light_object),
        "light_path": prim.prim_path,
        "usd_family": prim.usd_family,
        "previous_authored_light_form": prim.authored_light_form,
    }
    if _string_value(match_source):
        provenance["match_source"] = _string_value(match_source)
    current_form = _current_authored_light_form(light_object)
    if not provenance["previous_authored_light_form"] or not current_form:
        return [
            _unsupported_light_form_edit(
                location=edit_location(
                    usd_prim_path=prim.prim_path,
                    usd_layer_id=resolved_layer,
                    blender_property_path="data.type",
                    provenance=provenance,
                ),
                current_authored_light_form=current_form,
            )
        ]
    topology_edit = _light_form_topology_edit(
        light_object,
        location=edit_location(
            usd_prim_path=prim.prim_path,
            usd_layer_id=resolved_layer,
            blender_property_path=_light_form_blender_property_path(provenance["previous_authored_light_form"], current_form),
            provenance=provenance,
        ),
        previous_authored_light_form=provenance["previous_authored_light_form"],
        current_authored_light_form=current_form,
        previous_usd_family=provenance["usd_family"],
    )
    if topology_edit is not None:
        return [topology_edit]
    edits: list[InteractiveEdit] = []
    for attribute in policies.light.usd_attribute_values(light_object):
        target = edit_location(
            usd_prim_path=prim.prim_path,
            usd_attribute=attribute.name,
            usd_layer_id=resolved_layer,
            blender_property_path=attribute.blender_property_path,
            provenance=provenance,
        )
        edits.append(
            light_value_edit(
                light_object,
                property_name=attribute.blender_property_path,
                usd_attribute=attribute.name,
                value=attribute.value,
                metadata=dict(attribute.metadata),
                location=target,
            )
        )
    return edits


def world_value_edit(
    world: Any,
    *,
    property_name: str = "world_dome",
    usd_attribute: str = "inputs:intensity",
    value: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
    location: Mapping[str, Any] | None = None,
) -> InteractiveEdit:
    return _value_edit(
        DataAuthority.VIEW,
        world,
        property_name=property_name,
        usd_attribute=usd_attribute,
        value=value,
        metadata=metadata,
        location=location,
    )


def world_value_edits_from_prim(
    world: Any,
    prim: world_dome_usd_prim.WorldDomeUsdPrim,
    *,
    value_edit_conversion_policies: ValueEditConversionPolicies | None = None,
    usd_layer_id: str | None = None,
) -> list[InteractiveEdit]:
    policies = _value_edit_conversion_policies(value_edit_conversion_policies)
    resolved_layer = _usd_layer_id(world, override=usd_layer_id)
    topology_edit = _world_topology_edit(
        world,
        prim,
        policies=policies,
        usd_layer_id=resolved_layer,
    )
    if topology_edit is not None:
        return [topology_edit]
    attributes = policies.world.usd_attribute_values(world)
    if not attributes:
        return []
    conversion_diagnostics = {
        "status": "supported_value",
        "attributes": [attribute.name for attribute in attributes],
        **dict(attributes[0].metadata),
    }
    provenance = {
        **_provenance_from_blender_id(world),
        "dome_owner_path": prim.prim_path,
        "usd_family": prim.usd_family,
        "world_dome_conversion": conversion_diagnostics,
    }
    edits: list[InteractiveEdit] = []
    for attribute in attributes:
        target = edit_location(
            usd_prim_path=prim.prim_path,
            usd_attribute=attribute.name,
            usd_layer_id=resolved_layer,
            blender_property_path=attribute.blender_property_path,
            provenance=provenance,
        )
        edits.append(
            world_value_edit(
                world,
                property_name=attribute.blender_property_path,
                usd_attribute=attribute.name,
                value=attribute.value,
                metadata=dict(attribute.metadata),
                location=target,
            )
        )
    return edits


def _world_topology_edit(
    world: Any,
    prim: world_dome_usd_prim.WorldDomeUsdPrim,
    *,
    policies: ValueEditConversionPolicies,
    usd_layer_id: str,
) -> InteractiveEdit | None:
    """Node-based worlds are topology for the live-edit route (task04-04).

    The value-editable world is the solid-color world only (flat
    ``World.color`` or one bare Background node's color/strength). Any
    node-based world — environment texture, sky texture, mixed graphs —
    changes what the dome represents: TOPOLOGY, generation route, with the
    policy's reason recorded on the edit.
    """

    spec_fn = getattr(policies.world, "world_dome_spec", None)
    if not callable(spec_fn):
        return None
    spec = spec_fn(world)
    if _string_value(getattr(spec, "status", "")) != STATUS_TOPOLOGY:
        return None
    reason = _string_value(getattr(spec, "reason", ""))
    change_kind = (
        "environment_texture"
        if reason == world_dome_conversion.ENVIRONMENT_TEXTURE_CHANGED
        else "world_node_graph"
    )
    location = edit_location(
        usd_prim_path=prim.prim_path,
        usd_layer_id=usd_layer_id,
        blender_property_path="node_tree",
        provenance={
            **_provenance_from_blender_id(world),
            "dome_owner_path": prim.prim_path,
            "usd_family": prim.usd_family,
            "world_dome_conversion": {
                "status": _string_value(getattr(spec, "status", "")),
                "reason": reason,
            },
        },
    )
    return InteractiveEdit(
        shape=EditShape.TOPOLOGY,
        data_authority=DataAuthority.VIEW,
        **dict(location),
        value=reason,
        previous_value="",
        metadata={
            "topology_change_kinds": (change_kind,),
            "world_topology_reason": reason,
        },
    )


def _world_presence_topology_edit(
    source_id: Any,
    *,
    dome_prim_path: str,
    usd_family: str,
    world_present: bool,
    usd_layer_id: str | None = None,
) -> InteractiveEdit:
    """World added/removed where the authored generation disagrees.

    Adding a world where none was authored (or removing the world while
    the authored generation still carries the dome) changes which prims
    exist: TOPOLOGY (``world_assignment`` →
    ``world_datablock_assignment_is_topology``), generation route, never a
    value update (task04-04 clarification). Built only on presence
    divergence, so the authored dome presence is the inverse of the
    Blender-side world presence.
    """

    provenance = {
        **_provenance_from_blender_id(source_id),
        "dome_owner_path": dome_prim_path,
    }
    if usd_family:
        provenance["usd_family"] = usd_family
    location = edit_location(
        usd_prim_path=dome_prim_path,
        usd_layer_id=_usd_layer_id(source_id, override=usd_layer_id),
        blender_property_path="world",
        provenance=provenance,
    )
    return InteractiveEdit(
        shape=EditShape.TOPOLOGY,
        data_authority=DataAuthority.VIEW,
        **dict(location),
        value="world" if world_present else "none",
        previous_value="none" if world_present else "world_dome",
        metadata={
            "topology_change_kinds": ("world_assignment",),
            "world_present": bool(world_present),
            "authored_dome_present": not world_present,
        },
    )


def _world_edits_for_update(
    id_data: Any,
    resolver: UsdPrimResolver,
    policies: ValueEditConversionPolicies,
) -> list[InteractiveEdit]:
    """Build the world lane's edits for one depsgraph update (task04-04).

    World datablock updates carry value edits (solid-color worlds), a
    node-graph topology edit, or a presence edit when no dome was authored.
    Generic Scene updates are presentation noise here; scene-generation
    identity tracking owns World assignment changes. Dome resolution failures
    other than the missing prim (stage unavailable, wrong prim type, missing
    attributes) fail closed.
    """

    if not _is_world_id(id_data):
        return []
    world = id_data
    resolution = resolver.resolve_world_dome()
    prim = resolution.value
    if prim is not None:
        return world_value_edits_from_prim(
            world,
            prim,
            value_edit_conversion_policies=policies,
        )
    if (
        world is not None
        and str(resolution.error_reason) == world_dome_usd_prim.ERROR_NO_DOME_PRIM
    ):
        dome_path = (
            _string_value(resolution.diagnostics.get("prim_path", ""))
            or world_dome_conversion.DEFAULT_DOME_OWNER_PATH
        )
        return [
            _world_presence_topology_edit(
                id_data,
                dome_prim_path=dome_path,
                usd_family="",
                world_present=True,
            )
        ]
    return []


def uv_value_edit_from_prim(
    mesh_owner: Any,
    prim: uv_usd_prim.UvUsdPrim,
    *,
    usd_layer_id: str | None = None,
    loop_order_validation: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> InteractiveEdit | None:
    snapshot = uv_usd_prim.active_uv_snapshot(mesh_owner)
    if snapshot.get("status") != uv_usd_prim.RESOLVED:
        return None
    previous_topology = _string_value(
        (loop_order_validation or {}).get("topology_fingerprint", "")
    )
    current_topology = _string_value(snapshot.get("topology_fingerprint", ""))
    if previous_topology and current_topology != previous_topology:
        return _mesh_topology_edit(
            mesh_owner,
            usd_prim_path=prim.prim_path,
            previous_fingerprint=previous_topology,
            current_fingerprint=current_topology,
            usd_layer_id=usd_layer_id,
        )
    if uv_usd_prim.cached_loop_order_validation_is_valid(loop_order_validation, snapshot, prim):
        validation = dict(loop_order_validation or {})
    else:
        validation = uv_usd_prim.validate_loop_order(snapshot, prim)
    if validation.get("status") != uv_usd_prim.RESOLVED:
        return None

    resolved_layer = _usd_layer_id(mesh_owner, override=usd_layer_id)
    uv_layer_name = _string_value(snapshot.get("uv_layer_name", ""))
    validation_record = uv_usd_prim.validation_record(validation)
    target = edit_location(
        usd_prim_path=prim.prim_path,
        usd_attribute=prim.target_attribute,
        usd_property_path=prim.prim_path + "." + prim.target_attribute,
        usd_layer_id=resolved_layer,
        blender_property_path=uv_usd_prim.DEFAULT_BLENDER_PROPERTY_PATH,
        provenance={
            **_provenance_from_blender_id(mesh_owner),
            "uv_loop_order_validation": validation_record,
        },
    )
    return InteractiveEdit(
        shape=EditShape.VALUE,
        data_authority=DataAuthority.VIEW,
        **target,
        value=tuple(snapshot.get("uv_values", ())),
        metadata={
            **dict(metadata or {}),
            "uv_layer_name": uv_layer_name,
            "element_count": int(snapshot.get("element_count", 0) or 0),
            "topology_fingerprint": _string_value(snapshot.get("topology_fingerprint", "")),
            "uv_digest": _string_value(snapshot.get("uv_digest", "")),
            "loop_order_validation": validation_record,
        },
    )


def uv_value_edits_from_resolver(
    mesh_owner: Any,
    resolver: UsdPrimResolver,
    *,
    usd_layer_id: str | None = None,
) -> list[InteractiveEdit]:
    topology_change = getattr(resolver, "mesh_topology_change", lambda _mesh: None)(
        mesh_owner
    )
    if topology_change is not None:
        return [
            _mesh_topology_edit(
                mesh_owner,
                usd_prim_path=_string_value(topology_change["usd_prim_path"]),
                previous_fingerprint=_string_value(
                    topology_change["previous_fingerprint"]
                ),
                current_fingerprint=_string_value(
                    topology_change["current_fingerprint"]
                ),
                usd_layer_id=usd_layer_id,
            )
        ]
    resolution = resolver.resolve_uv(mesh_owner)
    if resolution.value is None:
        return []
    edit = uv_value_edit_from_prim(
        mesh_owner,
        resolution.value,
        usd_layer_id=usd_layer_id,
        loop_order_validation=resolver.uv_loop_order_validation(resolution.value.prim_path),
    )
    return [edit] if edit is not None else []


def _mesh_topology_edit(
    mesh_owner: Any,
    *,
    usd_prim_path: str,
    previous_fingerprint: str,
    current_fingerprint: str,
    usd_layer_id: str | None,
) -> InteractiveEdit:
    return InteractiveEdit(
        shape=EditShape.TOPOLOGY,
        data_authority=DataAuthority.VIEW,
        **edit_location(
            usd_prim_path=usd_prim_path,
            usd_layer_id=_usd_layer_id(mesh_owner, override=usd_layer_id),
            blender_property_path="vertices",
            provenance=_provenance_from_blender_id(mesh_owner),
        ),
        value=current_fingerprint,
        previous_value=previous_fingerprint,
        metadata={
            "topology_change_kinds": ("mesh_topology",),
            "topology_reasons": (topology_edit_fallback.MESH_TOPOLOGY_CHANGED,),
        },
    )


def property_edit(
    id_data: Any,
    *,
    data_authority: DataAuthority | str = DataAuthority.VIEW,
    property_name: str,
    usd_attribute: str,
    location: Mapping[str, Any] | None = None,
) -> InteractiveEdit:
    return _value_edit(
        data_authority=_parse_data_authority(data_authority),
        id_data=id_data,
        property_name=property_name,
        usd_attribute=usd_attribute,
        value=None,
        metadata=None,
        location=location,
    )


def _value_edit(
    data_authority: DataAuthority,
    id_data: Any,
    *,
    property_name: str,
    usd_attribute: str,
    value: Any | None,
    metadata: Mapping[str, Any] | None,
    location: Mapping[str, Any] | None,
) -> InteractiveEdit:
    metadata_dict = dict(metadata or {})
    return InteractiveEdit(
        shape=EditShape.VALUE,
        data_authority=data_authority,
        **dict(location or edit_location_from_blender_id(
            id_data, usd_attribute=usd_attribute, blender_property_path=property_name
        )),
        value=_property_value(id_data, property_name) if value is None else value,
        metadata=metadata_dict,
    )


def build_interactive_edits_from_depsgraph(
    depsgraph: Any,
    *,
    value_edit_conversion_policies: ValueEditConversionPolicies | None = None,
    usd_prim_resolver: UsdPrimResolver | None = None,
    light_objects: Any = (),
    worlds: Any = (),
    selection_resolution: Mapping[str, Any] | None = None,
    write_target_input_usd_path: str | None = None,
    write_target_ignored_layer_identifiers: Iterable[str] = (),
) -> list[InteractiveEdit]:
    ignored_layer_identifiers = tuple(
        str(identifier)
        for identifier in write_target_ignored_layer_identifiers
        if str(identifier)
    )
    policies = _value_edit_conversion_policies(value_edit_conversion_policies)
    light_lookup = _light_object_lookup(light_objects)
    edits: list[InteractiveEdit] = []
    # One depsgraph event can report several updated IDs for one light edit
    # (the evaluated Object and its Light data): emit each resolved light's
    # edit set once per builder call (task04-03).
    seen_light_prim_paths: set[str] = set()
    # Likewise a world change can report both the Scene and the World ID:
    # emit the world lane's edit set once per builder call (task04-04).
    seen_world_dome_paths: set[str] = set()
    # Node-tree-only world updates (e.g. adding an unlinked node reports
    # only the embedded ShaderNodeTree ID, task04-04 gap) are collected
    # during the loop and handled in a post-pass, so an event that also
    # reports the World keeps the world lane's behavior byte-identical.
    node_tree_world_updates: list[tuple[Any, Any]] = []
    for update in getattr(depsgraph, "updates", ()):
        id_data = getattr(update, "id", update)
        id_data = getattr(id_data, "original", id_data)
        update_edits: list[InteractiveEdit] = []
        try:
            edit = _interactive_edit_from_blender_id(id_data)
        except ValueError:
            continue
        if edit is not None:
            update_edits.append(edit)
        else:
            if usd_prim_resolver is not None and hasattr(id_data, "matrix_world"):
                object_resolution = usd_prim_resolver.resolve_blender_object(id_data)
                if object_resolution.value is not None:
                    update_edits.append(
                        object_transform_edit(
                            id_data,
                            data_authority=DataAuthority.VIEW,
                            location=edit_location(
                                usd_prim_path=object_resolution.value,
                                usd_attribute="xformOp:transform",
                                blender_property_path="matrix_world",
                                provenance={
                                    **_provenance_from_blender_id(id_data),
                                    **object_resolution.diagnostics_dict(),
                                },
                            ),
                        )
                    )
            if usd_prim_resolver is not None and _is_material_id(id_data):
                # Coarse linked-input topology routing applies to USD-import
                # materials (``sourceUsdPath`` identity): their authored graph
                # is not converter-owned, so any graph wiring re-composes.
                # Authoring-identity materials (``ov.usd.prim_path``) instead
                # use the precise authored-connection divergence check inside
                # the value lane (``_material_graph_topology_edit``), which
                # keeps value edits alive on unchanged graphs (task04-02).
                if (
                    not usd_paths.authoring_prim_path(id_data)
                    and usd_paths.source_usd_path_from_blender_id(id_data)
                    and _material_requires_topology_edit(id_data, policies)
                ):
                    topology_edit = material_graph_topology_edit_from_resolver(
                        id_data,
                        usd_prim_resolver,
                    )
                    if topology_edit is not None:
                        update_edits.append(topology_edit)
                else:
                    update_edits.extend(
                        material_value_edits_from_resolver(
                            id_data,
                            usd_prim_resolver,
                            value_edit_conversion_policies=policies,
                        )
                    )
            if usd_prim_resolver is not None:
                for light_object in _light_objects_for_update(id_data, light_lookup):
                    resolution = usd_prim_resolver.resolve_light(light_object)
                    if resolution.value is not None:
                        if resolution.value.prim_path in seen_light_prim_paths:
                            continue
                        seen_light_prim_paths.add(resolution.value.prim_path)
                        update_edits.extend(
                            light_value_edits_from_prim(
                                light_object,
                                resolution.value,
                                value_edit_conversion_policies=policies,
                                match_source=str(
                                    resolution.diagnostics.get("match_source", "")
                                ),
                            )
                        )
            if usd_prim_resolver is not None:
                world_edits = _world_edits_for_update(
                    id_data,
                    usd_prim_resolver,
                    policies,
                )
                if world_edits:
                    dome_path = world_edits[0].usd_prim_path
                    if dome_path not in seen_world_dome_paths:
                        seen_world_dome_paths.add(dome_path)
                        update_edits.extend(world_edits)
            if (
                usd_prim_resolver is not None
                and _is_mesh_id(id_data)
                and _mesh_geometry_updated(update, id_data)
            ):
                update_edits.extend(
                    uv_value_edits_from_resolver(
                        id_data,
                        usd_prim_resolver,
                    )
                )
            if usd_prim_resolver is not None and not update_edits:
                owner_world = _world_for_node_tree_update(id_data, worlds)
                if owner_world is not None:
                    node_tree_world_updates.append((id_data, owner_world))
        update_edits = [
            _with_write_target_resolution(
                _with_changed_blender_id(edit, id_data),
                input_usd_path=write_target_input_usd_path,
                ignored_layer_identifiers=ignored_layer_identifiers,
            )
            for edit in update_edits
        ]
        edits.extend(_with_selection_resolution(edit, id_data, selection_resolution) for edit in update_edits)
    for id_data, owner_world in node_tree_world_updates:
        classification_edit = _world_node_tree_classification_edit(
            owner_world,
            usd_prim_resolver,
            policies,
        )
        if classification_edit is None:
            continue
        if classification_edit.usd_prim_path in seen_world_dome_paths:
            # The same event reported the World/Scene ID: the world lane
            # already emitted the real edit set for this change.
            continue
        seen_world_dome_paths.add(classification_edit.usd_prim_path)
        classification_edit = _with_write_target_resolution(
            classification_edit,
            input_usd_path=write_target_input_usd_path,
            ignored_layer_identifiers=ignored_layer_identifiers,
        )
        edits.append(
            _with_selection_resolution(classification_edit, id_data, selection_resolution)
        )
    return edits


def _world_for_node_tree_update(id_data: Any, worlds: Any) -> Any | None:
    """Map a node-tree-only depsgraph update back to its owning World.

    Blender 5.1 reports only the embedded ShaderNodeTree ID for some
    intermediate node-graph states (e.g. adding an unlinked node), which
    reaches no lane (task04-04 gap). Embedded node trees expose no owner
    pointer, so the owning World is found by identity against the caller's
    world datablocks (evaluated tree copies map back via ``original``).
    """

    if not _is_node_tree_id(id_data):
        return None
    candidates = {id(id_data)}
    original = getattr(id_data, "original", None)
    if original is not None:
        candidates.add(id(original))
    for world in tuple(worlds or ()):
        tree = getattr(world, "node_tree", None)
        if tree is None:
            continue
        if id(tree) in candidates:
            return world
    return None


def _is_node_tree_id(id_data: Any) -> bool:
    # Node trees carry ``nodes`` + ``links`` and, unlike the IDs that embed
    # them (worlds, materials, lights), no ``node_tree`` attribute.
    return (
        hasattr(id_data, "nodes")
        and hasattr(id_data, "links")
        and not hasattr(id_data, "node_tree")
    )


def _world_node_tree_classification_edit(
    world: Any,
    resolver: UsdPrimResolver,
    policies: ValueEditConversionPolicies,
) -> InteractiveEdit | None:
    """Report-only record for a node-tree-only world update (task04-07).

    When the intermediate state classifies TOPOLOGY (environment texture
    or node-graph shape), the report originates from classification: the
    ``unsupported_reason`` metadata routes the planner to an UNSUPPORTED
    plan (mechanism NONE — no RPC, no refinement reset, no reconcile churn
    for the intermediate state) while the explicit ``classification``
    metadata keeps the record and the once-per-key report on the topology
    vocabulary ("applies on next scene update"). A still-supported world
    shape (e.g. an unlinked utility node next to the bare Background)
    emits nothing — nothing render-relevant changed.
    """

    spec_fn = getattr(policies.world, "world_dome_spec", None)
    if not callable(spec_fn):
        return None
    spec = spec_fn(world)
    if _string_value(getattr(spec, "status", "")) != STATUS_TOPOLOGY:
        return None
    reason = _string_value(getattr(spec, "reason", ""))
    dome_path = world_dome_conversion.DEFAULT_DOME_OWNER_PATH
    if resolver is not None:
        resolution = resolver.resolve_world_dome()
        if resolution.value is not None:
            dome_path = resolution.value.prim_path
        else:
            dome_path = (
                _string_value(resolution.diagnostics.get("prim_path", ""))
                or dome_path
            )
    location = edit_location(
        usd_prim_path=dome_path,
        usd_layer_id=_usd_layer_id(world),
        blender_property_path="node_tree",
        provenance={
            **_provenance_from_blender_id(world),
            "dome_owner_path": dome_path,
            "classification_origin": "world_node_tree_update",
            "world_dome_conversion": {
                "status": _string_value(getattr(spec, "status", "")),
                "reason": reason,
            },
        },
    )
    return InteractiveEdit(
        shape=EditShape.VALUE,
        data_authority=DataAuthority.VIEW,
        **dict(location),
        value=reason,
        previous_value="",
        metadata={
            "unsupported_reason": reason,
            "classification": CLASSIFICATION_TOPOLOGY,
            "topology_change_kinds": (
                ("environment_texture",)
                if reason == world_dome_conversion.ENVIRONMENT_TEXTURE_CHANGED
                else ("world_node_graph",)
            ),
        },
    )


def _value_edit_conversion_policies(
    policies: ValueEditConversionPolicies | None,
) -> ValueEditConversionPolicies:
    return policies or default_value_edit_conversion_policies()


def _material_requires_topology_edit(
    material: Any,
    policies: ValueEditConversionPolicies,
) -> bool:
    predicate = getattr(policies.material, "requires_topology_edit", None)
    return bool(callable(predicate) and predicate(material))


def _interactive_edit_from_blender_id(
    id_data: Any,
) -> InteractiveEdit | None:
    raw_authority = usd_paths.id_property(id_data, usd_paths.DATA_AUTHORITY_PROP, "")
    blender_property_path = _configured_property_name(id_data, "")
    usd_attribute = _configured_usd_attribute(id_data, "")
    if raw_authority in {None, ""}:
        return None
    authority = _parse_data_authority(raw_authority)
    if blender_property_path == "uv_layers.active":
        return None
    if blender_property_path == "matrix_world" or authority == DataAuthority.SIM:
        return object_transform_edit(id_data, data_authority=authority)
    if blender_property_path == "world_dome":
        return world_value_edit(id_data, property_name=blender_property_path, usd_attribute=usd_attribute or "inputs:intensity")
    if blender_property_path in {"energy", "data.type", "data.shape"}:
        return light_value_edit(id_data, property_name=blender_property_path, usd_attribute=usd_attribute or "inputs:intensity")
    if blender_property_path:
        return material_value_edit(id_data, property_name=blender_property_path, usd_attribute=usd_attribute or "inputs:diffuseColor")
    return property_edit(
        id_data,
        data_authority=authority,
        property_name=blender_property_path,
        usd_attribute=usd_attribute,
    )


def _with_selection_resolution(
    edit: InteractiveEdit,
    id_data: Any,
    selection_resolution: Mapping[str, Any] | None,
) -> InteractiveEdit:
    selection_record = _selection_record_for_id(id_data, selection_resolution)
    if selection_record is None:
        return edit
    return replace(
        edit,
        provenance={
            **dict(edit.provenance),
            "selection_resolution": selection_record,
        },
        metadata=dict(edit.metadata),
    )


def _with_changed_blender_id(edit: InteractiveEdit, id_data: Any) -> InteractiveEdit:
    provenance = dict(edit.provenance)
    changed_id = _provenance_from_blender_id(id_data)
    for key in ("blender_id_kind", "blender_session_uid"):
        provenance.pop(key, None)
        if key in changed_id:
            provenance[key] = changed_id[key]
    return replace(edit, provenance=provenance, metadata=dict(edit.metadata))


def _with_write_target_resolution(
    edit: InteractiveEdit,
    *,
    input_usd_path: str | None,
    ignored_layer_identifiers: Iterable[str],
) -> InteractiveEdit:
    target_kind = _write_target_kind(edit)
    resolution = write_target_resolution.resolve_write_target(
        str(input_usd_path or ""),
        usd_prim_path=edit.usd_prim_path,
        target_kind=target_kind,
        usd_property_name=edit.usd_attribute,
        explicit_usd_layer_id=edit.usd_layer_id,
        ignored_layer_identifiers=ignored_layer_identifiers,
    )
    provenance = dict(edit.provenance)
    provenance.pop("write_target_resolution", None)
    provenance.pop("write_target_error_reason", None)
    if resolution.status is write_target_resolution.WriteTargetResolutionStatus.ERROR:
        provenance["write_target_error_reason"] = str(resolution.error_reason)
    return replace(
        edit,
        usd_layer_id=resolution.usd_layer_id or "",
        provenance=provenance,
        metadata=dict(edit.metadata),
    )


def _write_target_kind(edit: InteractiveEdit) -> str:
    if edit.shape == EditShape.TOPOLOGY and edit.blender_property_path in {"node_tree", "material_slots"}:
        return write_target_resolution.TARGET_KIND_RELATIONSHIP
    if edit.shape == EditShape.TOPOLOGY:
        return write_target_resolution.TARGET_KIND_PRIM
    if edit.usd_attribute:
        return write_target_resolution.TARGET_KIND_ATTRIBUTE
    return ""


def _selection_record_for_id(
    id_data: Any,
    selection_resolution: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(selection_resolution, Mapping):
        return None
    records = selection_resolution.get("sources", ())
    if not isinstance(records, (list, tuple)):
        return None
    session_uid = int(getattr(id_data, "session_uid", 0) or 0)
    if session_uid <= 0:
        return None
    for record in records:
        if not isinstance(record, Mapping):
            continue
        record_session_uids = {
            int(record.get(key, 0) or 0)
            for key in (
                "source_session_uid",
                "source_data_session_uid",
                "owner_session_uid",
                "owner_data_session_uid",
            )
        }
        if session_uid in record_session_uids:
            return dict(record)
    return None


def _parse_data_authority(value: DataAuthority | str | None) -> DataAuthority:
    if isinstance(value, DataAuthority):
        return value
    normalized = str(value or "").strip().lower().replace("-", "_")
    if not normalized:
        return DataAuthority.VIEW
    try:
        return DataAuthority(normalized)
    except ValueError as exc:
        raise ValueError(f"Unsupported data authority: {value!r}") from exc


def _configured_property_name(
    id_data: Any,
    default: str,
) -> str:
    return _string_value(
        usd_paths.id_property(id_data, usd_paths.BLENDER_PROPERTY_PATH_PROP, default)
    ) or default


def _configured_usd_attribute(
    id_data: Any,
    default: str,
) -> str:
    return _string_value(usd_paths.id_property(id_data, USD_ATTRIBUTE_PROP, default)) or default


def _provenance_from_blender_id(id_data: Any) -> dict[str, Any]:
    provenance: dict[str, Any] = {"source": "blender"}
    session_uid = int(getattr(id_data, "session_uid", 0) or 0)
    identifier = _string_value(
        getattr(getattr(id_data, "bl_rna", None), "identifier", "")
    ).upper()
    if session_uid > 0:
        provenance["blender_session_uid"] = session_uid
    if identifier:
        provenance["blender_id_kind"] = identifier
    name = _string_value(getattr(id_data, "name_full", getattr(id_data, "name", "")))
    if name:
        provenance["name"] = name
    blender_type = _string_value(getattr(id_data, "type", ""))
    if blender_type:
        provenance["type"] = blender_type
    return provenance


def _usd_layer_id(id_data: Any, *, override: str | None = None) -> str:
    if override is not None:
        return _string_value(override)
    return _string_value(usd_paths.id_property(id_data, usd_paths.USD_LAYER_ID_PROP, ""))


def _current_authored_light_form(light_object: Any) -> str:
    light_data = getattr(light_object, "data", light_object)
    return light_conversion.authored_light_form(
        _string_value(getattr(light_data, "type", "")).upper(),
        _string_value(getattr(light_data, "shape", "")).upper(),
    )


def _light_form_blender_property_path(previous_form: str, current_form: str) -> str:
    if previous_form.startswith("AREA_") and current_form.startswith("AREA_"):
        return "data.shape"
    return "data.type"


def _unsupported_light_form_edit(
    *,
    location: Mapping[str, Any],
    current_authored_light_form: str,
) -> InteractiveEdit:
    return InteractiveEdit(
        shape=EditShape.VALUE,
        data_authority=DataAuthority.VIEW,
        **dict(location),
        value=current_authored_light_form,
        previous_value="",
        metadata={
            "unsupported_reason": light_conversion.MISSING_PREVIOUS_AUTHORED_LIGHT_FORM,
            "current_authored_light_form": current_authored_light_form,
        },
    )


def _light_form_topology_edit(
    light_object: Any,
    *,
    location: Mapping[str, Any],
    previous_authored_light_form: str,
    current_authored_light_form: str,
    previous_usd_family: str,
) -> InteractiveEdit | None:
    light_data = getattr(light_object, "data", light_object)
    current_family = light_conversion.exported_light_family(
        _string_value(getattr(light_data, "type", "")).upper(),
        _string_value(getattr(light_data, "shape", "")).upper(),
    )
    if previous_authored_light_form == current_authored_light_form:
        return None
    return InteractiveEdit(
        shape=EditShape.TOPOLOGY,
        data_authority=DataAuthority.VIEW,
        **dict(location),
        value=current_authored_light_form,
        previous_value=previous_authored_light_form,
        metadata={
            "topology_change_kinds": ("light_form",),
            "previous_authored_light_form": previous_authored_light_form,
            "current_authored_light_form": current_authored_light_form,
            "previous_usd_family": previous_usd_family,
            "current_usd_family": current_family,
            "topology_attribute_values": _light_form_topology_attribute_values(
                light_object,
                current_authored_light_form=current_authored_light_form,
            ),
        },
    )


def _light_form_topology_attribute_values(
    light_object: Any,
    *,
    current_authored_light_form: str,
) -> tuple[dict[str, Any], ...]:
    attributes = [
        {
            "name": attribute.name,
            "value": attribute.value,
            "value_type": attribute.value_type,
            "blender_property_path": attribute.blender_property_path,
            "metadata": dict(attribute.metadata),
        }
        for attribute in light_conversion.usd_attribute_values(light_object)
    ]
    if current_authored_light_form == light_conversion.AUTHORED_LIGHT_FORM_POINT:
        attributes.extend(
            (
                {
                    "name": "inputs:shaping:cone:angle",
                    "value": 180.0,
                    "value_type": "Float",
                    "blender_property_path": "type",
                    "metadata": {"source_property": "type", "light_form_policy": "clear_spot_shape"},
                },
                {
                    "name": "inputs:shaping:cone:softness",
                    "value": 0.0,
                    "value_type": "Float",
                    "blender_property_path": "type",
                    "metadata": {"source_property": "type", "light_form_policy": "clear_spot_shape"},
                },
            )
        )
    return tuple(attributes)


def _property_value(id_data: Any, property_name: str) -> Any:
    if property_name.startswith("principled:"):
        return _principled_input_value(id_data, property_name.split(":", 1)[1])
    value = getattr(id_data, property_name, None)
    if value is None:
        value = usd_paths.id_property(id_data, property_name, None)
    return _plain_value(value)


def _principled_input_value(material: Any, input_name: str) -> Any:
    node_tree = getattr(material, "node_tree", None)
    for node in getattr(node_tree, "nodes", ()):
        node_type = _string_value(getattr(node, "type", ""))
        node_name = _string_value(getattr(node, "name", ""))
        if node_type != "BSDF_PRINCIPLED" and node_name != "Principled BSDF":
            continue
        inputs = getattr(node, "inputs", {})
        getter = getattr(inputs, "get", None)
        socket = getter(input_name) if callable(getter) else None
        if socket is None:
            return None
        return _plain_value(getattr(socket, "default_value", None))
    return None


def _is_material_id(id_data: Any) -> bool:
    blender_type = _string_value(getattr(id_data, "type", ""))
    if blender_type == "MATERIAL":
        return True
    return hasattr(id_data, "diffuse_color")


def _is_world_id(id_data: Any) -> bool:
    blender_type = _string_value(getattr(id_data, "type", ""))
    if blender_type == "WORLD":
        return True
    # bpy.types.Light also carries ``color`` and ``use_nodes`` (light node
    # trees); without the ``energy`` exclusion a depsgraph Light data
    # update misrouted into the world lane and rewrote the dome from the
    # light's values (task04-04 review). Worlds have no ``energy``.
    return (
        hasattr(id_data, "color")
        and hasattr(id_data, "use_nodes")
        and not hasattr(id_data, "energy")
    )


def _is_scene_id(id_data: Any) -> bool:
    # bpy.types.Scene has no ``type`` attribute; the ``world`` + ``render``
    # pair distinguishes Scene IDs from every other depsgraph-reported ID
    # (task04-04: scene updates carry world add/remove divergence).
    if _string_value(getattr(id_data, "type", "")) == "SCENE":
        return True
    return hasattr(id_data, "world") and hasattr(id_data, "render")


_LIGHT_DATA_TYPES = frozenset({"POINT", "SPOT", "SUN", "AREA"})


def _light_object_lookup(light_objects: Any) -> dict[str, dict[Any, list[Any]]]:
    by_identity: dict[tuple[str, int], list[Any]] = {}
    by_data_name: dict[str, list[Any]] = {}
    for obj in tuple(light_objects or ()):
        if _string_value(getattr(obj, "type", "")) != "LIGHT":
            continue
        for key in _blender_id_lookup_keys(obj):
            by_identity.setdefault(key, []).append(obj)
        data = getattr(obj, "data", None)
        if data is not None:
            for key in _blender_id_lookup_keys(data):
                by_identity.setdefault(key, []).append(obj)
            data_name = _string_value(getattr(data, "name_full", getattr(data, "name", "")))
            if data_name:
                by_data_name.setdefault(data_name, []).append(obj)
    return {"by_identity": by_identity, "by_data_name": by_data_name}


def _blender_id_lookup_keys(value: Any) -> tuple[tuple[str, int], ...]:
    keys = [("python", id(value))]
    as_pointer = getattr(value, "as_pointer", None)
    pointer = int(as_pointer() or 0) if callable(as_pointer) else 0
    if pointer > 0:
        keys.append(("pointer", pointer))
    session_uid = int(getattr(value, "session_uid", 0) or 0)
    if session_uid > 0:
        keys.append(("session_uid", session_uid))
    return tuple(keys)


def _light_objects_for_update(
    id_data: Any,
    lookup: Mapping[str, Mapping[Any, list[Any]]],
) -> tuple[Any, ...]:
    by_identity = lookup.get("by_identity", {})
    for key in _blender_id_lookup_keys(id_data):
        if direct := tuple(by_identity.get(key, ())):
            return direct
    # Depsgraph updates carry evaluated ID copies whose Python identity
    # differs from the originals the caller passed in ``light_objects``
    # (blender-live-render task04-03): map the update back through the
    # original datablock, then through the light-data name (evaluation
    # keeps ID names), so light data edits reach their owning objects.
    original = getattr(id_data, "original", None)
    if original is not None:
        for key in _blender_id_lookup_keys(original):
            if direct := tuple(by_identity.get(key, ())):
                return direct
    if _is_light_data_id(id_data):
        data_name = _string_value(getattr(id_data, "name_full", getattr(id_data, "name", "")))
        direct = tuple(lookup.get("by_data_name", {}).get(data_name, ()))
        if direct:
            return direct
    if _string_value(getattr(id_data, "type", "")) == "LIGHT":
        return (id_data,)
    return ()


def _is_light_data_id(id_data: Any) -> bool:
    return (
        _string_value(getattr(id_data, "type", "")).upper() in _LIGHT_DATA_TYPES
        and hasattr(id_data, "energy")
    )




def _is_mesh_id(id_data: Any) -> bool:
    if _string_value(getattr(id_data, "type", "")) == "MESH":
        return True
    if hasattr(id_data, "uv_layers"):
        return True
    data = getattr(id_data, "data", None)
    return _string_value(getattr(data, "type", "")) == "MESH" or hasattr(data, "uv_layers")


def _mesh_geometry_updated(update: Any, id_data: Any) -> bool:
    marker = getattr(update, "is_updated_geometry", None)
    if marker is not None:
        return bool(marker)
    return hasattr(id_data, "uv_layers")


def _matrix_rows(value: Any) -> list[list[float]]:
    if hasattr(value, "to_4x4"):
        value = value.to_4x4()
    rows = list(value)
    if len(rows) != 4:
        raise ValueError("Blender transform matrix must have four rows")
    matrix: list[list[float]] = []
    for row in rows:
        row_values = list(row)
        if len(row_values) != 4:
            raise ValueError("Blender transform matrix rows must have four values")
        matrix.append([float(item) for item in row_values])
    return _transpose_matrix4(matrix)


def _runtime_body_matrix_world(obj: Any) -> Any:
    matrix_world = getattr(obj, "matrix_world")
    offset = _matrix_from_id_property(usd_paths.id_property(obj, BODY_VISUAL_OFFSET_MATRIX_PROP, None))
    if offset is None:
        return matrix_world
    try:
        return matrix_world @ offset.inverted()
    except Exception as exc:
        raise ValueError("Blender body visual offset matrix could not be applied") from exc


def _matrix_from_id_property(value: Any) -> Any | None:
    if value in (None, ""):
        return None
    try:
        from mathutils import Matrix  # type: ignore
    except Exception as exc:
        raise ValueError("mathutils is required for body visual offset matrices") from exc
    values = _plain_value(value)
    if isinstance(values, tuple) and len(values) == 4 and all(
        isinstance(row, tuple) and len(row) == 4 for row in values
    ):
        return Matrix(values)
    if isinstance(values, tuple) and len(values) == 16:
        return Matrix([values[index : index + 4] for index in range(0, 16, 4)])
    raise ValueError("Body visual offset matrix must contain 16 values")


def _transpose_matrix4(matrix: list[list[float]]) -> list[list[float]]:
    return [[matrix[row][column] for row in range(4)] for column in range(4)]


def _plain_value(value: Any) -> Any:
    if isinstance(value, (str, bytes, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_plain_value(item) for item in value)
    try:
        return tuple(value)
    except TypeError:
        return value


def _string_value(value: Any) -> str:
    return str(value or "").strip()

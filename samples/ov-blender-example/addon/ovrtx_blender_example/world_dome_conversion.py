# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Blender World to OVRTX dome value edit conversion policy."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Sequence

from . import usd_value_edit_support
from .value_edit_conversion import (
    BLENDER_DATABLOCK_NON_RENDER_FIELD_REASONS,
    FieldClassification,
    STATUS_NON_RENDER,
    STATUS_SUPPORTED,
    STATUS_TOPOLOGY,
    STATUS_UNSUPPORTED,
    UsdAttributeValue,
    classify_mapped_field,
    float_value,
    node_input,
    socket_is_linked,
)


DOME_LIGHT_SCALE = 360.0 * math.pi
DOME_POLICY_VERSION = "ovrtx_0_3_world_dome_parity_2026_06_11"

ENVIRONMENT_TEXTURE_CHANGED = "topology_world_environment_texture_changed"
# Any node-based world beyond one bare Background node (sky texture, mixed
# graphs, ambiguous outputs) is topology for the live-edit route
# (blender-live-render task04-04): the value-editable world is the
# solid-color world only.
WORLD_NODE_GRAPH_CHANGED = "world_node_graph_changes_are_topology"
# Adding a world where none existed (or removing the world) changes which
# prims the authored generation contains: topology, generation route.
WORLD_ASSIGNMENT_CHANGED = "world_datablock_assignment_is_topology"

DEFAULT_DOME_OWNER_PATH = "/World/StudioDome"

SUPPORTED_USD_ATTRIBUTES = usd_value_edit_support.WORLD_USD_VALUE_TYPES
EDIT_VALUE_ATTRIBUTES_BY_FIELD = MappingProxyType(
    {
        "color": tuple(SUPPORTED_USD_ATTRIBUTES),
        "background_color": tuple(SUPPORTED_USD_ATTRIBUTES),
        "background_strength": tuple(SUPPORTED_USD_ATTRIBUTES),
        "world_dome": tuple(SUPPORTED_USD_ATTRIBUTES),
    }
)
EDIT_VALUE_CONCEPTS = frozenset({"world.color_strength"})
EXPORT_VALUE_CONCEPTS = EDIT_VALUE_CONCEPTS
EDIT_TOPOLOGY_CONCEPTS = frozenset()
EDIT_TOPOLOGY_KINDS = frozenset()

_TOPOLOGY_FIELD_REASONS = {
    "world": WORLD_ASSIGNMENT_CHANGED,
    "use_nodes": "world_node_mode_changes_are_topology",
    "node_tree": WORLD_NODE_GRAPH_CHANGED,
    "environment_texture": ENVIRONMENT_TEXTURE_CHANGED,
    "inputs:texture:file": ENVIRONMENT_TEXTURE_CHANGED,
}

_UNSUPPORTED_FIELD_REASONS = {
    "use_eevee_finite_volume": "unsupported_world_volume_mode",
    "lightgroup": "unsupported_world_lightgroup",
    "probe_resolution": "unsupported_world_probe_resolution",
    "sun_threshold": "unsupported_world_sun_shadow_settings",
    "sun_angle": "unsupported_world_sun_shadow_settings",
    "use_sun_shadow": "unsupported_world_sun_shadow_settings",
    "sun_shadow_maximum_resolution": "unsupported_world_sun_shadow_settings",
    "sun_shadow_filter_radius": "unsupported_world_sun_shadow_settings",
    "use_sun_shadow_jitter": "unsupported_world_sun_shadow_settings",
    "sun_shadow_jitter_overblur": "unsupported_world_sun_shadow_settings",
}

_NON_RENDER_FIELD_REASONS = {
    "name": "non_runtime_world_identifier",
    "name_full": "non_runtime_world_identifier",
    **BLENDER_DATABLOCK_NON_RENDER_FIELD_REASONS,
}


@dataclass(frozen=True)
class WorldDomeSpec:
    status: str
    reason: str
    effective_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0)
    peak: float = 0.0
    color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    intensity: float = 0.0
    source_fields: tuple[str, ...] = ()
    scene_linear_rgb_assumption: bool = True


def classify_field(world: Any, property_name: str) -> FieldClassification:
    field = str(property_name or "").strip()
    classification = classify_mapped_field(
        field,
        non_render=_NON_RENDER_FIELD_REASONS,
        topology=_TOPOLOGY_FIELD_REASONS,
        unsupported=_UNSUPPORTED_FIELD_REASONS,
    )
    if classification is not None:
        return classification
    if field == "color" and not bool(getattr(world, "use_nodes", False)):
        return FieldClassification(
            STATUS_SUPPORTED,
            "supported_world_flat_color",
            EDIT_VALUE_ATTRIBUTES_BY_FIELD[field],
        )
    if field in {"background_color", "background_strength"} and bool(getattr(world, "use_nodes", False)):
        spec = world_dome_spec(world)
        if spec.status == STATUS_SUPPORTED:
            return FieldClassification(
                STATUS_SUPPORTED,
                "supported_world_background_value",
                EDIT_VALUE_ATTRIBUTES_BY_FIELD[field],
            )
        return FieldClassification(spec.status, spec.reason)
    return FieldClassification(STATUS_UNSUPPORTED, "unsupported_world_field")


def world_dome_spec(source: Any) -> WorldDomeSpec:
    world = _world_from_source(source)
    if world is None:
        return _supported_spec(
            (0.0, 0.0, 0.0),
            source_fields=("scene.world",),
            reason="no_world",
        )
    if bool(getattr(world, "use_nodes", False)):
        return _node_world_dome_spec(world)
    return _supported_spec(
        _rgb_tuple(getattr(world, "color", (0.0, 0.0, 0.0))),
        source_fields=("World.color",),
        reason="flat_world_color",
    )


def usd_attribute_values(source: Any) -> tuple[UsdAttributeValue, ...]:
    spec = world_dome_spec(source)
    if spec.status != STATUS_SUPPORTED:
        return ()
    common_metadata = {
        "conversion_policy": DOME_POLICY_VERSION,
        "dome_light_scale": DOME_LIGHT_SCALE,
        "formula": "effective_rgb -> color=effective_rgb/peak; intensity=peak*DOME_LIGHT_SCALE",
        "effective_rgb": spec.effective_rgb,
        "peak": spec.peak,
        "source_fields": spec.source_fields,
        "scene_linear_rgb_assumption": spec.scene_linear_rgb_assumption,
        "target_dome_prim": DEFAULT_DOME_OWNER_PATH,
    }
    return (
        UsdAttributeValue(
            "inputs:intensity",
            spec.intensity,
            SUPPORTED_USD_ATTRIBUTES["inputs:intensity"],
            "world_dome",
            common_metadata,
        ),
        UsdAttributeValue(
            "inputs:color",
            spec.color,
            SUPPORTED_USD_ATTRIBUTES["inputs:color"],
            "world_dome",
            common_metadata,
        ),
    )


def _node_world_dome_spec(world: Any) -> WorldDomeSpec:
    node_tree = getattr(world, "node_tree", None)
    nodes = tuple(getattr(node_tree, "nodes", ()) or ())
    if _environment_texture_nodes(nodes):
        return WorldDomeSpec(STATUS_TOPOLOGY, ENVIRONMENT_TEXTURE_CHANGED)
    background_nodes = _background_nodes(nodes)
    if len(background_nodes) != 1:
        # Node-based worlds beyond one bare Background node (no background,
        # ambiguous backgrounds, mixed graphs) are topology for the
        # live-edit route (blender-live-render task04-04 clarification):
        # the graph shape changed, not a value — generation route.
        return WorldDomeSpec(STATUS_TOPOLOGY, WORLD_NODE_GRAPH_CHANGED)
    background = background_nodes[0]
    color_socket = node_input(background, "Color")
    strength_socket = node_input(background, "Strength")
    if socket_is_linked(color_socket) or socket_is_linked(strength_socket):
        # A linked Background input means a node graph drives the
        # background (sky texture, mixed graphs): node-based world →
        # topology for the live-edit route (task04-04).
        return WorldDomeSpec(STATUS_TOPOLOGY, WORLD_NODE_GRAPH_CHANGED)
    color = _rgb_tuple(_socket_default(color_socket, (0.0, 0.0, 0.0)))
    strength = float_value(_socket_default(strength_socket, 1.0), 1.0)
    effective_rgb = tuple(channel * max(0.0, strength) for channel in color)
    return _supported_spec(
        effective_rgb,
        source_fields=("Background.Color", "Background.Strength"),
        reason="background_color_strength",
    )


def _supported_spec(
    effective_rgb: Sequence[float],
    *,
    source_fields: tuple[str, ...],
    reason: str,
) -> WorldDomeSpec:
    values = list(effective_rgb)[:3]
    if len(values) != 3:
        rgb = (0.0, 0.0, 0.0)
    else:
        rgb = tuple(max(0.0, float(channel)) for channel in values)
    peak = max(rgb)
    if peak <= 0.0:
        return WorldDomeSpec(
            STATUS_SUPPORTED,
            reason,
            effective_rgb=rgb,
            peak=0.0,
            color=(0.0, 0.0, 0.0),
            intensity=0.0,
            source_fields=source_fields,
        )
    return WorldDomeSpec(
        STATUS_SUPPORTED,
        reason,
        effective_rgb=rgb,
        peak=peak,
        color=tuple(channel / peak for channel in rgb),
        intensity=peak * DOME_LIGHT_SCALE,
        source_fields=source_fields,
    )


def environment_texture_dome(source: Any) -> dict[str, Any] | None:
    """The common HDRI world shape as a textured-dome description.

    Matches exactly one Background node whose Color input is fed directly
    by exactly one Environment Texture node with an on-disk image, and an
    unlinked Strength. Returns the resolved texture path and dome
    intensity for authored-scene conversion; anything else (packed or
    missing image, several textures, mixed graphs) returns ``None``.

    This describes the *authored* dome only. The live value-edit lane
    keeps classifying environment-texture worlds as topology
    (``ENVIRONMENT_TEXTURE_CHANGED``): changing the HDRI is a new
    generation, not a value update.
    """

    world = _world_from_source(source)
    if world is None or not bool(getattr(world, "use_nodes", False)):
        return None
    node_tree = getattr(world, "node_tree", None)
    nodes = tuple(getattr(node_tree, "nodes", ()) or ())
    env_nodes = _environment_texture_nodes(nodes)
    background_nodes = _background_nodes(nodes)
    if len(env_nodes) != 1 or len(background_nodes) != 1:
        return None
    background = background_nodes[0]
    color_socket = node_input(background, "Color")
    if not socket_is_linked(color_socket):
        return None
    links = tuple(getattr(color_socket, "links", ()) or ())
    if len(links) != 1 or getattr(links[0], "from_node", None) is not env_nodes[0]:
        return None
    strength_socket = node_input(background, "Strength")
    if socket_is_linked(strength_socket):
        return None
    texture_file = _image_disk_path(getattr(env_nodes[0], "image", None))
    if not texture_file:
        return None
    strength = max(0.0, float_value(_socket_default(strength_socket, 1.0), 1.0))
    return {
        "texture_file": texture_file,
        "intensity": strength * DOME_LIGHT_SCALE,
        "color": (1.0, 1.0, 1.0),
        "reason": "background_environment_texture",
        "source_fields": ("Background.Strength", "EnvironmentTexture.image"),
    }


def _image_disk_path(image: Any) -> str:
    """An on-disk file for a Blender image, or empty string.

    Delegates to the shared texture materializer: on-disk images resolve
    in place, packed images (the common case in downloaded scenes) export
    their bytes once to the texture cache; generated images degrade.
    """

    from .texture_materialization import materialized_image_path

    return materialized_image_path(image)


def _world_from_source(source: Any) -> Any:
    if source is None:
        return None
    if hasattr(source, "world") and not hasattr(source, "color"):
        return getattr(source, "world", None)
    return source


def _background_nodes(nodes: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(node for node in nodes if _node_kind(node) in {"BACKGROUND", "ShaderNodeBackground"})


def _environment_texture_nodes(nodes: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(node for node in nodes if _node_kind(node) in {"TEX_ENVIRONMENT", "ShaderNodeTexEnvironment"})


def _node_kind(node: Any) -> str:
    for name in ("type", "bl_idname"):
        value = str(getattr(node, name, "") or "")
        if value:
            return value
    return str(getattr(node, "name", "") or "")


def _socket_default(socket: Any, default: Any) -> Any:
    if socket is None:
        return default
    return getattr(socket, "default_value", default)


def _rgb_tuple(value: Any) -> tuple[float, float, float]:
    # Real Blender color values (``bpy_prop_array`` socket defaults,
    # ``mathutils.Color``) are not ``collections.abc.Sequence`` instances;
    # an isinstance check silently read every real world as black
    # (blender-live-render task04-04). Accept any non-string iterable.
    if isinstance(value, (str, bytes)):
        return (0.0, 0.0, 0.0)
    try:
        values = list(value)
    except TypeError:
        return (0.0, 0.0, 0.0)
    if len(values) < 3:
        return (0.0, 0.0, 0.0)
    return tuple(max(0.0, float_value(values[index], 0.0)) for index in range(3))

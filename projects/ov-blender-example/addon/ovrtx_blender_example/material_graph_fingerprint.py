# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Output-affecting graph identity for authored material generation invalidation.

This is not a shader converter. A changed signature requests the existing
generation route; it does not make an unsupported graph supported. Unlinked
Principled values already owned by the native value lane are deliberately
excluded. No Blender imports, pixel reads, disk reads, or runtime calls occur.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from .material_value_conversion import EDIT_VALUE_ATTRIBUTES_BY_FIELD, _principled_node

_LIVE_INPUTS = frozenset(key.removeprefix("principled:") for key in EDIT_VALUE_ATTRIBUTES_BY_FIELD)
_UI_PROPERTIES = frozenset({
    "rna_type", "name", "label", "location", "location_absolute", "width", "width_hidden", "height",
    "dimensions", "select", "show_options", "show_preview", "hide", "parent",
    "use_custom_color", "color", "warning_propagation", "bl_idname", "type",
})
_PARAMETERS = (
    "mute", "operation", "blend_type", "clamp", "use_clamp", "vector_type",
    "interpolation", "projection", "projection_blend", "extension", "uv_map",
    "attribute_name", "space", "invert", "distribution", "subsurface_method",
)


def material_graph_signature(material: Any) -> str:
    """Digest the active surface graph, excluding supported native constants."""
    return _graph_signature(material, native_principled_values=True)


def world_graph_signature(world: Any) -> str:
    """Digest the active World graph, including Background and HDRI settings."""
    return _graph_signature(world, native_principled_values=False)


def _graph_signature(owner: Any, *, native_principled_values: bool) -> str:
    tree = getattr(owner, "node_tree", None)
    nodes = tuple(getattr(tree, "nodes", ()) or ())
    outputs = [node for node in nodes if _type(node) in {
        "OUTPUT_MATERIAL", "OUTPUT_WORLD", "ShaderNodeOutputMaterial", "ShaderNodeOutputWorld",
    }]
    active = [node for node in outputs if bool(getattr(node, "is_active_output", False))]
    roots = active or outputs
    if not roots:
        roots = [node for node in nodes if _type(node) in {"BSDF_PRINCIPLED", "ShaderNodeBsdfPrincipled"}]
    # Node collection order is not render semantics. Names only order roots;
    # names themselves never participate in the digest.
    roots.sort(key=lambda node: str(getattr(node, "name", "")))
    visiting: set[tuple[str, int]] = set()
    native_node = _principled_node(owner) if native_principled_values else None
    native_identity = _node_identity(native_node) if native_node is not None else None

    def encode_node(node: Any) -> Any:
        identity = _node_identity(node)
        if identity in visiting:
            return ("cycle", _type(node))
        visiting.add(identity)
        try:
            is_principled = _type(node) in {"BSDF_PRINCIPLED", "ShaderNodeBsdfPrincipled"}
            inputs = []
            for index, socket in enumerate(tuple(getattr(node, "inputs", ()) or ())):
                name = str(getattr(socket, "name", index))
                links = tuple(getattr(socket, "links", ()) or ())
                if links:
                    value = []
                    for link in links:
                        upstream = getattr(link, "from_node", None)
                        output = getattr(link, "from_socket", None)
                        value.append((str(getattr(output, "identifier", getattr(output, "name", ""))),
                                      encode_node(upstream)))
                    inputs.append((name, "linked", value))
                elif not (identity == native_identity and is_principled and name in _LIVE_INPUTS):
                    inputs.append((name, "value", _scalar(getattr(socket, "default_value", None))))
            outputs = [(str(getattr(socket, "name", index)), _scalar(getattr(socket, "default_value", None)))
                       for index, socket in enumerate(tuple(getattr(node, "outputs", ()) or ()))]
            image = getattr(node, "image", None)
            image_state = None if image is None else (
                str(getattr(image, "name_full", getattr(image, "name", ""))),
                str(getattr(image, "filepath", "")), str(getattr(image, "source", "")),
                str(getattr(getattr(image, "library", None), "filepath", "")),
                str(getattr(getattr(image, "colorspace_settings", None), "name", "")),
                str(getattr(getattr(image, "colorspace_settings", None), "is_data", "")),
                str(getattr(image, "alpha_mode", "")),
                bool(getattr(image, "is_dirty", False)), _scalar(getattr(image, "size", ())),
            )
            ramp = getattr(node, "color_ramp", None)
            ramp_state = None if ramp is None else (
                str(getattr(ramp, "interpolation", "")), str(getattr(ramp, "color_mode", "")),
                str(getattr(ramp, "hue_interpolation", "")),
                [(float(getattr(element, "position", 0)), _scalar(getattr(element, "color", ())))
                 for element in tuple(getattr(ramp, "elements", ()) or ())],
            )
            mapping = getattr(node, "mapping", None)
            curve_state = None if mapping is None else (
                [[(_scalar(getattr(point, "location", ())), str(getattr(point, "handle_type", "")))
                  for point in tuple(getattr(curve, "points", ()) or ())]
                 for curve in tuple(getattr(mapping, "curves", ()) or ())],
                _scalar(getattr(mapping, "black_level", ())), _scalar(getattr(mapping, "white_level", ())),
                str(getattr(mapping, "extend", "")),
            )
            group_nodes = tuple(getattr(getattr(node, "node_tree", None), "nodes", ()) or ())
            group_outputs = [child for child in group_nodes
                             if _type(child) in {"GROUP_OUTPUT", "NodeGroupOutput"}]
            active_group_outputs = [child for child in group_outputs
                                    if bool(getattr(child, "is_active_output", False))]
            group_state = [encode_node(child) for child in active_group_outputs or group_outputs]
            return (_type(node), _parameters(node), inputs, outputs, image_state, ramp_state, group_state,
                    curve_state)
        finally:
            visiting.remove(identity)

    payload = (bool(getattr(owner, "use_nodes", False)), [encode_node(node) for node in roots])
    return hashlib.sha256(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def _node_identity(node: Any) -> tuple[str, int]:
    """RNA wrappers can be recreated on each link/collection lookup.

    Pointer identity is used only for comparisons during this traversal, never
    serialized into the semantic digest. Fixtures without RNA retain normal
    Python object identity. A zero/dead RNA pointer is not an identity proof.
    """
    pointer = getattr(node, "as_pointer", None)
    if callable(pointer):
        try:
            value = int(pointer())
        except (ReferenceError, TypeError, ValueError):
            value = 0
        if value:
            return ("RNA", value)
    return ("PYTHON", id(node))


def _type(node: Any) -> str:
    return str(getattr(node, "type", getattr(node, "bl_idname", "")))


def _parameters(node: Any) -> dict[str, Any]:
    names = set(_PARAMETERS)
    for prop in tuple(getattr(getattr(node, "bl_rna", None), "properties", ()) or ()):
        if str(getattr(prop, "type", "")) in {"BOOLEAN", "INT", "FLOAT", "STRING", "ENUM"}:
            names.add(str(getattr(prop, "identifier", "")))
    result = {}
    for name in sorted(names - _UI_PROPERTIES):
        try:
            value = getattr(node, name)
        except (AttributeError, ReferenceError):
            continue
        converted = _scalar(value)
        if converted is not None:
            result[name] = converted
    return result


def _scalar(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    try:
        sequence = tuple(value)
    except TypeError:
        return None
    if all(isinstance(item, (bool, int, float, str)) for item in sequence):
        return sequence
    return None

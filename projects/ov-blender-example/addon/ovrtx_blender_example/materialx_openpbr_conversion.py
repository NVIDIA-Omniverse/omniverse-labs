# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert Blender materials into a composition-ready USD scene entry."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping

from . import usd_paths as usd_paths
from .render_requests import MaterialPresentationLayer


_SOURCE = "materialx_openpbr"
_STATUS_GENERATED = "generated"
_STATUS_UNSUPPORTED_NO_ENTRY = "unsupported_no_entry"
_SCOPE = "OVRTX_Materials"
_EMISSION_LUMINANCE_SCALE = 120.0 * math.pi * math.pi
EXPORT_VALUE_CONCEPTS = frozenset(
    {
        "material.base_color",
        "material.roughness",
        "material.metallic",
        "material.opacity",
        "material.transmission",
        "material.ior",
        "material.specular",
        "material.anisotropy",
        "material.coat",
        "material.subsurface",
        "material.emission",
    }
)
EXPORT_TOPOLOGY_CONCEPTS = frozenset({"material.graph"})
_SUPPORTED_PRINCIPLED_INPUTS = (
    "Base Color",
    "Roughness",
    "Metallic",
    "Alpha",
    "Transmission Weight",
    "IOR",
    "Specular IOR Level",
    "Specular Tint",
    "Anisotropic",
    "Coat Weight",
    "Coat Roughness",
    "Coat IOR",
    "Coat Tint",
)
_SUPPORTED_NODE_TYPES = {
    "OUTPUT_MATERIAL",
    "BSDF_DIFFUSE",
    "BSDF_HAIR",
    "BSDF_HAIR_PRINCIPLED",
    "BSDF_PRINCIPLED",
    "BSDF_TRANSPARENT",
    "ADD_SHADER",
    "BLACKBODY",
    "BRIGHTCONTRAST",
    "BUMP",
    "CURVE_RGB",
    "EMISSION",
    "GAMMA",
    "HUE_SAT",
    "INVERT",
    "LIGHT_PATH",
    "MATH",
    "MIX",
    "MIX_RGB",
    "MIX_SHADER",
    "NORMAL_MAP",
    "MAPPING",
    "SEPARATE_COLOR",
    "SEPRGB",
    "TEX_COORD",
    "TEX_SKY",
    "UVMAP",
    "VALTORGB",
    "RGB",
    "VALUE",
    "VECT_MATH",
}
_EPSILON = 1.0e-6
_SURFACE_NODE_TYPES = {
    "ADD_SHADER",
    "BSDF_DIFFUSE",
    "BSDF_HAIR",
    "BSDF_HAIR_PRINCIPLED",
    "BSDF_PRINCIPLED",
    "BSDF_TRANSPARENT",
    "EMISSION",
    "MIX_SHADER",
}
_TEXTURE_INPUTS = {
    "Base Color": ("base_color", "color3f", "ND_image_color3", "sRGB"),
    "Metallic": ("base_metalness", "float", "ND_image_color3", "raw"),
    "Roughness": ("specular_roughness", "float", "ND_image_color3", "raw"),
    "Alpha": ("geometry_opacity", "float", "ND_image_color3", "raw"),
    "Transmission Weight": ("transmission_weight", "float", "ND_image_color3", "raw"),
    "Emission Color": ("emission_color", "color3f", "ND_image_color3", "sRGB"),
    "Color": ("emission_color", "color3f", "ND_image_color3", "sRGB"),
}
_CHANNEL_INDEX = {
    "R": 0,
    "Red": 0,
    "X": 0,
    "G": 1,
    "Green": 1,
    "Y": 1,
    "B": 2,
    "Blue": 2,
    "Z": 2,
}
_COLORSPACE_RAW_NAMES = frozenset(
    {
        "non-color",
        "noncolor",
        "raw",
        "generic data",
        "data",
        "utility - raw",
        "linear",
        "linear rec.709",
        "linear rec.709 (srgb)",
        "linear bt.709",
        "lin_rec709",
        "linear srgb",
        "aces2065-1",
        "acescg",
        "aces - acescg",
        "linear aces",
        "xyz",
        "linear cie-xyz d65",
    }
)
_PASSIVE_DEFAULTS = {
    "Anisotropic": 0.0,
    "Anisotropic IOR Level": 0.5,
    "Anisotropic Rotation": 0.0,
    "Coat Normal": None,
    "Diffuse Roughness": 0.0,
    "Coat Tint": [1.0, 1.0, 1.0, 1.0],
    "Coat IOR": (1.45, 1.5),
    "Coat Roughness": 0.03,
    "Coat Weight": 0.0,
    "Emission Color": [1.0, 1.0, 1.0, 1.0],
    "Emission Strength": 0.0,
    "IOR": (1.45, 1.5),
    "Normal": None,
    "Sheen Roughness": 0.5,
    "Sheen Tint": [1.0, 1.0, 1.0, 1.0],
    "Sheen Weight": 0.0,
    "Specular IOR Level": 0.5,
    "Specular Tint": [1.0, 1.0, 1.0, 1.0],
    "Subsurface Anisotropy": 0.0,
    "Subsurface IOR": (1.4, 1.5),
    "Subsurface Radius": [1.0, 0.2, 0.1],
    "Subsurface Scale": 0.05,
    "Subsurface Weight": 0.0,
    "Tangent": None,
    "Thin Film IOR": 1.33,
    "Thin Film Thickness": 0.0,
    "Transmission Weight": 0.0,
    "Weight": 1.0,
}


class MaterialSceneConversionStatus(str, Enum):
    OK = "ok"
    ERROR = "error"


@dataclass(frozen=True)
class MaterialSceneConversionResult:
    status: MaterialSceneConversionStatus
    value: MaterialPresentationLayer | None = None
    error_reason: str | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status is MaterialSceneConversionStatus.OK:
            if self.error_reason is not None:
                raise ValueError("successful material scene conversion cannot have an error reason")
            if self.value is not None and not isinstance(self.value, MaterialPresentationLayer):
                raise TypeError("successful material scene conversion value must be a presentation layer")
            return
        if self.status is not MaterialSceneConversionStatus.ERROR:
            raise ValueError(f"invalid material scene conversion status: {self.status}")
        if self.value is not None:
            raise ValueError("failed material scene conversion cannot have a scene entry")
        if not isinstance(self.error_reason, str) or not self.error_reason.strip():
            raise ValueError("failed material scene conversion requires an error reason")


_MATERIALX_BINDING_IDENTITY_CACHE: dict[str, Mapping[str, Any]] = {}
_ACTIVE_INPUT_LINKS: dict[int, Any] | None = None


@dataclass(frozen=True)
class _MaterialBinding:
    """A Blender material plus explicit USD stage prim paths to rebind."""

    material: Any
    binding_targets: tuple[str, ...]
    identity: Mapping[str, Any] = field(default_factory=dict)


def _overlay_from_bindings(
    bindings: Iterable[_MaterialBinding],
) -> dict[str, Any]:
    """Build a request-ready temporary USD layer from explicit bindings."""

    used_identifiers: set[str] = set()
    material_records: list[dict[str, Any]] = []
    material_blocks: list[list[str]] = []
    binding_records: list[tuple[str, str]] = []

    for binding in bindings:
        material = binding.material
        material_name = _material_name(material)
        raw_targets = tuple(str(value) for value in binding.binding_targets)
        target_paths = _valid_target_paths(raw_targets)
        invalid_targets = tuple(target for target in raw_targets if target not in target_paths)
        result = _classify_material(material, target_paths, invalid_targets, binding.identity)
        if result["status"] != _STATUS_GENERATED:
            material_records.append(result)
            continue

        material_id = _unique_identifier(_sanitize_identifier(material_name), used_identifiers)
        material_path = f"/{_SCOPE}/{material_id}"
        values = result["openpbr_values"]
        result = {
            **result,
            "generated_material_path": material_path,
            "binding_targets": list(target_paths),
        }
        material_records.append(result)
        material_blocks.append(_material_block_lines(material_id, values))
        binding_records.extend((target_path, material_path) for target_path in target_paths)

    layer_body = _overlay_body(material_blocks, _binding_tree_lines(binding_records))
    generated_records = [
        record for record in material_records if record.get("status") == _STATUS_GENERATED
    ]
    overlay = {
        "source": _SOURCE,
        "status": _STATUS_GENERATED if generated_records else _STATUS_UNSUPPORTED_NO_ENTRY,
        "layer_body": layer_body,
        "material_count": len(material_records),
        "generated_material_paths": [
            str(record.get("generated_material_path", "")) for record in generated_records
        ],
        "binding_targets": [
            target
            for record in generated_records
            for target in list(record.get("binding_targets", ()))
        ],
        "materials": material_records,
    }
    return {
        **overlay,
        "digest": _digest_json(
            {
                "source": _SOURCE,
                "layer_body": layer_body,
            }
        ),
    }


def scene_layer_from_materials(
    materials: Iterable[Any],
    input_usd_path: str,
    *,
    allow_stock_fallback: bool = False,
) -> MaterialSceneConversionResult:
    """Convert every USD-bound source material into one OVRTX presentation layer."""

    selected = tuple(materials)
    if not selected or not input_usd_path:
        return MaterialSceneConversionResult(
            MaterialSceneConversionStatus.OK,
            diagnostics={
                "selected_materials": tuple(_material_name(material) for material in selected),
                "skipped_materials": (),
            },
        )

    identity = _materialx_binding_identity(input_usd_path)
    overlay = _scene_overlay_from_materials(selected, identity)
    material_records = tuple(overlay.get("materials", ()))
    unsupported = tuple(
        record
        for record in material_records
        if str(record.get("status", "")) != _STATUS_GENERATED
    )
    if unsupported and not allow_stock_fallback:
        reasons = []
        for record in unsupported:
            material_reasons = [
                str(reason) for reason in record.get("blocking_reasons", ()) if str(reason)
            ]
            identity = record.get("identity", {})
            if isinstance(identity, Mapping):
                identity_reason = str(identity.get("reason", "") or "")
                if identity_reason and identity_reason not in material_reasons:
                    material_reasons.append(identity_reason)
            reasons.append(
                f"{record.get('material_name', '<unknown>')}:"
                + ",".join(material_reasons)
            )
        return MaterialSceneConversionResult(
            MaterialSceneConversionStatus.ERROR,
            error_reason="material_scene_conversion_failed: " + "; ".join(reasons),
            diagnostics=overlay,
        )
    if unsupported:
        overlay = {
            **overlay,
            "stock_fallback_materials": [
                str(record.get("material_name", "")) for record in unsupported
            ],
        }
    if not material_records:
        return MaterialSceneConversionResult(
            MaterialSceneConversionStatus.OK,
            diagnostics=overlay,
        )

    layer_body = str(overlay.get("layer_body", "") or "")
    binding_targets = tuple(
        str(path) for path in overlay.get("binding_targets", ()) if str(path)
    )
    if not binding_targets:
        return MaterialSceneConversionResult(
            MaterialSceneConversionStatus.OK,
            diagnostics=overlay,
        )
    layer = MaterialPresentationLayer(
        target_path=min(binding_targets) if binding_targets else "",
        layer_body=layer_body.rstrip(),
        authored_properties=tuple(
            (path, "material:binding") for path in binding_targets
        ),
        digest_content={
            "source": _SOURCE,
            "digest": str(overlay.get("digest", "")),
            "layer_body": layer_body,
        },
        diagnostics=_layer_diagnostics(overlay),
    )
    return MaterialSceneConversionResult(
        MaterialSceneConversionStatus.OK,
        value=layer,
        diagnostics=overlay,
    )


def _scene_overlay_from_materials(
    materials: Iterable[Any],
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the automatic scene overlay for every supported material."""

    selected = tuple(materials)
    bindings: list[_MaterialBinding] = []
    skipped_materials: list[dict[str, Any]] = []
    for material in selected:
        resolution = _resolve_binding(material, identity)
        if str(resolution.get("status", "")) in {
            "missing_source_usd_path",
            "no_binding_targets",
        } and int(resolution.get("candidate_count", 0) or 0) == 0:
            skipped_materials.append(
                {
                    "material_name": _material_name(material),
                    "status": "skipped_unbound_material",
                    "reason": str(resolution.get("reason", "")),
                    "identity": dict(resolution),
                }
            )
            continue
        bindings.append(
            _MaterialBinding(
                material,
                tuple(resolution.get("raw_binding_targets", ())),
                identity=resolution,
            )
        )
    overlay = dict(_overlay_from_bindings(bindings))
    overlay["selection_policy"] = "all_supported"
    overlay["selected_materials"] = [_material_name(material) for material in selected]
    if skipped_materials:
        overlay["skipped_materials"] = skipped_materials
    return overlay


def _load_materialx_binding_identity(usd_path: Any) -> dict[str, Any]:
    """Load USD stage material binding targets for preview overlays."""

    try:
        from pxr import Usd, UsdShade  # type: ignore
    except Exception as exc:
        return {
            "available": False,
            "reason": "pxr_unavailable:" + type(exc).__name__,
            "material_paths": (),
            "bindings": (),
        }
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        return {
            "available": False,
            "reason": "stage_open_failed",
            "material_paths": (),
            "bindings": (),
        }
    return _materialx_binding_identity_from_prims(
        stage.Traverse(),
        material_type=UsdShade.Material,
    )


def _materialx_binding_identity(input_usd_path: str) -> Mapping[str, Any]:
    key = str(Path(input_usd_path).expanduser())
    identity = _MATERIALX_BINDING_IDENTITY_CACHE.get(key)
    if identity is None:
        identity = _load_materialx_binding_identity(key)
        _MATERIALX_BINDING_IDENTITY_CACHE[key] = identity
    return identity


def _materialx_binding_identity_from_prims(
    prims: Iterable[Any],
    *,
    material_type: Any | None = None,
) -> dict[str, Any]:
    material_paths: list[str] = []
    bindings: list[dict[str, str]] = []
    for prim in prims:
        path = _prim_path(prim)
        if not path:
            continue
        if _is_material_prim(prim, material_type):
            material_paths.append(path)
        material_path = _bound_material_path(prim)
        if material_path:
            bindings.append({"material_path": material_path, "binding_target": path})
    return {
        "available": True,
        "reason": "",
        "material_paths": tuple(material_paths),
        "bindings": tuple(bindings),
    }


def _resolve_binding(
    material: Any,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    material_name = _material_name(material)
    source_path = usd_paths.source_usd_path_from_blender_id(material)
    match_source = "source_usd_path" if source_path else ""
    base = {
        "status": "identity_unavailable",
        "reason": "",
        "material_name": material_name,
        "source_usd_path": source_path,
        "material_path": "",
        "binding_targets": [],
        "raw_binding_targets": [],
        "candidate_count": 0,
        "match_source": match_source,
    }
    if not identity.get("available", False):
        return {**base, "reason": str(identity.get("reason", ""))}
    if not source_path:
        source_path = _material_path_from_name(material_name, identity)
        if not source_path:
            return {**base, "status": "missing_source_usd_path", "reason": "missing_source_usd_path"}
        match_source = "material_name"
    bindings = [
        dict(binding)
        for binding in identity.get("bindings", ())
        if str(binding.get("material_path", "")) == source_path
    ]
    raw_targets = tuple(str(binding.get("binding_target", "")) for binding in bindings)
    targets = _valid_target_paths(raw_targets)
    if not targets:
        return {
            **base,
            "status": "no_binding_targets",
            "reason": "no_usd_binding_targets",
            "material_path": source_path,
            "raw_binding_targets": list(raw_targets),
            "candidate_count": len(bindings),
            "source_usd_path": source_path,
            "match_source": match_source,
        }
    return {
        **base,
        "status": "resolved",
        "reason": "",
        "material_path": source_path,
        "source_usd_path": source_path,
        "binding_targets": list(targets),
        "raw_binding_targets": list(raw_targets),
        "candidate_count": len(targets),
        "match_source": match_source,
    }


def _material_path_from_name(material_name: str, identity: Mapping[str, Any]) -> str:
    expected = _sanitize_identifier(material_name)
    matches = [
        str(path)
        for path in identity.get("material_paths", ())
        if str(path).rstrip("/").rsplit("/", 1)[-1] == expected
    ]
    return matches[0] if len(matches) == 1 else ""


def _classify_material(
    material: Any,
    binding_targets: tuple[str, ...],
    invalid_binding_targets: tuple[str, ...],
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    material_name = _material_name(material)
    base = {
        "material_name": material_name,
        "source_usd_path": usd_paths.source_usd_path_from_blender_id(material),
        "identity": dict(identity),
        "status": _STATUS_UNSUPPORTED_NO_ENTRY,
        "blocking_reasons": [],
        "node_inventory": [],
        "openpbr_values": {},
        "generated_material_path": "",
        "binding_targets": list(binding_targets),
        "invalid_binding_targets": list(invalid_binding_targets),
    }
    binding_reasons = []
    if invalid_binding_targets:
        binding_reasons.append("invalid_binding_targets")
    if not binding_targets:
        binding_reasons.append("missing_binding_targets")
    if binding_reasons:
        return {**base, "blocking_reasons": binding_reasons}

    node_tree = getattr(material, "node_tree", None)
    nodes = list(getattr(node_tree, "nodes", ()) or ())
    if node_tree is None or not nodes:
        return {**base, "blocking_reasons": ["missing_node_tree"]}

    global _ACTIVE_INPUT_LINKS
    previous_input_links = _ACTIVE_INPUT_LINKS
    _ACTIVE_INPUT_LINKS = _node_tree_input_links(node_tree)
    try:
        active_output = _active_output_node(nodes)
        if active_output is None:
            return {
                **base,
                "node_inventory": [_node_diagnostics(node, None) for node in nodes],
                "blocking_reasons": ["missing_active_material_output"],
            }

        surface_node = _output_surface_node(active_output)
        active_node_ids = {
            _blender_identity(node)
            for node in (*_surface_graph_nodes(surface_node), active_output)
        }
        inventory = [
            _node_diagnostics(node, surface_node)
            for node in nodes
            if _blender_identity(node) in active_node_ids
        ]
        blocking_reasons = _blocking_reasons(inventory)
        if blocking_reasons:
            return {**base, "node_inventory": inventory, "blocking_reasons": blocking_reasons}
        values = _openpbr_values_from_surface(surface_node)
        if values is None:
            return {
                **base,
                "node_inventory": inventory,
                "blocking_reasons": [f"unsupported_surface:{_node_type(surface_node) or 'missing'}"],
            }
        return {
            **base,
            "status": _STATUS_GENERATED,
            "node_inventory": inventory,
            "openpbr_values": values,
        }
    finally:
        _ACTIVE_INPUT_LINKS = previous_input_links


def _openpbr_values_from_surface(surface_node: Any | None) -> dict[str, Any] | None:
    node_type = _node_type(surface_node)
    if node_type == "BSDF_PRINCIPLED":
        return _principled_openpbr_values(surface_node)
    if node_type == "BSDF_DIFFUSE":
        return _diffuse_openpbr_values(surface_node)
    if node_type in {"BSDF_HAIR", "BSDF_HAIR_PRINCIPLED"}:
        return _hair_openpbr_values(surface_node)
    if node_type == "BSDF_TRANSPARENT":
        return _transparent_openpbr_values(surface_node)
    if node_type == "EMISSION":
        return _emission_openpbr_values(surface_node)
    if node_type == "ADD_SHADER":
        return _add_shader_openpbr_values(surface_node)
    if node_type == "MIX_SHADER":
        return _mix_shader_openpbr_values(surface_node)
    return None


def _principled_openpbr_values(node: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "base_color": _color3_input(node, "Base Color", (0.8, 0.8, 0.8)),
        "specular_roughness": _float_input(node, "Roughness", 0.5),
        "base_metalness": _float_input(node, "Metallic", 0.0),
        "geometry_opacity": _float_input(node, "Alpha", 1.0),
        "textures": {},
    }
    textures = values["textures"]
    for socket_name in ("Base Color", "Metallic", "Roughness", "Alpha"):
        record = _texture_record_from_socket(node, socket_name)
        if record:
            textures[record["openpbr_input"]] = record
    transmission_weight = _float_input(node, "Transmission Weight", 0.0)
    transmission_texture = _texture_record_from_socket(node, "Transmission Weight")
    if transmission_texture or transmission_weight > 1.0e-6:
        values["transmission_weight"] = transmission_weight
        if transmission_texture:
            textures["transmission_weight"] = transmission_texture
        elif any(abs(channel - 1.0) > 1.0e-6 for channel in values["base_color"]):
            values["transmission_color"] = values["base_color"]
    _apply_coat_values(node, values)
    _apply_subsurface_values(node, values)
    _apply_specular_values(node, values)
    normal = _normal_texture_record(node)
    if normal:
        textures["geometry_normal"] = normal
    emission_strength = _float_input(node, "Emission Strength", 0.0)
    emission_texture = _texture_record_from_socket(node, "Emission Color")
    emission_color = _color3_input(node, "Emission Color", (1.0, 1.0, 1.0))
    emission_color_linked = _socket_linked(_input_socket(node, "Emission Color"))
    if emission_color_linked and not emission_texture:
        static_color = _linked_static_color(node, "Emission Color")
        if static_color is not None:
            emission_color = static_color
        else:
            emission_strength = 0.0
    if emission_strength > 1.0e-6 and (emission_texture or any(channel > 1.0e-6 for channel in emission_color)):
        values["emission_luminance"] = emission_strength * _EMISSION_LUMINANCE_SCALE
        values["emission_color"] = emission_color
        if emission_texture:
            textures["emission_color"] = emission_texture
    return values


def _apply_coat_values(node: Any, values: dict[str, Any]) -> None:
    weight = _float_input(node, "Coat Weight", 0.0)
    if weight <= _EPSILON:
        return
    values["coat_weight"] = weight
    values["coat_roughness"] = _float_input(node, "Coat Roughness", 0.03)
    values["coat_ior"] = _float_input(node, "Coat IOR", 1.5)
    tint = _color3_input(node, "Coat Tint", (1.0, 1.0, 1.0))
    if any(abs(channel - 1.0) > _EPSILON for channel in tint):
        values["coat_color"] = tint


def _apply_subsurface_values(node: Any, values: dict[str, Any]) -> None:
    weight = _float_input(node, "Subsurface Weight", 0.0)
    if weight <= _EPSILON:
        return
    radius = _color3_input(node, "Subsurface Radius", (1.0, 0.2, 0.1))
    scale = _float_input(node, "Subsurface Scale", 0.05)
    effective_radius = [max(0.0, channel * scale) for channel in radius]
    max_radius = max(effective_radius)
    if max_radius <= _EPSILON:
        return
    values["subsurface_weight"] = weight
    values["subsurface_color"] = values["base_color"]
    values["subsurface_radius"] = max_radius
    values["subsurface_radius_scale"] = tuple(channel / max_radius for channel in effective_radius)
    anisotropy = _float_input(node, "Subsurface Anisotropy", 0.0)
    if abs(anisotropy) > _EPSILON:
        values["subsurface_scatter_anisotropy"] = anisotropy


def _apply_specular_values(node: Any, values: dict[str, Any]) -> None:
    ior = _float_input(node, "IOR", 1.45)
    if abs(ior - 1.5) > _EPSILON:
        values["specular_ior"] = ior
    textures = values.get("textures", {})
    has_metalness_texture = isinstance(textures, Mapping) and "base_metalness" in textures
    metalness = float(values.get("base_metalness", 0.0))
    if metalness > _EPSILON and not has_metalness_texture:
        weight = 1.0
    else:
        weight = 2.0 * _float_input(node, "Specular IOR Level", 0.5)
    if abs(weight - 1.0) > _EPSILON:
        values["specular_weight"] = weight
    tint = _color3_input(node, "Specular Tint", (1.0, 1.0, 1.0))
    if weight > _EPSILON and any(abs(channel - 1.0) > _EPSILON for channel in tint):
        values["specular_color"] = tint
    anisotropy = _float_input(node, "Anisotropic", 0.0)
    if abs(anisotropy) > _EPSILON:
        values["specular_roughness_anisotropy"] = anisotropy


def _emission_openpbr_values(node: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "base_color": (0.0, 0.0, 0.0),
        "specular_roughness": 1.0,
        "base_metalness": 0.0,
        "geometry_opacity": 1.0,
        "specular_weight": 0.0,
        "emission_luminance": _float_input(node, "Strength", 1.0) * _EMISSION_LUMINANCE_SCALE,
        "emission_color": _color3_input(node, "Color", (1.0, 1.0, 1.0)),
        "textures": {},
    }
    emission_texture = _texture_record_from_socket(node, "Color")
    if emission_texture:
        values["textures"]["emission_color"] = emission_texture
    elif _socket_linked(_input_socket(node, "Color")):
        static_color = _linked_static_color(node, "Color")
        if static_color is not None:
            values["emission_color"] = static_color
        else:
            values["emission_luminance"] = 0.0
    return values


def _diffuse_openpbr_values(node: Any) -> dict[str, Any]:
    return {
        "base_color": _color3_input(node, "Color", (0.8, 0.8, 0.8)),
        "specular_roughness": 0.9,
        "base_metalness": 0.0,
        "geometry_opacity": 1.0,
        "specular_weight": 0.0,
        "textures": {},
    }


def _hair_openpbr_values(node: Any) -> dict[str, Any]:
    return {
        "base_color": _color3_input(node, "Color", (0.5, 0.3, 0.2)),
        "specular_roughness": 0.7,
        "base_metalness": 0.0,
        "geometry_opacity": 1.0,
        "specular_weight": 0.0,
        "textures": {},
    }


def _transparent_openpbr_values(node: Any) -> dict[str, Any]:
    return {
        "base_color": _color3_input(node, "Color", (1.0, 1.0, 1.0)),
        "specular_roughness": 0.5,
        "base_metalness": 0.0,
        "geometry_opacity": 0.0,
        "specular_weight": 0.0,
        "textures": {},
    }


def _add_shader_openpbr_values(node: Any) -> dict[str, Any] | None:
    branches = _shader_input_nodes(node)
    emission_node = next((branch for branch in branches if _node_type(branch) == "EMISSION"), None)
    other_node = next((branch for branch in branches if branch is not None and branch is not emission_node), None)
    if emission_node is None:
        return _openpbr_values_from_surface(other_node)
    emission_values = _emission_openpbr_values(emission_node)
    values = _openpbr_values_from_surface(other_node) if other_node is not None else None
    if values is None:
        return {**emission_values, "textures": dict(emission_values.get("textures", {}))}
    values = {**values, "textures": dict(values.get("textures", {}))}
    values["emission_luminance"] = emission_values["emission_luminance"]
    values["emission_color"] = emission_values["emission_color"]
    emission_texture = emission_values.get("textures", {}).get("emission_color")
    if emission_texture:
        values["textures"]["emission_color"] = emission_texture
    return values


def _mix_shader_openpbr_values(node: Any, *, depth: int = 4) -> dict[str, Any] | None:
    if depth <= 0:
        return None
    branches = list(_shader_input_nodes(node))
    if not branches:
        return None
    transparent_index = next(
        (index for index, branch in enumerate(branches) if _node_type(branch) == "BSDF_TRANSPARENT"),
        None,
    )
    if transparent_index is not None and len(branches) >= 2:
        other_index = 1 - transparent_index if transparent_index in {0, 1} else 0
        other = branches[other_index] if other_index < len(branches) else None
        values = _openpbr_values_from_mixed_branch(other, depth=depth - 1)
        if values is None:
            return None
        values = {**values, "textures": dict(values.get("textures", {}))}
        factor_socket = _input_socket(node, "Factor") or _input_socket(node, "Fac")
        opacity_texture = _opacity_texture_record_from_socket(factor_socket)
        invert = transparent_index == 1
        if opacity_texture:
            opacity_texture = {**opacity_texture, "invert": bool(opacity_texture.get("invert", False)) != invert}
            values["textures"]["geometry_opacity"] = opacity_texture
        else:
            factor = _float_input_any(node, ("Factor", "Fac"), 0.5)
            values["geometry_opacity"] = (1.0 - factor) if invert else factor
        return values
    for branch in branches:
        values = _openpbr_values_from_mixed_branch(branch, depth=depth - 1)
        if values is not None:
            return values
    return None


def _openpbr_values_from_mixed_branch(node: Any | None, *, depth: int) -> dict[str, Any] | None:
    if node is None:
        return None
    if _node_type(node) == "MIX_SHADER":
        return _mix_shader_openpbr_values(node, depth=depth)
    return _openpbr_values_from_surface(node)


def _node_diagnostics(node: Any, surface_node: Any | None) -> dict[str, Any]:
    node_type = _node_type(node)
    if node_type == "TEX_IMAGE":
        classification = "supported" if _image_asset_path(getattr(node, "image", None)) else "unsupported"
    else:
        classification = "supported" if node_type in _SUPPORTED_NODE_TYPES else "unsupported"
    socket_records = [_socket_diagnostics(node, socket, surface_node) for socket in _input_sockets(node)]
    if any(record["classification"] in {"unsupported", "unknown"} for record in socket_records):
        classification = "unsupported"
    return {
        "name": str(getattr(node, "name", "")),
        "type": node_type,
        "classification": classification,
        "inputs": socket_records,
    }


def _socket_diagnostics(node: Any, socket: Any, surface_node: Any | None) -> dict[str, Any]:
    node_type = _node_type(node)
    socket_name = str(getattr(socket, "name", ""))
    linked = _socket_linked(socket)
    default_value = _plain_value(getattr(socket, "default_value", None))
    target_node = _socket_link_target_node(socket)
    record = {
        "name": socket_name,
        "linked": linked,
        "default_value": default_value,
        "classification": "fallback",
        "reason": "",
    }
    if node_type == "OUTPUT_MATERIAL" and socket_name == "Surface":
        if target_node is surface_node and surface_node is not None:
            return {**record, "classification": "supported"}
        if _node_type(target_node) in _SURFACE_NODE_TYPES:
            return {**record, "classification": "supported"}
        if target_node is None or not bool(getattr(node, "is_active_output", True)):
            return {**record, "classification": "fallback", "reason": "passive_default"}
        return {**record, "classification": "unsupported", "reason": "unsupported_surface"}
    if node_type == "ADD_SHADER":
        if str(getattr(socket, "type", "")) == "SHADER" and linked:
            return {**record, "classification": "supported"}
        return record
    if node_type == "EMISSION":
        if socket_name == "Color":
            if linked:
                if (
                    _texture_record_from_socket(node, socket_name)
                    or _linked_static_color(node, socket_name) is not None
                ):
                    return {**record, "classification": "supported"}
                return {**record, "classification": "unsupported", "reason": "linked_supported_input:Color"}
            return {**record, "classification": "supported"}
        if socket_name == "Strength":
            return {**record, "classification": "supported"}
        return record
    if node_type == "NORMAL_MAP":
        if socket_name == "Color":
            if linked and _texture_record_from_socket(node, socket_name):
                return {**record, "classification": "supported"}
            if linked:
                return {**record, "classification": "unsupported", "reason": "linked_supported_input:Normal"}
        if socket_name == "Strength":
            return {**record, "classification": "supported"}
        return record
    if node_type != "BSDF_PRINCIPLED":
        return record
    if socket_name in _SUPPORTED_PRINCIPLED_INPUTS:
        if linked:
            if _texture_record_from_socket(node, socket_name):
                return {**record, "classification": "supported"}
            return {
                **record,
                "classification": "unsupported",
                "reason": f"linked_supported_input:{socket_name}",
            }
        return {**record, "classification": "supported"}
    if socket_name == "Normal":
        if linked:
            if _normal_texture_record(node):
                return {**record, "classification": "supported"}
            return {
                **record,
                "classification": "unsupported",
                "reason": "linked_supported_input:Normal",
            }
        if _passive_default(socket_name, default_value):
            return {**record, "classification": "fallback", "reason": "passive_default"}
        return {**record, "classification": "supported"}
    if socket_name == "Emission Color":
        if linked:
            if (
                _texture_record_from_socket(node, socket_name)
                or _linked_static_color(node, socket_name) is not None
            ):
                return {**record, "classification": "supported"}
            return {
                **record,
                "classification": "unsupported",
                "reason": f"linked_supported_input:{socket_name}",
            }
        if _passive_default(socket_name, default_value):
            return {**record, "classification": "fallback", "reason": "passive_default"}
        return {**record, "classification": "supported"}
    if socket_name == "Emission Strength":
        if linked:
            return {
                **record,
                "classification": "unsupported",
                "reason": f"linked_unsupported_input:{socket_name}",
            }
        emission_color = _color3_input(node, "Emission Color", (1.0, 1.0, 1.0))
        if _float_input(node, socket_name, 0.0) <= 1.0e-6 or max(emission_color) <= 1.0e-6:
            return {**record, "classification": "fallback", "reason": "passive_default"}
        return {**record, "classification": "supported"}
    if socket_name == "Weight":
        return {**record, "classification": "fallback", "reason": "passive_default"}
    if socket_name.startswith("Subsurface "):
        if _float_input(node, "Subsurface Weight", 0.0) <= _EPSILON:
            return {**record, "classification": "fallback", "reason": "passive_default"}
        if socket_name == "Subsurface Scale":
            return {**record, "classification": "supported"}
        if socket_name in {
            "Subsurface Weight",
            "Subsurface Radius",
            "Subsurface Anisotropy",
        }:
            if not linked:
                return {**record, "classification": "supported"}
            return {
                **record,
                "classification": "unsupported",
                "reason": f"linked_supported_input:{socket_name}",
            }
        if _passive_default(socket_name, default_value):
            return {**record, "classification": "fallback", "reason": "passive_default"}
    if linked:
        return {
            **record,
            "classification": "unsupported",
            "reason": f"linked_unsupported_input:{socket_name}",
        }
    if _passive_default(socket_name, default_value):
        return {**record, "classification": "fallback", "reason": "passive_default"}
    return {
        **record,
        "classification": "unsupported",
        "reason": f"unsupported_non_default_input:{socket_name}",
    }


def _blocking_reasons(node_inventory: Iterable[Mapping[str, Any]]) -> list[str]:
    reasons: list[str] = []
    for node in node_inventory:
        if node.get("classification") == "unsupported" and node.get("type") not in _SUPPORTED_NODE_TYPES:
            reasons.append(f"unsupported_node:{node.get('type', '')}")
        for socket in node.get("inputs", ()):
            reason = str(socket.get("reason", ""))
            if socket.get("classification") in {"unsupported", "unknown"} and reason:
                reasons.append(reason)
    return sorted(set(reasons))


def _overlay_body(material_blocks: list[list[str]], binding_lines: list[str]) -> str:
    if not material_blocks:
        return ""
    lines = [f'def Scope "{_SCOPE}"', "{"]
    for index, block in enumerate(material_blocks):
        if index:
            lines.append("")
        lines.extend("    " + line if line else "" for line in block)
    lines.append("}")
    if binding_lines:
        lines.append("")
        lines.extend(binding_lines)
    lines.append("")
    return "\n".join(lines)


def _material_block_lines(material_id: str, values: Mapping[str, Any]) -> list[str]:
    material_path = f"/{_SCOPE}/{material_id}"
    shader_path = f"{material_path}/ND_open_pbr_surface_surfaceshader"
    color = values["base_color"]
    textures = _textures_with_output_paths(material_path, values.get("textures", {}))
    lines = [
        f'def Material "{_usda_string(material_id)}"',
        "{",
        f"    token outputs:mtlx:surface.connect = <{shader_path}.outputs:out>",
        "",
        '    def Shader "ND_open_pbr_surface_surfaceshader" (',
        '        prepend apiSchemas = ["NodeGraphNodeAPI"]',
        "    )",
        "    {",
        '        uniform token info:id = "ND_open_pbr_surface_surfaceshader"',
        f"        color3f inputs:base_color = {_usda_color3(color)}",
    ]
    if "base_color" in textures:
        lines.append(_input_connect_line("color3f", "base_color", textures["base_color"]))
    lines.append(f"        float inputs:base_metalness = {_usda_float(values['base_metalness'])}")
    if "base_metalness" in textures:
        lines.append(_input_connect_line("float", "base_metalness", textures["base_metalness"]))
    lines.append(f"        float inputs:geometry_opacity = {_usda_float(values['geometry_opacity'])}")
    if "geometry_opacity" in textures:
        lines.append(_input_connect_line("float", "geometry_opacity", textures["geometry_opacity"]))
    lines.append(f"        float inputs:specular_roughness = {_usda_float(values['specular_roughness'])}")
    if "specular_roughness" in textures:
        lines.append(_input_connect_line("float", "specular_roughness", textures["specular_roughness"]))
    if "geometry_normal" in textures:
        lines.append(_input_connect_line("float3", "geometry_normal", textures["geometry_normal"]))
    if "transmission_weight" in values:
        lines.append(f"        float inputs:transmission_weight = {_usda_float(float(values['transmission_weight']))}")
    if "transmission_weight" in textures:
        lines.append(_input_connect_line("float", "transmission_weight", textures["transmission_weight"]))
    if "transmission_color" in values:
        lines.append(f"        color3f inputs:transmission_color = {_usda_color3(values['transmission_color'])}")
    for input_name in (
        "specular_weight",
        "specular_ior",
        "specular_roughness_anisotropy",
        "coat_weight",
        "coat_roughness",
        "coat_ior",
        "subsurface_weight",
        "subsurface_radius",
        "subsurface_scatter_anisotropy",
    ):
        if input_name in values:
            lines.append(f"        float inputs:{input_name} = {_usda_float(float(values[input_name]))}")
    for input_name in ("specular_color", "coat_color", "subsurface_color", "subsurface_radius_scale"):
        if input_name in values:
            lines.append(f"        color3f inputs:{input_name} = {_usda_color3(values[input_name])}")
    if "subsurface_color" in values and "base_color" in textures:
        lines.append(_input_connect_line("color3f", "subsurface_color", textures["base_color"]))
    if "emission_luminance" in values:
        lines.append(f"        float inputs:emission_luminance = {_usda_float(float(values['emission_luminance']))}")
    if "emission_color" in values:
        lines.append(f"        color3f inputs:emission_color = {_usda_color3(values['emission_color'])}")
    if "emission_color" in textures:
        lines.append(_input_connect_line("color3f", "emission_color", textures["emission_color"]))
    lines.extend(["        token outputs:out", "    }"])
    lines.extend(_texture_shader_lines(material_path, textures))
    lines.append("}")
    return lines


def _textures_with_output_paths(
    material_path: str,
    textures: Any,
) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    if not isinstance(textures, Mapping):
        return normalized
    for input_name, texture in textures.items():
        if not isinstance(texture, Mapping):
            continue
        record = dict(texture)
        if str(input_name) == "geometry_normal":
            record["output_path"] = f"{material_path}/ND_normalmap_float_geometry_normal.outputs:out"
        elif record.get("extract_channel") is not None:
            record["extract_shader_name"] = f"ND_extract_color3_{input_name}"
            if record.get("invert"):
                record["invert_shader_name"] = f"ND_invert_float_{input_name}"
                record["output_path"] = f"{material_path}/{record['invert_shader_name']}.outputs:out"
            elif record.get("post_image_op"):
                op = dict(record["post_image_op"])
                op["shader_name"] = f"{op['info_id']}_{input_name}"
                record["post_image_op"] = op
                record["output_path"] = f"{material_path}/{op['shader_name']}.outputs:out"
            else:
                record["output_path"] = f"{material_path}/{record['extract_shader_name']}.outputs:out"
        elif record.get("post_image_op"):
            op = dict(record["post_image_op"])
            op["shader_name"] = f"{op['info_id']}_{input_name}"
            record["post_image_op"] = op
            record["output_path"] = f"{material_path}/{op['shader_name']}.outputs:out"
        else:
            record["output_path"] = f"{material_path}/{record['shader_name']}.outputs:out"
        normalized[str(input_name)] = record
    return normalized


def _input_connect_line(value_type: str, input_name: str, texture: Mapping[str, Any]) -> str:
    return f"        {value_type} inputs:{input_name}.connect = <{texture['output_path']}>"


def _texture_shader_lines(material_path: str, textures: Mapping[str, Mapping[str, Any]]) -> list[str]:
    lines: list[str] = []
    for input_name in sorted(textures):
        texture = textures[input_name]
        shader_name = str(texture["shader_name"])
        info_id = str(texture["info_id"])
        lines.extend(
            [
                "",
                f'    def Shader "{_usda_string(shader_name)}" (',
                '        prepend apiSchemas = ["NodeGraphNodeAPI"]',
                "    )",
                "    {",
                f'        uniform token info:id = "{info_id}"',
                f"        asset inputs:file = @{_usda_asset_path(str(texture['asset_path']))}@ (",
                f'            colorSpace = "{str(texture["color_space"])}"',
                "        )",
                f"        {texture['image_output_type']} outputs:out",
                "    }",
            ]
        )
        if texture.get("extract_channel") is not None:
            extract_path = f"{material_path}/{texture['extract_shader_name']}.outputs:out"
            lines.extend(
                [
                    "",
                    f'    def Shader "{_usda_string(str(texture["extract_shader_name"]))}" (',
                    '        prepend apiSchemas = ["NodeGraphNodeAPI"]',
                    "    )",
                    "    {",
                    '        uniform token info:id = "ND_extract_color3"',
                    f"        color3f inputs:in.connect = <{material_path}/{shader_name}.outputs:out>",
                    f"        int inputs:index = {int(texture['extract_channel'])}",
                    "        float outputs:out",
                    "    }",
                ]
            )
            if texture.get("invert"):
                lines.extend(
                    [
                        "",
                        f'    def Shader "{_usda_string(str(texture["invert_shader_name"]))}" (',
                        '        prepend apiSchemas = ["NodeGraphNodeAPI"]',
                        "    )",
                        "    {",
                        '        uniform token info:id = "ND_invert_float"',
                        f"        float inputs:in.connect = <{extract_path}>",
                        "        float outputs:out",
                        "    }",
                    ]
                )
            elif texture.get("post_image_op"):
                lines.extend(_post_image_op_shader_lines(material_path, texture, extract_path))
        elif texture.get("post_image_op"):
            image_path = f"{material_path}/{shader_name}.outputs:out"
            lines.extend(_post_image_op_shader_lines(material_path, texture, image_path))
        if input_name == "geometry_normal":
            if texture.get("bump"):
                height_to_normal_path = (
                    f"{material_path}/ND_heighttonormal_vector3_geometry_normal.outputs:out"
                )
                lines.extend(
                    [
                        "",
                        '    def Shader "ND_heighttonormal_vector3_geometry_normal" (',
                        '        prepend apiSchemas = ["NodeGraphNodeAPI"]',
                        "    )",
                        "    {",
                        '        uniform token info:id = "ND_heighttonormal_vector3"',
                        f"        float inputs:in.connect = <{material_path}/{shader_name}.outputs:out>",
                        f"        float inputs:scale = {_usda_float(float(texture.get('scale', 1.0)))}",
                        "        float3 outputs:out",
                        "    }",
                        "",
                        '    def Shader "ND_normalmap_float_geometry_normal" (',
                        '        prepend apiSchemas = ["NodeGraphNodeAPI"]',
                        "    )",
                        "    {",
                        '        uniform token info:id = "ND_normalmap_float"',
                        f"        float3 inputs:in.connect = <{height_to_normal_path}>",
                        "        float3 outputs:out",
                        "    }",
                    ]
                )
            else:
                convert_path = f"{material_path}/ND_convert_color3_vector3_geometry_normal.outputs:out"
                lines.extend(
                    [
                        "",
                        '    def Shader "ND_convert_color3_vector3_geometry_normal" (',
                        '        prepend apiSchemas = ["NodeGraphNodeAPI"]',
                        "    )",
                        "    {",
                        '        uniform token info:id = "ND_convert_color3_vector3"',
                        f"        color3f inputs:in.connect = <{material_path}/{shader_name}.outputs:out>",
                        "        float3 outputs:out",
                        "    }",
                        "",
                        '    def Shader "ND_normalmap_float_geometry_normal" (',
                        '        prepend apiSchemas = ["NodeGraphNodeAPI"]',
                        "    )",
                        "    {",
                        '        uniform token info:id = "ND_normalmap_float"',
                        f"        float3 inputs:in.connect = <{convert_path}>",
                        f"        float inputs:scale = {_usda_float(float(texture.get('scale', 1.0)))}",
                        "        float3 outputs:out",
                        "    }",
                    ]
                )
    return lines


def _post_image_op_shader_lines(
    material_path: str,
    texture: Mapping[str, Any],
    input_path: str,
) -> list[str]:
    op = texture.get("post_image_op")
    if not isinstance(op, Mapping):
        return []
    value_type = str(op["value_type"])
    shader_name = str(op["shader_name"])
    info_id = str(op["info_id"])
    return [
        "",
        f'    def Shader "{_usda_string(shader_name)}" (',
        '        prepend apiSchemas = ["NodeGraphNodeAPI"]',
        "    )",
        "    {",
        f'        uniform token info:id = "{info_id}"',
        f"        {value_type} inputs:in1.connect = <{input_path}>",
        f"        float inputs:in2 = {_usda_float(float(op['in2']))}",
        f"        {value_type} outputs:out",
        "    }",
    ]


def _binding_tree_lines(bindings: Iterable[tuple[str, str]]) -> list[str]:
    root: dict[str, Any] = {}
    for target_path, material_path in bindings:
        node = root
        for part in usd_paths.path_parts(target_path):
            node = node.setdefault(part, {})
        node["__material_path__"] = material_path
    lines: list[str] = []
    _append_binding_tree_lines(lines, root, depth=0)
    return lines


def _append_binding_tree_lines(lines: list[str], node: Mapping[str, Any], *, depth: int) -> None:
    indent = "    " * depth
    material_path = str(node.get("__material_path__", ""))
    if material_path:
        lines.append(f"{indent}rel material:binding = <{material_path}>")
    for part, child in node.items():
        if part == "__material_path__":
            continue
        lines.append(f'{indent}over "{_usda_string(str(part))}"')
        lines.append(f"{indent}{{")
        _append_binding_tree_lines(lines, child, depth=depth + 1)
        lines.append(f"{indent}}}")


def _active_material_surface(material: Any) -> Any | None:
    node_tree = getattr(material, "node_tree", None)
    nodes = list(getattr(node_tree, "nodes", ()) or ())
    active_output = _active_output_node(nodes)
    return _output_surface_node(active_output) if active_output is not None else None


def _surface_graph_nodes(node: Any | None, seen: set[int] | None = None) -> tuple[Any, ...]:
    if node is None:
        return ()
    if seen is None:
        seen = set()
    node_id = _blender_identity(node)
    if node_id in seen:
        return ()
    seen.add(node_id)
    nodes = [node]
    for socket in _input_sockets(node):
        linked = _socket_link_target_node(socket)
        if linked is not None:
            nodes.extend(_surface_graph_nodes(linked, seen))
    return tuple(nodes)


def _active_output_node(nodes: Iterable[Any]) -> Any | None:
    for node in nodes:
        if _node_type(node) == "OUTPUT_MATERIAL" and bool(getattr(node, "is_active_output", True)):
            return node
    return None


def _output_surface_node(output_node: Any) -> Any | None:
    socket = _input_socket(output_node, "Surface")
    return _socket_link_target_node(socket)


def _input_socket(node: Any, name: str) -> Any | None:
    inputs = getattr(node, "inputs", None)
    getter = getattr(inputs, "get", None)
    if callable(getter):
        return getter(name)
    if isinstance(inputs, Mapping):
        return inputs.get(name)
    for socket in inputs or ():
        if str(getattr(socket, "name", "")) == name:
            return socket
    return None


def _input_sockets(node: Any) -> list[Any]:
    inputs = getattr(node, "inputs", None)
    if isinstance(inputs, Mapping):
        return list(inputs.values())
    return list(inputs or ())


def _shader_input_nodes(node: Any) -> tuple[Any | None, ...]:
    return tuple(
        _socket_link_target_node(socket)
        for socket in _input_sockets(node)
        if str(getattr(socket, "type", "")) == "SHADER" or str(getattr(socket, "name", "")) == "Shader"
    )


def _socket_linked(socket: Any | None) -> bool:
    if socket is None:
        return False
    is_linked = getattr(socket, "is_linked", None)
    if is_linked is not None:
        return bool(is_linked)
    return bool(getattr(socket, "links", ()))


def _has_linked_input(node: Any, name: str) -> bool:
    return _socket_linked(_input_socket(node, name))


def _socket_link_target_node(socket: Any | None) -> Any | None:
    if socket is None:
        return None
    if getattr(socket, "is_linked", None) is False:
        return None
    if _ACTIVE_INPUT_LINKS is not None:
        link = _ACTIVE_INPUT_LINKS.get(_blender_identity(socket))
        return getattr(link, "from_node", None)
    links = list(getattr(socket, "links", ()) or ())
    if not links:
        return None
    return getattr(links[0], "from_node", None)


def _blender_identity(value: Any) -> int:
    as_pointer = getattr(value, "as_pointer", None)
    return int(as_pointer()) if callable(as_pointer) else id(value)


def _node_tree_input_links(node_tree: Any) -> dict[int, Any] | None:
    links = getattr(node_tree, "links", None)
    if links is None:
        return None
    return {
        _blender_identity(link.to_socket): link
        for link in links
        if getattr(link, "to_socket", None) is not None
    }


def _node_type(node: Any | None) -> str:
    return str(getattr(node, "type", ""))


def _texture_record_from_socket(node: Any, socket_name: str) -> dict[str, Any] | None:
    slot = _TEXTURE_INPUTS.get(socket_name)
    if slot is None:
        return None
    openpbr_input, value_type, info_id, fallback_color_space = slot
    socket = _input_socket(node, socket_name)
    texture_node, inverted = _linked_texture_info(socket)
    if texture_node is None:
        return None
    asset_path = _image_asset_path(getattr(texture_node, "image", None))
    if not asset_path:
        return None
    shader_name = f"{info_id}_{openpbr_input}"
    image_output_type = "color3f" if info_id == "ND_image_color3" else value_type
    return {
        "openpbr_input": openpbr_input,
        "shader_name": shader_name,
        "info_id": info_id,
        "asset_path": asset_path,
        "color_space": _image_color_space(getattr(texture_node, "image", None), fallback_color_space),
        "value_type": value_type,
        "image_output_type": image_output_type,
        "extract_channel": _scalar_texture_channel(socket)
        if value_type == "float" and image_output_type == "color3f"
        else None,
        "invert": inverted
        if openpbr_input in {"geometry_opacity", "transmission_weight"}
        else False,
        "post_image_op": _post_image_op_record(node, socket_name, value_type),
    }


def _post_image_op_record(node: Any, socket_name: str, value_type: str) -> dict[str, Any] | None:
    source = _socket_link_target_node(_input_socket(node, socket_name))
    if source is None:
        return None
    if value_type == "color3f":
        if _node_type(source) != "GAMMA":
            return None
        color_source = _socket_link_target_node(_input_socket(source, "Color"))
        if _node_type(color_source) != "TEX_IMAGE":
            return None
        gamma = _float_input(source, "Gamma", 1.0)
        if abs(gamma - 1.0) <= _EPSILON:
            return None
        return {
            "info_id": "ND_power_color3FA",
            "value_type": "color3f",
            "in2": gamma,
        }
    if value_type != "float" or _node_type(source) != "MATH":
        return None
    operation = str(getattr(source, "operation", ""))
    if operation not in {"MULTIPLY", "ADD"}:
        return None
    const_value: float | None = None
    linked_to_texture = False
    for socket in _input_sockets(source)[:2]:
        linked_source = _socket_link_target_node(socket)
        if linked_source is not None:
            if _find_texture_node(linked_source) is not None:
                linked_to_texture = True
            continue
        try:
            const_value = float(getattr(socket, "default_value", 0.0))
        except (TypeError, ValueError):
            pass
    if not linked_to_texture or const_value is None:
        return None
    return {
        "info_id": "ND_multiply_float" if operation == "MULTIPLY" else "ND_add_float",
        "value_type": "float",
        "in2": const_value,
    }


def _opacity_texture_record_from_socket(socket: Any | None) -> dict[str, Any] | None:
    texture_node, inverted = _linked_texture_info(socket)
    if texture_node is None:
        return None
    asset_path = _image_asset_path(getattr(texture_node, "image", None))
    if not asset_path:
        return None
    return {
        "openpbr_input": "geometry_opacity",
        "shader_name": "ND_image_color3_geometry_opacity",
        "info_id": "ND_image_color3",
        "asset_path": asset_path,
        "color_space": _image_color_space(getattr(texture_node, "image", None), "raw"),
        "value_type": "float",
        "image_output_type": "color3f",
        "extract_channel": _scalar_texture_channel(socket),
        "invert": inverted,
    }


def _normal_texture_record(node: Any) -> dict[str, Any] | None:
    socket = _input_socket(node, "Normal")
    source = _socket_link_target_node(socket)
    if source is None:
        return None
    source_type = _node_type(source)
    bump = False
    scale = 1.0
    texture_socket = socket
    if source_type == "NORMAL_MAP":
        texture_socket = _input_socket(source, "Color")
        scale = _float_input(source, "Strength", 1.0)
    elif source_type == "BUMP":
        bump = True
        texture_socket = _input_socket(source, "Height")
        scale = _float_input(source, "Strength", 1.0)
        if bool(getattr(source, "invert", False)):
            scale = -scale
    texture_node = _linked_texture_node(texture_socket)
    if texture_node is None:
        return None
    asset_path = _image_asset_path(getattr(texture_node, "image", None))
    if not asset_path:
        return None
    shader_name = "ND_image_float_geometry_normal" if bump else "ND_image_color3_geometry_normal"
    return {
        "openpbr_input": "geometry_normal",
        "shader_name": shader_name,
        "info_id": "ND_image_float" if bump else "ND_image_color3",
        "asset_path": asset_path,
        "color_space": _image_color_space(getattr(texture_node, "image", None), "raw"),
        "value_type": "float" if bump else "color3f",
        "image_output_type": "float" if bump else "color3f",
        "scale": scale,
        "bump": bump,
    }


def _linked_texture_node(socket: Any | None, *, depth: int = 8) -> Any | None:
    node, _ = _linked_texture_info(socket, depth=depth)
    return node


def _linked_texture_info(socket: Any | None, *, depth: int = 8) -> tuple[Any | None, bool]:
    node = _socket_link_target_node(socket)
    return _find_texture_info(node, depth=depth)


def _find_texture_node(node: Any | None, *, depth: int = 8) -> Any | None:
    found, _ = _find_texture_info(node, depth=depth)
    return found


def _find_texture_info(
    node: Any | None,
    *,
    depth: int = 8,
    inverted: bool = False,
) -> tuple[Any | None, bool]:
    if node is None or depth <= 0:
        return None, False
    if _node_type(node) == "TEX_IMAGE":
        return node, inverted
    if _node_type(node) == "INVERT":
        inverted = not inverted
    for socket in _texture_chain_inputs(node):
        found, child_inverted = _find_texture_info(
            _socket_link_target_node(socket),
            depth=depth - 1,
            inverted=inverted,
        )
        if found is not None:
            return found, child_inverted
    return None, False


def _texture_chain_inputs(node: Any) -> tuple[Any, ...]:
    node_type = _node_type(node)
    if node_type in {"MIX", "MIX_RGB"}:
        all_sockets = tuple(_input_sockets(node))
        mix_sockets = tuple(
            socket
            for socket in all_sockets
            if str(getattr(socket, "name", "")) in {"A", "B", "Color1", "Color2"}
        )
        linked_color_sockets = tuple(
            socket
            for socket in mix_sockets
            if _socket_linked(socket) and _socket_is_color_input(socket)
        )
        linked_mix_sockets = tuple(socket for socket in mix_sockets if _socket_linked(socket))
        linked_sockets = tuple(socket for socket in all_sockets if _socket_linked(socket))
        return (
            linked_color_sockets
            or linked_mix_sockets
            or linked_sockets
            or mix_sockets
            or all_sockets
        )
    preferred_by_type = {
        "BUMP": ("Height",),
        "INVERT": ("Color",),
        "NORMAL_MAP": ("Color",),
        "SEPARATE_COLOR": ("Color", "Image"),
        "SEPRGB": ("Image",),
        "VALTORGB": ("Fac",),
    }
    preferred = preferred_by_type.get(node_type, ())
    sockets = tuple(
        socket
        for socket in (_input_socket(node, name) for name in preferred)
        if socket is not None
    )
    return sockets or tuple(_input_sockets(node))


def _linked_static_color(node: Any, socket_name: str) -> tuple[float, float, float] | None:
    return _static_color_from_node(_socket_link_target_node(_input_socket(node, socket_name)))


def _static_color_from_node(node: Any | None, *, depth: int = 8) -> tuple[float, float, float] | None:
    if node is None or depth <= 0:
        return None
    node_type = _node_type(node)
    if node_type == "TEX_IMAGE":
        return None
    if node_type == "RGB":
        return _output_color3(node, "Color")
    if node_type == "VALUE":
        value = _output_float(node, "Value")
        return (value, value, value) if value is not None else None
    if node_type == "BLACKBODY":
        return _kelvin_rgb(_float_input(node, "Temperature", 6500.0))
    if node_type == "TEX_SKY":
        return (0.8, 0.9, 1.0)
    if node_type == "INVERT":
        color = _static_color_from_node(
            _socket_link_target_node(_input_socket(node, "Color")),
            depth=depth - 1,
        )
        return tuple(1.0 - channel for channel in color) if color is not None else None
    if node_type in {"BRIGHTCONTRAST", "CURVE_RGB", "GAMMA", "HUE_SAT"}:
        return _static_color_from_node(
            _socket_link_target_node(_input_socket(node, "Color")),
            depth=depth - 1,
        )
    if node_type in {"MIX", "MIX_RGB"}:
        colors = []
        for socket in _input_sockets(node):
            if str(getattr(socket, "name", "")) not in {"A", "B", "Color1", "Color2"}:
                continue
            color = _static_color_from_node(_socket_link_target_node(socket), depth=depth - 1)
            if color is not None:
                colors.append(color)
        if colors:
            return tuple(sum(color[index] for color in colors) / len(colors) for index in range(3))
    return None


def _output_color3(node: Any, name: str) -> tuple[float, float, float] | None:
    value = _output_default(node, name)
    if value is None:
        return None
    try:
        channels = tuple(float(channel) for channel in list(value)[:3])
    except (TypeError, ValueError):
        return None
    return channels if len(channels) == 3 else None


def _output_float(node: Any, name: str) -> float | None:
    value = _output_default(node, name)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _output_default(node: Any, name: str) -> Any | None:
    outputs = getattr(node, "outputs", None)
    if outputs is None:
        return None
    if hasattr(outputs, "get"):
        socket = outputs.get(name)
    else:
        try:
            socket = outputs[name]
        except (KeyError, IndexError, TypeError):
            socket = None
    return getattr(socket, "default_value", None)


def _kelvin_rgb(kelvin: float) -> tuple[float, float, float]:
    temp = max(10.0, min(400.0, kelvin / 100.0))
    if temp <= 66.0:
        red = 255.0
        green = 99.4708025861 * math.log(temp) - 161.1195681661
        blue = 0.0 if temp <= 19.0 else 138.5177312231 * math.log(temp - 10.0) - 305.0447927307
    else:
        red = 329.698727446 * ((temp - 60.0) ** -0.1332047592)
        green = 288.1221695283 * ((temp - 60.0) ** -0.0755148492)
        blue = 255.0
    return tuple(max(0.0, min(1.0, channel / 255.0)) for channel in (red, green, blue))


def _socket_is_color_input(socket: Any) -> bool:
    return str(getattr(socket, "type", "")).upper() in {"RGBA", "COLOR"}


def _scalar_texture_channel(socket: Any | None) -> int:
    link = _first_link(socket)
    node = getattr(link, "from_node", None) if link is not None else None
    depth = 8
    while node is not None and depth > 0:
        if _node_type(node) in {"SEPARATE_COLOR", "SEPRGB"}:
            from_socket_name = str(getattr(getattr(link, "from_socket", None), "name", ""))
            channel = _CHANNEL_INDEX.get(from_socket_name)
            return int(channel) if channel is not None else 0
        next_link = None
        for input_socket in _input_sockets(node):
            candidate = _first_link(input_socket)
            if candidate is not None:
                next_link = candidate
                break
        link = next_link
        node = getattr(link, "from_node", None) if link is not None else None
        depth -= 1
    return 0


def _first_link(socket: Any | None) -> Any | None:
    if socket is not None and _ACTIVE_INPUT_LINKS is not None:
        return _ACTIVE_INPUT_LINKS.get(_blender_identity(socket))
    links = list(getattr(socket, "links", ()) or ()) if socket is not None else []
    return links[0] if links else None


def _image_asset_path(image: Any) -> str:
    if image is None:
        return ""
    for candidate in _image_path_candidates(image):
        path = _absolute_image_path(candidate)
        if path and path.is_file():
            return str(path)
    materialized = _materialized_image_path(image)
    if materialized:
        return materialized
    return ""


def _image_path_candidates(image: Any) -> tuple[str, ...]:
    values: list[str] = []
    filepath_from_user = getattr(image, "filepath_from_user", None)
    if callable(filepath_from_user):
        try:
            values.append(str(filepath_from_user() or ""))
        except Exception:
            pass
    for attr in ("filepath", "filepath_raw"):
        value = str(getattr(image, attr, "") or "")
        if value:
            values.append(value)
    return tuple(dict.fromkeys(value for value in values if value))


def _absolute_image_path(value: str) -> Path | None:
    if not value:
        return None
    try:
        import bpy  # type: ignore

        path = Path(bpy.path.abspath(value)).expanduser()
    except Exception:
        path = Path(value).expanduser()
    return path if path.is_absolute() else path.resolve()


def _materialized_image_path(image: Any) -> str:
    if not (getattr(image, "packed_file", None) is not None or bool(getattr(image, "has_data", False))):
        return ""
    target_dir = _texture_output_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / _texture_filename(image)
    packed = getattr(image, "packed_file", None)
    packed_data = getattr(packed, "data", None)
    if packed_data is not None:
        try:
            target.write_bytes(bytes(packed_data))
        except Exception:
            return ""
        return str(target) if target.is_file() else ""
    if not target.is_file():
        save = getattr(image, "save", None)
        if not callable(save):
            return ""
        try:
            save(filepath=str(target))
        except Exception:
            return ""
    return str(target) if target.is_file() else ""


def _texture_output_dir() -> Path:
    root = Path(
        os.environ.get("OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR")
        or Path(tempfile.gettempdir()) / "ov-blender-example" / "temporary-usd-layers"
    ).expanduser()
    return root / "materialx-openpbr-textures"


def _texture_filename(image: Any) -> str:
    raw_name = str(getattr(image, "name", "") or "texture")
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(raw_name).name) or "texture"
    if "." not in name:
        name += ".png"
    return name


def _image_color_space(image: Any, fallback: str) -> str:
    name = ""
    try:
        name = str(image.colorspace_settings.name or "") if image is not None else ""
    except AttributeError:
        name = ""
    low = name.lower()
    if "srgb" in low:
        return "sRGB"
    if low in _COLORSPACE_RAW_NAMES:
        return "raw"
    return fallback


def _float_input(node: Any, name: str, default: float) -> float:
    socket = _input_socket(node, name)
    if socket is None:
        return float(default)
    try:
        return float(getattr(socket, "default_value", default))
    except (TypeError, ValueError):
        return float(default)


def _float_input_any(node: Any, names: Iterable[str], default: float) -> float:
    for name in names:
        socket = _input_socket(node, name)
        if socket is not None:
            return _float_input(node, name, default)
    return float(default)


def _color3_input(node: Any, name: str, default: tuple[float, float, float]) -> tuple[float, float, float]:
    socket = _input_socket(node, name)
    value = getattr(socket, "default_value", default) if socket is not None else default
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError, IndexError):
        return default


def _passive_default(name: str, value: Any) -> bool:
    if name in {"Coat Normal", "Tangent"}:
        return value is None or _value_close(value, [0.0, 0.0, 0.0])
    expected = _PASSIVE_DEFAULTS.get(name, None)
    if expected is None:
        return value is None or value == "" or value is False
    if isinstance(expected, tuple):
        return any(_value_close(value, item) for item in expected)
    return _value_close(value, expected)


def _value_close(value: Any, expected: Any) -> bool:
    if isinstance(value, (list, tuple)):
        if isinstance(expected, (int, float)) and len(value) == 1:
            return _value_close(value[0], expected)
        if isinstance(expected, (list, tuple)) and len(value) >= len(expected):
            return all(_value_close(value[index], expected[index]) for index in range(len(expected)))
        return False
    try:
        return abs(float(value) - float(expected)) <= 1.0e-6
    except (TypeError, ValueError):
        return value == expected


def _valid_target_paths(paths: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(path for path in (str(value) for value in paths) if _known_prim_path(path)))


def _material_name(material: Any) -> str:
    return str(getattr(material, "name_full", getattr(material, "name", "")) or "Material")


def _prim_path(prim: Any) -> str:
    try:
        return str(prim.GetPath())
    except Exception:
        return str(getattr(prim, "path", ""))


def _is_material_prim(prim: Any, material_type: Any | None) -> bool:
    if material_type is not None:
        try:
            return bool(prim.IsA(material_type))
        except Exception:
            pass
    type_name = str(getattr(prim, "type_name", ""))
    if not type_name:
        getter = getattr(prim, "GetTypeName", None)
        if callable(getter):
            try:
                type_name = str(getter())
            except Exception:
                type_name = ""
    return type_name == "Material"


def _bound_material_path(prim: Any) -> str:
    relationship = None
    getter = getattr(prim, "GetRelationship", None)
    if callable(getter):
        try:
            relationship = getter("material:binding")
        except Exception:
            relationship = None
    if relationship is None:
        relationship = getattr(prim, "material_binding", None)
    targets = []
    target_getter = getattr(relationship, "GetTargets", None)
    if callable(target_getter):
        try:
            targets = list(target_getter())
        except Exception:
            targets = []
    elif relationship:
        targets = list(relationship if isinstance(relationship, (list, tuple)) else (relationship,))
    for target in targets:
        path = str(target)
        if _known_prim_path(path):
            return path
    return ""


def _sanitize_identifier(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not value:
        value = "Material"
    if value[0].isdigit():
        value = "_" + value
    return value


def _unique_identifier(base: str, used: set[str]) -> str:
    candidate = base
    index = 2
    while candidate in used:
        candidate = f"{base}_{index}"
        index += 1
    used.add(candidate)
    return candidate


def _plain_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return str(value)


def _usda_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _usda_asset_path(value: str) -> str:
    return value.replace("\\", "/").replace("@", "%40")


def _usda_float(value: float) -> str:
    return f"{float(value):.9g}"


def _usda_color3(value: Iterable[float]) -> str:
    red, green, blue = tuple(value)
    return f"({_usda_float(red)}, {_usda_float(green)}, {_usda_float(blue)})"


def _digest_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _known_prim_path(value: str) -> bool:
    value = str(value or "").strip()
    return (
        bool(value)
        and value != "/"
        and value != "???"
        and value.startswith("/")
        and "." not in value
        and "[" not in value
        and "]" not in value
        and not value.endswith("/")
    )


def _layer_diagnostics(overlay: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics = {
        "source": _SOURCE,
        "path": "",
        "digest": str(overlay.get("digest", "")),
        "status": str(overlay.get("status", "")),
        "generated": True,
        "material_count": int(overlay.get("material_count", 0) or 0),
        "generated_material_paths": list(overlay.get("generated_material_paths", ())),
        "binding_targets": list(overlay.get("binding_targets", ())),
        "materials": [dict(record) for record in overlay.get("materials", ())],
    }
    if "selection_policy" in overlay:
        diagnostics["selection_policy"] = str(overlay.get("selection_policy", ""))
    return diagnostics


__all__ = [
    "MaterialSceneConversionResult",
    "MaterialSceneConversionStatus",
    "scene_layer_from_materials",
]

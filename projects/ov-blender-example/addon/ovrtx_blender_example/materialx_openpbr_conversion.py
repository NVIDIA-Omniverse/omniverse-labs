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
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

from . import usd_paths as usd_paths
from . import materialx_node_network as node_network
from .sky_texture_radiance import sky_texture_radiance
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
    # Blender 5.3 added dispersion to the Principled BSDF; 5.2 files have no such sockets.
    "Transmission Dispersion Scale",
    "Transmission Dispersion Abbe Number",
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
    "ATTRIBUTE",
    "BSDF_DIFFUSE",
    "BSDF_GLASS",
    "BSDF_GLOSSY",
    "BSDF_HAIR",
    "BSDF_HAIR_PRINCIPLED",
    "BSDF_PRINCIPLED",
    "BSDF_TRANSLUCENT",
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
    "OBJECT_INFO",
    "SEPARATE_COLOR",
    "SEPRGB",
    "TEX_COORD",
    "TEX_NOISE",
    "TEX_SKY",
    "TEX_WAVE",
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
    "BSDF_GLASS",
    "BSDF_GLOSSY",
    "BSDF_HAIR",
    "BSDF_HAIR_PRINCIPLED",
    "BSDF_PRINCIPLED",
    "BSDF_TRANSLUCENT",
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
    "Transmission Dispersion Abbe Number": 20.0,
    "Transmission Dispersion Scale": 0.0,
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
    parameter_layouts: dict[Any, Any] = {}

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
        values = _deconflicted_values(material_id, result["openpbr_values"], parameter_layouts)
        if values is None:
            material_records.append(
                {
                    **result,
                    "status": _STATUS_UNSUPPORTED_NO_ENTRY,
                    "blocking_reasons": ["materialx_parameter_name_collision"],
                    "openpbr_values": {},
                }
            )
            continue
        result = {
            **result,
            "openpbr_values": values,
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


_INPUT_LINE = re.compile(r"^\s+\S+ inputs:([A-Za-z0-9_:]+)(\.connect)?\s*=")
_MAX_PARAMETER_PADDING = 24


def _parameter_layout(lines: Iterable[str]) -> tuple[tuple, tuple]:
    """``(parameter names, wiring)`` of one material's authored shaders.

    The installed renderer names a compiled material's parameters after the node
    definition, its index among nodes of that definition and the input
    (``ND_add_vector2__1_in2``), and registers those names for the whole stage. Two
    materials with the same set of unconnected parameter names but different
    wiring (scale -> rotate -> offset versus offset -> rotate -> scale) cannot both
    compile: one of them logs "Unable to find original SdfPath for key" and renders
    with the default material. ``parameter names`` is that set (as counts per node
    definition and input); ``wiring`` identifies the connection structure.
    """

    shaders: list[dict[str, Any]] = []
    index: dict[str, int] = {}
    current: dict[str, Any] | None = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("def Shader "):
            current = {"id": "", "declared": [], "connected": {}}
            index[stripped.split('"')[1]] = len(shaders)
            shaders.append(current)
            continue
        if current is None:
            continue
        if stripped.startswith("uniform token info:id"):
            current["id"] = stripped.split('"')[1]
            continue
        match = _INPUT_LINE.match(line)
        if not match:
            continue
        if match.group(2):
            source = line.split("<", 1)[1].split(">", 1)[0].rsplit("/", 1)[-1].split(".", 1)[0]
            current["connected"][match.group(1)] = source
        else:
            current["declared"].append(match.group(1))
    counts: dict[tuple[str, str], int] = {}
    wiring = []
    for shader in shaders:
        parameters = sorted(name for name in shader["declared"] if name not in shader["connected"])
        for name in parameters:
            counts[(shader["id"], name)] = counts.get((shader["id"], name), 0) + 1
        wiring.append(
            (
                shader["id"],
                tuple(sorted((name, index.get(source, -1)) for name, source in shader["connected"].items())),
                tuple(parameters),
            )
        )
    return tuple(sorted(counts.items())), tuple(wiring)


def _padded_values(values: Mapping[str, Any], count: int) -> dict[str, Any] | None:
    """``values`` with ``count`` neutral ``+ 0`` nodes appended to one translated network.

    Each pad node adds a parameter name, so the padded material's parameter-name set
    differs from an unpadded material with the same wiring problem. None when the
    material has no network to extend.
    """

    textures = values.get("textures") or {}
    for name in sorted(textures):
        record = textures[name]
        network = record.get("network") if isinstance(record, Mapping) else None
        if not network:
            continue
        nodes = list(network["nodes"])
        out = str(network["out"])
        color = network["out_type"] == "color3f"
        for number in range(1, count + 1):
            pad_name = f"ND_net_{record.get('openpbr_input', name)}_pad_{number:02d}"
            nodes.append(
                {
                    "name": pad_name,
                    "info_id": "ND_add_color3FA" if color else "ND_add_float",
                    "out_type": network["out_type"],
                    "inputs": [
                        {"type": network["out_type"], "name": "in1", "const": None, "connect": out},
                        {"type": "float", "name": "in2", "const": (0.0,), "connect": ""},
                    ],
                }
            )
            out = pad_name
        padded = {**record, "network": {**network, "nodes": nodes, "out": out}}
        return {**values, "textures": {**textures, name: padded}}
    return None


def _deconflicted_values(
    material_id: str, values: Mapping[str, Any], seen: dict[Any, Any]
) -> Mapping[str, Any] | None:
    """``values`` made safe to author next to the materials already in ``seen``.

    ``seen`` maps parameter-name sets to the wiring that owns them (see
    ``_parameter_layout``). A different wiring with the same names gets neutral pad
    nodes until its names are unique; a material that cannot be padded is
    rejected (None) so it keeps its stock shading instead of rendering with the
    renderer's default material.
    """

    candidate: Mapping[str, Any] | None = values
    for padding in range(_MAX_PARAMETER_PADDING + 1):
        names, wiring = _parameter_layout(_material_block_lines(material_id, candidate))
        owner = seen.get(names)
        if owner is None or owner == wiring:
            seen[names] = wiring
            return candidate
        candidate = _padded_values(values, padding + 1)
        if candidate is None:
            return None
    return None


def scene_layer_from_materials(
    materials: Iterable[Any],
    input_usd_path: str,
    *,
    allow_stock_fallback: bool = False,
    binding_identity: Mapping[str, Any] | None = None,
    known_material_paths: Mapping[int, str] | None = None,
) -> MaterialSceneConversionResult:
    """Convert every USD-bound source material into one OVRTX presentation layer.

    ``known_material_paths`` maps Blender material session UIDs to the prim
    paths a current scene generation exported them at; it takes precedence
    over name matching, which cannot identify library-linked materials.
    """

    selected = tuple(materials)
    if not selected or not input_usd_path:
        return MaterialSceneConversionResult(
            MaterialSceneConversionStatus.OK,
            diagnostics={
                "selected_materials": tuple(_material_name(material) for material in selected),
                "skipped_materials": (),
            },
        )

    identity = (binding_identity if binding_identity is not None
                else _materialx_binding_identity(input_usd_path))
    overlay = _scene_overlay_from_materials(selected, identity, known_material_paths)
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
    known_material_paths: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    """Build the automatic scene overlay for every supported material."""

    selected = tuple(materials)
    bindings: list[_MaterialBinding] = []
    skipped_materials: list[dict[str, Any]] = []
    for material in selected:
        resolution = _resolve_binding(material, identity, known_material_paths)
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
    known_material_paths: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    material_name = _material_name(material)
    source_path = usd_paths.source_usd_path_from_blender_id(material)
    match_source = "source_usd_path" if source_path else ""
    known_path = _known_material_path(material, identity, known_material_paths)
    if known_path:
        # The current scene generation exported this exact material here.
        source_path, match_source = known_path, "scene_generation_identity"
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
    by_material = identity.get("bindings_by_material")
    bindings = (
        by_material.get(source_path, ()) if isinstance(by_material, Mapping)
        else [binding for binding in identity.get("bindings", ())
              if str(binding.get("material_path", "")) == source_path]
    )
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


def _known_material_path(
    material: Any,
    identity: Mapping[str, Any],
    known_material_paths: Mapping[int, str] | None,
) -> str:
    if not known_material_paths:
        return ""
    uid = getattr(material, "session_uid", None)
    path = known_material_paths.get(int(uid)) if uid is not None else None
    if not path:
        return ""
    material_paths = identity.get("material_paths")
    if material_paths is not None and str(path) not in {str(item) for item in material_paths}:
        return ""
    return str(path)


def _material_path_from_name(material_name: str, identity: Mapping[str, Any]) -> str:
    expected = _sanitize_identifier(material_name)
    by_name = identity.get("materials_by_name")
    matches = (
        by_name.get(expected, ()) if isinstance(by_name, Mapping)
        else [str(path) for path in identity.get("material_paths", ())
              if str(path).rstrip("/").rsplit("/", 1)[-1] == expected]
    )
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

    global _ACTIVE_INPUT_LINKS, _ACTIVE_MATERIAL, _ACTIVE_NETWORK_CACHE
    previous_input_links = _ACTIVE_INPUT_LINKS
    previous_material = _ACTIVE_MATERIAL
    previous_cache = _ACTIVE_NETWORK_CACHE
    nodes, _ACTIVE_INPUT_LINKS = _flattened_node_tree(node_tree)
    _ACTIVE_MATERIAL = material
    _ACTIVE_NETWORK_CACHE = {}
    try:
        active_output = _active_output_node(nodes)
        if active_output is None:
            return {
                **base,
                "node_inventory": [_node_diagnostics(node, None) for node in nodes],
                "blocking_reasons": ["missing_active_material_output"],
            }

        surface_node = _output_surface_node(active_output)
        active_graph_nodes = _surface_graph_nodes(surface_node)
        active_node_ids = {
            _blender_identity(node)
            for node in (*active_graph_nodes, active_output)
        }
        inventory = [
            _node_diagnostics(node, surface_node)
            for node in nodes
            if _blender_identity(node) in active_node_ids
        ]
        blocking_reasons = _blocking_reasons(inventory)
        blocking_reasons.extend(_image_coordinate_blocking_reasons(active_graph_nodes))
        if blocking_reasons:
            return {**base, "node_inventory": inventory, "blocking_reasons": blocking_reasons}
        values = _openpbr_values_from_surface(surface_node)
        if values is None:
            return {
                **base,
                "node_inventory": inventory,
                "blocking_reasons": [f"unsupported_surface:{_node_type(surface_node) or 'missing'}"],
            }
        getter = getattr(material, "get", None)
        thin_walled = getter("ovrtx_thin_walled", False) if callable(getter) else False
        if type(thin_walled) is not bool:
            return {**base, "node_inventory": inventory,
                    "blocking_reasons": ["ovrtx_thin_walled_must_be_boolean"]}
        if thin_walled:
            volume_socket = _input_socket(active_output, "Volume")
            if volume_socket is not None and _socket_linked(volume_socket):
                return {**base, "node_inventory": inventory,
                        "blocking_reasons": ["ovrtx_thin_walled_unsupported_volume_absorption"]}
            if _node_type(surface_node) != "BSDF_PRINCIPLED" or not (
                float(values.get("transmission_weight", 0.0)) > _EPSILON
                or "transmission_weight" in values.get("textures", {})
            ):
                return {**base, "node_inventory": inventory,
                        "blocking_reasons": ["ovrtx_thin_walled_requires_principled_transmission"]}
            values["geometry_thin_walled"] = True
        approximations = sorted(
            {
                *(
                    f"{name}:{op}"
                    for name, texture in (values.get("textures") or {}).items()
                    if isinstance(texture, Mapping)
                    for op in texture.get("approximated_ops", ())
                ),
                *(str(note) for note in values.get("approximations", ())),
                *_displacement_approximations(active_output),
            }
        )
        translation_failures = sorted(
            f"{name}:{texture['network_failure']}"
            for name, texture in (values.get("textures") or {}).items()
            if isinstance(texture, Mapping) and texture.get("network_failure")
        )
        error_bounds = sorted(
            f"{name}:{bound}"
            for name, texture in (values.get("textures") or {}).items()
            if isinstance(texture, Mapping)
            for bound in texture.get("error_bounds", ())
        )
        return {
            **base,
            "status": _STATUS_GENERATED,
            "node_inventory": inventory,
            "openpbr_values": values,
            **({"approximations": approximations} if approximations else {}),
            **({"error_bounds": error_bounds} if error_bounds else {}),
            **({"translation_failures": translation_failures} if translation_failures else {}),
        }
    finally:
        _ACTIVE_INPUT_LINKS = previous_input_links
        _ACTIVE_MATERIAL = previous_material
        _ACTIVE_NETWORK_CACHE = previous_cache


_EXTENSION_ADDRESS_MODES = {
    "REPEAT": "periodic",
    "EXTEND": "clamp",
    "CLIP": "constant",
    "MIRROR": "mirror",
}
_DEFAULT_ADDRESS_MODE = "periodic"


def _image_address_mode(texture_node: Any) -> str | None:
    """MaterialX address mode for an Image Texture extension (None: no equivalent).

    REPEAT is MaterialX ``periodic``, EXTEND ``clamp`` and MIRROR ``mirror``.
    CLIP (transparent black outside the image) is ``constant``, which returns the
    image node's default colour, black; only the alpha of CLIP has no counterpart
    and the converter never reads image alpha.
    """

    return _EXTENSION_ADDRESS_MODES.get(str(getattr(texture_node, "extension", "REPEAT")))


def _image_address_modes(texture_node: Any) -> dict[str, str]:
    return {"address_mode": _image_address_mode(texture_node) or _DEFAULT_ADDRESS_MODE}


def _displacement_approximations(output_node: Any) -> list[str]:
    """Material Output Displacement is not translated; a linked one is reported."""

    socket = _input_socket(output_node, "Displacement")
    return ["material_output:DISPLACEMENT_not_translated"] if _socket_linked(socket) else []


def _image_coordinate_blocking_reasons(nodes: Iterable[Any]) -> list[str]:
    """Report authored coordinate behavior omitted by image conversion.

    Shared by initial presentation, source fallback and direct graph updates.
    Default active mesh UV remains supported; unused nodes never reach here.
    """
    reasons = []
    for node in nodes:
        if _node_type(node) != "TEX_IMAGE":
            continue
        vector = _input_socket(node, "Vector")
        if _socket_linked(vector) and _uv_chain(_socket_link(vector)) is None:
            reasons.append("image coordinate graph is not translated: " + node.name)
        if getattr(node, "projection", "FLAT") != "FLAT":
            reasons.append("image projection is not translated: " + node.name)
        if _image_address_mode(node) is None:
            reasons.append("image extension mode is not translated: " + node.name)
    return reasons


_IDENTITY_UV_TRANSFORM = ((1.0, 1.0), (0.0, 0.0))
_ACTIVE_MATERIAL: Any | None = None
# Network translations are pure functions of one classification's node graph.
_ACTIVE_NETWORK_CACHE: dict[Any, Any] | None = None


def _translate_cached(prefix: str, socket: Any, kind: str) -> tuple[Any, Any]:
    cache = _ACTIVE_NETWORK_CACHE
    key = (prefix, _blender_identity(socket), kind)
    if cache is not None and key in cache:
        hit = cache[key]
        if isinstance(hit, Exception):
            raise hit
        return hit
    try:
        result = node_network.translate_socket(prefix, socket, kind)
    except node_network.Untranslatable as exc:
        if cache is not None:
            cache[key] = exc
        raise
    if cache is not None:
        cache[key] = result
    return result


def _material_uv_layouts(material: Any) -> list[tuple[str, tuple[str, ...]]] | None:
    """(active render UV name, all UV names) for each mesh bound to ``material``.

    None when the meshes cannot be enumerated. Only evaluated for materials
    that contain a UV Map node naming a map.
    """

    if material is None:
        return None
    try:
        import bpy  # type: ignore
    except Exception:
        return None
    layouts: list[tuple[str, tuple[str, ...]]] = []
    seen: set[int] = set()
    try:
        for obj in bpy.data.objects:
            if getattr(obj, "type", "") != "MESH":
                continue
            if not any(getattr(slot, "material", None) == material for slot in obj.material_slots):
                continue
            data = obj.data
            key = _blender_identity(data)
            if key in seen:
                continue
            seen.add(key)
            layers = list(data.uv_layers)
            active = next((layer.name for layer in layers if layer.active_render), layers[0].name if layers else "")
            layouts.append((active, tuple(layer.name for layer in layers)))
    except Exception:
        return None
    return layouts


def _named_uv_primvar(name: str) -> str | None:
    """Primvar a named UV Map reads: "" for the exported render UV (``st``), else its name.

    The stock export authors the active render UV map as ``st`` and every other
    map under its own name. A name resolves only when every mesh bound to the
    material has the map and agrees on whether it is the render UV; anything
    else is rejected because one shared material cannot read two primvars.
    """

    layouts = _material_uv_layouts(_ACTIVE_MATERIAL)
    if not layouts:
        return None
    if any(name not in names for _, names in layouts):
        return None
    active = {render for render, _ in layouts}
    if active == {name}:
        return ""
    if name in active:
        return None
    return _sanitize_identifier(name)


def _uv_chain(
    link: Any | None, *, depth: int = 8
) -> tuple[tuple[float, float], tuple[float, float], str, tuple[tuple[str, Any], ...] | None] | None:
    """``(scale, offset, primvar, steps)``: the UV an image coordinate link reads.

    Generated image nodes (and stock UsdUVTexture readers) sample the exported
    active render UV (primvar ""). Texture Coordinate UV and an unnamed UV Map
    are that UV; a UV Map naming a map reads that map when every bound mesh
    agrees (see ``_named_uv_primvar``); Mapping nodes with unlinked values apply
    scale, Z rotation and offset in UV space (chains compose). ``scale``/``offset``
    is the composed affine transform when there is no rotation; with a rotation
    ``steps`` lists the operations in application order instead
    (``("scale", (sx, sy))``, ``("rotate", degrees)``, ``("offset", (ox, oy))``).
    None for anything else.
    """

    chain = _uv_step_chain(link, depth=depth)
    if chain is None:
        return None
    steps, primvar = chain
    if any(kind == "rotate" for kind, _ in steps):
        return (*_IDENTITY_UV_TRANSFORM, primvar, tuple(steps))
    scale, offset = (1.0, 1.0), (0.0, 0.0)
    for kind, value in steps:
        if kind == "scale":
            scale = (scale[0] * value[0], scale[1] * value[1])
            offset = (offset[0] * value[0], offset[1] * value[1])
        else:
            offset = (offset[0] + value[0], offset[1] + value[1])
    return scale, offset, primvar, None


def _uv_step_chain(link: Any | None, *, depth: int = 8) -> tuple[list[tuple[str, Any]], str] | None:
    if link is None or depth <= 0:
        return None
    source = getattr(link, "from_node", None)
    output = str(getattr(getattr(link, "from_socket", None), "name", ""))
    node_type = _node_type(source)
    if node_type == "TEX_COORD":
        return ([], "") if output == "UV" else None
    if node_type == "UVMAP":
        if output != "UV":
            return None
        name = str(getattr(source, "uv_map", "") or "")
        if not name:
            return [], ""
        primvar = _named_uv_primvar(name)
        return None if primvar is None else ([], primvar)
    if node_type != "MAPPING" or output != "Vector":
        return None
    vector_type = str(getattr(source, "vector_type", "POINT"))
    if vector_type not in {"POINT", "TEXTURE", "VECTOR"}:
        return None
    values = {}
    for name in ("Location", "Rotation", "Scale"):
        socket = _input_socket(source, name)
        if socket is None or _socket_linked(socket):
            return None
        try:
            values[name] = tuple(float(value) for value in socket.default_value)
        except (AttributeError, TypeError, ValueError):
            return None
        if len(values[name]) != 3:
            return None
    rx, ry, rz = values["Rotation"]
    degrees = math.degrees(rz)
    vector = _input_socket(source, "Vector")
    if not _socket_linked(vector):
        return None
    inner = _uv_step_chain(_socket_link(vector), depth=depth - 1)
    if inner is None:
        return None
    steps, primvar = inner
    # A tilt about X or Y leaves the UV plane. The vector's z is 0 when it comes straight
    # from a UV, so a single tilt only shortens one axis by cos(angle) before the Z rotation.
    # Anything else (both tilts, or a tilt after another Mapping) is not translated.
    tilt: tuple[str, Any] | None = None
    if abs(rx) > _EPSILON or abs(ry) > _EPSILON:
        fed_by_mapping = _node_type(getattr(_socket_link(vector), "from_node", None)) == "MAPPING"
        if (abs(rx) > _EPSILON and abs(ry) > _EPSILON) or fed_by_mapping:
            return None
        tilt = ("scale", (1.0, math.cos(rx))) if abs(rx) > _EPSILON else ("scale", (math.cos(ry), 1.0))
    (lx, ly, _), (sx, sy, _) = values["Location"], values["Scale"]
    tilts = [tilt] if tilt is not None else []
    if vector_type == "POINT":
        own = [("scale", (sx, sy)), *tilts, ("rotate", degrees), ("offset", (lx, ly))]
    elif vector_type == "VECTOR":
        own = [("scale", (sx, sy)), *tilts, ("rotate", degrees)]
    else:  # TEXTURE applies the inverse transform
        if abs(sx) <= _EPSILON or abs(sy) <= _EPSILON:
            return None
        own = [("offset", (-lx, -ly)), ("rotate", -degrees), *tilts, ("scale", (1.0 / sx, 1.0 / sy))]
    steps = steps + [
        step for step in own
        if not (
            (step[0] == "rotate" and abs(step[1]) <= _EPSILON)
            or (step[0] == "scale" and all(abs(v - 1.0) <= _EPSILON for v in step[1]))
            or (step[0] == "offset" and all(abs(v) <= _EPSILON for v in step[1]))
        )
    ]
    return steps, primvar


def _uv_coordinate_transform(
    link: Any | None, *, depth: int = 8
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """2D affine ``uv * scale + offset`` a linked image coordinate applies to its UV.

    None for anything that is not a plain affine transform (including a rotation,
    which ``_uv_chain`` reports as ordered steps).
    """

    chain = _uv_chain(link, depth=depth)
    if chain is None or chain[3] is not None:
        return None
    return (chain[0], chain[1])


def _image_uv_transform(texture_node: Any) -> dict[str, Any] | None:
    """UV transform for ``texture_node`` (None when it reads the default render UV unchanged)."""

    vector = _input_socket(texture_node, "Vector")
    if not _socket_linked(vector):
        return None
    chain = _uv_chain(_socket_link(vector))
    if chain is None:
        return None
    scale, offset, primvar, steps = chain
    if steps is None and not primvar and all(
        abs(a - b) <= _EPSILON
        for pair, identity in zip((scale, offset), _IDENTITY_UV_TRANSFORM)
        for a, b in zip(pair, identity)
    ):
        return None
    return {
        "scale": scale,
        "offset": offset,
        **({"primvar": primvar} if primvar else {}),
        **({"steps": steps} if steps is not None else {}),
    }


def _socket_link(socket: Any | None) -> Any | None:
    """The single link feeding ``socket``, honouring the active link index."""

    if socket is None or getattr(socket, "is_linked", None) is False:
        return None
    if _ACTIVE_INPUT_LINKS is not None:
        return _ACTIVE_INPUT_LINKS.get(_blender_identity(socket))
    links = list(getattr(socket, "links", ()) or ())
    return links[0] if links else None


def _openpbr_values_from_surface(surface_node: Any | None) -> dict[str, Any] | None:
    node_type = _node_type(surface_node)
    if node_type == "BSDF_PRINCIPLED":
        return _principled_openpbr_values(surface_node)
    if node_type == "BSDF_DIFFUSE":
        return _diffuse_openpbr_values(surface_node)
    if node_type == "BSDF_GLOSSY":
        return _glossy_openpbr_values(surface_node)
    if node_type == "BSDF_GLASS":
        return _glass_openpbr_values(surface_node)
    if node_type == "BSDF_TRANSLUCENT":
        return _translucent_openpbr_values(surface_node)
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
    _apply_folded_constants(
        node,
        values,
        (
            ("Base Color", "base_color", "color3"),
            ("Metallic", "base_metalness", "float"),
            ("Roughness", "specular_roughness", "float"),
            ("Alpha", "geometry_opacity", "float"),
        ),
    )
    transmission_weight = _float_input(node, "Transmission Weight", 0.0)
    transmission_texture = _texture_record_from_socket(node, "Transmission Weight")
    if transmission_texture or transmission_weight > 1.0e-6:
        values["transmission_weight"] = transmission_weight
        if transmission_texture:
            textures["transmission_weight"] = transmission_texture
        elif any(abs(channel - 1.0) > 1.0e-6 for channel in values["base_color"]):
            values["transmission_color"] = values["base_color"]
    # Blender 5.3 dispersion: a zero scale is Blender's default and means none,
    # so the Abbe number only matters when the scale is set.
    dispersion_scale = _float_input(node, "Transmission Dispersion Scale", 0.0)
    if dispersion_scale > _EPSILON:
        values["transmission_dispersion_scale"] = dispersion_scale
        values["transmission_dispersion_abbe_number"] = _float_input(
            node, "Transmission Dispersion Abbe Number", 20.0)
    _apply_coat_values(node, values)
    _apply_subsurface_values(node, values)
    _apply_specular_values(node, values)
    _apply_normal(node, values)
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


def _apply_folded_constants(
    node: Any,
    values: dict[str, Any],
    sockets: Iterable[tuple[str, str, str]],
) -> None:
    """Linked sockets without a texture whose upstream graph folds to a constant."""

    textures = values.get("textures", {})
    for socket_name, key, kind in sockets:
        if key in textures or not _socket_linked(_input_socket(node, socket_name)):
            continue
        folded = _linked_const(node, socket_name, kind)
        if folded is not None:
            values[key] = tuple(float(c) for c in folded) if kind == "color3" else float(folded[0])


def _apply_normal(node: Any, values: dict[str, Any]) -> None:
    """Normal map or bump texture, or a reported omission of an untranslatable bump."""

    normal = _normal_texture_record(node)
    if normal:
        values["textures"]["geometry_normal"] = normal
    elif _normal_bump_droppable(node):
        report = _dropped_bump_report(node)
        if report:
            values.setdefault("approximations", []).append(report)


def _normal_bump_droppable(node: Any) -> bool:
    """A Bump whose height is no image: its micro-normal is omitted and reported.

    Only a Bump feeding Normal with nothing chained into the Bump's own Normal
    qualifies; the surface keeps its geometric normal. A Bump over a constant
    height is flat, so dropping it is exact and nothing is reported.
    """

    source = _socket_link_target_node(_input_socket(node, "Normal"))
    if _node_type(source) != "BUMP":
        return False
    return not _socket_linked(_input_socket(source, "Normal"))


def _dropped_bump_report(node: Any) -> str:
    bump = _socket_link_target_node(_input_socket(node, "Normal"))
    if not _socket_linked(_input_socket(bump, "Height")):
        return ""
    height = _socket_link_target_node(_input_socket(bump, "Height"))
    types = sorted({_node_type(item) for item in _upstream_nodes(height)} - _COORDINATE_NODE_TYPES)
    return "geometry_normal:BUMP(" + "+".join(types) + ")"


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
    textures: dict[str, Any] = {}
    base_color = _color3_input(node, "Color", (0.8, 0.8, 0.8))
    texture = _texture_record_from_socket(node, "Color", slot="Base Color")
    if texture:
        textures["base_color"] = texture
    elif _socket_linked(_input_socket(node, "Color")):
        static_color = _linked_static_color(node, "Color")
        if static_color is not None:
            base_color = static_color
    values = {
        "base_color": base_color,
        "specular_roughness": 0.9,
        "base_metalness": 0.0,
        "geometry_opacity": 1.0,
        "specular_weight": 0.0,
        "textures": textures,
    }
    _apply_normal(node, values)
    return values


def _glossy_openpbr_values(node: Any) -> dict[str, Any]:
    """Glossy BSDF as a rough metal: its colour is the reflectance, roughness carries over.

    Blender's Glossy has no Fresnel term (colour is the reflected fraction), which
    is what an OpenPBR metal with that base colour reflects at every angle to
    first order. Mixed with a diffuse BSDF it becomes a dielectric specular layer
    instead (see ``_layer_glossy``).
    """

    values: dict[str, Any] = {
        "base_color": _color3_input(node, "Color", (0.8, 0.8, 0.8)),
        "specular_roughness": _resolved_float(node, "Roughness", 0.5),
        "base_metalness": 1.0,
        "geometry_opacity": 1.0,
        "textures": {},
    }
    textures = values["textures"]
    color = _texture_record_from_socket(node, "Color", slot="Base Color")
    if color:
        textures["base_color"] = color
    elif _socket_linked(_input_socket(node, "Color")):
        folded = _linked_static_color(node, "Color")
        if folded is not None:
            values["base_color"] = folded
    roughness = _texture_record_from_socket(node, "Roughness")
    if roughness:
        textures["specular_roughness"] = roughness
    anisotropy = _resolved_float(node, "Anisotropy", 0.0)
    if abs(anisotropy) > _EPSILON:
        values["specular_roughness_anisotropy"] = anisotropy
    _apply_normal(node, values)
    return values


def _glass_openpbr_values(node: Any) -> dict[str, Any]:
    """Glass BSDF as OpenPBR transmission with the glass colour as transmission colour."""

    color = _color3_input(node, "Color", (1.0, 1.0, 1.0))
    values: dict[str, Any] = {
        "base_color": color,
        "specular_roughness": _resolved_float(node, "Roughness", 0.0),
        "base_metalness": 0.0,
        "geometry_opacity": 1.0,
        "transmission_weight": 1.0,
        "textures": {},
    }
    textures = values["textures"]
    texture = _texture_record_from_socket(node, "Color", slot="Base Color")
    if texture:
        textures["base_color"] = texture
        values["transmission_color_from_base"] = True
    elif _socket_linked(_input_socket(node, "Color")):
        folded = _linked_static_color(node, "Color")
        if folded is not None:
            color = folded
            values["base_color"] = folded
    if any(abs(channel - 1.0) > _EPSILON for channel in color) or texture:
        values["transmission_color"] = color
    roughness = _texture_record_from_socket(node, "Roughness")
    if roughness:
        textures["specular_roughness"] = roughness
    ior = _resolved_float(node, "IOR", 1.5)
    if abs(ior - 1.5) > _EPSILON:
        values["specular_ior"] = ior
    _apply_normal(node, values)
    return values


def _translucent_openpbr_values(node: Any) -> dict[str, Any]:
    """Translucent BSDF: lit from the front like a diffuse surface.

    OpenPBR has no diffuse-transmission lobe, so light passing through the
    surface is omitted; this is reported as an approximation.
    """

    textures: dict[str, Any] = {}
    base_color = _color3_input(node, "Color", (0.8, 0.8, 0.8))
    texture = _texture_record_from_socket(node, "Color", slot="Base Color")
    if texture:
        textures["base_color"] = texture
    elif _socket_linked(_input_socket(node, "Color")):
        folded = _linked_static_color(node, "Color")
        if folded is not None:
            base_color = folded
    values = {
        "base_color": base_color,
        "specular_roughness": 0.9,
        "base_metalness": 0.0,
        "geometry_opacity": 1.0,
        "specular_weight": 0.0,
        "textures": textures,
        "approximations": ["BSDF_TRANSLUCENT:modelled_as_diffuse"],
    }
    _apply_normal(node, values)
    return values


def _mix_factor(node: Any) -> float | None:
    """Mix Shader factor when it is a constant (typed or folded from a node graph)."""

    for name in ("Factor", "Fac"):
        socket = _input_socket(node, name)
        if socket is None:
            continue
        if not _socket_linked(socket):
            return _float_input(node, name, 0.5)
        folded = _linked_const(node, name, "float")
        return float(folded[0]) if folded is not None else None
    return None


# Largest dielectric F0 a specular layer represents; stronger mixes use metalness.
_MAX_SPECULAR_LAYER_F0 = 0.25


def _layer_glossy(base: dict[str, Any], glossy: dict[str, Any], weight: float) -> dict[str, Any]:
    """Blend a Glossy BSDF into a diffuse surface as weight ``w`` of mirror reflection.

    Blender's ``Mix(Diffuse, Glossy, w)`` reflects ``w * colour`` at normal incidence.
    OpenPBR's dielectric specular reflects ``F0(ior) * specular_color`` there, so the
    IOR is solved from ``F0 = w`` (``ior = (1 + sqrt w) / (1 - sqrt w)``) and the glossy
    colour becomes the specular colour; the fresnel growth towards grazing angles is
    the remaining difference. Beyond ``F0 = 0.25`` the layer becomes partial
    metalness (a coarser match).
    """

    notes = list(base.get("approximations", ())) + list(glossy.get("approximations", ()))
    if weight <= _EPSILON:
        return {**base, "approximations": notes} if notes else base
    values = {**base, "textures": dict(base.get("textures", {}))}
    glossy_textures = glossy.get("textures", {})
    if "specular_roughness" in glossy_textures:
        values["textures"]["specular_roughness"] = glossy_textures["specular_roughness"]
    values["specular_roughness"] = glossy["specular_roughness"]
    if "specular_roughness_anisotropy" in glossy:
        values["specular_roughness_anisotropy"] = glossy["specular_roughness_anisotropy"]
    values.pop("specular_weight", None)
    glossy_color = tuple(glossy["base_color"])
    if "base_color" in glossy_textures or any(abs(channel - 1.0) > _EPSILON for channel in glossy_color):
        values["specular_color"] = glossy_color
    if "base_color" in glossy_textures and "base_color" not in values["textures"]:
        notes.append("MIX_SHADER:glossy_colour_texture_omitted")
    if weight <= _MAX_SPECULAR_LAYER_F0:
        root = math.sqrt(max(weight, 0.0))
        ior = (1.0 + root) / (1.0 - root)
        if abs(ior - 1.5) > _EPSILON:
            values["specular_ior"] = ior
        notes.append("MIX_SHADER:glossy_specular_layer_matched_at_normal_incidence")
    else:
        diffuse_color = tuple(base["base_color"])
        values["base_metalness"] = min(weight, 1.0)
        values["base_color"] = tuple(
            d * (1.0 - weight) + g * weight for d, g in zip(diffuse_color, glossy_color)
        )
        values.pop("specular_color", None)
        notes.append("MIX_SHADER:glossy_weight_above_0.25_modelled_as_partial_metalness")
    values["approximations"] = notes
    return values


def _glossy_mix_values(node: Any, branches: list[Any], glossy_index: int, depth: int) -> dict[str, Any] | None:
    factor = _mix_factor(node)
    if factor is None:
        return None
    other = branches[1 - glossy_index]
    if other is None or _node_type(other) not in {"BSDF_DIFFUSE", "BSDF_TRANSLUCENT"}:
        return None
    base = _openpbr_values_from_surface(other)
    if base is None:
        return None
    weight = factor if glossy_index == 1 else 1.0 - factor
    weight = min(max(weight, 0.0), 1.0)
    glossy = _glossy_openpbr_values(branches[glossy_index])
    if weight >= 1.0 - _EPSILON:
        return glossy
    return _layer_glossy(base, glossy, weight)


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
    glossy_index = next(
        (index for index, branch in enumerate(branches) if _node_type(branch) == "BSDF_GLOSSY"),
        None,
    )
    if glossy_index is not None and len(branches) == 2:
        return _glossy_mix_values(node, branches, glossy_index, depth)
    for index, branch in enumerate(branches):
        values = _openpbr_values_from_mixed_branch(branch, depth=depth - 1)
        if values is not None:
            # Only this branch is converted; any other connected shader is reported.
            omitted = [
                f"MIX_SHADER:omitted_{_node_type(other)}"
                for position, other in enumerate(branches)
                if position != index and other is not None
            ]
            if omitted:
                values = {**values, "approximations": [*values.get("approximations", ()), *omitted]}
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
    if node_type == "BSDF_DIFFUSE":
        if socket_name == "Color" and linked:
            if (
                _texture_record_from_socket(node, socket_name, slot="Base Color")
                or _linked_static_color(node, socket_name) is not None
            ):
                return {**record, "classification": "supported"}
            return {**record, "classification": "unsupported", "reason": "linked_supported_input:Color"}
        if socket_name == "Normal" and linked:
            if _normal_texture_record(node) or _normal_bump_droppable(node):
                return {**record, "classification": "supported"}
            return {**record, "classification": "unsupported", "reason": "linked_supported_input:Normal"}
        if socket_name == "Roughness" and linked:
            return {**record, "classification": "unsupported", "reason": "linked_unsupported_input:Roughness"}
        return record
    if node_type in {"BSDF_GLOSSY", "BSDF_GLASS", "BSDF_TRANSLUCENT"}:
        return _bsdf_socket_diagnostics(node, node_type, socket_name, linked, record)
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
            if _texture_record_from_socket(node, socket_name) or _folded_principled_input(node, socket_name):
                return {**record, "classification": "supported"}
            return {
                **record,
                "classification": "unsupported",
                "reason": f"linked_supported_input:{socket_name}",
            }
        return {**record, "classification": "supported"}
    if socket_name == "Normal":
        if linked:
            if _normal_texture_record(node) or _normal_bump_droppable(node):
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


def _bsdf_socket_diagnostics(
    node: Any, node_type: str, socket_name: str, linked: bool, record: dict[str, Any]
) -> dict[str, Any]:
    """Glossy, Glass and Translucent BSDF inputs: textures and folded constants are translated."""

    if not linked:
        return {**record, "classification": "supported"}
    if socket_name == "Color":
        if _texture_record_from_socket(node, "Color", slot="Base Color") or _linked_static_color(node, "Color") is not None:
            return {**record, "classification": "supported"}
        return {**record, "classification": "unsupported", "reason": "linked_supported_input:Color"}
    if socket_name == "Roughness" and node_type != "BSDF_TRANSLUCENT":
        if _texture_record_from_socket(node, "Roughness") or _linked_const(node, "Roughness", "float") is not None:
            return {**record, "classification": "supported"}
        return {**record, "classification": "unsupported", "reason": "linked_supported_input:Roughness"}
    if socket_name in {"IOR", "Anisotropy"} and node_type != "BSDF_TRANSLUCENT":
        if _linked_const(node, socket_name, "float") is not None:
            return {**record, "classification": "supported"}
        return {**record, "classification": "unsupported", "reason": f"linked_unsupported_input:{socket_name}"}
    if socket_name == "Normal":
        if _normal_texture_record(node) or _normal_bump_droppable(node):
            return {**record, "classification": "supported"}
        return {**record, "classification": "unsupported", "reason": "linked_supported_input:Normal"}
    return {**record, "classification": "unsupported", "reason": f"linked_unsupported_input:{socket_name}"}


def _folded_principled_input(node: Any, socket_name: str) -> bool:
    kind = {"Base Color": "color3", "Metallic": "float", "Roughness": "float", "Alpha": "float"}.get(socket_name)
    return kind is not None and _linked_const(node, socket_name, kind) is not None


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
    if values.get("geometry_thin_walled", False):
        lines.append("        bool inputs:geometry_thin_walled = true")
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
        if values.get("transmission_color_from_base") and "base_color" in textures:
            lines.append(_input_connect_line("color3f", "transmission_color", textures["base_color"]))
    for input_name in (
        "transmission_dispersion_scale",
        "transmission_dispersion_abbe_number",
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
        if record.get("network"):
            record["output_path"] = f"{material_path}/{record['network']['out']}.outputs:out"
            normalized[str(input_name)] = record
            continue
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
        if texture.get("network"):
            lines.extend(_network_shader_lines(material_path, texture["network"]))
            continue
        shader_name = str(texture["shader_name"])
        info_id = str(texture["info_id"])
        lines.extend(
            _image_shader_lines(
                material_path,
                shader_name,
                info_id,
                str(texture["asset_path"]),
                str(texture["color_space"]),
                str(texture["image_output_type"]),
                texture.get("uv_transform"),
                str(texture.get("address_mode", _DEFAULT_ADDRESS_MODE)),
            )
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
                lines.extend(_height_normal_lines(material_path, texture, shader_name))
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


def _height_normal_lines(material_path: str, texture: Mapping[str, Any], source_name: str) -> list[str]:
    """Height (float output of ``source_name``) to a shading normal for a Bump node."""

    height_to_normal_path = f"{material_path}/ND_heighttonormal_vector3_geometry_normal.outputs:out"
    return [
        "",
        '    def Shader "ND_heighttonormal_vector3_geometry_normal" (',
        '        prepend apiSchemas = ["NodeGraphNodeAPI"]',
        "    )",
        "    {",
        '        uniform token info:id = "ND_heighttonormal_vector3"',
        f"        float inputs:in.connect = <{material_path}/{source_name}.outputs:out>",
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


def _image_shader_lines(
    material_path: str,
    shader_name: str,
    info_id: str,
    asset_path: str,
    color_space: str,
    output_type: str,
    uv_transform: Mapping[str, Any] | None,
    address_mode: str,
) -> list[str]:
    """One MaterialX image node with its coordinate nodes and address mode."""

    lines: list[str] = []
    texcoord_lines: list[str] = []
    if uv_transform:
        uv_lines, uv_output = _uv_transform_shader_lines(material_path, shader_name, uv_transform)
        lines.extend(uv_lines)
        texcoord_lines = [f"        float2 inputs:texcoord.connect = <{material_path}/{uv_output}.outputs:out>"]
    address_lines = (
        [
            f'        string inputs:uaddressmode = "{address_mode}"',
            f'        string inputs:vaddressmode = "{address_mode}"',
        ]
        if address_mode != _DEFAULT_ADDRESS_MODE
        else []
    )
    lines.extend(
        [
            "",
            f'    def Shader "{_usda_string(shader_name)}" (',
            '        prepend apiSchemas = ["NodeGraphNodeAPI"]',
            "    )",
            "    {",
            f'        uniform token info:id = "{info_id}"',
            f"        asset inputs:file = @{_usda_asset_path(asset_path)}@ (",
            f'            colorSpace = "{color_space}"',
            "        )",
            *address_lines,
            *texcoord_lines,
            f"        {output_type} outputs:out",
            "    }",
        ]
    )
    return lines


_NETWORK_LITERALS = {
    "float": lambda value: _usda_float(value if not isinstance(value, (tuple, list)) else value[0]),
    "color3f": lambda value: _usda_color3(value),
    "float2": lambda value: f"({_usda_float(value[0])}, {_usda_float(value[1])})",
    "float3": lambda value: f"({_usda_float(value[0])}, {_usda_float(value[1])}, {_usda_float(value[2])})",
    "int": lambda value: str(int(value)),
    "bool": lambda value: "true" if value else "false",
    "string": lambda value: '"' + _usda_string(str(value)) + '"',
}


def _network_shader_lines(material_path: str, network: Mapping[str, Any]) -> list[str]:
    """Author the MaterialX nodes of one translated node network."""

    lines: list[str] = []
    for node in network["nodes"]:
        name = str(node["name"])
        image = node.get("image")
        if image:
            lines.extend(
                _image_shader_lines(
                    material_path,
                    name,
                    str(node["info_id"]),
                    str(image["asset_path"]),
                    str(image["color_space"]),
                    "color3f",
                    image.get("uv_transform"),
                    str(image.get("address_mode", _DEFAULT_ADDRESS_MODE)),
                )
            )
            continue
        body = []
        for row in node["inputs"]:
            input_type, input_name = str(row["type"]), str(row["name"])
            if row.get("connect"):
                body.append(
                    f"        {input_type} inputs:{input_name}.connect = "
                    f"<{material_path}/{row['connect']}.outputs:out>"
                )
            else:
                body.append(
                    f"        {input_type} inputs:{input_name} = "
                    f"{_NETWORK_LITERALS[input_type](row['const'])}"
                )
        lines.extend(
            [
                "",
                f'    def Shader "{_usda_string(name)}" (',
                '        prepend apiSchemas = ["NodeGraphNodeAPI"]',
                "    )",
                "    {",
                f'        uniform token info:id = "{node["info_id"]}"',
                *body,
                f"        {node['out_type']} outputs:out",
                "    }",
            ]
        )
    return lines


def _uv_transform_shader_lines(
    material_path: str,
    shader_name: str,
    uv_transform: Mapping[str, Any],
) -> tuple[list[str], str]:
    """Coordinate nodes feeding one image node and the name of the one to connect.

    A plain render UV is ``ND_texcoord_vector2``; a named UV map is read from its
    primvar with ``ND_geompropvalue_vector2``. A Blender Mapping in UV space adds
    ``texcoord * scale + offset``.
    """

    scale = tuple(float(value) for value in uv_transform["scale"])
    offset = tuple(float(value) for value in uv_transform["offset"])
    primvar = str(uv_transform.get("primvar", "") or "")
    steps = uv_transform.get("steps")
    identity = steps is None and all(
        abs(a - b) <= _EPSILON
        for pair, expected in zip((scale, offset), _IDENTITY_UV_TRANSFORM)
        for a, b in zip(pair, expected)
    )
    base = (
        ("uv", "ND_geompropvalue_vector2", [f'        string inputs:geomprop = "{_usda_string(primvar)}"'])
        if primvar
        else ("uv", "ND_texcoord_vector2", ["        int inputs:index = 0"])
    )
    stages = [base]
    if steps is not None:
        previous = f"{shader_name}_uv"
        for index, (kind, value) in enumerate(steps):
            suffix = f"uv_{index}_{kind}"
            if kind == "scale":
                info_id, body = "ND_multiply_vector2", [
                    f"        float2 inputs:in2 = ({_usda_float(value[0])}, {_usda_float(value[1])})"]
            elif kind == "offset":
                info_id, body = "ND_add_vector2", [
                    f"        float2 inputs:in2 = ({_usda_float(value[0])}, {_usda_float(value[1])})"]
            else:
                info_id, body = "ND_rotate2d_vector2", [f"        float inputs:amount = {_usda_float(value)}"]
            input_name = "in" if kind == "rotate" else "in1"
            stages.append((suffix, info_id, [
                f"        float2 inputs:{input_name}.connect = <{material_path}/{previous}.outputs:out>", *body]))
            previous = f"{shader_name}_{suffix}"
    elif not identity:
        stages.extend([
        ("uv_scale", "ND_multiply_vector2", [
            f"        float2 inputs:in1.connect = <{material_path}/{shader_name}_uv.outputs:out>",
            f"        float2 inputs:in2 = ({_usda_float(scale[0])}, {_usda_float(scale[1])})",
        ]),
        ("uv_offset", "ND_add_vector2", [
            f"        float2 inputs:in1.connect = <{material_path}/{shader_name}_uv_scale.outputs:out>",
            f"        float2 inputs:in2 = ({_usda_float(offset[0])}, {_usda_float(offset[1])})",
        ]),
        ])
    lines: list[str] = []
    for suffix, info_id, body in stages:
        lines.extend([
            "",
            f'    def Shader "{_usda_string(shader_name)}_{suffix}" (',
            '        prepend apiSchemas = ["NodeGraphNodeAPI"]',
            "    )",
            "    {",
            f'        uniform token info:id = "{info_id}"',
            *body,
            "        float2 outputs:out",
            "    }",
        ])
    return lines, f"{shader_name}_{stages[-1][0]}"


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


class _FlattenError(Exception):
    """A node group arrangement this conversion does not flatten."""


class _ConstantNode:
    """Stands in for an unconnected node-group input feeding the group's inner nodes."""

    def __init__(self, node_type: str, output_name: str, value: Any) -> None:
        self.type = node_type
        self.name = "Group Input constant"
        self.inputs: list[Any] = []
        self.outputs = {output_name: SimpleNamespace(name=output_name, default_value=value)}


@dataclass(frozen=True)
class _FlatLink:
    from_node: Any
    from_socket: Any


def _flattened_node_tree(node_tree: Any) -> tuple[list[Any], dict[int, Any] | None]:
    """Nodes and per-input-socket links of ``node_tree`` with node groups inlined.

    Each group node is replaced by its inner nodes: links leaving a Group node
    start at the node feeding the inner Group Output, and links leaving an inner
    Group Input start at whatever feeds the Group node's matching input (or a
    constant for an unconnected one). A group tree instanced twice in one
    material, or any other arrangement that cannot be resolved, keeps the Group
    node, which the converter then rejects as an unsupported node.
    """

    nodes = list(getattr(node_tree, "nodes", ()) or ())
    root_links = _node_tree_input_links(node_tree)
    if root_links is None or not any(
        _node_type(node) == "GROUP" and getattr(node, "node_tree", None) is not None for node in nodes
    ):
        return nodes, root_links
    flat_nodes: list[Any] = []
    flat_links: dict[int, Any] = {}
    used_trees: set[int] = set()

    def socket_key(socket: Any) -> str:
        return str(getattr(socket, "identifier", "") or getattr(socket, "name", ""))

    def group_context(group: Any, tree_links: dict[int, Any], context: dict[str, Any]) -> dict[str, Any]:
        inner: dict[str, Any] = {}
        for socket in _input_sockets(group):
            link = tree_links.get(_blender_identity(socket)) if _socket_linked(socket) else None
            if link is not None:
                resolved = resolve(link, tree_links, context)
                if resolved is not None:
                    inner[socket_key(socket)] = resolved
                continue
            socket_type = str(getattr(socket, "type", ""))
            if socket_type == "RGBA":
                node = _ConstantNode("RGB", "Color", tuple(getattr(socket, "default_value", ())))
                inner[socket_key(socket)] = _FlatLink(node, SimpleNamespace(name="Color"))
            elif socket_type == "VALUE":
                node = _ConstantNode("VALUE", "Value", float(getattr(socket, "default_value", 0.0)))
                inner[socket_key(socket)] = _FlatLink(node, SimpleNamespace(name="Value"))
        return inner

    def resolve(link: Any, tree_links: dict[int, Any], context: dict[str, Any]) -> Any | None:
        source = getattr(link, "from_node", None)
        source_type = _node_type(source)
        output = getattr(link, "from_socket", None)
        if source_type == "GROUP" and getattr(source, "node_tree", None) is not None:
            inner_tree = source.node_tree
            inner_links = _node_tree_input_links(inner_tree)
            if inner_links is None:
                raise _FlattenError("group tree has no links")
            group_output = next(
                (node for node in inner_tree.nodes
                 if _node_type(node) == "GROUP_OUTPUT" and bool(getattr(node, "is_active_output", True))),
                None,
            )
            if group_output is None:
                raise _FlattenError("group has no output node")
            target = next(
                (socket for socket in _input_sockets(group_output)
                 if socket_key(socket) == socket_key(output)),
                None,
            )
            inner_link = inner_links.get(_blender_identity(target)) if target is not None else None
            if inner_link is None:
                return None
            return resolve(inner_link, inner_links, group_context(source, tree_links, context))
        if source_type == "GROUP_INPUT":
            return context.get(socket_key(output))
        return link

    def walk(tree: Any, tree_links: dict[int, Any], context: dict[str, Any], depth: int) -> None:
        if depth > 6:
            raise _FlattenError("node groups nested too deeply")
        for node in tree.nodes:
            node_type = _node_type(node)
            if node_type == "GROUP" and getattr(node, "node_tree", None) is not None:
                inner_tree = node.node_tree
                if id(inner_tree) in used_trees:
                    raise _FlattenError("node group instanced more than once")
                used_trees.add(id(inner_tree))
                inner_links = _node_tree_input_links(inner_tree)
                if inner_links is None:
                    raise _FlattenError("group tree has no links")
                walk(inner_tree, inner_links, group_context(node, tree_links, context), depth + 1)
                continue
            if node_type in {"GROUP_INPUT", "GROUP_OUTPUT"}:
                continue
            flat_nodes.append(node)
            for socket in _input_sockets(node):
                link = tree_links.get(_blender_identity(socket))
                if link is None:
                    continue
                resolved = resolve(link, tree_links, context)
                if resolved is not None:
                    flat_links[_blender_identity(socket)] = resolved

    try:
        walk(node_tree, root_links, {}, 0)
    except _FlattenError:
        return nodes, root_links
    return flat_nodes, flat_links


def _node_type(node: Any | None) -> str:
    return str(getattr(node, "type", ""))


def _texture_record_from_socket(node: Any, socket_name: str, *, slot: str = "") -> dict[str, Any] | None:
    """Texture feeding ``socket_name``; ``slot`` names the _TEXTURE_INPUTS mapping
    when the socket name alone is ambiguous (Diffuse BSDF "Color" is base colour,
    Emission "Color" is emission colour).

    A chain of one image through the operations the image record applies keeps
    its compact record. Anything else is translated as a MaterialX node network
    (``record["network"]``); when the network is untranslatable the first-image
    record reports what it omits in ``approximated_ops``."""
    chain = _image_chain_texture_record(node, socket_name, slot=slot)
    if chain is not None and not chain.get("approximated_ops"):
        return chain
    network = _network_texture_record(node, socket_name, slot=slot, fallback=chain)
    return network if network is not None else chain


def _network_texture_record(
    node: Any, socket_name: str, *, slot: str, fallback: dict[str, Any] | None
) -> dict[str, Any] | None:
    spec = _TEXTURE_INPUTS.get(slot or socket_name)
    socket = _input_socket(node, socket_name)
    if spec is None or not _socket_linked(socket):
        return None
    openpbr_input, value_type, _info_id, _color_space = spec
    kind = "color3" if value_type == "color3f" else "float"
    try:
        net, value = _translate_cached(f"ND_net_{openpbr_input}", socket, kind)
    except node_network.Untranslatable as exc:
        if fallback is not None:
            fallback["network_failure"] = str(exc)
        return None
    if value.is_const:
        return None  # a constant: the caller reads it with _linked_const
    record = dict(fallback or {})
    record.update(
        {
            "openpbr_input": openpbr_input,
            "network": {
                "nodes": net.nodes,
                "out": value.node,
                "out_type": "color3f" if kind == "color3" else "float",
            },
            "approximated_ops": tuple(sorted(set(net.approximations))),
            "error_bounds": tuple(net.error_bounds),
            "invert": False,
            "extract_channel": None,
            "post_image_op": None,
        }
    )
    return record


def _linked_const(node: Any, socket_name: str, kind: str) -> tuple[float, ...] | None:
    """Value of a linked socket whose upstream network folds to a constant, else None."""

    socket = _input_socket(node, socket_name)
    if not _socket_linked(socket):
        return None
    try:
        _net, value = _translate_cached("ND_const", socket, kind)
    except node_network.Untranslatable:
        return None
    return value.const if value.is_const else None


def _resolved_float(node: Any, socket_name: str, default: float) -> float:
    """Float input honouring a linked socket that folds to a constant."""

    folded = _linked_const(node, socket_name, "float") if _socket_linked(_input_socket(node, socket_name)) else None
    return float(folded[0]) if folded is not None else _float_input(node, socket_name, default)


def _image_chain_texture_record(node: Any, socket_name: str, *, slot: str = "") -> dict[str, Any] | None:
    slot = _TEXTURE_INPUTS.get(slot or socket_name)
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
    record = {
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
        "uv_transform": _image_uv_transform(texture_node),
        **_image_address_modes(texture_node),
    }
    applied = set()
    if record["post_image_op"]:
        applied.add("GAMMA" if value_type == "color3f" else "MATH")
    if record["invert"]:
        applied.add("INVERT")
    if record["extract_channel"] is not None:
        applied.update({"SEPARATE_COLOR", "SEPRGB"})
    record["approximated_ops"] = _approximated_texture_ops(socket, applied)
    return record


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
        "uv_transform": _image_uv_transform(texture_node),
        **_image_address_modes(texture_node),
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
        # Blender scales the height gradient by Distance; Strength then blends
        # towards the bumped normal, so for the small strengths bumps use the
        # slope is Strength * Distance. Ignoring Distance (default scenes use
        # 0.1) made image bumps about ten times too strong.
        scale = _float_input(source, "Strength", 1.0) * _float_input(source, "Distance", 1.0)
        if bool(getattr(source, "invert", False)):
            scale = -scale
    texture_node = _linked_texture_node(texture_socket)
    if texture_node is None:
        return None
    asset_path = _image_asset_path(getattr(texture_node, "image", None))
    if not asset_path:
        return None
    shader_name = "ND_image_float_geometry_normal" if bump else "ND_image_color3_geometry_normal"
    record = {
        "openpbr_input": "geometry_normal",
        "shader_name": shader_name,
        "info_id": "ND_image_float" if bump else "ND_image_color3",
        "asset_path": asset_path,
        "color_space": _image_color_space(getattr(texture_node, "image", None), "raw"),
        "value_type": "float" if bump else "color3f",
        "image_output_type": "float" if bump else "color3f",
        "scale": scale,
        "bump": bump,
        "uv_transform": _image_uv_transform(texture_node),
        **_image_address_modes(texture_node),
    }
    record["approximated_ops"] = _approximated_texture_ops(texture_socket, set())
    return record


def _linked_texture_node(socket: Any | None, *, depth: int = 8) -> Any | None:
    node, _ = _linked_texture_info(socket, depth=depth)
    return node


_TRANSPARENT_CHAIN_NODES = frozenset({"REROUTE"})


def _approximated_texture_ops(socket: Any | None, applied: set[str]) -> tuple[str, ...]:
    """Node types between ``socket`` and its texture that the generated image omits.

    The texture walk follows a chain to the first image it reaches (the
    long-standing Principled policy, also used for Diffuse BSDF). Operations it
    does not reproduce are reported here so conversion never approximates
    silently; ``applied`` names the ones the record translated.
    """

    start = _socket_link_target_node(socket)
    path = _texture_path(start)
    reported = [
        _node_type(node) for node in path
        if _node_type(node) not in _TRANSPARENT_CHAIN_NODES and _node_type(node) not in applied
    ]
    on_path = {_blender_identity(node) for node in path}
    for other in _upstream_nodes(start):
        other_type = _node_type(other)
        if (
            _blender_identity(other) not in on_path
            and other_type not in _COORDINATE_NODE_TYPES
            and other_type not in _TRANSPARENT_CHAIN_NODES
            and other_type != "TEX_IMAGE"
            and other_type not in reported
        ):
            reported.append(other_type)
    return tuple(reported)


_COORDINATE_NODE_TYPES = frozenset({"MAPPING", "TEX_COORD", "UVMAP"})


def _upstream_nodes(start: Any | None, *, depth: int = 12) -> list[Any]:
    """Every node upstream of ``start`` except what hangs below an image's coordinates."""

    found: list[Any] = []
    seen: set[int] = set()

    def visit(node: Any | None, remaining: int) -> None:
        if node is None or remaining <= 0 or _blender_identity(node) in seen:
            return
        seen.add(_blender_identity(node))
        found.append(node)
        if _node_type(node) == "TEX_IMAGE":
            return
        for socket in _input_sockets(node):
            visit(_socket_link_target_node(socket), remaining - 1)

    visit(start, depth)
    return found


def _texture_path(node: Any | None, *, depth: int = 8) -> tuple[Any, ...]:
    """Nodes the texture walk passes through before its image (image excluded)."""

    if node is None or depth <= 0:
        return ()
    if _node_type(node) == "TEX_IMAGE":
        return ()
    for socket in _texture_chain_inputs(node):
        child = _socket_link_target_node(socket)
        if _find_texture_node(child, depth=depth - 1) is not None:
            return (node, *_texture_path(child, depth=depth - 1))
    return (node,)


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
    folded = _linked_const(node, socket_name, "color3")
    if folded is not None:
        return (float(folded[0]), float(folded[1]), float(folded[2]))
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
        return sky_texture_radiance(node)[0]
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
            from .texture_materialization import packed_image_suffix
            payload = bytes(packed_data)
            suffix = packed_image_suffix(
                payload, Path(str(getattr(image, "filepath", "") or "")).suffix or target.suffix,
            )
            target = target.with_suffix(suffix)
            temporary = target.with_suffix(target.suffix + ".tmp")
            temporary.write_bytes(payload)
            temporary.replace(target)
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

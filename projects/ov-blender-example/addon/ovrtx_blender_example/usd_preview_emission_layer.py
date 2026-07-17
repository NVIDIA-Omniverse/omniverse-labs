# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""USDPreviewSurface material repair for Blender USD exports."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import materialx_openpbr_conversion
from . import ovrtx_scene_composition
from .render_requests import MaterialPresentationLayer


_SOURCE = "usd_preview_material_repair"


@dataclass(frozen=True)
class _EmissionRecord:
    material_name: str
    material_path: str
    preview_surface_path: str
    texture_path: str
    scale: float = 1.0
    color_space: str = "sRGB"
    uv_path: str = ""


@dataclass(frozen=True)
class _DiffuseRecord:
    material_name: str
    material_path: str
    preview_surface_path: str
    texture_path: str
    asset_path: str = ""
    color_space: str = "sRGB"
    uv_path: str = ""


@dataclass(frozen=True)
class _OpacityRecord:
    material_name: str
    preview_surface_path: str
    texture_path: str


@dataclass(frozen=True)
class _SurfaceFallbackRecord:
    material_name: str
    material_path: str
    shader_path: str


def scene_layer_from_materials(
    materials: Iterable[Any],
    input_usd_path: str,
) -> MaterialPresentationLayer | None:
    """Patch USDPreview material links Blender's USD exporter leaves out."""

    selected = tuple(materials)
    if not selected or not input_usd_path:
        return None
    identity = materialx_openpbr_conversion._materialx_binding_identity(input_usd_path)
    if not identity.get("available", False):
        return None
    material_index = _usd_preview_material_index(input_usd_path)
    if not material_index:
        return None

    emission_records: list[_EmissionRecord] = []
    diffuse_records: list[_DiffuseRecord] = []
    opacity_records: list[_OpacityRecord] = []
    surface_records: list[_SurfaceFallbackRecord] = []
    skipped: list[dict[str, str]] = []
    seen_material_paths: set[str] = set()
    for material in selected:
        binding = materialx_openpbr_conversion._resolve_binding(material, identity)
        material_name = str(binding.get("material_name", ""))
        material_path = str(binding.get("material_path", ""))
        if not material_path or material_path in seen_material_paths:
            continue
        seen_material_paths.add(material_path)
        material_info = material_index.get(material_path)
        if material_info is None:
            surface_record = _surface_fallback_record(material_name, material_path, material)
            if surface_record is not None:
                surface_records.append(surface_record)
                continue
            skipped.append(
                {
                    "material_name": material_name,
                    "reason": "missing_usd_preview_surface",
                }
            )
            continue
        classified = materialx_openpbr_conversion._classify_material(
            material,
            tuple(str(path) for path in binding.get("binding_targets", ()) if str(path)),
            (),
            identity,
        )
        diffuse_record = _diffuse_repair_record(
            material_name,
            material_path,
            material_info,
            classified,
        )
        if diffuse_record is not None:
            diffuse_records.append(diffuse_record)
        opacity_record = _opacity_repair_record(material_name, material_info)
        if opacity_record is not None:
            opacity_records.append(opacity_record)
        if material_info.get("emissive_connection"):
            continue
        texture = _emission_texture_record(classified)
        if texture is None:
            continue
        asset_path = str(texture.get("asset_path", ""))
        if not asset_path:
            continue
        emission_records.append(
            _EmissionRecord(
                material_name=material_name,
                material_path=material_path,
                preview_surface_path=str(material_info["preview_surface_path"]),
                texture_path=asset_path,
                scale=_emission_scale(classified),
                color_space=str(texture.get("color_space", "sRGB") or "sRGB"),
                uv_path=str(material_info.get("uv_path", "")),
            )
        )
    return _layer_from_records(
        emission_records,
        diffuse_records,
        opacity_records,
        surface_records,
        skipped,
    )


def _usd_preview_material_index(input_usd_path: str) -> dict[str, dict[str, Any]]:
    try:
        from pxr import Usd  # type: ignore
    except ModuleNotFoundError:
        return {}
    stage = Usd.Stage.Open(str(input_usd_path))
    if stage is None:
        return {}
    result: dict[str, dict[str, Any]] = {}
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Material":
            continue
        preview_surface_path = ""
        uv_path = ""
        diffuse_connection = ""
        emissive_connection = ""
        opacity_connection = ""
        textures_by_path: dict[str, dict[str, Any]] = {}
        textures_by_stem: dict[str, str] = {}
        for child in prim.GetChildren():
            if child.GetTypeName() != "Shader":
                continue
            shader_id = child.GetAttribute("info:id").Get()
            if shader_id == "UsdPreviewSurface":
                preview_surface_path = str(child.GetPath())
                diffuse = child.GetAttribute("inputs:diffuseColor")
                connections = diffuse.GetConnections() if diffuse and diffuse.IsValid() else ()
                diffuse_connection = str(connections[0].GetPrimPath()) if connections else ""
                emissive = child.GetAttribute("inputs:emissiveColor")
                connections = emissive.GetConnections() if emissive and emissive.IsValid() else ()
                emissive_connection = str(connections[0].GetPrimPath()) if connections else ""
                opacity = child.GetAttribute("inputs:opacity")
                connections = opacity.GetConnections() if opacity and opacity.IsValid() else ()
                opacity_connection = str(connections[0].GetPrimPath()) if connections else ""
            elif shader_id == "UsdPrimvarReader_float2" and not uv_path:
                uv_path = str(child.GetPath())
            elif shader_id == "UsdUVTexture":
                file_attr = child.GetAttribute("inputs:file")
                asset_path = _asset_path_text(file_attr.Get()) if file_attr else ""
                texture_path = str(child.GetPath())
                textures_by_path[texture_path] = {
                    "asset_path": asset_path,
                    "stem": Path(asset_path).stem,
                    "has_bias": bool(child.HasAttribute("inputs:bias")),
                    "has_scale": bool(child.HasAttribute("inputs:scale")),
                }
                if asset_path:
                    textures_by_stem.setdefault(Path(asset_path).stem, texture_path)
        if preview_surface_path:
            result[str(prim.GetPath())] = {
                "preview_surface_path": preview_surface_path,
                "uv_path": uv_path,
                "diffuse_connection": diffuse_connection,
                "emissive_connection": emissive_connection,
                "opacity_connection": opacity_connection,
                "textures_by_path": textures_by_path,
                "textures_by_stem": textures_by_stem,
            }
    return result


def _emission_texture_record(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    values = record.get("openpbr_values")
    if not isinstance(values, Mapping):
        return None
    textures = values.get("textures")
    if not isinstance(textures, Mapping):
        return None
    texture = textures.get("emission_color")
    return texture if isinstance(texture, Mapping) else None


def _base_color_texture_record(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    values = record.get("openpbr_values")
    if not isinstance(values, Mapping):
        return None
    textures = values.get("textures")
    if not isinstance(textures, Mapping):
        return None
    texture = textures.get("base_color")
    return texture if isinstance(texture, Mapping) else None


def _emission_scale(record: Mapping[str, Any]) -> float:
    values = record.get("openpbr_values")
    if not isinstance(values, Mapping):
        return 1.0
    luminance = values.get("emission_luminance")
    if luminance is None:
        return 1.0
    return max(
        0.0,
        float(luminance) / float(materialx_openpbr_conversion._EMISSION_LUMINANCE_SCALE),
    )


def _diffuse_repair_record(
    material_name: str,
    material_path: str,
    material_info: Mapping[str, Any],
    classified: Mapping[str, Any],
) -> _DiffuseRecord | None:
    textures_by_path = material_info.get("textures_by_path")
    textures_by_stem = material_info.get("textures_by_stem")
    if not isinstance(textures_by_path, Mapping) or not isinstance(textures_by_stem, Mapping):
        return None
    diffuse_path = str(material_info.get("diffuse_connection", ""))
    diffuse_texture = textures_by_path.get(diffuse_path)
    if not isinstance(diffuse_texture, Mapping):
        return None
    stem = str(diffuse_texture.get("stem", ""))
    if "_SSS" not in stem:
        return None
    base_path = str(textures_by_stem.get(stem.replace("_SSS", "_BaseColor"), ""))
    if base_path:
        return _DiffuseRecord(
            material_name=material_name,
            material_path=material_path,
            preview_surface_path=str(material_info["preview_surface_path"]),
            texture_path=base_path,
        )
    texture = _base_color_texture_record(classified)
    if texture is None:
        return None
    asset_path = str(texture.get("asset_path", ""))
    if not asset_path:
        return None
    return _DiffuseRecord(
        material_name=material_name,
        material_path=material_path,
        preview_surface_path=str(material_info["preview_surface_path"]),
        texture_path=f"{material_path}/OVRTX_BaseColor_Texture",
        asset_path=asset_path,
        color_space=str(texture.get("color_space", "sRGB") or "sRGB"),
        uv_path=str(material_info.get("uv_path", "")),
    )


def _opacity_repair_record(
    material_name: str,
    material_info: Mapping[str, Any],
) -> _OpacityRecord | None:
    textures_by_path = material_info.get("textures_by_path")
    if not isinstance(textures_by_path, Mapping):
        return None
    opacity_path = str(material_info.get("opacity_connection", ""))
    opacity_texture = textures_by_path.get(opacity_path)
    if not isinstance(opacity_texture, Mapping):
        return None
    stem = str(opacity_texture.get("stem", "")).lower()
    if "opacity" not in stem:
        return None
    if not bool(opacity_texture.get("has_bias")) and not bool(opacity_texture.get("has_scale")):
        return None
    return _OpacityRecord(
        material_name=material_name,
        preview_surface_path=str(material_info["preview_surface_path"]),
        texture_path=opacity_path,
    )


def _surface_fallback_record(
    material_name: str,
    material_path: str,
    material: Any,
) -> _SurfaceFallbackRecord | None:
    node_tree = getattr(material, "node_tree", None)
    if node_tree is None:
        return None
    outputs = [
        node
        for node in getattr(node_tree, "nodes", ())
        if str(getattr(node, "type", "")) == "OUTPUT_MATERIAL"
        and bool(getattr(node, "is_active_output", False))
    ]
    if not outputs:
        return None
    inputs = getattr(outputs[0], "inputs", {})
    surface = inputs.get("Surface") if hasattr(inputs, "get") else None
    volume = inputs.get("Volume") if hasattr(inputs, "get") else None
    if bool(getattr(surface, "is_linked", False)) or not bool(
        getattr(volume, "is_linked", False)
    ):
        return None
    return _SurfaceFallbackRecord(
        material_name=material_name,
        material_path=material_path,
        shader_path=f"{material_path}/OVRTX_Volume_Surface",
    )


def _layer_from_records(
    emission_records: Iterable[_EmissionRecord],
    diffuse_records: Iterable[_DiffuseRecord] = (),
    opacity_records: Iterable[_OpacityRecord] = (),
    surface_records: Iterable[_SurfaceFallbackRecord] = (),
    skipped: Iterable[Mapping[str, str]] = (),
) -> MaterialPresentationLayer | None:
    selected_emission = tuple(emission_records)
    selected_diffuse = tuple(diffuse_records)
    selected_opacity = tuple(opacity_records)
    selected_surface = tuple(surface_records)
    if (
        not selected_emission
        and not selected_diffuse
        and not selected_opacity
        and not selected_surface
    ):
        return None
    definitions: list[tuple[str, str, list[str]]] = []
    for record in selected_emission:
        texture_path = f"{record.material_path}/OVRTX_Emission_Texture"
        texture_attributes = [
            'uniform token info:id = "UsdUVTexture"',
            "asset inputs:file = "
            f"@{materialx_openpbr_conversion._usda_asset_path(record.texture_path)}@",
            f'token inputs:sourceColorSpace = "{_usda_string(record.color_space)}"',
        ]
        if abs(record.scale - 1.0) > 1.0e-6:
            texture_attributes.append(
                f"float4 inputs:scale = ({_float(record.scale)}, {_float(record.scale)}, {_float(record.scale)}, 1)"
            )
        if record.uv_path:
            texture_attributes.append(
                f"float2 inputs:st.connect = <{record.uv_path}.outputs:result>"
            )
        texture_attributes.append("color3f outputs:rgb")
        definitions.append((texture_path, "Shader", texture_attributes))
        definitions.append(
            (
                record.preview_surface_path,
                "",
                [
                    "color3f inputs:emissiveColor.connect = "
                    f"<{texture_path}.outputs:rgb>"
                ],
            )
        )
    for record in selected_diffuse:
        if record.asset_path:
            texture_attributes = [
                'uniform token info:id = "UsdUVTexture"',
                "asset inputs:file = "
                f"@{materialx_openpbr_conversion._usda_asset_path(record.asset_path)}@",
                f'token inputs:sourceColorSpace = "{_usda_string(record.color_space)}"',
            ]
            if record.uv_path:
                texture_attributes.append(
                    f"float2 inputs:st.connect = <{record.uv_path}.outputs:result>"
                )
            texture_attributes.append("color3f outputs:rgb")
            definitions.append((record.texture_path, "Shader", texture_attributes))
        definitions.append(
            (
                record.preview_surface_path,
                "",
                [
                    "color3f inputs:diffuseColor.connect = "
                    f"<{record.texture_path}.outputs:rgb>"
                ],
            )
        )
    for record in selected_opacity:
        definitions.append(
            (
                record.texture_path,
                "",
                [
                    "float4 inputs:bias = (0, 0, 0, 0)",
                    "float4 inputs:scale = (1, 1, 1, 1)",
                ],
            )
        )
        definitions.append(
            (
                record.preview_surface_path,
                "",
                [
                    "float inputs:opacity = 1",
                    "float inputs:opacity.connect = "
                    f"<{record.texture_path}.outputs:r>",
                ],
            )
        )
    for record in selected_surface:
        definitions.append(
            (
                record.shader_path,
                "Shader",
                [
                    'uniform token info:id = "UsdPreviewSurface"',
                    "float inputs:clearcoat = 0",
                    "float inputs:clearcoatRoughness = 0.03",
                    "color3f inputs:diffuseColor = (0.8, 0.8, 0.8)",
                    "float inputs:ior = 1.5",
                    "float inputs:metallic = 0",
                    "float inputs:opacity = 1",
                    "float inputs:roughness = 0.5",
                    "float inputs:specular = 0.5",
                    "token outputs:surface",
                ],
            )
        )
        definitions.append(
            (
                record.material_path,
                "",
                [
                    "token outputs:surface.connect = "
                    f"<{record.shader_path}.outputs:surface>"
                ],
            )
        )
    layer_body = "\n".join(
        ovrtx_scene_composition._usda_typed_def_tree(_merged_definitions(definitions))
    ).rstrip()
    digest_records: list[dict[str, Any]] = [
        {
            "kind": "emission",
            "material_name": record.material_name,
            "material_path": record.material_path,
            "preview_surface_path": record.preview_surface_path,
            "texture_path": record.texture_path,
            "scale": record.scale,
            "color_space": record.color_space,
            "uv_path": record.uv_path,
        }
        for record in selected_emission
    ] + [
        {
            "kind": "diffuse",
            "material_name": record.material_name,
            "material_path": record.material_path,
            "preview_surface_path": record.preview_surface_path,
            "texture_path": record.texture_path,
            "asset_path": record.asset_path,
            "color_space": record.color_space,
            "uv_path": record.uv_path,
        }
        for record in selected_diffuse
    ] + [
        {
            "kind": "opacity",
            "material_name": record.material_name,
            "preview_surface_path": record.preview_surface_path,
            "texture_path": record.texture_path,
        }
        for record in selected_opacity
    ] + [
        {
            "kind": "surface_fallback",
            "material_name": record.material_name,
            "material_path": record.material_path,
            "shader_path": record.shader_path,
        }
        for record in selected_surface
    ]
    target_paths = [
        record.preview_surface_path
        for record in (*selected_emission, *selected_diffuse, *selected_opacity)
    ] + [record.material_path for record in selected_surface]
    return MaterialPresentationLayer(
        target_path=min(target_paths),
        layer_body=layer_body,
        authored_properties=tuple(
            (record.preview_surface_path, "inputs:emissiveColor")
            for record in selected_emission
        )
        + tuple(
            (record.preview_surface_path, "inputs:diffuseColor")
            for record in selected_diffuse
        )
        + tuple((record.texture_path, "inputs:bias") for record in selected_opacity)
        + tuple((record.texture_path, "inputs:scale") for record in selected_opacity)
        + tuple(
            (record.preview_surface_path, "inputs:opacity")
            for record in selected_opacity
        )
        + tuple(
            (record.material_path, "outputs:surface")
            for record in selected_surface
        ),
        digest_content={"source": _SOURCE, "records": digest_records},
        diagnostics={
            "source": _SOURCE,
            "status": "generated",
            "digest": hashlib.sha256(
                json.dumps(digest_records, sort_keys=True).encode("utf-8")
            ).hexdigest()[:16],
            "material_count": len({record["material_name"] for record in digest_records}),
            "materials": sorted({record["material_name"] for record in digest_records}),
            "skipped_materials": [dict(item) for item in skipped],
        },
    )


def _usda_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _merged_definitions(
    definitions: Iterable[tuple[str, str, list[str]]],
) -> list[tuple[str, str, list[str]]]:
    merged: dict[str, tuple[str, list[str]]] = {}
    order = []
    for path, type_name, lines in definitions:
        if path not in merged:
            merged[path] = (type_name, list(lines))
            order.append(path)
            continue
        existing_type, existing_lines = merged[path]
        if existing_type and type_name and existing_type != type_name:
            raise ValueError(f"conflicting USD definition types for {path}")
        existing_lines.extend(lines)
        merged[path] = (existing_type or type_name, existing_lines)
    return [(path, *merged[path]) for path in order]


def _asset_path_text(value: Any) -> str:
    return str(getattr(value, "path", value) or "").strip("@")


def _float(value: Any) -> str:
    return f"{float(value):.9g}"


__all__ = ["scene_layer_from_materials"]

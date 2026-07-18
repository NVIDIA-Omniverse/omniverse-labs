#!/usr/bin/env python3
"""Print a read-only JSON audit of Blender light and World authoring state."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import bpy


SUPPORTED_TYPES = {"POINT", "SPOT", "SUN", "AREA"}


def _finite(values) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _safe_number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _socket_default(socket):
    """Resolve direct constant links; return (value, indeterminate)."""
    if socket is None:
        return None, False
    if not socket.is_linked:
        return socket.default_value, False
    links = list(socket.links)
    if len(links) == 1 and hasattr(links[0].from_socket, "default_value"):
        return links[0].from_socket.default_value, False
    return None, True


def _args() -> argparse.Namespace:
    raw = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene")
    parser.add_argument("--lights", nargs="*")
    parser.add_argument("--require-effective-lighting", action="store_true")
    parser.add_argument("--allow-unsupported-types", action="store_true")
    return parser.parse_args(raw)


def _reachable_upstream(output) -> set[Any]:
    reachable = {output}
    pending = [output]
    while pending:
        node = pending.pop()
        for socket in node.inputs:
            for link in socket.links:
                if link.from_node not in reachable:
                    reachable.add(link.from_node)
                    pending.append(link.from_node)
    return reachable


def _light_record(obj, require_supported: bool) -> dict[str, Any]:
    data = obj.data
    issues: list[str] = []
    warnings: list[str] = []
    matrix = [value for row in obj.matrix_world for value in row]
    color = list(data.color)
    energy = float(data.energy)
    if not _finite(matrix):
        issues.append("nonfinite_world_transform")
    if not _finite(color) or any(value < 0.0 for value in color):
        issues.append("invalid_color")
    if not math.isfinite(energy) or energy < 0.0:
        issues.append("invalid_energy")
    if require_supported and data.type not in SUPPORTED_TYPES:
        issues.append("unsupported_light_type")
    elif data.type not in SUPPORTED_TYPES:
        warnings.append("light_type_requires_addon_capability_check")

    type_fields: dict[str, Any] = {}
    if data.type in {"POINT", "SPOT"}:
        type_fields["shadow_soft_size"] = float(data.shadow_soft_size)
        if not math.isfinite(type_fields["shadow_soft_size"]) or type_fields["shadow_soft_size"] < 0.0:
            issues.append("invalid_shadow_soft_size")
    if data.type == "SPOT":
        type_fields.update(spot_size=float(data.spot_size), spot_blend=float(data.spot_blend))
        if not 0.0 < type_fields["spot_size"] <= math.pi:
            issues.append("invalid_spot_size")
        if not 0.0 <= type_fields["spot_blend"] <= 1.0:
            issues.append("invalid_spot_blend")
    if data.type == "SUN":
        type_fields["angle"] = float(data.angle)
        if not 0.0 <= type_fields["angle"] <= math.pi:
            issues.append("invalid_sun_angle")
    if data.type == "AREA":
        type_fields.update(shape=str(data.shape), size=float(data.size))
        if not math.isfinite(type_fields["size"]) or type_fields["size"] <= 0.0:
            issues.append("invalid_area_size")
        if data.shape in {"RECTANGLE", "ELLIPSE"}:
            type_fields["size_y"] = float(data.size_y)
            if not math.isfinite(type_fields["size_y"]) or type_fields["size_y"] <= 0.0:
                issues.append("invalid_area_size_y")

    return {
        "name": obj.name,
        "data": data.name,
        "type": data.type,
        "energy": _safe_number(energy),
        "color": color,
        "visible_render": not obj.hide_render,
        "type_fields": type_fields,
        "effective": not obj.hide_render and energy > 0.0 and max(color, default=0.0) > 0.0,
        "issues": issues,
        "warnings": warnings,
        "ok": not issues,
    }


def _world_record(world) -> dict[str, Any]:
    if world is None:
        return {"name": None, "effective": False, "issues": [], "warnings": ["scene_has_no_world"], "ok": True}

    issues: list[str] = []
    warnings: list[str] = []
    node_tree = world.node_tree
    if node_tree is None:
        return {
            "name": world.name,
            "effective": False,
            "backgrounds": [],
            "environments": [],
            "issues": ["world_has_no_node_tree"],
            "warnings": [],
            "ok": False,
        }
    outputs = [node for node in node_tree.nodes if node.type == "OUTPUT_WORLD" and getattr(node, "is_active_output", True)]
    output = outputs[0] if outputs else None
    if output is None:
        issues.append("missing_active_world_output")
        reachable = set()
    else:
        surface = output.inputs.get("Surface")
        if surface is None or not surface.is_linked:
            issues.append("world_surface_unlinked")
        reachable = _reachable_upstream(output)
    if len(outputs) > 1:
        warnings.append("multiple_active_world_outputs")

    backgrounds = [node for node in reachable if node.type == "BACKGROUND"]
    environments = [node for node in reachable if node.type == "TEX_ENVIRONMENT"]
    if not backgrounds:
        issues.append("no_reachable_background")
    background_records = []
    effective = False
    for node in backgrounds:
        color_socket = node.inputs.get("Color")
        strength_socket = node.inputs.get("Strength")
        color_value, color_indeterminate = _socket_default(color_socket)
        strength_value, strength_indeterminate = _socket_default(strength_socket)
        color = list(color_value) if color_value is not None and hasattr(color_value, "__iter__") else []
        strength = float(strength_value) if strength_value is not None else float("nan")
        linked_color = bool(color_socket and color_socket.is_linked)
        node_issues = []
        if strength_indeterminate or color_indeterminate:
            warnings.append(f"background:{node.name}:linked_value_indeterminate")
        elif not math.isfinite(strength) or strength < 0.0:
            node_issues.append("invalid_background_strength")
        if not linked_color and (not _finite(color) or any(value < 0.0 for value in color[:3])):
            node_issues.append("invalid_background_color")
        if not node_issues and not strength_indeterminate and not color_indeterminate and strength > 0.0 and max(color[:3], default=0.0) > 0.0:
            effective = True
        issues.extend(f"background:{node.name}:{issue}" for issue in node_issues)
        background_records.append({"name": node.name, "strength": _safe_number(strength), "color": color,
                                   "color_linked": linked_color,
                                   "indeterminate": strength_indeterminate or color_indeterminate,
                                   "issues": node_issues})

    environment_records = []
    for node in environments:
        image = node.image
        node_issues = []
        exists = False
        packed = bool(image and image.packed_file)
        filepath = None
        if image is None:
            node_issues.append("environment_has_no_image")
        else:
            filepath = bpy.path.abspath(image.filepath, library=image.library)
            exists = packed or (bool(filepath) and Path(filepath).is_file())
            if not exists:
                node_issues.append("environment_image_missing")
        issues.extend(f"environment:{node.name}:{issue}" for issue in node_issues)
        environment_records.append({"name": node.name, "image": image.name if image else None, "filepath": filepath, "packed": packed, "exists": exists, "issues": node_issues})

    return {
        "name": world.name,
        "effective": effective and all(not record["issues"] for record in environment_records),
        "backgrounds": background_records,
        "environments": environment_records,
        "issues": issues,
        "warnings": warnings,
        "ok": not issues,
    }


def audit(request: dict[str, Any] | None = None) -> dict[str, Any]:
    request = dict(request or {})
    scene_name = request.get("scene")
    scene = bpy.data.scenes.get(scene_name) if scene_name else bpy.context.scene
    if scene is None:
        return {"schema": "blender_lighting_world_audit.v1", "ok": False, "checks": {"scene_resolved": False}, "scene": scene_name, "missing_lights": [], "lights": [], "world": None}

    requested_names = request.get("lights")
    missing: list[str] = []
    objects = []
    if requested_names is None:
        objects = [obj for obj in scene.objects if obj.type == "LIGHT" and not obj.hide_render]
    else:
        for name in requested_names:
            obj = scene.objects.get(name)
            if obj is None or obj.type != "LIGHT":
                missing.append(name)
            else:
                objects.append(obj)
    require_supported = not bool(request.get("allow_unsupported_types", False))
    lights = [_light_record(obj, require_supported) for obj in objects]
    world = _world_record(scene.world)
    effective = any(record["effective"] for record in lights) or world["effective"]
    checks = {
        "scene_resolved": True,
        "all_requested_lights_resolved": not missing,
        "all_lights_valid": all(record["ok"] for record in lights),
        "world_valid": world["ok"],
        "effective_lighting": effective if request.get("require_effective_lighting", False) else True,
    }
    return {
        "schema": "blender_lighting_world_audit.v1",
        "ok": all(checks.values()),
        "blender_version": bpy.app.version_string,
        "scene": scene.name,
        "checks": checks,
        "missing_lights": missing,
        "lights": lights,
        "world": world,
        "boundary": "Blender authoring state only; native OVRTX conversion is not tested.",
    }


if "LIGHTING_AUDIT_REQUEST" in globals():
    print(json.dumps(audit(LIGHTING_AUDIT_REQUEST), sort_keys=True))
elif __name__ == "__main__":
    args = _args()
    request = {
        "scene": args.scene,
        "lights": args.lights if args.lights else None,
        "require_effective_lighting": args.require_effective_lighting,
        "allow_unsupported_types": args.allow_unsupported_types,
    }
    result = audit(request)
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["ok"] else 2)

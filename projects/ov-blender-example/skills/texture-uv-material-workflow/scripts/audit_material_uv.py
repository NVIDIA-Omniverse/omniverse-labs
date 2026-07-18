"""Print a read-only JSON audit of visible mesh UV and image-material state."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import bpy


EPSILON = 1.0e-12


def _node_type(node: bpy.types.Node) -> str:
    return str(getattr(node, "bl_idname", type(node).__name__))


def _active_output(material: bpy.types.Material) -> bpy.types.Node | None:
    outputs = [node for node in material.node_tree.nodes if node.type == "OUTPUT_MATERIAL"]
    return next((node for node in outputs if getattr(node, "is_active_output", False)), outputs[0] if outputs else None)


def _reachable_upstream(output: bpy.types.Node | None) -> set[bpy.types.Node]:
    if output is None:
        return set()
    reachable: set[bpy.types.Node] = {output}
    pending = [output]
    while pending:
        node = pending.pop()
        for socket in node.inputs:
            for link in socket.links:
                source = link.from_node
                if source not in reachable:
                    reachable.add(source)
                    pending.append(source)
    return reachable


def _downstream_roles(node: bpy.types.Node, reachable: set[bpy.types.Node]) -> set[str]:
    """Classify reachable image use without assuming node names or layout."""
    roles: set[str] = set()
    pending = [node]
    visited: set[bpy.types.Node] = {node}
    while pending:
        current = pending.pop()
        for socket in current.outputs:
            for link in socket.links:
                destination = link.to_node
                if destination not in reachable:
                    continue
                destination_type = _node_type(destination)
                input_name = link.to_socket.name.casefold()
                if destination_type == "ShaderNodeNormalMap" and input_name == "color":
                    roles.add("data")
                elif destination_type == "ShaderNodeBump" and input_name in {"height", "normal"}:
                    roles.add("data")
                elif destination_type == "ShaderNodeBsdfPrincipled":
                    if input_name in {"base color", "emission color"}:
                        roles.add("color")
                    elif input_name == "alpha":
                        roles.add("alpha")
                    elif input_name in {
                        "anisotropic ior level",
                        "coat weight",
                        "metallic",
                        "normal",
                        "roughness",
                        "specular ior level",
                        "transmission weight",
                    }:
                        roles.add("data")
                elif destination_type == "ShaderNodeOutputMaterial" and input_name == "displacement":
                    roles.add("data")
                if destination not in visited:
                    visited.add(destination)
                    pending.append(destination)
    return roles


def _upstream_uv_sources(image_node: bpy.types.Node) -> tuple[set[str], bool]:
    vector = image_node.inputs.get("Vector")
    if vector is None or not vector.is_linked:
        return {"__ACTIVE__"}, True
    names: set[str] = set()
    uses_uv = False
    pending = [link.from_node for link in vector.links]
    visited: set[bpy.types.Node] = set()
    while pending:
        node = pending.pop()
        if node in visited:
            continue
        visited.add(node)
        if _node_type(node) == "ShaderNodeUVMap":
            names.add(str(node.uv_map) if node.uv_map else "__ACTIVE__")
            uses_uv = True
        elif _node_type(node) == "ShaderNodeTexCoord":
            for output in node.outputs:
                if output.name == "UV" and any(link.to_node in visited or link.to_node is image_node for link in output.links):
                    names.add("__ACTIVE__")
                    uses_uv = True
        for socket in node.inputs:
            pending.extend(link.from_node for link in socket.links)
    return names, uses_uv


def _image_record(node: bpy.types.Node, reachable: set[bpy.types.Node]) -> dict[str, Any]:
    image = node.image
    roles = sorted(_downstream_roles(node, reachable))
    if image is None:
        return {
            "node": node.name,
            "image": None,
            "source": None,
            "roles": roles,
            "colorspace": None,
            "expected_colorspace": None,
            "exists": False,
            "packed": False,
            "ok": False,
            "issues": ["image_node_has_no_image"],
        }

    source = str(image.source)
    packed = image.packed_file is not None
    exists = True
    file_name = None
    if source == "FILE":
        resolved = Path(bpy.path.abspath(image.filepath, library=image.library))
        file_name = resolved.name
        exists = packed or resolved.is_file()
    elif source == "TILED":
        resolved = Path(bpy.path.abspath(image.filepath, library=image.library))
        file_name = resolved.name
        exists = packed or bool(list(resolved.parent.glob(resolved.name.replace("<UDIM>", "[0-9][0-9][0-9][0-9]"))))

    expected = None
    if "color" in roles and "data" not in roles:
        expected = "sRGB"
    elif "data" in roles and "color" not in roles:
        expected = "Non-Color"
    colorspace = str(image.colorspace_settings.name)
    issues: list[str] = []
    if not exists:
        issues.append("image_file_missing")
    if expected is not None and colorspace != expected:
        issues.append("colorspace_mismatch")
    if "color" in roles and "data" in roles:
        issues.append("image_used_for_color_and_data")

    uv_names, uses_uv = _upstream_uv_sources(node)
    return {
        "node": node.name,
        "image": image.name,
        "file_name": file_name,
        "source": source,
        "roles": roles,
        "colorspace": colorspace,
        "expected_colorspace": expected,
        "exists": exists,
        "packed": packed,
        "uses_uv": uses_uv,
        "uv_layers": sorted(uv_names),
        "ok": not issues,
        "issues": issues,
    }


def _material_record(material: bpy.types.Material | None) -> dict[str, Any]:
    if material is None:
        return {
            "name": None,
            "ok": False,
            "issues": ["empty_material_slot"],
            "required_uv_layers": [],
            "allows_udim": False,
            "images": [],
        }
    issues: list[str] = []
    # ``use_nodes`` is deprecated in Blender 5.1. The node tree plus a linked
    # active output is the forward-compatible validity check.
    if material.node_tree is None:
        issues.append("nodes_disabled")
        return {
            "name": material.name,
            "ok": False,
            "issues": issues,
            "required_uv_layers": [],
            "allows_udim": False,
            "images": [],
        }

    output = _active_output(material)
    surface_linked = bool(output and output.inputs.get("Surface") and output.inputs["Surface"].is_linked)
    if output is None:
        issues.append("material_output_missing")
    elif not surface_linked:
        issues.append("surface_output_unlinked")
    reachable = _reachable_upstream(output)
    image_nodes = [node for node in material.node_tree.nodes if node.type == "TEX_IMAGE"]
    reachable_images = [node for node in image_nodes if node in reachable]
    orphan_images = sorted(node.name for node in image_nodes if node not in reachable)
    image_records = [_image_record(node, reachable) for node in reachable_images]
    if any(not record["ok"] for record in image_records):
        issues.append("reachable_image_invalid")

    required_uv_layers: set[str] = set()
    for record in image_records:
        if record.get("uses_uv"):
            required_uv_layers.update(record.get("uv_layers", []))
    return {
        "name": material.name,
        "ok": not issues,
        "issues": issues,
        "active_output": output.name if output else None,
        "surface_linked": surface_linked,
        "reachable_node_count": len(reachable),
        "orphan_image_nodes": orphan_images,
        "required_uv_layers": sorted(required_uv_layers),
        "allows_udim": any(record.get("source") == "TILED" for record in image_records),
        "images": image_records,
    }


def _uv_polygon_area(mesh: bpy.types.Mesh, polygon: bpy.types.MeshPolygon, layer: bpy.types.MeshUVLoopLayer) -> float:
    coordinates = [layer.data[index].uv for index in polygon.loop_indices]
    return abs(sum(
        coordinates[index].x * coordinates[(index + 1) % len(coordinates)].y
        - coordinates[(index + 1) % len(coordinates)].x * coordinates[index].y
        for index in range(len(coordinates))
    )) * 0.5


def _mesh_record(
    obj: bpy.types.Object,
    depsgraph: bpy.types.Depsgraph,
    materials: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        issues: list[str] = []
        slot_materials = [slot.material for slot in obj.material_slots]
        used_slot_indices = sorted({int(poly.material_index) for poly in mesh.polygons})
        invalid_faces = [
            int(poly.index)
            for poly in mesh.polygons
            if poly.material_index >= len(slot_materials) or slot_materials[poly.material_index] is None
        ]
        unused_slots = [index for index in range(len(slot_materials)) if index not in used_slot_indices]
        if invalid_faces:
            issues.append("faces_without_material")

        required_layers: set[str] = set()
        allows_udim = bool(obj.get("uv_allow_udim", False))
        for material in slot_materials:
            if material is None:
                continue
            record = materials[material.name]
            required_layers.update(record["required_uv_layers"])
            allows_udim = allows_udim or bool(record["allows_udim"])

        active_name = mesh.uv_layers.active.name if mesh.uv_layers.active else None
        resolved_required = {active_name if name == "__ACTIVE__" else name for name in required_layers}
        missing_layers = sorted(name for name in resolved_required if name is None or mesh.uv_layers.get(name) is None)
        if missing_layers:
            issues.append("required_uv_layer_missing")

        layer_records: list[dict[str, Any]] = []
        layers_to_check = list(mesh.uv_layers)
        for layer in layers_to_check:
            nonfinite = 0
            out_of_range = 0
            for datum in layer.data:
                values = (float(datum.uv.x), float(datum.uv.y))
                if not all(math.isfinite(value) for value in values):
                    nonfinite += 1
                elif not allows_udim and not all(-EPSILON <= value <= 1.0 + EPSILON for value in values):
                    out_of_range += 1
            degenerate_faces = [
                int(poly.index)
                for poly in mesh.polygons
                if float(poly.area) > EPSILON and _uv_polygon_area(mesh, poly, layer) <= EPSILON
            ]
            layer_issues: list[str] = []
            if nonfinite:
                layer_issues.append("nonfinite_uv")
            if out_of_range:
                layer_issues.append("uv_out_of_unit_tile")
            if degenerate_faces:
                layer_issues.append("degenerate_uv_faces")
            if layer.name in resolved_required and layer_issues:
                issues.append(f"required_uv_layer_invalid:{layer.name}")
            layer_records.append({
                "name": layer.name,
                "active": layer.name == active_name,
                "render_active": bool(layer.active_render),
                "loop_count": len(layer.data),
                "nonfinite_loop_count": nonfinite,
                "out_of_range_loop_count": out_of_range,
                "degenerate_face_count": len(degenerate_faces),
                "degenerate_faces": degenerate_faces[:25],
                "ok": not layer_issues,
                "issues": layer_issues,
            })

        return {
            "name": obj.name,
            "ok": not issues,
            "issues": issues,
            "vertices": len(mesh.vertices),
            "polygons": len(mesh.polygons),
            "material_slots": [material.name if material else None for material in slot_materials],
            "used_material_slots": used_slot_indices,
            "unused_material_slots": unused_slots,
            "faces_without_material_count": len(invalid_faces),
            "faces_without_material": invalid_faces[:25],
            "required_uv_layers": sorted(name for name in resolved_required if name is not None),
            "missing_uv_layers": missing_layers,
            "allows_udim": allows_udim,
            "uv_layers": layer_records,
        }
    finally:
        evaluated.to_mesh_clear()


def build_audit() -> dict[str, Any]:
    scene = bpy.context.scene
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    objects = [obj for obj in scene.objects if obj.type == "MESH" and not obj.hide_render]

    used_materials = {
        slot.material.name: slot.material
        for obj in objects
        for slot in obj.material_slots
        if slot.material is not None
    }
    material_records = {name: _material_record(material) for name, material in sorted(used_materials.items())}
    mesh_records = [_mesh_record(obj, depsgraph, material_records) for obj in sorted(objects, key=lambda item: item.name)]

    failed_materials = [name for name, record in material_records.items() if not record["ok"]]
    failed_meshes = [record["name"] for record in mesh_records if not record["ok"]]
    checks = {
        "renderable_mesh_present": bool(objects),
        "materials_valid": not failed_materials,
        "mesh_uv_assignments_valid": not failed_meshes,
    }
    return {
        "schema": "blender_material_uv_audit.v1",
        "ok": all(checks.values()),
        "status": "pass" if all(checks.values()) else "fail",
        "blender_version": bpy.app.version_string,
        "scene": scene.name,
        "checks": checks,
        "failed_materials": failed_materials,
        "failed_meshes": failed_meshes,
        "materials": list(material_records.values()),
        "meshes": mesh_records,
    }


def main() -> int:
    result = build_audit()
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 2


if "MATERIAL_UV_AUDIT_REQUEST" in globals():
    print(json.dumps(build_audit(), sort_keys=True))
elif __name__ == "__main__":
    raise SystemExit(main())

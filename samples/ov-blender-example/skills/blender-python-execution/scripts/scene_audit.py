"""Print a read-only JSON audit of render and evaluated-geometry readiness."""

from __future__ import annotations

import json
import math
import sys
import argparse
from collections.abc import Iterable

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


def _finite(values: Iterable[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _check(checks: list[dict[str, object]], name: str, passed: bool, detail: object) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def _material_has_surface(material: bpy.types.Material | None) -> bool:
    # ``use_nodes`` is deprecated in Blender 5.1 and removed in 6.0. A usable
    # node material is established by the node tree and reachable output.
    if material is None or material.node_tree is None:
        return False
    outputs = [node for node in material.node_tree.nodes if node.bl_idname == "ShaderNodeOutputMaterial"]
    active = [node for node in outputs if getattr(node, "is_active_output", False)] or outputs
    return any(node.inputs.get("Surface") and node.inputs["Surface"].is_linked for node in active)


def _evaluated_world_corners(obj: bpy.types.Object, depsgraph: bpy.types.Depsgraph) -> list[Vector]:
    evaluated = obj.evaluated_get(depsgraph)
    return [evaluated.matrix_world @ Vector(corner) for corner in evaluated.bound_box]


def _mesh_record(obj: bpy.types.Object, depsgraph: bpy.types.Depsgraph) -> dict[str, object]:
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        matrix = evaluated.matrix_world
        world_vertices = [matrix @ vertex.co for vertex in mesh.vertices]
        finite_world_vertices = [vertex for vertex in world_vertices if _finite(vertex)]
        finite_vertices = len(finite_world_vertices)
        if finite_world_vertices:
            minimum = [min(vertex[index] for vertex in finite_world_vertices) for index in range(3)]
            maximum = [max(vertex[index] for vertex in finite_world_vertices) for index in range(3)]
        else:
            minimum = maximum = None
        slots = [slot.material for slot in obj.material_slots]
        invalid_material_faces = sum(
            1
            for polygon in mesh.polygons
            if polygon.material_index >= len(slots) or slots[polygon.material_index] is None
        )
        degenerate_faces = sum(1 for polygon in mesh.polygons if not math.isfinite(polygon.area) or polygon.area <= 1e-12)
        materials = sorted({material.name for material in slots if material is not None})
        unreachable = sorted(material.name for material in slots if material and not _material_has_surface(material))
        return {
            "name": obj.name,
            "vertices": len(mesh.vertices),
            "edges": len(mesh.edges),
            "polygons": len(mesh.polygons),
            "finite_world_vertices": finite_vertices,
            "nonfinite_world_vertices": len(world_vertices) - finite_vertices,
            "degenerate_polygons": degenerate_faces,
            "world_bounds": {"min": minimum, "max": maximum},
            "material_slots": len(obj.material_slots),
            "materials": materials,
            "unassigned_material_polygons": invalid_material_faces,
            "materials_without_reachable_surface": unreachable,
            "uv_layers": [layer.name for layer in mesh.uv_layers],
            "visible_render": not obj.hide_render,
        }
    finally:
        evaluated.to_mesh_clear()


def _camera_coverage(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    targets: list[bpy.types.Object],
    depsgraph: bpy.types.Depsgraph,
    margin: float,
    include_descendants: bool,
    include_instances: bool,
) -> dict[str, object]:
    requested = list(targets)
    expanded: list[bpy.types.Object] = []
    seen: set[str] = set()
    def add(obj: bpy.types.Object) -> None:
        if obj.name_full in seen:
            return
        seen.add(obj.name_full)
        expanded.append(obj)
        if include_descendants:
            for child in obj.children:
                add(child)
    for obj in requested:
        add(obj)

    points: list[Vector] = []
    per_object: dict[str, int] = {}
    geometry = [obj for obj in expanded if obj.type not in {"EMPTY", "CAMERA", "LIGHT", "SPEAKER"}]
    for obj in geometry:
        corners = _evaluated_world_corners(obj, depsgraph)
        points.extend(corners)
        per_object[obj.name] = len(corners)
    instance_count = 0
    if include_instances:
        included = {getattr(obj, "original", obj).name_full for obj in expanded}
        for instance in depsgraph.object_instances:
            if not instance.is_instance or instance.object is None:
                continue
            source = getattr(instance.object, "original", instance.object)
            parent = getattr(instance.parent, "original", instance.parent) if instance.parent else None
            if source.name_full not in included and (parent is None or parent.name_full not in included):
                continue
            if instance.object.type in {"EMPTY", "CAMERA", "LIGHT", "SPEAKER"}:
                continue
            corners = [instance.matrix_world @ Vector(corner) for corner in instance.object.bound_box]
            points.extend(corners)
            instance_count += 1
    projected = [world_to_camera_view(scene, camera, point) for point in points]
    camera_inverse = camera.matrix_world.inverted()
    depths = [float(-(camera_inverse @ point).z) for point in points]
    outside = [
        index
        for index, point in enumerate(projected)
        if point.x < margin or point.x > 1.0 - margin or point.y < margin or point.y > 1.0 - margin
    ]
    behind = [index for index, depth in enumerate(depths) if depth <= 0.0]
    clipped = [
        index
        for index, depth in enumerate(depths)
        if depth < camera.data.clip_start or depth > camera.data.clip_end
    ]
    finite = all(_finite((point.x, point.y, point.z)) for point in projected) and _finite(depths)
    rect = None
    if projected:
        rect = {
            "min": [min(float(point.x) for point in projected), min(float(point.y) for point in projected)],
            "max": [max(float(point.x) for point in projected), max(float(point.y) for point in projected)],
        }
    return {
        "ok": bool(points) and finite and not outside and not behind and not clipped,
        "margin": margin,
        "requested_objects": [obj.name for obj in requested],
        "expanded_objects": [obj.name for obj in expanded],
        "geometry_objects": [obj.name for obj in geometry],
        "included_instance_count": instance_count,
        "points_per_object": per_object,
        "point_count": len(points),
        "projected_rect": rect,
        "depth_range": [min(depths), max(depths)] if depths else None,
        "outside_margin_indices": outside,
        "behind_camera_indices": behind,
        "outside_clip_indices": clipped,
        "finite": finite,
    }


def _request_from_argv() -> dict[str, object]:
    request = globals().get("BLENDER_AUDIT_REQUEST")
    if isinstance(request, dict):
        return request
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--targets", nargs="*", default=[])
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--no-descendants", action="store_true")
    parser.add_argument("--no-instances", action="store_true")
    values, _ = parser.parse_known_args(argv)
    return {"target_names": values.targets, "margin": values.margin,
            "include_descendants": not values.no_descendants,
            "include_instances": not values.no_instances}


def _reachable_file_images(scene: bpy.types.Scene, meshes: list[bpy.types.Object]) -> list[bpy.types.Image]:
    """Return file images reachable from assigned surfaces or the scene World."""
    trees_and_outputs = []
    materials = {slot.material for obj in meshes for slot in obj.material_slots if slot.material}
    for material in materials:
        tree = material.node_tree
        if tree:
            outputs = [node for node in tree.nodes if node.bl_idname == "ShaderNodeOutputMaterial"]
            active = [node for node in outputs if getattr(node, "is_active_output", False)] or outputs
            trees_and_outputs.extend((tree, node) for node in active)
    if scene.world and scene.world.node_tree:
        tree = scene.world.node_tree
        outputs = [node for node in tree.nodes if node.bl_idname == "ShaderNodeOutputWorld"]
        active = [node for node in outputs if getattr(node, "is_active_output", False)] or outputs
        trees_and_outputs.extend((tree, node) for node in active)

    images = set()
    for _tree, output in trees_and_outputs:
        pending, seen = [output], set()
        while pending:
            node = pending.pop()
            if node in seen:
                continue
            seen.add(node)
            image = getattr(node, "image", None)
            if image and image.source == "FILE":
                images.add(image)
            for socket in node.inputs:
                pending.extend(link.from_node for link in socket.links)
    return list(images)


def build_audit(request: dict[str, object] | None = None) -> dict[str, object]:
    request = request or {}
    scene = bpy.context.scene
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    checks: list[dict[str, object]] = []

    _check(checks, "active_camera", scene.camera is not None, scene.camera.name if scene.camera else None)
    _check(
        checks,
        "positive_resolution",
        scene.render.resolution_x > 0 and scene.render.resolution_y > 0,
        [int(scene.render.resolution_x), int(scene.render.resolution_y)],
    )

    nonfinite: list[str] = []
    for obj in scene.objects:
        matrix_values = [value for row in obj.matrix_world for value in row]
        if not _finite(matrix_values):
            nonfinite.append(obj.name)
    _check(checks, "finite_world_transforms", not nonfinite, sorted(nonfinite))

    visible_meshes = [obj for obj in scene.objects if obj.type == "MESH" and not obj.hide_render]
    _check(checks, "renderable_mesh_present", bool(visible_meshes), len(visible_meshes))

    mesh_records: list[dict[str, object]] = []
    evaluation_errors: list[dict[str, str]] = []
    for obj in visible_meshes:
        try:
            mesh_records.append(_mesh_record(obj, depsgraph))
        except Exception as exc:  # Preserve all object-level failures in one audit.
            evaluation_errors.append({"object": obj.name, "error": f"{type(exc).__name__}: {exc}"})
    _check(checks, "evaluated_meshes", not evaluation_errors, evaluation_errors)
    invalid_geometry = sorted(
        record["name"]
        for record in mesh_records
        if record["vertices"] == 0
        or record["polygons"] == 0
        or record["nonfinite_world_vertices"]
        or record["degenerate_polygons"]
    )
    _check(checks, "valid_render_geometry", not invalid_geometry, invalid_geometry)

    raw_targets = request.get("targets", request.get("target_names", []))
    target_names = [str(name) for name in raw_targets]
    margin = float(request.get("margin", 0.05))
    include_descendants = bool(request.get("include_descendants", True))
    include_instances = bool(request.get("include_instances", True))
    if not 0.0 <= margin < 0.5:
        raise ValueError("margin must be in [0, 0.5)")
    missing_targets = [name for name in target_names if bpy.data.objects.get(name) is None]
    coverage = None
    if target_names and scene.camera and not missing_targets:
        coverage = _camera_coverage(
            scene,
            scene.camera,
            [bpy.data.objects[name] for name in target_names],
            depsgraph,
            margin,
            include_descendants,
            include_instances,
        )
    if target_names:
        _check(checks, "camera_targets_exist", not missing_targets, missing_targets)
        _check(checks, "camera_contains_targets", bool(coverage and coverage["ok"]), coverage)

    missing_images = sorted(
        image.name
        for image in _reachable_file_images(scene, visible_meshes)
        if image.source == "FILE" and not image.packed_file and not image.has_data
    )
    _check(checks, "file_images_loaded", not missing_images, missing_images)

    failed = [item["name"] for item in checks if not item["passed"]]
    return {
        "schema": "blender_scene_audit.v2",
        "ok": not failed,
        "status": "pass" if not failed else "fail",
        "scene": scene.name,
        "render_engine": scene.render.engine,
        "camera": scene.camera.name if scene.camera else None,
        "checks": checks,
        "failed_checks": failed,
        "meshes": mesh_records,
        "camera_coverage": coverage,
        "lighting": {
            "lights": sorted(obj.name for obj in scene.objects if obj.type == "LIGHT" and not obj.hide_render),
            "world": scene.world.name if scene.world else None,
            "world_has_node_tree": bool(scene.world and scene.world.node_tree),
        },
        "output": {
            "filepath": scene.render.filepath,
            "file_format": scene.render.image_settings.file_format,
            "resolution": [scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage],
            "pixel_aspect": [scene.render.pixel_aspect_x, scene.render.pixel_aspect_y],
        },
    }


def main() -> None:
    print(json.dumps(build_audit(_request_from_argv()), sort_keys=True))


if "BLENDER_AUDIT_REQUEST" in globals():
    print(json.dumps(build_audit(BLENDER_AUDIT_REQUEST), sort_keys=True))
elif __name__ == "__main__":
    main()

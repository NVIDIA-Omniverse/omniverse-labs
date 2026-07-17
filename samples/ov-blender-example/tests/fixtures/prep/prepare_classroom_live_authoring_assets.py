#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare a furniture-free Classroom scene and append-only workstation library.

Run inside Blender with the official Classroom file open::

    blender --background classroom.blend --python prepare_classroom_live_authoring_assets.py -- \
      --output-dir /path/to/local/output

The outputs are local authoring inputs, not redistributable fixtures. This
preflight creates no SimReady or USD Physics properties.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Sequence


STUDENT_PATTERN = re.compile(r"chair(?:[._]\d+)?", re.IGNORECASE)
SOURCE_COLLECTION = "schoolDesk"
WORKSTATION_LIBRARY_COLLECTION = "ClassroomLiveWorkstationAsset"
EMISSION_STRENGTHS = {
    "blackBoardLight": 1.0,
    "dayLight_portal": 20.0,
    "ceillingLamp_light": 2.0,
}
EMISSION_COLORS = {
    "blackBoardLight": (0.8, 0.8, 0.8, 1.0),
    "dayLight_portal": (0.8, 0.9, 1.0, 1.0),
    "ceillingLamp_light": (1.0, 0.89416665, 0.8075703, 1.0),
}
MATRIX_OVERRIDE_PROP = "ovrtx.matrix_override"
CLASSROOM_LIGHT_INTENSITIES = {
    "blackBoard_light": 1126.36,
    "coridor_ceilingLight": 281250.0,
    "exterior_fillLight": 118911.0,
    "sun": 1184.35,
}
CEILING_LIGHT_INTENSITY = 925.926
SPOT_LIGHT_INTENSITY = 300000.0

CHAIR_MEMBERS = frozenset(
    {
        "Box280.002",
        "Box295.002",
        "Box296.002",
        "Box297.002",
        "Box298.002",
        "Box299.002",
        "Cube.024",
        "Cube.033",
        "Sphere123.003",
        "Sphere123.004",
    }
)
DESK_MEMBERS = frozenset(
    {
        "Cylinder813.000",
        "Cylinder813.001",
        "Cylinder813.003",
        "Line121.002",
        "Line122.002",
        "Line136.002",
        "Line137.002",
        "Plane.003",
        "Plane.004",
        "Sphere140.003",
        "Sphere140.004",
        "Sphere141.003",
        "Sphere141.004",
    }
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(_script_args() if argv is None else argv)
    try:
        import bpy  # type: ignore
        from mathutils import Matrix  # type: ignore
    except ImportError as exc:
        raise SystemExit(f"run this script inside Blender: {exc}")

    source = Path(bpy.data.filepath).resolve()
    if not source.is_file():
        raise SystemExit("open the official Classroom .blend before running preflight")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    working_blend = output_dir / "classroom-furniture-free.blend"
    workstation_library = output_dir / "classroom-workstation-library.blend"
    report_path = args.report or output_dir / "classroom-live-authoring-assets.json"

    source_collection = bpy.data.collections.get(SOURCE_COLLECTION)
    if source_collection is None:
        raise SystemExit(f"source collection not found: {SOURCE_COLLECTION}")
    source_meshes = {
        obj.name: obj
        for obj in source_collection.all_objects
        if obj.type == "MESH"
    }
    expected = CHAIR_MEMBERS | DESK_MEMBERS
    if set(source_meshes) != expected:
        missing = sorted(expected.difference(source_meshes))
        extra = sorted(set(source_meshes).difference(expected))
        raise SystemExit(f"schoolDesk role contract mismatch; missing={missing}, extra={extra}")

    students = tuple(
        sorted(
            (
                obj
                for obj in bpy.context.scene.objects
                if STUDENT_PATTERN.fullmatch(obj.name) and obj.instance_collection == source_collection
            ),
            key=lambda obj: obj.name,
        )
    )
    if len(students) != 20:
        raise SystemExit(f"expected 20 student placements, found {len(students)}")

    desk_bounds = combined_bounds(source_meshes[name] for name in sorted(DESK_MEMBERS))
    desk_top_bounds = combined_bounds(source_meshes[name] for name in ("Plane.003", "Plane.004"))
    chair_bounds = combined_bounds(source_meshes[name] for name in sorted(CHAIR_MEMBERS))
    chair_seat_bounds = combined_bounds((source_meshes["Box297.002"],))
    chair_back_bounds = combined_bounds((source_meshes["Box280.002"],))
    desk_proxies = desk_proxy_specs(desk_bounds, top_bounds=desk_top_bounds)
    chair_proxies = chair_proxy_specs(
        chair_bounds,
        chair_seat_bounds,
        chair_back_bounds,
        back_axis="-Y",
    )

    workstation_collection = bpy.data.collections.new(WORKSTATION_LIBRARY_COLLECTION)
    desk_root = bpy.data.objects.new("DeskTemplate", None)
    desk_root["classroom:role"] = "desk"
    desk_root["classroom:massKg"] = 15.0
    workstation_collection.objects.link(desk_root)
    desk_member_records = []
    for name in sorted(DESK_MEMBERS):
        copied = local_copy(bpy, source_meshes[name], f"DeskVisual_{name}")
        workstation_collection.objects.link(copied)
        copied.parent = desk_root
        copied.matrix_parent_inverse = Matrix.Identity(4)
        copied.matrix_basis = source_meshes[name].matrix_world.copy()
        copied["classroom:sourceObject"] = name
        copied["classroom:role"] = "desk_visual"
        desk_member_records.append(member_record(copied, name))
    for spec in desk_proxies:
        proxy = box_object(bpy, f"DeskCollider_{spec['name']}", spec)
        workstation_collection.objects.link(proxy)
        proxy.parent = desk_root
        proxy.matrix_parent_inverse = Matrix.Identity(4)
        proxy["classroom:role"] = "desk_collider"

    chair_root = bpy.data.objects.new("ChairTemplate", None)
    chair_root["classroom:role"] = "chair"
    chair_root["classroom:massKg"] = 6.0
    workstation_collection.objects.link(chair_root)
    chair_member_records = []
    for name in sorted(CHAIR_MEMBERS):
        copied = local_copy(bpy, source_meshes[name], f"ChairVisual_{name}")
        workstation_collection.objects.link(copied)
        copied.parent = chair_root
        copied.matrix_parent_inverse = Matrix.Identity(4)
        copied.matrix_basis = source_meshes[name].matrix_world.copy()
        copied["classroom:sourceObject"] = name
        copied["classroom:role"] = "chair_visual"
        chair_member_records.append(member_record(copied, name))
    for spec in chair_proxies:
        proxy = box_object(bpy, f"ChairCollider_{spec['name']}", spec)
        workstation_collection.objects.link(proxy)
        proxy.parent = chair_root
        proxy.matrix_parent_inverse = Matrix.Identity(4)
        proxy["classroom:role"] = "chair_collider"

    if workstation_library.exists():
        workstation_library.unlink()
    bpy.data.libraries.write(
        str(workstation_library),
        {workstation_collection},
        path_remap="ABSOLUTE",
        fake_user=True,
    )

    placements = []
    for index, student in enumerate(students):
        placements.append(
            {
                "source_instance": student.name,
                "desk_owner": f"Desk_{index:02d}",
                "chair_owner": f"Chair_{index:02d}",
                "world_transform": matrix_rows(student.matrix_world),
                "translation": [float(value) for value in student.matrix_world.translation],
            }
        )

    for student in students:
        bpy.data.objects.remove(student, do_unlink=True)
    bpy.data.collections.remove(source_collection)
    for obj in tuple(bpy.data.objects):
        if str(obj.get("classroom:role", "")).startswith(("chair", "desk")):
            bpy.data.objects.remove(obj, do_unlink=True)
    for collection in tuple(bpy.data.collections):
        if collection.name == WORKSTATION_LIBRARY_COLLECTION:
            bpy.data.collections.remove(collection)

    beauty_repairs = apply_beauty_repairs(bpy)

    student_instances_remaining = [
        obj.name
        for obj in bpy.context.scene.objects
        if STUDENT_PATTERN.fullmatch(obj.name) and obj.instance_collection is not None
    ]
    furniture_roles_remaining = [
        obj.name
        for obj in bpy.context.scene.objects
        if str(obj.get("classroom:role", "")).startswith(("chair", "desk"))
    ]
    physics_keys = sorted(
        {
            key
            for obj in bpy.context.scene.objects
            for key in obj.keys()
            if "physics" in key.lower() or "physx" in key.lower() or "simready" in key.lower()
        }
    )
    if student_instances_remaining or furniture_roles_remaining or physics_keys:
        raise RuntimeError(
            "opening scene is not furniture/physics free: "
            f"instances={student_instances_remaining}, furniture={furniture_roles_remaining}, keys={physics_keys}"
        )

    bpy.ops.wm.save_as_mainfile(filepath=str(working_blend))
    reveal_order = reveal_order_from_placements(placements)
    report = {
        "schema_version": 1,
        "artifact_id": "classroom-live-authoring-local-assets",
        "status": "pass",
        "source": {
            "blend": str(source),
            "sha256": sha256(source),
            "student_collection": SOURCE_COLLECTION,
            "student_placement_count": len(placements),
        },
        "role_contract": {
            "desk_members": sorted(DESK_MEMBERS),
            "chair_members": sorted(CHAIR_MEMBERS),
            "desk_proxy_roles": [spec["name"] for spec in desk_proxies],
            "chair_proxy_roles": [spec["name"] for spec in chair_proxies],
            "chair_back_axis": "-Y",
        },
        "workstation_template": {
            "collection": WORKSTATION_LIBRARY_COLLECTION,
            "desk_members": desk_member_records,
            "chair_members": chair_member_records,
            "mesh_data_shared_across_placements": True,
            "identity_basis": "all 20 source placements instance the same schoolDesk collection",
        },
        "placements": placements,
        "reveal_order": reveal_order,
        "opening_scene": {
            "student_instance_count": 0,
            "chair_role_object_count": 0,
            "desk_role_object_count": 0,
            "physics_property_keys": [],
        },
        "beauty_repairs": beauty_repairs,
        "outputs": {
            "working_blend": str(working_blend),
            "working_blend_sha256": sha256(working_blend),
            "workstation_library": str(workstation_library),
            "workstation_library_sha256": sha256(workstation_library),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "report": str(report_path)}, sort_keys=True))
    return 0


def apply_beauty_repairs(bpy: Any) -> dict[str, Any]:
    """Represent the accepted Classroom look in stock-exportable Blender data."""

    scene = bpy.context.scene
    instancers = sorted(
        (obj for obj in scene.objects if obj.instance_collection is not None),
        key=lambda obj: obj.name,
    )
    if instancers:
        bpy.ops.object.select_all(action="DESELECT")
        for obj in instancers:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = instancers[0]
        result = bpy.ops.object.duplicates_make_real(
            use_base_parent=True,
            use_hierarchy=True,
        )
        if set(result) != {"FINISHED"}:
            raise RuntimeError(f"could not realize Classroom collection instances: {sorted(result)}")
    remaining_instancers = [obj.name for obj in scene.objects if obj.instance_collection is not None]
    if remaining_instancers:
        raise RuntimeError(f"Classroom collection instances remain: {remaining_instancers}")

    baked_armature_modifiers = []
    for obj in sorted((item for item in scene.objects if item.type == "MESH"), key=lambda item: item.name):
        modifiers = [modifier for modifier in obj.modifiers if modifier.type == "ARMATURE"]
        if not modifiers:
            continue
        if obj.data.library is not None or obj.data.users > 1:
            obj.data = obj.data.copy()
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        for modifier in modifiers:
            name = modifier.name
            result = bpy.ops.object.modifier_apply(modifier=name)
            if set(result) != {"FINISHED"}:
                raise RuntimeError(f"could not bake Classroom armature modifier: {obj.name}.{name}")
            baked_armature_modifiers.append(f"{obj.name}.{name}")
    armature_objects = sorted(
        (obj for obj in scene.objects if obj.type == "ARMATURE"),
        key=lambda obj: obj.name,
    )
    armature_object_names = [obj.name for obj in armature_objects]
    for obj in armature_objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    parented_objects = sorted(
        (obj for obj in scene.objects if obj.parent is not None),
        key=lambda obj: obj.name,
    )
    for obj in parented_objects:
        world = obj.matrix_world.copy()
        obj.parent = None
        obj.matrix_world = world

    non_rendered_objects = sorted(
        (
            obj
            for obj in scene.objects
            if obj.type in {"CAMERA", "LIGHT", "MESH"}
            and (
                obj.hide_render
                or (
                    obj.users_collection
                    and all(collection.hide_render for collection in obj.users_collection)
                )
            )
        ),
        key=lambda obj: obj.name,
    )
    non_rendered_object_names = [obj.name for obj in non_rendered_objects]
    for obj in non_rendered_objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    scene_objects = set(scene.objects)
    spots = sorted(
        (
            obj
            for obj in bpy.data.objects
            if obj.type == "LIGHT"
            and obj.data.type == "SPOT"
            and obj not in scene_objects
        ),
        key=lambda obj: obj.name,
    )
    if len(spots) != 5:
        raise RuntimeError(f"expected 5 orphaned Classroom spotlights, found {len(spots)}")
    repaired_spots = []
    for source_obj in spots:
        source_name = source_obj.name
        source_matrix = source_obj.matrix_basis.copy()
        obj = source_obj.copy()
        obj.data = source_obj.data.copy()
        bpy.data.objects.remove(source_obj, do_unlink=True)
        obj.name = source_name
        scene.collection.objects.link(obj)
        obj[MATRIX_OVERRIDE_PROP] = [
            float(source_matrix[row][column])
            for row in range(4)
            for column in range(4)
        ]
        obj.data.type = "POINT"
        obj.data.energy = blender_energy_for_usd_intensity("SPOT", SPOT_LIGHT_INTENSITY)
        obj.data.normalize = False
        obj.data.use_temperature = False
        repaired_spots.append(obj)

    repaired_lights = {}
    for obj in sorted((item for item in scene.objects if item.type == "LIGHT"), key=lambda item: item.name):
        target = CLASSROOM_LIGHT_INTENSITIES.get(obj.name)
        if obj.name.startswith("Point."):
            target = CEILING_LIGHT_INTENSITY
        if target is None:
            continue
        if obj.data.library is not None or obj.data.users > 1:
            obj.data = obj.data.copy()
        obj.data.energy = blender_energy_for_usd_intensity(obj.data.type, target)
        obj.data.normalize = False
        obj.data.use_temperature = False
        repaired_lights[obj.name] = target

    curves = sorted((obj for obj in scene.objects if obj.type == "CURVE"), key=lambda obj: obj.name)
    thin = [obj for obj in curves if sum(len(spline.points) for spline in obj.data.splines) == 2]
    thick = [obj for obj in curves if sum(len(spline.points) for spline in obj.data.splines) == 9]
    if len(curves) != 56 or len(thin) != 48 or len(thick) != 8:
        raise RuntimeError(
            f"expected 56 Classroom blind curves (48 thin, 8 thick), found "
            f"{len(curves)} ({len(thin)} thin, {len(thick)} thick)"
        )
    converted_curve_data = set()
    for obj in curves:
        pointer = int(obj.data.as_pointer())
        if pointer in converted_curve_data:
            continue
        converted_curve_data.add(pointer)
        bevel_depth = float(obj.data.bevel_depth)
        for spline in obj.data.splines:
            for point in spline.points:
                point.radius = bevel_depth
        obj.data.bevel_depth = 1.0

    simplified_emission_materials = []
    for name, expected_strength in EMISSION_STRENGTHS.items():
        material = bpy.data.materials.get(name)
        if material is None or material.node_tree is None:
            raise RuntimeError(f"missing Classroom emission material: {name}")
        nodes = [node for node in material.node_tree.nodes if node.type == "EMISSION"]
        if len(nodes) != 1:
            raise RuntimeError(f"expected one emission node in {name}, found {len(nodes)}")
        strength = float(nodes[0].inputs["Strength"].default_value)
        if not math.isclose(strength, expected_strength, abs_tol=1.0e-6):
            raise RuntimeError(f"unexpected emission strength for {name}: {strength}")
        material.use_nodes = True
        material.node_tree.nodes.clear()
        output = material.node_tree.nodes.new("ShaderNodeOutputMaterial")
        emission = material.node_tree.nodes.new("ShaderNodeEmission")
        emission.inputs["Color"].default_value = EMISSION_COLORS[name]
        emission.inputs["Strength"].default_value = expected_strength
        material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
        simplified_emission_materials.append(name)

    unlinked_memberships = []
    for obj in sorted(scene.objects, key=lambda item: item.name):
        collections = sorted(obj.users_collection, key=lambda item: item.name)
        if scene.collection not in collections:
            scene.collection.objects.link(obj)
        for collection in tuple(obj.users_collection):
            if collection == scene.collection:
                continue
            collection.objects.unlink(obj)
            unlinked_memberships.append({"object": obj.name, "collection": collection.name})

    return {
        "spotlights": [obj.name for obj in repaired_spots],
        "spotlight_intensity": SPOT_LIGHT_INTENSITY,
        "spotlight_normalize": False,
        "spotlight_transform_override": MATRIX_OVERRIDE_PROP,
        "light_intensities": repaired_lights,
        "blind_curve_count": len(curves),
        "thin_blind_curve_count": len(thin),
        "thick_blind_curve_count": len(thick),
        "emission_strengths": dict(EMISSION_STRENGTHS),
        "simplified_emission_materials": simplified_emission_materials,
        "realized_collection_instances": [obj.name for obj in instancers],
        "baked_armature_modifiers": baked_armature_modifiers,
        "armature_objects_removed": armature_object_names,
        "flattened_parent_count": len(parented_objects),
        "non_rendered_objects_removed": non_rendered_object_names,
        "duplicate_collection_memberships_removed": unlinked_memberships,
    }


def blender_energy_for_usd_intensity(light_type: str, intensity: float) -> float:
    """Invert Blender's stock USD light-energy conversion."""

    value = float(intensity)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("USD light intensity must be finite and non-negative")
    return value * (4.0 if str(light_type).upper() == "SUN" else math.pi)


def reveal_order_from_placements(placements: Sequence[dict[str, Any]]) -> list[str]:
    """Order rows far-to-near and alternate left/right within each row."""

    remaining = sorted(placements, key=lambda item: (-float(item["translation"][1]), float(item["translation"][0])))
    rows: list[list[dict[str, Any]]] = []
    for placement in remaining:
        if not rows or abs(float(rows[-1][0]["translation"][1]) - float(placement["translation"][1])) > 0.35:
            rows.append([placement])
        else:
            rows[-1].append(placement)
    ordered: list[str] = []
    for row in rows:
        values = sorted(row, key=lambda item: float(item["translation"][0]))
        take_left = True
        while values:
            selected = values.pop(0 if take_left else -1)
            ordered.append(str(selected["source_instance"]))
            take_left = not take_left
    return ordered


def combined_bounds(objects: Iterable[Any]) -> dict[str, list[float]]:
    from mathutils import Vector  # type: ignore

    corners = [
        obj.matrix_world @ Vector(corner)
        for obj in objects
        for corner in obj.bound_box
    ]
    return {
        "min": [min(float(corner[axis]) for corner in corners) for axis in range(3)],
        "max": [max(float(corner[axis]) for corner in corners) for axis in range(3)],
    }


def desk_proxy_specs(
    bounds: dict[str, list[float]],
    *,
    top_bounds: dict[str, list[float]] | None = None,
) -> list[dict[str, Any]]:
    low, high = bounds["min"], bounds["max"]
    size = [high[index] - low[index] for index in range(3)]
    top_low = (top_bounds or bounds)["min"]
    top_high = (top_bounds or bounds)["max"]
    top_size = [top_high[index] - top_low[index] for index in range(3)]
    top_center = [(top_low[index] + top_high[index]) / 2.0 for index in range(3)]
    leg = max(min(size[0], size[1]) * 0.06, 0.035)
    height = max(top_low[2] - low[2], 0.1)
    result = [box_spec("Top", top_center, top_size)]
    for name, x, y in (
        ("LegFL", low[0], low[1]),
        ("LegFR", high[0], low[1]),
        ("LegBL", low[0], high[1]),
        ("LegBR", high[0], high[1]),
    ):
        result.append(box_spec(name, [x, y, low[2] + height / 2.0], [leg, leg, height]))
    return result


def chair_proxy_specs(
    bounds: dict[str, list[float]],
    seat_bounds: dict[str, list[float]],
    back_bounds: dict[str, list[float]],
    *,
    back_axis: str,
) -> list[dict[str, Any]]:
    if back_axis != "-Y":
        raise ValueError("the validated schoolDesk chair back axis is -Y")
    low, high = bounds["min"], bounds["max"]
    seat_low, seat_high = seat_bounds["min"], seat_bounds["max"]
    size = [seat_high[index] - seat_low[index] for index in range(3)]
    center = [(seat_low[index] + seat_high[index]) / 2.0 for index in range(3)]
    leg = max(min(size[0], size[1]) * 0.07, 0.03)
    height = seat_low[2] - low[2]
    result = [box_spec("Seat", center, size)]
    back_low, back_high = back_bounds["min"], back_bounds["max"]
    result.append(
        box_spec(
            "Back",
            [(back_low[index] + back_high[index]) / 2.0 for index in range(3)],
            [back_high[index] - back_low[index] for index in range(3)],
        )
    )
    for name, x, y in (
        ("LegFL", seat_low[0] + leg / 2.0, seat_low[1] + leg / 2.0),
        ("LegFR", seat_high[0] - leg / 2.0, seat_low[1] + leg / 2.0),
        ("LegBL", seat_low[0] + leg / 2.0, seat_high[1] - leg / 2.0),
        ("LegBR", seat_high[0] - leg / 2.0, seat_high[1] - leg / 2.0),
    ):
        result.append(
            box_spec(
                name,
                [x, y, low[2] + height / 2.0],
                [leg, leg, height],
            )
        )
    return result


def box_spec(name: str, center: Sequence[float], size: Sequence[float]) -> dict[str, Any]:
    if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in size):
        raise ValueError(f"invalid box size for {name}: {size}")
    return {"name": name, "center": [float(value) for value in center], "size": [float(value) for value in size]}


def box_object(bpy: Any, name: str, spec: dict[str, Any]) -> Any:
    sx, sy, sz = (float(value) / 2.0 for value in spec["size"])
    vertices = [
        (x, y, z)
        for x in (-sx, sx)
        for y in (-sy, sy)
        for z in (-sz, sz)
    ]
    faces = ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))
    mesh = bpy.data.meshes.new(f"{name}Mesh")
    mesh.from_pydata(vertices, (), faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = tuple(spec["center"])
    obj.display_type = "WIRE"
    obj.hide_render = True
    obj["classroom:proxyShape"] = "box"
    obj["classroom:proxySize"] = list(spec["size"])
    return obj


def local_copy(bpy: Any, source: Any, name: str) -> Any:
    copied = source.copy()
    copied.data = source.data.copy()
    copied.name = name
    for slot_index, slot in enumerate(copied.material_slots):
        if slot.material is not None:
            copied.material_slots[slot_index].material = slot.material.copy()
    return copied


def member_record(obj: Any, source_name: str) -> dict[str, Any]:
    return {
        "source_object": source_name,
        "object": obj.name,
        "mesh": obj.data.name,
        "materials": [slot.material.name for slot in obj.material_slots if slot.material],
    }


def matrix_rows(matrix: Any) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    return parser.parse_args(list(argv))


def _script_args() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Author the small committed .blend fixture for direct user-scene export."""

from __future__ import annotations

import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
ADDON_PATH = ROOT / "addon"
DEFAULT_OUTPUT = ROOT / "tests" / "fixtures" / "assets" / "hero_cube" / "scene.blend"


def main() -> None:
    import bpy  # type: ignore
    from mathutils import Vector  # type: ignore

    sys.path.insert(0, str(ADDON_PATH))
    import ovrtx_blender_example
    from ovrtx_blender_example import light_value_conversion

    output = _output_path()
    output.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_homefile(use_empty=True)
    ovrtx_blender_example.register()
    scene = bpy.context.scene
    scene.name = "Hero Cube"
    scene.render.resolution_x = 640
    scene.render.resolution_y = 480
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.world = bpy.data.worlds.new("Night Studio")
    scene.world.color = (0.025, 0.035, 0.055)
    scene.unit_settings.system = "METRIC"
    scene.ovrtx_example.min_samples = 16
    scene.ovrtx_example.max_samples = 16

    bpy.ops.mesh.primitive_cube_add(location=(0.0, 0.0, 1.0))
    cube = bpy.context.active_object
    cube.name = "Hero Cube"
    cube.rotation_euler[2] = math.radians(22.5)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.edge_split(type="EDGE")
    bpy.ops.object.mode_set(mode="OBJECT")
    cube_material = bpy.data.materials.new("Hero Blue")
    _set_principled_base_color(cube_material, (0.05, 0.28, 0.8, 1.0))
    cube.data.materials.append(cube_material)

    bpy.ops.mesh.primitive_plane_add(size=12.0, location=(0.0, 0.0, 0.0))
    floor = bpy.context.active_object
    floor.name = "Ground Plane"
    floor_material = bpy.data.materials.new("Warm Ground")
    _set_principled_base_color(floor_material, (0.32, 0.18, 0.08, 1.0))
    floor.data.materials.append(floor_material)

    bpy.ops.object.light_add(type="AREA", location=(3.5, -3.0, 6.0))
    key = bpy.context.active_object
    key.name = "Key Light"
    key.data.energy = 50000.0 / light_value_conversion.MEASURED_LIGHT_SCALE
    key.data.shape = "DISK"
    key.data.size = 4.0
    _point_at(key, Vector((0.0, 0.0, 1.0)))

    bpy.ops.object.light_add(type="AREA", location=(-3.0, 1.5, 3.5))
    fill = bpy.context.active_object
    fill.name = "Fill Light"
    fill.data.energy = 20000.0 / light_value_conversion.MEASURED_LIGHT_SCALE
    fill.data.size = 3.0
    _point_at(fill, Vector((0.0, 0.0, 1.0)))

    bpy.ops.object.camera_add(location=(5.6, -6.2, 4.2))
    camera = bpy.context.active_object
    camera.name = "Camera"
    camera.data.lens = 52.0
    _point_at(camera, Vector((0.0, 0.0, 0.8)))
    scene.camera = camera

    bpy.ops.wm.save_as_mainfile(filepath=str(output), check_existing=False)
    print(f"authored_blend_user_scene={output}")


def _point_at(obj: object, target: object) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _set_principled_base_color(material: object, color: tuple[float, float, float, float]) -> None:
    material.diffuse_color = color
    material.use_nodes = True
    principled = next(node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
    principled.inputs["Base Color"].default_value = color
    principled.inputs["Roughness"].default_value = 0.38


def _output_path() -> Path:
    argv = sys.argv
    if "--" not in argv:
        return DEFAULT_OUTPUT
    values = argv[argv.index("--") + 1 :]
    return Path(values[0]).expanduser().resolve() if values else DEFAULT_OUTPUT


if __name__ == "__main__":
    main()

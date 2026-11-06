# SPDX-License-Identifier: Apache-2.0
"""Shared authored scene for stock-Blender render-output acceptance."""
import bpy
from mathutils import Vector


def create_scene(addon):
    addon.register()
    bpy.context.preferences.view.show_splash = False
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.render.engine = 'OVRTX_EXAMPLE'
    scene.render.resolution_x, scene.render.resolution_y = 321, 241
    scene.render.resolution_percentage = 100
    scene.ovrtx_example.max_samples = 32
    scene.ovrtx_example.dlss_enabled = False
    # Both implementations use the same explicitly authored neutral exposure.
    # Factory-default request admission has a separate stock Blender test.
    scene.ovrtx_example.native_auto_exposure = False
    scene.ovrtx_example.native_exposure_white_point_scale = 10
    scene.ovrtx_example.color_presentation_mode = 'scene_linear_hdr'
    scene.view_settings.view_transform = 'Standard'
    scene.world.use_nodes = True
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = .15
    bpy.ops.mesh.primitive_cube_add(location=(-1, 0, 1))
    cube = bpy.context.object
    cube.name = 'Animated orange subject'
    material = bpy.data.materials.new('Orange emission')
    material.use_nodes = True
    shader = material.node_tree.nodes['Principled BSDF']
    shader.inputs['Base Color'].default_value = (.8, .03, .01, 1)
    shader.inputs['Emission Color'].default_value = (.8, .03, .01, 1)
    shader.inputs['Emission Strength'].default_value = 4
    cube.data.materials.append(material)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, location=(2, 1, 1))
    sphere = bpy.context.object
    sphere.name = 'Blue stationary reference'
    blue = bpy.data.materials.new('Blue')
    blue.use_nodes = True
    blue_shader = blue.node_tree.nodes['Principled BSDF']
    blue_shader.inputs['Base Color'].default_value = (.01, .04, .8, 1)
    blue_shader.inputs['Emission Color'].default_value = (.01, .04, .8, 1)
    blue_shader.inputs['Emission Strength'].default_value = 1
    sphere.data.materials.append(blue)
    bpy.ops.mesh.primitive_plane_add(size=30)
    bpy.context.object.name = 'Ground'
    bpy.ops.object.light_add(type='AREA', location=(2, -3, 7))
    light = bpy.context.object
    light.data.energy, light.data.size = 1200, 4
    light.rotation_euler = (Vector((0, 0, 1))-light.location).to_track_quat('-Z', 'Y').to_euler()
    bpy.ops.object.camera_add(location=(7, -11, 6))
    camera = bpy.context.object
    camera.name = 'Output camera'
    camera.rotation_euler = (Vector((0, 0, 1))-camera.location).to_track_quat('-Z', 'Y').to_euler()
    scene.camera = camera
    scene.frame_start, scene.frame_end = 1, 3
    for frame, x in ((1, -2), (2, 0), (3, 2)):
        cube.location.x = x
        cube.keyframe_insert(data_path='location', frame=frame)
    scene.frame_set(1)
    bpy.context.view_layer.update()
    return scene, cube, camera


def raw_pixels(result):
    """Owned bottom-left RGBA bytes for both retained and current result APIs."""
    if result.linear_rgba16f:
        method = getattr(result, 'linear_rgba16f_bytes', None)
        return method() if method else bytes(result.linear_rgba16f)
    return result.rgba8_bytes()

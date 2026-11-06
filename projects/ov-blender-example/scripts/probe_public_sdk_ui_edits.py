#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Live Blender UI edit fixture controlled by files in its private output dir.

Run in a Blender window using launch_public_sdk_blender.sh. The main-thread
timer performs ordinary Blender edits; the add-on owns conversion and runtime
submission. An external capture process records this actual X11 window.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import bpy
from mathutils import Vector

source = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(source / 'addon'))
import ovrtx_blender_example as addon
from ovrtx_blender_example import engine

output = Path(os.environ['OV_BLENDER_PUBLIC_SDK_OUTPUT']) / 'ui-edits'
output.mkdir(parents=True, exist_ok=True)
addon.register()
bpy.context.preferences.view.show_splash = False
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)


def material(name, color, roughness=0.35):
    result = bpy.data.materials.new(name)
    result.use_nodes = True
    shader = result.node_tree.nodes.get('Principled BSDF')
    shader.inputs['Base Color'].default_value = (*color, 1)
    shader.inputs['Roughness'].default_value = roughness
    return result


bpy.ops.mesh.primitive_cube_add(location=(0, 0, 1))
cube = bpy.context.object
cube.name = 'Edit Cube'
cube.data.materials.append(material('Cube Orange', (0.8, 0.13, 0.025)))
bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=0.8, location=(-2.6, 0.5, 0.8))
sphere = bpy.context.object
sphere.name = 'Reference Sphere'
sphere.data.materials.append(material('Sphere Teal', (0.03, 0.45, 0.5)))
for polygon in sphere.data.polygons:
    polygon.use_smooth = True
bpy.ops.mesh.primitive_plane_add(size=200)
floor = bpy.context.object
floor.name = 'Floor'
floor.data.materials.append(material('Floor Gray', (0.2, 0.2, 0.2), 0.7))
bpy.ops.object.light_add(type='AREA', location=(2, -3, 7))
light = bpy.context.object
light.name = 'Key Light'
light.data.energy, light.data.shape, light.data.size = 1700, 'DISK', 5
light.rotation_euler = (Vector((0, 0, 0)) - light.location).to_track_quat('-Z', 'Y').to_euler()
bpy.ops.object.camera_add(location=(7, -11, 6))
camera = bpy.context.object
camera.name = 'Demo Camera'
camera.rotation_euler = (Vector((-0.6, 0, 0.8)) - camera.location).to_track_quat('-Z', 'Y').to_euler()
scene = bpy.context.scene
scene.name = 'Public SDK viewport edits'
scene.camera = camera
scene.render.engine = 'OVRTX_EXAMPLE'
scene.render.resolution_x, scene.render.resolution_y = 960, 640
scene.render.resolution_percentage = 100
scene.ovrtx_example.max_samples = 32
scene.world.color = (0.05, 0.05, 0.05)
bpy.ops.object.select_all(action='DESELECT')
cube.select_set(True)
bpy.context.view_layer.objects.active = cube
for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        space = area.spaces.active
        space.region_3d.view_perspective = 'CAMERA'
        space.region_3d.view_camera_zoom = 5
        space.shading.type = 'RENDERED'
        space.overlay.show_overlays = False
        area.tag_redraw()
    elif area.type == 'PROPERTIES':
        area.spaces.active.context = 'RENDER'
bpy.ops.wm.save_as_mainfile(filepath=str(output / 'Public SDK viewport edits.blend'))
started = time.monotonic()
last_command = 0
command_result = None
depsgraph_events = []


def observe_depsgraph(_scene, depsgraph):
    rows = [{'name': update.id.name, 'type': update.id.bl_rna.identifier,
             'geometry': update.is_updated_geometry, 'transform': update.is_updated_transform}
            for update in depsgraph.updates]
    depsgraph_events.append({'time': time.monotonic() - started, 'ids': rows})
    del depsgraph_events[:-20]


bpy.app.handlers.depsgraph_update_post.append(observe_depsgraph)


def execute(command):
    action = command['action']
    if action == 'transform':
        cube.location.x = float(command.get('x', 1.5))
        cube.rotation_euler.z = float(command.get('angle', 0.55))
        cube.scale.z = float(command.get('scale_z', 1.25))
    elif action == 'material':
        shader = cube.data.materials[0].node_tree.nodes.get('Principled BSDF')
        shader.inputs['Base Color'].default_value = tuple(command.get('rgba', [0.08, 0.2, 0.8, 1]))
        shader.inputs['Roughness'].default_value = float(command.get('roughness', 0.18))
    elif action == 'light':
        light.data.energy = float(command.get('energy', 650))
        light.data.color = tuple(command.get('color', [1, 0.65, 0.3]))
    elif action == 'tag_material':
        cube.data.materials[0].node_tree.update_tag()
        cube.data.materials[0].update_tag()
    elif action == 'material_sockets':
        shader = cube.data.materials[0].node_tree.nodes.get('Principled BSDF')
        for name, value in command['values'].items():
            shader.inputs[name].default_value = value
    elif action == 'texture':
        tree = cube.data.materials[0].node_tree
        shader = tree.nodes.get('Principled BSDF')
        texture = tree.nodes.get('Live Checker Texture')
        if command.get('connected', True):
            if texture is None:
                generated = bpy.data.images.new('Live Checker Source', width=64, height=64)
                pixels = []
                for y in range(64):
                    for x in range(64):
                        pixels.extend((0.9, 0.03, 0.02, 1) if ((x // 8 + y // 8) % 2) else (0.03, 0.9, 0.08, 1))
                generated.pixels[:] = pixels
                generated.filepath_raw = str(output / 'live-checker.png')
                generated.file_format = 'PNG'
                generated.save()
                texture = tree.nodes.new('ShaderNodeTexImage')
                texture.name = 'Live Checker Texture'
                texture.image = bpy.data.images.load(generated.filepath_raw, check_existing=False)
                bpy.data.images.remove(generated)
            tree.links.new(texture.outputs['Color'], shader.inputs['Base Color'])
        else:
            for link in tuple(shader.inputs['Base Color'].links):
                tree.links.remove(link)
    elif action == 'camera':
        camera.location.x = float(command.get('x', 4))
        camera.rotation_euler = (Vector((-0.6, 0, 0.8)) - camera.location).to_track_quat('-Z', 'Y').to_euler()
    elif action == 'add':
        bpy.ops.mesh.primitive_cube_add(size=1.2, location=(-1, 1.8, 0.6))
        obj = bpy.context.object
        obj.name = 'Added During Live View'
        obj.data.materials.append(material('Added Green', (0.08, 0.8, 0.12)))
    elif action == 'delete':
        obj = bpy.data.objects.get('Added During Live View')
        if obj:
            bpy.data.objects.remove(obj, do_unlink=True)
    elif action == 'geometry':
        # Ordinary mesh-data edits must traverse the same generation owner as
        # interactive Edit Mode, without replacing the external native stage.
        for vertex in cube.data.vertices:
            if vertex.co.z > 0:
                vertex.co.x *= 0.25
                vertex.co.z *= 1.6
        cube.data.update()
    elif action == 'restart':
        return engine.restart_ovrtx_workers()
    elif action == 'save':
        path = output / 'Public SDK viewport edited.blend'
        bpy.ops.wm.save_as_mainfile(filepath=str(path))
        return {'applied': action, 'blend': str(path)}
    elif action == 'render':
        previous_path = scene.render.filepath
        try:
            scene.render.filepath = str(output / 'final-render.png')
            scene.render.image_settings.file_format = 'PNG'
            bpy.ops.render.render(write_still=True)
        finally:
            scene.render.filepath = previous_path
        return {'applied': action, 'image': str(output / 'final-render.png')}
    elif action == 'quit':
        bpy.ops.wm.quit_blender()
    else:
        raise ValueError(f'Unknown fixture action: {action}')
    bpy.context.view_layer.update()
    return {'applied': action}


def tick():
    global last_command, command_result
    try:
        path = output / 'command.json'
        if path.exists():
            command = json.loads(path.read_text())
            if command['id'] > last_command:
                last_command = command['id']
                command_result = execute(command)
        presented, errors = [], []
        for instance in engine._live_viewport_engines():
            error = getattr(instance, '_viewport_reported_error', '')
            if error:
                errors.append(error)
            frame = getattr(instance, '_presented_frame', None)
            if frame is not None and frame.render_result is not None:
                result = frame.render_result
                payload = result.rgba8_bytes(origin='top_left')
                presented.append({'width': result.width, 'height': result.height,
                                  'samples': result.completed_samples,
                                  'publication': frame.publication_index,
                                  'sha256': hashlib.sha256(payload).hexdigest(),
                                  'render_var': result.render_var,
                                  'stage_ordinal': result.native_timings.get('stage_ordinal'),
                                  'session': {key: result.native_timings.get(key) for key in
                                      ('runtime_pid', 'runtime_epoch', 'simulation_id', 'worker_load_requests', 'native_stage_id')}})
        report = {'pid': os.getpid(), 'elapsed': time.monotonic() - started,
                  'command_id': last_command, 'command_result': command_result,
                  'depsgraph_events': depsgraph_events,
                  'presented': presented, 'errors': errors,
                  'cube_location': list(cube.location), 'cube_rotation': list(cube.rotation_euler),
                  'cube_color': list(cube.data.materials[0].node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value),
                  'sdk_imports_in_blender': [n for n in ('ovrtx', 'ovphysx', 'ovstage', 'ovstream', 'warp') if n in sys.modules]}
        from ovrtx_blender_example import scene_generation_sessions
        runtime = scene_generation_sessions._runtimes.get(scene.session_uid)
        if runtime is not None:
            diagnostic = runtime.workflow.diagnostics()
            report['edit_workflow'] = {**diagnostic,
                'events': diagnostic['events'][-32:], 'edit_records': diagnostic['edit_records'][-8:]}
        report['edit_bridge'] = engine.interactive_edit_bridge_diagnostics()
        report['generation'] = scene_generation_sessions.diagnostics_for_scene(scene)
        report['texture_connected'] = bool(cube.data.materials[0].node_tree.nodes.get(
            'Principled BSDF').inputs['Base Color'].is_linked)
        temp = output / 'status.tmp'
        temp.write_text(json.dumps(report, indent=2) + '\n')
        temp.replace(output / 'status.json')
    except Exception:
        (output / 'failure.txt').write_text(traceback.format_exc())
        traceback.print_exc()
    return 0.25


bpy.app.timers.register(tick, first_interval=1)

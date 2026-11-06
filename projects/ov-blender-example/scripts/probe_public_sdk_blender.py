#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Blender 5.2 smoke: authored scene -> unchanged engine -> external SDK.

Run using the experiment launcher, --background --python this_file -- --output
an_empty_directory. This authors its own tiny scene and does not open user files.
"""
import argparse
import json
import hashlib
import os
from pathlib import Path
import sys
import time

import bpy
from mathutils import Vector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=1)
    parser.add_argument('--animation', action='store_true', help='Use Blender Render Animation with keyed cube motion')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    args.output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(source / 'addon'))
    import ovrtx_blender_example as addon
    addon.register()
    report = {'blender_version': bpy.app.version_string, 'frames': []}
    try:
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        bpy.ops.mesh.primitive_cube_add(location=(0, 0, 1))
        cube = bpy.context.object
        material = bpy.data.materials.new('Public SDK orange')
        material.use_nodes = True
        bsdf = material.node_tree.nodes.get('Principled BSDF')
        bsdf.inputs['Base Color'].default_value = (0.8, 0.15, 0.025, 1)
        bsdf.inputs['Roughness'].default_value = 0.35
        cube.data.materials.append(material)
        bpy.ops.mesh.primitive_plane_add(size=20)
        bpy.ops.object.light_add(type='AREA', location=(2, -3, 6))
        light = bpy.context.object
        light.data.energy, light.data.shape, light.data.size = 1000, 'DISK', 5
        light.rotation_euler = (Vector((0, 0, 0)) - light.location).to_track_quat('-Z', 'Y').to_euler()
        bpy.ops.object.camera_add(location=(5, -7, 4))
        camera = bpy.context.object
        camera.rotation_euler = (Vector((0, 0, 1)) - camera.location).to_track_quat('-Z', 'Y').to_euler()
        scene = bpy.context.scene
        scene.camera = camera
        scene.render.engine = 'OVRTX_EXAMPLE'
        scene.render.resolution_x, scene.render.resolution_y = 320, 240
        scene.render.resolution_percentage = 100
        scene.ovrtx_example.max_samples = 4
        scene.world.color = (0.05, 0.05, 0.05)
        if args.animation:
            scene.frame_start, scene.frame_end = 1, args.frames
            for index in range(args.frames):
                cube.rotation_euler.z = 0.25 * index
                cube.keyframe_insert(data_path='rotation_euler', frame=index + 1)
            scene.frame_set(1)
        bpy.ops.wm.save_as_mainfile(filepath=str(args.output / 'probe.blend'))
        if args.animation:
            scene.render.filepath = str(args.output / 'animation-')
            scene.render.image_settings.file_format = 'PNG'
            started = time.monotonic()
            bpy.ops.render.render(animation=True)
            report['animation_seconds'] = time.monotonic() - started
        for index in range(args.frames):
            started = time.monotonic()
            if args.animation:
                path = args.output / f'animation-{index + 1:04d}.png'
            else:
                scene.frame_set(index + 1)
                cube.rotation_euler.z = 0.15 * index
                scene.render.filepath = str(args.output / f'frame-{index + 1:03d}.png')
                scene.render.image_settings.file_format = 'PNG'
                bpy.ops.render.render(write_still=True)
                path = Path(scene.render.filepath)
            if not path.is_file() or path.stat().st_size < 100:
                raise RuntimeError('Blender produced no image')
            image = bpy.data.images.load(str(path), check_existing=False)
            values = list(image.pixels)
            rgb = [v for i, v in enumerate(values) if i % 4 != 3]
            if not rgb or max(rgb) - min(rgb) < 0.01:
                raise RuntimeError('Blender output is blank or uniform')
            report['frames'].append({'path': str(path), 'seconds': time.monotonic() - started,
                                     'minimum': min(rgb), 'maximum': max(rgb),
                                     'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            bpy.data.images.remove(image)
        if args.animation and len({frame['sha256'] for frame in report['frames']}) != args.frames:
            raise RuntimeError('Animated frames did not change')
        forbidden = [name for name in ('ovrtx', 'ovphysx', 'ovstage', 'ovstream', 'warp') if name in sys.modules]
        report['sdk_imports_in_blender'] = forbidden
        if forbidden:
            raise RuntimeError(f'NVIDIA SDK imported inside Blender: {forbidden}')
        report['ok'] = True
    except Exception as exc:
        report.update(ok=False, error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        addon.unregister()
        (args.output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()

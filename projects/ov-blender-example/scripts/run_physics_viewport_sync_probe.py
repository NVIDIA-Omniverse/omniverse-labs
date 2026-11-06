# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run in factory-startup Blender: authored geometry, native physics, render, rewind.

blender --background --factory-startup --python-exit-code 17 --python <this file>
  -- --runtime-root <materialized runtime> --output <artifact directory>
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
import uuid

import bpy
from mathutils import Matrix, Vector
import numpy as np


def add_hull_collider(obj):
    """Give obj an owner identity and one convex-hull collider record."""
    from ovrtx_blender_example.physics_endpoints import initialize_owner
    initialize_owner(obj)
    collider = obj.ov.physics.colliders.add()
    collider.name = collider.persistent_id = uuid.uuid4().hex
    collider.schema_opinion = 'APPLY'
    collider.shape = 'CONVEX_HULL'
    return collider


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--addon-root', type=Path, default=Path(__file__).resolve().parents[1] / 'addon')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
    from public_sdk_physics_fixture import add_arguments, configure
    add_arguments(parser)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    args.device = 'cpu'
    args.output = args.output.resolve()
    args.runtime_root = args.runtime_root.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(args.addon_root.resolve()))
    from ovrtx_blender_example import authoring_properties, properties, bundled_runtime
    from ovrtx_blender_example import scene_generation_sessions, interactive_operator_state as mirror
    from ovrtx_blender_example import render_requests
    from ovrtx_blender_example.blender_signal_translation import RenderRequestTranslator
    from ovrtx_blender_example.blender_signals import BlenderRenderSignal, BlenderRenderSignalSource, BlenderRenderIntent
    from ovrtx_blender_example.ovrtx_session_controller import OvrtxSessionController
    from ovrtx_blender_example.ovrtx_value_updates import OvrtxTransformValue, OvrtxAttributeValue
    from ovrtx_blender_example.runtime_scheduler import RuntimeTickStatus
    from ovrtx_blender_example.interactive_edit_planner import (
        InteractiveEdit, InteractiveEditPlanner, EditShape, DataAuthority, edit_location)
    from ovrtx_blender_example.scene_generation import blender_id
    from ovrtx_blender_example.rtpt_worker_config import worker_config_path

    configure(args)

    def free_port():
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            return sock.getsockname()[1]

    if args.public_sdk:
        defaults = None
        os.environ['OV_BLENDER_EXAMPLE_OVPHYSX_DEVICE'] = 'cpu'
    else:
        physics_address = f'127.0.0.1:{free_port()}'
        defaults = bundled_runtime.defaults(root=args.runtime_root, ovrtx_port=str(free_port()),
                                            ovphysx_address=physics_address)
        sys.path.insert(0, defaults.native_client_path)
        os.environ.update({
            'OV_BLENDER_EXAMPLE_WORKER_COMMAND': defaults.worker_command,
            'OV_BLENDER_EXAMPLE_NATIVE_CLIENT_PATH': defaults.native_client_path,
            'OV_BLENDER_EXAMPLE_OVPHYSX_WORKER_COMMAND': defaults.ovphysx_worker_command,
            'OV_BLENDER_EXAMPLE_OVPHYSX_ADDRESS': physics_address,
            'OV_BLENDER_EXAMPLE_OVPHYSX_NATIVE_CLIENT_PATH': defaults.ovphysx_native_client_path,
            'OV_BLENDER_EXAMPLE_OVPHYSX_WORKER_LOG': str(args.output / 'physics-worker.log'),
            'OVPHYSX_ROOT': defaults.ovphysx_root,
            'OV_BLENDER_EXAMPLE_WORKER_LOG': str((args.output / 'renderer-worker.log').resolve()),
            'OVRTX_WORKER_RENDERER_LOG': str((args.output / 'renderer.log').resolve()),
        })
    # No shared-stage opt-in: this must follow ordinary current-scene requests.
    os.environ.pop('OV_BLENDER_EXAMPLE_SHARED_STAGE', None)
    authoring_properties.register()
    properties.register()
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, 121
    scene.render.resolution_x, scene.render.resolution_y = 320, 240
    scene.render.resolution_percentage = 100
    scene.ovrtx_example.min_samples = scene.ovrtx_example.max_samples = 16
    bpy.ops.mesh.primitive_cube_add(location=(0, 0, 3))
    cube = bpy.context.object
    cube.name = 'SyncCube'
    # A simulated body needs a uniformly scaled frame: bake the box extent into
    # the mesh so the cube keeps its (0.5, 0.75, 0.25) size at unit scale.
    cube.data.transform(Matrix.Diagonal((0.5, 0.75, 0.25, 1.0)))
    cube.rotation_euler.z = 0.3
    cube.ov.physics.rigid_body.schema_opinion = 'APPLY'
    cube.ov.physics.rigid_body.mass_kg = 1
    add_hull_collider(cube)
    material = bpy.data.materials.new('Red marker')
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (1, 0, 0, 1)
    bsdf.inputs['Emission Color'].default_value = (1, 0, 0, 1)
    bsdf.inputs['Emission Strength'].default_value = 1
    cube.data.materials.append(material)
    bpy.ops.mesh.primitive_cube_add(location=(0, 0, -0.5))
    floor = bpy.context.object
    floor.name = 'Floor'
    floor.scale = (5, 5, 0.5)
    add_hull_collider(floor)
    scene.ov.physics_scene.schema_opinion = 'APPLY'
    scene.ov.physics_scene.gravity_direction = (0, 0, -1)
    scene.ov.physics_scene.gravity_magnitude = 9.81
    bpy.ops.object.light_add(type='SUN')
    light = bpy.context.object
    light.data.energy = 2
    bpy.ops.object.camera_add(location=(7, -7, 5))
    camera = bpy.context.object
    camera.rotation_euler = (Vector((0, 0, 1.5)) - camera.location).to_track_quat('-Z', 'Y').to_euler()
    scene.camera = camera
    bpy.context.view_layer.update()
    initial_world = cube.matrix_world.copy()
    generation = scene_generation_sessions.generation_for_scene(scene, work_root=args.output / 'generations')
    paths = scene_generation_sessions.current_generation_object_paths(scene)
    assert paths[cube.session_uid] == generation.blender_prim_paths[blender_id(cube, 'OBJECT')].object_path
    assert not cube.get('ovrtx.usd_prim_path'), 'Probe must exercise untagged objects'
    camera_path = generation.blender_prim_paths[blender_id(camera, 'OBJECT')].schema_path
    projection = camera.calc_matrix_camera(bpy.context.evaluated_depsgraph_get(), x=320, y=240)
    region_data = SimpleNamespace(view_perspective='PERSP', window_matrix=projection)
    context = SimpleNamespace(scene=scene, screen=SimpleNamespace(is_animation_playing=False),
                              region=SimpleNamespace(width=320, height=240), region_data=region_data)
    translator = RenderRequestTranslator()
    renderer = OvrtxSessionController()
    owner = scene_generation_sessions._owners[scene.session_uid]
    runtime = scene_generation_sessions.AuthoringGenerationRuntime(renderer, owner)
    physics, scheduler = runtime.ovphysx, runtime.scheduler
    light_path = generation.blender_prim_paths[blender_id(light, 'OBJECT')].schema_path
    owner.retain_transform_values((OvrtxTransformValue(paths[cube.session_uid], tuple(tuple(row) for row in initial_world.transposed())),))
    owner.retain_attribute_values((OvrtxAttributeValue(light_path, 'inputs:intensity', 2.0, 'Float'),))
    lock = mirror.PhysicsPlaybackLock()
    evidence = []
    observed_runtimes, streams = [], set()
    final_result = dict(status='running', checks=evidence, cleanup_errors=[])
    if args.public_sdk:
        from ovrtx_blender_example.direct_sdk_client import PublicSdkRenderClient
        original_copy = PublicSdkRenderClient._copy_frame
        def copied(client, result, *positional, **keywords):
            if client.runtime not in observed_runtimes:
                observed_runtimes.append(client.runtime)
            if result.get('shm'):
                streams.add(result['shm']['stream_name'])
            return original_copy(client, result, *positional, **keywords)
        PublicSdkRenderClient._copy_frame = copied

    def request_for(frame, view_position=(5, -8, 6)):
        scene.frame_set(frame)
        region_data.view_perspective = 'CAMERA' if view_position is None else 'PERSP'
        if view_position is None:
            view_world = camera.matrix_world
        else:
            rotation = (Vector((0, 0, 1.5)) - Vector(view_position)).to_track_quat('-Z', 'Y')
            view_world = Matrix.LocRotScale(Vector(view_position), rotation, Vector((1, 1, 1)))
        region_data.view_matrix = view_world.inverted()
        return translator.translate(BlenderRenderSignal(
            source=BlenderRenderSignalSource.VIEW_DRAW, intent=BlenderRenderIntent.VIEWPORT,
            scene=scene, context=context, current_scene_generation=True,
            input_usd_path=generation.materialize_usd(), camera_prim_path=camera_path,
            render_product_path=properties.DEFAULT_RENDER_PRODUCT_PATH))

    def queue_light(intensity):
        edit = InteractiveEdit(shape=EditShape.VALUE, data_authority=DataAuthority.VIEW,
            **edit_location(usd_prim_path=light_path, usd_attribute='inputs:intensity',
                            blender_property_path='energy',
                            provenance={'source': 'probe', 'light_path': light_path, 'usd_family': 'DistantLight'}),
            value=intensity)
        scheduler.submit_edit(InteractiveEditPlanner().plan(edit).to_intent())

    def tick(request, complete=False):
        def apply(port, project):
            result = scheduler.tick_viewport(render_requests.tick(request), ovrtx_updates=port,
                                            project_complete_pose=complete or project)
            port.update_transforms((OvrtxTransformValue(request.camera_prim_path, request.camera_matrix),))
            return result
        result = renderer.apply_runtime_updates(apply)
        assert result.status != RuntimeTickStatus.FAILED, result
        pending, diagnostics = mirror.prepare_runtime_pose_mirror(
            result.physics_pose_set, bpy, context, object_paths_by_session_uid=paths,
            lock_runtime_owned=False)
        applied = mirror.apply_pending_runtime_pose_mirror(bpy, pending, lock)
        assert applied and applied['mirrored_count'] == 1, (diagnostics, applied)
        # The mirror must keep the authored (baked, unit) Object scale exactly.
        assert max(abs(a - b) for a, b in zip(cube.matrix_world.decompose()[2], (1.0, 1.0, 1.0))) < 1e-5
        return result

    def capture(label, request, result):
        print('Rendering ' + label, flush=True)
        rendered = renderer.render(request, additional_samples=16)
        if rendered.linear_rgba16f:
            colors = np.frombuffer(rendered.linear_rgba16f, dtype=np.float16).reshape(rendered.height, rendered.width, 4).astype(np.float32)
            pixels = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
        else:
            pixels = np.frombuffer(rendered.rgba8_bytes(), dtype=np.uint8).reshape(rendered.height, rendered.width, 4)
            colors = pixels.astype(np.float32) / 255
        print('Frame', rendered.frame_format, 'range', float(colors.min()), float(colors.max()), flush=True)
        image = bpy.data.images.new(label, width=rendered.width, height=rendered.height)
        image.pixels.foreach_set(colors.reshape(-1))
        image.filepath_raw = str(args.output / (label + '.png'))
        image.file_format = 'PNG'
        image.save()
        # Bottom-left origin matches Blender's clip-to-window projection.
        red = (pixels[:, :, 0] > 70) & (pixels[:, :, 0] > 1.8 * pixels[:, :, 1]) & (pixels[:, :, 0] > 1.8 * pixels[:, :, 2])
        yy, xx = np.nonzero(red)
        assert len(xx) > 20, ('No rendered red cube', request.camera_matrix, request.camera_projection, request.color_presentation)
        actual = [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())]
        projected = []
        for corner in cube.bound_box:
            clip = projection @ region_data.view_matrix @ cube.matrix_world @ Vector((*corner, 1))
            projected.append(((clip.x / clip.w + 1) * 160, (clip.y / clip.w + 1) * 120))
        expected = [min(p[0] for p in projected), min(p[1] for p in projected),
                    max(p[0] for p in projected), max(p[1] for p in projected)]
        error = max(abs(a - b) for a, b in zip(actual, expected))
        record = dict(label=label, blender_z=float(cube.matrix_world.translation.z),
                      simulation_time_ns=result.simulation_time_ns, timeline_reset=result.timeline_reset,
                      rendered_bbox=actual, blender_bbox=expected, max_pixel_error=error)
        rgba_path, colors_path = args.output / (label + '.npy'), args.output / (label + '-colors.npy')
        np.save(rgba_path, pixels)
        np.save(colors_path, colors)
        record.update(rgba_path=rgba_path.name, rgba_sha256=hashlib.sha256(rgba_path.read_bytes()).hexdigest(),
                      colors_path=colors_path.name, pixel_origin='bottom_left',
                      projection_matrix=[list(row) for row in projection],
                      view_matrix=[list(row) for row in region_data.view_matrix],
                      cube_world_matrix=[list(row) for row in cube.matrix_world],
                      cube_bound_box=[list(row) for row in cube.bound_box], native=dict(rendered.native_timings))
        if args.public_sdk:
            physics_owner = physics.controller._physics_client.runtime
            if physics_owner not in observed_runtimes:
                observed_runtimes.append(physics_owner)
            record['physics_runtime'] = dict(identity=asdict(physics_owner.identity),
                                             health=physics_owner.request('health', {}))
        evidence.append(record)
        (args.output / 'progress.json').write_text(json.dumps(evidence, indent=2))
        assert error < 3, record
        print(json.dumps(record), flush=True)

    # Session startup authors launch-time quality settings. Leave the supplied
    # runtime exactly as found, including its authenticated configuration bytes.
    config_path = worker_config_path(defaults.worker_command) if defaults is not None else None
    original_config = config_path.read_bytes() if config_path is not None else None
    try:
        request = request_for(1)
        assert request.timeline_controls_enabled
        runtime.activate_blocking(generation, request)
        from pxr import Usd
        Usd.Stage.Open(renderer._spec.ovrtx_scene_composition.composed_scene_path).Export(str(args.output / 'render-input.usda'))
        initial = tick(request, complete=True)
        capture('initial', request, initial)
        for frame in range(2, 122):
            queue_light(2.0 + frame / 121)
            request = request_for(frame)
            settled = tick(request)
            # Several producers can publish before Blender presents a frame.
            # Do not hide duplicate-write bugs by rendering between every tick.
        assert abs(cube.location.z - 0.25) < 0.02, cube.location.z
        capture('settled', request, settled)
        physics_generation = physics.physics_generation
        assert runtime.ovrtx.deactivate() in {'stopped', 'not_found'}
        request = request_for(121)
        runtime.activate_blocking(generation, request)
        # Edits arriving during re-entry are drained by activation. Then Numpad 0
        # supplies another camera value before any rendering has sealed them.
        for intensity in (4.0, 5.0):
            queue_light(intensity)
            runtime.activate_blocking(generation, request)
        tick(request, complete=True)
        request = request_for(121, None)
        runtime.activate_blocking(generation, request)
        camera_switched = tick(request)
        capture('reentered_camera', request, camera_switched)
        assert physics.physics_generation == physics_generation
        request = request_for(1)
        runtime.activate_blocking(generation, request)
        reset = tick(request)
        assert reset.timeline_reset and reset.simulation_time_ns == 0
        assert max(abs(cube.matrix_world[r][c] - initial_world[r][c]) for r in range(4) for c in range(4)) < 1e-5
        capture('rewound', request, reset)
        request = request_for(1, (-6, -7, 5))
        moved_camera = tick(request)
        capture('orbited', request, moved_camera)
        assert moved_camera.simulation_time_ns == 0, 'Orbiting a paused scene advanced physics'
        for frame in range(2, 122):
            request = request_for(frame, (-6, -7, 5))
            replay = tick(request)
        assert abs(cube.location.z - 0.25) < 0.02
        capture('replayed', request, replay)
        final_result['status'] = 'passed'
    finally:
        # Every independent owner gets its close even when an earlier close or
        # numeric assertion fails. Keep the original exception and raw result.
        for name, close in [('scheduler', scheduler.shutdown), ('physics', physics.deactivate),
                            ('renderer', renderer.shutdown)]:
            try:
                status = close()
                final_result.setdefault('shutdown_statuses', {})[name] = status
                if name != 'scheduler' and status not in {'stopped', 'not_found'}:
                    final_result['cleanup_errors'].append(dict(owner=name, status=status))
            except Exception as exc:
                final_result['cleanup_errors'].append(dict(owner=name, error=repr(exc)))
        if args.public_sdk:
            PublicSdkRenderClient._copy_frame = original_copy
            from run_ovstream_raw_frame_smoke import resource_state
            final_result['runtimes_after_shutdown'] = [owner.diagnostics() for owner in observed_runtimes]
            final_result['resources_after_shutdown'] = {stream: resource_state(stream) for stream in sorted(streams)}
        forbidden = {'ovrtx', 'ovphysx', 'ovstage', 'ovstream', 'warp', 'ovusd', 'ovmaterial', 'ovwarp'}
        final_result['vendor_sdk_imports'] = sorted(key for key in sys.modules if key.split('.')[0] in forbidden)
        try:
            if config_path is not None and original_config is not None:
                config_path.write_bytes(original_config)
        finally:
            if sys.exc_info()[0] is not None or final_result['cleanup_errors']:
                final_result['status'] = 'failed'
            (args.output / 'result.json').write_text(json.dumps(final_result, indent=2, default=str))
    if args.public_sdk:
        from run_physics_workflows import validate_viewport_result, verify_viewport_artifacts
        validate_viewport_result(final_result)
        verify_viewport_artifacts(final_result, args.output)


if __name__ == '__main__':
    main()

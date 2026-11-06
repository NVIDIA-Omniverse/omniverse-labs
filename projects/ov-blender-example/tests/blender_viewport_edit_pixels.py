# SPDX-License-Identifier: Apache-2.0
"""Actual view_update/view_draw edit acceptance; renderer runs on its owner thread."""
import json
import hashlib
import os
from pathlib import Path
import runpy
import time
import traceback

import bpy
import numpy as np

source = Path(__file__).resolve().parents[1]
fixture = runpy.run_path(str(source / 'scripts/probe_public_sdk_ui_edits.py'))
engine = fixture['engine']
scene = fixture['scene']
scene.ovrtx_example.color_presentation_mode = 'ldr_rgba8_display_passthrough'
out = Path(os.environ['OV_BLENDER_PUBLIC_SDK_OUTPUT']) / 'viewport-acceptance'
out.mkdir(parents=True, exist_ok=True)
report = {'status': 'running', 'scope': 'actual rendered viewport callbacks', 'cases': []}
actions = [('initial', None), ('held', None),
           ('transform', {'action': 'transform', 'x': 2}),
           ('material', {'action': 'material', 'rgba': [.03, .1, .9, 1]}),
           ('light', {'action': 'light', 'energy': 100}),
           ('texture', {'action': 'texture'}),
           ('packed_texture', {'action': 'packed_texture'}),
           ('geometry', {'action': 'geometry'}),
           ('add', {'action': 'add'}), ('delete', {'action': 'delete'})]
index = 0
armed = False
started = time.monotonic()
changed_at = started
previous = None
threshold = .2
publication = -1
identity = None
attempt_publication = -1
added_identity = None


def finish():
    (out / 'result.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    # Quit from the main-thread timer; add-on lifecycle owns runtime shutdown.
    bpy.ops.wm.quit_blender()
    return None


def tick():
    global index, armed, changed_at, previous, threshold, publication, identity
    global attempt_publication, added_identity
    try:
        if time.monotonic() - changed_at > 180:
            raise TimeoutError('Viewport did not acknowledge ' + actions[index][0])
        name, action = actions[index]
        if not armed:
            if action:
                if action['action'] == 'packed_texture':
                    packed = bpy.data.images.new('Packed Only Checker', width=64, height=64)
                    rgba = [(0.03, .04, .95, 1) if ((x // 8 + y // 8) % 2) else (.95, .85, .02, 1)
                            for y in range(64) for x in range(64)]
                    packed.pixels.foreach_set(np.array(rgba, dtype=np.float32).ravel())
                    packed.filepath_raw = str(out / 'intentionally-missing-packed.png')
                    packed.file_format = 'PNG'
                    packed.save()
                    bpy.data.images.remove(packed)
                    packed = bpy.data.images.load(str(out / 'intentionally-missing-packed.png'), check_existing=False)
                    packed.name = 'Packed Only Checker'
                    packed.pack()
                    Path(packed.filepath_raw).unlink()
                    assert packed.packed_file and not Path(packed.filepath_raw).exists()
                    fixture['cube'].data.materials[0].node_tree.nodes.get('Live Checker Texture').image = packed
                    fixture['cube'].data.materials[0].update_tag()
                    report['packed_texture_sha256'] = hashlib.sha256(bytes(packed.packed_file.data)).hexdigest()
                elif action['action'] == 'add':
                    # The retained small cube is visible but below the frozen
                    # whole-image noise gate after dimming the ordinary light.
                    # Use a clear foreground footprint; keep light and gate.
                    bpy.ops.mesh.primitive_cube_add(size=2.4, location=(-1, -1.8, 1.2))
                    added = bpy.context.object
                    added.name = 'Added During Live View'
                    from ovrtx_blender_example.scene_generation import blender_id
                    added_identity = blender_id(added, 'OBJECT')
                    added.data.materials.append(fixture['material']('Added Green', (0.08, 0.8, 0.12)))
                    bpy.context.view_layer.update()
                    report['added_fixture'] = dict(size=2.4, location=[-1, -1.8, 1.2])
                else:
                    fixture['execute'](action)
            changed_at = time.monotonic()
            armed = True
            return .2
        if time.monotonic() - changed_at < 2:
            return .2
        for instance in engine._live_viewport_engines():
            error = getattr(instance, '_viewport_reported_error', '')
            if error:
                raise AssertionError(error)
            frame = getattr(instance, '_presented_frame', None)
            if frame is None or frame.render_result is None:
                continue
            result = frame.render_result
            if result.completed_samples < 16:
                continue
            if action and frame.publication_index <= publication:
                continue
            payload = result.rgba8_bytes(origin='top_left')
            pixels = np.frombuffer(payload, np.uint8).reshape(result.height, result.width, 4).copy()
            assert pixels[..., :3].std() > 5
            delta = None if previous is None else float(np.abs(pixels.astype(float) - previous).mean())
            if action and frame.publication_index != attempt_publication:
                # Preserve below-gate evidence too: an accepted generation and
                # a new publication alone do not prove a visible scene edit.
                from ovrtx_blender_example import scene_generation_sessions
                from pxr import Usd
                diagnostic = scene_generation_sessions.diagnostics_for_scene(bpy.context.scene)
                attempt = dict(name=name, pixel_mae=delta, samples=result.completed_samples,
                               publication=frame.publication_index, generation=diagnostic,
                               native_timings=result.native_timings)
                np.save(out / (name + '-attempt.npy'), pixels)
                owner = scene_generation_sessions._owners[bpy.context.scene.session_uid]
                stage = Usd.Stage.Open(owner.current_generation.usd_path)
                stage.Flatten().Export(str(out / (name + '-attempt.usda')))
                (out / (name + '-attempt.json')).write_text(json.dumps(attempt, indent=2, default=str) + '\n')
                attempt_publication = frame.publication_index
            if action and delta <= threshold:
                continue
            if name in {'add', 'delete'}:
                # A prior generation may still publish while the edit is in
                # flight. Check ownership only after the visible-edit gate.
                owner = scene_generation_sessions._owners[bpy.context.scene.session_uid]
                stage = Usd.Stage.Open(owner.current_generation.usd_path)
                added_paths = [str(prim.GetPath()) for prim in stage.Traverse()
                               if prim.GetName() == 'Added_During_Live_View']
                assert bool(added_paths) == (name == 'add'), (name, added_paths)
                assert (added_identity in owner.current_generation.blender_prim_paths) == (name == 'add')
            if name == 'delete':
                restored = np.load(out / 'geometry.npy')
                restoration_mae = float(np.abs(pixels.astype(float) - restored.astype(float)).mean())
                assert restoration_mae < 2, ('delete did not restore the pre-add image', restoration_mae)
                report['delete_restoration_mae'] = restoration_mae
            current = tuple(result.native_timings.get(key) for key in ('runtime_pid', 'native_stage_id'))
            assert all(current), current
            if identity is None:
                identity = current
            assert identity == current, (name, identity, current)
            if name == 'held':
                threshold = max(.2, 5 * delta)
                report['pixel_change_threshold'] = threshold
            from ovrtx_blender_example import scene_generation_sessions
            if name == 'packed_texture':
                from pxr import Sdf, Usd
                owner = scene_generation_sessions._owners[bpy.context.scene.session_uid]
                stage = Usd.Stage.Open(owner.current_generation.usd_path)
                from ovrtx_blender_example.scene_generation import blender_id
                mapping = owner.current_generation.blender_prim_paths[blender_id(fixture['cube'], 'OBJECT')]
                uvs = stage.GetPrimAtPath(mapping.schema_path).GetAttribute('primvars:st').Get()
                assert uvs is not None and len(uvs) >= 4, 'packed checker UV mapping absent'
                matching_assets = []
                from ovrtx_blender_example.texture_materialization import materialized_image_path
                reference_path = Path(materialized_image_path(bpy.data.images['Packed Only Checker']))
                assert hashlib.sha256(reference_path.read_bytes()).hexdigest() == report['packed_texture_sha256']
                reference_image = bpy.data.images.load(str(reference_path), check_existing=False)
                reference_pixels = np.array(reference_image.pixels[:])
                for prim in stage.Traverse():
                    for attr in prim.GetAttributes():
                        value = attr.Get()
                        if isinstance(value, Sdf.AssetPath):
                            path = Path(value.resolvedPath or value.path)
                            if path.is_file():
                                decoded = bpy.data.images.load(str(path), check_existing=False)
                                if tuple(decoded.size) == tuple(reference_image.size) and np.allclose(
                                        np.array(decoded.pixels[:]), reference_pixels, atol=1 / 255, rtol=0):
                                    matching_assets.append(str(path))
                                bpy.data.images.remove(decoded)
                bpy.data.images.remove(reference_image)
                assert matching_assets, 'accepted USD does not preserve decoded packed texels'
                rgb = pixels[..., :3].astype(float)
                blue = (rgb[..., 2] > rgb[..., 0] + 20) & (rgb[..., 2] > rgb[..., 1] + 20)
                yellow = (rgb[..., 0] > rgb[..., 2] + 20) & (rgb[..., 1] > rgb[..., 2] + 20)
                assert blue.sum() > 16 and yellow.sum() > 16, 'packed checker colors absent'
                report['packed_texture'] = dict(materialized_assets=matching_assets,
                    uv_count=len(uvs), blue_pixels=int(blue.sum()), yellow_pixels=int(yellow.sum()), original_file_exists=False)
            report['cases'].append(dict(name=name, pixel_mae=delta, identity=current,
                                       publication=frame.publication_index, samples=result.completed_samples,
                                       generation=scene_generation_sessions.diagnostics_for_scene(bpy.context.scene)))
            np.save(out / (name + '.npy'), pixels)
            previous = pixels
            publication = frame.publication_index
            index += 1
            armed = False
            (out / 'result.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
            if index == len(actions):
                report['status'] = 'passed'
                return finish()
            break
        return .2
    except Exception:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        return finish()


bpy.app.timers.register(tick, first_interval=2)

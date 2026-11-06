# SPDX-License-Identifier: Apache-2.0
"""Actual stock Blender undo/redo/file reload and unit ownership contracts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

import bpy
import numpy as np
from pxr import Sdf, Usd, UsdGeom

sys.path.insert(0, str(Path(__file__).parents[1] / 'addon'))
from ovrtx_blender_example import scene_generation_sessions as sessions
from ovrtx_blender_example.scene_generation import SceneGenerationOwner, blender_id

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--generated-packed', action='store_true')
args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=True)
report = {'status': 'running', 'checks': [], 'scope': 'stock Blender composed stage and actual undo/file lifecycle; CPU only'}
report['blender_version'] = bpy.app.version_string
report['source_sha256'] = {str(path.relative_to(Path(__file__).parents[1])): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in (Path(__file__).parents[1] / 'addon').rglob('*.py')}
owner = SceneGenerationOwner(out / 'generations')


def capture(name):
    scene = bpy.context.scene
    bpy.context.view_layer.update()
    generation = owner.replace(scene)
    stage = Usd.Stage.Open(generation.usd_path)
    stage.Flatten().Export(str(out / (name + '.usda')))
    names = {mapping.blender_id_name for mapping in generation.blender_prim_paths.values()}
    report['checks'].append(dict(name=name, generation=generation.number, owners=sorted(names)))
    return generation, stage, names


try:
    bpy.context.preferences.edit.use_global_undo = True
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_cube_add()
    cube = bpy.context.object
    cube.name = 'LifecycleSubject'
    bpy.ops.object.camera_add(location=(0, -8, 4))
    bpy.context.scene.camera = bpy.context.object
    first, stage, names = capture('initial')
    assert 'LifecycleSubject' in names
    bpy.ops.ed.undo_push(message='accepted subject')
    bpy.data.objects.remove(cube, do_unlink=True)
    bpy.ops.ed.undo_push(message='deleted subject')
    deleted, stage, names = capture('deleted')
    assert 'LifecycleSubject' not in names
    assert bpy.ops.ed.undo() == {'FINISHED'}
    restored, stage, names = capture('undo_restored')
    assert 'LifecycleSubject' in names
    assert bpy.ops.ed.redo() == {'FINISHED'}
    redone, stage, names = capture('redo_deleted')
    assert 'LifecycleSubject' not in names
    assert bpy.ops.ed.undo() == {'FINISHED'}
    restored, stage, names = capture('second_undo')
    scene = bpy.context.scene
    scene.unit_settings.system = 'METRIC'
    scene.unit_settings.scale_length = .01
    generation, stage, names = capture('units_centimeters')
    cube = bpy.data.objects['LifecycleSubject']
    prim = stage.GetPrimAtPath(generation.blender_prim_paths[blender_id(cube, 'OBJECT')].object_path)
    actual = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(prim)).T
    expected = np.diag([.01, .01, .01, 1]) @ np.array(cube.matrix_world)
    assert np.allclose(actual, expected, atol=1e-6), (actual, expected)
    report['units_physical_matrix'] = actual.tolist()
    report['meters_per_blender_unit'] = .01
    packed = bpy.data.images.new('Lifecycle Packed Texture', width=8, height=8)
    packed.pixels.foreach_set(np.tile([.8, .1, .03, 1], 64).astype(np.float32))
    packed.filepath_raw = str(out / 'intentionally-missing-packed.png')
    packed.file_format = 'PNG'
    if not args.generated_packed:
        packed.save()
        bpy.data.images.remove(packed)
        packed = bpy.data.images.load(str(out / 'intentionally-missing-packed.png'), check_existing=False)
        packed.name = 'Lifecycle Packed Texture'
    packed.pack()
    if Path(packed.filepath_raw).exists():
        Path(packed.filepath_raw).unlink()
    assert packed.packed_file and not Path(packed.filepath_raw).exists()
    packed_digest = hashlib.sha256(bytes(packed.packed_file.data)).hexdigest()
    material = bpy.data.materials.new('Lifecycle Packed Material')
    material.use_nodes = True
    texture = material.node_tree.nodes.new('ShaderNodeTexImage')
    texture.image = packed
    material.node_tree.links.new(texture.outputs['Color'], material.node_tree.nodes.get('Principled BSDF').inputs['Base Color'])
    cube.data.materials.append(material)
    packed_generation, packed_stage, _ = capture('packed_texture')
    from ovrtx_blender_example.texture_materialization import materialized_image_path
    reference_path = Path(materialized_image_path(packed))
    assert hashlib.sha256(reference_path.read_bytes()).hexdigest() == packed_digest
    reference_image = bpy.data.images.load(str(reference_path), check_existing=False)
    reference_pixels = np.array(reference_image.pixels[:])
    matching_assets = []
    report['packed_asset_diagnostics'] = []
    for prim in packed_stage.Traverse():
        for attr in prim.GetAttributes():
            value = attr.Get()
            if isinstance(value, Sdf.AssetPath):
                asset_path = Path(value.resolvedPath or value.path)
                diagnostic = dict(path=str(asset_path), exists=asset_path.is_file(), authored=value.path)
                report['packed_asset_diagnostics'].append(diagnostic)
                if asset_path.is_file():
                    decoded = bpy.data.images.load(str(asset_path), check_existing=False)
                    diagnostic.update(size=list(decoded.size), reference_size=list(reference_image.size))
                    if tuple(decoded.size) == tuple(reference_image.size):
                        diagnostic['max_pixel_error'] = float(np.max(np.abs(np.array(decoded.pixels[:]) - reference_pixels)))
                    if tuple(decoded.size) == tuple(reference_image.size) and np.allclose(
                            np.array(decoded.pixels[:]), reference_pixels, atol=1 / 255, rtol=0):
                        matching_assets.append(str(asset_path))
                    bpy.data.images.remove(decoded)
    bpy.data.images.remove(reference_image)
    assert matching_assets, 'composed USD does not preserve decoded packed texels'
    mesh = packed_stage.GetPrimAtPath(packed_generation.blender_prim_paths[blender_id(cube, 'OBJECT')].schema_path)
    assert len(mesh.GetAttribute('primvars:st').Get()) >= 4
    report['packed_texture_assets'] = matching_assets
    path = out / 'accepted.blend'
    bpy.ops.wm.save_as_mainfile(filepath=str(path))
    bpy.data.objects.remove(bpy.data.objects['LifecycleSubject'], do_unlink=True)
    capture('unsaved_deleted')
    # Use the production pre-load cleanup on real Blender RNA destruction.
    bpy.app.handlers.load_pre.append(sessions.load_pre)
    bpy.ops.wm.open_mainfile(filepath=str(path))
    assert sessions.diagnostics()['load_pre_completed']
    generation, stage, names = capture('file_restored')
    assert 'LifecycleSubject' in names
    packed = bpy.data.images['Lifecycle Packed Texture']
    assert hashlib.sha256(bytes(packed.packed_file.data)).hexdigest() == packed_digest
    assert not Path(packed.filepath_raw).exists()
    from ovrtx_blender_example.texture_materialization import materialized_image_path
    materialized = Path(materialized_image_path(packed))
    assert materialized.is_file() and hashlib.sha256(materialized.read_bytes()).hexdigest() == packed_digest
    report['packed_texture_reload'] = dict(sha256=packed_digest, materialized_path=str(materialized),
                                         original_file_exists=False)
    cube = bpy.data.objects['LifecycleSubject']
    prim = stage.GetPrimAtPath(generation.blender_prim_paths[blender_id(cube, 'OBJECT')].object_path)
    assert np.allclose(np.array(UsdGeom.XformCache().GetLocalToWorldTransform(prim)).T,
                       np.diag([.01, .01, .01, 1]) @ np.array(cube.matrix_world), atol=1e-6)
    # Actual RNA admission is independent of low-level USD authoring/physics.
    from ovrtx_blender_example import properties, authoring_properties
    from ovrtx_blender_example.blender_callback_adapters import BlenderRenderCallbackAdapter, ExactStageRenderCallbackAdapter
    from ovrtx_blender_example.blender_signal_translation import BlenderSignalTranslationError
    properties.register()
    authoring_properties.register()
    scene = bpy.context.scene
    sessions._owners[scene.session_uid] = owner
    calls = []
    def provide(scene):
        calls.append(scene.session_uid)
        return sessions.generation_for_scene(scene)
    adapter = BlenderRenderCallbackAdapter(generation_for_scene=provide)
    report['candidate_admission'] = []
    for field in ('matte', 'holdout'):
        for value in ('ON', 'OFF'):
            accepted = owner.current_generation
            setattr(cube.ov.rendering, field, value)
            before = len(calls)
            for callback in (lambda: adapter.final_render_from_scene(scene),
                             lambda: adapter.view_update_from_scene(scene, bpy.context),
                             lambda: adapter.view_draw_from_scene(scene, bpy.context)):
                try:
                    callback()
                except BlenderSignalTranslationError as exc:
                    assert 'unavailable' in str(exc)
                else:
                    raise AssertionError('unadmitted RNA override accepted')
                assert owner.current_generation is accepted and owner.pending_generation is None
            assert len(calls) == before + 1, 'unchanged refusal rebuilt the population'
            rejected = owner._failed[-1]
            exact = ExactStageRenderCallbackAdapter(input_usd_path=rejected.usd_path,
                camera_prim_path=rejected.blender_prim_paths[blender_id(scene.camera, 'OBJECT')].schema_path,
                render_product_path=properties.DEFAULT_RENDER_PRODUCT_PATH)
            assert exact.final_render_from_scene(scene).input_usd_path == rejected.usd_path
            setattr(cube.ov.rendering, field, 'INHERIT')
            adapter.final_render_from_scene(scene)
            if owner.pending_generation is not None:
                assert not owner.pending_generation.unadmitted_rendering_overrides
                owner.accept(owner.pending_generation)
            report['candidate_admission'].append(dict(field=field, value=value,
                callbacks_refused=3, accepted_preserved=True, exact_stage_preserved=True, inherit_recovered=True))
    assert not any(name in sys.modules for name in ('ovrtx', 'ovstage', 'ovstream', 'ovphysx', 'warp'))
    report['status'] = 'passed'
except Exception:
    report['status'] = 'failed'
    report['error'] = traceback.format_exc()
    raise
finally:
    owner.close()
    (out / 'result.json').write_text(json.dumps(report, indent=2) + '\n')

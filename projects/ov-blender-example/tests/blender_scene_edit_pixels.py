# SPDX-License-Identifier: Apache-2.0
"""Stock Blender current-scene edits through the production external worker.

Run with an isolated profile and locked OV_BLENDER_PUBLIC_SDK_PYTHON. This
fixture qualifies scene compilation/transactions; foreground viewport scheduling
is measured separately. No native NVIDIA module is imported into Blender.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import traceback

import bpy
from mathutils import Vector
import numpy as np
from pxr import Usd, UsdGeom

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'addon'))
from ovrtx_blender_example import authoring_properties, color_presentation, properties
from ovrtx_blender_example.blender_callback_adapters import BlenderRenderCallbackAdapter
from ovrtx_blender_example.ovrtx_session_controller import OvrtxSessionController
from ovrtx_blender_example.scene_generation import SceneGenerationOwner, blender_id

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=True)
report = {'status': 'running', 'cases': [], 'scope': 'current-scene generation and external renderer; not foreground viewport qualification'}
report['source_sha256'] = {str(path.relative_to(Path(__file__).parents[1])): hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in Path(__file__).parents[1].rglob('*.py') if '__pycache__' not in path.parts}
report['blender_version'] = bpy.app.version_string
properties.register()
authoring_properties.register()
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
scene = bpy.context.scene
scene.render.resolution_x, scene.render.resolution_y = 320, 240
scene.render.resolution_percentage = 100
scene.ovrtx_example.max_samples = 16
scene.ovrtx_example.dlss_enabled = False
scene.ovrtx_example.color_presentation_mode = color_presentation.MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH
bpy.ops.mesh.primitive_cube_add(location=(0, 0, 1))
cube = bpy.context.object
cube.name = 'Subject'
material = bpy.data.materials.new('SubjectMaterial')
material.use_nodes = True
shader = material.node_tree.nodes.get('Principled BSDF')
shader.inputs['Base Color'].default_value = (.8, .03, .02, 1)
cube.data.materials.append(material)
bpy.ops.mesh.primitive_plane_add(size=20)
floor = bpy.context.object
floor.name = 'Floor'
bpy.ops.object.light_add(type='AREA', location=(2, -3, 6))
light = bpy.context.object
light.data.energy = 1800
light.data.shape = 'DISK'
light.data.size = 3
light.rotation_euler = (Vector((0, 0, 1)) - light.location).to_track_quat('-Z', 'Y').to_euler()
bpy.ops.object.camera_add(location=(5, -8, 4))
camera = bpy.context.object
camera.rotation_euler = (Vector((0, 0, 1)) - camera.location).to_track_quat('-Z', 'Y').to_euler()
scene.camera = camera
scene.world.use_nodes = True
background = scene.world.node_tree.nodes.get('Background')
background.inputs['Color'].default_value = (.05, .05, .05, 1)
background.inputs['Strength'].default_value = .3
bpy.context.view_layer.update()
owner = SceneGenerationOwner(out / 'generations')
controller = OvrtxSessionController()
callback = BlenderRenderCallbackAdapter(generation_for_scene=lambda scene: owner.reuse())
identity = None
last_pixels = None
threshold = .2


def render(name, affected=(), *, expected_changed=True, check=None, fresh=False, new_stage=False,
           retained_request=None):
    global identity, last_pixels
    bpy.context.view_layer.update()
    if fresh:
        # Final/animation capture exports evaluated geometry for this frame;
        # transform-only depsgraph edits belong to the live value lane.
        owner.replace(scene)
    candidate = owner.reconcile(scene, set(affected)) if affected and not fresh else None
    generation = owner.reuse()
    request = retained_request
    if request is None:
        request = callback.final_render_from_scene(scene)
        try:
            controller.ensure(request)
        except Exception:
            if candidate is not None:
                owner.reject(candidate)
            raise
    else:
        # A typed native-value diagnostic measures the already accepted scene.
        # Rebuilding the final request here would author the changed Blender
        # value into a new composition and measure population replacement.
        assert not affected and not fresh and not new_stage
    if candidate is not None:
        owner.accept(candidate)
    result = controller.render(request, additional_samples=16)
    pixels = np.frombuffer(result.rgba8_bytes(origin='top_left'), np.uint8).reshape(240, 320, 4).copy()
    assert pixels[..., :3].std() > 5 and pixels[..., 3].max() > 0, name
    client, session = controller._active()
    health = client._call('health', {})
    report['runtime_versions'] = health['versions']
    current = (health['pid'], health['native_stage_id'])
    if new_stage:
        assert identity != current, (name, 'expected explicit stage replacement')
        identity = current
    if identity is None:
        identity = current
    assert identity == current, (name, identity, current)
    delta = None if last_pixels is None else float(np.abs(pixels.astype(float) - last_pixels).mean())
    stage = Usd.Stage.Open(generation.usd_path)
    stage.Flatten().Export(str(out / (name + '.usda')))
    if check:
        check(stage, generation)
    np.save(out / (name + '.npy'), pixels)
    report['cases'].append(dict(name=name, pixel_mae=delta, generation=generation.number,
                               identity=current, pixels_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                               scene_update=client.startup_diagnostics.get('scene_update'),
                               mappings={item.blender_id_name: item.schema_path for item in generation.blender_prim_paths.values()}))
    last_pixels = pixels
    (out / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    if expected_changed and delta is not None:
        assert delta > threshold, (name, delta, threshold)
    return pixels


try:
    owner.replace(scene)
    first = render('initial')
    held = render('held', expected_changed=False)
    noise = float(np.abs(first.astype(float) - held).mean())
    # Freeze the per-fixture visible-change threshold before evaluating edits.
    threshold = max(.2, 5 * noise)
    report['pixel_change_threshold'] = threshold
    cube_id = blender_id(cube, 'OBJECT')
    shader.inputs['Base Color'].default_value = (.02, .04, .85, 1)
    render('material_blue', [blender_id(material, 'MATERIAL')])
    light.data.energy = 150
    render('light_dim', [blender_id(light.data, 'LIGHT')])
    # Qualify the candidate adapter separately from stock scene re-export.
    from ovrtx_blender_example import direct_light_updates
    from ovrtx_blender_example.generation_runtime_adapters import _replay_ovrtx_values
    from ovrtx_blender_example.ovrtx_value_updates import OvrtxAttributeValue
    from ovrtx_blender_example.runtime_scheduler import RuntimeTickStatus
    generation = owner.reuse()
    light_contract = direct_light_updates.capture_contract(Usd.Stage.Open(generation.usd_path), scene, generation)
    direct_before = last_pixels.copy()
    direct_client, _ = controller._active()
    direct_counts = direct_client.runtime.diagnostics()['request_counts'].copy()
    direct_request = callback.final_render_from_scene(scene)
    report['direct_light_adapter'] = []
    for energy, label in [(600, 'direct_light_brighter'), (150, 'direct_light_restored')]:
        light.data.energy = energy
        bpy.context.view_layer.update()
        planned = direct_light_updates.plan(scene, generation, {blender_id(light.data, 'LIGHT')},
            light_contract, depsgraph=bpy.context.evaluated_depsgraph_get())
        assert not planned.fallback and planned.values
        assert any(row['attribute_name'] == 'inputs:intensity' for row in planned.values)
        values = tuple(OvrtxAttributeValue(row['prim_path'], row['attribute_name'], row['value'], row['value_type'])
                       for row in planned.values)
        acknowledged = controller.apply_runtime_updates(lambda port, _: _replay_ovrtx_values(port, (), values))
        assert acknowledged.status != RuntimeTickStatus.FAILED and acknowledged.values_written
        light_contract = planned.next_contract
        pixels = render(label, retained_request=direct_request)
        counts = direct_client.runtime.diagnostics()['request_counts']
        assert counts.get('update_scene', 0) == direct_counts.get('update_scene', 0)
        assert counts.get('load', 0) == direct_counts.get('load', 0)
        report['direct_light_adapter'].append(dict(energy=energy, values=planned.values,
            rgb_mean=float(pixels[..., :3].mean()), generation=generation.number))
        if energy == 600:
            assert pixels[..., :3].mean() > direct_before[..., :3].mean() + threshold
        else:
            assert float(np.abs(pixels.astype(float) - direct_before).mean()) <= max(2, threshold)
    background.inputs['Color'].default_value = (.8, .8, .8, 1)
    background.inputs['Strength'].default_value = 8
    before_world = last_pixels.copy()
    def world_value(stage, generation):
        domes = [prim for prim in stage.Traverse() if prim.GetTypeName() == 'DomeLight']
        assert len(domes) == 1
        report['world_authored_values'] = {attr.GetName(): str(attr.Get()) for attr in domes[0].GetAttributes() if attr.HasAuthoredValueOpinion()}
    bright = render('world_bright', [blender_id(scene.world, 'WORLD')], check=world_value)
    assert bright[..., :3].mean() > before_world[..., :3].mean() + threshold
    cube.hide_render = True
    def hidden(stage, generation):
        prim = stage.GetPrimAtPath(generation.blender_prim_paths[cube_id].schema_path)
        assert UsdGeom.Imageable(prim).ComputeVisibility() == 'invisible'
    render('hidden', [cube_id], check=hidden)
    cube.hide_render = False
    render('restored', [cube_id])
    for vertex in cube.data.vertices:
        vertex.co.z *= 1.8
    cube.data.update()
    render('geometry', [blender_id(cube.data, 'MESH')])
    mod = cube.modifiers.new('Subdivision', 'SUBSURF')
    mod.levels = mod.render_levels = 2
    render('modifier', [cube_id])
    duplicate = cube.copy()
    duplicate.data = cube.data.copy()
    scene.collection.objects.link(duplicate)
    duplicate.location.x = -2.5
    duplicate_id = blender_id(duplicate, 'OBJECT')
    render('duplicate', [duplicate_id])
    bpy.data.objects.remove(duplicate, do_unlink=True)
    render('delete', [duplicate_id], check=lambda stage, generation: (
        None if duplicate_id not in generation.blender_prim_paths else (_ for _ in ()).throw(AssertionError('stale deleted owner'))))
    parent = bpy.data.objects.new('Parent', None)
    scene.collection.objects.link(parent)
    parent.location.x = 2
    cube.parent = parent
    bpy.context.view_layer.update()
    cube.matrix_parent_inverse = parent.matrix_world.inverted()
    parent.location.x = -1
    cube.scale = (2, 2, 2)
    def world_transform(stage, generation):
        prim = stage.GetPrimAtPath(generation.blender_prim_paths[cube_id].object_path)
        actual = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(prim)).T
        expected = np.array(cube.matrix_world)
        assert np.allclose(actual, expected, atol=1e-5), (actual, expected)
        report['parent_inverse_world_matrix'] = actual.tolist()
    render('parent_inverse', [cube_id, blender_id(parent, 'OBJECT')], check=world_transform)
    parent.location.x -= 1
    # Notify only the parent: descendant invalidation must discover the cube.
    render('parent_descendant', [blender_id(parent, 'OBJECT')], check=world_transform)
    cube.location.x = 2
    cube.keyframe_insert(data_path='location', frame=1)
    cube.location.x = -2
    cube.keyframe_insert(data_path='location', frame=2)
    scene.frame_set(1)
    render('animation_1', [cube_id], fresh=True, check=world_transform)
    scene.frame_set(2)
    render('animation_2', [cube_id], fresh=True, check=world_transform)
    # Shared datablock dependents must all receive a changed geometry export.
    cube.animation_data_clear()
    cube.parent = None
    cube.matrix_parent_inverse.identity()
    cube.location = (-1.5, 0, 1)
    cube.scale = (1, 1, 1)
    cube.modifiers.clear()
    render('dependent_setup', fresh=True)
    linked = cube.copy()
    linked.name = 'LinkedDependent'
    linked.data = cube.data
    scene.collection.objects.link(linked)
    linked.location.x = 1.5
    linked_id = blender_id(linked, 'OBJECT')
    render('linked_user_added', [linked_id])
    for vertex in cube.data.vertices:
        vertex.co.z *= .3
    cube.data.update()
    def dependent_points(stage, generation):
        meshes = [UsdGeom.Mesh(stage.GetPrimAtPath(generation.blender_prim_paths[item].schema_path))
                  for item in (cube_id, linked_id)]
        points = [np.asarray(mesh.GetPointsAttr().Get()) for mesh in meshes]
        assert points[0].shape == points[1].shape and np.allclose(*points)
        assert max(points[0][:, 2]) < 1
    render('shared_geometry_dependents', [blender_id(cube.data, 'MESH')], check=dependent_points)
    # A live curve edit and a collection instance are distinct export families.
    curve_data = bpy.data.curves.new('AcceptanceCurve', 'CURVE')
    curve_data.dimensions = '3D'
    curve_data.bevel_depth = .25
    spline = curve_data.splines.new('POLY')
    spline.points.add(2)
    for point, co in zip(spline.points, [(-2,-2,1,1), (0,-2,1,1), (2,-2,1,1)]):
        point.co = co
    curve = bpy.data.objects.new('AcceptanceCurve', curve_data)
    scene.collection.objects.link(curve)
    curve.data.materials.append(material)
    curve_id = blender_id(curve, 'OBJECT')
    render('curve_added', [curve_id])
    spline.points[1].co.z = 3
    render('curve_edited', [blender_id(curve_data, 'CURVE')])
    # The v9 curve occluded the instance. Establish a visible instance baseline
    # while preserving the frozen initial image-noise threshold.
    curve.hide_render = True
    render('curve_hidden_for_instance', [curve_id])
    prototype = bpy.data.collections.new('InstancePrototype')
    proto = cube.copy()
    proto.data = cube.data.copy()
    proto.location = (0, 0, 0)
    prototype.objects.link(proto)
    instance = bpy.data.objects.new('AcceptanceInstance', None)
    instance.instance_type = 'COLLECTION'
    instance.instance_collection = prototype
    instance.location = (0, -1, 2)
    scene.collection.objects.link(instance)
    instance_id = blender_id(instance, 'OBJECT')
    def instance_present(stage, generation):
        prim = stage.GetPrimAtPath(generation.blender_prim_paths[instance_id].object_path)
        assert prim and any(child.GetTypeName() in {'Mesh', 'Cube', 'Xform'} for child in prim.GetChildren())
    render('collection_instance_added', [instance_id], check=instance_present)
    instance.location.x = 3
    render('collection_instance_moved', fresh=True, check=instance_present)
    bpy.data.objects.remove(instance, do_unlink=True)
    render('collection_instance_deleted', [instance_id])
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=.65, location=(0, -1, 2))
    emitter = bpy.context.object
    emitter.name = 'AcceptanceHair'
    emitter.data.materials.append(material)
    emitter_id = blender_id(emitter, 'OBJECT')
    render('hair_emitter', [emitter_id])
    emitter.modifiers.new('Hair', type='PARTICLE_SYSTEM')
    hair = emitter.particle_systems[0].settings
    hair.type = 'HAIR'
    hair.count = 64
    hair.hair_length = .9
    hair.root_radius, hair.tip_radius, hair.radius_scale = 1, .1, .04
    hair.child_type = 'SIMPLE'
    hair.child_percent = hair.rendered_child_count = 2
    hair.render_step = 3
    def hair_authored(stage, generation):
        curves = [UsdGeom.BasisCurves(prim) for prim in stage.Traverse() if prim.GetTypeName() == 'BasisCurves']
        assert curves
        curve = next(item for item in curves if item.GetTypeAttr().Get() == 'cubic')
        assert curve.GetBasisAttr().Get() == 'catmullRom'
        assert curve.GetWidthsInterpolation() == 'vertex'
        assert UsdGeom.Imageable(curve).ComputeVisibility() == 'inherited'
        assert len(curve.GetWidthsAttr().Get()) == len(curve.GetPointsAttr().Get())
        assert np.isclose(curve.GetWidthsAttr().Get()[0], hair.radius_scale, rtol=.01)
    render('hair_visible', [emitter_id], check=hair_authored)
    hair.radius_scale = .1
    render('hair_width_changed', [emitter_id], check=hair_authored)
    def curves_hidden(stage, generation):
        curves = [prim for prim in stage.Traverse() if prim.GetTypeName() == 'BasisCurves']
        assert curves and all(UsdGeom.Imageable(prim).ComputeVisibility() == 'invisible' for prim in curves)
    emitter.hide_render = True
    render('hair_hidden', [emitter_id], check=curves_hidden)
    emitter.hide_render = False
    render('hair_restored', [emitter_id], check=hair_authored)
    # Shadow visibility is measured separately from unavailable matte/holdout
    # overrides. Their retained v14 raw diagnostics did not establish alpha.
    linked.hide_render = curve.hide_render = emitter.hide_render = True
    cube.location = (0, 0, 1)
    light.data.energy = 1800
    background.inputs['Color'].default_value = (.05, .05, .05, 1)
    background.inputs['Strength'].default_value = .3
    control_before = render('visibility_control_baseline', fresh=True, check=curves_hidden)
    control_held = render('visibility_control_held', expected_changed=False)
    control_noise = float(np.abs(control_before.astype(float) - control_held).mean())
    report['candidate_control_threshold'] = max(.2, 5 * control_noise)
    cube.visible_shadow = False
    shadow_pixels = render('shadow_disabled', [cube_id], expected_changed=False, check=lambda stage, generation:
        report.update(shadow_authored=stage.GetPrimAtPath(generation.blender_prim_paths[cube_id].schema_path)
                      .GetAttribute('primvars:doNotCastShadows').Get()))
    assert report['shadow_authored'] is True
    cube.visible_shadow = True
    shadow_restored = render('shadow_restored', [cube_id], expected_changed=False)
    assert float(np.abs(shadow_restored.astype(float) - control_held).mean()) <= max(2, report['candidate_control_threshold'])
    report['candidate_control_diagnostics'] = {
        name: dict(rgb_mae=float(np.abs(after[..., :3].astype(float) - before[..., :3]).mean()),
                   rgb_mean_change=float((after[..., :3].astype(float) - before[..., :3]).mean()),
                   alpha_mae=float(np.abs(after[..., 3].astype(float) - before[..., 3]).mean()),
                   alpha_min=int(after[..., 3].min()), alpha_max=int(after[..., 3].max()))
        for name, before, after in [('shadow', control_held, shadow_pixels)]}
    assert report['candidate_control_diagnostics']['shadow']['rgb_mae'] > report['candidate_control_threshold']
    assert report['candidate_control_diagnostics']['shadow']['rgb_mean_change'] > report['candidate_control_threshold']
    report['matte_holdout'] = 'unavailable in normal Blender rendering; actual RNA refusal qualified separately'
    # Return to ordinary visible geometry for lifecycle pixel assertions.
    cube.visible_shadow = True
    cube.ov.rendering.matte = cube.ov.rendering.holdout = 'INHERIT'
    linked.hide_render = curve.hide_render = emitter.hide_render = False
    render('candidate_controls_restored', fresh=True)
    # Render compositions use meters. Blender units change physical transforms,
    # while the native stage's meter-valued metadata and identity remain stable.
    scene.unit_settings.system = 'METRIC'
    scene.unit_settings.scale_length = .01
    def physical_units(stage, generation):
        prim = stage.GetPrimAtPath(generation.blender_prim_paths[cube_id].object_path)
        actual = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(prim)).T
        expected = np.diag([.01, .01, .01, 1]) @ np.array(cube.matrix_world)
        assert np.allclose(actual, expected, atol=1e-6), (actual, expected)
        composed = Usd.Stage.Open(controller._spec.ovrtx_scene_composition.composed_scene_path)
        assert composed.HasAuthoredMetadata('metersPerUnit')
        assert UsdGeom.GetStageMetersPerUnit(composed) == 1
        unit_client, unit_session = controller._active()
        assert unit_client.states[unit_session].stage_metadata['metersPerUnit'] == 1
        receipt = unit_client.startup_diagnostics['scene_update']
        assert receipt['scene_updated'] is True
        assert receipt['population_update'] != 'no_change'
        report['units'] = dict(meters_per_blender_unit=.01, render_meters_per_unit=1,
                              world_matrix=actual.tolist(), expected_world_matrix=expected.tolist(),
                              scene_update=receipt)
    render('units_centimeters', fresh=True, expected_changed=False, check=physical_units)
    # Actual Blender undo/redo restores RNA and exported ownership together.
    bpy.context.preferences.edit.use_global_undo = True
    accepted_pixels = last_pixels.copy()
    bpy.ops.ed.undo_push(message='accepted edit scene')
    bpy.data.objects.remove(bpy.data.objects['LinkedDependent'], do_unlink=True)
    bpy.data.objects['AcceptanceCurve'].parent = bpy.data.objects['Parent']
    bpy.data.materials['SubjectMaterial'].node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value = (.9, .6, .02, 1)
    extra = bpy.data.objects['Subject'].copy()
    extra.name = 'UndoAdded'
    extra.location = (0, -2, 3)
    scene.collection.objects.link(extra)
    bpy.ops.ed.undo_push(message='compound add delete parent material')
    changed_pixels = render('undo_candidate_edits', fresh=True)
    assert bpy.ops.ed.undo() == {'FINISHED'}
    scene = bpy.context.scene
    def undo_owners(stage, generation):
        names = {item.blender_id_name for item in generation.blender_prim_paths.values()}
        assert 'LinkedDependent' in names and 'UndoAdded' not in names
        assert bpy.data.objects['AcceptanceCurve'].parent is None
    restored_pixels = render('undo_restored', fresh=True, check=undo_owners)
    restoration_tolerance = max(2, threshold)
    report['undo_restore_mae'] = float(np.abs(restored_pixels.astype(float) - accepted_pixels).mean())
    assert report['undo_restore_mae'] <= restoration_tolerance
    assert bpy.ops.ed.redo() == {'FINISHED'}
    scene = bpy.context.scene
    redone_pixels = render('redo_restored', fresh=True)
    report['redo_restore_mae'] = float(np.abs(redone_pixels.astype(float) - changed_pixels).mean())
    assert report['redo_restore_mae'] <= restoration_tolerance
    assert bpy.ops.ed.undo() == {'FINISHED'}
    scene = bpy.context.scene
    render('undo_before_file', fresh=True, check=undo_owners)
    accepted_file = out / 'accepted-scene.blend'
    bpy.ops.wm.save_as_mainfile(filepath=str(accepted_file))
    bpy.data.objects.remove(bpy.data.objects['LinkedDependent'], do_unlink=True)
    render('unsaved_file_edit', fresh=True)
    controller.shutdown()
    controller = OvrtxSessionController()
    bpy.ops.wm.open_mainfile(filepath=str(accepted_file))
    scene = bpy.context.scene
    file_pixels = render('file_reloaded', fresh=True, new_stage=True, check=undo_owners)
    report['file_restore_mae'] = float(np.abs(file_pixels.astype(float) - accepted_pixels).mean())
    assert report['file_restore_mae'] <= restoration_tolerance
    assert not any(name in sys.modules for name in ('ovrtx', 'ovstage', 'ovstream', 'ovphysx', 'warp'))
    report['native_modules_in_blender'] = []
    report['status'] = 'passed'
except Exception:
    report['status'] = 'failed'
    report['error'] = traceback.format_exc()
    raise
finally:
    controller.shutdown()
    owner.close()
    (out / 'result.json').write_text(json.dumps(report, indent=2) + '\n')

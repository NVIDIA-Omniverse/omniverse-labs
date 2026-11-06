# SPDX-License-Identifier: Apache-2.0
"""Real Blender dependency flags for shader drags and topology fallback.

Run stock Blender with --background --factory-startup --python this_file
-- output_directory. The generic fixture uses the factory mesh/camera/light;
it does not load SDKs, open supplied assets, save the scene or measure FPS.
"""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from ovrtx_blender_example import authoring_properties, scene_generation_sessions as sessions
from ovrtx_blender_example import scene_generation as generation

authoring_properties.register()
out = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
out.mkdir(parents=True, exist_ok=True)
scene = bpy.context.scene
obj = bpy.data.objects['Cube']
material = bpy.data.materials.new('Slider fixture')
material.use_nodes = True
obj.data.materials.clear()
obj.data.materials.append(material)
bpy.context.view_layer.update()
uid = scene.session_uid
accepted = SimpleNamespace(scene_object_count=len(scene.objects),
    blender_prim_paths={generation.blender_id(material, 'MATERIAL'): object()},
    mesh_shading_inputs=generation._mesh_shading_inputs_for_scene(scene))
sessions._owners[uid] = SimpleNamespace(current_generation=accepted, pending_generation=None)
sessions._runtimes[uid] = SimpleNamespace()
sessions._dirty.pop(uid, None)
records = []

def observe(_scene, graph):
    records.append({'shader_only': sessions.shader_only_generation_update(scene, graph),
        'updates': [{'kind': row.id.bl_rna.identifier, 'geometry':row.is_updated_geometry,
                     'transform':row.is_updated_transform, 'shading':row.is_updated_shading}
                    for row in graph.updates]})

bpy.app.handlers.depsgraph_update_post.append(observe)
try:
    socket = material.node_tree.nodes['Principled BSDF'].inputs['Roughness']
    for value in (.1, .3, .6, .9):
        socket.default_value = value
        bpy.context.view_layer.update()
    shader_records = tuple(records)
    assert len(shader_records) >= 4, shader_records
    assert all(row['shader_only'] for row in shader_records), shader_records
    records.clear()
    obj.data.vertices[0].co.x += .3
    obj.data.update()
    bpy.context.view_layer.update()
    assert records and any(not row['shader_only'] for row in records), records
    geometry_records = tuple(records)
    obj.hide_render = True
    assert not sessions._mesh_shading_unchanged(accepted, obj)
    obj.hide_render = False
    obj.data.materials[0] = bpy.data.materials.new('Replacement binding')
    assert not sessions._mesh_shading_unchanged(accepted, obj)
    (out/'blender-slider-notifications.json').write_text(json.dumps({
        'blender': bpy.app.version_string, 'shader_records':shader_records,
        'geometry_records':geometry_records, 'visibility_and_binding_fallback':True}))
    print('Material slider notification fixture passed', flush=True)
finally:
    bpy.app.handlers.depsgraph_update_post.remove(observe)
    sessions._owners.pop(uid,None)
    sessions._runtimes.pop(uid,None)
    authoring_properties.unregister()

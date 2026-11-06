"""Stock RNA/evaluated camera -> current-scene viewport request, SDK-free."""
from pathlib import Path
import json
import sys
from types import SimpleNamespace

import bpy
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'addon'))
from ovrtx_blender_example import properties, render_requests
from ovrtx_blender_example.blender_callback_adapters import BlenderRenderCallbackAdapter
from ovrtx_blender_example.blender_signal_translation import RenderRequestTranslator
from ovrtx_blender_example.blender_signals import BlenderRenderSignal, BlenderRenderSignalSource, BlenderRenderIntent
from ovrtx_blender_example.scene_generation import BlenderPrimPath, blender_id
from ovrtx_blender_example.direct_camera_updates import projection_values

out = Path(sys.argv[sys.argv.index('--') + 1])
properties.register()
scene = bpy.context.scene
scene.ovrtx_example.sync_viewport_camera = False
scene.render.resolution_x = 321
scene.render.resolution_y = 241
scene.render.resolution_percentage = 100
camera = scene.camera
camera.location = (7, 8, 9)
replacement = camera.copy()
replacement.data = camera.data.copy()
replacement.name = 'ReplacementCamera'
replacement.location = (2, 4, 6)
replacement.data.lens = 80
scene.collection.objects.link(replacement)
generation = SimpleNamespace(blender_prim_paths={blender_id(c, 'OBJECT'):
    BlenderPrimPath(c.name, 'CAMERA', '/World/'+c.name, '/World/'+c.name+'/Camera')
    for c in (camera, replacement)}, materialize_usd=lambda: str(out.with_suffix('.unused.usda')))
translator = RenderRequestTranslator(include_material_presentation=False)
adapter = BlenderRenderCallbackAdapter(generation_for_scene=lambda _:generation, translator=translator)
report = {'status': 'running', 'blender_version': bpy.app.version_string, 'cases': []}

def capture(name):
    bpy.context.view_layer.update()
    graph = bpy.context.evaluated_depsgraph_get()
    evaluated = graph.scene_eval
    request = adapter.view_draw(bpy.context, graph)
    assert request.camera_matrix is None and request.scene_camera_matrix is None
    expected = generation.blender_prim_paths[blender_id(evaluated.camera.original, 'OBJECT')].schema_path
    assert request.camera_prim_path == expected
    assert request.native_scene_camera_matrix == render_requests.scene_camera_world_matrix(evaluated)
    assert request.camera_projection is not None
    values = {r['attribute_name']: r['value'] for r in projection_values(expected, request.camera_projection)}
    report['cases'].append({'name':name,'camera_path':expected,'values':values,
                            'pose':request.native_scene_camera_matrix})
    return values

try:
    camera.data.lens = 75
    assert capture('lens')['focalLength'] == 75
    camera.data.sensor_width = 20
    assert capture('aperture')['horizontalAperture'] == 20
    camera.data.shift_x = .3
    assert abs(capture('offset')['horizontalApertureOffset'] - 6) < 1e-5
    camera.data.clip_start = 15
    assert capture('clip')['clippingRange'][0] == 15
    camera.data.dof.use_dof = True
    camera.data.dof.aperture_fstop = .5
    camera.data.dof.focus_distance = 2
    values = capture('dof')
    assert values['fStop'] == 50 and values['focusDistance'] == 2
    scene.camera = replacement
    assert capture('replacement')['focalLength'] == 80
    assert report['cases'][-1]['pose'][3][:3] == (2.,4.,6.)
    scene.camera = camera
    assert capture('restore')['focalLength'] == 75
    exact = translator.translate(BlenderRenderSignal(BlenderRenderSignalSource.VIEW_DRAW,
        BlenderRenderIntent.VIEWPORT,scene,'exact-stage.usda',camera_prim_path='/USD/Camera',
        render_product_path=properties.DEFAULT_RENDER_PRODUCT_PATH,context=bpy.context))
    assert exact.camera_projection is None and exact.native_scene_camera_matrix is None
    report['sdk_modules'] = [n for n in ('ovrtx','ovphysx','ovstage','ovstream','warp') if n in sys.modules]
    assert not report['sdk_modules']
    report['status'] = 'passed'
finally:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n')
    properties.unregister()
print('AUTHORED_CAMERA_REQUEST', json.dumps(report), flush=True)

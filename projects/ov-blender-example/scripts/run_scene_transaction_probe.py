# SPDX-License-Identifier: Apache-2.0
"""Locked external-SDK rollback with native readback and owned OVStream pixels."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import sysconfig
import traceback

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--usd-site-packages', type=Path, required=True,
                    help='Separate consumer site-packages for test-only USD fixture authoring')
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=True)
sys.path[:0] = [str(root / 'runtime'), str(root / 'addon')]
from runtime_identity import verified_identity
identity = verified_identity()
# Freeze the locked numerical package origin before adding test-only USD.
# Pillow is test-only and not in the SDK lock; textures use the add-on's stdlib PNG writer.
import numpy as np
locked_site = Path(sysconfig.get_path('purelib')).resolve()
module_origins = {}
for module in (np,):
    origin = Path(module.__file__).resolve()
    origin.relative_to(locked_site)
    module_origins[module.__name__] = dict(path=str(origin), sha256=hashlib.sha256(origin.read_bytes()).hexdigest())
# Test-only USD fixture authoring; production SDK workers do not import pxr.
sys.path.append(str(args.usd_site_packages.absolute()))
from pxr import Usd, Sdf
import pxr
Path(pxr.__file__).resolve().relative_to(args.usd_site_packages.resolve())
from ovrtx_blender_example.immutable_scene_store import ImmutableSceneStore
from ovrtx_blender_example.semantic_aov import write_rgb_png
from public_sdk_backend import RendererBackend
from direct_native_structure import inspect
from ovrtx_blender_example.ovstream_shm import HeldFrame, OvstreamShmReader
from ovrtx_blender_example.ovstream_raw_frame import RawFrameAssembler

backend = RendererBackend(out)
report = {'status': 'running', 'runtime_identity': identity, 'checks': []}
report['locked_module_origins'] = module_origins
report['test_usd_origin'] = str(Path(pxr.__file__).resolve())
report['source_sha256'] = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                           for folder in ('runtime', 'addon') for path in (root / folder).rglob('*.py')}


def pixels(name):
    result = backend.dispatch('render', {'render_var': 'LdrColor', 'samples': 32})
    held = HeldFrame(**result['shm'])
    try:
        with OvstreamShmReader(stream_name=held.stream_name, epoch=held.epoch, producer_pid=held.producer_pid) as reader:
            assembler = RawFrameAssembler(held, result['raw_frame'], result['chunk'])
            while not assembler.append(reader, held, result['raw_frame'], result['chunk']):
                result = backend.dispatch('next_raw_chunk', {'copy_token': held.copy_token})
                held = HeldFrame(**result['shm'])
            copied = assembler.finish()
        plane = next(item for item in copied.planes if item.plane.name == 'rgba8')
        value = np.frombuffer(plane.data, np.uint8).reshape(96, 128, 4).copy()
        assert value[..., :3].std() > 5
        np.save(out / (name + '.npy'), value)
        return value
    finally:
        backend.dispatch('release', {'copy_token': held.copy_token})


try:
    source = out / 'mutable-source.usda'
    texture = out / 'environment.png'
    write_rgb_png(texture, np.full((16, 32, 3), (255, 220, 180), np.uint8))
    source.write_text((root / 'tests/fixtures/public_sdk_render.usda').read_text().replace(
        'float inputs:intensity = 2000', 'float inputs:intensity = 2000\n asset inputs:texture:file = @environment.png@')
        + '\ndef Material "Material" {\n float inputs:marker = 1\n}\n')
    composition = out / 'composition.usda'
    composition.write_text('#usda 1.0\n(\n subLayers = [@mutable-source.usda@]\n)\n')
    store = ImmutableSceneStore(out / 'accepted')
    initial = store.capture(composition)
    params = dict(scene_path=initial.path,
                  sensor_paths=['/Render/Camera'], width=128, height=96, settings={})
    backend.dispatch('load', params)
    stage_id = backend.native_stage_id
    value = dict(prim_path='/World/Light', attribute_name='inputs:intensity', value_type='Float', value=500)
    backend.dispatch('update', {'values': [value]})
    backend.dispatch('material_graph_update', {'updates': [dict(material_path='/Material',
        layer_body='#usda 1.0\nover "Material" {\n float inputs:marker = 7\n}\n')]})
    backend.dispatch('instance_graph_update', {'updates': [dict(owner_id=7,
        projection_path='/World/_DirectGeometry/Object_7', hidden_source_path='/World/Cube',
        layer_body='#usda 1.0\nover "World" {\n def Xform "_DirectGeometry" {\n def Sphere "Object_7" {\n double radius = 0.35\n double3 xformOp:translate = (2,0,0)\n uniform token[] xformOpOrder = ["xformOp:translate"]\n}\n}\n}\n')]})
    before = pixels('before')
    held = pixels('held')
    noise = float(np.abs(before.astype(float) - held).mean())
    # Freeze restoration tolerance before injecting the candidate failure.
    tolerance = max(2.0, 5 * noise)
    source.write_text(source.read_text().replace('double size = 2', 'double size = 4'))
    write_rgb_png(texture, np.full((16, 32, 3), (40, 90, 255), np.uint8))
    candidate = store.capture(composition)
    rejected = backend.dispatch('update_scene', {**params, 'scene_path': candidate.path, 'settings': {'/invalid/setting': True}})
    assert rejected['scene_updated'] is False and rejected['rollback_succeeded'] is True, rejected
    assert backend.scene_config == params and backend.native_stage_id == stage_id
    state = inspect(backend, {'prim_paths': ['/World/Light'], 'attributes': ['inputs:intensity']})
    assert state['columns']['/World/Light']['inputs:intensity']['preview'] == [500.0], state
    inline = inspect(backend, {'prim_paths': ['/Material', '/World/_DirectGeometry/Object_7'],
        'attributes': ['inputs:marker', 'radius']})
    assert inline['columns']['/Material']['inputs:marker']['preview'] == [7.0], inline
    assert np.allclose(inline['columns']['/World/_DirectGeometry/Object_7']['radius']['preview'], [.35]), inline
    restored = pixels('restored')
    mae = float(np.abs(restored.astype(float) - held).mean())
    assert mae <= tolerance, (mae, tolerance)
    report['checks'].append(dict(name='rejected_population_restores_native_values_and_pixels', result=rejected,
                                native_readback=state, inline_sources=inline, pixel_mae=mae, tolerance=tolerance, same_stage=True))
    # The same external file and texture changes must also remain admissible.
    accepted = backend.dispatch('update_scene', {**params, 'scene_path': candidate.path})
    assert accepted['scene_updated'] and backend.native_stage_id == stage_id, accepted
    changed = pixels('external_changes_accepted')
    delta = float(np.abs(changed.astype(float) - restored).mean())
    assert delta > max(2, 5 * noise), (delta, noise)
    cube = inspect(backend, {'prim_paths': ['/World/Cube'], 'attributes': ['size']})
    assert cube['columns']['/World/Cube']['size']['preview'] == [4.0], cube
    report['checks'].append(dict(name='external_layer_texture_changes_accepted', result=accepted,
        pixel_mae=delta, native_readback=cube))
    params = {**params, 'scene_path': candidate.path}
    store.retain([candidate.path])
    same_source = backend.dispatch('update_scene', dict(params))
    assert same_source['scene_updated'] and backend.native_stage_id == stage_id, same_source
    same_pixels = pixels('same_source_reaccepted')
    same_delta = float(np.abs(same_pixels.astype(float) - changed).mean())
    assert same_delta <= tolerance, (same_delta, tolerance)
    report['checks'].append(dict(name='same_immutable_source_reaccepted', result=same_source, pixel_mae=same_delta))
    # A failed rollback must poison subsequent work, never appear restored.
    source.write_text(source.read_text().replace('double size = 4', 'double size = 6'))
    poison_candidate = store.capture(composition)
    original = backend.population_owner.restore
    def reject_restore(*args, **kwargs):
        raise RuntimeError('injected native restore failure')
    backend.population_owner.restore = reject_restore
    rejected = backend.dispatch('update_scene', {**params, 'scene_path': poison_candidate.path,
        'settings': {'/invalid/setting': True}})
    assert rejected['rollback_succeeded'] is False
    assert 'Unsupported public renderer setting' in rejected['error'], rejected
    refused = []
    for operation in ('render', 'update', 'update_scene', 'resize'):
        try:
            backend.dispatch(operation, {})
        except RuntimeError as error:
            assert 'reload required' in str(error)
            refused.append(operation)
        else:
            raise AssertionError(operation + ' accepted uncertain state')
    report['checks'].append(dict(name='rollback_failure_quarantines_native_owner', refused=refused, result=rejected))
    backend.population_owner.restore = original
    report['status'] = 'passed'
except Exception:
    report['status'] = 'failed'
    report['error'] = traceback.format_exc()
    raise
finally:
    backend.unload()
    report['unloaded_stage'] = backend.stage is None
    (out / 'result.json').write_text(json.dumps(report, indent=2, default=str) + '\n')

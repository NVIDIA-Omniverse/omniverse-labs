"""Bounded actual SDK consumption of immutable Windows textures and rollback.

Run only under the exclusive native lease with the locked worker interpreter.
The --prepare-only route exports/compiles USD in stock Blender without a worker.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import traceback

import bpy
import numpy as np
from pxr import Ar, Gf, Sdf, Usd, UsdGeom, UsdShade

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'addon'))
from ovrtx_blender_example.direct_sdk_client import PublicSdkRenderClient
from ovrtx_blender_example.direct_runtime_session import RuntimeWorkerError
from ovrtx_blender_example.ovrtx_runtime_client import RenderClientError
from ovrtx_blender_example.ovrtx_session import build_spec
from ovrtx_blender_example.render_requests import RenderRequest
from ovrtx_blender_example.win32_shm import existing_resources

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True, type=Path)
parser.add_argument('--payload', required=True, type=Path)
parser.add_argument('--prepare-only', action='store_true')
args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=True)
report = {'status': 'running', 'qualification': not args.prepare_only,
          'scope': 'immutable asset capture/native consumption/rollback; not viewport performance',
          'blender_version': bpy.app.version_string, 'frames': [], 'requests': []}
client = None
streams = set()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def archive(path):
    path = Path(path)
    checksum = digest(path)
    target = out / 'captured-bytes' / (checksum + path.suffix)
    target.parent.mkdir(exist_ok=True)
    if not target.exists():
        shutil.copy2(path, target)
    assert digest(target) == checksum
    return {'path':str(path), 'sha256':checksum, 'copy':str(target)}


def extended(path):
    return Path('\\\\?\\' + str(path))


def sized_path(base, name, length):
    count = length - len(str(base)) - len(name) - 2
    assert 1 <= count <= 255
    return base / ('p' * count) / name


def scene_file(name, texture):
    path = out / (name + '.usda')
    stage = Usd.Stage.CreateNew(str(path))
    world = stage.DefinePrim('/World', 'Xform')
    stage.SetDefaultPrim(world)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    mesh = UsdGeom.Mesh.Define(stage, '/World/Plane')
    mesh.CreatePointsAttr([(-10,-10,0),(10,-10,0),(10,10,0),(-10,10,0)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0,1,2,3])
    mesh.CreateSubdivisionSchemeAttr('none')
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.vertex).Set([(0,0),(1,0),(1,1),(0,1)])
    camera = UsdGeom.Camera.Define(stage, '/World/Camera')
    camera.AddTranslateOp().Set((0,0,5))
    camera.CreateClippingRangeAttr(Gf.Vec2f(.1, 100))
    material = UsdShade.Material.Define(stage, '/World/Material')
    surface = UsdShade.Shader.Define(stage, '/World/Material/Surface')
    surface.CreateIdAttr('UsdPreviewSurface')
    surface.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set((0,0,0))
    surface.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(1)
    texture_shader = UsdShade.Shader.Define(stage, '/World/Material/Texture')
    texture_shader.CreateIdAttr('UsdUVTexture')
    texture_shader.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture.as_posix()))
    texture_shader.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('raw')
    reader = UsdShade.Shader.Define(stage, '/World/Material/ST')
    reader.CreateIdAttr('UsdPrimvarReader_float2')
    reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
    texture_shader.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.CreateOutput('result', Sdf.ValueTypeNames.Float2))
    surface.CreateInput('emissiveColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(
        texture_shader.CreateOutput('rgb', Sdf.ValueTypeNames.Float3))
    material.CreateSurfaceOutput().ConnectToSource(surface.CreateOutput('surface', Sdf.ValueTypeNames.Token))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()
    return path


def inspect_owned():
    state = next(iter(client.states.values()))
    stage = Usd.Stage.Open(state.native_scene_path)
    value = stage.GetPrimAtPath('/World/Material/Texture').GetAttribute('inputs:file').Get()
    resolved = str(Ar.GetResolver().Resolve(value.path))
    asset = Path(resolved)
    assert resolved and value.resolvedPath and len(str(asset)) == 244
    assert not str(asset).startswith('\\\\?\\')
    return {'path': str(asset), 'length': len(str(asset)), 'sha256': digest(asset),
            'scene': state.native_scene_path, 'scene_sha256': digest(state.native_scene_path),
            'asset_archive': archive(asset), 'scene_archive': archive(state.native_scene_path)}


def render(name, identity, spec):
    previous_session_samples = client.states[identity].session_samples
    result = client.render_result(identity, selected_sensor_paths=spec.sensor_paths,
                                  render_var='HdrColor', additional_samples=32)
    image = np.frombuffer(result.linear_rgba16f_bytes(origin='top_left'), np.float16).reshape(97,129,4).astype(np.float32)
    reader_key = client._reader_key
    streams.add(reader_key[0])
    np.save(out / (name + '.npy'), image)
    report['frames'].append({'name': name, 'array_sha256': digest(out / (name + '.npy')),
        'samples': result.completed_samples, 'session_samples': result.session_completed_samples,
        'product_id': result.product_id, 'native_timings': result.native_timings, 'reader_key':reader_key,
        'owned': inspect_owned(), 'health': client._call('health', {})})
    (out / 'result.json').write_text(json.dumps(report, indent=2))
    assert np.isfinite(image).all() and result.product_id == spec.sensor_paths[0]
    # Every fixture capture follows load, accepted replacement or rejected
    # replacement rollback. _apply_scene_config dispatches resize, which resets
    # backend.sample_count. Logical session work is a separate client counter.
    assert result.completed_samples == 32
    assert result.session_completed_samples == previous_session_samples + 32
    return image


try:
    original = extended(args.payload).read_bytes()
    assert len(original) == 465
    assert hashlib.sha256(original).hexdigest() == 'a17a0a25c46ed4e7b5826d1b19e832da0560500d8a2c1a58a45b0fc6336165a9'
    paths = {
        'short': out / 'gray.exr',
        'long_name': sized_path(out / 'name', 'a' * 64 + '.exr', 189),
        'long_source': sized_path(out / 'source', 'color_0C0C0C.exr', 266),
        'red_candidate': sized_path(out / 'candidate', 'color_FF0000.exr', 266)}
    red = out / 'red.exr'
    image = bpy.data.images.new('Red', width=1, height=1, float_buffer=True)
    image.pixels[:] = [1,0,0,1]
    image.file_format = 'OPEN_EXR'
    image.filepath_raw = str(red)
    image.save()
    bpy.data.images.remove(image)
    for name, path in paths.items():
        extended(path.parent).mkdir(parents=True, exist_ok=True)
        extended(path).write_bytes(red.read_bytes() if name == 'red_candidate' else original)
    report['sources'] = {name: {'path': str(path), 'length': len(str(path)),
        'sha256': hashlib.sha256(extended(path).read_bytes()).hexdigest()} for name,path in paths.items()}
    os.environ['OV_BLENDER_EXAMPLE_AUTHORING_WORK_DIR'] = str(out / 'authoring')
    os.environ['OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR'] = str(out / 'composition')
    requests = {name: RenderRequest(input_usd_path=str(scene_file(name,path)),
        camera_prim_path='/World/Camera', width=129, height=97, max_samples=32,
        dlss_enabled=False, color_presentation={'render_var':'HdrColor'}) for name,path in paths.items()}
    specs = {name: build_spec(request) for name,request in requests.items()}
    report['compositions'] = {name: {'path': spec.ovrtx_scene_composition.composed_scene_path,
        'sha256': digest(spec.ovrtx_scene_composition.composed_scene_path)} for name,spec in specs.items()}
    if args.prepare_only:
        report['status'] = 'prepared'
    else:
        # Exercise the normal output-root configuration, not a substituted store.
        suffix = f'/ov-public-renderer-{os.getpid()}-1234567890/accepted-scene'
        base = sized_path(out, 'workers', 160 - len(suffix))
        os.environ['OV_BLENDER_PUBLIC_SDK_OUTPUT'] = str(base)
        client = PublicSdkRenderClient(requests['short'])
        assert len(str(client.runtime.output_root / 'accepted-scene')) == 160
        raw_call = client._call
        reject_next = False

        def observe(op, params):
            global reject_next
            wire = params
            if op == 'update_scene' and reject_next:
                reject_next = False
                # A typed stale-predecessor delta is rejected by the real worker
                # inside its rollback transaction. No backend monkeypatch.
                wire = {**params, 'population_delta': {
                    'path': params['scene_path'], 'predecessor_path': params['scene_path']}}
            row = None
            if op in {'load', 'update_scene', 'unload', 'render'}:
                row = {'operation':op, 'params':wire}
                if op in {'load', 'update_scene'}:
                    snapshot = client._scene_store().snapshots[wire['scene_path']]
                    row['scene_archive'] = archive(snapshot.path)
                    row['asset_archives'] = [archive(path) for path in sorted(snapshot.assets)]
                report['requests'].append(row)
            try:
                result = raw_call(op, wire)
            except RenderClientError as error:
                if row is not None:
                    row['exception'] = {'type':type(error).__name__, 'message':str(error),
                        'cause_type':type(error.__cause__).__name__, 'cause_message':str(error.__cause__)}
                raise
            else:
                if row is not None:
                    row['result'] = result
                return result
        client._call = observe
        reference = None
        for name in ('short', 'long_name', 'long_source'):
            client.configure_session(requests[name])
            identity = client.start_session(specs[name])
            image = render(name, identity, specs[name])
            assert image[10:-10,10:-10,:3].mean() > .01, 'Texture must actually light the emissive plane'
            if reference is None:
                reference = image
            else:
                error = float(np.abs(image-reference).mean())
                report['frames'][-1]['short_reference_mae'] = error
                assert error <= .005
            if name != 'long_source':
                client.delete_simulation(identity)
                assert not tuple(client._scene_store().root.iterdir())
        accepted = inspect_owned()
        previous_load = dict(client.runtime._load)
        extended(paths['long_source']).write_bytes(red.read_bytes())
        reject_next = True
        try:
            client.update_scene(identity, specs['red_candidate'])
        except RenderClientError as error:
            assert isinstance(error.__cause__, RuntimeWorkerError)
            assert str(error.__cause__) == ('Scene update rejected; predecessor restored: '
                                             'Population delta predecessor is stale')
        else:
            raise AssertionError('The stale-predecessor candidate must be rejected')
        rejection = report['requests'][-1]
        assert rejection['operation'] == 'update_scene' and 'exception' in rejection
        assert 'result' not in rejection
        rejection['acknowledged_load_unchanged'] = client.runtime._load == previous_load
        assert rejection['acknowledged_load_unchanged']
        rejection['health_after_rejection'] = client._call('health', {})
        assert rejection['health_after_rejection']['scene_failure'] is None
        assert client.runtime.diagnostics()['uncertain_operation'] == ''
        assert inspect_owned() == accepted
        restored = render('rollback', identity, specs['long_source'])
        # _apply_scene_config -> dispatch('resize') resets actual accumulation.
        # The native-v1 counter-only failure demonstrated32, with64 logical work.
        assert report['requests'][-1]['result']['completed_samples'] == 32
        assert report['frames'][-1]['session_samples'] == 64
        assert float(np.abs(restored-reference).mean()) <= .005
        assert len(client._scene_store().snapshots) == 1
        client.update_scene(identity, specs['red_candidate'])
        changed = render('accepted_red', identity, specs['red_candidate'])
        assert report['requests'][-1]['result']['completed_samples'] == 32
        rgb = changed[10:-10,10:-10,:3].mean(axis=(0,1))
        assert rgb[0] > .5 and max(rgb[1:]) < .05, rgb
        assert float(np.abs(changed-reference).mean()) > .05
        report['red_mean'] = rgb.tolist()
        assert inspect_owned()['sha256'] == digest(red)
        report['status'] = 'passed'
except BaseException:
    report['status'] = 'failed'
    report['traceback'] = traceback.format_exc()
    raise
finally:
    if client is not None:
        report['resources_before_shutdown'] = {name:existing_resources(name) for name in sorted(streams)}
        report['runtime_before_shutdown'] = client.runtime.diagnostics()
        report['shutdown'] = client.shutdown()
        report['runtime_after_shutdown'] = client.runtime.diagnostics()
        report['store_cleanup_errors'] = client._scene_store().cleanup_errors
        report['store_remaining_files'] = [str(p) for p in client._scene_store().root.iterdir()]
        report['resources_after_shutdown'] = {name:existing_resources(name) for name in sorted(streams)}
        if report['status'] == 'passed':
            assert any(exists for row in report['resources_before_shutdown'].values() for exists in row.values())
        assert not report['store_cleanup_errors'] and not report['store_remaining_files']
        assert not any(exists for row in report['resources_after_shutdown'].values() for exists in row.values())
    report['sdk_modules'] = sorted(name for name in sys.modules if name.split('.')[0]
        in {'ovrtx','ovphysx','ovusd','ovmaterial','ovwarp','ovstage','ovstream','warp'})
    assert not report['sdk_modules']
    (out / 'result.json').write_text(json.dumps(report, indent=2, default=str))

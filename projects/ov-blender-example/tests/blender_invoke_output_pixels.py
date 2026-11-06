# SPDX-License-Identifier: Apache-2.0
"""Actual native GUI INVOKE cold still/animation and viewport recovery acceptance."""
import argparse, dataclasses, hashlib, json, os, sys, tempfile, threading, time, traceback
from pathlib import Path
import bpy
import numpy as np
source = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--repo', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--mode', choices=('cold_still', 'cold_animation', 'viewport_still', 'exact_stage_viewport_still'), required=True)
p.add_argument('--reference', type=Path, required=True)
p.add_argument('--viewport-reference', type=Path, required=True)
a = p.parse_args(sys.argv[sys.argv.index('--') + 1:])
root = a.repo.resolve()
out = a.output.resolve()
out.relative_to(root)
out.mkdir(parents=True, exist_ok=False)
sys.path[:0] = [str(source / 'addon'), str(source / 'tests')]
import ovrtx_blender_example as addon
from ovrtx_blender_example import engine, scene_generation_sessions
from blender_render_output_fixture import create_scene, raw_pixels
reference = json.loads(a.reference.read_text())
viewport_reference = json.loads(a.viewport_reference.read_text())
assert reference['status'] == viewport_reference['status'] == 'passed'
tempfile.tempdir = str(out / 'temp')
Path(tempfile.tempdir).mkdir()
scene, cube, camera = create_scene(addon)
scene.frame_set(3 if a.mode == 'cold_animation' else 1)
scene.render.image_settings.file_format = 'OPEN_EXR'
scene.render.image_settings.color_mode = 'RGBA'
scene.render.image_settings.color_depth = '32'
scene.render.filepath = str(out / 'animation_' if a.mode == 'cold_animation' else out / 'still.exr')
os.environ['OV_BLENDER_EXAMPLE_FIXED_VIEWPORT_RESOLUTION'] = '321x241'
report = {'status': 'running', 'mode': a.mode, 'events': [], 'frames': [], 'sdk_modules': [], 'source_sha256': {str(q.relative_to(source)): hashlib.sha256(q.read_bytes()).hexdigest() for q in (source / 'addon').rglob('*.py')}}
report['shared_scene_sha256'] = hashlib.sha256((source / 'tests/blender_render_output_fixture.py').read_bytes()).hexdigest()
assert report['shared_scene_sha256'] == reference['shared_scene_sha256'] == viewport_reference['shared_scene_sha256']
report['references'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (a.reference, a.viewport_reference)}
started = time.monotonic()
completed_at = None
invoked = False
draw = None
before = None
original_job = engine._run_final_render_job
original_draw = engine.OvrtxExampleRenderEngine._draw_viewport_texture
final_job_active = False
trace_sequence = 0
final_sample_calls = []
final_mutations = []


def trace_default(value):
    if dataclasses.is_dataclass(value):
        return {field.name: getattr(value, field.name) for field in dataclasses.fields(value)}
    return str(value)


def trace(name, value):
    global trace_sequence
    trace_sequence += 1
    path = out / f'trace-{trace_sequence:04}-{name}.json'
    path.write_text(json.dumps(value, indent=2, default=trace_default) + '\n')
    return path


def snapshot_composition(label, path):
    if not path:
        return
    from pxr import Usd
    stage = Usd.Stage.Open(str(path))
    assert stage is not None, path
    destination = out / f'{label}.usda'
    stage.Flatten().Export(str(destination))
    trace('composition', {'label': label, 'source': str(path), 'flattened': str(destination),
                          'sha256': hashlib.sha256(destination.read_bytes()).hexdigest()})


# Observe existing client boundaries without changing native commands or samples.
from ovrtx_blender_example.direct_sdk_client import PublicSdkRenderClient
original_call = PublicSdkRenderClient._call
original_render_result = PublicSdkRenderClient.render_result


def observe_call(self, op, params):
    if op in ('load', 'update_scene'):
        snapshot_composition(f'population-{trace_sequence:04}', params.get('scene_path'))
    observed = op in ('load', 'update_scene', 'update', 'retarget_camera', 'resize')
    if observed and final_job_active:
        final_mutations.append({'op': op, 'thread': threading.get_native_id()})
    if observed:
        trace('native-request', {'op': op, 'final_job_active': final_job_active, 'params': params})
    result = original_call(self, op, params)
    if observed:
        trace('native-response', {'op': op, 'result': result})
    return result


def observe_render_result(self, simulation_id, **kwargs):
    result = original_render_result(self, simulation_id, **kwargs)
    if final_job_active:
        final_sample_calls.append({'requested': kwargs['additional_samples'], 'completed': result.completed_samples, 'thread': threading.get_native_id()})
        trace('final-acquired-samples', {'requested': kwargs, 'completed_samples': result.completed_samples,
              'session_completed_samples': result.session_completed_samples,
              'simulation_time_ns': result.simulation_time_ns, 'runtime': result.native_timings})
    return result
PublicSdkRenderClient._call = observe_call
PublicSdkRenderClient.render_result = observe_render_result

def event(name, **kw):
    report['events'].append(dict(event=name, native_thread=threading.get_native_id(), main=threading.current_thread() is threading.main_thread(), at=time.monotonic(), **kw))

def persist():
    (out / 'result.json').write_text(json.dumps(report, indent=2, default=str) + '\n')

def array(result):
    return np.frombuffer(raw_pixels(result), '<f2' if result.linear_rgba16f else np.uint8).reshape(result.height, result.width, 4).astype(np.float32) / (1 if result.linear_rgba16f else 255)

def observe_job(request, *args, **kwargs):
    global final_job_active
    trace('final-request', request)
    controller = kwargs.get('controller')
    if controller is not None:
        spec = controller._spec
        snapshot_composition('final-composed', spec.ovrtx_scene_composition.composed_scene_path)
        client = controller._client
        state = client.states[controller._simulation_id]
        trace('final-state', {'spec': spec, 'desired': list(state.desired.values()),
                              'samples': state.samples, 'session_samples': state.session_samples,
                              'time_ns': state.time_ns, 'request': client.request})
    final_sample_calls.clear()
    final_mutations.clear()
    final_job_active = True
    final_thread = threading.get_native_id()
    try:
        outcome = original_job(request, *args, **kwargs)
    finally:
        final_job_active = False
    report['final_native_trace'] = {'sample_calls': list(final_sample_calls), 'mutations': list(final_mutations), 'job_thread': final_thread}
    persist()
    if a.mode == 'viewport_still' and controller is not None:
        assert not final_mutations, final_mutations
        assert all(row['thread'] == final_thread for row in final_sample_calls), final_sample_calls
        assert [row['requested'] for row in final_sample_calls] == [1, 1, 2, 4, 8, 16], final_sample_calls
        assert [row['completed'] for row in final_sample_calls] == [1, 2, 4, 8, 16, 32], final_sample_calls
    result = outcome.get('result')
    if result is not None:
        pixels = array(result)
        path = out / f'raw-{request.timeline_frame}.npy'
        np.save(path, pixels)
        assert result.product_id == request.render_product_path
        assert result.render_var == 'HdrColor' and result.completed_samples == request.max_samples == 32
        assert np.isfinite(pixels).all() and pixels[..., :3].max() > 1
        report['frames'].append(dict(frame=request.timeline_frame, product_id=result.product_id, raw_path=str(path), samples=result.completed_samples, runtime=dict(result.native_timings)))
        persist()
    return outcome
engine._run_final_render_job = observe_job

def observe_draw(self, context, texture, result, scene=None):
    global draw
    original_draw(self, context, texture, result, scene)
    if context.area == area:
        draw = (self, result, texture, time.monotonic(), self._presented_frame)
engine.OvrtxExampleRenderEngine._draw_viewport_texture = observe_draw
def prepare_exact_stage(scene, output):
    # Export the same frozen authored scene once, then exercise the production
    # exact-stage adapter and hosted final lane with this immutable USD input.
    from pxr import Ar, Usd, UsdUtils
    import shutil
    from ovrtx_blender_example.ovrtx_session import build_spec
    request = engine._render_callback_adapter('exact-fixture-export').final_render_from_scene(scene)
    spec = build_spec(request)
    stage = Usd.Stage.Open(spec.ovrtx_scene_composition.composed_scene_path)
    assert stage is not None
    path = output / 'exact-input.usda'
    layer = stage.Flatten()
    assets = output / 'exact-assets'
    assets.mkdir(exist_ok=True)
    manifest = {}
    def retain_asset(asset):
        if not asset:
            return asset
        resolved = Path(str(Ar.GetResolver().Resolve(asset)))
        assert resolved.is_file(), ('Unresolved exact fixture asset', asset)
        digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
        # Keep the original short basename: the immutable native store places
        # it below its own hash/UUID directories on stock Windows Blender.
        target = assets / digest / resolved.name
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            shutil.copy2(resolved, target)
        assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
        manifest[str(target)] = {'source': str(resolved), 'sha256': digest}
        return str(target)
    with Ar.ResolverContextBinder(stage.GetPathResolverContext()):
        UsdUtils.ModifyAssetPaths(layer, retain_asset)
    layer.Export(str(path))
    scene_generation_sessions.close()
    # Reopen only the standalone export after its temporary owner is gone.
    check = Usd.Stage.Open(str(path))
    assert check is not None
    verified = set()
    def verify_asset(asset):
        if not asset:
            return asset
        resolved = Path(str(Ar.GetResolver().Resolve(asset))).resolve()
        resolved.relative_to(output.resolve())
        assert resolved.is_file(), ('Exact fixture lost asset after close', asset)
        assert hashlib.sha256(resolved.read_bytes()).hexdigest() == manifest[str(resolved)]['sha256']
        verified.add(str(resolved))
        return asset
    with Ar.ResolverContextBinder(check.GetPathResolverContext()):
        UsdUtils.ModifyAssetPaths(check.Flatten(), verify_asset)
    assert verified == set(manifest)
    engine.configure_exact_stage(input_usd_path=str(path), camera_prim_path=request.camera_prim_path,
                                 render_product_path=request.render_product_path)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'camera': request.camera_prim_path, 'product': request.render_product_path,
            'assets': manifest, 'assets_verified_after_owner_close': True}


if a.mode == 'exact_stage_viewport_still':
    report['exact_stage'] = prepare_exact_stage(scene, out)
    persist()
area = next((q for q in bpy.context.screen.areas if q.type == 'VIEW_3D'))
if a.mode in ('viewport_still', 'exact_stage_viewport_still'):
    area.spaces.active.region_3d.view_perspective = 'CAMERA'
    area.spaces.active.overlay.show_overlays = False
    area.spaces.active.shading.type = 'RENDERED'

def complete(scene, *args):
    global completed_at
    completed_at = time.monotonic()
    event('render_complete')
    persist()

def canceled(scene, *args):
    report['status'] = 'failed'
    report['error'] = 'Unexpected actual native GUI render cancellation'
    event('render_cancel')
    persist()
bpy.app.handlers.render_complete.append(complete)
bpy.app.handlers.render_cancel.append(canceled)

def finish(error=None):
    report['sdk_modules'] = [name for name in ('ovrtx', 'ovphysx', 'ovstage', 'ovstream', 'warp') if name in sys.modules]
    report['status'] = 'failed' if error or report['sdk_modules'] else 'passed'
    if error:
        report['error'] = str(error)
    report['elapsed_seconds'] = time.monotonic() - started
    persist()
    bpy.ops.wm.quit_blender()
    return None

def validate_saved_exr(frames, images):
    """Check actual Blender publication, including retained opaque alpha policy."""
    if report.get('saved_exr_validated'):
        return
    for row, raw in zip(frames, images):
        path = out / (f"animation_{row['frame']:04}.exr" if a.mode == 'cold_animation' else 'still.exr')
        image = bpy.data.images.load(str(path), check_existing=False)
        try:
            assert tuple(image.size) == (raw.shape[1], raw.shape[0])
            saved = np.empty(raw.size, np.float32)
            image.pixels.foreach_get(saved)
            saved = saved.reshape(raw.shape)
            saved_path = out / f"saved-{row['frame']}.npy"
            np.save(saved_path, saved)
            assert np.array_equal(saved[..., :3], raw[..., :3]), 'GUI EXR changed scene-linear RGB'
            assert np.all(saved[..., 3] == 1), 'GUI EXR changed retained opaque presentation alpha'
            row['saved_exr'] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                'decoded_path': str(saved_path), 'rgb_exact': True, 'alpha_one': True}
        finally:
            bpy.data.images.remove(image)
    report['saved_exr_validated'] = True
    persist()


def tick():
    global invoked, before
    try:
        if report['status'] == 'failed':
            return finish(report['error'])
        if time.monotonic() - started > 420:
            raise TimeoutError('Native GUI lifecycle watchdog')
        if not invoked:
            if a.mode in ('viewport_still', 'exact_stage_viewport_still'):
                if draw is None or draw[1].completed_samples < 32:
                    return 0.1
                if draw[4] is None or draw[4].snapshot_key != draw[0]._written_snapshot_key:
                    return 0.1
                before = draw
                trace('viewport-request', draw[0]._viewport_request)
                np.save(out / 'viewport-before.npy', array(draw[1]))
                report['viewport_before'] = {'engine': id(draw[0]), 'texture': id(draw[2]), 'runtime': dict(draw[1].native_timings), 'publication': draw[4].publication_index}
            else:
                scene_generation_sessions.close()
                assert int(scene.session_uid) not in scene_generation_sessions._owners
                report['cold_no_generation'] = True
            event('invoke', frame=scene.frame_current)
            status = bpy.ops.render.render('INVOKE_DEFAULT', animation=a.mode == 'cold_animation', write_still=a.mode != 'cold_animation')
            assert status == {'RUNNING_MODAL'}, status
            invoked = True
            persist()
            return 0.1
        if completed_at is None:
            return 0.1
        frames = report['frames']
        assert [q['frame'] for q in frames] == ([1, 2, 3] if a.mode == 'cold_animation' else [1])
        images = [np.load(q['raw_path']) for q in frames]
        validate_saved_exr(frames, images)
        if a.mode == 'cold_animation':
            changes = [float(np.abs(x - y).mean()) for x, y in zip(images, images[1:])]
            assert min(changes) > 0.005, changes
            assert all(((out / f'animation_{i:04}.exr').is_file() for i in (1, 2, 3)))
            report['animation_pixel_changes'] = changes
        else:
            assert (out / 'still.exr').is_file()
            baseline = np.load(next((q for q in reference['cases'] if q['name'] == 'still_hdr'))['raw_path'])
            error = float(np.abs(images[0] - baseline).mean())
            report['frozen_hdr_comparison'] = {'mae': error, 'limit': 0.005}
            persist()
            assert error <= 0.005, error
        if a.mode in ('viewport_still', 'exact_stage_viewport_still'):
            area.tag_redraw()
            if draw is None or draw[3] <= completed_at or draw[1].completed_samples < 32:
                return 0.1
            current = draw[4]
            if current is None or current is before[4] or current.snapshot_key != draw[0]._written_snapshot_key:
                return 0.1
            error = float(np.abs(array(draw[1]) - array(before[1])).mean())
            limit = viewport_reference['thresholds']['HdrColor']
            assert error <= limit, (error, limit)
            assert draw[1].product_id == draw[0]._viewport_request.render_product_path
            np.save(out / 'viewport-after.npy', array(draw[1]))
            report['viewport_after'] = {'engine': id(draw[0]), 'texture': id(draw[2]), 'runtime': dict(draw[1].native_timings), 'publication': current.publication_index, 'fresh_frame_object': True, 'pixel_mae': error, 'limit': limit}
        return finish()
    except BaseException as exc:
        report['traceback'] = traceback.format_exc()
        return finish(exc)
bpy.app.timers.register(tick, first_interval=0.5)

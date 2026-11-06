# SPDX-License-Identifier: Apache-2.0
"""Actual Blender F12/animation output, lossless color and cancel/resume checks.

Run in stock Blender with locked external SDK and an isolated user profile.
The cancellation case injects the job cancellation predicate after a native
sample, ORed with the actual Blender test_break callback. This is an injected
signal, not physical Esc input. Rendering, teardown and resume use production.
"""
import argparse
import ast
import hashlib
import json
import os
import shutil
from pathlib import Path
import sys
import traceback
import threading
import time

import bpy
import numpy as np

source = Path(__file__).resolve().parents[1]
addon_root=Path(os.environ.get('OV_OUTPUT_ADDON_ROOT',str(source/'addon')))
sys.path[:0] = [str(addon_root), str(source/'tests')]
import ovrtx_blender_example as addon
from ovrtx_blender_example import engine, properties
from blender_render_output_fixture import create_scene, raw_pixels

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--calibrate-baseline',action='store_true')
parser.add_argument('--reference',type=Path)
parser.add_argument('--cancel-resume-only',action='store_true')
args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=True)
report = {'status': 'running', 'cases': [], 'blender_version': bpy.app.version_string,
          'baseline_calibration':args.calibrate_baseline,'addon_root':str(addon_root),
          'fixture_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'source_sha256': {str(p.relative_to(addon_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in addon_root.rglob('*.py')}}
scene, cube, camera = create_scene(addon)
report['shared_scene_sha256']=hashlib.sha256((source/'tests/blender_render_output_fixture.py').read_bytes()).hexdigest()
reference=None
if not args.calibrate_baseline:
    if args.reference is None:raise ValueError('Candidate qualification requires a frozen matched baseline reference')
    reference=json.loads(args.reference.read_text())
    assert reference['status']=='passed' and reference['baseline_calibration']
    assert reference['shared_scene_sha256']==report['shared_scene_sha256']
    report['baseline_reference']={'path':str(args.reference),'sha256':hashlib.sha256(args.reference.read_bytes()).hexdigest()}
written = []
native_samples = []
original_write = engine.OvrtxExampleRenderEngine._write_blender_result
original_render = engine.OvrtxSessionController.render
original_break = engine.OvrtxExampleRenderEngine.__dict__.get('test_break')
original_profiled = engine.OvrtxExampleRenderEngine._render_profiled
original_job = engine._run_final_render_job
cancel_armed = False


def invoke_expected_cancellation(render):
    """Blender's ERROR report propagates its no-output cancellation flag."""
    try:
        return render(), None
    except RuntimeError as error:
        if str(error) not in ('Error: OVRTX render cancelled', 'Error: OVRTX render cancelled\n'):
            raise
        return None, str(error)


def cancel_event(name, **values):
    if cancel_armed:
        with (out/'cancel-events.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(event=name, time=time.time(),
                thread=threading.get_ident(), native_calls=len(native_samples),
                writes=len(written), **values), default=str)+'\n')


def observe_profiled(self, depsgraph):
    bound = self.test_break
    cancel_event('render_enter', bound_repr=repr(bound),
        bound_function=getattr(getattr(bound, '__func__', None), '__name__', None),
        frame=depsgraph.scene.frame_current)
    try:
        return original_profiled(self, depsgraph)
    finally:
        cancel_event('render_exit')


def observe_job(request, *args, **kwargs):
    if cancel_armed:
        real_cancel = kwargs.get('cancel_requested')
        def injected_cancel():
            actual = bool(real_cancel()) if real_cancel is not None else False
            injected = len(native_samples) > sample_count
            cancel_event('cancel_predicate', actual_blender=actual,
                injected_after_sample=injected, value=actual or injected)
            return actual or injected
        # RNA's builtin self.test_break cannot be replaced by a class-level
        # Python assignment. Keep that real callback and inject at the existing
        # job predicate boundary, which production checks between native chunks.
        kwargs['cancel_requested'] = injected_cancel
    cancel_event('job_enter', max_samples=request.max_samples,
        prepared=kwargs.get('controller') is not None,
        cancel_callable=repr(kwargs.get('cancel_requested')))
    result = original_job(request, *args, **kwargs)
    cancel_event('job_exit', status=result.get('status'))
    return result


engine.OvrtxExampleRenderEngine._render_profiled = observe_profiled
engine._run_final_render_job = observe_job


def restore_break():
    if original_break is None:
        if 'test_break' in engine.OvrtxExampleRenderEngine.__dict__:
            delattr(engine.OvrtxExampleRenderEngine,'test_break')
    else:
        engine.OvrtxExampleRenderEngine.test_break=original_break


def observe_native(self, *args, **kwargs):
    result = original_render(self, *args, **kwargs)
    native_samples.append(result.completed_samples)
    return result


def observe_write(self, result):
    raw = raw_pixels(result)
    dtype = '<f2' if result.linear_rgba16f else 'uint8'
    pixels = np.frombuffer(raw, dtype).reshape(result.height,result.width,4).copy()
    written.append((result,pixels))
    # Never abort the engine callback before Blender receives its RenderResult:
    # Blender's EXR writer assumes a completed result after the render callback.
    original_write(self,result)


def validate_result(result,pixels,*,minimum_blue_pixels=50):
    if not args.calibrate_baseline:
        assert result.product_id == properties.DEFAULT_RENDER_PRODUCT_PATH
    assert result.render_var == 'HdrColor', 'Blender final-render contract must stay scene-linear HDR'
    assert np.isfinite(pixels).all() and pixels[..., :3].astype(np.float32).std() > .01
    upper=1 if result.linear_rgba16f else 255
    assert pixels[...,3].min()>=0 and pixels[...,3].max()<=upper, ('Alpha outside normalized range',float(pixels[...,3].min()),float(pixels[...,3].max()))
    assert pixels[...,3].max()>0
    red=pixels[...,0].astype(float);blue=pixels[...,2].astype(float)
    assert np.count_nonzero(red>blue*2+.01)>50, 'Orange subject missing or channel order changed'
    assert np.count_nonzero(blue>red*2+.01)>minimum_blue_pixels, 'Blue reference missing or channel order changed'


def logged_native(self, *args, **kwargs):
    cancel_event('native_enter', additional_samples=kwargs.get('additional_samples'))
    result = observe_native(self, *args, **kwargs)
    cancel_event('native_exit', completed_samples=result.completed_samples)
    return result


def logged_write(self, result):
    cancel_event('write_enter', completed_samples=result.completed_samples)
    return observe_write(self, result)


engine.OvrtxExampleRenderEngine._write_blender_result = logged_write
engine.OvrtxSessionController.render = logged_native


def save_report():
    (out/'result.json').write_text(json.dumps(report,indent=2,default=str)+'\n')


def capture(name, *, mode='scene_linear_hdr', frame=1):
    scene.ovrtx_example.color_presentation_mode = mode
    scene.frame_set(frame)
    scene.render.image_settings.file_format = 'OPEN_EXR' if mode == 'scene_linear_hdr' else 'PNG'
    scene.render.image_settings.color_depth = '16' if mode == 'scene_linear_hdr' else '8'
    path = out/(name + ('.exr' if mode == 'scene_linear_hdr' else '.png'))
    scene.render.filepath = str(path)
    before = len(written)
    native_before=len(native_samples)
    bpy.ops.render.render(write_still=True)
    assert len(written) == before+1, (name,'missing result')
    result,pixels = written[-1]
    assert len(native_samples)>native_before,'Final capture reused a held result without actual native render calls'
    assert (result.width,result.height)==(321,241)
    assert result.completed_samples == scene.ovrtx_example.max_samples
    np.save(out/(name+'.npy'),pixels)
    validate_result(result,pixels)
    image = bpy.data.images.load(str(path),check_existing=False)
    saved = np.empty(result.width*result.height*4,np.float32)
    image.pixels.foreach_get(saved)
    saved = saved.reshape(result.height,result.width,4)
    assert np.isfinite(saved).all() and saved[...,:3].std()>.01
    np.save(out/(name+'-saved.npy'),saved)
    worker_log=os.environ.get('OV_BLENDER_EXAMPLE_WORKER_LOG')
    if worker_log and Path(worker_log).is_file():shutil.copy2(worker_log,out/(name+'-worker.log'))
    if mode == 'scene_linear_hdr':
        assert pixels[...,:3].max()>1, 'HDR clipped before Blender'
        assert np.array_equal(saved[...,:3],pixels[...,:3].astype(np.float32)), 'EXR changed scene-linear RGB pixels'
        assert np.all(saved[...,3]==1), 'EXR changed baseline opaque presentation alpha'
    bpy.data.images.remove(image)
    report['cases'].append({'name':name,'frame':frame,'product_id':getattr(result,'product_id',properties.DEFAULT_RENDER_PRODUCT_PATH),
        'render_var':result.render_var,'size':[result.width,result.height],
        'samples':result.completed_samples,'min':float(pixels.min()),'max':float(pixels.max()),
        'alpha_min':float(pixels[...,3].min()),'alpha_max':float(pixels[...,3].max()),
        'alpha_zero_pixels':int(np.count_nonzero(pixels[...,3]==0)),
        'saved_alpha_min':float(saved[...,3].min()),'saved_alpha_max':float(saved[...,3].max()),
        'hdr_alpha_policy':'raw half alpha preserved; baseline Blender presentation alpha=1',
        'runtime':dict(result.native_timings),'raw_sha256':hashlib.sha256(pixels.tobytes()).hexdigest(),
        'saved_path':str(path),'raw_path':str(out/(name+'.npy')),
        'saved_pixels_path':str(out/(name+'-saved.npy')),'native_render_batches':len(native_samples)-native_before,
        'requested_viewport_color_mode':mode,'output_format':scene.render.image_settings.file_format,
        'camera':camera.name,'object_world_x':cube.matrix_world.translation.x})
    save_report()
    return pixels.astype(np.float32)


def verify_frozen_baseline_capture():
    """Permit only the reviewed candidate-animation correction against v6."""
    if reference is None or reference['fixture_sha256']==report['fixture_sha256']:
        return
    assert reference['fixture_sha256']=='80da8948e5abcd10a05c9729f6eb42c559268ba3d20bd425055114006a5c7b90'
    tree=ast.parse(Path(__file__).read_text())
    names={'restore_break','observe_native','observe_write','validate_result','save_report','capture'}
    nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    validation=next(node for node in nodes if node.name=='validate_result')
    assert [arg.arg for arg in validation.args.kwonlyargs]==['minimum_blue_pixels']
    assert [ast.literal_eval(value) for value in validation.args.kw_defaults]==[50]
    validation.args.kwonlyargs=[];validation.args.kw_defaults=[]
    class FreezeStillDefault(ast.NodeTransformer):
        def visit_Name(self,node):
            return ast.Constant(value=50) if node.id=='minimum_blue_pixels' else node
    FreezeStillDefault().visit(validation)
    run=next(node for node in tree.body if isinstance(node,ast.Try))
    body=run.body
    if isinstance(body[0],ast.If) and ast.unparse(body[0].test)=='args.cancel_resume_only':
        body=body[0].orelse  # full-mode capture/comparison statements stay exact
    end=next(index for index,node in enumerate(body)
             if isinstance(node,ast.If) and ast.unparse(node.test)=='args.calibrate_baseline')
    nodes+=body[:end+2]  # captures, frozen baseline calibration and comparisons
    digest=hashlib.sha256(ast.dump(ast.Module(body=nodes,type_ignores=[]),include_attributes=False).encode()).hexdigest()
    assert digest=='86cd0dbaa91d11520edb2484eaeeedb06748c21e8fe0c30e8f25f6b7ebbb0e71'
    report['baseline_fixture_compatibility']={'original_fixture_sha256':reference['fixture_sha256'],
        'baseline_capture_ast_sha256':digest,
        'scope':'Still capture, scene hash, settings, calibration and comparisons unchanged; candidate animation visibility assertion corrected for authored frame3 occlusion'}


verify_frozen_baseline_capture()
try:
    if args.cancel_resume_only:
        if args.calibrate_baseline:
            raise ValueError('Cancellation-only mode requires the frozen baseline')
        first = capture('still_hdr')
        threshold = reference['thresholds']['HdrColor']
        baseline_case=next(c for c in reference['cases'] if c['name']=='still_hdr')
        baseline=np.load(baseline_case['raw_path']).astype(np.float32)
        error=float(np.abs(first-baseline).mean())
        assert error<=threshold,(error,threshold)
        report['baseline_comparisons']=[{'mode':'HdrColor','mae':error,'limit':threshold}]
        report['frozen_change_threshold']=threshold
        report['scope']='Selected cancellation/resume; full still/animation cells reported separately'
        scene.frame_set(3)  # original full-mode animation leaves this same authored frame
    else:
        first = capture('still_hdr')
        held = capture('still_hdr_repeat')
        noise = float(np.abs(first-held).mean())
        threshold = max(.005,noise*3) if args.calibrate_baseline else reference['thresholds']['HdrColor']
        png = capture('still_png',mode='ldr_rgba8_display_passthrough')
        png_held=capture('still_png_repeat',mode='ldr_rgba8_display_passthrough')
        saved_png=np.load(out/'still_png-saved.npy')
        saved_png_repeat=np.load(out/'still_png_repeat-saved.npy')
        if args.calibrate_baseline:
            report['thresholds']={'HdrColor':threshold,'SavedPNG':max(.005,float(np.abs(saved_png-saved_png_repeat).mean())*3)}
            report['threshold_rationale']='3x actual baseline rerender MAE, minimum0.005; final native frames are HdrColor for both EXR and PNG; SavedPNG is decoded float RGBA, frozen before candidate'
            report['status']='passed'
            save_report()
            bpy.ops.wm.quit_blender()
            raise SystemExit(0)
        for mode,pixels,case_name in (('HdrColor',first,'still_hdr'),('SavedPNG',saved_png,'still_png')):
            baseline_case=next(c for c in reference['cases'] if c['name']==case_name)
            baseline=np.load(baseline_case['saved_pixels_path' if mode=='SavedPNG' else 'raw_path']).astype(np.float32)
            error=float(np.abs(pixels-baseline).mean())
            report.setdefault('baseline_comparisons',[]).append({'mode':mode,'mae':error,'limit':reference['thresholds'][mode]})
            assert error<=reference['thresholds'][mode],(mode,error,reference['thresholds'][mode])
        report['frozen_change_threshold']=threshold
        # Exercise Blender's actual animation operator and output order.
        scene.ovrtx_example.color_presentation_mode = 'scene_linear_hdr'
        scene.render.image_settings.file_format = 'OPEN_EXR'
        scene.render.image_settings.color_depth = '16'
        scene.render.filepath = str(out/'animation_')
        count = len(written)
        bpy.ops.render.render(animation=True)
        frames = written[count:]
        assert len(frames)==3
        for index,(result,pixels) in enumerate(frames,1):
            np.save(out/f'animation_{index}.npy',pixels)
            # At x2 the orange cube covers nearly all the blue sphere: the exact
            # shared-scene camera ray cast leaves only38 blue primary pixels.
            # Preserve the nonoccluded channel-order check on frames1/2; frame3
            # must retain a blue sliver and show its expected occlusion.
            validate_result(result,pixels,minimum_blue_pixels=0 if index==3 else 50)
            assert (out/f'animation_{index:04}.exr').is_file()
            if index==3:
                def blue_count(image):
                    return int(np.count_nonzero(image[...,2].astype(float)>image[...,0].astype(float)*2+.01))
                assert blue_count(pixels)<blue_count(frames[0][1])/2
                report['frame3_occlusion']={'blue_pixels':blue_count(pixels),'frame1_blue_pixels':blue_count(frames[0][1]),
                    'expected':'nonzero blue sliver, less than half its unoccluded area'}
        deltas=[float(np.abs(frames[i][1].astype(float)-frames[i-1][1]).mean()) for i in (1,2)]
        assert all(delta>threshold for delta in deltas),(deltas,threshold)
        report['cases'].append({'name':'blender_animation','frames':[1,2,3],'pixel_changes':deltas})
    # Inject at the existing job predicate, retaining Blender's real callback.
    before=len(written)
    sample_count=len(native_samples)
    scene.ovrtx_example.max_samples=4096
    cancel_armed=True
    cancel_event('operator_enter', sample_count=sample_count, written_before=before)
    operator_result,operator_exception=invoke_expected_cancellation(bpy.ops.render.render)
    cancel_event('operator_exit', operator_result=sorted(operator_result) if operator_result is not None else None,
                 operator_exception=operator_exception)
    report['cancel_diagnostic']={'sample_count_before':sample_count,
        'sample_count_after':len(native_samples),'writes_before':before,
        'writes_after':len(written),'operator_result':sorted(operator_result) if operator_result is not None else None,
        'operator_exception':operator_exception}
    save_report()
    assert len(native_samples)>sample_count and len(written)==before
    report['cases'].append({'name':'cancel','signal':'Injected job predicate after native sample OR real Blender test_break; no physical Esc input',
        'native_batches':len(native_samples)-sample_count,'published_results':0})
    restore_break()
    cancel_armed=False
    scene.ovrtx_example.max_samples=32
    resumed=capture('resume_hdr',frame=2)
    assert float(np.abs(resumed-first).mean())>threshold
    report['sdk_imports_in_blender']=[name for name in ('ovrtx','ovphysx','ovstage','ovstream','warp') if name in sys.modules]
    assert not report['sdk_imports_in_blender']
    report['status']='passed'
except Exception:
    report['status']='failed'
    report['error']=traceback.format_exc()
finally:
    engine.OvrtxExampleRenderEngine._write_blender_result=original_write
    restore_break()
    engine.OvrtxSessionController.render=original_render
    engine.OvrtxExampleRenderEngine._render_profiled=original_profiled
    engine._run_final_render_job=original_job
    save_report()
    print('FINAL_OUTPUT_RESULT',json.dumps(report,default=str),flush=True)
    bpy.ops.wm.quit_blender()

# SPDX-License-Identifier: Apache-2.0
"""Real Windows viewport presentation/lifecycle acceptance with owned screenshots.

Instrumentation observes the completed production draw only. Camera/settings
changes go through Blender RNA/operators; there is no render-refresh injection.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import bpy
import numpy as np
from mathutils import Quaternion

source=Path(__file__).resolve().parents[1]
addon_root=Path(os.environ.get('OV_OUTPUT_ADDON_ROOT',str(source/'addon')))
sys.path[:0]=[str(addon_root),str(source/'tests')]
import ovrtx_blender_example as addon
from ovrtx_blender_example import engine, ovrtx_scene_composition, properties
from blender_render_output_fixture import create_scene, raw_pixels
from viewport_output_baseline import (verify_fixture_compatibility, load_dlss_reference,
    compare_dlss_policy, dlss_frame_ready)

parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--calibrate-baseline',action='store_true')
parser.add_argument('--reference',type=Path)
parser.add_argument('--dlss-reference',type=Path)
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
os.environ['OV_BLENDER_EXAMPLE_FIXED_VIEWPORT_RESOLUTION']='321x241'
scene,cube,camera=create_scene(addon)
report={'status':'running','cases':[],'blender_version':bpy.app.version_string,
        'baseline_calibration':args.calibrate_baseline,'addon_root':str(addon_root),
        'fixture_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'actual stock Blender view_update/view_draw and screenshot, no performance claim'}
report['shared_scene_sha256']=hashlib.sha256((source/'tests/blender_render_output_fixture.py').read_bytes()).hexdigest()
reference=None
dlss_reference=None
if not args.calibrate_baseline:
    if args.reference is None:raise ValueError('Candidate qualification requires a frozen matched baseline reference')
    reference=json.loads(args.reference.read_text())
    assert reference['status']=='passed' and reference['baseline_calibration']
    report['baseline_fixture_compatibility']=verify_fixture_compatibility(__file__,reference['fixture_sha256'])
    assert reference['shared_scene_sha256']==report['shared_scene_sha256']
    report['baseline_reference']={'path':str(args.reference),'sha256':hashlib.sha256(args.reference.read_bytes()).hexdigest()}
    if args.dlss_reference is None:raise ValueError('Viewport qualification requires pinned same-policy DLSS baseline evidence')
    dlss_reference=load_dlss_reference(args.dlss_reference,report['shared_scene_sha256'])
area=next(a for a in bpy.context.screen.areas if a.type=='VIEW_3D')
secondary=None
draws={}
accepted_loads=[]
accepted_updates=[]
original_call=None
if not args.calibrate_baseline:
    from ovrtx_blender_example.direct_sdk_client import PublicSdkRenderClient
    original_call=PublicSdkRenderClient._call
    def observe_call(self,op,params):
        reply=original_call(self,op,params)
        if op=='load':
            accepted_loads.append({'settings':dict(params.get('settings',{})),
                'sensor_paths':list(params['sensor_paths']),'render_var':params['render_var'],
                'scene_path':params['scene_path'],'accepted_at':time.monotonic()})
        if op=='update':
            assert reply['updated_count']==len(params['values'])
            accepted_updates.append({'values':list(params['values']),'accepted_at':time.monotonic()})
        return reply
    PublicSdkRenderClient._call=observe_call
original_draw=engine.OvrtxExampleRenderEngine._draw_viewport_texture


def observed_draw(self,context,texture,result,scene=None):
    original_draw(self,context,texture,result,scene)
    draws[context.area.as_pointer()]=(self,result,texture,time.monotonic())


engine.OvrtxExampleRenderEngine._draw_viewport_texture=observed_draw


def enable_view(target,perspective='CAMERA'):
    target.type='VIEW_3D'
    target.spaces.active.region_3d.view_perspective=perspective
    target.spaces.active.overlay.show_overlays=False
    target.spaces.active.shading.type='RENDERED'
    target.tag_redraw()


enable_view(area)
stage=0
armed=False
since=time.monotonic()
pixels_by_name={}
screens_by_name={}
noise={'HdrColor':.005,'LdrColor':.005}
if reference:noise.update(reference['thresholds'])
expected_size=(321,241)
pending_capture=None
camera_defaults={key:getattr(camera.data,key) for key in ('lens','sensor_width','shift_x','clip_start','clip_end')}
quality_defaults={key:getattr(scene.ovrtx_example,key) for key in properties.RTPT_RENDER_SETTINGS}
alternate_camera=None


def action(name):
    global secondary,expected_size,alternate_camera
    if name=='hdr_exposure':scene.view_settings.exposure=2
    elif name=='hdr_gamma':scene.view_settings.gamma=2
    elif name=='hdr_gamma_reset':scene.view_settings.gamma=1
    elif name=='hdr_view_transform':scene.view_settings.view_transform='AgX'
    elif name=='hdr_view_transform_reset':scene.view_settings.view_transform='Standard'
    elif name in {'hdr_staging_on','hdr_staging_off'}:
        os.environ['OVRTX_HDR_GPU_FLOAT32']='1' if name=='hdr_staging_on' else '0'
        scene.ovrtx_example.max_samples+=1
    elif name in {'hdr_exposure_reset','ldr_exposure_reset'}:scene.view_settings.exposure=0
    elif name=='ldr':scene.ovrtx_example.color_presentation_mode='ldr_rgba8_display_passthrough'
    elif name=='ldr_exposure':scene.view_settings.exposure=2
    elif name=='resize':
        expected_size=(513,257)
        scene.render.resolution_x,scene.render.resolution_y=expected_size
        os.environ['OV_BLENDER_EXAMPLE_FIXED_VIEWPORT_RESOLUTION']='513x257'
    elif name=='resize_reset':
        expected_size=(321,241)
        scene.render.resolution_x,scene.render.resolution_y=expected_size
        os.environ['OV_BLENDER_EXAMPLE_FIXED_VIEWPORT_RESOLUTION']='321x241'
    elif name=='dlss_default':scene.ovrtx_example.dlss_enabled=True
    elif name=='dlss_performance':scene.ovrtx_example.dlss_enabled=False
    elif name=='perspective':
        region=area.spaces.active.region_3d
        region.view_perspective='PERSP'
        region.view_location=(0,0,1)
        region.view_distance=14
        region.view_rotation=Quaternion((0,0,1),.5) @ camera.rotation_euler.to_quaternion()
    elif name=='orthographic':area.spaces.active.region_3d.view_perspective='ORTHO'
    elif name=='reset_camera':
        with bpy.context.temp_override(area=area,region=next(r for r in area.regions if r.type=='WINDOW')):
            bpy.ops.ovrtx_example.reset_viewport_camera()
    elif name=='camera_lens':camera.data.lens=75
    elif name=='camera_aperture':camera.data.sensor_width=20
    elif name=='camera_offset':camera.data.shift_x=.3
    elif name=='camera_clip':camera.data.clip_start=15
    elif name=='camera_dof':
        camera.data.dof.use_dof=True
        camera.data.dof.focus_distance=2
        camera.data.dof.aperture_fstop=.5
    elif name=='camera_replace':
        alternate_camera=camera.copy();alternate_camera.data=camera.data.copy()
        scene.collection.objects.link(alternate_camera)
        alternate_camera.location.x-=3
        scene.camera=alternate_camera
    elif name.startswith('camera_restore_'):
        scene.camera=camera
        for key,value in camera_defaults.items():setattr(camera.data,key,value)
        camera.data.dof.use_dof=False
    elif name=='quality_edit':
        scene.ovrtx_example.min_samples=8
        scene.ovrtx_example.max_samples=48
        for key,value in {'rtpt_max_bounces':0,'rtpt_max_specular_and_transmission_bounces':0,
                          'rtpt_max_volume_bounces':0,'rtpt_firefly_filter_enabled':False}.items():
            setattr(scene.ovrtx_example,key,value)
    elif name=='quality_restore':
        scene.ovrtx_example.min_samples=1
        scene.ovrtx_example.max_samples=34
        for key,value in quality_defaults.items():setattr(scene.ovrtx_example,key,value)
    elif name=='final_render':
        scene.render.filepath=str(out/'viewport_to_F12.png')
        scene.render.image_settings.file_format='PNG'
        bpy.ops.render.render(write_still=True)
        assert (out/'viewport_to_F12.png').is_file()
    elif name=='editor_away':
        area.type='TEXT_EDITOR'
    elif name=='editor_return':enable_view(area)
    elif name=='two_views':
        scene.ovrtx_example.sync_viewport_camera=True
        area.spaces.active.region_3d.view_perspective='CAMERA'
        secondary=next(a for a in bpy.context.screen.areas if a.type=='PROPERTIES')
        enable_view(secondary,'PERSP')
        region=secondary.spaces.active.region_3d
        region.view_location=(0,0,1)
        region.view_distance=14
        region.view_rotation=Quaternion((0,0,1),-.7) @ camera.rotation_euler.to_quaternion()
    elif name=='move_one_view':
        region=secondary.spaces.active.region_3d
        region.view_rotation=Quaternion((0,0,1),.6) @ region.view_rotation
    elif name=='view_layer':
        layer=scene.view_layers.new('Alternate visible layer')
        cube.hide_set(True,view_layer=layer)
        bpy.context.window.view_layer=layer
    elif name=='view_layer_restore':bpy.context.window.view_layer=scene.view_layers[0]
    if area.type=='VIEW_3D':area.tag_redraw()


stages=['initial_hdr','held_hdr','hdr_exposure','hdr_exposure_reset','hdr_gamma','hdr_gamma_reset','hdr_view_transform','hdr_view_transform_reset','hdr_staging_on','hdr_staging_off','ldr','held_ldr',
        'ldr_exposure','ldr_exposure_reset','resize','resize_reset','dlss_default',
        'dlss_performance','perspective','held_perspective','orthographic','reset_camera',
        'camera_lens','camera_restore_lens','camera_aperture','camera_restore_aperture','camera_offset','camera_restore_offset','camera_clip','camera_restore_clip',
        'camera_dof','camera_restore_dof','camera_replace','camera_restore_replacement','quality_edit','quality_restore',
        'final_render','after_final','editor_away','editor_return','two_views',
        'move_one_view','view_layer','view_layer_restore']
if args.calibrate_baseline:stages=['initial_hdr','held_hdr','ldr','held_ldr']


def write_report():
    (out/'result.json').write_text(json.dumps(report,indent=2,default=str)+'\n')


def finish():
    engine.OvrtxExampleRenderEngine._draw_viewport_texture=original_draw
    if original_call is not None:PublicSdkRenderClient._call=original_call
    report['accepted_native_loads']=accepted_loads
    report['accepted_native_updates']=accepted_updates
    report['sdk_imports_in_blender']=[n for n in ('ovrtx','ovphysx','ovstage','ovstream','warp') if n in sys.modules]
    if report['sdk_imports_in_blender']:report['status']='failed'
    write_report()
    bpy.ops.wm.quit_blender()
    return None


def pixels(result):
    raw=raw_pixels(result)
    data=np.frombuffer(raw,'<f2' if result.linear_rgba16f else np.uint8).reshape(result.height,result.width,4)
    return data.astype(np.float32)/(1 if result.linear_rgba16f else 255)


def capture(name):
    rows=[];images=[]
    targets=[area]+([secondary] if secondary is not None else [])
    for target in targets:
        observed=draws.get(target.as_pointer())
        if observed is None:return False
        instance,result,texture,stamp=observed
        if stamp<since or result.completed_samples<scene.ovrtx_example.max_samples:return False
        frame=instance._presented_frame
        if not dlss_frame_ready(name,instance,result,frame,accepted_loads,since,report):return False
        if frame is None or frame.snapshot_key!=instance._written_snapshot_key:return False
        if (result.width,result.height)!=expected_size:return False
        hdr=scene.ovrtx_example.color_presentation_mode=='scene_linear_hdr'
        if result.render_var!=('HdrColor' if hdr else 'LdrColor'):return False
        if name=='hdr_staging_on' and not result.presentation_rgba32f:return False
        if name=='hdr_staging_off' and result.presentation_rgba32f:return False
        image=pixels(result)
        assert np.isfinite(image).all() and image[...,:3].std()>.01
        product_id=getattr(result,'product_id',instance._viewport_request.render_product_path)
        assert product_id==instance._viewport_request.render_product_path
        rows.append({'engine':id(instance),'texture':id(texture),'product_id':product_id,
            'render_var':result.render_var,'samples':result.completed_samples,'size':list(expected_size),
            'publication':frame.publication_index,'snapshot_current':True,
            'runtime':dict(result.native_timings),'raw_sha256':hashlib.sha256(image.tobytes()).hexdigest(),
            'camera_prim_path':instance._viewport_request.camera_prim_path,
            'camera_matrix':instance._viewport_request.camera_matrix,
            'dlss_enabled':instance._viewport_request.dlss_enabled,
            'rtpt_quality':dict(instance._viewport_request.rtpt_quality),
            'min_samples':instance._viewport_request.min_samples,
            'max_samples':instance._viewport_request.max_samples,
            'composition':ovrtx_scene_composition.diagnostics(instance._ovrtx_scene_composition),
            'accepted_load':accepted_loads[-1] if accepted_loads else None})
        images.append(image)
    if len(rows)>1:
        assert len({r['engine'] for r in rows})==2
        assert len({r['texture'] for r in rows})==2
        assert float(np.abs(images[0]-images[1]).mean())>noise[result.render_var],'two cameras share pixels'
    screenshots=out/(name+'.png')
    bpy.ops.screen.screenshot(filepath=str(screenshots))
    image=bpy.data.images.load(str(screenshots),check_existing=False)
    shot=np.empty(image.size[0]*image.size[1]*4,np.float32)
    image.pixels.foreach_get(shot)
    shot=shot.reshape(image.size[1],image.size[0],4)
    # Central viewport crop excludes UI/edges; actual display pixels, after OCIO.
    x,y,w,h=area.x,area.y,area.width,area.height
    crop=shot[y+h//4:y+3*h//4,x+w//4:x+3*w//4,:3].copy()
    bpy.data.images.remove(image)
    assert crop.std()>.005
    pixels_by_name[name]=images
    screens_by_name[name]=crop
    for i,im in enumerate(images):
        raw_path=out/f'{name}-{i}.npy'
        np.save(raw_path,im)
        rows[i]['raw_path']=str(raw_path)
    report['cases'].append({'name':name,'views':rows,'view_layer':bpy.context.window.view_layer.name,
                            'screenshot':str(screenshots),'crop_std':float(crop.std())})
    return True


def compare(name):
    def delta(a,b,index=0):return float(np.abs(pixels_by_name[a][index]-pixels_by_name[b][index]).mean())
    if name=='held_hdr':
        if args.calibrate_baseline:noise['HdrColor']=max(.005,3*delta('held_hdr','initial_hdr'))
        assert delta('held_hdr','initial_hdr')<=noise['HdrColor']
    elif name=='held_ldr':
        if args.calibrate_baseline:noise['LdrColor']=max(.005,3*delta('held_ldr','ldr'))
        assert delta('held_ldr','ldr')<=noise['LdrColor']
    elif name in {'hdr_exposure','hdr_gamma','hdr_view_transform'}:
        assert float(np.abs(screens_by_name[name]-screens_by_name['held_hdr']).mean())>.01
        assert delta(name,'held_hdr')<=noise['HdrColor'],'Blender display exposure changed native scene-linear pixels'
    elif name=='ldr_exposure':
        assert delta(name,'held_ldr')<=noise['LdrColor'],'LDR exposure incorrectly alters renderer pixels'
        assert float(np.abs(screens_by_name[name]-screens_by_name['ldr']).mean())<.01,'double display transform on LDR'
    elif name=='hdr_staging_off':assert delta(name,'hdr_staging_on')<=noise['HdrColor']
    elif name=='held_perspective':assert delta(name,'perspective')<=noise['LdrColor']
    elif name in {'perspective','orthographic'}:
        before='dlss_performance' if name=='perspective' else 'held_perspective'
        assert delta(name,before)>noise['LdrColor']
    elif name=='reset_camera':
        assert not scene.ovrtx_example.sync_viewport_camera
        assert report['cases'][-1]['views'][0]['camera_matrix'] is None
        assert delta(name,'orthographic')>noise['LdrColor']
        assert delta(name,'dlss_performance')<=noise['LdrColor'],'reset did not restore authored scene camera'
    elif name.startswith('camera_restore_'):
        assert delta(name,'reset_camera')<=noise['LdrColor'],'camera state did not restore pixels'
    elif name in {'camera_lens','camera_aperture','camera_offset','camera_clip','camera_dof','camera_replace'}:
        assert delta(name,'reset_camera')>noise['LdrColor'],'camera edit did not reach pixels'
    elif name in {'quality_edit','quality_restore'}:
        request=draws[area.as_pointer()][0]._viewport_request
        for key in properties.RTPT_RENDER_SETTINGS:
            assert request.rtpt_quality[key]==getattr(scene.ovrtx_example,key),(key,request.rtpt_quality)
        effective={}
        for key,spec in properties.RTPT_RENDER_SETTINGS.items():
            expected=spec.to_wire(request.rtpt_quality[key])
            carb_path='/'+'/'.join(spec.attribute.split(':')[1:])
            matching=[row for batch in accepted_updates if batch['accepted_at']>since
                      for row in batch['values'] if row['attribute_name']==spec.attribute]
            if matching:
                assert matching[-1]['value']==expected,(key,matching[-1],expected)
                effective[key]={'route':'acknowledged_live_value','value':matching[-1]['value']}
            else:
                load=accepted_loads[-1]
                assert load['accepted_at']>since and load['settings'][carb_path]==expected,(key,load,expected)
                effective[key]={'route':'accepted_load_setting','value':load['settings'][carb_path]}
        report['cases'][-1]['effective_quality']=effective
        assert request.min_samples==scene.ovrtx_example.min_samples
        assert request.max_samples==scene.ovrtx_example.max_samples
        assert report['cases'][-1]['views'][0]['samples']==request.max_samples
        if name=='quality_restore':assert delta(name,'reset_camera')<=noise['LdrColor']
        else:assert delta(name,'reset_camera')>noise['LdrColor'],'quality changes acknowledged but no visible effect'
    elif name in {'after_final','editor_return'}:
        assert delta(name,'reset_camera')<=noise['LdrColor'],'return path retained another camera/frame'
    elif name=='dlss_default':
        assert report['cases'][-1]['views'][0]['dlss_enabled'] is True
        assert accepted_loads[-1]['accepted_at']>since
        assert '/rtx/post/dlss/execMode' not in accepted_loads[-1]['settings']
        compare_dlss_policy(name,report,pixels_by_name,dlss_reference,noise['LdrColor'])
    elif name=='dlss_performance':
        assert report['cases'][-1]['views'][0]['dlss_enabled'] is False
        assert accepted_loads[-1]['accepted_at']>since
        assert accepted_loads[-1]['settings']['/rtx/post/dlss/execMode']=='performance'
        assert delta(name,'resize_reset')<=noise['LdrColor'],'DLSS preset reset did not restore output'
        compare_dlss_policy(name,report,pixels_by_name,dlss_reference,noise['LdrColor'])
    elif name=='move_one_view':
        assert delta(name,'two_views',0)<=noise['LdrColor'],'moving second camera contaminated first viewport'
        assert delta(name,'two_views',1)>noise['LdrColor'],'second camera retained stale pixels'
    elif name=='view_layer':assert delta(name,'move_one_view')>noise['LdrColor'],'view layer visibility did not reach pixels'
    elif name=='view_layer_restore':assert delta(name,'move_one_view')<=noise['LdrColor'],'restored view layer contaminated pixels'
    if reference and name in {'initial_hdr','ldr'}:
        mode='HdrColor' if name=='initial_hdr' else 'LdrColor'
        baseline_case=next(c for c in reference['cases'] if c['name']==name)
        baseline=np.load(baseline_case['views'][0]['raw_path'])
        error=float(np.abs(pixels_by_name[name][0]-baseline).mean())
        report.setdefault('baseline_comparisons',[]).append({'mode':mode,'mae':error,'limit':noise[mode]})
        assert error<=noise[mode],(mode,error,noise[mode])


def tick():
    global stage,armed,since
    try:
        name=stages[stage]
        if not armed:
            action(name);since=time.monotonic();armed=True
            return .25
        if time.monotonic()-since>180:raise TimeoutError('No current presented frame for '+name)
        for target in (area,secondary):
            if target is not None:target.tag_redraw()
        if time.monotonic()-since<2:return .25
        if name=='editor_away':
            report['cases'].append({'name':name,'editor':area.type})
        elif not capture(name):return .25
        else:compare(name)
        stage+=1;armed=False
        write_report()
        if stage==len(stages):
            report['thresholds']=dict(noise)
            report['threshold_rationale']='3x matched baseline within-mode repeat MAE, minimum0.005; LDR normalized0..1; frozen before candidate'
            report['status']='passed'
            return finish()
        return .25
    except Exception:
        report['status']='failed';report['error']=traceback.format_exc()
        return finish()


bpy.app.timers.register(tick,first_interval=1)

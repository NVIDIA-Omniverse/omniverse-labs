import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as S
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).parent))
from viewport_output_baseline import (ORIGINAL_FIXTURE, ORIGINAL_FUNCTIONS,
    ORIGINAL_MODULE, verify_fixture_compatibility, load_dlss_reference, compare_dlss_policy, dlss_frame_ready)

FIXTURE=Path(__file__).with_name('blender_viewport_output_pixels.py')


def test_viewport_baseline_functions_preserved():
    verified=verify_fixture_compatibility(FIXTURE,ORIGINAL_FIXTURE)
    assert verified['normalized_functions_sha256']==ORIGINAL_FUNCTIONS
    assert verified['normalized_module_sha256']==ORIGINAL_MODULE


@pytest.mark.parametrize('before,after',[
    ("camera.data.lens=75","camera.data.lens=55"),
    ("crop.std()>.005","crop.std()>.001"),
    ("max(.005,3*delta('held_ldr','ldr'))","max(.1,3*delta('held_ldr','ldr'))"),
    ("assert accepted_loads[-1]['accepted_at']>since","assert True"),
    ("assert delta(name,'resize_reset')<=noise['LdrColor']","assert True"),
    ("stages=['initial_hdr','held_hdr'","stages=['held_hdr','initial_hdr'"),
    ("noise={'HdrColor':.005,'LdrColor':.005}","noise={'HdrColor':.1,'LdrColor':.1}"),
    ("scene,cube,camera=create_scene(addon)","scene,cube,camera=create_scene(addon);scene.view_settings.exposure=2"),
])
def test_other_viewport_gates_cannot_change(tmp_path,before,after):
    text=FIXTURE.read_text();assert before in text
    path=tmp_path/'fixture.py';path.write_text(text.replace(before,after))
    with pytest.raises(AssertionError):verify_fixture_compatibility(path,ORIGINAL_FIXTURE)


def baseline(tmp_path):
    directory=tmp_path/'_build'/'baseline';directory.mkdir(parents=True)
    retained=tmp_path/'retained';retained.mkdir();source=retained/'addon.py';source.write_text('original')
    rows=[];policies={};files={}
    def pin(path):files[path.relative_to(tmp_path).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    for name,enabled in [('dlss_default',True),('dlss_performance',False)]:
        raw=directory/(name+'.npy');pixels=np.full((241,321,4),.25,np.float32);np.save(raw,pixels)
        config=directory/(name+'.json');config.write_text(json.dumps({} if enabled else {'rtx':{'post':{'dlss':{'execMode':0}}}}))
        pin(raw);pin(config)
        row={'dlss_enabled':enabled,'size':[321,241],'samples':34,'render_var':'LdrColor','product_id':'/Product',
             'dlss_authored_opinion':{'value':None if enabled else '0'},'raw_sha256':hashlib.sha256(pixels.tobytes()).hexdigest()}
        rows.append({'name':name,'views':[row]})
        policies[name]={'enabled':enabled,'product_id':'/Product','authored_opinion':None if enabled else '0',
            'config_value':None if enabled else 0,'raw_path':raw.relative_to(tmp_path).as_posix(),'config_path':config.relative_to(tmp_path).as_posix()}
    reference=directory/'result.json';reference.write_text(json.dumps({'status':'passed','qualification':False,
        'fixture_sha256':ORIGINAL_FIXTURE,'shared_scene_sha256':'shared','cases':rows}));pin(reference)
    receipt=directory/'receipt.json';receipt.write_text(json.dumps({'loaded_baseline_changed':[],'source_changed':[],
        'original_fixtures_unchanged':True,'reference_unchanged':True,'environment_overrides':{'OV_OUTPUT_ADDON_ROOT':str(retained)},
        'loaded_baseline_before':{'addon.py':hashlib.sha256(source.read_bytes()).hexdigest()},
        'baseline_worker_sha256':'worker','baseline_addon_pin':'pin'}));pin(receipt)
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'shared_scene_sha256':'shared',
        'reference_path':reference.relative_to(tmp_path).as_posix(),'receipt_path':receipt.relative_to(tmp_path).as_posix(),
        'files':files,'policies':policies,'baseline_worker_sha256':'worker','baseline_addon_pin':'pin'}))
    return reference,manifest,source


@pytest.mark.parametrize('target',['raw','config','source','shared_scene'])
def test_stale_baseline_is_rejected(tmp_path,target):
    reference,manifest,source=baseline(tmp_path)
    path={'raw':reference.with_name('dlss_default.npy'),'config':reference.with_name('dlss_default.json'),'source':source}.get(target)
    if path:path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(AssertionError):
        load_dlss_reference(reference,'wrong' if target=='shared_scene' else 'shared',manifest_path=manifest)


def test_policy_pixels_use_unchanged_limit_and_identity(tmp_path):
    reference,manifest,_=baseline(tmp_path);loaded=load_dlss_reference(reference,'shared',manifest_path=manifest)
    report={'cases':[{'views':[{'dlss_enabled':True,'product_id':'/Product','size':[321,241]}]}]}
    pixels={'dlss_default':[np.full((241,321,4),.25,np.float32)]}
    compare_dlss_policy('dlss_default',report,pixels,loaded,.0054700004697434985)
    pixels['dlss_default'][0]+=.01
    with pytest.raises(AssertionError):compare_dlss_policy('dlss_default',report,pixels,loaded,.0054700004697434985)
    pixels['dlss_default'][0][:]=.25;report['cases'][-1]['views'][0]['dlss_enabled']=False
    with pytest.raises(AssertionError):compare_dlss_policy('dlss_default',report,pixels,loaded,.0054700004697434985)


@pytest.mark.parametrize('stale',['load','setting','stage','path','presentation','opinion',None])
def test_policy_capture_waits_for_actual_replacement(monkeypatch,stale):
    result=S(native_timings={'native_stage_id':'old' if stale=='stage' else 'new','scene_path':'old' if stale=='path' else 'current'})
    frame=S(render_result=None if stale=='presentation' else result)
    instance=S(_ovrtx_scene_composition=S(composed_scene_path='current'),_viewport_request=S(render_product_path='/Product'))
    attribute=S(HasAuthoredValueOpinion=lambda:stale=='opinion',Get=lambda:'performance')
    monkeypatch.setitem(sys.modules,'pxr',S(Usd=S(Stage=S(Open=lambda path:S(GetPrimAtPath=lambda path:S(GetAttribute=lambda name:attribute))))))
    loads=[{'accepted_at':0 if stale=='load' else 2,'settings':{'/rtx/post/dlss/execMode':'performance'} if stale=='setting' else {}}]
    report={'cases':[{'name':'resize_reset','views':[{'runtime':{'native_stage_id':'old'}}]}]}
    assert dlss_frame_ready('dlss_default',instance,result,frame,loads,1,report) is (stale is None)

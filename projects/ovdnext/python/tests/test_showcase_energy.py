# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Analytic and selection oracles for the independently authored CPU control."""
import importlib.util
import math
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pytest


def module():
    path=Path(__file__).resolve().parents[2]/"tools/evidence_showcase.py"
    spec=importlib.util.spec_from_file_location("evidence_showcase",path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def inputs():
    return (dict(units="SI",workPolicy="diagnostic-no-work-accounting",mass=2,
        inertia=[1,2,3],comPose=[0,0,0,1,0,0,1],gravity=[0,0,-10],reference=[0,0,0]),
        dict(pose=[0,0,0,1,1,2,3],linear=[2,0,0],angular=[0,0,1]))


def test_mechanical_energy_uses_world_com_and_mass_frame():
    tool=module();setup,sample=inputs()
    energy,reason=tool.energy_from_sample(setup,sample)
    assert not reason and energy.components==pytest.approx((4,1.5,80))
    assert energy.total==pytest.approx(85.5)
    setup['comPose']=[0,math.sqrt(.5),0,math.sqrt(.5),0,0,1]
    rotated,reason=tool.energy_from_sample(setup,sample)
    assert not reason and rotated.components[1]==pytest.approx(.5)


@pytest.mark.parametrize('change',[
    lambda s,p:s.update(units='unknown'),lambda s,p:s.pop('workPolicy'),
    lambda s,p:s.update(mass=None),lambda s,p:p.update(linear=[math.nan,0,0]),
    lambda s,p:p.update(pose=[0,0,0,0,1,2,3]),lambda s,p:s.pop('inertia')])
def test_incomplete_or_invalid_inputs_are_inconclusive(change):
    setup,sample=inputs();change(setup,sample)
    energy,reason=module().energy_from_sample(setup,sample)
    assert energy is None and reason


def test_handover_selection_has_seven_points_fixed_threshold_and_no_pass_label():
    tool=module();setup,sample=inputs();energy,_=tool.energy_from_sample(setup,sample)
    rows=[dict(tilt=.5 if i<4 else -.5,energy=energy) for i in range(9)]
    report=tool.handover_windows(rows,1/240)
    assert report['eligible']==1 and report['excluded']=={}
    window=report['windows'][0]
    assert window['indices']==tuple(range(1,8)) and window['anchor']==4
    assert window['fraction']==0 and window['status']=='Observed'
    assert window['tolerance']==pytest.approx(.01)
    increased=tool.MechanicalEnergy((8,1.5,80),80,0)
    rows[7]['energy']=increased
    warning=tool.handover_windows(rows,1/240)['windows'][0]
    assert warning['status']=='Warning' and warning['fraction']>warning['tolerance']


def test_window_exclusions_and_eight_window_limit_are_explicit():
    tool=module();setup,sample=inputs();energy,_=tool.energy_from_sample(setup,sample)
    rows=[dict(tilt=.5 if (i//5)%2==0 else -.5,energy=energy) for i in range(70)]
    result=tool.handover_windows(rows,1/480)
    assert len(result['windows'])==8 and result['eligible']>8
    rows[0]['energy']=None
    result=tool.handover_windows(rows,1/480)
    assert not result['windows'] and result['excluded']['initial-energy']>0
    with pytest.raises(ValueError):tool.handover_windows(rows,float('nan'))
    with pytest.raises(ValueError):tool.handover_windows(rows,.1)


def test_sdk_ordinal_comparison_requires_the_same_actual_timestep():
    tool=module()
    assert tool.compare_same_timebase(dict(stepSeconds=1/240),dict(stepSeconds=1/240))
    assert not tool.compare_same_timebase(dict(stepSeconds=1/240),dict(stepSeconds=1/480))


def test_phase_domain_cannot_be_guessed_from_frame_parity():
    tool=module();actor=SimpleNamespace(recording_segment=0,scene_context=123,scene_object_uid=9,supported=True)
    scope=SimpleNamespace(segment=0,context=123,scene_uid=9,scene_mapped=True,monotonic=True,
        consecutive=True,scene_recreated=False,first=1,last=5,count=5)
    domain=SimpleNamespace(supported=True,frame_scopes_complete=True,scopes=[scope])
    tool._phase_domain(domain,actor,2)
    for field,value in [('consecutive',False),('monotonic',False),('scene_recreated',True),('first',2),('count',4)]:
        previous=getattr(scope,field);setattr(scope,field,value)
        with pytest.raises(ValueError):tool._phase_domain(domain,actor,2)
        setattr(scope,field,previous)


def test_live_log_requires_complete_pair_and_ordered_pre_post_samples(tmp_path):
    tool=module();path=tmp_path/'live.json'
    valid=dict(format='ovdnext.cpu-rocking-control',version=1,complete=True,
        sdkRevision='b97365c6ddb32fda8469bc788ec18024862b0368',solver='CPU-TGS',buildGPUData=False,
        units='SI',workPolicy='diagnostic-no-work-accounting',steps=1,hz=240,
        stepSeconds=0.004166666883975267,samples=[dict(step=1,phase='pre'),dict(step=1,phase='post')])
    path.write_text(json.dumps(valid));assert tool.read_live_log(path)['steps']==1
    for change in [lambda d:d.update(complete=False),lambda d:d.update(version=True),
                   lambda d:d.update(stepSeconds=1/240),lambda d:d['samples'].reverse()]:
        broken=json.loads(json.dumps(valid));change(broken);path.write_text(json.dumps(broken))
        with pytest.raises(ValueError):tool.read_live_log(path)
    path.write_text(json.dumps(valid));path.with_name('live.json.failed').write_text('failed')
    with pytest.raises(ValueError,match='marked'):tool.read_live_log(path)


def test_cli_requires_its_explicit_native_root_before_creating_outputs(tmp_path):
    script=Path(__file__).resolve().parents[2]/'tools/evidence_showcase.py'
    arguments=[]
    for name in ('left','right','left-log','right-log','inspector','native-root','output'):
        arguments.extend(('--'+name,str(tmp_path/name)))
    result=subprocess.run([sys.executable,'-B',str(script),*arguments],capture_output=True,text=True,timeout=15)
    assert result.returncode!=0 and 'Explicit native root' in result.stderr
    assert not (tmp_path/'output.assets').exists()


@pytest.mark.parametrize('failed_role',['left','right'])
def test_generate_rejects_original_failed_log_before_staging(tmp_path,monkeypatch,failed_role):
    tool=module();args=SimpleNamespace(output=tmp_path/'evidence.json',inspector=tmp_path/'unused-inspector')
    for role in ('left','right'):
        source=tmp_path/(role+'.ovd');source.write_bytes(b'Native boundary is isolated in this staging regression')
        path=tmp_path/(role+'.json')
        log=dict(format='ovdnext.cpu-rocking-control',version=1,complete=True,
            sdkRevision='b97365c6ddb32fda8469bc788ec18024862b0368',solver='CPU-TGS',buildGPUData=False,
            units='SI',workPolicy='diagnostic-no-work-accounting',steps=1,hz=240,
            stepSeconds=0.004166666883975267,clientSha256=tool._sha(tool.ROOT/'examples/evidence_timeline/rocking_capture.cpp'),
            samples=[dict(step=1,phase='pre'),dict(step=1,phase='post')])
        path.write_text(json.dumps(log));setattr(args,role,source);setattr(args,role+'_log',path)
    original=getattr(args,failed_role+'_log')
    original.with_name(original.name+'.failed').write_text('Final recorder close failed')
    captured=[];closed=[]
    def capture(role,source,log,cache):
        assert role!=failed_role,'Failed recorder pair reached native ingestion'
        captured.append(role)
        return dict(proof=SimpleNamespace(path=source,sha256=tool._sha(source)),
            loaded=SimpleNamespace(close=lambda:closed.append(role)))
    monkeypatch.setattr(tool,'_capture',capture)
    with pytest.raises(ValueError,match='Recorder marked this pair failed'):
        tool.generate(args)
    assert not args.output.exists()
    assets=args.output.with_name(args.output.name+'.assets')
    assert not (assets/(failed_role+'-live.json')).exists()
    assert captured==closed==([] if failed_role=='left' else ['left'])
    receipt=json.loads((assets/'generation-failed.json').read_text())
    assert receipt['complete'] is False and 'Recorder marked this pair failed' in receipt['error']


def test_nonzero_orientation_matches_actual_sdk_cpu_control_stdout():
    from ovd_ui.evidence import parse_evidence
    path=Path(__file__).parent/'fixtures/evidence/v2-cpu-control-orientation.json'
    evidence=parse_evidence(path.read_bytes())
    finding=evidence.findings[0];sdk=finding.data['evidence']
    assert sdk['measure']['distanceKind']=='TransformOrientationRadians'
    assert sdk['distance']==pytest.approx(0.026185872073672505,abs=1e-15)
    assert not finding.reasons

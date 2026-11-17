# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qualified CPU-control evidence; explicit engineering thresholds, no solver verdict."""
from collections import Counter
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))

from ovd_ui.evidence_energy import MechanicalEnergy, mechanical_energy


def energy_from_sample(setup, sample):
    try:
        if setup['units']!='SI' or setup['workPolicy']!='diagnostic-no-work-accounting':
            raise ValueError('Explicit SI units and diagnostic work assumptions required')
        return mechanical_energy(setup['mass'],setup['inertia'],setup['comPose'],sample['pose'],
            sample['linear'],sample['angular'],setup['gravity'],setup['reference']),''
    except (KeyError,TypeError,ValueError,OverflowError) as error:
        return None,'Inconclusive energy inputs: '+str(error)


def handover_windows(samples, dt):
    if not math.isfinite(dt) or not any(math.isclose(dt,1/hz,rel_tol=1e-6) for hz in (240,480)):
        raise ValueError('This recipe requires the actual finite 240 or 480 Hz timestep')
    excluded,windows,eligible=Counter(),[],0
    initial=samples[0]['energy'] if samples else None
    for i in range(1,len(samples)):
        a,b=samples[i-1]['tilt'],samples[i]['tilt']
        if not all(math.isfinite(t) for t in (a,b)) or a*b>=0:continue
        reason=''
        if abs(b)>=1:reason='tilt-outside-one-degree'
        elif i<3 or i+3>=len(samples):reason='incomplete-seven-point-window'
        elif initial is None:reason='initial-energy'
        else:
            points=samples[i-3:i+4]
            before,after=points[0]['energy'],points[-1]['energy']
            if any(p['energy'] is None for p in points):reason='missing-energy-inputs'
            elif before.total<=max(0,.1*initial.total,10*before.roundoff):reason='small-reference-energy'
        if reason:excluded[reason]+=1;continue
        fraction=(after.total-before.total)/before.total
        absolute=max(.01*abs(before.total),before.roundoff+after.roundoff)
        tolerance=absolute/before.total
        eligible+=1
        if len(windows)<8:
            windows.append(dict(indices=tuple(range(i-3,i+4)),anchor=i,fraction=fraction,
                tolerance=tolerance,status='Warning' if fraction>tolerance else 'Observed'))
    return dict(windows=windows,eligible=eligible,excluded=dict(excluded),unselected=max(0,eligible-8))


def compare_same_timebase(left,right):
    return left['stepSeconds']==right['stepSeconds']


def _unique(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('Duplicate JSON key')
        result[key]=value
    return result


def read_live_log(path):
    if path.with_name(path.name+'.failed').exists():raise ValueError('Recorder marked this pair failed')
    if path.stat().st_size>64*1024*1024:raise ValueError('Live log exceeds 64 MiB')
    data=json.loads(path.read_bytes(),object_pairs_hook=_unique,
        parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Nonfinite live log')))
    if (type(data) is not dict or data.get('format')!='ovdnext.cpu-rocking-control' or type(data.get('version')) is not int or data.get('version')!=1 or data.get('complete') is not True
            or data.get('sdkRevision')!='b97365c6ddb32fda8469bc788ec18024862b0368'
            or data.get('solver')!='CPU-TGS' or data.get('buildGPUData') is not False
            or data.get('units')!='SI' or data.get('workPolicy')!='diagnostic-no-work-accounting'):
        raise ValueError('Requires a complete qualified CPU-control log with explicit assumptions')
    steps,hz=data.get('steps'),data.get('hz')
    if type(steps) is not int or not 1<=steps<=4800 or type(hz) is not int or hz not in (240,480):
        raise ValueError('Unsupported step count/rate')
    dt=struct.unpack('<f',struct.pack('<f',1/hz))[0]
    if data.get('stepSeconds')!=dt:raise ValueError('Timestep differs from actual recipe float')
    samples=data.get('samples')
    if type(samples) is not list or len(samples)!=2*steps:raise ValueError('Incomplete pre/post log')
    for i,row in enumerate(samples):
        if type(row) is not dict or type(row.get('step')) is not int or row.get('step')!=i//2+1 or row.get('phase')!=('post' if i%2 else 'pre'):
            raise ValueError('Live step/phase sequence differs')
    return data


def _sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def _tool(inspector, arguments, assets, name, allowed=(0,)):
    command=[str(inspector),*map(str,arguments)]
    result=subprocess.run(command,cwd=assets,capture_output=True,timeout=60)
    (assets/(name+'.stdout.json')).write_bytes(result.stdout)
    (assets/(name+'.stderr.txt')).write_bytes(result.stderr)
    (assets/(name+'.command.json')).write_text(json.dumps(dict(argv=command,exit=result.returncode,
        toolSha256=_sha(inspector)),indent=2)+'\n',encoding='utf-8')
    if result.returncode not in allowed:raise ValueError(f'Inspector {name} failed: exit {result.returncode}')
    return json.loads(result.stdout,object_pairs_hook=_unique)


FIELDS={'mass':(1,'kg','mass'),'massSpaceInertiaTensor':(3,'kg*m^2','inertia'),
    'cMassLocalPose':(7,'m','comPose'),'gravity':(3,'m/s^2','gravity'),
    'globalPose':(7,'m','pose'),'linearVelocity':(3,'m/s','linear'),'angularVelocity':(3,'rad/s','angular')}


def _phase_domain(domain,actor,steps):
    scopes=[s for s in domain.scopes if (s.segment,s.context,s.scene_uid)==(actor.recording_segment,actor.scene_context,actor.scene_object_uid)]
    if not domain.supported or not domain.frame_scopes_complete or not actor.supported or len(scopes)!=1:
        raise ValueError('Unsupported or incomplete native phase domain')
    scope=scopes[0]
    if (not scope.scene_mapped or not scope.monotonic or not scope.consecutive or scope.scene_recreated
            or (scope.first,scope.last,scope.count)!=(1,2*steps+1,2*steps+1)):
        raise ValueError('Recipe phase domain has gaps, resets, recreation or unexpected frame bounds')


def _capture(role,path,log,cache):
    from ovd_ui.ingestion import load_capture
    from ovd_ui.evidence_binding import _namespace,_property
    loaded=load_capture(path,temporary_directory=cache)
    try:
        if loaded.verified_payload is None:raise ValueError('Finalized capture snapshot unavailable')
        session=loaded.session;index=_namespace(session,lambda:False)
        candidates=[r for group in index[1].values() for r in group if r.class_name=='PxRigidDynamic'
            and str(r.api_handle)==log['actorHandle'] and str(r.scene_context)==log['sceneHandle']]
        if len(candidates)!=1:raise ValueError('Live actor handle does not select one original native source identity')
        actor=candidates[0];scene=index[2][actor.scene_instance_id]
        _phase_domain(index[0],actor,log['steps'])
        if actor.recording_segment!=0 or str(scene.api_handle)!=log['sceneHandle']:
            raise ValueError('Unexpected source scene/segment')
        props={name:(scene if name=='gravity' else actor,*_property(session,scene if name=='gravity' else actor,index,name,count))
            for name,(count,_,_) in FIELDS.items()}
        selected=index[4][actor.scene_instance_id]
        snapshots=[];posts=[]
        for row in log['samples']:
            step=row['step'];post=row['phase']=='post';frame=2*step if post else 2*step-1
            view=session.presentation(selected.id,frame,inspect_ids=[actor.instance_id,scene.instance_id],
                read_requests=[(identity.instance_id,p.id,0,FIELDS[name][0]*(4 if code==8 else 8)) for name,(identity,p,code) in props.items()],include_evidence=True)
            if not view.ready or not view.request_applied or view.frame!=frame:raise ValueError('Native frame request was not applied')
            observed={}
            for name,(identity,prop,code) in props.items():
                count,_,key=FIELDS[name];width=4 if code==8 else 8
                stamp=view.evidence_sample_stamp(identity.instance_id,prop.id)
                frozen=view.evidence_identity(identity.instance_id)
                if frozen is None or frozen.object_uid!=identity.object_uid or frozen.api_handle!=identity.api_handle:
                    raise ValueError('Frozen source identity differs')
                if stamp is None or not stamp.supported or not stamp.has_sample:raise ValueError('Required captured field/source missing')
                values=struct.unpack('<'+str(count)+('f' if code==8 else 'd'),view.read(identity.instance_id,prop.id))
                expected=row[key] if count!=1 else [row[key]]
                if tuple(expected)!=values:raise ValueError(f'{role} {name} live/recorded mismatch at step {step} {row["phase"]}')
                if name in ('globalPose','linearVelocity','angularVelocity') and stamp.frame!=(frame if post else frame-1):
                    raise ValueError('Dynamic pre/post source-stamp proof failed')
                observed[name]=dict(value=list(values),stamp=stamp.frame)
            snapshots.append(observed)
            if post:
                setup=dict(units=log['units'],workPolicy=log['workPolicy'],mass=observed['mass']['value'][0],
                    inertia=observed['massSpaceInertiaTensor']['value'],comPose=observed['cMassLocalPose']['value'],
                    gravity=observed['gravity']['value'],reference=[0,0,.1177/2])
                energy,reason=energy_from_sample(setup,dict(pose=observed['globalPose']['value'],
                    linear=observed['linearVelocity']['value'],angular=observed['angularVelocity']['value']))
                if reason:raise ValueError(reason)
                # The tilt is recomputed from recorded quaternion components.
                x,y,z,w=observed['globalPose']['value'][:4]
                tilt=math.degrees(math.asin(max(-1.,min(1.,2*(w*x-y*z)))))
                if not math.isclose(tilt,row['tilt'],abs_tol=2e-5):raise ValueError('Recorded/logged tilt differs')
                posts.append(dict(step=step,frame=frame,tilt=tilt,energy=energy,observations=observed))
        return dict(loaded=loaded,actor=actor,scene=scene,posts=posts,role=role,log=log,
            proof=loaded.verified_payload,report=handover_windows(posts,log['stepSeconds']),verifiedSamples=len(snapshots)*len(FIELDS))
    except BaseException:
        loaded.close();raise


def _frame(capture,n):
    return dict(recordingSegment=str(capture['actor'].recording_segment),context=str(capture['actor'].scene_context),recordedFrame=str(n))


def _scalars(capture,identity,scene_identity):
    role=capture['role'];findings=[];observations={}
    def observation(name,point):
        item=point['observations'][name];stamp=item['stamp'];frame=point['frame']
        key=f'{role}-{name}-{frame}'
        observations[key]=dict(id=key,role=role,identity=scene_identity if name=='gravity' else identity,
            attribute=name,units=FIELDS[name][1],requested=_frame(capture,frame),source=_frame(capture,stamp) if stamp else None,
            sampleState='Exact' if stamp==frame else 'Carried',origin='native-recorded',value=item['value'],
            note='Actual native stored value/source, checked against paired live SDK log')
        return key
    for number,window in enumerate(capture['report']['windows']):
        points=[];first=capture['posts'][window['indices'][0]]
        supporting=[observation(name,first) for name in ('mass','massSpaceInertiaTensor','cMassLocalPose','gravity')]
        ref=f'{role}-reference-{number}'
        observations[ref]=dict(id=ref,role=role,identity=identity,attribute='energyReferencePosition',units='m',
            requested=_frame(capture,first['frame']),source=None,sampleState='Carried',origin='producer-log',
            value=[0,0,.1177/2],note='Explicit analytic reference at upright half-height; not a captured physical fact')
        for index in window['indices']:
            sample=capture['posts'][index];energy=sample['energy']
            points.append(dict(frame=_frame(capture,sample['frame']),phase='post',valueJ=energy.total,
                components=dict(zip(('translationalJ','rotationalJ','gravitationalJ'),energy.components)),
                observations=[observation(name,sample) for name in ('globalPose','linearVelocity','angularVelocity')]))
        findings.append(dict(id=f'{role}-handover-{number+1}',kind='scalar-window',role=role,identity=identity,
            anchor=_frame(capture,capture['posts'][window['anchor']]['frame']),points=points,supporting=[*supporting,ref],
            measure=dict(name='mechanicalEnergy.relativeHandoverChange',units='fraction',value=window['fraction'],reference=0,
                tolerance=window['tolerance'],status=window['status']),
            assumptions='SI assumed from explicit CPU recipe; COM/mass-frame energy. External contact/constraint work is not measured. Threshold is diagnostic, not a solver-error bound.',
            message='Seven evaluated post-step points around a recorded tilt sign change; other spans are unevaluated.'))
    return findings,list(observations.values())


def generate(args):
    from ovd_ui.evidence import parse_evidence
    from ovd_ui.evidence_binding import resolve_bindings,verify_payload
    output=args.output.resolve();assets=output.with_name(output.name+'.assets')
    if output.exists() or assets.exists():raise ValueError('Evidence output and assets must be new paths')
    assets.mkdir()
    captures=[]
    try:
        for role in ('left','right'):
            source=getattr(args,role).resolve();logpath=getattr(args,role+'_log').resolve()
            if logpath.with_name(logpath.name+'.failed').exists():
                raise ValueError('Recorder marked this pair failed')
            staged_log=assets/(role+'-live.json');shutil.copyfile(logpath,staged_log);log=read_live_log(staged_log)
            if log['clientSha256']!=_sha(ROOT/'examples/evidence_timeline/rocking_capture.cpp'):
                raise ValueError('Recorder source differs from the compiled client provenance')
            capture=_capture(role,source,log,assets);captures.append(capture)
            shutil.copyfile(capture['proof'].path,assets/(role+'.ovd'))
            if _sha(assets/(role+'.ovd'))!=capture['proof'].sha256:raise ValueError('Staged inspector input differs')
        left,right=captures;inspector=args.inspector.resolve()
        version=subprocess.run([str(inspector),'--version'],capture_output=True,check=True,timeout=15)
        (assets/'inspector-version.txt').write_bytes(version.stdout+version.stderr)
        common=['--left-id',left['actor'].object_uid,'--right-id',right['actor'].object_uid,
            '--left-segment',0,'--right-segment',0,'--left-context',left['actor'].scene_context,'--right-context',right['actor'].scene_context,
            '--start',2,'--end',2*min(len(left['posts']),len(right['posts'])),'--stride',2,'--offset',0,'--tolerance',0]
        findings=[];observations=[]
        comparisons=[('velocity',['--units','m/s']),('position',['--attribute','globalPose','--distance','position','--units','m']),
                     ('orientation',['--attribute','globalPose','--distance','orientation'])] if compare_same_timebase(left['log'],right['log']) else []
        for name,extra in comparisons:
            sdk=_tool(inspector,['compare','left.ovd','right.ovd',*common,*extra],assets,'compare-'+name,(0,1))
            findings.append(dict(id='sdk-'+name,kind='comparison',evidence=sdk))
        for capture in captures:
            role=capture['role'];base=['inspect',role+'.ovd','--segment',0,'--context',capture['actor'].scene_context,'--frame',2]
            body=_tool(inspector,[*base,'--id',capture['actor'].object_uid,'--attribute','globalPose'],assets,role+'-body')
            scene=_tool(inspector,[*base,'--id',capture['scene'].object_uid,'--attribute','gravity'],assets,role+'-scene')
            scalar,values=_scalars(capture,body['sample']['identity'],scene['sample']['identity'])
            findings.extend(scalar);observations.extend(values)
        provenance=dict(control='Independent CPU scene/initial-condition diagnostic; no GPU reproduction/fix/root-cause claim',
            sdkRevision=left['log']['sdkRevision'],inspectorSha256=_sha(inspector),
            sdkComparison='Equal actual timesteps: recorded post frames aligned' if comparisons else 'Different timesteps: no SDK ordinal comparison; scalar windows retain their own times and widths',
            captures={c['role']:dict(hz=c['log']['hz'],actualStepSeconds=c['log']['stepSeconds'],tilt=c['log']['initialTiltDegrees'],
                verifiedFieldSamples=c['verifiedSamples'],eligible=c['report']['eligible'],excluded=c['report']['excluded'],
                unselected=c['report']['unselected'],selected=len(c['report']['windows'])) for c in captures},
            coverage='All logged pre/post fields match native stored values; dynamic exact post stamps2*n and prior-state pre stamps were checked for every step. Contacts/work unmeasured; other spans unevaluated.')
        doc=dict(format='ovdnext.timeline-evidence',version=1,captures={c['role']:dict(sha256=c['proof'].sha256,
            byteLength=str(c['proof'].byte_length),sdkVersion=c['log']['sdkVersion'],provenance='Explicit CPU-only checked SDK and independent recorder; paired log/tool records in assets',
            phaseMapping=dict(kind='sdk-pre-post',firstPostFrame='2',firstCapturedStep='1',stepSeconds=c['log']['stepSeconds'],
                provenance=f"Verified all {len(c['posts'])} live pre/post steps against native frame scopes, typed values and source stamps")) for c in captures},
            observations=observations,findings=findings,provenance=json.dumps(provenance,separators=(',',':')))
        encoded=json.dumps(doc,allow_nan=False,separators=(',',':')).encode();parsed=parse_evidence(encoded)
        for capture in captures:
            verify_payload(capture['proof'],lambda:False);binding=resolve_bindings(parsed,capture['proof'],capture['loaded'].session)
            applicable={f.id for f in parsed.findings if f.kind=='comparison' or f.data['role']==capture['role']}
            failures={str(key):value for key,value in binding.reasons.items() if value and key[1]==capture['role'] and key[0] in applicable}
            if failures:raise ValueError('Generated evidence failed native binding: '+str(failures))
        with output.open('xb') as stream:stream.write(encoded);stream.flush();os.fsync(stream.fileno())
        (assets/'generation.json').write_text(json.dumps(dict(complete=True,outputSha256=_sha(output),provenance=provenance),indent=2)+'\n')
        return provenance
    except BaseException as error:
        (assets/'generation-failed.json').write_text(json.dumps(dict(complete=False,error=str(error)),indent=2)+'\n')
        raise
    finally:
        for capture in captures:capture['loaded'].close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('left','right','left-log','right-log','inspector','native-root','output'):
        parser.add_argument('--'+name,required=True,type=Path)
    args=parser.parse_args();native=args.native_root.resolve();sys.path.insert(0,str(native))
    suffix='.pyd' if sys.platform=='win32' else '.so'
    spec=importlib.util.find_spec('_pvd_native')
    if not any(native.glob('_pvd_native*'+suffix)) or spec is None or Path(spec.origin).resolve().parent!=native:
        parser.error('Explicit native root must contain the selected built _pvd_native module')
    try:print(json.dumps(generate(args),indent=2))
    except (ValueError,OSError,RuntimeError,KeyError,subprocess.SubprocessError) as error:
        parser.exit(1,'Evidence generation failed: '+str(error)+'\n')


if __name__=='__main__':main()

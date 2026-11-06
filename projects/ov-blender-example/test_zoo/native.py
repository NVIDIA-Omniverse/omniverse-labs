#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Test-only Pillow: taken from the consumer/test environment (scripts/requirements-public-sdk-consumer.txt), never the shipped SDK runtime.
"""Measure compiled zoo generations through the existing typed public SDK client.

No SDK imports or custom worker: --addon-source supplies the real client/runtime.
Pixel changes are recorded alongside identity and noise controls, never inferred
from acknowledged writes. Physics and specialized shading still need their gates.
"""
import argparse
from dataclasses import asdict,dataclass,field
import hashlib
import json
import os
import re
from pathlib import Path
import sys
import time


def authored_mode_evidence(path):
    """Read bounded text-layer opinions without loading USD/NVIDIA into client.

    This is provenance, not a replacement for USD composition or native readback.
    Conflicting token opinions remain unresolved rather than guessing strength.
    """
    pending=[Path(path).resolve()];seen=set();opinions=[];total=0
    while pending:
        layer=pending.pop()
        if layer in seen or layer.suffix not in {'.usd','.usda'}:continue
        seen.add(layer)
        if len(seen)>128:raise ValueError('Mode provenance exceeds128 text layers')
        if not layer.is_file():continue
        total+=layer.stat().st_size
        if total>16*1024**2:raise ValueError('Mode provenance exceeds16MiB')
        with layer.open('rb') as stream:
            if stream.read(10).startswith(b'PXR-USDC'):continue
        content=layer.read_text()
        for token in re.findall(r'\btoken\s+omni:rtx:rendermode\s*=\s*"([^"]+)"',content):
            opinions.append({'layer':str(layer),'token':token})
        for asset in re.findall(r'@([^@]+)@',content):
            target=Path(asset)
            if not target.is_absolute():target=layer.parent/target
            if target.suffix in {'.usd','.usda'}:pending.append(target.resolve())
    tokens=sorted({row['token'] for row in opinions})
    return {'authored_mode':tokens[0] if len(tokens)==1 else None,'opinions':opinions,
            'verification':'consistent text-layer opinions; native mode readback unavailable'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,help='Zoo author/export output root')
    parser.add_argument('--collection',type=Path,help='Collection manifest from collect.py; preserves immutable source runs')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--addon-source',type=Path,required=True)
    parser.add_argument('--python',required=True,help='Existing external public-SDK Python')
    parser.add_argument('--cases',default='mesh.cube,material.base_color,geometry_nodes.realized_instances')
    parser.add_argument('--gpu',type=int,default=2)
    parser.add_argument('--samples',type=int,default=16)
    parser.add_argument('--render-var',choices=('LdrColor','HdrColor'),default='LdrColor')
    args=parser.parse_args()
    if bool(args.input)==bool(args.collection):parser.error('Provide exactly one of --input or --collection')
    sources=({row['case']:Path(row['result']) for row in json.loads(args.collection.read_text())['cases'] if row.get('result')}
             if args.collection else {p.parent.name:p for p in args.input.glob('*/result.json')})
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    os.environ['OV_BLENDER_PUBLIC_SDK_PYTHON']=args.python
    os.environ['OV_BLENDER_PUBLIC_SDK_GPU']=str(args.gpu)
    os.environ['OVRTX_ACTIVE_CUDA_GPUS']=str(args.gpu)
    os.environ['OV_BLENDER_PUBLIC_SDK_OUTPUT']=str(args.output/'workers')
    sys.path.insert(0,str(args.addon_source.resolve()/'addon'))
    from ovrtx_blender_example.direct_sdk_client import PublicSdkRenderClient
    from ovrtx_blender_example.ovrtx_scene_composition import OvrtxSceneComposition
    from ovrtx_blender_example.ovrtx_session import OvrtxSessionSpec
    from ovrtx_blender_example.render_requests import RenderRequest
    from PIL import Image
    import numpy as np
    @dataclass(frozen=True)
    class ZooSpec(OvrtxSessionSpec):
        stage_metadata:dict=field(default_factory=dict)
    def spec_for(step):
        usd=step['usd'];path=Path(usd['composed_scene'])
        if 'stage_metadata' not in usd:
            raise ValueError('Re-export case with current generator to record composed stage metadata')
        width,height=step['blender'].get('size',(512,384))
        return ZooSpec(OvrtxSceneComposition(str(path),str(path),(),hashlib.sha256(path.read_bytes()).hexdigest(),True),
            tuple(usd['sensor_paths']),width,height,usd['camera_path'],'composed_scene',
            render_var=args.render_var,output_size_value_route=True,stage_metadata=usd['stage_metadata'])
    candidates=[]
    for token in args.cases.split(','):
        selected=(sorted(sources) if token=='all' else sorted(key for key in sources if key.startswith(token[:-1]))
                  if token.endswith('.*') else [token])
        candidates.extend(key for key in selected if key not in candidates)
    reports=[]
    for case in candidates:
        directory=args.output/case;directory.mkdir(exist_ok=True)
        report={'case':case,'status':'running','frames':[],'semantic_acceptance':'not_inferred_from_acknowledgment'}
        client=None
        try:
            authored=json.loads(sources[case].read_text())
            report['source_result']=str(sources[case].resolve())
            report['source_result_sha256']=hashlib.sha256(sources[case].read_bytes()).hexdigest()
            if authored['status']!='export_passed':
                raise ValueError(f"Author/export gate is {authored['status']}: {authored.get('error','')}")
            steps=authored['steps']
            request=RenderRequest(dlss_enabled=False)
            client=PublicSdkRenderClient(request)
            spec=spec_for(steps[0])
            report['render_configuration']={'request':asdict(request),
                'translated_settings':client._load_params(spec)['settings'],
                'mode':authored_mode_evidence(steps[0]['usd']['composed_scene']),
                'dlss_note':'False uses the existing addon Performance-preset mapping; it is not a verified DLSS-off switch.',
                'native_mode_selection':'RenderRequest exposes no render-mode field; baseline composer authors RealTimePathTracing.',
                'samples_per_render_call':args.samples,'render_var':args.render_var}
            identity=client.start_session(spec)
            original=client.runtime.request('health',{})
            persistent=(original['_runtime_pid'],original['_runtime_epoch'],original['native_stage_id'])
            report['initial_identity']={'pid':persistent[0],'epoch':persistent[1],'native_stage_id':persistent[2]}
            previous=None;noise=0.
            for index,step in enumerate(steps):
                desired=spec_for(step)
                if index:
                    if not client.can_update_scene(spec,desired):
                        raise RuntimeError('Accepted candidate requires unsupported stage metadata transition')
                    client.update_scene(identity,desired)
                    spec=desired
                started=time.monotonic()
                result=client.render_result(identity,selected_sensor_paths=spec.sensor_paths,
                    render_var=args.render_var,additional_samples=args.samples)
                array=np.frombuffer(result.rgba8,dtype=np.uint8).reshape(result.height,result.width,4).copy()
                if array.shape!=(spec.height,spec.width,4):raise AssertionError('Wrong output shape')
                if len(result.rgba8)!=spec.width*spec.height*4:raise AssertionError('Incomplete owned pixels')
                if args.render_var=='HdrColor':
                    linear=np.frombuffer(result.linear_rgba16f,dtype='<f2')
                    if linear.size!=spec.width*spec.height*4 or not np.isfinite(linear).all():
                        raise AssertionError('Invalid exact HDR output')
                if int(array[:,:,:3].max())-int(array[:,:,:3].min())<3:
                    raise AssertionError('Uniform/empty-looking image requires investigation')
                Image.fromarray(array).transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(directory/f'{index:02d}-{step["operation"]}.png')
                health=client.runtime.request('health',{})
                now=(health['_runtime_pid'],health['_runtime_epoch'],health['native_stage_id'])
                if now!=persistent:raise AssertionError('Edit recreated runtime process/stage')
                mae=(float(np.abs(array.astype(float)-previous.astype(float)).mean())
                     if previous is not None and previous.shape==array.shape else None)
                frame={'operation':step['operation'],'seconds':time.monotonic()-started,
                       'pid':now[0],'epoch':now[1],'native_stage_id':now[2],
                       'width':result.width,'height':result.height,'transport':result.native_timings.get('transport'),
                       'sha256':hashlib.sha256(result.rgba8).hexdigest(),'pixel_mae_from_previous':mae,
                       'changed_above_noise':mae>max(.1,3*noise) if mae is not None else None}
                if index==0:
                    repeat=client.render_result(identity,selected_sensor_paths=spec.sensor_paths,
                        render_var=args.render_var,additional_samples=args.samples)
                    stable=np.frombuffer(repeat.rgba8,dtype=np.uint8).reshape(array.shape)
                    noise=float(np.abs(stable.astype(float)-array.astype(float)).mean())
                    report['no_edit_pixel_mae']=noise
                    Image.fromarray(stable).transpose(Image.Transpose.FLIP_TOP_BOTTOM).save(directory/'no-edit-control.png')
                report['frames'].append(frame);previous=array
                (directory/'result.json').write_text(json.dumps(report,indent=2)+'\n')
            report['status']='native_frames_measured'
            report['same_session_for_all_edits']=True
            report['semantic_acceptance']='Per-feature BRDF/physics/primvar gates still required; images and differences retained'
            report['runtime_diagnostics']=client.runtime.diagnostics()
        except Exception as error:
            report.update(status='gap',error=str(error),error_class=type(error).__name__)
        finally:
            if client is not None:
                client.shutdown()
                report['cleanup_diagnostics']=client.runtime.diagnostics()
        (directory/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        reports.append(report)
        (args.output/'summary.json').write_text(json.dumps({'schema':1,'results':reports},indent=2)+'\n')
        print(case+': '+report['status'],flush=True)


if __name__=='__main__':main()

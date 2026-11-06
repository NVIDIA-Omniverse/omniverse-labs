#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run one serialized stock-Windows render-output probe in an isolated profile."""
import argparse
import array
import ast
import struct
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import time



def producer_pids(value):
    found=set()
    if isinstance(value,dict):
        for key,item in value.items():
            if key in ('runtime_pid','launcher_pid') and type(item) is int and item>0:found.add(item)
            found.update(producer_pids(item))
    elif isinstance(value,list):
        for item in value:found.update(producer_pids(item))
    return found


def process_running(pid):
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=(wintypes.DWORD,wintypes.BOOL,wintypes.DWORD)
    kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes=(wintypes.HANDLE,wintypes.DWORD)
    kernel.WaitForSingleObject.restype=wintypes.DWORD
    kernel.CloseHandle.argtypes=(wintypes.HANDLE,)
    handle=kernel.OpenProcess(0x00100000,False,pid)
    if not handle:
        error=ctypes.get_last_error()
        if error==87:return False
        raise ctypes.WinError(error)
    try:
        status=kernel.WaitForSingleObject(handle,0)
        if status not in (0,258):raise ctypes.WinError(ctypes.get_last_error())
        return status==258
    finally:kernel.CloseHandle(handle)



def read_float32_npy(path):
    with Path(path).open('rb') as stream:
        assert stream.read(6)==b'\x93NUMPY'
        major,minor=stream.read(2)
        size=struct.unpack('<H' if major==1 else '<I',stream.read(2 if major==1 else 4))[0]
        header=ast.literal_eval(stream.read(size).decode('latin1'))
        assert header['descr']=='<f4' and not header['fortran_order'],header
        values=array.array('f');values.frombytes(stream.read())
        return header['shape'],values


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--kind',choices=('final','viewport','invoke'),required=True)
    parser.add_argument('--invoke-mode',choices=('cold_still','cold_animation','viewport_still','exact_stage_viewport_still'))
    parser.add_argument('--viewport-reference',type=Path)
    parser.add_argument('--baseline',action='store_true',help='Calibrate the same fixture with the retained repaired SIL baseline')
    parser.add_argument('--single-baseline-run',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--reference',type=Path,help='Passing matched baseline result.json, frozen before candidate evaluation')
    parser.add_argument('--dlss-reference',type=Path,help='Pinned retained-baseline DLSS policy result for full viewport qualification')
    args=parser.parse_args()
    if sys.platform!='win32':raise SystemExit('Windows execution only; Linux qualification is deferred')
    repo=args.repo.resolve()
    source=Path(__file__).resolve().parents[1]
    out=args.output.resolve()
    out.relative_to(repo)
    out.mkdir(parents=True,exist_ok=False)
    blender=repo/'_build/lts-validation-5.2.1/install/blender-5.2.1-windows-x64/blender.exe'
    python=repo/'_build/blender-ovrtx-ovstream/runtime-lock/env/Scripts/python.exe'
    env=dict(os.environ)
    for key in tuple(env):
        if key.startswith(('OV_BLENDER','OVRTX_','OV_OUTPUT_')) or key in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV'):
            env.pop(key)
    for key in ('BLENDER_USER_CONFIG','BLENDER_USER_SCRIPTS','BLENDER_USER_DATAFILES',
                'OPTIX_CACHE_PATH','CUDA_CACHE_PATH','WARP_CACHE_PATH','LOCALAPPDATA','APPDATA','TEMP','TMP'):
        directory=out/'environment'/key
        directory.mkdir(parents=True)
        env[key]=str(directory)
    env['PYTHONDONTWRITEBYTECODE']='1'
    env.update(OV_BLENDER_PUBLIC_SDK_OUTPUT=str(out/'workers'),
               OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR=str(out/'generations'),
               OMNICLIENT_HUB_MODE='disabled')
    baseline_worker=None
    baseline_config=None
    baseline_config_sha256=None
    if args.kind=='invoke' and (args.baseline or not args.invoke_mode or not args.viewport_reference):
        raise ValueError('INVOKE acceptance requires candidate mode and frozen viewport reference')
    if args.baseline:
        retained=repo/'_build/blender-ovrtx-ovstream/source-retention/baseline-repair/vendor/ov-blender-example-internal/public/addon'
        runtime=repo/'_build/official-addon/runtime-store/runtimes/windows-x64/current'
        baseline_worker=repo/'_build/blender-ovrtx-ovstream/camera/native-build-vs2026/bin/ovrtx-bridge-server.exe'
        # The retained addon authors startup Carb settings into package-root.
        # Give it an independent package, never modify the retained comparator.
        source_package=runtime/'runtime/ovrtx-bridge-server'
        baseline_config=source_package/'ovrtx.config.json'
        baseline_config_sha256=hashlib.sha256(baseline_config.read_bytes()).hexdigest()
        package=out/'baseline-package'
        shutil.copytree(source_package,package)
        with socket.socket() as listener:
            listener.bind(('127.0.0.1',0));port=listener.getsockname()[1]
        env.update(OV_OUTPUT_ADDON_ROOT=str(retained),
            OV_BLENDER_EXAMPLE_BUNDLED_ADDON_ROOT=str(runtime),
            OV_BLENDER_EXAMPLE_NATIVE_CLIENT_PATH=str(runtime/'native/client'),
            OV_BLENDER_EXAMPLE_NATIVE_CLIENT_MODULE='ovrtx_bridge_client',
            OV_BLENDER_EXAMPLE_WORKER_COMMAND=subprocess.list2cmdline([
                str(baseline_worker),'--address','127.0.0.1','--port',str(port),
                '--package-root',str(package)]),
            OV_BLENDER_EXAMPLE_WORKER_LOG=str(out/'baseline-worker.log'),
            OVRTX_WORKER_RENDERER_LOG=str(out/'baseline-renderer.log'),OVRTX_ACTIVE_CUDA_GPUS='0')
    else:
        if args.reference is None:raise ValueError('Candidate qualification requires --reference from matched baseline calibration')
        env['OV_BLENDER_PUBLIC_SDK_PYTHON']=str(python)
    script=source/'tests'/dict(final='blender_final_output_pixels.py',viewport='blender_viewport_output_pixels.py',invoke='blender_invoke_output_pixels.py')[args.kind]
    command=[str(blender),'--factory-startup','--python-exit-code','17']
    if args.kind=='final':command+=['--background']
    command+=['--python',str(script),'--','--output',str(out/'proof')]
    command+=['--calibrate-baseline'] if args.baseline else ['--reference',str(args.reference.resolve())]
    if args.kind=='viewport' and not args.baseline:
        if args.dlss_reference is None:raise ValueError('Viewport qualification requires --dlss-reference')
        command+=['--dlss-reference',str(args.dlss_reference.resolve())]
    if args.kind=='invoke':
        command+=['--repo',str(repo),'--mode',args.invoke_mode,'--viewport-reference',str(args.viewport_reference.resolve())]
    git_root=source.parents[1]
    git=['git','-c',f'safe.directory={git_root.as_posix()}','-C',str(git_root)]
    dirty=subprocess.check_output([*git,'diff','--binary'])
    receipt={'source_commit':subprocess.check_output([*git,'rev-parse','HEAD'],text=True).strip(),
        'dirty_diff_sha256':hashlib.sha256(dirty).hexdigest(),
        'runtime_lock_sha256':hashlib.sha256((source/'runtime/public-sdk-lock.json').read_bytes()).hexdigest(),
        'fixture_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),
        'os':platform.platform(),'runner_python':sys.version,'command':command,
        'environment_overrides':{k:v for k,v in env.items() if os.environ.get(k)!=v},
        'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True)}
    if baseline_worker:
        receipt['baseline_worker_sha256']=hashlib.sha256(baseline_worker.read_bytes()).hexdigest()
        receipt['baseline_config_source_sha256']=baseline_config_sha256
        receipt['baseline_package_copy']=str(package)
        receipt['baseline_addon_pin']='365a167c07b3f2670c68d275c943b19fce5d5b6c'
    if args.kind=='invoke':
        shutil.copy2(script,out/'fixture_snapshot.py')
        shutil.copy2(Path(__file__),out/'runner_snapshot.py')
        receipt['source_before']={str(q.relative_to(source)):hashlib.sha256(q.read_bytes()).hexdigest() for q in (source/'addon').rglob('*.py')}
    receipt_path=out/'receipt.json'
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    started=time.monotonic()
    with (out/'blender.log').open('w',encoding='utf-8') as log:
        process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        receipt['blender_pid']=process.pid
        receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
        try:
            returncode=process.wait(timeout=480 if args.kind=='invoke' else 2400)
        except subprocess.TimeoutExpired:
            if args.kind!='invoke':raise
            receipt['watchdog_timeout_seconds']=480
            process.kill()  # Only the Blender Popen owned by this acceptance run.
            returncode=process.wait(timeout=15)
    receipt.update(exit_code=returncode,elapsed_seconds=time.monotonic()-started)
    if args.kind=='invoke':
        source_after={str(q.relative_to(source)):hashlib.sha256(q.read_bytes()).hexdigest() for q in (source/'addon').rglob('*.py')}
        receipt['source_changed']=[key for key in receipt['source_before'] if receipt['source_before'][key]!=source_after.get(key)]
    result_path=out/'proof/result.json'
    receipt['result']=json.loads(result_path.read_text()) if result_path.is_file() else None
    if baseline_config is not None:
        assert hashlib.sha256(baseline_config.read_bytes()).hexdigest()==baseline_config_sha256,'Retained baseline config changed'
    observed_pids=producer_pids(receipt['result'])
    deadline=time.monotonic()+15
    alive=set(observed_pids)
    while alive:
        alive={pid for pid in alive if process_running(pid)}
        if not alive or time.monotonic()>=deadline:break
        time.sleep(.1)
    receipt['observed_producer_pids']=sorted(observed_pids)
    receipt['surviving_producer_pids']=sorted(alive)
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    if alive:raise SystemExit(f'Owned producer processes survived Blender exit: {sorted(alive)}')
    if not args.baseline and receipt['result'] and receipt['result']['status']=='passed':
        assert observed_pids,'Candidate acceptance must observe actual external worker identity'
    if returncode or receipt.get('source_changed') or not receipt['result'] or receipt['result']['status']!='passed':
        raise SystemExit(f'Render output {args.kind} failed; inspect {receipt_path}')
    if args.baseline and args.kind=='viewport' and not args.single_baseline_run:
        repeat=out/'independent-repeat'
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--repo',str(repo),
            '--output',str(repeat),'--kind','viewport','--baseline','--single-baseline-run'],check=True)
        repeated_receipt=json.loads((repeat/'receipt.json').read_text())
        assert repeated_receipt['blender_pid']!=receipt['blender_pid']
        repeated=repeated_receipt['result']
        current=receipt['result']
        current['held_frame_thresholds']=dict(current['thresholds'])
        measurements=[]
        for mode,name in (('HdrColor','initial_hdr'),('LdrColor','ldr')):
            first_case=next(c for c in current['cases'] if c['name']==name)
            second_case=next(c for c in repeated['cases'] if c['name']==name)
            shape,first_pixels=read_float32_npy(first_case['views'][0]['raw_path'])
            second_shape,second_pixels=read_float32_npy(second_case['views'][0]['raw_path'])
            assert shape==second_shape and len(first_pixels)==len(second_pixels)
            mae=sum(abs(a-b) for a,b in zip(first_pixels,second_pixels))/len(first_pixels)
            current['thresholds'][mode]=max(.005,3*mae)
            measurements.append({'mode':mode,'mae':mae,'frozen_limit':current['thresholds'][mode]})
        current['independent_repeatability']={'blender_pids':[receipt['blender_pid'],repeated_receipt['blender_pid']],
            'repeat_receipt':str(repeat/'receipt.json'),'repeat_receipt_sha256':hashlib.sha256((repeat/'receipt.json').read_bytes()).hexdigest(),'measurements':measurements}
        current['threshold_rationale']='3x actual fresh-process matched baseline MAE, minimum0.005, LDRnormalized0..1; frozen before candidate. Held captures only measure stability.'
        result_path.write_text(json.dumps(current,indent=2)+'\n')
        receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'status':'passed','kind':args.kind,'receipt':str(receipt_path)}))


if __name__=='__main__':main()

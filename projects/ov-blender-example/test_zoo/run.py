#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Serial, bounded Blender zoo author/export runner. Never labels native output passed."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from catalog import BY_ID, SMOKE, manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blender',type=Path)
    parser.add_argument('--addon-source',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[2]/'out/test-zoo')
    parser.add_argument('--cases',default='smoke',help='smoke, all, family.*, or comma-separated case IDs')
    parser.add_argument('--list',action='store_true')
    parser.add_argument('--preview',action='store_true',help='Render labeled Workbench authoring previews; not native evidence')
    parser.add_argument('--edits',action='store_true',help='Apply and export each case edit sequence')
    parser.add_argument('--timeout',type=int,default=120)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.list:
        print(json.dumps(manifest(),indent=2)); return
    if args.blender is None or not args.blender.is_file():
        parser.error('--blender must identify an existing Blender5.2 executable')
    selected=[]
    for token in args.cases.split(','):
        ids=(list(BY_ID) if token=='all' else list(SMOKE) if token=='smoke' else
             [key for key in BY_ID if key.startswith(token[:-1])] if token.endswith('.*') else [token])
        for key in ids:
            if key not in BY_ID: parser.error(f'Unknown case: {key}')
            if key not in selected: selected.append(key)
    args.output=args.output.resolve(); args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'catalog.json').write_text(json.dumps(manifest(),indent=2)+'\n')
    results=[]
    for key in selected:
        case_dir=args.output/key; case_dir.mkdir(exist_ok=True)
        result_path=case_dir/'result.json'
        if args.resume and result_path.exists():
            results.append(json.loads(result_path.read_text())); continue
        used=sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())
        if used > 1024**3: raise RuntimeError('Generated output reached1GiB budget')
        command=[str(args.blender),'-b','-noaudio','--factory-startup','--threads','2','--python',
                 str(Path(__file__).with_name('blender_generate.py')),'--','--case',key,
                 '--output',str(case_dir),'--addon-source',str(args.addon_source.resolve())]
        if args.preview: command.append('--preview')
        if args.edits: command.append('--edits')
        started=time.monotonic()
        try:
            with (case_dir/'blender.log').open('w') as log:
                environment=os.environ.copy();environment['XDG_CACHE_HOME']=str(args.output/'_cache')
                environment['BLENDER_USER_CONFIG']=str(args.output/'_config')
                environment['OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR']=str(case_dir/'composition')
                (case_dir/'temporary').mkdir(exist_ok=True)
                environment['TMPDIR']=str(case_dir/'temporary')
                completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=args.timeout,env=environment)
            result=json.loads(result_path.read_text()) if result_path.exists() else {
                'case':key,'status':'failed','error':'Blender did not produce case report'}
            result['exit_code']=completed.returncode
        except subprocess.TimeoutExpired:
            result={'case':key,'status':'blocked','error':'Blender case exceeded bounded timeout'}
        result['elapsed_seconds']=time.monotonic()-started
        result_path.write_text(json.dumps(result,indent=2)+'\n')
        results.append(result)
        (args.output/'summary.json').write_text(json.dumps({'schema':1,'results':results,
            'counts':dict(Counter(r['status'] for r in results)),
            'native_validation':'not_run; export is not renderer acceptance'},indent=2)+'\n')
        print(f"{key}: {result['status']}",flush=True)
    print(json.dumps(dict(Counter(r['status'] for r in results))))


if __name__=='__main__': main()

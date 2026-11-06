#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Index immutable zoo runs; later sources supersede earlier case reports."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from catalog import CASES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--assets',type=Path)
    args=parser.parse_args(); rows=[]
    for case in CASES:
        selected=None
        for source in args.sources:
            path=source.resolve()/case.id/'result.json'
            if path.is_file():selected=path
        row={'case':case.id,'family':case.family,'classification':case.classification,
             'status':'not_run','native_status':'not_run'}
        if selected:
            data=json.loads(selected.read_text()); defects=[]
            for step in data.get('steps',[]):
                usd=step.get('usd',{})
                if 'stage_metadata' not in usd:defects.append('missing_stage_metadata')
                composed=Path(usd.get('composed_scene',''))
                if not composed.is_file():defects.append('missing_composition')
                elif not composed.is_relative_to(selected.parent):defects.append('composition_outside_case')
            error=data.get('error')
            if not error and data['status']=='conversion_rejected':
                error=next((s.get('usd',{}).get('material_conversion_error') for s in data.get('steps',[])
                            if s.get('usd',{}).get('material_conversion_error')),None)
            row.update(status=data['status'],result=str(selected),
                       result_sha256=hashlib.sha256(selected.read_bytes()).hexdigest(),
                       provenance_defects=sorted(set(defects)),error=error)
            row['native_ready']=data['status']=='export_passed' and not defects
        rows.append(row)
    files=sorted(Path(__file__).parent.glob('*.py'))
    report={'schema':1,'cases':rows,'counts':dict(Counter(r['status'] for r in rows)),
            'native_ready':sum(r.get('native_ready',False) for r in rows),
            'source_runs':[str(p.resolve()) for p in args.sources],
            'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            'acceptance':'Author/export and native image measurements are separate gates.'}
    if args.assets:
        report['assets_manifest']=str(args.assets.resolve())
        report['assets_manifest_sha256']=hashlib.sha256(args.assets.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'cases':len(rows),'counts':report['counts'],'native_ready':report['native_ready']}))


if __name__=='__main__':main()

"""Combine bounded body-control shards and rerun every cross-case assertion."""
import argparse
import json
from pathlib import Path
from public_sdk_physics_evidence import PUBLIC, body_contract, digest, validate_directory

def aggregate(shards, output, device, mode):
    metrics={}; identity=None; metadata=None; seen=set(); receipts=[]
    for directory in map(Path,shards):
        row=validate_directory(directory,'body_controls',device,mode,partial=True)
        if identity is not None and row['metadata']['identity']!=identity: raise ValueError('Mixed adapter/native identities')
        if metadata is not None and row['metadata'].get('local_manifest_override')!=metadata.get('local_manifest_override'):raise ValueError('Mixed qualification manifests')
        if row['process_key'] in seen or set(metrics)&set(row['result']['metrics']): raise ValueError('Duplicate case or process')
        seen.add(row['process_key']);metrics.update(row['result']['metrics'])
        identity=row['metadata']['identity'];metadata=row['metadata']
        receipts.append({'directory':str(directory.resolve()),'acceptance_sha256':digest(directory/'acceptance.json')})
    checks=body_contract().assert_metrics(metrics)
    result={'device':device,'checks':checks,'metrics':metrics,'qualification_mode':mode}
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    (output/'runtime-identity.json').write_text(json.dumps(metadata,indent=2)+'\n')
    accepted=dict(row['acceptance']);accepted.pop('process_sha256',None)
    accepted.update(results_sha256=digest(output/'results.json'),assertions_sha256=digest(PUBLIC/'tests/body_controls_assertions.py'),shards=receipts)
    (output/'acceptance.json').write_text(json.dumps(accepted,indent=2)+'\n')
    validate_directory(output,'body_controls',device,mode)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--shards',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','gpu'),required=True);p.add_argument('--qualification-mode',choices=('measurement','candidate','production'),required=True)
    a=p.parse_args();aggregate(a.shards,a.output,a.device,a.qualification_mode)

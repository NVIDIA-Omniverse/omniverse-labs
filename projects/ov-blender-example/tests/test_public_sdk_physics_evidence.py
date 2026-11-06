"""Admission rejects incomplete, stale, mixed and falsely labeled qualification."""
import copy
from datetime import datetime,timezone,timedelta
import json
import math
from pathlib import Path
import sys
import pytest

PUBLIC=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PUBLIC/'scripts'))
from public_sdk_physics_evidence import body_contract,digest,identity_hash,product_source_hash,validate_directory,validate_metadata,executed_sources
from aggregate_public_sdk_body_controls import aggregate

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))

def valid_metrics():
    result={name:dict(position=[0,0,2],rotation=[0,0,0,1],angle_x=.1,angle_z=0,rebound_height=.2,at_24=[0,0,1]) for name in body_contract().CASE_NAMES}
    for name in ('free','free_cm'):result[name].update(position=[4,0,2],angle_z=2)
    for name in ('box','box_cm','capsule'):result[name]['position']=[0,0,.5]
    for name in ('hinge_mass2','hinge_density','hinge_material_density'):result[name]['angle_x']=.2
    for name in ('hinge_inertia2','hinge_principal'):result[name]['angle_x']=.05
    result['hinge_com']['angle_x']=.15
    result['friction_low']['position'][0]=2;result['friction_high']['position'][0]=.2
    result['restitution_high']['rebound_height']=.8
    result['gravity_half']['at_24'][2]=1.1;result['gravity_x']['at_24']=[.2,0,2]
    result['static_high']['position'][0]=.5*math.sin(math.radians(15))
    result['static_low']['at_24'][0]=.03
    return result

def metadata(identity,device='cpu',mode='measurement'):
    return {'identity':identity,'device':device,'qualification_mode':mode,'measurement_only':mode=='measurement',
            'local_manifest_override':'candidate.json' if mode=='candidate' else None,
            'admission_checks':[{'status':'measurement-only' if mode=='measurement' else mode,'local_override':mode=='candidate','device':device,'identity_sha256':identity_hash(identity)}]}

def valid_family_result(family,device):
    if family=='parented':
        return {'device':device,'metrics':{**{'ic_'+phase+'_first':[8,0,6] for phase in ('start','reset','mid')},'kinematic_rider_last':[3.5,2,3]},'checks':['initial condition at '+phase+' lands at the requested world pose' for phase in ('start','reset','mid')]}
    if family=='bones':
        names=[device+'-1.0',device+'-0.01']
        return {'device':device,'metrics':[{'case':name,'root_error_m':0,'moving_displacement_m':[.1,.2]} for name in names],
                'checks':[name+' '+suffix for name in names for suffix in ('fixed transformed root under gravity','same and mixed mechanism real motion','isolated and full physics trajectories agree','renamed label resolves same stable identity')]}
    names=['fixed-x-limit','revolute-x-limit','revolute-y-limit','prismatic-x-limit','prismatic-z-drive','d6-y-limit','d6-z-drive','static_anchor-x-limit'];checks=[]
    for name in names:
        checks.extend(name+' '+suffix for suffix in ('complete finite poses','input unchanged','all compiled source layers unchanged','no authoring pose writeback'))
        suffixes=('gravity attachment',) if name.startswith('fixed') else ('swings under gravity','static anchor holds joint radius') if name.startswith('static_anchor') else ('moves under gravity','angular limit') if name.startswith('revolute') else ('configured axis','linear limits','loaded drive' if name.endswith('drive') else 'gravity translation')
        checks.extend(name+' '+suffix for suffix in suffixes)
    return {'device':device,'status':'pass','metrics':[{'case':name,'steps':360} for name in names],'checks':checks}

def bind(directory,family,identity,device='cpu',mode='measurement',pid=123):
    from public_sdk_physics_evidence import SCRIPTS
    meta=metadata(identity,device,mode)
    if mode=='candidate':
        manifest=directory.parent/'candidate.json';write(manifest,{'schema_version':1,'measurements':[]});meta['local_manifest_override']=str(manifest)
    else:manifest=PUBLIC/'addon/ovrtx_blender_example/public_sdk_physics_admission.json'
    if mode!='measurement':meta['admission_checks'][0]['manifest_path']=str(manifest)
    write(directory/'runtime-identity.json',meta)
    now=datetime.now(timezone.utc)
    process={'pid':pid,'exit_code':0,'terminal_exit_code':0,'natural_job_active':0,'guard_closed':True,'terminal_confirmed':True,
             'forced_cleanup':False,'cleanup_errors':[],'started_utc':(now-timedelta(seconds=1)).isoformat(),'ended_utc':(now+timedelta(seconds=1)).isoformat()}
    process['command']=['blender','--python',str(PUBLIC/'tests'/SCRIPTS[family]),'--','--output',str(directory),'--device',device,'--qualification-mode',mode]
    write(directory/'process.json',process)
    inputs={'source':str(PUBLIC),'source_files_sha256':executed_sources(),'command':process['command'],'device':device,'qualification_mode':mode}
    if mode=='candidate':inputs.update(manifest_path=str(manifest),manifest_sha256=digest(manifest))
    write(directory/'inputs.json',inputs)
    accepted={'family':family,'device':device,'qualification_mode':mode,'measurement_only':mode=='measurement','identity_sha256':identity_hash(identity),
              'fixture_sha256':digest(PUBLIC/'tests'/SCRIPTS[family]),'results_sha256':digest(directory/'results.json'),
              'product_source_sha256':product_source_hash(),'process_sha256':digest(directory/'process.json'),'inputs_sha256':digest(directory/'inputs.json')}
    if family=='body_controls':accepted['assertions_sha256']=digest(PUBLIC/'tests/body_controls_assertions.py')
    if family=='bake_contact':
        process=copy.deepcopy(process);process['command']+=['--offline']
        write(directory/'offline-process.json',process)
        inputs['command']=process['command'];write(directory/'offline-inputs.json',inputs)
        accepted.update(offline_results_sha256=digest(directory/'offline-results.json'),offline_process_sha256=digest(directory/'offline-process.json'))
        accepted['offline-inputs_sha256']=digest(directory/'offline-inputs.json')
    write(directory/'acceptance.json',accepted)

@pytest.fixture
def shards(tmp_path):
    identity={'adapter_sha256':'b'*64,'native_sha256':{'physics.dll':'a'*64}}
    values=list(valid_metrics().items());paths=[]
    for i in range(2):
        d=tmp_path/str(i);write(d/'results.json',{'device':'cpu','metrics':dict(values[i*13:(i+1)*13])})
        bind(d,'body_controls',identity,pid=100+i);paths.append(d)
    return paths

def test_aggregate_rechecks_all_twenty_numerical_assertions(shards,tmp_path):
    result=aggregate(shards,tmp_path/'combined','cpu','measurement')
    assert len(result['metrics'])==26 and len(result['checks'])==20
    validate_directory(tmp_path/'combined','body_controls','cpu','measurement')

@pytest.mark.parametrize('failure',['missing','duplicate','hull_cm','static','nan','source','identity','device','mode','cleanup','timeout','stale','changed','inputs_hash','helper_snapshot','command'])
def test_shards_reject_incomplete_or_untrustworthy_evidence(shards,tmp_path,failure):
    path=shards[1];r=json.loads((path/'results.json').read_text());m=json.loads((path/'runtime-identity.json').read_text())
    if failure=='missing':r['metrics'].pop('hinge_hull')
    if failure=='duplicate':r['metrics']['free']=valid_metrics()['free']
    if failure=='hull_cm':r['metrics']['hinge_hull_cm']['angle_x']+=.1
    if failure=='static':r['metrics']['static_low']['at_24'][0]=1
    if failure=='nan':r['metrics']['hinge_hull']['position'][0]=float('nan')
    if failure=='identity':m['identity']['adapter_sha256']='c'*64
    if failure=='device':m['device']='gpu'
    if failure=='mode':m['admission_checks'][0]['status']='production'
    write(path/'results.json',r);write(path/'runtime-identity.json',m)
    accepted=json.loads((path/'acceptance.json').read_text());accepted['results_sha256']=digest(path/'results.json');accepted['identity_sha256']=identity_hash(m['identity'])
    if failure=='source':accepted['product_source_sha256']='0'*64
    process=json.loads((path/'process.json').read_text())
    if failure=='cleanup':process['forced_cleanup']=True
    if failure=='timeout':process['started_utc']=(datetime.now(timezone.utc)-timedelta(seconds=700)).isoformat()
    if failure=='stale':process['ended_utc']=(datetime.now(timezone.utc)-timedelta(seconds=50)).isoformat();process['started_utc']=(datetime.now(timezone.utc)-timedelta(seconds=60)).isoformat()
    write(path/'process.json',process);accepted['process_sha256']=digest(path/'process.json')
    if failure=='changed':accepted['results_sha256']='0'*64
    inputs=json.loads((path/'inputs.json').read_text())
    if failure=='helper_snapshot':inputs['source_files_sha256']['tests/public_sdk_physics_fixture.py']='0'*64
    if failure=='command':inputs['command']=['python','invented.py']
    write(path/'inputs.json',inputs);accepted['inputs_sha256']=digest(path/'inputs.json')
    if failure=='inputs_hash':accepted['inputs_sha256']='0'*64
    write(path/'acceptance.json',accepted)
    with pytest.raises((ValueError,AssertionError)):aggregate(shards,tmp_path/'combined','cpu','measurement')

def test_candidate_cannot_claim_production(shards):
    row=json.loads((shards[0]/'runtime-identity.json').read_text())
    row.update(qualification_mode='production',measurement_only=False)
    with pytest.raises(ValueError,match='Actual admission'):validate_metadata(row,'cpu','production')

@pytest.mark.parametrize('failure',['device','identity','manifest'])
def test_candidate_pinned_proof_matches_actual_admission(tmp_path,failure):
    identity={'adapter_sha256':'b'*64,'native_sha256':{'native.dll':'a'*64}}
    d=tmp_path/'cell';write(d/'results.json',{'device':'cpu','metrics':valid_metrics()});bind(d,'body_controls',identity,mode='candidate')
    if failure in ('device','identity'):
        m=json.loads((d/'runtime-identity.json').read_text())
        m['admission_checks'][0]['device' if failure=='device' else 'identity_sha256']='gpu' if failure=='device' else '0'*64
        write(d/'runtime-identity.json',m)
    else:
        inputs=json.loads((d/'inputs.json').read_text());other=tmp_path/'other.json';write(other,{'different':True})
        inputs.update(manifest_path=str(other),manifest_sha256=digest(other));write(d/'inputs.json',inputs)
        accepted=json.loads((d/'acceptance.json').read_text());accepted['inputs_sha256']=digest(d/'inputs.json');write(d/'acceptance.json',accepted)
    with pytest.raises(ValueError):validate_directory(d,'body_controls','cpu','candidate')

def _rekey_inputs(directory,separator,tamper=False):
    inputs=json.loads((directory/'inputs.json').read_text())
    files={k.replace('\\','/').replace('/',separator):v for k,v in inputs['source_files_sha256'].items()}
    if tamper:files[next(iter(files))]='0'*64
    inputs['source_files_sha256']=files;write(directory/'inputs.json',inputs)
    accepted=json.loads((directory/'acceptance.json').read_text());accepted['inputs_sha256']=digest(directory/'inputs.json')
    write(directory/'acceptance.json',accepted)

def test_executed_source_snapshot_is_posix_keyed():
    files=executed_sources()
    assert files and all('\\' not in key for key in files) and 'scripts/probe_public_sdk_mass_inertia.py' in files

@pytest.mark.parametrize('separator',['/','\\'])
def test_windows_and_linux_recorded_snapshots_validate_on_either_host(tmp_path,separator):
    identity={'adapter_sha256':'b'*64,'native_sha256':{'physics.so':'a'*64}}
    d=tmp_path/'cell';write(d/'results.json',{'device':'cpu','metrics':valid_metrics()});bind(d,'body_controls',identity)
    _rekey_inputs(d,separator)
    validate_directory(d,'body_controls','cpu','measurement')

@pytest.mark.parametrize('separator',['/','\\'])
def test_portable_snapshot_still_rejects_changed_digest(tmp_path,separator):
    identity={'adapter_sha256':'b'*64,'native_sha256':{'physics.so':'a'*64}}
    d=tmp_path/'cell';write(d/'results.json',{'device':'cpu','metrics':valid_metrics()});bind(d,'body_controls',identity)
    _rekey_inputs(d,separator,tamper=True)
    with pytest.raises(ValueError,match='snapshot'):validate_directory(d,'body_controls','cpu','measurement')

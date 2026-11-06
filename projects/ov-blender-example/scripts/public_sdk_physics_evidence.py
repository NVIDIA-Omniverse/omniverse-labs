"""Fail-closed provenance checks shared by bounded qualification and recording."""
import hashlib
import importlib.util
import json
import math
from datetime import datetime
from pathlib import Path

PUBLIC = Path(__file__).resolve().parents[1]
SCRIPTS = {'body_controls':'blender_native_body_controls.py','joints':'blender_native_joints.py',
           'kinematic':'blender_native_kinematic.py','parented':'blender_native_parented_poses.py',
           'bones':'blender_bone_mechanisms.py','bake_contact':'blender_bake_contact_standalone.py',
           'live_mass':'blender_public_physics_generation_reuse.py'}

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def identity_hash(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

def product_source_hash():
    return identity_hash({p.relative_to(PUBLIC).as_posix():digest(p)
                          for folder in ('addon','runtime') for p in (PUBLIC/folder).rglob('*.py')})

def executed_sources():
    paths=[p for folder in ('addon','runtime') for p in (PUBLIC/folder).rglob('*.py')]
    paths.extend(p for p in (PUBLIC/'tests').glob('*.py') if not p.name.startswith('test_') and p.name!='conftest.py')
    paths.append(PUBLIC/'scripts/probe_public_sdk_mass_inertia.py')
    return {p.relative_to(PUBLIC).as_posix():digest(p) for p in paths}

def portable_source_snapshot(files):
    """Key a recorded snapshot by POSIX path, so Windows and Linux records compare equal."""
    return {str(k).replace('\\','/'):v for k,v in files.items()} if isinstance(files,dict) else None

def validate_inputs(directory,accepted,process,family,device,mode,*,offline=False):
    prefix='offline-' if offline else ''
    path=directory/(prefix+'inputs.json')
    if accepted.get(prefix+'inputs_sha256')!=digest(path):raise ValueError('Changed execution inputs')
    inputs=read(path)
    if portable_source_snapshot(inputs.get('source_files_sha256'))!=executed_sources():raise ValueError('Executed source/helper snapshot changed or incomplete')
    if inputs.get('command')!=process.get('command') or inputs.get('device')!=device or inputs.get('qualification_mode')!=mode:
        raise ValueError('Owned process does not match recorded execution inputs')
    if Path(inputs.get('source','')).resolve()!=PUBLIC.resolve():raise ValueError('Recorded source path differs from verified source')
    if mode=='candidate':
        manifest=Path(inputs.get('manifest_path',''))
        if not manifest.is_file() or inputs.get('manifest_sha256')!=digest(manifest):raise ValueError('Candidate manifest changed after execution')
        if manifest.resolve()!=Path(read(directory/'runtime-identity.json')['local_manifest_override']).resolve():raise ValueError('Pinned candidate differs from actual admission manifest')

def body_contract():
    path = PUBLIC/'tests/body_controls_assertions.py'
    spec = importlib.util.spec_from_file_location('body_controls_contract',path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def validate_metadata(metadata, device, mode=None):
    mode = mode or metadata.get('qualification_mode')
    if mode not in ('measurement','candidate','production') or metadata.get('qualification_mode') != mode:
        raise ValueError('Missing or mixed qualification mode')
    if metadata.get('device') != device or metadata.get('measurement_only') is not (mode == 'measurement'):
        raise ValueError('Wrong qualification device or measurement bypass')
    identity = metadata.get('identity')
    if not isinstance(identity,dict) or not identity.get('adapter_sha256') or not identity.get('native_sha256'):
        raise ValueError('Missing exact adapter/native identity')
    hashes = [identity['adapter_sha256'], *identity['native_sha256'].values()]
    if any(not isinstance(v,str) or len(v)!=64 or any(c not in '0123456789abcdef' for c in v) for v in hashes):
        raise ValueError('Malformed adapter/native hashes')
    override = mode == 'candidate'
    if bool(metadata.get('local_manifest_override')) != override:
        raise ValueError('Qualification manifest override does not match mode')
    checks = metadata.get('admission_checks')
    expected = 'measurement-only' if mode == 'measurement' else mode
    if not checks or any(c.get('status') != expected or c.get('local_override') is not override for c in checks):
        raise ValueError('Actual admission status does not match qualification mode')
    if any(c.get('device')!=device for c in checks):raise ValueError('Actual admission device differs from metadata')
    if mode!='measurement':
        manifest=Path(metadata['local_manifest_override']) if override else PUBLIC/'addon/ovrtx_blender_example/public_sdk_physics_admission.json'
        if any(Path(c.get('manifest_path','')).resolve()!=manifest.resolve() for c in checks):raise ValueError('Admission did not use the selected manifest')
        if any(c.get('identity_sha256')!=identity_hash(identity) for c in checks):raise ValueError('Actual admission identity differs from metadata')
    return identity

def validate_process(process, result_path):
    if (process.get('exit_code') != 0 or process.get('terminal_exit_code') != 0
            or process.get('natural_job_active') != 0 or process.get('guard_closed') is not True
            or process.get('terminal_confirmed') is not True or process.get('forced_cleanup') is not False
            or process.get('cleanup_errors') != [] or not isinstance(process.get('pid'),int)):
        raise ValueError('Qualification requires successful natural owned process cleanup')
    start = datetime.fromisoformat(process['started_utc']).timestamp()
    end = datetime.fromisoformat(process['ended_utc']).timestamp()
    if not 0 < end-start < 600:
        raise ValueError('Qualification shard exceeds the bounded invocation window')
    if not start-2 <= Path(result_path).stat().st_mtime <= end+2:
        raise ValueError('Stale result outside its owned process interval')
    return (process['pid'],process['started_utc'])

def validate_command(process,directory,family,device,mode,*,offline=False):
    command=process.get('command',[])
    def option(name):
        try:return command[command.index(name)+1]
        except (ValueError,IndexError):raise ValueError('Missing executed fixture argument: '+name)
    if Path(option('--python')).resolve()!=(PUBLIC/'tests'/SCRIPTS[family]).resolve() or Path(option('--output')).resolve()!=Path(directory).resolve():
        raise ValueError('Owned process did not execute the recorded fixture/output')
    if option('--device')!=device or (not offline and option('--qualification-mode')!=mode):
        raise ValueError('Owned process used another device or qualification mode')
    if offline != ('--offline' in command): raise ValueError('Wrong offline execution boundary')

def finite_tree(value):
    if isinstance(value,float) and not math.isfinite(value):
        raise ValueError('Nonfinite numerical evidence')
    if isinstance(value,dict):
        for v in value.values(): finite_tree(v)
    if isinstance(value,(list,tuple)):
        for v in value: finite_tree(v)

def validate_result(result, family, device, *, partial=False):
    finite_tree(result)
    if not isinstance(result,dict) or (family != 'bake_contact' and result.get('device') != device):
        raise ValueError('Missing fixture/device result')
    if family == 'body_controls':
        metrics = result.get('metrics',{})
        if partial:
            if not metrics or not set(metrics).issubset(body_contract().CASE_NAMES):
                raise ValueError('Invalid body-control shard roster')
        else:
            body_contract().assert_metrics(metrics)
    elif family in ('joints','parented','bones'):
        if not result.get('metrics') or not result.get('checks') or not all(isinstance(c,str) and c for c in result['checks']):
            raise ValueError('Missing numerical metrics or completed fixture assertions')
        if family=='joints' and result.get('status')!='pass': raise ValueError('Joint acceptance absent')
        metrics=result['metrics'];checks=set(result['checks'])
        if family=='joints':
            names={'fixed-x-limit','revolute-x-limit','revolute-y-limit','prismatic-x-limit','prismatic-z-drive','d6-y-limit','d6-z-drive','static_anchor-x-limit'}
            if len(metrics)!=8 or {r.get('case') for r in metrics}!=names or any(r.get('steps')!=360 for r in metrics):raise ValueError('Incomplete joint case roster')
            for name in names:
                required={name+' complete finite poses',name+' input unchanged',name+' all compiled source layers unchanged',name+' no authoring pose writeback'}
                if name.startswith('fixed'):required.add(name+' gravity attachment')
                elif name.startswith('static_anchor'):required.update({name+' swings under gravity',name+' static anchor holds joint radius'})
                elif name.startswith('revolute'):required.update({name+' moves under gravity',name+' angular limit'})
                else:required.update({name+' configured axis',name+' linear limits',name+(' loaded drive' if name.endswith('drive') else ' gravity translation')})
                if not required.issubset(checks):raise ValueError('Incomplete joint numerical assertions')
        elif family=='parented':
            if set(metrics)!={'ic_start_first','ic_reset_first','ic_mid_first','kinematic_rider_last'}:raise ValueError('Incomplete parented pose roster')
            for phase in ('start','reset','mid'):
                if math.dist(metrics['ic_'+phase+'_first'],(8,0,6))>=.01 or 'initial condition at '+phase+' lands at the requested world pose' not in checks:raise ValueError('Parented initial pose failed')
            rider=metrics['kinematic_rider_last']
            if abs(rider[2]-3)>=.05 or rider[0]<=3.3:raise ValueError('Parented kinematic motion failed')
        else:
            names={device+'-1.0',device+'-0.01'}
            if len(metrics)!=2 or {r.get('case') for r in metrics}!=names:raise ValueError('Incomplete bone meter/cm roster')
            for row in metrics:
                if row['root_error_m']>=.01 or not row['moving_displacement_m'] or min(row['moving_displacement_m'])<=.02:raise ValueError('Bone numerical motion failed')
                required={row['case']+' '+suffix for suffix in ('fixed transformed root under gravity','same and mixed mechanism real motion','isolated and full physics trajectories agree','renamed label resolves same stable identity')}
                if not required.issubset(checks):raise ValueError('Incomplete bone numerical assertions')
    elif family == 'kinematic':
        metrics=result.get('metrics',{})
        if set(metrics) != {'object_held','object_moving','bone_held','bone_moving'}:
            raise ValueError('Incomplete kinematic cases')
        for kind in ('object','bone'):
            if not abs(metrics[kind+'_held']['final_dynamic'][0])<.001 or not metrics[kind+'_moving']['final_dynamic'][0]>.3:
                raise ValueError('Kinematic numerical acceptance failed')
    elif family == 'bake_contact':
        if set(result) != {'object','bone'}: raise ValueError('Incomplete bake cases')
        for row in result.values():
            if not (row['selected_motion_m']>.1 and row['shared_error']<=1e-4 and max(row['playback_error'])<=1e-4 and row['samples']>1):
                raise ValueError('Bake numerical acceptance failed')
    elif family == 'live_mass':
        required=('exact_state_preserved','mass_update_preserved_stage_state_time','automatic_and_explicit_inertia_parity','other_physics_changes_and_explicit_reset_replaced')
        if any(result.get(k) is not True for k in required) or result.get('preserved_steps',0)<=0 or result.get('visual_reuse_count',0)<1:
            raise ValueError('Incomplete live mass acceptance')
    else: raise ValueError('Unknown fixture family')

def validate_directory(directory, family, device, mode=None, *, partial=False):
    directory=Path(directory)
    metadata=read(directory/'runtime-identity.json')
    identity=validate_metadata(metadata,device,mode)
    accepted=read(directory/'acceptance.json')
    result_path=directory/'results.json'
    result=read(result_path)
    validate_result(result,family,device,partial=partial)
    expected={'family':family,'device':device,'qualification_mode':metadata['qualification_mode'],
              'measurement_only':metadata['measurement_only'],'identity_sha256':identity_hash(identity),
              'fixture_sha256':digest(PUBLIC/'tests'/SCRIPTS[family]),'results_sha256':digest(result_path),
              'product_source_sha256':product_source_hash()}
    if family=='body_controls': expected['assertions_sha256']=digest(PUBLIC/'tests/body_controls_assertions.py')
    if any(accepted.get(k)!=v for k,v in expected.items()): raise ValueError('Changed or mismatched fixture acceptance provenance')
    if family=='bake_contact':
        offline=read(directory/'offline-results.json');finite_tree(offline)
        if set(offline)!={'object','bone'} or any(len(v)!=2 or max(v)>1e-4 for v in offline.values()): raise ValueError('Incomplete offline bake numerical proof')
        if accepted.get('offline_results_sha256')!=digest(directory/'offline-results.json') or accepted.get('offline_process_sha256')!=digest(directory/'offline-process.json'): raise ValueError('Offline proof changed')
        validate_process(read(directory/'offline-process.json'),directory/'offline-results.json')
        validate_command(read(directory/'offline-process.json'),directory,family,device,metadata['qualification_mode'],offline=True)
        validate_inputs(directory,accepted,read(directory/'offline-process.json'),family,device,metadata['qualification_mode'],offline=True)
    if 'shards' in accepted:
        if family!='body_controls' or partial: raise ValueError('Unexpected recursive shard evidence')
        merged={}; seen=set()
        for shard in accepted['shards']:
            path=Path(shard['directory'])
            if digest(path/'acceptance.json')!=shard['acceptance_sha256']: raise ValueError('Changed shard receipt')
            row=validate_directory(path,family,device,metadata['qualification_mode'],partial=True)
            if row['metadata']['identity']!=identity: raise ValueError('Mixed shard identities')
            if row['metadata'].get('local_manifest_override')!=metadata.get('local_manifest_override'):raise ValueError('Mixed shard qualification manifests')
            if row['process_key'] in seen or set(merged)&set(row['result']['metrics']): raise ValueError('Duplicate shard/process/case')
            seen.add(row['process_key']);merged.update(row['result']['metrics'])
        if merged!=result['metrics']: raise ValueError('Aggregate differs from measured shards')
    else:
        if accepted.get('process_sha256')!=digest(directory/'process.json'): raise ValueError('Changed owned process receipt')
        process_key=validate_process(read(directory/'process.json'),result_path)
        validate_command(read(directory/'process.json'),directory,family,device,metadata['qualification_mode'])
        validate_inputs(directory,accepted,read(directory/'process.json'),family,device,metadata['qualification_mode'])
        return dict(metadata=metadata,result=result,acceptance=accepted,process_key=process_key)
    return dict(metadata=metadata,result=result,acceptance=accepted,process_key=None)

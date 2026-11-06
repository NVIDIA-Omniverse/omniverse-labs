"""Recorder preserves platform evidence and refuses incomplete measurements."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
from test_public_sdk_physics_evidence import bind, valid_metrics, valid_family_result

PUBLIC = Path(__file__).resolve().parents[1]
SCRIPTS = {'body_controls':'blender_native_body_controls.py', 'joints':'blender_native_joints.py',
           'kinematic':'blender_native_kinematic.py', 'parented':'blender_native_parented_poses.py',
           'bones':'blender_bone_mechanisms.py', 'bake_contact':'blender_bake_contact_standalone.py'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def evidence(root, identity, measurement_only=True):
    identity.setdefault('adapter_sha256','b'*64)
    identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    for family, script in SCRIPTS.items():
        for device in ('cpu', 'gpu'):
            directory = root/f'{family}-{device}'
            write(directory/'runtime-identity.json', {'identity':identity,'device':device,'measurement_only':measurement_only})
            result={'device':device,'metrics':{'measured':[1,2,3]},'checks':['completed numerical fixture assertion']}
            if family in ('joints','parented','bones'): result=valid_family_result(family,device)
            if family=='body_controls': result['metrics']=valid_metrics()
            if family=='joints': result['status']='pass'
            if family=='kinematic': result['metrics']={kind+'_'+motion:{'final_dynamic':[0 if motion=='held' else .5,0,0]} for kind in ('object','bone') for motion in ('held','moving')}
            if family=='bake_contact': result={kind:dict(selected_motion_m=.2,shared_error=0,playback_error=[0,0],samples=120) for kind in ('object','bone')}
            write(directory/'results.json',result)
            accepted = {'family':family, 'device':device,'measurement_only':measurement_only, 'identity_sha256':identity_hash, 'fixture_sha256':digest(PUBLIC/'tests'/script),
                        'results_sha256':digest(directory/'results.json')}
            if family == 'bake_contact':
                write(directory/'offline-results.json', {'object':[0,0],'bone':[0,0]})
                accepted['offline_results_sha256'] = digest(directory/'offline-results.json')
            bind(directory,family,identity,device,'measurement' if measurement_only else 'production')
    mass = []
    for device in ('cpu', 'gpu'):
        directory = root/f'mass-{device}'
        mass.append(directory)
        write(directory/'runtime-identity.json', {'identity':identity,'device':device,'measurement_only':measurement_only})
        write(directory/'results.json', {'device':device, 'exact_state_preserved':True,
            'mass_update_preserved_stage_state_time':True, 'automatic_and_explicit_inertia_parity':True,
            'other_physics_changes_and_explicit_reset_replaced':True, 'preserved_steps':3, 'visual_reuse_count':1,
            'fixture_sha256':digest(PUBLIC/'tests/blender_public_physics_generation_reuse.py'),
            'inertia_fixture_sha256':digest(PUBLIC/'scripts/probe_public_sdk_mass_inertia.py')})
        write(directory/'inertia-parity/results.json', {'status':'pass','runtime_identity':identity})
        bind(directory,'live_mass',identity,device,'measurement' if measurement_only else 'production')
    return mass


def record(root, destination, mass, production=None, production_mass=None, candidate=False):
    prefix='candidate' if candidate else 'production'
    extra = (['--'+prefix+'-evidence-root',str(production),'--'+prefix+'-live-mass-evidence',*map(str,production_mass)] if production else [])
    return subprocess.run([sys.executable,str(PUBLIC/'scripts/record_public_sdk_physics_admission.py'),
        '--evidence-root',str(root),'--destination',str(destination),'--live-mass-evidence',*map(str,mass),*extra],
        capture_output=True, text=True, timeout=15)


def test_recorder_adds_windows_and_replaces_only_same_identity(tmp_path):
    identity = {'backend':'public_sdk','platform':'win32','native_sha256':{'ovphysx/bin/ovphysx.dll':'a'*64}}
    root = tmp_path/'evidence'
    mass = evidence(root, identity)
    linux = {'identity':{'platform':'linux','adapter_sha256':'previous'},'evidence':{'preserve':'exact'}}
    destination = tmp_path/'local.json'
    write(destination, {'schema_version':1,'measurements':[linux]})
    for _ in range(2):
        result = record(root,destination,mass)
        assert result.returncode == 0, result.stderr
        rows = json.loads(destination.read_text())['measurements']
        assert len(rows) == 2 and rows[0] == linux
        assert rows[1]['identity'] == identity and len(rows[1]['evidence']) == 14
        assert rows[1]['status'] == 'candidate'
    assert not list(tmp_path.glob('*.tmp'))


@pytest.mark.parametrize('failure',[None,'mixed','destination_alias'])
def test_candidate_promotion_preserves_actual_candidate_provenance(tmp_path,failure):
    identity={'backend':'public_sdk','platform':'win32','native_sha256':{'native.dll':'a'*64}}
    measured=tmp_path/'measured';mass=evidence(measured,identity)
    candidate=tmp_path/'candidate';candidate_mass=evidence(candidate,identity,measurement_only=False)
    for family in (*SCRIPTS,'live_mass'):
        for device in ('cpu','gpu'):
            directory=candidate/('mass-'+device if family=='live_mass' else family+'-'+device)
            bind(directory,family,identity,device,'candidate')
    if failure=='mixed':
        bind(candidate/'joints-gpu','joints',identity,'gpu','production')
    destination=candidate/'candidate.json' if failure=='destination_alias' else tmp_path/'admission.json'
    before=destination.read_bytes() if destination.exists() else None
    result=record(measured,destination,mass,candidate,candidate_mass,candidate=True)
    if failure:
        assert result.returncode!=0
        assert destination.read_bytes()==before if before is not None else not destination.exists()
        if failure=='destination_alias':assert 'immutable candidate qualification manifest' in result.stderr
    else:
        assert result.returncode==0,result.stderr
        row=json.loads(destination.read_text())['measurements'][0]
        assert row['status']=='production' and row['qualification_mode']=='candidate'
        for cell in row['evidence'].values():
            assert cell['qualification_mode']=='candidate'
            assert all(c['status']=='candidate' and c['local_override'] is True for c in cell['qualification_admission_checks'])


@pytest.mark.parametrize('failure', ['changed_result', 'empty_binaries', 'invalid_existing'])
def test_failed_record_does_not_replace_existing_evidence(tmp_path, failure):
    identity = {'backend':'public_sdk','platform':'win32','native_sha256':{} if failure=='empty_binaries' else {'native.dll':'a'*64}}
    root = tmp_path/'evidence'
    mass = evidence(root,identity)
    destination = tmp_path/'local.json'
    original = '{invalid' if failure=='invalid_existing' else json.dumps({'schema_version':1,'measurements':[]})
    destination.write_text(original)
    if failure == 'changed_result':
        write(root/'joints-gpu/results.json', {'modified':True})
    result = record(root,destination,mass)
    assert result.returncode != 0
    assert destination.read_text() == original
    assert not list(tmp_path.glob('*.tmp'))


@pytest.mark.parametrize('failure', [None, 'measurement_bypass', 'wrong_identity', 'wrong_family', 'tampered_results', 'missing_mass_proof'])
def test_production_promotion_requires_gated_same_identity_runs(tmp_path, failure):
    identity = {'backend':'public_sdk','platform':'win32','native_sha256':{'native.dll':'a'*64}}
    measured = tmp_path/'measured'
    measured_mass = evidence(measured, identity)
    production = tmp_path/'production'
    production_mass = evidence(production, identity, measurement_only=False)
    directory = production/'joints-gpu'
    if failure in ('measurement_bypass','wrong_identity'):
        path = directory/'runtime-identity.json'
        metadata = json.loads(path.read_text())
        if failure == 'measurement_bypass': metadata['measurement_only'] = True
        else: metadata['identity']['platform'] = 'linux'
        write(path, metadata)
    elif failure == 'wrong_family':
        path = directory/'acceptance.json'; accepted=json.loads(path.read_text()); accepted['family']='kinematic'; write(path,accepted)
    elif failure == 'tampered_results': write(directory/'results.json', {'changed':True})
    elif failure == 'missing_mass_proof': (production_mass[0]/'inertia-parity/results.json').unlink()
    destination=tmp_path/'admission.json'
    result=record(measured,destination,measured_mass,production,production_mass)
    if failure:
        assert result.returncode != 0
        assert not destination.exists()
    else:
        assert result.returncode == 0, result.stderr
        row=json.loads(destination.read_text())['measurements'][0]
        assert row['status']=='production'
        assert all(cell['production_measurement_only'] is False and len(cell['production_results_sha256'])==64 for cell in row['evidence'].values())

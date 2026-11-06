"""Public SDK admission is exact, device-specific and family-specific."""
import copy
import json
from types import SimpleNamespace

import pytest

from ovrtx_blender_example import physics_capabilities as admission


def record(identity, device, family, status='production'):
    return {'identity': identity, 'devices': [device], 'families': [family], 'status': status,
            'evidence': {f'{family}/{device}': {'fixture_sha256': 'a'*64, 'results_sha256': 'b'*64,
                         'production_results_sha256': 'c'*64, 'production_measurement_only': False}}}


@pytest.fixture
def measured(tmp_path, monkeypatch):
    identity = {"backend": "public_sdk", "platform": "linux", "architecture": "x86_64",
                "versions": {"ovphysx": "0.6.3", "ovstage": "0.2.0.377349"},
                "native_sha256": {"ovphysx/lib/libovphysx.so": "a" * 64},
                "adapter_sha256": "b" * 64, "runtime_lock": {"lock_sha256": "c" * 64},
                "adapter_components": {"base": "d" * 64}}
    monkeypatch.setattr(admission, "__file__", str(tmp_path / "physics_capabilities.py"))
    (tmp_path / "public_sdk_physics_admission.json").write_text(json.dumps({"measurements": [
        record(identity, "cpu", "kinematic")]}))
    return identity


def unavailable_native():
    raise AssertionError("Public SDK admission must not import a SIL client")


def test_only_measured_device_and_family_are_admitted(measured):
    admission.validate_fixed_runtime(SimpleNamespace(device="cpu"), unavailable_native,
                                     lambda: measured, family="kinematic")
    for device, family in (("gpu", "kinematic"), ("cpu", "joints"), ("cpu", "body_controls"), ("cpu", "live_mass")):
        with pytest.raises(ValueError, match="no measured"):
            admission.validate_fixed_runtime(SimpleNamespace(device=device), unavailable_native,
                                             lambda: measured, family=family)


@pytest.mark.parametrize("field", ["native_sha256", "adapter_sha256", "versions", "platform", "architecture"])
def test_changed_runtime_or_adapter_loses_admission(measured, field):
    changed = copy.deepcopy(measured)
    changed[field] = "different"
    with pytest.raises(ValueError, match="no measured"):
        admission.validate_fixed_runtime(SimpleNamespace(device="cpu"), unavailable_native,
                                         lambda: changed, family="kinematic")


def test_missing_evidence_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(admission, "__file__", str(tmp_path / "physics_capabilities.py"))
    with pytest.raises(ValueError, match="no measured"):
        admission.validate_fixed_runtime(SimpleNamespace(device="cpu"), unavailable_native,
                                         lambda: {"backend": "public_sdk"}, family="kinematic")


def test_world_space_client_keeps_parented_world_pose():
    from ovrtx_blender_example.ovphysx_stage import parent_local_poses
    client = SimpleNamespace(pose_write_space="world")
    controller = SimpleNamespace(_physics_client=client)
    pose = SimpleNamespace(prim_path="/Parent/Body", translate=(8, 0, 6), orient=(0, 0, 0, 1))
    assert parent_local_poses(controller, [pose]) == (pose,)


def test_local_windows_evidence_adds_exact_admission_without_replacing_linux(measured, tmp_path, monkeypatch):
    windows = copy.deepcopy(measured)
    windows.update(platform='win32', architecture='AMD64', native_sha256={'ovphysx/bin/ovphysx.dll':'c'*64})
    local = tmp_path / 'local.json'
    local.write_text(json.dumps({'schema_version':1, 'measurements':[
        record(windows, 'gpu', 'joints')]}))
    monkeypatch.setenv('OV_BLENDER_PUBLIC_SDK_PHYSICS_ADMISSION',str(local))
    admission.validate_fixed_runtime(SimpleNamespace(device='cpu'), unavailable_native, lambda:measured, family='kinematic')
    admission.validate_fixed_runtime(SimpleNamespace(device='gpu'), unavailable_native, lambda:windows, family='joints')
    for device, family in [('cpu','joints'), ('gpu','kinematic')]:
        with pytest.raises(ValueError, match='no measured'):
            admission.validate_fixed_runtime(SimpleNamespace(device=device), unavailable_native, lambda:windows, family=family)
    windows['native_sha256']['ovphysx/bin/ovphysx.dll'] = 'd'*64
    with pytest.raises(ValueError, match='no measured'):
        admission.validate_fixed_runtime(SimpleNamespace(device='gpu'), unavailable_native, lambda:windows, family='joints')


@pytest.mark.parametrize('content', [None, '{bad json', '{}', '{"schema_version":2,"measurements":[]}', '{"measurements":[1]}'])
def test_invalid_explicit_local_evidence_fails_closed(measured, tmp_path, monkeypatch, content):
    local = tmp_path/'local.json'
    if content is not None:
        local.write_text(content)
    monkeypatch.setenv('OV_BLENDER_PUBLIC_SDK_PHYSICS_ADMISSION', str(local))
    with pytest.raises(ValueError, match='Invalid public SDK admission manifest'):
        admission.validate_fixed_runtime(SimpleNamespace(device='cpu'), unavailable_native, lambda:measured, family='kinematic')


def test_empty_native_identity_is_not_admitted_even_if_manifest_matches(measured, tmp_path):
    measured['native_sha256'] = {}
    (tmp_path/'public_sdk_physics_admission.json').write_text(json.dumps({'measurements':[
        {'identity':measured,'devices':['cpu'],'families':['kinematic']}]}))
    with pytest.raises(ValueError, match='missing native binaries'):
        admission.validate_fixed_runtime(SimpleNamespace(device='cpu'), unavailable_native, lambda:measured, family='kinematic')


@pytest.mark.parametrize('mutation', ['candidate', 'missing_status', 'no_production_proof', 'bypass', 'wrong_cell', 'missing_lock', 'missing_adapter'])
def test_measurement_or_incomplete_shipped_evidence_cannot_admit(measured, tmp_path, mutation):
    row = record(measured, 'cpu', 'kinematic')
    proof = row['evidence']['kinematic/cpu']
    if mutation == 'candidate': row['status'] = 'candidate'
    elif mutation == 'missing_status': row.pop('status')
    elif mutation == 'no_production_proof': proof.pop('production_results_sha256')
    elif mutation == 'bypass': proof['production_measurement_only'] = True
    elif mutation == 'wrong_cell': row['evidence'] = {'joints/cpu': proof}
    elif mutation == 'missing_lock': measured.pop('runtime_lock')
    elif mutation == 'missing_adapter': measured.pop('adapter_components')
    (tmp_path/'public_sdk_physics_admission.json').write_text(json.dumps({'measurements':[row]}))
    with pytest.raises(ValueError, match='no measured'):
        admission.validate_fixed_runtime(SimpleNamespace(device='cpu'), unavailable_native, lambda:measured, family='kinematic')


def test_candidate_requires_explicit_local_qualification_path(measured, tmp_path, monkeypatch):
    row = record(measured, 'cpu', 'kinematic', status='candidate')
    row['evidence']['kinematic/cpu'].pop('production_results_sha256')
    path = tmp_path/'qualification.json'
    path.write_text(json.dumps({'measurements':[row]}))
    (tmp_path/'public_sdk_physics_admission.json').unlink()
    with pytest.raises(ValueError, match='no measured'):
        admission.validate_fixed_runtime(SimpleNamespace(device='cpu'), unavailable_native, lambda:measured, family='kinematic')
    monkeypatch.setenv('OV_BLENDER_PUBLIC_SDK_PHYSICS_ADMISSION', str(path))
    admission.validate_fixed_runtime(SimpleNamespace(device='cpu'), unavailable_native, lambda:measured, family='kinematic')


def test_candidate_live_geometry_is_unavailable_without_its_own_family(measured, monkeypatch):
    from ovrtx_blender_example.direct_sdk_client import PublicSdkPhysicsClient
    client = PublicSdkPhysicsClient.__new__(PublicSdkPhysicsClient)
    client.config = SimpleNamespace(device='cpu')
    client.public_sdk_identity = lambda: measured
    client._import_native_client = unavailable_native
    client.runtime = SimpleNamespace(request=lambda *args: pytest.fail('Unadmitted geometry reached native worker'))
    assert client.supports_static_geometry_updates is False
    with pytest.raises(ValueError, match='no measured production admission'):
        client.write_physics_geometry([{'prim_path': '/World/Body', 'radius': 2}])
    client.config.device = 'gpu'
    client.public_sdk_identity = lambda: pytest.fail('GPU geometry must reject before probing')
    assert client.supports_static_geometry_updates is False

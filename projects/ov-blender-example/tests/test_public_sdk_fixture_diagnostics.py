"""CPU controls for current SDK fixture diagnostics and real admission refusals."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def configured(tmp_path, monkeypatch):
    directory = ROOT / 'addon/ovrtx_blender_example'
    package = types.ModuleType('_fixture_actual_policy')
    package.__path__ = [str(directory)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    policy = load(package.__name__ + '.physics_capabilities', directory / 'physics_capabilities.py')
    helper = load('_current_sdk_fixture_test', Path(__file__).with_name('public_sdk_physics_fixture.py'))
    monkeypatch.delenv('OV_BLENDER_PUBLIC_SDK_PHYSICS_ADMISSION', raising=False)
    rows = json.loads((directory / 'public_sdk_physics_admission.json').read_text())['measurements']
    identity = copy.deepcopy(next(row['identity'] for row in rows
                                 if row['identity']['platform'] == 'win32'
                                 and row['status'] == 'production' and 'joints' in row['families']))
    config = types.SimpleNamespace(device='cpu')
    def forbidden(*args, **kwargs):
        raise AssertionError('No native import or RPC is permitted by a rejection control')
    runtime = types.SimpleNamespace(identity=('actual-session', 1, 7), request=forbidden)
    helper._state = types.SimpleNamespace(client=types.SimpleNamespace(runtime=runtime),
        identities=[identity], admissions=[], validate=policy.validate_fixed_runtime, output=tmp_path)
    return helper, policy, config, identity, forbidden


def test_real_production_validator_accepts_original_and_refuses_both_copies(configured):
    helper, policy, config, identity, forbidden = configured
    receipt = policy.validate_fixed_runtime(config, forbidden, lambda: identity, family='joints')
    assert receipt['status'] == 'production' and receipt['local_override'] is False
    before = copy.deepcopy((helper._state.identities, helper._state.admissions))
    rows = helper.reject_changed_identities(config)
    assert [row['label'] for row in rows] == ['missing native identity', 'unknown adapter identity']
    assert 'missing native binaries' in rows[0]['error']
    assert 'no measured joints/cpu admission' in rows[1]['error']
    assert (helper._state.identities, helper._state.admissions) == before


def test_unexpected_validator_error_is_not_relabelled_as_refusal(configured):
    helper, _, config, _, _ = configured
    error = OSError('unexpected failure')
    def failed(*args, **kwargs):
        raise error
    helper._state.validate = failed
    with pytest.raises(OSError) as caught:
        helper.reject_changed_identities(config)
    assert caught.value is error


def test_false_admission_cannot_pass_negative_control(configured):
    helper, _, config, _, _ = configured
    helper._state.validate = lambda *args, **kwargs: {'status': 'production'}
    with pytest.raises(AssertionError, match='was admitted'):
        helper.reject_changed_identities(config)


def test_measurement_bypass_refused_before_import_or_selection(configured):
    helper, *_ = configured
    with pytest.raises(ValueError, match='normal production admission'):
        helper.configure(types.SimpleNamespace(measure_unadmitted=True, public_sdk=True))


def test_capabilities_are_actual_session_advertisement(configured):
    helper, *_ = configured
    helper._state.client.runtime.identity = types.SimpleNamespace(capabilities=(('typed_frames', 1),))
    assert helper.current_capabilities() == {'typed_frames': 1}


def test_local_override_refused_before_selection(configured, monkeypatch):
    helper, *_ = configured
    monkeypatch.setenv('OV_BLENDER_PUBLIC_SDK_PHYSICS_ADMISSION', 'unapproved.json')
    with pytest.raises(ValueError, match='local admission override'):
        helper.configure(types.SimpleNamespace(measure_unadmitted=False, public_sdk=True))


@pytest.mark.parametrize('mode,override,valid', [('production', '', True),
    ('production', 'local.json', False), ('measurement', '', True),
    ('measurement', 'local.json', False), ('candidate', '', False),
    ('candidate', 'local.json', True)])
def test_explicit_modes_enforce_override_provenance(configured, monkeypatch, mode, override, valid):
    helper, *_ = configured
    monkeypatch.setenv('OV_BLENDER_PUBLIC_SDK_PHYSICS_ADMISSION', override)
    args = types.SimpleNamespace(measure_unadmitted=False, public_sdk=True, qualification_mode=mode)
    if valid:
        assert helper.qualification_mode(args) == mode
    else:
        with pytest.raises(ValueError): helper.qualification_mode(args)


@pytest.mark.parametrize('receipt', [{}, {'status': 'candidate', 'local_override': False},
    {'status': 'production', 'local_override': True}, {'status': 'production'}])
def test_non_shipped_admission_refused(configured, receipt):
    helper, *_ = configured
    with pytest.raises(AssertionError, match='shipped production admission'):
        helper.require_production_admission(receipt)

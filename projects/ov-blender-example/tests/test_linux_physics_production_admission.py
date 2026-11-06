"""Every shipped Linux SDK/adapter tuple needs its own production physics proof."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from ovrtx_blender_example import physics_capabilities

PUBLIC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PUBLIC / 'runtime'))
from physics_adapter_identity import recorded_identity
from ovrtx_blender_example.public_sdk_lock import load_lock


def current_linux_identity():
    rows = json.loads((PUBLIC / 'addon/ovrtx_blender_example/public_sdk_physics_admission.json').read_text())['measurements']
    versions = load_lock(PUBLIC / 'runtime/public-sdk-lock.json')['versions']
    matching = [row['identity'] for row in rows
                if row['identity'].get('platform') == 'linux'
                and row['identity'].get('versions') == versions]
    assert matching, 'No measured Linux native identity for the shipped SDK tuple'
    identity = copy.deepcopy(matching[-1])
    identity['adapter_sha256'], identity['adapter_components'] = recorded_identity()
    return identity


def forbidden_native():
    raise AssertionError('Public SDK admission must not use the retired SIL client')


@pytest.mark.parametrize('device', ['cpu', 'gpu'])
@pytest.mark.parametrize('family', ['body_controls', 'joints', 'kinematic', 'live_mass'])
def test_current_linux_adapter_has_shipped_production_admission(monkeypatch, device, family):
    monkeypatch.delenv('OV_BLENDER_PUBLIC_SDK_PHYSICS_ADMISSION', raising=False)
    identity = current_linux_identity()
    receipt = physics_capabilities.validate_fixed_runtime(
        SimpleNamespace(device=device), forbidden_native, lambda: identity, family=family)
    assert receipt['status'] == 'production'
    assert receipt['local_override'] is False
    assert Path(receipt['manifest_path']) == PUBLIC / 'addon/ovrtx_blender_example/public_sdk_physics_admission.json'


@pytest.mark.parametrize('field', ['native_sha256', 'adapter_sha256', 'runtime_lock', 'architecture', 'versions'])
def test_current_linux_proof_does_not_admit_changed_identity(monkeypatch, field):
    monkeypatch.delenv('OV_BLENDER_PUBLIC_SDK_PHYSICS_ADMISSION', raising=False)
    identity = current_linux_identity()
    identity[field] = 'unmeasured'
    with pytest.raises(ValueError, match='no measured'):
        physics_capabilities.validate_fixed_runtime(
            SimpleNamespace(device='cpu'), forbidden_native, lambda: identity, family='joints')

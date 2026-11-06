"""Actual Windows worker identity through default admission and rejected joints."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

from pxr import Gf, Usd, UsdPhysics

PUBLIC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PUBLIC / 'addon'))
from ovrtx_blender_example.direct_sdk_client import PublicSdkPhysicsClient
from ovrtx_blender_example.physics_capabilities import validate_joint_runtime, validate_fixed_runtime
from ovrtx_blender_example.shared_stage_config import InteractiveSharedStageConfig

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
assert not os.environ.get('OV_BLENDER_PUBLIC_SDK_PHYSICS_ADMISSION')
args.output.mkdir(parents=True, exist_ok=True)
source = (PUBLIC / 'tests/fixtures/public_sdk_kinematic.usda').read_text()
cases = [('gpu-pair', 'gpu', 'body-to-body'), ('gpu-rotated', 'gpu', 'rotated joint frames'),
         ('cpu-acceleration', 'cpu', 'Acceleration joint drives'),
         ('cpu-disabled', 'cpu', 'static/disabled endpoint'),
         ('cpu-spherical', 'cpu', 'no validated runtime admission')]
config = InteractiveSharedStageConfig.from_env(str(args.output / 'unused.usda'))
client = PublicSdkPhysicsClient(config, 'restriction-validation')
results = []
try:
    client.start()
    identity = client.public_sdk_identity()
    for device in ('cpu', 'gpu'):
        for family in ('body_controls', 'joints', 'kinematic', 'live_mass'):
            receipt = validate_fixed_runtime(replace(config, device=device), client._import_native_client,
                                             client.public_sdk_identity, family=family)
            assert receipt['status'] == 'production' and receipt['local_override'] is False
            results.append({'admitted': receipt})
    for name, device, expected in cases:
        path = args.output / (name + '.usda')
        path.write_text(source)
        stage = Usd.Stage.Open(str(path))
        cls = UsdPhysics.SphericalJoint if name.endswith('spherical') else UsdPhysics.RevoluteJoint
        joint = cls.Define(stage, '/World/Joint')
        joint.CreateBody1Rel().SetTargets(['/World/Dynamic'])
        if name.endswith('pair'):
            joint.CreateBody0Rel().SetTargets(['/World/Kinematic'])
        if name.endswith('rotated'):
            joint.CreateLocalRot0Attr(Gf.Quatf(0.70710678, Gf.Vec3f(0, 0, 0.70710678)))
        if name.endswith('acceleration'):
            UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'angular').CreateTypeAttr('acceleration')
        if name.endswith('disabled'):
            stage.GetPrimAtPath('/World/Dynamic').GetAttribute('physics:rigidBodyEnabled').Set(False)
        stage.GetRootLayer().Save()
        try:
            validate_joint_runtime(str(path), replace(config, device=device), client._import_native_client,
                                   client.public_sdk_identity)
        except ValueError as exc:
            assert expected in str(exc), (name, str(exc))
            results.append({'case': name, 'rejected': str(exc)})
        else:
            raise AssertionError('Unadmitted joint accepted: ' + name)
    assert client.startup_diagnostics['loaded'] is False
    assert client.runtime.request('health', {})['loaded'] is False
    assert not any(name in sys.modules for name in ('ovphysx', 'ovstage', 'ovrtx', 'warp'))
finally:
    client.shutdown()
assert client.runtime.returncode is not None, 'Owned physics worker survived shutdown'
(args.output / 'results.json').write_text(json.dumps({'identity': identity, 'checks': results,
                                                    'worker_closed': client.runtime.returncode is not None}, indent=2))
print('DEFAULT_ADMISSION_RESTRICTIONS_PASS', len(results), flush=True)

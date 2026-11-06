# SPDX-License-Identifier: Apache-2.0
"""Bounded direct SDK render-only physics projection and final-state read probe.

This observes native capability; production admission is separately requalified.
Run in the locked SDK Python, with an owned process timeout below 600 seconds.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime'))
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
import warp as wp
from public_sdk_backend import RendererBackend, PhysicsBackend, _versions
from public_sdk_light_fixture import _generated_light_probe_fixture

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--physics-device', choices=('cpu', 'gpu'), default='cpu')
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
fixture = _generated_light_probe_fixture(args.output / 'scene.usda', 320, 200)
scene = Path(fixture['usd_path'])
text = scene.read_text()
start = text.index('    def Cube "LitCube"')
end = text.index('    def Camera "Camera"', start)
text = text[:start] + '''    def Xform "Body" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    ) {
        bool physics:rigidBodyEnabled = true
        float physics:mass = 1
        def Mesh "Visual" (
            prepend apiSchemas = ["MaterialBindingAPI", "PhysicsCollisionAPI"]
        ) {
            rel material:binding = </World/Looks/NeutralGrey>
            point3f[] points = [(-0.5,0,0), (0.5,0,0), (0.5,0,2), (-0.5,0,2)]
            int[] faceVertexCounts = [3,3]
            int[] faceVertexIndices = [0,2,1,0,3,2]
            uniform token subdivisionScheme = "none"
            normal3f[] normals = [(0,-1,0), (0,-1,0), (0,-1,0), (0,-1,0)]
            double3 xformOp:scale = (1.5, 1, 1)
            uniform token[] xformOpOrder = ["xformOp:scale"]
        }
    }
''' + text[end:]
text = text.replace('float inputs:intensity = 50000', 'float inputs:intensity = 500')
text = text.replace('float inputs:angle = 0.53', 'float inputs:angle = 0.53\nquatd xformOp:orient = (0.7071068, 0.7071068, 0, 0)\nuniform token[] xformOpOrder = ["xformOp:orient"]')
scene.write_text(text)
original_hash = hashlib.sha256(scene.read_bytes()).hexdigest()
# This diagnostic hosts both adapters serially; register the PhysX schema
# plugins before any USD population builds its immutable schema registry.
# Production uses separate renderer/physics worker processes.
physics = PhysicsBackend(args.output / 'physics')
owner = RendererBackend(args.output / 'render')
report = {'versions': _versions(), 'updates': []}
try:
    owner.dispatch('load', dict(scene_path=str(scene), sensor_paths=[fixture['render_product_path']], width=320, height=200, settings={}))
    stage_id = owner.native_stage_id
    initial_counters = dict(owner.update_diagnostics)
    def pixels(label):
        started = time.perf_counter()
        for _ in range(8):
            products = owner.renderer.step(set(owner.sensors), delta_time=1/60, ordinal=owner.ordinal)
        frame = products[owner.sensors[0]].frames[-1]
        buffer = next(v for k, v in frame.render_vars.items() if k.rsplit('/', 1)[-1] == 'LdrColor')
        mapped = buffer.map(device=owner.rtx.Device.CUDA)
        try:
            result = wp.from_dlpack(mapped).numpy()[..., :3].copy()
        finally:
            mapped.unmap()
        # Pixels stay .npy: the locked SDK Python has no Pillow (test-only).
        np.save(args.output / (label + '.npy'), result)
        report[label + '_render_seconds'] = time.perf_counter() - started
        return result
    baseline = pixels('rest')
    for label, x in (('moved', 1.1), ('returned', 0.0)):
        matrix = np.eye(4); matrix[3, 0] = x
        result = owner.update([
            dict(prim_path='/World/Body', attribute_name='omni:xform', value_type='Matrix4d', value=matrix.tolist()),
            dict(prim_path='/World/Body', attribute_name='omni:resetXformStack', value_type='Bool', value=True)])
        report['updates'].append(result)
        assert result['native_pose_route'] == 'native_inputs', result
        assert result['native_transform_auto_invalidation'], result
        with owner.ovstage.PathDictionary(owner.stage) as paths:
            worlds = owner.native_pose_fanout._worlds({'/World/Body', '/World/Body/Visual'}, owner.ordinal, paths)
        np.testing.assert_allclose(worlds['/World/Body'].reshape(4, 4), matrix, atol=1e-6)
        expected = matrix.copy(); expected[0, 0] = 1.5
        np.testing.assert_allclose(worlds['/World/Body/Visual'].reshape(4, 4), expected, atol=1e-6)
        captured = pixels(label)
        report[label + '_mean_abs_difference'] = float(np.abs(captured.astype(float) - baseline.astype(float)).mean())
    assert report['moved_mean_abs_difference'] > 1.0, report
    assert report['returned_mean_abs_difference'] < 0.5, report
    assert owner.native_stage_id == stage_id
    assert owner.population_owner.path == scene.resolve()
    assert not list((args.output / 'render').glob('native-poses*.usda'))
    assert hashlib.sha256(scene.read_bytes()).hexdigest() == original_hash
    report['update_counters'] = owner.update_diagnostics
    report['steady_renderer_resets'] = owner.update_diagnostics['renderer_resets'] - initial_counters['renderer_resets']
    report['steady_population_replacements'] = owner.update_diagnostics['usd_reprojections'] - initial_counters['usd_reprojections']
    report['initial_update_counters'] = initial_counters
    assert report['steady_renderer_resets'] == 0
    assert report['steady_population_replacements'] == 0
    report['native_stage_id'] = stage_id
    report['scene_sha256'] = original_hash
finally:
    owner.dispatch('unload', {})
    (args.output / 'render-proof.json').write_text(json.dumps(report, indent=2))
physics_scene = args.output / 'physics.usda'
physics_scene.write_text((ROOT / 'tests/fixtures/public_sdk_kinematic.usda').read_text()
    .replace('float physics:gravityMagnitude = 0', 'float physics:gravityMagnitude = 9.81'))
try:
    health = physics.dispatch('load', dict(scene_path=str(physics_scene),
        body_prims=['/World/Dynamic', '/World/Kinematic'], device=args.physics_device))
    initial = physics.dispatch('read', {})
    before_reads = physics.state_read_calls
    before_native_reads = physics.native_state_read_sessions
    for index in range(4):
        reply = physics.dispatch('step', dict(dt_ns=16666667,
            time_ns=(index + 1) * 16666667, read_states=False))
        assert 'states' not in reply and physics.state_read_calls == before_reads
    final = physics.dispatch('read', {})
    assert physics.state_read_calls == before_reads + 1 and physics.step_calls == 4
    assert physics.native_state_read_sessions == before_native_reads + 1
    initial_z = next(row for row in initial['states'] if row['prim_path'] == '/World/Dynamic')['translate']['z']
    final_z = next(row for row in final['states'] if row['prim_path'] == '/World/Dynamic')['translate']['z']
    assert final_z < initial_z
    assert next(row for row in final['states'] if row['prim_path'] == '/World/Kinematic')['translate']['z'] == 1.0
    report['physics'] = dict(device=args.physics_device, health=health, final=final,
        step_calls=physics.step_calls, publication_read_calls=physics.state_read_calls-before_reads,
        publication_native_state_reads=physics.native_state_read_sessions-before_native_reads)
finally:
    physics.dispatch('unload', {})
report['source_sha256'] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in (ROOT/'runtime/native_pose_fanout.py', ROOT/'runtime/public_sdk_backend.py', ROOT/'addon/ovrtx_blender_example/direct_sdk_client.py')}
(args.output / 'proof.json').write_text(json.dumps(report, indent=2))
print('RENDER_PHYSICS_PROJECTION_PASS ' + json.dumps(report), flush=True)

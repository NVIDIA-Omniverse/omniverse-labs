"""Real authored visual reuse, body-topology replacement and explicit reset."""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
import uuid

import bpy

source = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(source / 'addon'))
sys.path.insert(0, str(source / 'tests'))
from public_sdk_physics_fixture import add_arguments, configure
from ovrtx_blender_example import authoring_properties as props
from ovrtx_blender_example.physics_endpoints import initialize_owner
from ovrtx_blender_example.generation_runtime_adapters import OvphysxGenerationAdapter
from ovrtx_blender_example.scene_generation import SceneGenerationOwner
from ovrtx_blender_example.ovphysx_stage import OvphysxStageStatus

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--device', choices=('cpu', 'gpu'), default='cpu')
add_arguments(p)
args = p.parse_args(sys.argv[sys.argv.index('--') + 1:])
assert args.public_sdk and not args.measure_unadmitted
assert not os.environ.get('OV_BLENDER_PUBLIC_SDK_PHYSICS_ADMISSION')
configure(args)
os.environ['OV_BLENDER_EXAMPLE_OVPHYSX_DEVICE'] = args.device
props.register()
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
scene = bpy.context.scene
scene.ov.physics_scene.schema_opinion = 'APPLY'
scene.ov.physics_scene.gravity_magnitude = 0


def body(name, x):
    obj = bpy.data.objects.new(name, None)
    scene.collection.objects.link(obj)
    obj.location = (x, 0, 1)
    initialize_owner(obj)
    obj.ov.physics.rigid_body.schema_opinion = 'APPLY'
    obj.ov.physics.rigid_body.role = 'DYNAMIC'
    obj.ov.physics.rigid_body.initial_linear_velocity = (.25, 0, 0)
    obj.ov.physics.rigid_body.linear_damping = 0
    obj.ov.physics.rigid_body.angular_damping = 0
    collider = obj.ov.physics.colliders.add()
    collider.name = collider.persistent_id = uuid.uuid4().hex
    collider.schema_opinion, collider.shape, collider.radius_m = 'APPLY', 'ANALYTIC_SPHERE', .2
    return obj


original_body = body('ExistingBody', 100)
bpy.ops.mesh.primitive_cube_add(location=(0, 0, 2))
visual = bpy.context.object
bpy.context.view_layer.update()
owner = SceneGenerationOwner(args.output / 'generations')
adapter = OvphysxGenerationAdapter()
records = {}


def snapshot():
    controller = adapter.controller
    client = controller._physics_client
    health = client.runtime.request('health', {})
    states, raw = client.read_body_states(controller.current_simulation_time_ns)
    return dict(owner=asdict(client.runtime.identity), stage=health['stage_identity'],
                physics=health['physics_identity'], steps=controller.step_count,
                controller_time=controller.current_simulation_time_ns,
                backend_time=raw['simulation_time_ns'], states=sorted(states, key=lambda row: row['prim_path']))


try:
    generation = owner.replace(scene)
    assert adapter.activate(generation), adapter.last_error
    controller = adapter.controller
    for _ in range(10):
        assert controller.tick().status == OvphysxStageStatus.OK
    records['before_visual'] = snapshot()
    assert records['before_visual']['steps'] == 10 * controller.config.steps_per_update
    visual.location.x += 2
    bpy.context.view_layer.update()
    assert adapter.activate(owner.replace(scene)), adapter.last_error
    assert adapter.controller is controller
    records['after_visual'] = snapshot()
    assert records['before_visual'] == records['after_visual']
    assert adapter.reused_generation_count == 1
    assert controller.tick().status == OvphysxStageStatus.OK
    records['continued'] = snapshot()
    assert records['continued']['backend_time'] > records['after_visual']['backend_time']
    new_body = body('AddedBody', 120)
    bpy.context.view_layer.update()
    assert adapter.activate(owner.replace(scene)), adapter.last_error
    assert adapter.controller is not controller and controller.closed
    replacement = adapter.controller
    records['topology_reset'] = snapshot()
    reset = records['topology_reset']
    assert reset['owner']['pid'] != records['continued']['owner']['pid']
    assert reset['steps'] == reset['controller_time'] == reset['backend_time'] == 0
    assert len(reset['states']) == 2
    positions = sorted(tuple(s['translate'][a] for a in 'xyz') for s in reset['states'])
    assert all(math.dist(a, b) < 2e-5 for a, b in zip(positions, [(100, 0, 1), (120, 0, 1)]))
    assert replacement.tick().status == OvphysxStageStatus.OK
    records['before_explicit_reset'] = snapshot()
    assert adapter.reset(), adapter.last_error
    assert adapter.controller is not replacement and replacement.closed
    records['explicit_reset'] = snapshot()
    explicit = records['explicit_reset']
    assert explicit['owner']['pid'] != reset['owner']['pid']
    assert explicit['steps'] == explicit['controller_time'] == explicit['backend_time'] == 0
    assert tuple(original_body.location) == (100, 0, 1) and tuple(new_body.location) == (120, 0, 1)
    assert explicit['states'] == reset['states']
finally:
    assert adapter.deactivate() in {'stopped', 'not_found'}

(args.output / 'results.json').write_text(json.dumps(dict(status='passed', records=records,
    checks=['visual generation exact same owner/state/time', 'continued simulation',
            'added body resets topology to full authored roster', 'explicit reset restores exact initial state',
            'authoring channels remain exact']), indent=2) + '\n')

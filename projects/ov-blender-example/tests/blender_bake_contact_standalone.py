"""Native full-scene contact selection and truly standalone Object/bone bake fixtures.

Run normally with --runtime/--output, then in a fresh factory Blender process
with --offline --output pointing to the same directory (no addon import).
"""
import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid
import bpy
from mathutils import Matrix

p = argparse.ArgumentParser()
p.add_argument('--runtime', type=Path, default=Path('.'))
p.add_argument('--device', choices=('cpu','gpu'), default='cpu')
p.add_argument('--output', type=Path, required=True)
p.add_argument('--offline', action='store_true')
p.add_argument('--case', choices=('all','object','bone'), default='all')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import public_sdk_physics_fixture as sdk_fixture
sdk_fixture.add_arguments(p)
a = p.parse_args(sys.argv[sys.argv.index('--') + 1:])
out = a.output.resolve()
out.mkdir(parents=True, exist_ok=True)

def errors(world, desired):
    qa, qb = world.to_quaternion(), desired.to_quaternion()
    dot = abs(sum(float(x)*float(y) for x,y in zip(qa,qb)))
    norm = math.sqrt(sum(float(x)**2 for x in qa)*sum(float(x)**2 for x in qb))
    return (world.translation-desired.translation).length, 2*math.acos(min(1.,dot/norm))

def no_native_data(scene):
    assert scene.rigidbody_world is None
    assert all(o.rigid_body is None and o.rigid_body_constraint is None and not o.constraints for o in scene.objects)
    assert all(not b.constraints for o in scene.objects if o.type == 'ARMATURE' for b in o.pose.bones)

def evaluate(scene, oracle):
    maximum = [0., 0.]
    for sample in oracle['frames']:
        scene.frame_set(sample['frame'])
        bpy.context.view_layer.update()
        for row in oracle['owners']:
            obj = scene.objects[row['object']]
            world = obj.matrix_world @ obj.pose.bones[row['bone']].matrix if row['bone'] else obj.matrix_world
            error = errors(world, Matrix(sample['worlds'][row['path']]))
            assert max(error) <= 1e-4, (row, sample['frame'], error)
            maximum = [max(x,y) for x,y in zip(maximum,error)]
    no_native_data(scene)
    return maximum

if a.offline:
    assert 'ovrtx_blender_example' not in sys.modules and not hasattr(bpy.types.Scene, 'ov_physics_bake')
    result = {}
    for kind in (('object','bone') if a.case == 'all' else (a.case,)):
        directory = out / kind
        oracle = json.loads((directory/'offline-oracle.json').read_text())
        bpy.ops.wm.open_mainfile(filepath=str(directory/'baked.blend'))
        result[kind] = evaluate(bpy.context.scene, oracle)
        from bpy_extras.anim_utils import action_get_channelbag_for_slot
        obj = bpy.context.scene.objects[oracle['owners'][0]['object']]
        bag = action_get_channelbag_for_slot(obj.animation_data.action, obj.animation_data.action_slot)
        point = bag.fcurves[0].keyframe_points[0]
        old = point.co.y
        point.co.y += .125
        assert point.co.y != old and point.interpolation == 'LINEAR'
    (out/'offline-results.json').write_text(json.dumps(result, indent=2))
    print('PASS standalone/contact offline', result, flush=True)
else:
    assert a.runtime is not None
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'addon'))
    from ovrtx_blender_example import authoring_properties as props, physics_bake_ui as ui
    from ovrtx_blender_example.physics_endpoints import initialize_owner, initialize_joint
    from ovrtx_blender_example.physics_authoring import apply_anchor
    from ovrtx_blender_example.physics_bake_capture import BakeCaptureSettings, BakeCaptureSession
    from ovrtx_blender_example.physics_bake_conversion import convert_capture, _target, _world
    from ovrtx_blender_example.physics_bake_actions import build_actions, activate, deactivate
    from ovrtx_blender_example.physics_pose_presentation import apply, resolve
    from ovrtx_blender_example.interactive_operator_state import PhysicsPlaybackLock
    from ovrtx_blender_example.ovphysx_stage import OvphysxStageController
    from ovrtx_blender_example import scene_generation_sessions as sessions
    sdk_fixture.configure(a)
    props.register()
    ui.register()
    runtime = a.runtime.resolve()
    server = Path(__file__).resolve() if a.public_sdk else runtime/'bin/ovphysx_grpc_server.exe'
    os.environ['OVPHYSX_ROOT'] = str(runtime/'private/ovphysx-runtime')
    results = {}
    for kind in (('object','bone') if a.case == 'all' else (a.case,)):
        sessions.close()
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        scene = bpy.context.scene
        scene.frame_set(1)
        scene.ov.physics_scene.schema_opinion = 'APPLY'
        scene.ov.physics_scene.gravity_magnitude = 0
        directory = out/kind
        directory.mkdir(parents=True, exist_ok=True)
        if kind == 'bone':
            bpy.ops.object.armature_add()
            obj = bpy.context.object
            obj.name = 'StandaloneRig'
            target = initialize_owner(obj, obj.data.bones[0])
        else:
            obj = bpy.data.objects.new('SelectedSlider', None)
            scene.collection.objects.link(obj)
            target = initialize_owner(obj)
        physics = target.ov.physics
        physics.rigid_body.schema_opinion = 'APPLY'
        physics.rigid_body.role = 'DYNAMIC'
        physics.rigid_body.mass_kg = 1
        physics.rigid_body.initial_linear_velocity = (1,0,0) if kind == 'bone' else (0,0,0)
        collider = physics.colliders.add()
        collider.name = collider.persistent_id = uuid.uuid4().hex
        collider.schema_opinion, collider.shape, collider.radius_m = 'APPLY', 'ANALYTIC_SPHERE', .2
        handle = bpy.data.objects.new('WorldSlider', None)
        scene.collection.objects.link(handle)
        joint = initialize_joint(handle)
        joint.type, joint.axis = 'PRISMATIC', 'X'
        joint.endpoint_a.kind = 'WORLD'
        joint.endpoint_b.object = obj
        if kind == 'bone':
            joint.endpoint_b.kind = 'BONE'
            joint.endpoint_b.bone_name = target.name
            joint.endpoint_b.bone_id = obj.data.bones[target.name].ov.persistent_id
        joint.axes['transX'].state = 'FREE'
        bpy.context.view_layer.update()
        apply_anchor(handle, scene, bpy.context.evaluated_depsgraph_get())
        striker = None
        if kind == 'object':
            striker = bpy.data.objects.new('UnselectedUnconnectedStriker', None)
            scene.collection.objects.link(striker)
            striker.location = (-1,0,0)
            initialize_owner(striker)
            ph = striker.ov.physics
            ph.rigid_body.schema_opinion, ph.rigid_body.role = 'APPLY','DYNAMIC'
            ph.rigid_body.mass_kg = 1
            ph.rigid_body.initial_linear_velocity = (2,0,0)
            c = ph.colliders.add()
            c.name = c.persistent_id = uuid.uuid4().hex
            c.schema_opinion, c.shape, c.radius_m = 'APPLY','ANALYTIC_SPHERE',.2
        bpy.context.view_layer.update()
        assert (kind == 'bone') == any(o.type == 'ARMATURE' for o in scene.objects)
        no_native_data(scene)
        bpy.ops.wm.save_as_mainfile(filepath=str(directory/'authored.blend'))
        def capture(name, paths=None):
            with socket.socket() as sock:
                sock.bind(('127.0.0.1',0))
                address = f'127.0.0.1:{sock.getsockname()[1]}'
            def factory(config):
                return OvphysxStageController(replace(config,server=str(server),ovphysx_address=address,
                    ovphysx_worker_command=subprocess.list2cmdline([str(server),'--listen',address,'--device',a.device]),
                    device=a.device,worker_log_path=str(directory/(name+'.log')),
                    ovphysx_native_client_path=str(runtime/'client'),ovphysx_native_client_module='ovphysx_bridge_client'))
            session = BakeCaptureSession(scene,BakeCaptureSettings(1,1,25,24,substeps=5),directory/name,
                output_paths=paths,controller_factory=factory)
            try:
                while not session.poll():
                    time.sleep(.001)
            finally:
                session.close()
            host,port = address.rsplit(':',1)
            with socket.socket() as sock:
                sock.settimeout(.5)
                assert sock.connect_ex((host,int(port))) != 0
            return session.artifact
        all_capture = capture('all')
        targets = {_target(row).body_path:_target(row) for row in all_capture.manifest()['targets']}
        path = next(p for p,t in targets.items() if t.object_session_uid == obj.session_uid)
        assert len(targets) == (2 if striker else 1)
        assert all(bool(t.bone_identity) == (kind == 'bone') for t in targets.values())
        selected = capture('selected',(path,))
        assert selected.manifest()['output_paths'] == [path]
        assert selected.manifest()['body_paths'] == all_capture.manifest()['body_paths']
        shared_error = 0.
        for first,second in zip(all_capture.samples(),selected.samples(),strict=True):
            assert first['step'] == second['step'] and len(second['poses']) == len(targets)
            error = errors(_world(targets[path],first['poses'][path]),_world(targets[path],second['poses'][path]))
            shared_error = max(shared_error,*error)
        assert shared_error <= 1e-4
        motion = list(selected.samples())[-1]['poses'][path]['position'][0]
        assert motion > .1, motion
        control_motion = None
        if striker:
            striker.location.y = 3
            bpy.context.view_layer.update()
            control = capture('no-contact',(path,))
            control_motion = list(control.samples())[-1]['poses'][path]['position'][0]
            assert abs(control_motion) < 1e-4 and motion-control_motion > .1
            striker.location.y = 0
            bpy.context.view_layer.update()
        from ovrtx_blender_example.physics_bake_configuration import channel_configuration, TRANSFORMS, DELTAS
        debug_before = [(o.name, channel_configuration(o,o,TRANSFORMS+DELTAS), {n:list(getattr(o,n)) for n in TRANSFORMS+DELTAS}) for o in scene.objects]
        # Show real native samples through the transient Object/bone presentation seam.
        lock = PhysicsPlaybackLock()
        for sample in (list(selected.samples())[0],list(selected.samples())[-1]):
            poses = {p:{'translate':v['position'],'orient':v['rotation_xyzw']} for p,v in sample['poses'].items()}
            apply(scene,tuple(targets.values()),poses,lock,1,bpy.context.view_layer)
            channel = resolve(scene,targets[path])[1]
            world = obj.matrix_world @ channel.matrix if kind == 'bone' else obj.matrix_world
            assert max(errors(world,_world(targets[path],sample['poses'][path]))) <= 1e-4
        lock.clear(reason='test_capture_display_complete')
        (directory/'display-configuration.json').write_text(json.dumps({'before':debug_before, 'after':[(o.name, channel_configuration(o,o,TRANSFORMS+DELTAS), {n:list(getattr(o,n)) for n in TRANSFORMS+DELTAS}) for o in scene.objects]},indent=2))
        converted = convert_capture(scene,selected,directory/'converted')
        batch = build_actions(scene,converted)
        transaction = activate(scene,batch)
        while not transaction.step():
            time.sleep(.001)
        oracle = {'owners':[{'path':path,'object':obj.name,'bone':targets[path].bone_name}], 'frames':[]}
        for sample in selected.samples():
            oracle['frames'].append({'frame':sample['frame'],'worlds':{path:[list(row) for row in _world(targets[path],sample['poses'][path])]}})
        maximum = evaluate(scene,oracle)
        assert sessions.presentation_stopped(scene)
        if striker:
            assert striker.animation_data is None or striker.animation_data.action is None
        bpy.ops.wm.save_as_mainfile(filepath=str(directory/'baked.blend'))
        (directory/'offline-oracle.json').write_text(json.dumps(oracle,indent=2))
        results[kind] = dict(selected_motion_m=motion,no_contact_motion_m=control_motion,shared_error=shared_error,
            playback_error=maximum,body_count=len(targets),samples=len(oracle['frames']))
        deactivate(scene)
        print(kind,results[kind],flush=True)
    sessions.close()
    (out/'results.json').write_text(json.dumps(results,indent=2))
    print('PASS standalone/contact native',results,flush=True)




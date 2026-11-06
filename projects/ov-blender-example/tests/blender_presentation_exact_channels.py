"""Real RNA presentation/release preserves exact authored transform components."""
import json
from pathlib import Path
import sys
import tempfile
import bpy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'addon'))
from ovrtx_blender_example import authoring_properties as props
from ovrtx_blender_example.physics_endpoints import initialize_owner
from ovrtx_blender_example.physics_presentation_targets import presentation_targets
from ovrtx_blender_example.physics_pose_presentation import apply
from ovrtx_blender_example.interactive_operator_state import PhysicsPlaybackLock, apply_pending_runtime_pose_mirror
from ovrtx_blender_example.scene_generation import SceneGenerationOwner
from ovrtx_blender_example.physics_bake_configuration import channel_configuration, TRANSFORMS, DELTAS
props.register()
scene = bpy.context.scene
count = 0
for kind in ('OBJECT','BONE'):
    for mode in ('XYZ','QUATERNION','AXIS_ANGLE'):
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        if kind == 'BONE':
            bpy.ops.object.armature_add()
            obj = bpy.context.object
            channel = initialize_owner(obj,obj.data.bones[0])
        else:
            obj = bpy.data.objects.new('Owned',None)
            scene.collection.objects.link(obj)
            channel = initialize_owner(obj)
        unrelated = bpy.data.objects.new('Unowned',None)
        scene.collection.objects.link(unrelated)
        channel.rotation_mode = mode
        channel.location = (1.125,-2.25,3.5)
        channel.rotation_euler = (0.,0.,0.)
        channel.rotation_quaternion = (.9238795325,0.,0.,.3826834324)
        channel.rotation_axis_angle = (.37,0.,1.,0.)
        channel.scale = (1.2,)*3
        if kind == 'OBJECT':
            obj.delta_location = (.13,-.27,.09)
            obj.delta_rotation_euler = (.07,-.03,.11)
            obj.delta_rotation_quaternion = (.98078528,.19509032,0.,0.)
            obj.delta_scale = (1.1,)*3
        channel.ov.physics.rigid_body.schema_opinion = 'APPLY'
        channel.ov.physics.rigid_body.role = 'DYNAMIC'
        bpy.context.view_layer.update()
        names = TRANSFORMS + (DELTAS if kind == 'OBJECT' else ())
        before = json.dumps({name:list(getattr(channel,name)) for name in names},sort_keys=True)
        digest = channel_configuration(obj,channel,names)
        basis = channel.matrix_basis.copy()
        owner = SceneGenerationOwner(Path(tempfile.mkdtemp(prefix='ov-release-exact-')))
        generation = owner.replace(scene)
        targets = presentation_targets(generation)
        assert len(targets) == 1
        lock = PhysicsPlaybackLock()
        apply(scene,targets,{targets[0].body_path:{'translate':(7.,8.,9.),'orient':(0.,0.,.3826834324,.9238795325)}},lock,1,bpy.context.view_layer)
        assert channel.matrix_basis != basis
        unrelated.location.x = 17
        lock.clear(reason='roundtrip')
        bpy.context.view_layer.update()
        after = json.dumps({name:list(getattr(channel,name)) for name in names},sort_keys=True)
        assert after == before,(kind,mode,before,after)
        assert channel_configuration(obj,channel,names) == digest
        assert unrelated.location.x == 17 and not lock.is_active()
        assert max(abs(channel.matrix_basis[i][j]-basis[i][j]) for i in range(4) for j in range(4)) < 1e-6
        count += 1
        if kind == 'OBJECT':
            pending = {'poses_by_path':{'/World/Legacy':{'translate':(7.,8.,9.),'orient':(0.,0.,.3826834324,.9238795325)}},
                'object_paths_by_session_uid':{int(obj.session_uid):'/World/Legacy'}, 'owning_generation':1}
            result = apply_pending_runtime_pose_mirror(bpy,pending,lock)
            assert result['mirrored_count'] == 1, result
            assert channel.matrix_basis != basis
            unrelated.location.x = 19
            lock.clear(reason='legacy_roundtrip')
            bpy.context.view_layer.update()
            assert json.dumps({name:list(getattr(channel,name)) for name in names},sort_keys=True) == before
            assert channel_configuration(obj,channel,names) == digest
            assert unrelated.location.x == 19 and not lock.is_active()
            count += 1
print('PASS exact authored presentation restoration',count,flush=True)

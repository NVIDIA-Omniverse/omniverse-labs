"""CPU-only real RNA: batched sibling evaluation, parents, bones, rollback."""
from pathlib import Path
import json
import sys
from types import SimpleNamespace
import bpy
from mathutils import Matrix, Quaternion, Vector
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'addon'))
from ovrtx_blender_example import authoring_properties as props
from ovrtx_blender_example.physics_endpoints import initialize_owner
from ovrtx_blender_example.physics_presentation_targets import PhysicsPresentationTarget
from ovrtx_blender_example.physics_frames import RigidFrame
from ovrtx_blender_example.physics_pose_presentation import apply, release, ResolutionIndex
from ovrtx_blender_example.interactive_operator_state import PhysicsPlaybackLock

props.register()
scene = bpy.context.scene
bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
targets, poses, expected, originals = [], {}, {}, {}

def add(obj, bone=None, position=(0,0,0), scale=(1,1,1)):
    channel = initialize_owner(obj, bone)
    path = '/World/B' + str(len(targets))
    target = PhysicsPresentationTarget(path, channel.ov.physics.rigid_body.persistent_id,
        obj.session_uid, obj.ov.physics.persistent_id,
        bone.ov.persistent_id if bone else '', bone.name if bone else '',
        'DYNAMIC', True, 1., 1., RigidFrame(), RigidFrame(), scale)
    targets.append(target)
    poses[path] = dict(translate=position, orient=(0,0,0,1))
    expected[path] = Matrix.LocRotScale(Vector(position), Quaternion((1,0,0,0)), Vector(scale))
    originals[path] = channel.matrix_basis.copy()
    return target

for i in range(100):
    obj = bpy.data.objects.new('Sibling' + str(i), None); scene.collection.objects.link(obj)
    add(obj, position=(i, 2, 3))
parent = bpy.data.objects.new('Parent', None);scene.collection.objects.link(parent)
child = bpy.data.objects.new('Child', None);scene.collection.objects.link(child);child.parent=parent
add(parent, position=(4,5,6), scale=(2,3,4));add(child, position=(7,8,9), scale=(1,2,3))
bpy.ops.object.armature_add();rig=bpy.context.object;rig.scale=(2,3,4);rig.parent=parent
bpy.ops.object.mode_set(mode='EDIT')
root=rig.data.edit_bones[0];root.name='Root';root.head=(0,0,0);root.tail=(0,1,0)
child_bone=rig.data.edit_bones.new('Tip');child_bone.head=(0,1,0);child_bone.tail=(0,2,0);child_bone.parent=root
bpy.ops.object.mode_set(mode='OBJECT')
for bone in rig.data.bones: bone.inherit_scale='NONE'
bpy.context.view_layer.update()
add(rig,rig.data.bones['Root'],position=(1,2,3))
add(rig,rig.data.bones['Tip'],position=(4,5,6))

class Evaluation:
    def __init__(self): self.count=0
    def update(self): self.count+=1;bpy.context.view_layer.update()
layer=Evaluation();lock=PhysicsPlaybackLock()
index=ResolutionIndex(scene)
apply(scene,targets,poses,lock,1,layer,index=index)
assert layer.count <= 4, layer.count
for target in targets:
    obj,channel=index.resolve(target)
    actual=obj.matrix_world if channel is obj else obj.matrix_world @ channel.matrix
    error=max(abs(actual[i][j]-expected[target.body_path][i][j]) for i in range(4) for j in range(4))
    assert error < 1e-5,(target.body_path,error)
release(lock);bpy.context.view_layer.update()
for target in targets:
    obj,channel=ResolutionIndex(scene).resolve(target)
    assert max(abs(channel.matrix_basis[i][j]-originals[target.body_path][i][j]) for i in range(4) for j in range(4)) < 1e-5
print('PASS presentation batching',json.dumps(dict(targets=len(targets),evaluations=layer.count)),flush=True)

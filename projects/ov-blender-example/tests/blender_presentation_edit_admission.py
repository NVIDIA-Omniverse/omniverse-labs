"""Real RNA ownership checked before the existing runtime edit workflow."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import bpy
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from ovrtx_blender_example import authoring_properties as props
from ovrtx_blender_example.physics_endpoints import initialize_owner
from ovrtx_blender_example.physics_frames import RigidFrame
from ovrtx_blender_example.physics_presentation_targets import PhysicsPresentationTarget
from ovrtx_blender_example.interactive_operator_state import prepare_runtime_pose_mirror, apply_pending_runtime_pose_mirror
from ovrtx_blender_example.interactive_edit_planner import InteractiveEdit, EditShape, DataAuthority, edit_location
from ovrtx_blender_example.scene_generation_sessions import AuthoringGenerationRuntime
from ovrtx_blender_example.scene_generation import SceneGenerationOwner
from ovrtx_blender_example.shared_stage_composition import BodyPose

out = Path(sys.argv[sys.argv.index("--")+1]).resolve(); out.mkdir(parents=True, exist_ok=True)
props.register()
scene = bpy.context.scene
obj = bpy.data.objects.new("OwnedOutput", None); scene.collection.objects.link(obj)
initialize_owner(obj)
rig = bpy.data.objects.new("Rig", bpy.data.armatures.new("RigData")); scene.collection.objects.link(rig)
bpy.context.view_layer.objects.active = rig; rig.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
for name, x in (("Owned", 0), ("Control", 3)):
    bone = rig.data.edit_bones.new(name); bone.head = (x,0,0); bone.tail = (x,1,0)
bpy.ops.object.mode_set(mode="OBJECT")
initialize_owner(rig, rig.data.bones["Owned"])
def descriptor(owner, path, bone=None):
    return PhysicsPresentationTarget(path, owner.ov.physics.persistent_id, int(owner.session_uid),
        owner.ov.physics.persistent_id, bone.ov.persistent_id if bone else "", bone.name if bone else "",
        "DYNAMIC", True, 1., 1., RigidFrame(), RigidFrame())
targets = (descriptor(obj, "/World/Body"), descriptor(rig, "/World/Bone", rig.data.bones["Owned"]))
runtime = AuthoringGenerationRuntime(SimpleNamespace(deactivate=lambda: "not_found"), SceneGenerationOwner(out / "generation"))
forwarded = []
runtime.workflow.preview_edit_group = lambda edits: forwarded.append(edits) or ()
pending, _ = prepare_runtime_pose_mirror((BodyPose("/World/Body", (2,0,0), (0,0,0,1)),
    BodyPose("/World/Bone", (0,2,0), (0,0,0,1))), bpy, bpy.context, presentation_targets=targets,
    object_paths_by_session_uid={obj.session_uid: "/World/RenderedOwner", rig.session_uid: "/World/Rig"}, owning_generation=7)
applied = apply_pending_runtime_pose_mirror(bpy, pending, runtime.playback_lock)
assert applied["mirrored_count"] == 2
checks = []
def check(label, condition):
    assert condition, label
    checks.append(label)
def edit(path, prop):
    return InteractiveEdit(shape=EditShape.VALUE, data_authority=DataAuthority.VIEW,
        **edit_location(usd_layer_id="scene.usda", usd_prim_path=path, blender_property_path=prop), value=(99,0,0))
for path in ("/World/Body", "/World/RenderedOwner"):
    obj.location.x = 99
    result = runtime.submit_edit_group((edit(path, "matrix_world"),))
    check("Object edit rejected before forwarding " + path, result[0].reason == "physics_playback_locked" and not forwarded)
    check("Object displayed transform restored " + path, obj.location.x == 2)
bone = rig.pose.bones["Owned"]
display = bone.matrix_basis.copy()
bone.location.x = 99
result = runtime.submit_edit_group((edit("/World/Rig", bone.path_from_id()+".location"),))
check("owned bone edit rejected before forwarding", result[0].reason == "physics_playback_locked" and not forwarded)
check("owned bone displayed transform restored", bone.matrix_basis == display)
control = rig.pose.bones["Control"]
control.location.x = 3
runtime.submit_edit_group((edit("/World/Rig", control.path_from_id()+".location"),))
check("nonowned control bone remains editable", len(forwarded) == 1 and control.location.x == 3)
rig.location.x = 4
runtime.submit_edit_group((edit("/World/Rig", "matrix_world"),))
check("armature Object channel remains editable", len(forwarded) == 2 and rig.location.x == 4)
diagnostics = runtime.playback_lock.diagnostics()
check("presentation-only ownership diagnostics", diagnostics["active"] and diagnostics["locked_bone_count"] == 1
      and diagnostics["owning_physics_generation"] == 7)
runtime.playback_lock.clear(reason="test_finished")
check("release retains unrelated channels", not runtime.playback_lock.is_active() and control.location.x == 3 and rig.location.x == 4)
(out / "results.json").write_text(json.dumps({"checks": checks}, indent=2))
print("PRESENTATION_EDIT_ADMISSION_OK", len(checks), flush=True)

# SPDX-License-Identifier: Apache-2.0
"""Offline Blender operator, undo, persistence and actual panel-draw acceptance.

Run background for state tests; run without --background and pass --ui after
artifact directory for Properties editor screenshots from actual UI redraws.
"""
import json
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace

import bpy
from mathutils import Matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from ovrtx_blender_example import authoring_properties as props
from ovrtx_blender_example import physics_authoring as authoring
from ovrtx_blender_example import physics_ui
from ovrtx_blender_example.physics_frames import RigidFrame
from ovrtx_blender_example.interactive_operator_state import PhysicsPlaybackLock, apply_pending_runtime_pose_mirror
from ovrtx_blender_example.blender_callback_adapters import BlenderRenderCallbackAdapter
from ovrtx_blender_example.blender_signal_translation import BlenderSignalTranslationError

out = Path(sys.argv[sys.argv.index("--")+1]).resolve()
out.mkdir(parents=True, exist_ok=True)
def report_exception(kind, error, tb):
    (out / "failure.txt").write_text("".join(traceback.format_exception(kind, error, tb)))
    sys.__excepthook__(kind, error, tb)
sys.excepthook = report_exception
props.register()
checks = []


def check(name, condition):
    assert condition, name
    checks.append(name)


def rejected(fn, text):
    try:
        fn()
    except (ValueError, RuntimeError) as error:
        check(text, text.lower() in str(error).lower())
    else:
        raise AssertionError("expected rejection: " + text)


def select(obj):
    for other in bpy.context.selected_objects:
        other.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


for operator_name in (
    "physics_add_body", "physics_create_joint", "physics_anchor",
    "physics_edit_frame", "physics_joint_type", "physics_bind_bone",
    "physics_scene_opinion", "physics_collider_shape", "physics_repair_identity",
    "physics_edit_colliders",
):
    check(f"{operator_name} tooltip", bool(getattr(bpy.ops.ov, operator_name).get_rna_type().description))


bpy.ops.mesh.primitive_cube_add(location=(2, 0, 1))
body = bpy.context.object
body.name = "AuthoringBody"
rejected(lambda: bpy.ops.ov.physics_create_joint(), "Assign OV Body first")
check("rejected joint leaves no handle", not any(o.name.startswith("OV Joint") for o in bpy.data.objects))
rejected(lambda: physics_ui.require_joint_body(body), "Assign OV Body first")
check("body operator", bpy.ops.ov.physics_add_body() == {"FINISHED"})
check("mesh body gets default convex hull collider",
      len(body.ov.physics.colliders) == 1 and body.ov.physics.colliders[0].shape == "CONVEX_HULL"
      and body.ov.physics.colliders[0].schema_opinion == "APPLY" and bool(body.ov.physics.colliders[0].persistent_id))
body.ov.physics.persistent_id = ""
other = bpy.data.objects.new("UnassignedJointEndpoint", None)
bpy.context.scene.collection.objects.link(other)
other.select_set(True)
rejected(lambda: bpy.ops.ov.physics_create_joint(), "Assign OV Body first")
check("unassigned second endpoint leaves no handle", not any(o.name.startswith("OV Joint") for o in bpy.data.objects))
check("rejected joint leaves first identity unchanged", not body.ov.physics.persistent_id)
other.select_set(False)
bpy.data.objects.remove(other, do_unlink=True)
check("joint creation operator", bpy.ops.ov.physics_create_joint() == {"FINISHED"})
check("joint creation restores missing body identity", len(body.ov.physics.persistent_id) == 32)
check("joint tooltip is documented", bool(bpy.ops.ov.physics_create_joint.get_rna_type().description))
joint_obj = bpy.context.object
joint_obj.name = "AuthoringJoint"
joint = joint_obj.ov.physics.joint
check("one joint per Empty", joint_obj.type == "EMPTY" and joint.enabled and len(joint.axes) == 6)
original_a, original_b = tuple(joint.frame_a.translation), tuple(joint.frame_b.translation)
check("creation captures both frames", original_a == (-2, 0, -1) and original_b == (0, 0, 0))
joint_obj.location.x = 3
bpy.context.view_layer.update()
check("handle stages pending", authoring.anchor_pending(joint_obj))
check("staged movement preserves frames", original_a == tuple(joint.frame_a.translation))
rejected(lambda: authoring.require_scene_ready(bpy.context.scene), "pending joint anchor")
rejected(lambda: authoring.edit_frame(joint_obj, bpy.context.scene, "A", RigidFrame()), "pending common anchor")
adapter = BlenderRenderCallbackAdapter(generation_for_scene=lambda scene: (_ for _ in ()).throw(AssertionError("must not prepare generation")))
rejected(lambda: adapter.final_render_from_scene(bpy.context.scene), "pending joint anchor")

# Persist a pending edit, proving no transient cache participates in admission.
bpy.ops.wm.save_as_mainfile(filepath=str(out / "pending.blend"))
if "--ui" not in sys.argv:
    bpy.ops.wm.open_mainfile(filepath=str(out / "pending.blend"))
joint_obj, body = bpy.data.objects["AuthoringJoint"], bpy.data.objects["AuthoringBody"]
joint = joint_obj.ov.physics.joint
select(joint_obj)
check("pending still present" if "--ui" in sys.argv else "save reload preserves pending", authoring.anchor_pending(joint_obj))
check("cancel operator", bpy.ops.ov.physics_anchor(operation="CANCEL") == {"FINISHED"})
bpy.context.view_layer.update()
check("cancel restores committed handle", not authoring.anchor_pending(joint_obj) and tuple(joint_obj.location) == (0, 0, 0))

# Explicit undo pushes make the test deterministic in background Blender too.
bpy.ops.ed.undo_push(message="Before Pending")
joint_obj.location.x = 4
bpy.context.view_layer.update()
bpy.ops.ed.undo_push(message="Pending Anchor")
bpy.ops.ov.physics_anchor(operation="APPLY")
bpy.ops.ed.undo_push(message="Applied Anchor")
check("apply stores both frames", tuple(joint.frame_a.translation) == (2, 0, -1) and tuple(joint.frame_b.translation) == (4, 0, 0))
bpy.ops.ed.undo()
joint_obj = bpy.data.objects["AuthoringJoint"]
check("undo restores pending and old frame", authoring.anchor_pending(joint_obj) and tuple(joint_obj.ov.physics.joint.frame_a.translation) == original_a)
bpy.ops.ed.redo()
joint_obj, body = bpy.data.objects["AuthoringJoint"], bpy.data.objects["AuthoringBody"]
joint = joint_obj.ov.physics.joint
check("redo restores committed frames", not authoring.anchor_pending(joint_obj))
select(joint_obj)
joint_obj.location.x = 5
bpy.context.view_layer.update()
bpy.ops.ed.undo_push(message="Pending Cancel")
bpy.ops.ov.physics_anchor(operation="CANCEL")
bpy.ops.ed.undo_push(message="Cancelled Anchor")
bpy.ops.ed.undo()
joint_obj = bpy.data.objects["AuthoringJoint"]
check("undo cancel restores pending", authoring.anchor_pending(joint_obj))
bpy.ops.ed.redo()
joint_obj, body = bpy.data.objects["AuthoringJoint"], bpy.data.objects["AuthoringBody"]
joint = joint_obj.ov.physics.joint
select(joint_obj)
check("redo cancel restores committed handle", not authoring.anchor_pending(joint_obj))
joint_obj.location.x = 6
bpy.context.view_layer.update()
joint_obj.location.x = 4
bpy.context.view_layer.update()
check("transform cancellation restores pending state", not authoring.anchor_pending(joint_obj))
frame_before_body_move = tuple(joint.frame_a.translation)
body.location.y = 2
bpy.context.view_layer.update()
check("body motion never recaptures anchor", tuple(joint.frame_a.translation) == frame_before_body_move and not authoring.anchor_pending(joint_obj))
saved_handle = tuple(joint.committed_handle.translation)
saved_b = tuple(joint.frame_b.translation)
check("single frame operator", bpy.ops.ov.physics_edit_frame(side="A", translation=(.1, .2, .3), rotation=(2, 0, 0, 0)) == {"FINISHED"})
check("single frame preserves other and handle", tuple(joint.frame_b.translation) == saved_b and tuple(joint.committed_handle.translation) == saved_handle)
check("single frame rotation normalized", tuple(joint.frame_a.rotation) == (1, 0, 0, 0))
joint_obj.scale = (2, 2, 2)
bpy.context.view_layer.update()
rejected(lambda: authoring.apply_anchor(joint_obj, bpy.context.scene, bpy.context.evaluated_depsgraph_get()), "scale must be one")
joint_obj.scale = (1, 1, 1)
bpy.context.view_layer.update()

# Semantic axis settings survive type/axis switches; animation is diagnosed.
bpy.ops.ov.physics_joint_type(joint_type="REVOLUTE", axis="X")
joint.axes["rotX"].drive.target_position = .6
joint.axes["rotX"].drive.keyframe_insert(data_path="target_position", frame=1)
bpy.ops.ov.physics_joint_type(joint_type="PRISMATIC", axis="Z")
check("inactive animation diagnosed", bool(authoring.inactive_animation(joint_obj)))
check("type switch preserves semantic settings", abs(joint.axes["rotX"].drive.target_position-.6) < 1e-6)
rejected(lambda: authoring.require_scene_ready(bpy.context.scene), "inactive animated")
bpy.ops.ov.physics_joint_type(joint_type="REVOLUTE", axis="X")
check("reactivated semantic animation valid", not authoring.inactive_animation(joint_obj))

# Collection paths: survive deletion and keep named semantic identity.
select(body)
bpy.ops.ov.physics_edit_colliders()
bpy.ops.ov.physics_edit_colliders()
bpy.ops.ov.physics_edit_colliders()
colliders = body.ov.physics.colliders
ids = [r.persistent_id for r in colliders]
body.keyframe_insert(data_path="ov.physics.colliders[1].radius_m", frame=1)
body.keyframe_insert(data_path="ov.physics.colliders[2].radius_m", frame=1)
bpy.ops.ov.physics_edit_colliders(remove_id=ids[1])
paths = [c.data_path for _, c in authoring.action_curves(body.animation_data.action)]
check("removed collider channels removed", not any(ids[1] in path for path in paths))
check("survivor numeric animation repaired", paths == ['ov.physics.colliders["' + ids[2] + '"].radius_m'])
check("survivor path resolves", abs(body.path_resolve(paths[0])-.5) < 1e-6)
bpy.ops.ov.physics_edit_colliders(move_id=ids[2], direction=-1)
check("reorder retains semantic animation", body.ov.physics.colliders[0].persistent_id == ids[2] and abs(body.path_resolve(paths[0])-.5) < 1e-6)
# External driver references are repaired, but deleting a required input fails
# before removing records or any animation channels.
driver_owner = bpy.data.objects.new("ColliderDriver", None)
bpy.context.scene.collection.objects.link(driver_owner)
curve = driver_owner.driver_add("location", 0)
variable = curve.driver.variables.new()
variable.name = "radius"
variable.targets[0].id = body
variable.targets[0].data_path = "ov.physics.colliders[0].radius_m"
curve.driver.expression = "radius"
bpy.ops.ov.physics_edit_colliders(move_id=ids[2], direction=1)
check("external driver path repaired", ids[2] in variable.targets[0].data_path)
before_count = len(body.ov.physics.colliders)
rejected(lambda: bpy.ops.ov.physics_edit_colliders(remove_id=ids[2]), "driver input")
check("driver rejection retains collection", len(body.ov.physics.colliders) == before_count)

# Existing runtime lock owns joint handles and restores their prior channel locks.
select(joint_obj)
joint_obj.lock_location = (True, False, False)
locks = PhysicsPlaybackLock()
pending = {"scene_name": bpy.context.scene.name, "poses_by_path": {}, "lock_runtime_owned": True}
body["ovrtx.usd_prim_path"] = "/AuthoringBody"
pending["poses_by_path"] = {"/AuthoringBody": {"translate": tuple(body.location), "orient": (0, 0, 0, 1)}}
pending["owning_generation"] = 3
handle_before_mirror = tuple(tuple(row) for row in joint_obj.matrix_world)
apply_pending_runtime_pose_mirror(bpy, pending, locks)
check("runtime handle locked", tuple(joint_obj.lock_location) == (True, True, True))
check("runtime never moves joint handle", tuple(tuple(row) for row in joint_obj.matrix_world) == handle_before_mirror)
engine_key = "ovrtx_blender_example.engine"
prior_engine = sys.modules.get(engine_key)
sys.modules[engine_key] = SimpleNamespace(_live_viewport_engines=lambda: [SimpleNamespace(_physics_playback_lock=locks)])
try:
    rejected(lambda: authoring.apply_anchor(joint_obj, bpy.context.scene), "release pose ownership")
    rejected(lambda: bpy.ops.ov.physics_joint_type(joint_type="FIXED", axis="X"), "release pose ownership")
finally:
    if prior_engine is None:
        del sys.modules[engine_key]
    else:
        sys.modules[engine_key] = prior_engine
for reason in ("reset", "stop", "error"):
    locks.clear(reason=reason)
    check(reason + " restores handle locks", tuple(joint_obj.lock_location) == (True, False, False))
    locks.lock_object("ov-joint:" + joint.persistent_id, joint_obj, generation=3)
locks.clear(reason="done")

# The same user-facing body/collider/joint operations support pose-bone owners.
bpy.ops.object.armature_add()
rig = bpy.context.object
rig.name = "AuthoringRig"
bpy.ops.object.mode_set(mode="POSE")
check("pose body operator", bpy.ops.ov.physics_add_body() == {"FINISHED"})
check("pose collider operator", bpy.ops.ov.physics_edit_colliders() == {"FINISHED"})
check("bone collider authority", len(rig.pose.bones[0].ov.physics.colliders) == 1)
check("bone joint creation operator", bpy.ops.ov.physics_create_joint() == {"FINISHED"})
bone_joint = bpy.context.object.ov.physics.joint
check("bone joint uses stable endpoint", bone_joint.endpoint_a.kind == "BONE" and bone_joint.endpoint_a.bone_id == rig.data.bones[0].ov.persistent_id)

material = bpy.data.materials.new("AuthoringMaterial")
body.data.materials.append(material)
check("all panels registered", all(getattr(cls, "is_registered", False) for cls in physics_ui.CLASSES))
check("no Blender rigid-body data", bpy.context.scene.rigidbody_world is None and all(o.rigid_body is None and o.rigid_body_constraint is None for o in bpy.context.scene.objects))
check("no constraint stacks changed", not body.constraints and not joint_obj.constraints)
bpy.ops.wm.save_as_mainfile(filepath=str(out / "authoring.blend"))
report = {"ok": True, "blender": bpy.app.version_string, "checks": checks}
(out / "report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report))

if "--ui" in sys.argv:
    # The real Properties editor invokes the registered draw callbacks on redraw.
    area = max(bpy.context.screen.areas, key=lambda a: a.width*a.height)
    area.type = "PROPERTIES"
    area.spaces.active.context = "PHYSICS"
    select(joint_obj)
    screens = [("joint", "PHYSICS", joint_obj), ("body-colliders", "PHYSICS", body),
               ("material", "MATERIAL", body), ("scene", "SCENE", body)]
    capture_index = [0]
    def capture():
        index = capture_index[0]
        with bpy.context.temp_override(area=area):
            bpy.ops.screen.screenshot(filepath=str(out / (screens[index][0] + "-panels.png")))
        capture_index[0] += 1
        if capture_index[0] < len(screens):
            _, panel_context, obj = screens[capture_index[0]]
            select(obj)
            area.spaces.active.context = panel_context
            area.spaces.active.search_filter = "OV Physics Material" if panel_context == "MATERIAL" else ""
            area.tag_redraw()
            return 1.0
        (out / "ui-complete.json").write_text(json.dumps({"ok": True, "editor": area.type, "context": area.spaces.active.context}))
        bpy.ops.wm.quit_blender()
    bpy.app.timers.register(capture, first_interval=3)

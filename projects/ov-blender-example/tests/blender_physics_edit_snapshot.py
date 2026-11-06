"""Real RNA edit classification and complete composed-scene restart regression."""
import json
from pathlib import Path
import sys
import uuid
import bpy
from pxr import Usd, UsdPhysics, UsdGeom
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from ovrtx_blender_example import authoring_properties as props
from ovrtx_blender_example import physics_edit_snapshot as snapshots
from ovrtx_blender_example import scene_generation_sessions as sessions
from ovrtx_blender_example import interactive_operator_state as suppression
from ovrtx_blender_example.physics_endpoints import initialize_owner, initialize_joint
from ovrtx_blender_example.physics_authoring import apply_anchor, cancel_anchor, require_scene_ready
from ovrtx_blender_example import engine
from ovrtx_blender_example.blender_interactive_edit_builders import build_interactive_edits_from_depsgraph
from types import SimpleNamespace
props.register()
out = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
out.mkdir(parents=True, exist_ok=True)
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
scene = bpy.context.scene
checks = []
def check(label, condition):
    assert condition, label
    checks.append(label)

# Compare against the former full-schema predicate on real persisted RNA.
# Default pointer reads create containers; explicit default/null opinions and
# empty removed collections must retain their previous participation semantics.
audit = bpy.data.objects.new('Sparse RNA participation audit', None)
scene.collection.objects.link(audit)
def sparse_parity(label):
    previous = any(explicit and not path.endswith('show_indicator')
                   or path.endswith('.__count__') and value
                   for path, explicit, value in snapshots._leaves(audit.ov.physics))
    check(label, snapshots._authored(audit.ov.physics) == bool(previous))
sparse_parity('sparse default containers match full predicate')
audit.ov.physics.joint.show_indicator = False
sparse_parity('sparse presentation flag does not create participant')
sessions._physics_authoring_dirty.clear()
audit.ov.physics.rigid_body.mass_kg = audit.ov.physics.rigid_body.mass_kg
check('registered nested scalar latches authoring', scene.session_uid in sessions._physics_authoring_dirty)
sparse_parity('sparse explicit default mass remains authored')
audit.ov.physics.rigid_body.property_unset('mass_kg')
sparse_parity('sparse mass reset removes opinion')
audit.ov.physics.joint.endpoint_a.object = None
sparse_parity('sparse explicit null endpoint remains authored')
audit.ov.physics.joint.endpoint_a.property_unset('object')
sparse_parity('sparse null endpoint reset removes opinion')
audit.ov.physics.colliders.add()
sparse_parity('sparse default collection member remains authored')
audit.ov.physics.colliders.remove(0)
sparse_parity('sparse explicit empty collection remains authored')
bpy.data.objects.remove(audit, do_unlink=True)
def body(name, x):
    bpy.ops.mesh.primitive_cube_add(location=(x, 0, 2))
    obj = bpy.context.object
    obj.name = name
    initialize_owner(obj)
    obj.ov.physics.rigid_body.schema_opinion = "APPLY"
    c = obj.ov.physics.colliders.add()
    c.name = c.persistent_id = uuid.uuid4().hex
    c.schema_opinion = "APPLY"
    c.shape = "ANALYTIC_SPHERE"
    return obj
a, b, contact = body("A", 0), body("B", 3), body("UnconnectedContact", 1)
handle = bpy.data.objects.new("Joint", None)
scene.collection.objects.link(handle)
joint = initialize_joint(handle)
joint.enabled = True
joint.endpoint_a.object, joint.endpoint_b.object = a, b
bpy.context.view_layer.update()
apply_anchor(handle, scene, bpy.context.evaluated_depsgraph_get())
uid = scene.session_uid
base = sessions.generation_for_scene(scene, work_root=out / "generations")
owner = sessions._owners[uid]
def observe():
    return sessions.observe_physics_edits(scene)
check("baseline stable after export", not observe().restart)
bpy.ops.object.camera_add()
ordinary_camera = bpy.context.object
check("ordinary camera membership is not physics restart", not observe().restart)
check("ordinary camera queues no physics intent", uid not in sessions._physics_restart_pending)
ordinary_camera.ov.physics.rigid_body.mass_kg = ordinary_camera.ov.physics.rigid_body.mass_kg
check("explicit default physics opinion on prior nonparticipant detected", observe().restart)
bpy.data.objects.remove(ordinary_camera, do_unlink=True)
observe()
sessions._physics_restart_pending.clear()
sessions._dirty.clear()
# Invoke the actual registered callback against actual RNA. Forbidden work is
# trapped here so a classifier helper alone cannot establish callback safety.
original_submit = engine.submit_depsgraph_interactive_edits_to_active_viewports
def forbidden(*args, **kwargs):
    raise AssertionError("physics callback attempted value/RPC submission")
engine.submit_depsgraph_interactive_edits_to_active_viewports = forbidden
joint.break_force = 43
engine._live_interactive_edit_depsgraph_handler(scene, bpy.context.evaluated_depsgraph_get())
check("actual callback queues physics intent", uid in sessions._physics_restart_pending)
check("actual callback performs no value submission", not suppression.interactive_edit_bridge_diagnostics()["last_error"])
engine.submit_depsgraph_interactive_edits_to_active_viewports = original_submit
joint.break_force = 1e10
observe()
sessions._physics_restart_pending.clear()
before = snapshots.capture(scene)
check("unset material pointer read preserves opinion", not a.ov.physics.colliders[0].is_property_set("physics_material"))
joint.show_indicator = False
changes = observe()
check("display redraw only", changes.redraw and not changes.restart)
check("display does not replace generation", sessions.generation_for_scene(scene) is base)
for field in ("indicators_enabled", "indicators_selected_only", "indicators_colliders", "indicators_size"):
    prior = getattr(scene.ov, field)
    setattr(scene.ov, field, prior * 2 if field == "indicators_size" else not prior)
    changes = observe()
    check(field + " redraw only", changes.redraw and not changes.restart and not changes.structural)
    check(field + " queues no restart", uid not in sessions._physics_restart_pending)
    check(field + " preserves accepted generation", sessions.generation_for_scene(scene) is base and owner.current_generation is base)
    check(field + " creates no pending generation", owner.pending_generation is None)
handle.location.x += 1
bpy.context.view_layer.update()
changes = observe()
check("handle stages anchor only", changes.pending_anchor and not changes.restart)
check("pending handle leaves generation unchanged", sessions.generation_for_scene(scene) is base)
edits = build_interactive_edits_from_depsgraph(SimpleNamespace(updates=[SimpleNamespace(id=handle)]))
check("joint handles cannot enter live body/render transport", not edits)
try:
    require_scene_ready(scene)
except ValueError:
    check("pending anchor blocks simulation and bake", True)
else:
    check("pending anchor blocks simulation and bake", False)
cancel_anchor(handle, scene)
bpy.context.view_layer.update()
observe()
sessions._dirty.clear()
joint.break_force = 42
changes = observe()
check("joint value needs explicit restart", changes.restart and not changes.structural)
check("viewport retains accepted joint generation", sessions.generation_for_scene(scene) is base)
candidate = sessions.generation_for_scene(scene, explicit_physics_restart=True)
check("restart produces pending candidate", owner.pending_generation is candidate and owner.current_generation is base)
stage = Usd.Stage.Open(candidate.usd_path)
check("all bodies retained including unconnected contact", sum(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in stage.Traverse()) == 3)
joint_prim = next(p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint))
check("restart authors changed joint value", joint_prim.GetAttribute("physics:breakForce").Get() == 42)
owner.accept(candidate)
sessions._pending_affected.clear()
with suppression.suppress_interactive_edit_bridge():
    mirrored = suppression.apply_pending_runtime_pose_mirror(bpy, {
        "scene_name": scene.name,
        "object_paths_by_session_uid": {a.session_uid: "/World/A"},
        "poses_by_path": {"/World/A": {"translate": (10, 0, 2), "orient": (0, 0, 0, 1)}},
        "lock_runtime_owned": False,
    }, None)
check("real presentation mirror applied", mirrored["status"] == "applied" and a.location.x == 10)
check("delayed presentation update suppressed", not observe().body_input and not sessions._physics_restart_pending)
displayed_basis = a.matrix_basis.copy()
joint.break_force = 73
observe()
restarted = sessions.generation_for_scene(scene, explicit_physics_restart=True)
restarted_stage = Usd.Stage.Open(restarted.usd_path)
position = UsdGeom.XformCache().GetLocalToWorldTransform(restarted_stage.GetPrimAtPath("/World/A")).ExtractTranslation()
check("restart uses authoring initial pose instead of presentation", abs(position[0]) < 1e-5)
check("restart restores displayed transform", a.matrix_basis == displayed_basis)
owner.accept(restarted)
sessions._pending_affected.clear()
check("restart restoration emits no authoring feedback", not observe().body_input)
joint.endpoint_b.object = a
observe()
try:
    sessions.generation_for_scene(scene, explicit_physics_restart=True)
except Exception:
    check("failed restart restores displayed transform", a.matrix_basis == displayed_basis)
else:
    check("invalid restart rejected", False)
joint.endpoint_b.object = b
observe()
sessions._physics_restart_pending.clear()
a.location.x += 1
bpy.context.view_layer.update()
changes = observe()
check("authoring body pose uses existing input path", changes.body_input and not changes.restart)
joint.break_force = 74
observe()
authored = sessions.generation_for_scene(scene, explicit_physics_restart=True)
authored_stage = Usd.Stage.Open(authored.usd_path)
position = UsdGeom.XformCache().GetLocalToWorldTransform(authored_stage.GetPrimAtPath("/World/A")).ExtractTranslation()
check("later durable body edit updates initial transform", abs(position[0] - a.location.x) < 1e-5)
owner.accept(authored)
sessions._pending_affected.clear()
with suppression.suppress_interactive_edit_bridge():
    joint.break_torque = 17
check("presentation suppression never swallows authoring settings", observe().restart)
scene.ov.physics_scene.gravity_magnitude = 3
check("scene gravity snapshot", observe().restart)
joint.axes[0].drive.stiffness = 15
check("nested drive snapshot", observe().restart)
a.ov.physics.colliders[0].offset.translation[0] = .1
check("collider local frame snapshot", observe().restart)
a.scale = (2, 2, 2)
bpy.context.view_layer.update()
check("body scale requires geometry recompile", observe().structural)
a.scale = (1, 1, 1)
bpy.context.view_layer.update()
observe()
collider = a.ov.physics.colliders.add()
collider.name = collider.persistent_id = uuid.uuid4().hex
collider.shape = "ANALYTIC_SPHERE"
collider.schema_opinion = "APPLY"
sessions._poll_physics_edits()
check("Python collection add detected without depsgraph", uid in sessions._physics_restart_pending)
a.ov.physics.colliders.remove(1)
check("Python collection remove detected", observe().structural)
material = bpy.data.materials.new("PhysicsOnly")
a.ov.physics.colliders[0].physics_material = material
check("physics material dependency detected", observe().restart)
material.ov.physics.static_friction = .8
check("physics-only material value detected", observe().restart)
old = snapshots.capture(scene)
a.ov.physics.rigid_body.mass_kg = 0
check("explicit default opinion differs from unset", snapshots.classify(old, snapshots.capture(scene)).restart)
scene.unit_settings.scale_length = .01
check("scene units require structural restart", observe().structural)
scene.unit_settings.scale_length = 1
observe()
dup = a.copy()
scene.collection.objects.link(dup)
check("duplicate identity detected before compilation", observe().structural)
scene.collection.objects.unlink(dup)
check("unlink detected", observe().structural)
bpy.data.objects.remove(dup)
# Rest changes and PoseBone property groups are copied independently of rig instances.
bpy.ops.object.armature_add()
rig = bpy.context.object
observe()
rig.pose.bones[0].ov.physics.rigid_body.mass_kg = 4
check("bone body setting detected", observe().restart)
bpy.ops.object.mode_set(mode="EDIT")
rig.data.edit_bones[0].tail.x += .2
bpy.ops.object.mode_set(mode="OBJECT")
check("bone rest edit detected", observe().structural)
bpy.ops.object.mode_set(mode="EDIT")
rig.data.edit_bones[0].length *= 2
bpy.ops.object.mode_set(mode="OBJECT")
check("bone length-only rest edit detected", observe().structural)
for collection in tuple(rig.users_collection):
    collection.objects.unlink(rig)
observe()
# A failed complete rebuild retains accepted state and restart intent.
joint.endpoint_b.object = a
observe()
accepted = owner.current_generation
try:
    sessions.generation_for_scene(scene, explicit_physics_restart=True)
except Exception:
    check("failed restart preserves accepted generation", owner.current_generation is accepted)
    check("failed restart preserves dirty intent", uid in sessions._physics_restart_pending)
else:
    check("invalid endpoint rebuild rejected", False)
joint.endpoint_b.object = b
observe()
stage.Flatten().Export(str(out / "complete-contact-scene.usda"))
# Undo/redo replaces RNA wrappers; identity is session UID, never wrapper address.
bpy.context.preferences.edit.use_global_undo = True
bpy.ops.ed.undo_push(message="Before Physics Edit")
joint.break_force = 55
bpy.ops.ed.undo_push(message="After Physics Edit")
observe()
bpy.ops.ed.undo()
scene = bpy.context.scene
handle = bpy.data.objects["Joint"]
joint = handle.ov.physics.joint
check("undo detected", observe().restart)
bpy.ops.ed.redo()
scene = bpy.context.scene
handle = bpy.data.objects["Joint"]
joint = handle.ov.physics.joint
check("redo detected", observe().restart)
handle.location.x += 1
bpy.context.view_layer.update()
observe()
bpy.ops.wm.save_as_mainfile(filepath=str(out / "snapshot.blend"))
saved = snapshots.capture(scene)
bpy.ops.wm.open_mainfile(filepath=str(out / "snapshot.blend"))
scene = bpy.context.scene
handle = bpy.data.objects["Joint"]
try:
    require_scene_ready(scene)
except ValueError:
    check("pending anchor persists after reload", True)
else:
    check("pending anchor persists after reload", False)
sessions._physics_snapshots.clear()
sessions._physics_restart_pending.clear()
check("reload establishes fresh snapshot without stale wrappers", not observe().restart)
bpy.data.objects.remove(handle, do_unlink=True)
check("deletion detected", observe().structural)
(out / "results.json").write_text(json.dumps({"blender": bpy.app.version_string, "checks": checks}, indent=2))
print(f"PHYSICS_EDIT_SNAPSHOT_OK {len(checks)} checks")

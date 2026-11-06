# SPDX-License-Identifier: Apache-2.0
"""Duplicated physics Objects are reidentified automatically; the original keeps its ID."""
from pathlib import Path
import json
import sys
from types import SimpleNamespace
import uuid

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from ovrtx_blender_example import authoring_properties as props
from ovrtx_blender_example import scene_generation_sessions as sessions
from ovrtx_blender_example import user_messages
from ovrtx_blender_example.physics_authoring import apply_anchor
from ovrtx_blender_example.physics_endpoints import initialize_joint, initialize_owner

props.register()
out = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
out.mkdir(parents=True, exist_ok=True)
checks = []


def check(label, value):
    assert value, label
    checks.append(label)


def body(scene, name, location):
    obj = bpy.data.objects.new(name, None)
    scene.collection.objects.link(obj)
    obj.location = location
    initialize_owner(obj)
    physics = obj.ov.physics
    physics.rigid_body.schema_opinion = "APPLY"
    physics.rigid_body.mass_kg = 1
    collider = physics.colliders.add()
    collider.name = collider.persistent_id = uuid.uuid4().hex
    collider.schema_opinion = "APPLY"
    collider.shape = "ANALYTIC_SPHERE"
    return obj


def duplicate(obj):
    """Shift+D equivalent: copies the Object and its ID properties."""
    copy = obj.copy()
    obj.users_collection[0].objects.link(copy)
    return copy


def poll():
    sessions._poll_physics_edits()


def reported():
    return [m.text for m in user_messages.default_bus().take_pending()]


def ids(obj):
    p = obj.ov.physics
    return p.persistent_id, p.rigid_body.persistent_id, p.joint.persistent_id


def reset():
    bpy.ops.wm.read_homefile(use_empty=True)
    props.register()
    sessions._identity_owners.clear()
    user_messages.default_bus().reset()
    return bpy.context.scene


# In-session duplication of a body, a joint endpoint body, and a joint handle.
scene = reset()
anchor = body(scene, "Anchor", (0, 0, 2))
hanging = body(scene, "Hanging", (0, 1, 2))
handle = bpy.data.objects.new("Joint", None)
scene.collection.objects.link(handle)
handle.location = (0, 0.5, 2)
joint = initialize_joint(handle)
joint.enabled = True
joint.endpoint_a.object, joint.endpoint_b.object = anchor, hanging
bpy.context.view_layer.update()
apply_anchor(handle, scene, bpy.context.evaluated_depsgraph_get())
poll()  # First observation records the original owners.
originals = {obj.name: ids(obj) for obj in (anchor, hanging, handle)}
copies = [duplicate(obj) for obj in (hanging, handle, anchor)]
poll()
check("originals keep every identity", all(ids(scene.objects[n]) == v for n, v in originals.items()))
check("copies receive new identities",
      all(ids(copy)[0] not in {v[0] for v in originals.values()} for copy in copies))
check("copy body identities are unique", len({ids(o)[1] for o in scene.objects}) == len(scene.objects))
check("copied joint receives its own joint identity", ids(copies[1])[2] and ids(copies[1])[2] != originals["Joint"][2])
check("original joint endpoints are unchanged", joint.endpoint_a.object == anchor and joint.endpoint_b.object == hanging)
messages = reported()
check("repair is reported once", len(messages) == 1 and all(copy.name in messages[0] for copy in copies))
first = {copy.name: ids(copy) for copy in copies}
poll()
check("stable after repair", {copy.name: ids(copy) for copy in copies} == first and reported() == [])

# The deterministic repair gives the same identity when replayed (redo).
replay = duplicate(anchor)
replay_name = replay.name
poll()
replay_id = ids(replay)
bpy.data.objects.remove(replay, do_unlink=True)
replay = duplicate(anchor)
replay.name = replay_name
poll()
check("repeated repair is deterministic", ids(replay) == replay_id)
bpy.data.objects.remove(replay, do_unlink=True)
reported()

# Play after an unrepaired duplicate builds without user action.
PlaybackEngine = type("PlaybackEngine", (bpy.types.RenderEngine,),
                      {"bl_idname": "OVRTX_EXAMPLE", "bl_label": "OVRTX playback test"})
bpy.utils.register_class(PlaybackEngine)
scene.render.engine = "OVRTX_EXAMPLE"
sessions._runtime_for_owner = lambda uid, owner, **_: sessions._runtimes.setdefault(
    uid, SimpleNamespace(scheduler=None, reuse_blocked=False))
sessions._wake_preparation = lambda uid: None
uid = int(scene.session_uid)
try:
    late = duplicate(hanging)
    sessions.animation_playback_pre(scene)
    check("Play repairs a fresh duplicate", ids(late)[0] != originals["Hanging"][0])
    check("Play builds after duplication", sessions.physics_playback_error(scene) == ""
          and sessions._owners[uid].reuse() is not None)
finally:
    sessions._runtimes.pop(uid, None)
    owner = sessions._owners.pop(uid, None)
    if owner is not None:
        owner.close()
    scene.render.engine = "BLENDER_EEVEE"
    bpy.utils.unregister_class(PlaybackEngine)

# Play with a pending anchor, before any poll, keeps the original's identity.
scene = reset()
hanging = body(scene, "Hanging", (0, 1, 2))
handle = bpy.data.objects.new("PendingJoint", None)
scene.collection.objects.link(handle)
handle.location = (0, 0, 2)
joint = initialize_joint(handle)
joint.enabled = True
joint.type, joint.axis = "REVOLUTE", "X"
joint.endpoint_a.kind = "WORLD"
joint.endpoint_b.object = hanging
bpy.context.view_layer.update()
apply_anchor(handle, scene, bpy.context.evaluated_depsgraph_get())
poll()
original = ids(hanging)
handle.location.z += 0.25  # Stage a pending anchor edit.
bpy.context.view_layer.update()
copy = duplicate(hanging)
bpy.utils.register_class(PlaybackEngine)
scene.render.engine = "OVRTX_EXAMPLE"
uid = int(scene.session_uid)
try:
    sessions.animation_playback_pre(scene)
    check("pending-anchor Play succeeds", sessions.physics_playback_error(scene) == "")
    check("pending-anchor Play keeps the original's identity", ids(hanging) == original)
    check("pending-anchor Play reidentifies the copy", ids(copy)[0] != original[0])
finally:
    sessions._runtimes.pop(uid, None)
    owner = sessions._owners.pop(uid, None)
    if owner is not None:
        owner.close()
    scene.render.engine = "BLENDER_EEVEE"
    bpy.utils.unregister_class(PlaybackEngine)

# A malformed shared identity never stops the physics poll.
scene = reset()
odd = body(scene, "Odd", (0, 0, 0))
odd.ov.physics.persistent_id = "legacy-id"
poll()
duplicate(odd)
try:
    poll()
    check("malformed duplicate identity does not raise from the poll", True)
except Exception as exc:
    check(f"malformed duplicate identity does not raise from the poll: {exc!r}", False)
check("malformed identities are left for explicit repair",
      [o.ov.physics.persistent_id for o in scene.objects] == ["legacy-id", "legacy-id"])

# A saved file with collisions: without history the first-loaded Object keeps it.
scene = reset()
source = body(scene, "Cylinder", (0, 0, 0))
for name in ("Cylinder.001", "Cylinder.002"):
    duplicate(source).name = name
shared = ids(source)
path = out / "collisions.blend"
bpy.ops.wm.save_as_mainfile(filepath=str(path))
bpy.ops.wm.open_mainfile(filepath=str(path))
scene = bpy.context.scene
sessions._identity_owners.clear()
user_messages.default_bus().reset()
loaded = sorted((o for o in scene.objects if o.name.startswith("Cylinder")), key=lambda o: int(o.session_uid))
check("loaded collision precondition", all(ids(o)[0] == shared[0] for o in loaded))
poll()
check("lowest session uid keeps the identity", ids(loaded[0])[0] == shared[0])
check("later loaded copies are reidentified", len({ids(o)[0] for o in loaded}) == 3)
check("loaded repair is reported", any("Cylinder" in text for text in reported()))

# Linked data is never rewritten; the collision stays for explicit handling.
scene = reset()
linked_source = body(scene, "LinkedBody", (0, 0, 0))
library_path = out / "library.blend"
bpy.data.libraries.write(str(library_path), {linked_source})
shared = ids(linked_source)
with bpy.data.libraries.load(str(library_path), link=True) as (data_from, data_to):
    data_to.objects = ["LinkedBody"]
linked = data_to.objects[0]
scene.collection.objects.link(linked)
check("linked precondition", linked.library is not None and ids(linked)[0] == shared[0])
poll()
check("linked data is not rewritten", ids(linked) == shared)
check("local original keeps its identity", ids(linked_source)[0] == shared[0])

# Armature Objects are reidentified; bone identities are never rewritten.
scene = reset()
armature = bpy.data.armatures.new("Rig")
rig = bpy.data.objects.new("Rig", armature)
scene.collection.objects.link(rig)
initialize_owner(rig)
bone_ids = {}
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode="EDIT")
edit = armature.edit_bones.new("Bone")
edit.tail = (0, 0, 1)
bpy.ops.object.mode_set(mode="OBJECT")
armature.bones["Bone"].ov.persistent_id = uuid.uuid4().hex
bone_ids = {b.name: b.ov.persistent_id for b in armature.bones}
poll()
rig_copy = duplicate(rig)
poll()
check("armature copy Object is reidentified", ids(rig_copy)[0] != ids(rig)[0])
check("bone identities are untouched",
      {b.name: b.ov.persistent_id for b in rig_copy.data.bones} == bone_ids
      and {b.name: b.ov.persistent_id for b in armature.bones} == bone_ids)
check("no native rigid body world", scene.rigidbody_world is None)

(out / "results.json").write_text(json.dumps({"blender": bpy.app.version_string, "checks": checks}, indent=2))
print("PHYSICS_DUPLICATE_IDENTITY_OK", len(checks))

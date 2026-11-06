# SPDX-License-Identifier: Apache-2.0
"""A rejected Play-time rebuild is reported and keeps the accepted generation."""
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
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
scene = bpy.context.scene
uid = int(scene.session_uid)
checks = []


def check(label, value):
    assert value, label
    checks.append(label)


class PlaybackEngine(bpy.types.RenderEngine):
    # Playback handling is gated on the OVRTX engine identifier.
    bl_idname = "OVRTX_EXAMPLE"
    bl_label = "OVRTX playback test"


bpy.utils.register_class(PlaybackEngine)
scene.render.engine = "OVRTX_EXAMPLE"


def body(name, location):
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


hanging = body("Hanging", (0, 1, 2))
handle = bpy.data.objects.new("Joint", None)
scene.collection.objects.link(handle)
handle.location = (0, 0, 2)
joint = initialize_joint(handle)
joint.enabled = True
joint.type, joint.axis = "REVOLUTE", "X"
joint.endpoint_a.kind = "WORLD"
joint.endpoint_b.object = hanging
bpy.context.view_layer.update()
apply_anchor(handle, scene, bpy.context.evaluated_depsgraph_get())

# Physics preparation needs a worker; playback admission and reporting do not.
sessions._runtime_for_owner = lambda uid, owner, **_: sessions._runtimes.setdefault(
    uid, SimpleNamespace(scheduler=None, reuse_blocked=False))
sessions._wake_preparation = lambda uid: None
user_messages.default_bus().reset()


def messages():
    return [(m.level, m.text) for m in user_messages.default_bus().take_pending()]


try:
    sessions.animation_playback_pre(scene)
    owner = sessions._owners[uid]
    accepted = owner.current_generation
    check("valid Play builds without an error", accepted is not None and sessions.physics_playback_error(scene) == "")
    check("valid Play is silent", messages() == [])

    hanging.ov.physics.rigid_body.enabled = False
    sessions.observe_physics_edits(scene)
    sessions.animation_playback_pre(scene)
    reported = messages()
    check("rejected Play reports one error",
          len(reported) == 1 and reported[0][0] == "ERROR"
          and "Physics not updated" in reported[0][1] and "no enabled authored OV body" in reported[0][1])
    check("rejected Play shows viewport status", "no enabled authored OV body" in sessions.physics_playback_error(scene))
    check("rejected Play keeps accepted generation",
          owner.current_generation is accepted and getattr(owner, "pending_generation", None) is None)
    sessions.animation_playback_pre(scene)
    check("unchanged error is not repeated", messages() == [])

    hanging.ov.physics.rigid_body.enabled = True
    sessions.observe_physics_edits(scene)
    sessions.animation_playback_pre(scene)
    check("repaired Play clears the status", sessions.physics_playback_error(scene) == "")
    check("repaired Play reports resolution once", [level for level, _ in messages()] == ["INFO"])
    # Re-enabling restores the accepted scene exactly, so it is reused.
    check("repaired Play resolves to the accepted scene", owner.reuse().digest == accepted.digest)

    hanging.ov.physics.rigid_body.enabled = False
    sessions.observe_physics_edits(scene)
    sessions.animation_playback_pre(scene)
    check("recurrence after success is reported again", [level for level, _ in messages()] == ["ERROR"])
finally:
    sessions._runtimes.pop(uid, None)
    owner = sessions._owners.pop(uid, None)
    if owner is not None:
        owner.close()
    scene.render.engine = "BLENDER_EEVEE"
    bpy.utils.unregister_class(PlaybackEngine)

(out / "results.json").write_text(json.dumps({"blender": bpy.app.version_string, "checks": checks}, indent=2))
print("PHYSICS_PLAYBACK_ERRORS_OK", len(checks))

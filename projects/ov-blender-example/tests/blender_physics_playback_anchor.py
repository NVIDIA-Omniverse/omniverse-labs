# SPDX-License-Identifier: Apache-2.0
"""Playback admission commits pending handles and rolls back a failed batch."""
from pathlib import Path
import sys
from types import SimpleNamespace

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from ovrtx_blender_example import authoring_properties as props
from ovrtx_blender_example import physics_authoring as authoring
from ovrtx_blender_example import scene_generation_sessions as sessions, user_messages
from ovrtx_blender_example.blender_callback_adapters import BlenderRenderCallbackAdapter

props.register()
scene = bpy.context.scene


def make_joint(name):
    bpy.ops.mesh.primitive_cube_add()
    body = bpy.context.object
    body.name = name
    assert bpy.ops.ov.physics_add_body() == {"FINISHED"}
    assert bpy.ops.ov.physics_create_joint() == {"FINISHED"}
    return body, bpy.context.object


body_a, handle_a = make_joint("Playback Body A")
shared_id = body_a.ov.physics.persistent_id
handle_a.location.x = 1
bpy.context.view_layer.update()
assert authoring.anchor_pending(handle_a)
authoring.require_scene_ready(scene, allow_pending_anchors=True)
adapter = BlenderRenderCallbackAdapter(
    generation_for_scene=lambda _scene: None,
    viewport_generation_for_scene=lambda _scene: (_ for _ in ()).throw(RuntimeError("viewport provider reached")),
)
try:
    adapter.view_draw_from_scene(scene, bpy.context)
except RuntimeError as exc:
    assert "viewport provider reached" in str(exc), exc
else:
    raise AssertionError("Viewport did not reach scene generation with a pending anchor")
try:
    authoring.require_scene_ready(scene)
except ValueError as exc:
    assert "pending joint anchor" in str(exc)
else:
    raise AssertionError("Strict admission accepted a pending anchor")

body_b, handle_b = make_joint("Playback Body B")
handle_b.location.x = 2
handle_b.ov.physics.joint.endpoint_b.kind = "OBJECT"
handle_b.ov.physics.joint.endpoint_b.object = body_b
bpy.context.view_layer.update()
# Creating the second joint can repair existing duplicates. Introduce this
# collision immediately before the rejected batch so rollback is exercised.
copy = body_a.copy()
scene.collection.objects.link(copy)
assert copy.ov.physics.persistent_id == shared_id
history_before_failure = {key: dict(value) for key, value in sessions._identity_owners.items()}
user_messages.default_bus().take_pending()
frame_a_before = tuple(handle_a.ov.physics.joint.frame_a.translation)
try:
    authoring.apply_pending_anchors_for_playback(scene)
except ValueError as exc:
    assert "itself" in str(exc), exc
else:
    raise AssertionError("Invalid second joint was admitted")
assert authoring.anchor_pending(handle_a)
assert tuple(handle_a.ov.physics.joint.frame_a.translation) == frame_a_before
assert body_a.ov.physics.persistent_id == shared_id
assert copy.ov.physics.persistent_id == shared_id
assert handle_a.ov.physics.joint.endpoint_a.object == body_a
assert sessions._identity_owners == history_before_failure
assert not user_messages.default_bus().take_pending()

handle_b.ov.physics.joint.endpoint_b.kind = "WORLD"
handle_b.ov.physics.joint.endpoint_b.object = None
applied = authoring.apply_pending_anchors_for_playback(scene)
assert set(applied) == {handle_a.name, handle_b.name}
assert not authoring.anchor_pending(handle_a)
assert not authoring.anchor_pending(handle_b)
assert body_a.ov.physics.persistent_id == shared_id
assert copy.ov.physics.persistent_id and copy.ov.physics.persistent_id != shared_id
assert handle_a.ov.physics.joint.endpoint_a.object == body_a
assert len(user_messages.default_bus().take_pending()) == 1
authoring.require_scene_ready(scene)

class TestOvEngine(bpy.types.RenderEngine):
    bl_idname = "OVRTX_EXAMPLE"
    bl_label = "Test OVRTX"

    def render(self, _depsgraph):
        pass

bpy.utils.register_class(TestOvEngine)
handle_a.location.x += 1
bpy.context.view_layer.update()
assert authoring.anchor_pending(handle_a)
sessions.animation_playback_pre(scene)
assert authoring.anchor_pending(handle_a), "Non-OVRTX playback changed the anchor"
scene.render.engine = "OVRTX_EXAMPLE"
uid = int(scene.session_uid)
missing = object()
saved_owner = sessions._owners.get(uid, missing)
saved_runtime = sessions._runtimes.get(uid, missing)
saved_generation = sessions.generation_for_scene
saved_wake = sessions._wake_preparation
try:
    sessions._owners[uid] = object()
    sessions._runtimes[uid] = SimpleNamespace(
        reuse_blocked=False, scheduler=SimpleNamespace(baked_playback_paused=True))
    sessions.generation_for_scene = lambda _scene, **_kwargs: None
    sessions._wake_preparation = lambda _uid: None
    sessions.animation_playback_pre(scene)
finally:
    sessions.generation_for_scene = saved_generation
    sessions._wake_preparation = saved_wake
    if saved_owner is missing:
        sessions._owners.pop(uid, None)
    else:
        sessions._owners[uid] = saved_owner
    if saved_runtime is missing:
        sessions._runtimes.pop(uid, None)
    else:
        sessions._runtimes[uid] = saved_runtime
assert not authoring.anchor_pending(handle_a)
print("PLAYBACK_ANCHOR_ADMISSION_OK")

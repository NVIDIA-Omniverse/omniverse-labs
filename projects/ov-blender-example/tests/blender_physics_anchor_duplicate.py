# SPDX-License-Identifier: Apache-2.0
"""Joint anchor application repairs selected duplicates and rolls back failures."""
from pathlib import Path
import sys

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from ovrtx_blender_example import authoring_properties as props
from ovrtx_blender_example import physics_authoring as authoring
from ovrtx_blender_example import scene_generation_sessions as sessions, user_messages

props.register()
bpy.ops.mesh.primitive_cube_add()
body = bpy.context.object
assert bpy.ops.ov.physics_add_body() == {"FINISHED"}
assert bpy.ops.ov.physics_create_joint() == {"FINISHED"}
handle = bpy.context.object
joint = handle.ov.physics.joint

copy = body.copy()
copy.name = "Copied Body"
bpy.context.scene.collection.objects.link(copy)
old_identity = body.ov.physics.persistent_id
assert copy.ov.physics.persistent_id == old_identity

handle.location.x = 1
bpy.context.view_layer.update()
assert authoring.anchor_pending(handle)
assert bpy.ops.ov.physics_anchor(operation="APPLY") == {"FINISHED"}
assert not authoring.anchor_pending(handle)
assert body.ov.physics.persistent_id == old_identity
assert copy.ov.physics.persistent_id != old_identity
assert joint.endpoint_a.object == body
assert len({body.ov.physics.persistent_id, copy.ov.physics.persistent_id}) == 2

joint_copy = handle.copy()
bpy.context.scene.collection.objects.link(joint_copy)
old_joint_identity = joint.persistent_id
handle.location.x = 1.5
bpy.context.view_layer.update()
assert bpy.ops.ov.physics_anchor(operation="APPLY") == {"FINISHED"}
assert joint.persistent_id == old_joint_identity
assert joint_copy.ov.physics.joint.persistent_id != old_joint_identity
bpy.data.objects.remove(joint_copy, do_unlink=True)
committed_frame = tuple(joint.frame_a.translation)

# A later resolution error must not leave a half-completed identity repair.
second_copy = body.copy()
bpy.context.scene.collection.objects.link(second_copy)
identity_before_failure = body.ov.physics.persistent_id
joint.endpoint_b.kind = "OBJECT"
joint.endpoint_b.object = body
handle.location.x = 2
bpy.context.view_layer.update()
history_before_failure = {key: dict(value) for key, value in sessions._identity_owners.items()}
user_messages.default_bus().take_pending()
try:
    authoring.apply_anchor(handle, bpy.context.scene, bpy.context.evaluated_depsgraph_get(),
                           repair_duplicate_objects=True)
except ValueError as exc:
    assert "itself" in str(exc), exc
else:
    raise AssertionError("A self-connected joint was accepted")
assert body.ov.physics.persistent_id == identity_before_failure
assert second_copy.ov.physics.persistent_id == identity_before_failure
assert tuple(joint.frame_a.translation) == committed_frame
assert sessions._identity_owners == history_before_failure
assert not user_messages.default_bus().take_pending()

print("ANCHOR_DUPLICATE_REPAIR_OK")

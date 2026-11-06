# SPDX-License-Identifier: Apache-2.0
"""Restoring simulated transforms must not become a structural physics edit."""
from pathlib import Path
import sys
from types import SimpleNamespace

import bpy
from mathutils import Quaternion

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from ovrtx_blender_example import authoring_properties, physics_pose_presentation
from ovrtx_blender_example import scene_generation_sessions as sessions
from ovrtx_blender_example.interactive_operator_state import PhysicsPlaybackLock, suppress_interactive_edit_bridge
from ovrtx_blender_example.physics_frames import RigidFrame

authoring_properties.register()
scene = bpy.context.scene
bpy.ops.mesh.primitive_cube_add(location=(0, 0, 3))
body = bpy.context.object
body.scale = (0.49999988079071045, 0.5, 0.49999988079071045)
bpy.ops.ov.physics_add_body()
bpy.context.view_layer.update()
initial_channels = {name: tuple(getattr(body, name)) for name in
                    ("location", "rotation_euler", "rotation_quaternion", "rotation_axis_angle", "scale")}
lock = PhysicsPlaybackLock()
uid = int(scene.session_uid)
sessions._runtimes[uid] = SimpleNamespace(playback_lock=lock)
target = SimpleNamespace(body_path="/Body", enabled=True, role="DYNAMIC",
                         object_session_uid=int(body.session_uid), object_identity=body.ov.physics.persistent_id,
                         bone_identity="", body_offset=RigidFrame(), meters_per_blender_unit=1,
                         owner_uniform_scale=0.5)

def observe(scene, *_unused):
    sessions.observe_physics_edits(scene)

bpy.app.handlers.depsgraph_update_post.append(observe)
try:
    sessions.observe_physics_edits(scene)
    for direct_clear in (False, True, False, True):
        q = Quaternion((0.86, 0.3, 0.4, 0.1)).normalized()
        with suppress_interactive_edit_bridge():
            physics_pose_presentation.apply(scene, (target,), {
                "/Body": {"translate": (1, 2, 1), "orient": (q.x, q.y, q.z, q.w)}},
                lock, 0, bpy.context.view_layer)
        assert uid not in sessions._physics_restart_pending
        if direct_clear:
            lock.clear(reason="initial_condition_frame", frame1_cleared=True)
            bpy.context.view_layer.update()
        else:
            sessions.restore_physics_presentation(scene)
        assert all(tuple(getattr(body, name)) == value for name, value in initial_channels.items())
        assert uid not in sessions._physics_restart_pending, "Restore queued a false structural restart"
        changes = sessions.observe_physics_edits(scene)
        assert not changes.restart and not changes.body_input, changes

    # Preserve exact detection of real authoring edits; no global scale epsilon.
    body.scale.x += 0.000001
    bpy.context.view_layer.update()
    sessions.observe_physics_edits(scene)
    assert sessions._physics_restart_pending[uid].structural, "Small authored scale edit was lost"
    print("PHYSICS_RESTORE_SNAPSHOT_OK: stop and direct frame-one restore cycles; real scale edit retained")
finally:
    bpy.app.handlers.depsgraph_update_post.remove(observe)
    sessions._runtimes.pop(uid, None)

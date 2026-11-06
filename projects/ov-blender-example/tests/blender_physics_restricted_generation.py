# SPDX-License-Identifier: Apache-2.0
"""Scene generation must not write IDs inside RenderEngine.view_draw.

Blender disallows ID writes in RNA callbacks without FUNC_ALLOW_WRITE, such
as view_draw. That callback only runs for a drawn window, so this fixture needs
a GUI session, not --background.
"""
import os
from pathlib import Path
import sys
import traceback
import uuid

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from ovrtx_blender_example import authoring_properties
from ovrtx_blender_example import scene_generation_sessions as sessions
from ovrtx_blender_example.interactive_operator_state import suppress_interactive_edit_bridge
from ovrtx_blender_example.physics_authoring import apply_anchor
from ovrtx_blender_example.physics_endpoints import initialize_joint, initialize_owner
from ovrtx_blender_example.scene_generation import SceneGenerationError

authoring_properties.register()
scene = bpy.context.scene
uid = int(scene.session_uid)


def body(name, x):
    bpy.ops.mesh.primitive_cube_add(location=(x, 0, 2))
    obj = bpy.context.object
    obj.name = name
    initialize_owner(obj)
    obj.ov.physics.rigid_body.schema_opinion = "APPLY"
    collider = obj.ov.physics.colliders.add()
    collider.name = collider.persistent_id = uuid.uuid4().hex
    collider.schema_opinion = "APPLY"
    collider.shape = "ANALYTIC_SPHERE"
    return obj


a, b = body("A", 0), body("B", 3)
handle = bpy.data.objects.new("Joint", None)
scene.collection.objects.link(handle)
joint = initialize_joint(handle)
joint.enabled = True
joint.endpoint_a.object, joint.endpoint_b.object = a, b
bpy.context.view_layer.update()
apply_anchor(handle, scene, bpy.context.evaluated_depsgraph_get())

draws = []  # (mode, outcome) for every restricted view_draw
mode = ["viewport"]
builds = [0]
original_generation_for_scene = sessions.generation_for_scene


def counted_generation_for_scene(*args, **kwargs):
    builds[0] += 1
    return original_generation_for_scene(*args, **kwargs)


class RestrictedGenerationEngine(bpy.types.RenderEngine):
    bl_idname = "OV_TEST_RESTRICTED_GENERATION"
    bl_label = "Restricted Generation Test"

    def view_update(self, context, depsgraph):
        pass

    def view_draw(self, context, depsgraph):
        try:
            if mode[0] == "direct":
                outcome = original_generation_for_scene(depsgraph.scene)
            else:
                outcome = sessions.generation_for_viewport(depsgraph.scene)
        except Exception as exc:
            outcome = exc
        draws.append((mode[0], outcome))


def redraw():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.spaces.active.shading.type = "RENDERED"
                area.tag_redraw()


def wait_for(predicate, label, *, frames=60):
    for _ in range(frames):
        redraw()
        yield
        if predicate():
            return
    raise AssertionError(f"Timed out waiting for {label}; draws={draws[-3:]!r}")


def last():
    return draws[-1][1] if draws else None


def reset_owner():
    owner = sessions._owners.pop(uid, None)
    if owner is not None:
        owner.close()


def steps():
    # Invalid authoring: the deferred build fails once, the viewport reports
    # the real error, and redraws do not retry until a scene edit.
    a.ov.physics.rigid_body.enabled = False
    sessions.observe_physics_edits(scene)
    yield from wait_for(lambda: "no enabled authored OV body" in str(last()), "endpoint error")
    assert all(not isinstance(o, AttributeError) for _, o in draws), draws
    failed_builds = builds[0]
    for _ in range(5):
        redraw()
        yield
    assert builds[0] == failed_builds, "Failed build was retried without an edit"
    assert "no enabled authored OV body" in str(last())

    # Fixing the authoring clears the error and the deferred build succeeds.
    a.ov.physics.rigid_body.enabled = True
    sessions.observe_physics_edits(scene)
    yield from wait_for(lambda: not isinstance(last(), Exception) and last() is not None,
                        "accepted generation")
    assert sessions._owners[uid].current_generation is not None

    # Direct construction inside view_draw with a presented pose reports a
    # scene error instead of a raw RNA AttributeError, and changes nothing.
    reset_owner()
    with suppress_interactive_edit_bridge():
        a.location = (4, 5, 6)
    bpy.context.view_layer.update()
    mode[0] = "direct"
    draws.clear()
    yield from wait_for(lambda: bool(draws), "direct draw")
    assert isinstance(last(), SceneGenerationError), repr(last())
    assert tuple(a.location) == (4, 5, 6), "Presented pose was changed"

    # The writable path still exports the authored basis and restores the pose.
    mode[0] = "viewport"
    assert original_generation_for_scene(scene) is not None
    assert tuple(a.location) == (4, 5, 6), "Presented pose was not restored"
    print("PHYSICS_RESTRICTED_GENERATION_OK: deferred build, reported error without retry loop, "
          "recovery after edit, restricted pose restore reported")


sessions.generation_for_scene = counted_generation_for_scene
bpy.utils.register_class(RestrictedGenerationEngine)
scene.render.engine = RestrictedGenerationEngine.bl_idname
sequence = steps()


def tick():
    code = 0
    try:
        next(sequence)
        return 0.1
    except StopIteration:
        pass
    except Exception:
        traceback.print_exc()
        code = 17
    sessions.generation_for_scene = original_generation_for_scene
    reset_owner()
    sys.stdout.flush()
    os._exit(code)


bpy.app.timers.register(tick, first_interval=1.0)

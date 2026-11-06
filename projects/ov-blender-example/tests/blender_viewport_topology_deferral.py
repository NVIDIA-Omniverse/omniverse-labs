# SPDX-License-Identifier: Apache-2.0
"""Topology edits never reconcile inside RenderEngine.view_draw.

view_draw forbids ID writes (a sparse export sets the active Object), so the
viewport provider defers reconciliation to a main-thread timer. Needs a GUI
session: view_draw only runs for a drawn window.
"""
import os
from pathlib import Path
import sys
import traceback

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from ovrtx_blender_example import scene_generation_sessions as sessions
from ovrtx_blender_example.scene_generation import blender_id

scene = bpy.context.scene
uid = int(scene.session_uid)
draws = []


class DeferralEngine(bpy.types.RenderEngine):
    bl_idname = "OV_TEST_TOPOLOGY_DEFERRAL"
    bl_label = "Topology Deferral Test"

    def view_update(self, context, depsgraph):
        pass

    def view_draw(self, context, depsgraph):
        try:
            draws.append(sessions.generation_for_viewport(depsgraph.scene))
        except Exception as exc:
            draws.append(exc)


def redraw():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.spaces.active.shading.type = "RENDERED"
                area.tag_redraw()


def wait_for(predicate, label, frames=60):
    for _ in range(frames):
        redraw()
        yield
        if predicate():
            return
    raise AssertionError(f"Timed out waiting for {label}; last draws={draws[-3:]!r}")


def mapped(obj):
    owner = sessions._owners[uid]
    return blender_id(obj, "OBJECT") in owner.reuse().blender_prim_paths


def accept_pending():
    owner = sessions._owners[uid]
    if owner.pending_generation is not None:
        owner.accept(owner.pending_generation)


def edit(change, label, expect):
    draws.clear()
    affected = change()
    sessions.mark_scene_dirty(scene, affected)
    yield from wait_for(lambda: bool(draws) and expect(), label)
    errors = [d for d in draws if isinstance(d, Exception)]
    assert not errors, f"{label}: view_draw raised {errors[0]!r}"
    accept_pending()


def add():
    bpy.ops.mesh.primitive_cube_add(location=(3, 0, 0))
    bpy.context.object.name = "Added"
    return {blender_id(bpy.context.object, "OBJECT")}


def duplicate():
    source = scene.objects["Added"]
    copy = source.copy()
    copy.data = source.data.copy()
    copy.name = "Duplicated"
    scene.collection.objects.link(copy)
    bpy.context.view_layer.objects.active = copy  # the edit's own active change
    return {blender_id(copy, "OBJECT")}


def delete():
    target = scene.objects["Duplicated"]
    identity = blender_id(target, "OBJECT")
    bpy.data.objects.remove(target, do_unlink=True)
    return {identity}


def steps():
    sessions.generation_for_scene(scene)  # writable: initial generation
    yield from edit(add, "added object", lambda: mapped(scene.objects["Added"]))
    yield from edit(duplicate, "duplicated object", lambda: mapped(scene.objects["Duplicated"]))
    removed = blender_id(scene.objects["Duplicated"], "OBJECT")
    yield from edit(delete, "deleted object",
                    lambda: removed not in sessions._owners[uid].reuse().blender_prim_paths)
    yield from failed_reconcile_steps()
    print("VIEWPORT_TOPOLOGY_DEFERRAL_OK: add, duplicate and delete reconciled off view_draw; "
          "failed reconciles retry once per edit, never per redraw")


def idle(frames):
    for _ in range(frames):
        redraw()
        yield


def settle_candidate():
    # As activate_for_viewport does when the render runtime accepts a candidate.
    accept_pending()
    sessions._blocked_reconciliations.pop(uid, None)
    sessions._pending_affected.pop(uid, None)


def failed_reconcile_steps():
    """With the real engine depsgraph handler, the export's own selection
    writes must not count as edits and loop a failing reconcile."""
    from ovrtx_blender_example import engine, scene_generation
    real_export = scene_generation._stock_export
    failing = {"on": True}
    attempts = []

    def export(scene_, path, selected_objects_only=False, **kwargs):
        if selected_objects_only and failing["on"]:
            attempts.append(path)
            raise RuntimeError("injected sparse export failure")
        return real_export(scene_, path, selected_objects_only=selected_objects_only, **kwargs)

    scene_generation._stock_export = export
    bpy.app.handlers.depsgraph_update_post.append(engine._live_interactive_edit_depsgraph_handler)
    try:
        settle_candidate()
        bpy.ops.mesh.primitive_cube_add(location=(-3, 0, 0))
        bpy.context.object.name = "Failing"
        yield from idle(40)
        assert len(attempts) == 1, f"failed reconcile looped: {len(attempts)} attempts without edits"
        scene.objects["Failing"].location.x = -4
        yield from idle(30)
        assert len(attempts) == 2, f"re-edit retried {len(attempts) - 1} times"
        failing["on"] = False
        scene.objects["Failing"].location.x = -5
        yield from wait_for(lambda: mapped(scene.objects["Failing"]), "repaired reconcile")
        settle_candidate()
        failing["on"] = True
        bpy.ops.mesh.primitive_cube_add(location=(0, 3, 0))
        bpy.context.object.name = "Discarded"
        yield from idle(20)
        assert uid in sessions._blocked_reconciliations, "failing addition did not block"
        bpy.data.objects.remove(scene.objects["Discarded"], do_unlink=True)
        yield from idle(30)
        assert uid not in sessions._dirty and uid not in sessions._blocked_reconciliations, (
            "deleted failed addition left a stale reconcile")
    finally:
        bpy.app.handlers.depsgraph_update_post.remove(engine._live_interactive_edit_depsgraph_handler)
        scene_generation._stock_export = real_export


bpy.utils.register_class(DeferralEngine)
scene.render.engine = DeferralEngine.bl_idname
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
    owner = sessions._owners.pop(uid, None)
    if owner is not None:
        owner.close()
    sys.stdout.flush()
    os._exit(code)


bpy.app.timers.register(tick, first_interval=1.0)

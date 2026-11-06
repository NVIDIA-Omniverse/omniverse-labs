# SPDX-License-Identifier: Apache-2.0
"""Run without --background for actual GPU/lifecycle acceptance; artifacts via --."""
import importlib
import json
from pathlib import Path
import sys
import traceback
import uuid
import bpy
from mathutils import Quaternion

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from ovrtx_blender_example import authoring_properties as props
from ovrtx_blender_example import physics_overlay as overlay
from ovrtx_blender_example.physics_endpoints import initialize_owner, initialize_joint
from ovrtx_blender_example.physics_authoring import apply_anchor

output = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
output.mkdir(parents=True, exist_ok=True)
if not bpy.app.background:
    sys.stdout = sys.stderr = (output / "console.log").open("w", buffering=1)
checks = []
def check(name, condition):
    assert condition, name
    checks.append(name)

props.register()
bpy.context.preferences.view.show_splash = False
if bpy.app.background:
    check("headless owns no GPU service", overlay._service is None and not overlay._installed)
    props.unregister()
    print(json.dumps({"ok": True, "checks": checks}))
else:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_cube_add(location=(-1, 0, 0))
    body = bpy.context.object
    initialize_owner(body)
    body.ov.physics.rigid_body.schema_opinion = "APPLY"
    collider = body.ov.physics.colliders.add()
    collider.name = collider.persistent_id = uuid.uuid4().hex
    collider.schema_opinion = "APPLY"
    collider.shape = "ANALYTIC_SPHERE"
    handle = bpy.data.objects.new("Hinge Indicator", None)
    bpy.context.scene.collection.objects.link(handle)
    handle.location = (1, 0, 0)
    joint = initialize_joint(handle)
    joint.enabled = True
    joint.type = "REVOLUTE"
    joint.endpoint_a.object = body
    joint.endpoint_b.kind = "WORLD"
    bpy.context.view_layer.update()
    apply_anchor(handle, bpy.context.scene, bpy.context.evaluated_depsgraph_get())
    area = next(a for a in bpy.context.screen.areas if a.type == "VIEW_3D")
    region = next(r for r in area.regions if r.type == "WINDOW")
    area.spaces.active.region_3d.view_distance = 9
    area.spaces.active.region_3d.view_rotation = Quaternion((1, 0, 0, 0))
    bpy.context.scene.render.engine = "BLENDER_EEVEE"
    main = bpy.context.window
    state = {"step": 0}
    draw_counts = {"lines": 0, "labels": 0}
    monitor = []

    def hide_windows():
        if sys.platform == "win32":
            import ctypes
            import os
            callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
            def hide(hwnd, unused):
                pid = ctypes.c_ulong()
                ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                if pid.value == os.getpid():
                    ctypes.windll.user32.ShowWindow(hwnd, 0)
                return True
            ctypes.windll.user32.EnumWindows(callback_type(hide), 0)

    def state_probe():
        # Calls the actual callback in a GPU draw context, verifies state restoration.
        import gpu
        service = overlay._service
        if service is None or service.current() is None:
            return
        saved = (gpu.state.blend_get(), gpu.state.depth_test_get(), gpu.state.depth_mask_get())
        service.draw_lines()
        assert saved == (gpu.state.blend_get(), gpu.state.depth_test_get(), gpu.state.depth_mask_get())
        draw_counts["lines"] += 1

    def label_probe():
        service = overlay._service
        if service is not None and service.current() is not None:
            service.draw_labels()
            draw_counts["labels"] += 1

    monitor.append(bpy.types.SpaceView3D.draw_handler_add(state_probe, (), "WINDOW", "POST_VIEW"))
    monitor.append(bpy.types.SpaceView3D.draw_handler_add(label_probe, (), "WINDOW", "POST_PIXEL"))

    def step():
        try:
            hide_windows()
            service = overlay._service
            index = state["step"]
            if index == 0:
                service.refresh()
                check("prepared real GPU batches", bool(service.prepared) and any(p.strokes for p in service.prepared.values()))
                check("registered once", len(service.handles) == 2)
                overlay.register()
                check("idempotent register", len(service.handles) == 2)
            elif index == 1:
                check("actual solid GPU callbacks", draw_counts["lines"] > 0 and draw_counts["labels"] > 0)
                bpy.ops.screen.screenshot(filepath=str(output / "solid.png"))
                area.spaces.active.shading.type = "RENDERED"
                state["solid_count"] = draw_counts["lines"]
            elif index == 2:
                check("actual rendered GPU callbacks", draw_counts["lines"] > state["solid_count"])
                bpy.ops.screen.screenshot(filepath=str(output / "rendered.png"))
                area.spaces.active.overlay.show_overlays = False
                with bpy.context.temp_override(window=main, area=area, region=region):
                    check("overlay toggle immediately suppresses stale draw", service.current() is None)
                service.refresh()
                check("overlay off clears batches", not service.prepared)
                area.spaces.active.overlay.show_overlays = True
                area.spaces.active.shading.type = "SOLID"
                bpy.context.scene.ov.indicators_selected_only = True
                bpy.ops.object.select_all(action="DESELECT")
                service.refresh()
                check("selected only omits unselected geometry", all(not p.strokes for p in service.prepared.values()))
                bpy.context.scene.ov.indicators_selected_only = False
                body = bpy.data.objects["Cube"]
                body.hide_set(True)
                service.refresh()
                check("hidden collider omitted", all(not any(s.role == "collider-wire" for s in p.snapshot.strokes) for p in service.prepared.values()))
                body.hide_set(False)
                body.select_set(True)
                bpy.context.view_layer.objects.active = body
                with bpy.context.temp_override(window=main, area=area, region=region):
                    bpy.ops.view3d.localview(frame_selected=False)
                    service.refresh()
                    check("local view excludes joint handle", all(not p.snapshot.labels for p in service.prepared.values()))
                    bpy.ops.view3d.localview(frame_selected=False)
                bpy.ops.wm.window_new()
                hide_windows()
            elif index == 3:
                service.refresh()
                check("multiple windows get independent caches", len({key[0] for key in service.prepared}) >= 2)
                second = next(w for w in bpy.context.window_manager.windows if w != main)
                with bpy.context.temp_override(window=second):
                    bpy.ops.wm.window_close()
                service.refresh()
                check("closed window releases batches", len({key[0] for key in service.prepared}) == 1)
                bpy.ops.ed.undo_push(message="Indicator before move")
                bpy.data.objects["Hinge Indicator"].location.x += 1
                bpy.ops.ed.undo_push(message="Indicator after move")
                bpy.ops.ed.undo()
                check("undo immediately invalidates cache", not service.prepared)
            elif index == 4:
                service.refresh()
                check("undo rebuilds GPU cache", bool(service.prepared))
                bpy.ops.wm.save_as_mainfile(filepath=str(output / "overlay.blend"))
                bpy.ops.wm.open_mainfile(filepath=str(output / "overlay.blend"))
                check("load immediately invalidates cache", not service.prepared)
            elif index == 5:
                service.refresh()
                check("file load restores cache", bool(service.prepared))
                old = service
                importlib.reload(overlay)
                check("module reload removes old handlers", not old.handles and not old.prepared and old.shader is None and not bpy.app.timers.is_registered(old.timer))
                overlay.register()
                overlay._service.refresh()
                check("reload waits for a drawing context", overlay._service.shader is None and not overlay._service.prepared)
            elif index == 6:
                check("module reload recovers", bool(service.prepared) and service.shader is not None)
                old = service
                props.unregister()
                check("disable releases resources", not old.handles and not old.prepared and old.shader is None and old.font is None)
                check("disable removes timer and lifecycle callbacks", not bpy.app.timers.is_registered(old.timer) and not overlay._installed)
                props.register()
                overlay._service.refresh()
                check("reenable waits for a drawing context", overlay._service.shader is None and not overlay._service.prepared)
            elif index == 7:
                check("reenable restores drawing", bool(service.prepared) and service.shader is not None)
                props.unregister()
                for token in monitor:
                    bpy.types.SpaceView3D.draw_handler_remove(token, "WINDOW")
                (output / "result.json").write_text(json.dumps({"ok": True, "blender": bpy.app.version_string, "checks": checks, "draw_counts": draw_counts}, indent=2))
                bpy.ops.wm.quit_blender()
                return None
            state["step"] += 1
            for window in bpy.context.window_manager.windows:
                for current_area in window.screen.areas:
                    current_area.tag_redraw()
            return 3.0
        except Exception:
            (output / "result.json").write_text(json.dumps({"ok": False, "step": state["step"], "checks": checks, "error": traceback.format_exc()}, indent=2))
            bpy.ops.wm.quit_blender()
            return None

    bpy.app.timers.register(step, first_interval=3, persistent=True)

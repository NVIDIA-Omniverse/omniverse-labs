"""Actual Blender UI/modal operator with the fixed native CPU worker."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
import bpy
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from ovrtx_blender_example import authoring_properties as props, physics_bake_ui as ui
from ovrtx_blender_example.ovphysx_stage import OvphysxStageController
from ovrtx_blender_example import scene_generation_sessions as sessions
p = argparse.ArgumentParser()
p.add_argument("--blend", type=Path, required=True)
p.add_argument("--runtime", type=Path, default=Path("."))
p.add_argument("--output", type=Path, required=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from public_sdk_physics_fixture import add_arguments, configure
add_arguments(p)
a = p.parse_args(sys.argv[sys.argv.index("--") + 1:])
a.device = "cpu"
configure(a)
a.output.mkdir(parents=True, exist_ok=True)
sys.stdout = sys.stderr = (a.output / "console.log").open("w", buffering=1)
props.register()
ui.register()
bpy.ops.wm.open_mainfile(filepath=str(a.blend.resolve()))
scene = bpy.context.scene
settings = scene.ov_physics_bake
settings.initial_frame, settings.start_frame, settings.end_frame = 1, 2, 5
settings.substeps, settings.interval = 2, 1
scene.render.fps, scene.render.fps_base = 24, 1.001
runtime = a.runtime.resolve()
server = Path(__file__).resolve() if a.public_sdk else runtime / "bin/ovphysx_grpc_server.exe"
os.environ["OVPHYSX_ROOT"] = str(runtime / "private/ovphysx-runtime")
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    address = f"127.0.0.1:{sock.getsockname()[1]}"
def factory(config):
    return OvphysxStageController(replace(config, server=str(server), ovphysx_address=address,
        ovphysx_worker_command=subprocess.list2cmdline([str(server), "--listen", address, "--device", "cpu"]),
        device="cpu", worker_log_path=str(a.output / "worker.log"),
        ovphysx_native_client_path=str(runtime / "client"), ovphysx_native_client_module="ovphysx_bridge_client"))
original_session = ui.BakeCaptureSession
captured = []
def native_session(*args, **kwargs):
    session = original_session(*args, **kwargs, controller_factory=factory)
    captured.append(session)
    return session
ui.BakeCaptureSession = native_session
area = region = window = None
checks = []
state = {"phase": 0, "deadline": time.monotonic() + 180}
def invoke(name):
    with bpy.context.temp_override(window=window, area=area, region=region):
        return getattr(bpy.ops.ov, name)("INVOKE_DEFAULT")
def check(name, result):
    assert result, name
    checks.append(name)
def tick():
    global area, region, window, scene, settings
    try:
        assert time.monotonic() < state["deadline"], "modal bake timed out"
        phase = state["phase"]
        if phase == 0:
            window = bpy.context.window_manager.windows[0]
            area = next(area for area in window.screen.areas if area.type == "PROPERTIES")
            area.spaces.active.context = "SCENE"
            region = next(r for r in area.regions if r.type == "WINDOW")
            check("actual bake invoke", invoke("physics_bake_animation") == {"RUNNING_MODAL"})
            state["phase"] = 1
        elif phase == 1 and not ui._jobs:
            check("modal native bake creates grouped Actions", len(settings.records) == 4)
            check("capture later start/substeps/fractional FPS", captured[0].artifact.manifest()["timing"]["substeps"] == 2)
            with bpy.context.temp_override(window=window, area=area, region=region):
                bpy.ops.view2d.scroll_down(page=True)
                bpy.ops.view2d.scroll_down(page=True)
            state["phase"] = 1.5
        elif phase == 1.5:
            bpy.ops.screen.screenshot(filepath=str(a.output / "bake-panel.png"))
            settings.destination = "MAPPED"
            settings.mappings.add()
            state["phase"] = 1.55
            area.tag_redraw()
            return .3
        elif phase == 1.55:
            with bpy.context.temp_override(window=window, area=area, region=region):
                bpy.ops.view2d.scroll_down(page=True)
            state["phase"] = 1.6
            area.tag_redraw()
            return .3
        elif phase == 1.6:
            bpy.ops.screen.screenshot(filepath=str(a.output / "bake-mapping-panel.png"))
            settings.destination = "SOURCE"
            check("activation invoke", invoke("physics_bake_activate") == {"RUNNING_MODAL"})
            state["phase"] = 2
        elif phase == 2 and not ui._jobs:
            check("activated with physics gate", settings.active and sessions.presentation_stopped(scene))
            scene.frame_set(4)
            check("timeline preserves gate", sessions.presentation_stopped(scene))
            check("deactivation operator", bpy.ops.ov.physics_bake_deactivate() == {"FINISHED"})
            check("deactivated without native restart", not settings.active)
            state["actions"] = set(bpy.data.actions)
            check("repeat capture starts", invoke("physics_bake_animation") == {"RUNNING_MODAL"})
            window.scene = bpy.data.scenes.new("OtherScene")
            state["phase"] = 3
        elif phase == 3 and not ui._jobs:
            check("scene switch cancels originating capture", captured[-1].closed and sessions._bake_capture_owner is None)
            check("scene switch retains completed Actions", set(bpy.data.actions) == state["actions"])
            window.scene = scene
            state["phase"] = 3.5
        elif phase == 3.5:
            check("capture for build cancellation", invoke("physics_bake_animation") == {"RUNNING_MODAL"})
            state["phase"] = 4
        elif phase == 4:
            job = next(iter(ui._jobs), None)
            if job and job._builder:
                check("incremental builder is cancellable", not job._builder.done)
                bpy.ops.wm.save_as_mainfile(filepath=str(a.output / "cancel-build.blend"))
                check("save cancels pending builder", not ui._jobs and job._timer is None)
                check("cancel removes only private incomplete Actions", set(bpy.data.actions) == state["actions"])
                check("pending activation before undo", invoke("physics_bake_activate") == {"RUNNING_MODAL"})
                with bpy.context.temp_override(window=window, area=area, region=region):
                    bpy.ops.ed.undo()
                check("undo cancels modal activation", not ui._jobs)
                scene, settings = window.scene, window.scene.ov_physics_bake
                bpy.ops.wm.open_mainfile(filepath=str(a.output / "cancel-build.blend"))
                state["phase"] = 5
                return .3
        elif phase == 5:
            window = bpy.context.window_manager.windows[0]
            scene, settings = window.scene, window.scene.ov_physics_bake
            check("file load retains no modal resources", not ui._jobs)
            check("file load preserves completed records", len(settings.records) == 4)
            check("no stale RNA lifecycle errors", "Traceback" not in (a.output / "console.log").read_text())
            ui.unregister()
            check("disable removes modal resources", not ui._jobs)
            ui.register()
            check("reload restores durable records", len(scene.ov_physics_bake.records) == 4)
            ui.unregister()
            props.unregister()
            (a.output / "results.json").write_text(json.dumps(dict(passed=True, checks=checks), indent=2))
            bpy.ops.wm.quit_blender()
            return None
        return .01
    except Exception:
        (a.output / "results.json").write_text(json.dumps(dict(passed=False, checks=checks, error=traceback.format_exc()), indent=2))
        ui._cancel_jobs()
        bpy.ops.wm.quit_blender()
        return None
bpy.app.timers.register(tick, first_interval=1, persistent=True)

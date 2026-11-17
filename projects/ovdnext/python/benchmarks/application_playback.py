# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure production playback or paused Next-frame updates in Application.

Instrumentation is local to this process. Application.run_async still owns
loading, polling, transport, rendering and layout; the product's public asyncio
coordinator owns task joining and shutdown. glFinish makes GPU
completion explicit; its synchronization changes normal CPU/GPU overlap. These
observations are completed updates, not uninstrumented application FPS.
"""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
from time import perf_counter

import numpy as np
from PIL import Image
import omni.ui as ui
import _pvd_native
from ovd_ui import style
from ovd_ui.adapters import PvdDataAdapter
from ovd_ui.application import Application
from ovd_ui.lifecycle import RUNNER_ID
from ovd_ui.interaction import InteractionState
from ovd_ui.models import ObjectTreeModel
from ovd_ui.playback import FrameMode
from ovd_ui.renderer import ViewportRenderer
from ovd_ui.render_settings import FaceCulling, PlaneMode
from ovd_ui.widgets.object_browser import ObjectBrowser, _ObjectDelegate
from ovd_ui.widgets.properties import PropertyInspector

from rendering import make_job
from application_evidence import (GlxWitness, display_environment, effective_settings,
    matrices, runtime_libraries, source_geometry, submitted_geometry, surface)
from inspector_evidence import array_control, inspector_workload, publication_requests
from refresh_cost_evidence import GcObservation, gc_control_evidence, instrument_graph

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    paths = sorted(path for path in (ROOT / "python/ovd_ui").rglob("*")
                   if path.suffix in (".py", ".json", ".ttf", ".otf", ".svg", ".png"))
    paths += sorted((ROOT / "src/PVDSession").glob("*.*"))
    paths += [Path(__file__), ROOT / "python/benchmarks/rendering.py",
              ROOT / "python/benchmarks/application_evidence.py"]
    paths += [ROOT / "python/benchmarks/inspector_evidence.py"]
    paths += [ROOT / "python/benchmarks/refresh_cost_evidence.py"]
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


class Recorder:
    def __init__(self):
        self.current = None
        self.originals = []
        self.inspectors = {}
        self.array_controls = {}
        self.last_publication_requests = []
        self.draw_states = {}
        self.gc_observation = None
        self.graph_started = None

    def phase_interval(self, name, start, stop):
        if self.gc_observation is not None and self.current is not None:
            self.current.setdefault("_phase_intervals", []).append((name, start, stop))

    def graph_begin(self):
        if self.current is not None:
            if self.graph_started is not None:
                raise RuntimeError("Graph observation was unexpectedly reentered")
            self.graph_started = perf_counter()

    def graph_end(self):
        if self.current is not None:
            stopped = perf_counter()
            if self.graph_started is None:
                raise RuntimeError("Graph observation lacks its start boundary")
            self.phase_interval("graph_preparation_ms", self.graph_started, stopped)
            self.add("graph_preparation_ms", (stopped - self.graph_started) * 1000)
            self.graph_started = None

    def add(self, name, elapsed):
        if self.current is not None:
            self.current[name] = self.current.get(name, 0.) + elapsed
            self.current[name + "_calls"] = self.current.get(name + "_calls", 0) + 1

    def wrap(self, cls, name, phase, *, finish=False, tick=False):
        original = getattr(cls, name)
        self.originals.append((cls, name, original))

        def measured(instance, *args, **kwargs):
            if phase == "property_build_ms":
                self.array_controls[id(instance)] = []
            if tick and self.current is not None:
                action = args[0].__name__
                if action in ("tick", "step"):
                    key = "transport_ticks" if action == "tick" else "transport_steps"
                    self.current[key] = self.current.get(key, 0) + 1
            started = perf_counter()
            try:
                result = original(instance, *args, **kwargs)
            finally:
                submitted = perf_counter()
                self.add(phase, (submitted - started) * 1000)
                self.phase_interval(phase, started, submitted)
                if finish:
                    instance.context.finish()
                    self.add("render_complete_ms", (perf_counter() - started) * 1000)
            if phase == "property_build_ms":
                adapter = args[0]
                self.inspectors[id(instance)] = dict(
                    frame=getattr(adapter.session, "frame", None) if adapter is not None else None,
                    capture_key=getattr(adapter.session, "capture_key", None) if adapter is not None else None,
                    generation=getattr(adapter.session, "generation", None) if adapter is not None else None,
                    identities=list(args[1]))
            return result

        setattr(cls, name, measured)

    def restore(self):
        for cls, name, original in reversed(self.originals):
            setattr(cls, name, original)

    def observe_draw_state(self, witness):
        original = ViewportRenderer._draw
        self.originals.append((ViewportRenderer, "_draw", original))

        def draw(renderer, *args, **kwargs):
            self.draw_states[id(renderer)] = dict(context=witness.context(),
                                                   viewport=witness.viewport())
            return original(renderer, *args, **kwargs)

        ViewportRenderer._draw = draw

    def wrap_native_presentation(self):
        original = _pvd_native.Session.presentation
        self.originals.append((_pvd_native.Session, "presentation", original))

        def presentation(session, *args, **kwargs):
            requests = publication_requests(args, kwargs)
            self.last_publication_requests = [requests]
            if self.current is not None:
                self.current.setdefault("presentation_requests", []).append(requests)
            started = perf_counter()
            view = original(session, *args, **kwargs)
            stopped = perf_counter()
            self.add("native_presentation_ms", (stopped - started) * 1000)
            self.phase_interval("native_presentation_ms", started, stopped)
            self.add("native_extraction_ms", view.extraction_ms)
            if self.current is not None and (not view.ready or not view.request_applied):
                raise RuntimeError("Recorded-file playback unexpectedly deferred its presentation")
            return view

        _pvd_native.Session.presentation = presentation

    def observe_property_arrays(self):
        original = PropertyInspector._array
        self.originals.append((PropertyInspector, "_array", original))

        def build(inspector, adapter, identity, prop, count, read, **kwargs):
            result = original(inspector, adapter, identity, prop, count, read, **kwargs)
            observation = array_control(inspector, identity, prop, count,
                stride=kwargs.get("stride", 0), key=kwargs.get("key"))
            self.array_controls.setdefault(id(inspector), []).append(observation)
            return result

        PropertyInspector._array = build


class PlaybackMeasurement:
    def __init__(self, app, args, jobs, recorder):
        self.app, self.args, self.jobs, self.recorder = app, args, jobs, recorder
        self.index = 0
        self.phase = "loading"
        self.settle = 0
        self.iteration = -args.warmup
        self.results = []
        self.started = perf_counter()
        self.job_started = self.started
        self.row_started = 0.
        self.result = None
        self.frame_index = 0
        self.previous_capture_key = None
        self.glx = GlxWitness()
        self.recorder.observe_draw_state(self.glx)
        self.libraries_before = None

    def prepare(self):
        app, job = self.app, self.jobs[self.index]
        if any(name in app.renderer.device.casefold() for name in ("software", "llvmpipe")):
            raise RuntimeError("This benchmark requires hardware OpenGL")
        app._refresh(frame=job["frames"][0] - 1, scene_id=int(job["scene_id"]))
        app._transport(app.playback.set_mode, FrameMode.FULL)
        if self.args.settings_profile == "qt-reference":
            app._render_settings_changed(replace(app.render_settings,
                face_culling=FaceCulling.NONE, planes=PlaneMode.NONE,
                background_mode="reference", anti_aliasing=False))
        app.navigation.camera.look_at(job["eye"], job["target"], job["up"])
        app.navigation.camera.near = job["near"]
        app.navigation.camera.set_fov(60.)
        app.interaction.set_visible(job["excluded_actor_ids"], False)
        app._interaction_action(lambda: None)
        app.selection.clear()
        identity = 0  # The shared evidence contract uses zero for no selection.
        if job["selection"] == "actor":
            geometry = app.renderer.snapshot
            candidates = list(dict.fromkeys(int(identity) for identity, kind, scene in
                zip(geometry.object_ids, geometry.kinds, geometry.scene_ids)
                if kind != 4 and scene == int(job["scene_instance_id"])
                and identity not in job["excluded_actor_ids"]))
            identity = next((identity for identity in candidates
                if app.adapter.class_name(identity) == "PxRigidDynamic"), candidates[0])
            app.selection.publish([f"/objects/{identity}"], source="benchmark")
        app._refresh()
        self.result = {
            "job": job, "selected_id": identity,
            "selected_class": app.adapter.class_name(identity) if identity else None,
            "selected_name": app.adapter.name(identity) if identity else None,
            "open_and_setup_ms": (perf_counter() - self.job_started) * 1000,
            "settings": app.render_settings.to_dict(), "camera": app.navigation.state(),
            "overlay_mask": app.overlay_mask, "samples": [],
            "hidden_native_ids": sorted(app.interaction.hidden),
            "selected_property_count": len(app.adapter.properties(identity)) if identity else 0,
        }
        self.iteration = -self.args.warmup
        self.phase, self.settle = "settling", 4

    def current_surface(self):
        return surface(self.app, self.glx, self.args.condition,
                       self.recorder.draw_states[id(self.app.renderer)]["viewport"])

    def ready(self):
        self.result.update(effective_settings=effective_settings(self.app),
                           surface=self.current_surface(), **matrices(self.app),
                           inspector_workload=inspector_workload(self.app,
                               self.recorder.array_controls.get(id(self.app.property_inspector)),
                               self.recorder.last_publication_requests))
        self.result["palette_sha256"] = sha256(self.app.palette.path)
        self.expected_context = self.glx.context()
        if self.recorder.draw_states[id(self.app.renderer)]["context"] != self.expected_context:
            raise RuntimeError("Geometry and UI composition used different GLX contexts")
        actual = self.result["surface"]
        required = (actual["window_pixels"] if self.args.condition == "same-window"
                    else actual["geometry_fbo_pixels"])
        if required != [self.args.width, self.args.height]:
            raise RuntimeError("Requested actual window/geometry attachment condition was not met")
        if self.libraries_before is None:
            self.libraries_before = runtime_libraries()
        self.start_pass()

    def start_pass(self):
        app, job = self.app, self.jobs[self.index]
        # Rewind and all validation happen outside recorded frame intervals.
        app.playback.pause()
        app.seek(job["frames"][0] - 1, snap=False)
        self.frame_index = 0
        self.phase, self.settle = "rewind", 3

    def begin_row(self):
        if self.glx.context() != self.expected_context:
            raise RuntimeError("GLX context changed before the sample")
        self.app.renderer.context.finish()
        self.recorder.current = {"repetition": self.iteration}
        self.row_started = perf_counter()
        if self.args.workload == "paused-next":
            # Same handler as the Next button, then the real application loop
            # owns polling/rendering and the following UI frame. Dispatch is
            # included in this interval; sample validation is outside it.
            self.app._transport(self.app.playback.step, 1)

    def finish_row(self, wait_started):
        app, rec = self.app, self.recorder
        app.renderer.context.finish()
        finished = perf_counter()
        row = rec.current
        row["application_loop_ms"] = (wait_started - self.row_started) * 1000
        row["ui_frame_wait_ms"] = (finished - wait_started) * 1000
        row["completed_iteration_ms"] = (finished - self.row_started) * 1000
        rec.current = None
        if rec.gc_observation is not None:
            if row.get("graph_preparation_ms_calls") != 1 or rec.graph_started is not None:
                raise RuntimeError("Expected exactly one complete graph-preparation span")
            intervals = row.pop("_phase_intervals", [])
            intervals += [("application_loop_ms", self.row_started, wait_started),
                          ("ui_frame_wait_ms", wait_started, finished)]
            row["gc_observation"] = rec.gc_observation.window(self.row_started, finished, intervals)
            row["phase_intervals"] = [dict(name=name, start_ms=(start-self.row_started)*1000,
                stop_ms=(stop-self.row_started)*1000) for name, start, stop in intervals]
        if (self.glx.context() != self.expected_context or
                rec.draw_states[id(app.renderer)]["context"] != self.expected_context):
            raise RuntimeError("Geometry/UI GLX completion context changed during the sample")
        expected = self.jobs[self.index]["frames"][self.frame_index]
        ticks = int(self.args.workload == "normal-playback")
        if (app.frame != expected or row.get("transport_ticks", 0) != ticks
                or row.get("transport_steps", 0) != 1 - ticks):
            raise RuntimeError(f"Expected one {self.args.workload} update to {expected}, got {app.frame}: {row}")
        expected_playing = bool(ticks and expected < app.playback.clip.last)
        if bool(app.playback.playing) != expected_playing:
            raise RuntimeError("Playback state does not match the requested workload")
        if row.get("render_submit_ms_calls") != 1:
            raise RuntimeError("Each measured update must actually render exactly once")
        if row.get("native_presentation_ms_calls") != 1 or app.presentation.frame != expected:
            raise RuntimeError("Each measured update must publish exactly one matching native frame")
        if app.presentation_dirty:
            raise RuntimeError("Selected properties requested another publication after this frame")
        if self.result["selected_id"] and list(app.selection.get_snapshot().paths()) != [f'/objects/{self.result["selected_id"]}']:
            raise RuntimeError("The selected actor did not remain available throughout playback")
        if (app.scene_id != int(self.result["job"]["scene_id"]) or
                app.renderer.scene_id != int(self.result["job"]["scene_instance_id"])):
            raise RuntimeError("Scene changed during the sample")
        if (effective_settings(app) != self.result["effective_settings"] or
                app.render_settings.to_dict() != self.result["settings"] or
                sorted(app.interaction.hidden) != self.result["hidden_native_ids"]):
            raise RuntimeError("Rendering settings or visibility changed during the sample")
        camera = matrices(app)
        if (app.navigation.state() != self.result["camera"] or
                any(camera[key] != self.result[key] for key in camera)):
            raise RuntimeError("Camera changed during the sample")
        current_surface = self.current_surface()
        if current_surface != self.result["surface"]:
            raise RuntimeError("Actual window/geometry framebuffer changed during the sample")
        inspector = rec.inspectors.get(id(app.property_inspector), {})
        if self.result["selected_id"] and inspector != dict(frame=expected,
                capture_key=app.presentation.capture_key, generation=app.presentation.generation,
                identities=[self.result["selected_id"]]):
            raise RuntimeError("Selected inspector did not build from this owned native frame")
        if app.renderer.snapshot is not app.presentation.geometry_snapshot():
            raise RuntimeError("Renderer snapshot does not belong to the current owned presentation")
        row.update(frame=app.frame, presentation_frame=app.presentation.frame,
            scene_id=app.scene_id, segment_epoch=app.presentation.segment_epoch,
            instances=len(app.adapter.instances), tree_specs=len(app.object_model._specs),
            draw_objects=app.renderer.instance_count, draw_calls=app.renderer.draw_count,
            viewport=list(app.renderer.size), selected=list(app.selection.get_snapshot().paths()),
            expanded_items=sum(app.tree.is_expanded(item) for item in app.object_model.cache.values()),
            playback_active=bool(app.playback.playing),
            selected_native_ids=list(app.interaction.selected),
            hidden_native_ids=sorted(app.interaction.hidden),
            inspector_frame=inspector.get("frame"),
            inspector_publication=inspector,
            inspector_workload=inspector_workload(app,
                rec.array_controls.get(id(app.property_inspector)),
                row.get("presentation_requests", [])),
            native_alive_count=len(app.interaction.alive),
            source_geometry=source_geometry(app.renderer.snapshot, app.renderer.scene_id),
            **submitted_geometry(app.renderer), **camera)
        row["instrumentation"] = dict(untimed_validation_ms=(perf_counter() - finished) * 1000)
        if self.iteration >= 0:
            self.result["samples"].append(row)
        self.frame_index += 1
        if self.frame_index == len(self.jobs[self.index]["frames"]):
            app.playback.pause()
            self.iteration += 1
            if self.iteration < self.args.repeat:
                self.start_pass()
            else:
                self.finish_job()
        else:
            self.begin_row()

    def finish_job(self):
        self.app._sync_transport()
        samples = self.result["samples"]
        keys = sorted({key for row in samples for key in row if key.endswith("_ms")})
        self.result["timings_ms"] = {key: {"median": statistics.median(row.get(key, 0.) for row in samples),
            "p95": float(np.percentile([row.get(key, 0.) for row in samples], 95))} for key in keys}
        self.result["viewport"] = samples[0]["viewport"]
        if any(row["viewport"] != self.result["viewport"] for row in samples):
            raise RuntimeError("Viewport changed during this workload")
        self.result["device"] = self.app.renderer.device
        self.result["gl_version"] = self.app.renderer.context.info["GL_VERSION"]
        self.result["sample_info"] = self.app.renderer.sample_info()
        info = self.app.renderer.context.info
        self.result["gpu"] = dict(renderer=info["GL_RENDERER"], vendor=info["GL_VENDOR"],
            driver_version=info["GL_VERSION"].split("NVIDIA ")[-1], api="glx",
            context_api_version=info["GL_VERSION"])
        if sha256(self.app.palette.path) != self.result["palette_sha256"]:
            raise RuntimeError("Palette changed during measurement")
        self.result["camera_after"] = self.app.navigation.state()
        if self.result["camera"] != self.result["camera_after"]:
            raise RuntimeError("Camera changed during playback")
        self.results.append(self.result)
        # Full viewport pixels are read only after the timing passes, including
        # portions covered by UI islands in the composed window screenshot.
        viewport_rgba = self.app.renderer.framebuffer.read(components=4)
        self.result["viewport_rgba_sha256"] = hashlib.sha256(viewport_rgba).hexdigest()
        evidence = self.args.output.parent / (self.args.output.stem + "-evidence")
        shot = evidence / f'{Path(self.jobs[self.index]["capture"]).stem}-{self.jobs[self.index]["selection"]}-viewport.png'
        # Production shaders flip Y for the top-down OVUI texture. Native raw
        # framebuffer row order therefore already matches the displayed view.
        Image.frombytes("RGBA", self.app.renderer.size, viewport_rgba).save(shot)
        self.result["viewport_screenshot"] = str(shot)
        # Pausing schedules deferred transport widgets. Let those UI frames
        # complete before registering the screenshot, outside all timing rows.
        self.phase, self.settle = "evidence-settling", 4

    def capture_evidence(self):
        evidence = self.args.output.parent / (self.args.output.stem + "-evidence")
        shot = evidence / f'{Path(self.jobs[self.index]["capture"]).stem}-{self.jobs[self.index]["selection"]}.png'
        # The Inspector's native pre-swap screenshot facility is used only after
        # every measured repetition has finished, never in a recorded interval.
        if not ui._ui._schedule_screenshot(str(shot.resolve())):
            raise RuntimeError("Could not schedule benchmark evidence screenshot")
        self.result["screenshot"] = str(shot)
        self.phase, self.settle = "screenshot", 2

    def advance(self, wait_started):
        if perf_counter() - self.started > self.args.timeout:
            raise TimeoutError("Whole-application benchmark exceeded its bounded run time")
        if self.recorder.current is not None:
            self.finish_row(wait_started)
            return
        if self.phase == "loading":
            job = self.jobs[self.index]
            if (self.app.pending is None and self.app.session is not None
                    and self.app.session is self.app.workflow.display_session
                    and self.app.session.capture_key != self.previous_capture_key
                    and self.app.capture == Path(job["capture"])):
                self.prepare()
        elif self.phase in ("settling", "rewind", "evidence-settling", "screenshot"):
            self.settle -= 1
            if self.settle:
                return
            if self.phase == "settling":
                self.ready()
            elif self.phase == "evidence-settling":
                self.capture_evidence()
            elif self.phase == "rewind":
                self.app.playback.fps = 1_000_000.
                self.app.playback.direction = int(self.args.workload == "normal-playback")
                self.app.last_play = 0.
                self.phase = "playing"
                self.begin_row()
            else:
                self.index += 1
                if self.index == len(self.jobs):
                    self.app.running = False
                    self.phase = "done"
                else:
                    self.job_started = perf_counter()
                    self.previous_capture_key = self.app.session.capture_key
                    self.app.load(Path(self.jobs[self.index]["capture"]))
                    self.phase = "loading"


def run(args):
    before = source_hashes()
    native_hash = sha256(_pvd_native.__file__)
    recorder = Recorder()
    application_class = Application
    application_source = args.application_source or ROOT / "python/ovd_ui/application.py"
    application_hash = sha256(application_source)
    original_application_source, original_application_hash = application_source, application_hash
    source_transform = None
    if args.observe_refresh_costs:
        generated, source_transform = instrument_graph(application_source.read_bytes())
        application_source = args.output.with_suffix(".application-observed.py")
        if application_source.exists():
            raise RuntimeError("Refusing to overwrite an instrumented Application source")
        application_source.write_bytes(generated)
        application_hash = sha256(application_source)
        recorder.gc_observation = GcObservation()
    if args.application_source or args.observe_refresh_costs:
        spec = importlib.util.spec_from_file_location("ovd_ui.playback_benchmark_application", application_source)
        module = importlib.util.module_from_spec(spec)
        if args.observe_refresh_costs:
            module._pvd_benchmark_graph_begin = recorder.graph_begin
            module._pvd_benchmark_graph_end = recorder.graph_end
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        application_class = module.Application
    jobs = []
    for capture in args.captures:
        frames = list(range(args.start, args.start + args.frames))
        job = make_job(Path(capture).resolve(), frames, args.output.parent)
        if job["frames"] != frames:
            raise ValueError("The requested frame range extends beyond the capture")
        job["capture_sha256"] = sha256(job["capture"])
        for selection in args.selections:
            jobs.append({**job, "selection": selection})
    recorder.wrap_native_presentation()
    recorder.observe_property_arrays()
    phases = [(application_class, "_refresh", "application_refresh_ms"),
        (application_class, "_sync_interaction", "interaction_render_ms"),
        (application_class, "_sync_transport", "transport_widgets_ms"),
        (application_class, "_poll_workflow", "workflow_poll_ms"),
        (application_class, "_poll_load_progress", "load_poll_ms"),
        (PvdDataAdapter, "reload", "adapter_ms"),
        (PvdDataAdapter, "ownership_graph", "ownership_graph_ms"),
        (ObjectTreeModel, "populate", "tree_model_ms"),
        (ObjectBrowser, "refresh", "tree_widget_refresh_ms"),
        (_ObjectDelegate, "build_widget", "tree_widget_build_ms"),
        (PropertyInspector, "build", "property_build_ms"),
        (InteractionState, "update_frame", "interaction_frame_ms"),
        (ViewportRenderer, "set_snapshot", "snapshot_prepare_ms"),
        (ViewportRenderer, "_prepare", "geometry_prepare_ms")]
    for cls, name, phase in phases:
        recorder.wrap(cls, name, phase)
    recorder.wrap(application_class, "_transport", "transport_ms", tick=True)
    recorder.wrap(ViewportRenderer, "render", "render_submit_ms",
                  finish=args.synchronization == "viewport-and-update")
    app = application_class(Path(jobs[0]["capture"]), width=args.width, height=args.height, theme=args.theme)
    measurement = PlaybackMeasurement(app, args, jobs, recorder)
    original_next_frame = ui.next_frame

    async def next_frame():
        started = perf_counter()
        await original_next_frame()
        measurement.advance(started)

    ui.next_frame = next_frame
    def setup():
        ui.init("PVD production playback benchmark", width=args.width, height=args.height, max_fps=0)
        style.startup(args.theme)
    gc_controls_before = GcObservation.controls()
    if recorder.gc_observation is not None:
        recorder.gc_observation.start()
    try:
        app.run_with_backend(setup=setup)
    finally:
        recorder.restore()
        ui.next_frame = original_next_frame
        if recorder.gc_observation is not None:
            recorder.gc_observation.stop()
        gc_controls = gc_control_evidence(gc_controls_before)
    if not gc_controls["unchanged"]:
        raise RuntimeError("GC controls changed during sampling; discard this run")
    if app.run_error:
        raise app.run_error
    if measurement.phase != "done":
        raise RuntimeError("Application exited before all benchmark jobs completed")
    after = source_hashes()
    if (before != after or native_hash != sha256(_pvd_native.__file__) or application_hash != sha256(application_source)
            or original_application_hash != sha256(original_application_source)):
        raise RuntimeError("Production source or native module changed during sampling; discard this run")
    for result in measurement.results:
        if sha256(result["job"]["capture"]) != result["job"]["capture_sha256"]:
            raise RuntimeError("Capture changed during sampling; discard this run")
        result["screenshot_sha256"] = sha256(result["screenshot"])
        result["viewport_screenshot_sha256"] = sha256(result["viewport_screenshot"])
    libraries_after = runtime_libraries()
    for path, digest in measurement.libraries_before.items():
        if sha256(path) != digest:
            raise RuntimeError("Runtime library changed during sampling")
    if len({tuple(result["viewport"]) for result in measurement.results}) != 1:
        raise RuntimeError("The scenarios did not retain the same viewport dimensions")
    result = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Production Application.run_async: transport update + periodic polls + native owned presentation + tree/property/interaction + viewport GL completion + subsequent UI frame completion.",
        "runner": {"id": RUNNER_ID, "pump": "omni.ui.run_async",
            "ownership": "Both tasks joined, then application resources closed, then backend shutdown.",
            "comparison_limit": "Changes scheduling from historical ui.run(coroutine) reports. Those archives retain their original runner and are not relabelled or timing-equivalent."},
        "workload": args.workload,
        "dispatch": ("Unthrottled production playback tick inside the application loop."
            if args.workload == "normal-playback" else
            "The production Next-button transport handler runs immediately before the application loop; its synchronous work is included. Playback remains paused. No pointer-event delivery is measured."),
        "exclusions": "Initial open/setup, rewinds, warmup, screenshot readback, checksums, sample validation, intentional 60Hz throttle, live ingestion, diagnostics windows, pointer picking and profiler instrumentation.",
        "nesting": "Do not sum nested phases. transport contains refresh; refresh contains native/adapter/tree/interaction/snapshot preparation. render includes remaining geometry preparation. UI wait includes deferred property construction, layout, composition/swap/platform wait and final glFinish. native extraction includes native seek; it is not a second seek observation.",
        "synchronization": ("glFinish before each sample and after next_frame" +
            (", plus after renderer.render" if args.synchronization == "viewport-and-update" else "") +
            "; records completed latency with normal CPU/GPU overlap reduced. No application FPS or Qt parity claim."),
        "completion_policy": dict(mode=args.synchronization, pre_sample_finish=True,
            viewport_phase_finish=args.synchronization == "viewport-and-update", post_ui_finish=True,
            post_ui_context="composition", post_ui_context_verified=True, swap_interval_requested=1,
            context_evidence="The actual GLX context observed inside each geometry draw equals the current context after every UI frame and before each measured update."),
        "instrumentation": {
            "inside_sample": "Method wrappers/counters and a GLX context call, actual glGetIntegerv(GL_VIEWPORT) query and dictionary allocation at each viewport draw. This observer overhead remains in completed latency; it is not treated as neutral or subtracted.",
            "between_samples": "Actual GL attachment/viewport checks, native geometry serialization/hashes and invariants run after completion and before scheduling the next update. Their elapsed time is recorded separately; they perturb CPU caches, GPU idle time and the production 30Hz polling cadence. Poll call counts remain in each sample.",
            "scope": "Separated completed-update latency under instrumentation, not uninterrupted application throughput or FPS."},
        "gc_controls": gc_controls,
        "settings_profile": args.settings_profile,
        "runtime_libraries_before": measurement.libraries_before,
        "runtime_libraries_after": libraries_after, "display_environment": display_environment(),
        "display": os.environ.get("DISPLAY"),
        "vsync": "The GLFW backend requests swap interval1. The benchmark does not override driver/X-server synchronization; any effective swap wait remains in ui_frame_wait_ms.",
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "platform": platform.platform(), "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ("ovui", "moderngl", "numpy")},
        "window": [args.width, args.height], "theme": args.theme,
        "warmup_passes": args.warmup, "measured_passes": args.repeat,
        "sources": before, "native_sha256": native_hash, "captures": measurement.results,
        "application_source": str(application_source), "application_source_sha256": application_hash,
        "application_source_original": str(original_application_source),
        "application_source_original_sha256": original_application_hash,
        "refresh_cost_observation": dict(enabled=bool(recorder.gc_observation), source_transform=source_transform,
            gc=recorder.gc_observation.summary() if recorder.gc_observation is not None else None,
            limitations="GC callbacks, phase interval bookkeeping and two source-copy calls add in-clock overhead. GC overlap is wall-clock correlation, not causal CPU attribution. No GC controls are changed. The completed-update start/end and GL completion barriers are unchanged."),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({Path(r["job"]["capture"]).name + "/" + r["job"]["selection"]: r["timings_ms"]
        for r in measurement.results}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="*", default=["test/data/ue4_shooter.pxd2", "test/data/multi_geom.ovd"])
    parser.add_argument("--start", type=int, default=30)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--selections", nargs="+", choices=("none", "actor"), default=["none", "actor"])
    parser.add_argument("--workload", choices=("normal-playback", "paused-next"), default="normal-playback")
    parser.add_argument("--settings-profile", choices=("application", "qt-reference"), default="application")
    parser.add_argument("--synchronization", choices=("viewport-and-update", "completed-update"), default="viewport-and-update")
    parser.add_argument("--condition", choices=("same-window", "same-geometry-fbo"), default="same-window")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--theme", choices=("dark", "light"), default="dark")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--application-source", type=Path,
                        help="Saved production Application source; all other modules remain from the working tree")
    parser.add_argument("--observe-refresh-costs", action="store_true",
                        help="Observe normal GC callbacks and a reversible graph-preparation span in a benchmark-only Application copy")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.start < 2 or min(args.frames, args.repeat) < 1 or args.warmup < 0:
        parser.error("Start must be at least 2 to measure an advancing update; frames/repeat must be positive and warmup nonnegative")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error("Refusing to overwrite existing benchmark evidence")
    evidence = args.output.parent / (args.output.stem + "-evidence")
    if evidence.exists():
        parser.error("Refusing to overwrite an existing benchmark evidence directory")
    if len({Path(capture).stem for capture in args.captures}) != len(args.captures):
        parser.error("Captures must have distinct basenames for evidence screenshots")
    if len(set(args.selections)) != len(args.selections):
        parser.error("Selections must be distinct for evidence screenshots")
    evidence.mkdir()
    os.chdir(ROOT)
    with tempfile.TemporaryDirectory(prefix="pvd-playback-settings-") as settings:
        os.environ["XDG_DATA_HOME"] = settings
        run(args)

#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure an existing Junk Shop light edit through the production viewport."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
ADDON = ROOT / "addon"
FIXTURES = ROOT / "tests" / "fixtures"
for _path in (SCRIPTS, ADDON, FIXTURES):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import validation  # noqa: E402
import visual_comparison  # noqa: E402
from fixture_manifest import fixture_input, load_manifest  # noqa: E402
from ovrtx_blender_example import color_presentation, render_requests  # noqa: E402
from run_blender_navigation import _production_materialization, _view3d_context  # noqa: E402


SCHEMA_VERSION = 1
ARTIFACT_ID = "ovrtx-existing-light-edit-responsiveness"
FIXTURE_ID = "perf_junk_shop_1280x720"
MAX_SAMPLES = validation.VISUAL_SAMPLES[FIXTURE_ID]
OBJECT_NAME = "Area.027"
LIGHT_NAME = "Area.019"
TARGET_COLOR = (1.0, 0.0, 1.0)
DURATION_NS = 2_000_000_000
TARGET_HZ = 120
EDIT_COUNT = DURATION_NS * TARGET_HZ // 1_000_000_000
MEASUREMENT_TIMEOUT_S = 300
LOAD_INTERVAL_S = 1.0 / TARGET_HZ
COLOR_PRESENTATION = color_presentation.MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=FIXTURES)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "out" / "artifacts" / "existing-light-edit-responsiveness" / "result.json",
    )
    parser.add_argument("--blender-command", default="blender")
    parser.add_argument("--native-client-path", type=Path)
    parser.add_argument("--native-client-module", default="ovrtx_bridge_client")
    parser.add_argument("--worker-command")
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--inside-blender", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--config", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(list(argv))
    args.source_root = args.source_root.expanduser().resolve()
    if args.inside_blender and args.config is None:
        parser.error("--inside-blender requires --config")
    if not args.inside_blender and (
        args.native_client_path is None or not args.worker_command
    ):
        parser.error("--native-client-path and --worker-command are required")
    return args


def _argv() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]


def _color(authored: Sequence[float], elapsed_ns: int) -> list[float]:
    fraction = min(max(int(elapsed_ns) / DURATION_NS, 0.0), 1.0)
    return [
        float(start) + (target - float(start)) * fraction
        for start, target in zip(authored, TARGET_COLOR, strict=True)
    ]


def _light_state(light: Any) -> dict[str, Any]:
    return {
        "color": [float(value) for value in light.color],
        "energy": float(light.energy),
        "exposure": float(getattr(light, "exposure", 0.0)),
        "normalize": bool(getattr(light, "normalize", False)),
        "shape": str(light.shape),
        "size": float(light.size),
        "size_y": float(light.size_y),
    }


def _same_color(left: Sequence[float], right: Sequence[float]) -> bool:
    return len(left) == len(right) and all(
        math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1.0e-6)
        for a, b in zip(left, right, strict=True)
    )


def _validate_run(run: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    identity = run.get("identity", {})
    if identity != {
        "object_name": OBJECT_NAME,
        "object_match_count": 1,
        "object_type": "LIGHT",
        "light_data_name": LIGHT_NAME,
        "light_data_match_count": 1,
    }:
        failures.append("run.identity")
    if run.get("target_hz") != TARGET_HZ or run.get("status") != "pass":
        failures.append("run.status")
    authored = run.get("authored_state", {})
    terminal = run.get("terminal", {})
    blender_state = terminal.get("blender_state", {})
    if not _same_color(blender_state.get("color", ()), TARGET_COLOR):
        failures.append("run.terminal.blender_color")
    for field in ("energy", "exposure", "normalize", "shape", "size", "size_y"):
        if authored.get(field) != blender_state.get(field):
            failures.append(f"run.terminal.preserved_{field}")
    observations = run.get("observations")
    if not isinstance(observations, list) or not observations:
        return failures + ["run.observations"]
    try:
        monotonic = [int(event["monotonic_ns"]) for event in observations]
        if monotonic != sorted(monotonic):
            raise ValueError
        edits = [event for event in observations if event.get("kind") == "edit"]
        publications = [
            event for event in observations if event.get("kind") == "publication"
        ]
        if [event["edit_index"] for event in edits] != list(range(1, EDIT_COUNT + 1)):
            raise ValueError
        if [event["workload_elapsed_ns"] for event in edits] != [
            str(index * DURATION_NS // EDIT_COUNT)
            for index in range(1, EDIT_COUNT + 1)
        ]:
            raise ValueError
        previous_revision = edits[0]["first_presentation_revision"] - 1
        for event in edits:
            if (
                event["first_presentation_revision"] != previous_revision + 1
                or event["presentation_revision"]
                < event["first_presentation_revision"]
            ):
                raise ValueError
            previous_revision = event["presentation_revision"]
        if not _same_color(edits[-1]["color"], TARGET_COLOR):
            raise ValueError
        offer_timing = run["offer_timing"]
        offer_duration = int(edits[-1]["monotonic_ns"]) - int(
            edits[0]["monotonic_ns"]
        )
        if (
            offer_duration <= 0
            or offer_timing["duration_ns"] != str(offer_duration)
            or not math.isclose(
                offer_timing["mean_hz"],
                (len(edits) - 1) * 1_000_000_000 / offer_duration,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
        ):
            raise ValueError
        session_identity = run["session_identity"]
        if (
            session_identity["initial"] != session_identity["final"]
            or not session_identity["initial"]["simulation_id"]
            or not session_identity["initial"]["generation_digest"]
        ):
            raise ValueError
        by_revision = {
            revision: event
            for event in edits
            for revision in range(
                event["first_presentation_revision"],
                event["presentation_revision"] + 1,
            )
        }
        if len(publications) < 2 or len(
            {event["presentation_revision"] for event in publications}
        ) < 2:
            raise ValueError
        previous_publication = 0
        previous_edit_index = 0
        previous_presentation_revision = 0
        previous_visible = -1
        for event in publications:
            edit = by_revision[event["presentation_revision"]]
            if event["edit_index"] != edit["edit_index"]:
                raise ValueError
            if (
                event["edit_index"] < previous_edit_index
                or event["presentation_revision"] < previous_presentation_revision
            ):
                raise ValueError
            if event["publication_index"] <= previous_publication:
                raise ValueError
            visible = int(event["monotonic_ns"])
            if visible <= previous_visible or visible < int(edit["monotonic_ns"]):
                raise ValueError
            if event["applied_revision"] <= edit["applied_revision_before"]:
                raise ValueError
            previous_publication = event["publication_index"]
            previous_edit_index = event["edit_index"]
            previous_presentation_revision = event["presentation_revision"]
            previous_visible = visible
        terminal_edit = edits[-1]
        final_publication = terminal["publication"]
        if (
            final_publication["presentation_revision"]
            != terminal_edit["presentation_revision"]
            or final_publication["edit_index"] != EDIT_COUNT
            or final_publication["completed_samples"] < MAX_SAMPLES
        ):
            raise ValueError
        runtime_target = terminal["accepted_runtime_target"]
        if (
            runtime_target["attribute"] != "inputs:color"
            or not _same_color(runtime_target["value"], TARGET_COLOR)
            or runtime_target["value_type"] != "Color3f"
            or runtime_target["prim_path"] != terminal["light_prim_path"]
        ):
            raise ValueError
        application = terminal["runtime_application"]
        if (
            application.get("values_written") is not True
            or terminal["light_prim_path"] not in application.get("value_paths", ())
        ):
            raise ValueError
        image = terminal["image_candidate"]
        if (
            image.get("width") != 1280
            or image.get("height") != 720
            or not _sha256(image.get("sha256"))
            or image.get("publication_index")
            != final_publication["publication_index"]
            or image.get("presentation_revision")
            != final_publication["presentation_revision"]
        ):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        failures.append("run.observations_or_terminal")
    return failures


def validate_record(record: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    if (
        record.get("schema_version") != SCHEMA_VERSION
        or record.get("artifact_id") != ARTIFACT_ID
        or record.get("status") != "pass"
        or record.get("failure_reasons") != []
    ):
        failures.append("record.identity")
    workload = record.get("workload", {})
    if (
        workload.get("fixture_id") != FIXTURE_ID
        or not _sha256(workload.get("blend_file_sha256"))
        or workload.get("duration_ns") != str(DURATION_NS)
        or workload.get("target_hz") != TARGET_HZ
        or workload.get("edit_count") != EDIT_COUNT
        or workload.get("target_color") != list(TARGET_COLOR)
        or workload.get("sample_policy") != "elapsed-fraction-clamped-linear-v1"
        or workload.get("color_presentation") != COLOR_PRESENTATION
    ):
        failures.append("workload")
    run = record.get("run")
    if not isinstance(run, Mapping):
        return failures + ["run"]
    failures.extend(_validate_run(run))
    final_image = record.get("final_image", {})
    baseline_image = record.get("baseline_image", {})
    try:
        selected_image = run["terminal"]["image_candidate"]
        if (
            final_image["sha256"] != selected_image["sha256"]
            or final_image["presentation_revision"]
            != run["terminal"]["publication"]["presentation_revision"]
        ):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        failures.append("final_image")
    try:
        if baseline_image["sha256"] != run["baseline_image_candidate"]["sha256"]:
            raise ValueError
    except (KeyError, TypeError, ValueError):
        failures.append("baseline_image")
    if not _visible_change(record.get("baseline_comparison", {})):
        failures.append("baseline_comparison")
    return sorted(set(failures))


def _visible_change(comparison: Mapping[str, Any]) -> bool:
    metrics = comparison.get("metrics")
    # Reuse the visual regression budget; here exceeding it is the
    # expected evidence that the terminal light edit visibly changed the frame.
    return (
        comparison.get("outcome") == "regression"
        and isinstance(metrics, Mapping)
        and metrics.get("alpha_mismatches") == 0
    )


def _sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


class _Measurement:
    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        scene: Any,
        light_object: Any,
        light_prim_path: str,
        authored_state: Mapping[str, Any],
        materialization: Mapping[str, Any],
    ) -> None:
        self.config = config
        self.scene = scene
        self.light_object = light_object
        self.light = light_object.data
        self.light_prim_path = light_prim_path
        self.authored_state = dict(authored_state)
        self.materialization = dict(materialization)
        self.observations: list[dict[str, Any]] = []
        self.edits_by_revision: dict[int, dict[str, Any]] = {}
        self.last_frame: Any = None
        self.baseline_frame: Any = None
        self.terminal_frame: Any = None
        self.started_ns = 0
        self.last_edit_index = 0
        self.initial_session_identity: dict[str, Any] = {}
        self.failure = ""

    def post_pixel(self) -> None:
        from ovrtx_blender_example import engine

        engines = list(engine._ACTIVE_VIEWPORT_ENGINES)
        if len(engines) != 1:
            return
        instance = engines[0]
        frame = instance._presented_frame
        if frame is None or frame is self.last_frame:
            return
        self.last_frame = frame
        if (
            frame.publication_index != instance._presented_publication_index
            or frame.publication_index != instance._texture_snapshot_index
        ):
            self.failure = "POST_PIXEL publication/upload/draw identity mismatch"
            return
        if not self.started_ns:
            if frame.completed_samples >= int(self.config["max_samples"]):
                self.baseline_frame = frame
            return
        if not self.edits_by_revision:
            return
        revision = int(frame.presentation_revision)
        edit = self.edits_by_revision.get(revision)
        if edit is None:
            return
        event = {
            "kind": "publication",
            "monotonic_ns": str(time.perf_counter_ns()),
            "publication_index": int(frame.publication_index),
            "presentation_revision": revision,
            "applied_revision": int(frame.applied_revision),
            "edit_index": int(edit["edit_index"]),
            "completed_samples": int(frame.completed_samples),
        }
        self.observations.append(event)
        if (
            event["edit_index"] == EDIT_COUNT
            and event["completed_samples"] >= int(self.config["max_samples"])
        ):
            self.terminal_frame = frame

    def start(self) -> None:
        if self.baseline_frame is None:
            raise RuntimeError("baseline frame was not presented")
        self.initial_session_identity = self._session_identity()
        self.started_ns = time.perf_counter_ns()

    def offer_due_edits(self) -> None:
        import bpy

        if (
            not self.started_ns
            or self.terminal_frame is not None
        ):
            return
        elapsed = min(time.perf_counter_ns() - self.started_ns, DURATION_NS)
        due = min(EDIT_COUNT, int(elapsed * TARGET_HZ // 1_000_000_000))
        runtime = self._runtime()
        scheduler = runtime.scheduler
        if self.last_edit_index < due:
            index = self.last_edit_index + 1
            revision_before = int(scheduler.presentation_revision)
            applied_revision_before = int(scheduler.applied_revision)
            offered_ns = time.perf_counter_ns()
            workload_elapsed_ns = index * DURATION_NS // EDIT_COUNT
            color = _color(self.authored_state["color"], workload_elapsed_ns)
            self.light.color = color
            bpy.context.view_layer.update()
            revision = int(scheduler.presentation_revision)
            if revision <= revision_before:
                raise RuntimeError(
                    "light edit did not advance the scheduler revision: "
                    f"before={revision_before} after={revision}"
                )
            event = {
                "kind": "edit",
                "monotonic_ns": str(offered_ns),
                "workload_elapsed_ns": str(workload_elapsed_ns),
                "edit_index": index,
                "color": color,
                "applied_revision_before": applied_revision_before,
                "first_presentation_revision": revision_before + 1,
                "presentation_revision": revision,
            }
            self.observations.append(event)
            for represented_revision in range(revision_before + 1, revision + 1):
                self.edits_by_revision[represented_revision] = event
            self.last_edit_index = index

    def complete(self) -> dict[str, Any]:
        from ovrtx_blender_example import engine, scene_generation_sessions

        if self.failure:
            raise RuntimeError(self.failure)
        if self.last_edit_index != EDIT_COUNT or self.terminal_frame is None:
            raise RuntimeError("terminal edit did not reach a settled visible publication")
        runtime = self._runtime()
        generation = scene_generation_sessions.generation_for_scene(self.scene)
        retained = runtime.owner.retained_values_for(generation)[1]
        targets = [
            value
            for value in retained
            if value.prim_path == self.light_prim_path
            and value.attribute == "inputs:color"
        ]
        if len(targets) != 1:
            raise RuntimeError("terminal retained runtime light color is not unique")
        target = targets[0]
        baseline_image = engine._write_image(
            str(self.config["baseline_image_candidate"]),
            self.baseline_frame.render_result,
        )
        if baseline_image.get("error"):
            raise RuntimeError(str(baseline_image["error"]))
        image = engine._write_image(
            str(self.config["image_candidate"]), self.terminal_frame.render_result
        )
        if image.get("error"):
            raise RuntimeError(str(image["error"]))
        terminal_edit = self.edits_by_revision[int(self.terminal_frame.presentation_revision)]
        publication = next(
            event
            for event in reversed(self.observations)
            if event.get("kind") == "publication"
            and event["publication_index"] == self.terminal_frame.publication_index
        )
        application = dict(runtime.scheduler.diagnostics().get("last_edit_update", {}))
        final_session_identity = self._session_identity()
        if final_session_identity != self.initial_session_identity:
            raise RuntimeError("light edits replaced the live generation or OVRTX session")
        edit_events = [
            event for event in self.observations if event.get("kind") == "edit"
        ]
        offer_duration_ns = int(edit_events[-1]["monotonic_ns"]) - int(
            edit_events[0]["monotonic_ns"]
        )
        run = {
            "status": "pass",
            "target_hz": int(self.config["target_hz"]),
            "identity": dict(self.config["identity"]),
            "authored_state": self.authored_state,
            "materialization": self.materialization,
            "session_identity": {
                "initial": self.initial_session_identity,
                "final": final_session_identity,
            },
            "offer_timing": {
                "duration_ns": str(offer_duration_ns),
                "mean_hz": (len(edit_events) - 1)
                * 1_000_000_000
                / offer_duration_ns,
            },
            "observations": sorted(
                self.observations, key=lambda event: int(event["monotonic_ns"])
            ),
            "baseline_image_candidate": baseline_image,
            "terminal": {
                "edit": terminal_edit,
                "publication": publication,
                "blender_state": _light_state(self.light),
                "light_prim_path": self.light_prim_path,
                "accepted_runtime_target": {
                    "prim_path": target.prim_path,
                    "attribute": target.attribute,
                    "value": list(target.value),
                    "value_type": target.value_type,
                },
                "runtime_application": application,
                "image_candidate": {
                    **image,
                    "publication_index": int(self.terminal_frame.publication_index),
                    "presentation_revision": int(self.terminal_frame.presentation_revision),
                    "applied_revision": int(self.terminal_frame.applied_revision),
                    "completed_samples": int(self.terminal_frame.completed_samples),
                    "terminal_edit_index": int(terminal_edit["edit_index"]),
                },
            },
        }
        return run

    def timeout_record(self) -> dict[str, Any]:
        return {
            "status": "timeout",
            "target_hz": int(self.config["target_hz"]),
            "identity": dict(self.config["identity"]),
            "authored_state": self.authored_state,
            "edit_count": self.last_edit_index,
            "expected_edit_count": EDIT_COUNT,
            "session_identity": {
                "initial": self.initial_session_identity,
                "current": self._session_identity(),
            },
            "observations": sorted(
                self.observations, key=lambda event: int(event["monotonic_ns"])
            ),
        }

    def _runtime(self) -> Any:
        from ovrtx_blender_example import engine

        engines = list(engine._ACTIVE_VIEWPORT_ENGINES)
        if len(engines) != 1 or engines[0]._viewport_generation_runtime is None:
            raise RuntimeError("measurement requires one current-scene authoring runtime")
        return engines[0]._viewport_generation_runtime

    def _session_identity(self) -> dict[str, Any]:
        runtime = self._runtime()
        controller = runtime.ovrtx.controller
        diagnostics = controller.diagnostics()
        generation = runtime.owner.current_generation
        return {
            "controller_instance_id": hex(id(controller)),
            "simulation_id": diagnostics["simulation_id"],
            "lifecycle_event_count": len(diagnostics["lifecycle_events"]),
            "generation_number": generation.number,
            "generation_digest": generation.digest,
        }


def _inside_blender(args: argparse.Namespace) -> int:
    import bpy

    config = json.loads(args.config.read_text(encoding="utf-8"))
    for value in (config["addon_path"], config["native_client_path"]):
        if value and value not in sys.path:
            sys.path.insert(0, value)
    bpy.context.preferences.view.show_splash = False
    bpy.ops.wm.open_mainfile(filepath=config["blend_path"])
    os.environ["OV_BLENDER_EXAMPLE_WORKER_COMMAND"] = config["worker_command"]
    os.environ["OV_BLENDER_EXAMPLE_NATIVE_CLIENT_MODULE"] = config["native_client_module"]
    os.environ[color_presentation.ENV_COLOR_PRESENTATION_MODE] = COLOR_PRESENTATION

    import ovrtx_blender_example
    from ovrtx_blender_example import engine, scene_generation_sessions
    from ovrtx_blender_example.blender_callback_adapters import BlenderRenderCallbackAdapter
    from ovrtx_blender_example.interactive_setup import configure_scene, configure_viewports

    ovrtx_blender_example.register()
    state = SimpleNamespace(
        measurement=None,
        handler=None,
        full_area=False,
        deadline=None,
    )

    def quit_blender() -> None:
        ovrtx_blender_example.unregister()
        bpy.ops.wm.quit_blender()

    def fail(exc: BaseException) -> None:
        Path(config["error_result"]).write_text(
            json.dumps({"error": f"{type(exc).__name__}: {exc}"}, indent=2) + "\n",
            encoding="utf-8",
        )
        quit_blender()

    def post_pixel() -> None:
        try:
            if state.measurement is not None:
                state.measurement.post_pixel()
        except BaseException as exc:
            fail(exc)

    def write_timeout_record() -> None:
        measurement = state.measurement
        raw_result = Path(config["raw_result"])
        if measurement is None or raw_result.exists():
            return
        raw_result.write_text(
            json.dumps(
                measurement.timeout_record(),
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def pump() -> float | None:
        try:
            measurement = state.measurement
            raw_result = Path(config["raw_result"])
            if state.deadline is not None and time.monotonic() >= state.deadline:
                write_timeout_record()
                raise TimeoutError("light edit responsiveness run reached its timeout")
            if measurement.failure:
                raise RuntimeError(measurement.failure)
            if not measurement.started_ns:
                if measurement.baseline_frame is None:
                    return float(config["load_interval_s"])
                measurement.start()
                state.deadline = time.monotonic() + float(config["timeout_s"])
                timeout_timer = threading.Timer(
                    float(config["timeout_s"]),
                    write_timeout_record,
                )
                timeout_timer.daemon = True
                timeout_timer.start()
            measurement.offer_due_edits()
            if measurement.terminal_frame is None:
                return float(config["load_interval_s"])
            raw_result.write_text(
                json.dumps(measurement.complete(), allow_nan=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            quit_blender()
            return None
        except BaseException as exc:
            fail(exc)
            return None

    def setup() -> float | None:
        try:
            context = _view3d_context(bpy)
            if context is None:
                return 0.1
            window, area, region, space = context
            if not state.full_area:
                space.show_region_toolbar = False
                space.show_region_ui = False
                with bpy.context.temp_override(window=window, area=area):
                    bpy.ops.screen.screen_full_area(use_hide_panels=True)
                state.full_area = True
                return 0.25
            objects = [item for item in bpy.data.objects if item.name == OBJECT_NAME]
            lights = [item for item in bpy.data.lights if item.name == LIGHT_NAME]
            if len(objects) != 1 or len(lights) != 1:
                raise RuntimeError("Junk Shop light identity is missing or ambiguous")
            light_object = objects[0]
            if light_object.type != "LIGHT" or light_object.data is not lights[0]:
                raise RuntimeError("Area.027 does not resolve to LIGHT datablock Area.019")
            bpy.ops.object.select_all(action="DESELECT")
            light_object.select_set(True)
            bpy.context.view_layer.objects.active = light_object
            authored_state = _light_state(light_object.data)
            for field, value in authored_state.items():
                if field != "color" and hasattr(light_object.data, field):
                    setattr(light_object.data, field, value)
            light_object.data.color = authored_state["color"]
            bpy.context.view_layer.update()
            with bpy.context.temp_override(window=window, area=area, region=region, region_data=space.region_3d):
                scene = bpy.context.scene
                configure_scene(scene, config)
                generation = scene_generation_sessions.generation_for_scene(scene)
                request = BlenderRenderCallbackAdapter(
                    generation_for_scene=lambda _scene: generation
                ).final_render_from_scene(scene)
                config.update(camera_prim_path=request.camera_prim_path, render_product_path=request.render_product_path)
                configure_scene(scene, config)
                generation = scene_generation_sessions.generation_for_scene(scene)
                request = BlenderRenderCallbackAdapter(
                    generation_for_scene=lambda _scene: generation
                ).final_render_from_scene(scene)
                config.update(usd_path=generation.materialize_usd(), camera_prim_path=request.camera_prim_path, render_product_path=request.render_product_path)
                configure_viewports(bpy, scene, config, screen=window.screen)
                resolver, _light_objects = scene_generation_sessions.current_generation_edit_context(scene)
                resolution = resolver.resolve_light(light_object)
                if resolution.value is None:
                    raise RuntimeError("Area.019 did not resolve to one current-generation USD light")
                state.measurement = _Measurement(
                    config=config,
                    scene=scene,
                    light_object=light_object,
                    light_prim_path=resolution.value.prim_path,
                    authored_state=authored_state,
                    materialization=_production_materialization(generation, request),
                )
                state.handler = space.draw_handler_add(post_pixel, (), "WINDOW", "POST_PIXEL")
                area.tag_redraw()
                bpy.app.timers.register(pump, first_interval=0.01)
                return None
        except BaseException as exc:
            fail(exc)
            return None

    bpy.app.timers.register(setup, first_interval=0.5)
    return 0


def _subrun(args: argparse.Namespace, fixture: Mapping[str, Any]) -> dict[str, Any]:
    work_dir = args.output.resolve().parent / ".existing-light-edit-responsiveness"
    work_dir.mkdir(parents=True, exist_ok=True)
    raw_result = work_dir / "raw-result.json"
    error_result = work_dir / "error.json"
    baseline_image_candidate = work_dir / "baseline.png"
    image_candidate = work_dir / "final.png"
    for path in (raw_result, error_result, baseline_image_candidate, image_candidate):
        path.unlink(missing_ok=True)
    config = {
        "addon_path": str(ADDON),
        "source_root": str(args.source_root),
        "fixture_id": fixture["id"],
        "blend_path": str(Path(fixture["path"]).resolve()),
        "blend_sha256": fixture["sha256"],
        "camera_prim_path": "???",
        "width": 1280,
        "height": 720,
        "min_samples": 1,
        "max_samples": MAX_SAMPLES,
        "color_presentation": COLOR_PRESENTATION,
        "native_client_module": args.native_client_module,
        "native_client_path": str(args.native_client_path.resolve()),
        "worker_command": args.worker_command,
        "sync_viewport_camera": True,
        "selectable_imported_objects": False,
        "tag_dynamic_body_transforms": False,
        "shared_stage_composition": False,
        "body_root": "/World/PhysicsIsland/DynamicBodies",
        "composition_update_fps": 30.0,
        "viewport_redraw_pressure_mode": "scheduled",
        "forced_redraw_timer_type": "DRAW_WIN_SWAP",
        "interactive_duration_s": 0.0,
        "timeline_end_frame": 120,
        "viewport_orbit_distance": None,
        "viewport_pitch_offset_degrees": None,
        "viewport_clip_start": None,
        "viewport_clip_end": None,
        "target_hz": TARGET_HZ,
        "load_interval_s": LOAD_INTERVAL_S,
        "timeout_s": MEASUREMENT_TIMEOUT_S,
        "identity": {
            "object_name": OBJECT_NAME,
            "object_match_count": 1,
            "object_type": "LIGHT",
            "light_data_name": LIGHT_NAME,
            "light_data_match_count": 1,
        },
        "raw_result": str(raw_result),
        "error_result": str(error_result),
        "baseline_image_candidate": str(baseline_image_candidate),
        "image_candidate": str(image_candidate),
    }
    config_path = work_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    env = os.environ.copy()
    env.pop("OV_BLENDER_EXAMPLE_VIEWPORT_ARTIFACT", None)
    env.pop("OV_BLENDER_EXAMPLE_VIEWPORT_PROFILE", None)
    env.update(
        {
            "OVRTX_ACTIVE_CUDA_GPUS": "0",
            render_requests.ENV_FIXED_VIEWPORT_RESOLUTION: "1",
            "OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR": str(work_dir / "viewport-work"),
            "OV_BLENDER_EXAMPLE_WORKER_LOG": str(work_dir / "worker.log"),
        }
    )
    command = [
        args.blender_command,
        "--factory-startup",
        "--window-geometry",
        "20",
        "20",
        "1280",
        "720",
        "--python",
        str(Path(__file__).resolve()),
        "--",
        "--inside-blender",
        "--source-root",
        str(args.source_root),
        "--config",
        str(config_path),
    ]
    exit_status = validation.SubprocessExecutor(verbose=True).run(
        command, cwd=ROOT, output_dir=work_dir, label="blender", env=env
    ).exit_status
    run = json.loads(raw_result.read_text(encoding="utf-8")) if raw_result.exists() else None
    if run is not None and run.get("status") == "timeout":
        return run
    if exit_status or error_result.exists() or run is None:
        detail = error_result.read_text() if error_result.exists() else f"Blender exited {exit_status}"
        raise RuntimeError(f"load run failed: {detail.strip()}")
    failures = _validate_run(run)
    if failures:
        raise RuntimeError("load run validation failed: " + ", ".join(failures))
    return run


def _workload(fixture: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "fixture_id": FIXTURE_ID,
        "blend_file_sha256": fixture["sha256"],
        "duration_ns": str(DURATION_NS),
        "target_hz": TARGET_HZ,
        "edit_count": EDIT_COUNT,
        "object_name": OBJECT_NAME,
        "light_data_name": LIGHT_NAME,
        "target_color": list(TARGET_COLOR),
        "sample_policy": "elapsed-fraction-clamped-linear-v1",
        "color_presentation": COLOR_PRESENTATION,
        "width": 1280,
        "height": 720,
        "min_samples": 1,
        "max_samples": MAX_SAMPLES,
        "load_interval_ns": str(round(LOAD_INTERVAL_S * 1_000_000_000)),
    }


def _outer(args: argparse.Namespace) -> int:
    fixture = fixture_input(load_manifest(args.manifest.resolve()), FIXTURE_ID)
    base = {
        "schema_version": SCHEMA_VERSION,
        "artifact_id": ARTIFACT_ID,
        "status": "failed",
        "failure_reasons": [],
        "workload": _workload(fixture),
        "run": {},
        "final_image": {},
        "baseline_image": {},
        "baseline_comparison": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run: dict[str, Any] = {}
    record = dict(base)
    try:
        run = _subrun(args, fixture)
        if run["status"] == "timeout":
            raise TimeoutError(
                f"load run completed {run['edit_count']} of "
                f"{run['expected_edit_count']} edits before the five-minute timeout"
            )
        selected_source = Path(run["terminal"]["image_candidate"]["path"])
        selected_path = args.output.with_suffix(".png")
        shutil.copyfile(selected_source, selected_path)
        selected_sha = hashlib.sha256(selected_path.read_bytes()).hexdigest()
        baseline_source = Path(run["baseline_image_candidate"]["path"])
        baseline_path = args.output.with_name(f"{args.output.stem}.baseline.png")
        shutil.copyfile(baseline_source, baseline_path)
        baseline_sha = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
        final_publication = run["terminal"]["publication"]
        baseline_comparison = visual_comparison.compare(
            baseline_path,
            selected_path,
            expected_width=1280,
            expected_height=720,
            expected_presentation=COLOR_PRESENTATION,
            control_presentation=COLOR_PRESENTATION,
            contender_presentation=COLOR_PRESENTATION,
        )
        record = {
            **base,
            "status": "pass",
            "run": run,
            "baseline_image": {
                "path": str(baseline_path),
                "sha256": baseline_sha,
            },
            "baseline_comparison": baseline_comparison,
            "final_image": {
                "path": str(selected_path),
                "sha256": selected_sha,
                "selection_policy": "settled-publication-for-acknowledged-terminal-edit-v1",
                "publication_index": final_publication["publication_index"],
                "presentation_revision": final_publication["presentation_revision"],
                "terminal_edit_index": final_publication["edit_index"],
            },
        }
        failures = validate_record(record)
        if failures:
            raise RuntimeError("record validation failed: " + ", ".join(failures))
    except BaseException as exc:
        record = {
            **record,
            "status": "failed",
            "failure_reasons": [f"{type(exc).__name__}: {exc}"],
            "run": run,
        }
    args.output.write_text(json.dumps(record, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": record["status"], "result": str(args.output), "failure_reasons": record["failure_reasons"]}, indent=2, sort_keys=True))
    return 0 if record["status"] == "pass" else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(_argv() if argv is None else argv)
    return _inside_blender(args) if args.inside_blender else _outer(args)


if __name__ == "__main__":
    if "--inside-blender" in _argv():
        main()
    else:
        raise SystemExit(main())

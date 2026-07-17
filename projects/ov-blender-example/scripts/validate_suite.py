#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run one semantic validation suite directly."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import semantic_validation as inventory
import validation
import visual_comparison

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
FIXTURES = TESTS / "fixtures"
for path in (TESTS, FIXTURES):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from blender_test_support import blender_executable  # noqa: E402
from fixture_manifest import fixture_input, load_manifest, render_fixture  # noqa: E402


def _pytest(items: Sequence[str], deselect: Sequence[str] = ()) -> int:
    command = [sys.executable, "-m", "pytest", "-q", *items]
    command.extend(f"--deselect={item}" for item in deselect)
    return subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": ".", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        check=False,
    ).returncode


def _runtime(path: Path | None, addon_root: Path) -> dict[str, str]:
    if path is None:
        raise validation.ValidationError("--runtime-root is required for this suite")
    addon = str(addon_root)
    if addon not in sys.path:
        sys.path.insert(0, addon)
    from ovrtx_blender_example import bundled_runtime

    defaults = bundled_runtime.defaults(root=path.resolve())
    runtime = {
        "runtime_root": str(defaults.root),
        "ovrtx_worker_command": defaults.worker_command,
        "ovrtx_blender_client": defaults.native_client_path,
        "ovphysx_server": defaults.ovphysx_server,
        "ovphysx_blender_client": defaults.ovphysx_native_client_path,
        "ovphysx_bridge_root": defaults.ovphysx_bridge_runtime_root,
        "ovphysx_root": defaults.ovphysx_root,
        "ovruntime_root": defaults.ovruntime_root,
    }
    missing = [key for key, value in runtime.items() if key != "runtime_root" and not value]
    if missing:
        raise validation.ValidationError(
            "materialized runtime is incomplete: " + ", ".join(missing)
        )
    return runtime


def _probe(
    executor: validation.Executor,
    output: Path,
    runtime: Mapping[str, Any],
    blender: Path,
) -> bool:
    fixture = validation._fixture_manifest_record(ROOT, "demo_stair_drop_1280x720")
    workload = fixture[0] if fixture is not None else None
    check_output = output / "live-transform"
    check_output.mkdir()
    argv = [
        sys.executable,
        str(ROOT / "scripts/run_ovrtx_live_transform_probe.py"),
        "--output-dir", str(check_output),
        "--fixture-id", "demo_stair_drop_1280x720",
        "--manifest", str(ROOT / "tests/fixtures"),
        "--native-client-path", str(runtime["ovrtx_blender_client"]),
        "--worker-command", validation._worker_command(runtime),
    ]
    command = executor.run(
        argv,
        cwd=ROOT,
        output_dir=check_output,
        label="probe",
        env={"BLENDER_COMMAND": str(blender)},
    )
    result = validation._load_result(check_output / "result.json", command)
    if isinstance(result, dict):
        result["validation_color_presentation"] = (
            validation._live_transform_color_presentation(
                result, allow_legacy_ldr=False
            ) or "invalid"
        )
        validation._write_json(check_output / "result.json", result)
    failure = validation._probe_failure("live-transform", result, check_output)
    if not failure and (
        validation._probe_fixture_identity(result) != workload
    ):
        failure = "probe fixture identity does not match its manifest"
    return not failure


def _golden(suite: str, runtime: Mapping[str, str], output: Path | None) -> int:
    if output is None:
        raise validation.ValidationError("--output-dir is required for this suite")
    output.mkdir(parents=True, exist_ok=False)
    executor = validation.SubprocessExecutor(verbose=True)
    passed = True
    for fixture, golden in inventory.GOLDENS[suite]:
        case_output = output / golden
        case_output.mkdir()
        result = _golden_case(executor, fixture, golden, runtime, case_output)
        passed = result["outcome"] == "pass" and passed
    print(json.dumps({"suite": suite, "result": "pass" if passed else "fail"}))
    return 0 if passed else 1


def _golden_case(
    executor: validation.Executor,
    fixture_id: str,
    golden: str,
    runtime: Mapping[str, str],
    output: Path,
) -> dict[str, Any]:
    result_path = output / "result.json"
    candidate = output / "frame-0001.png"
    golden_root = ROOT / "docs" / "goldens" / golden / "single-frame"
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "ovrtx-golden-validation-case",
        "fixture": fixture_id,
        "golden": golden,
        "outcome": "unavailable",
        "candidate_image": str(candidate),
        "golden_evidence": validation._golden_evidence(golden, golden_root),
    }
    try:
        manifest = load_manifest(FIXTURES)
        selected = fixture_input(manifest, fixture_id)
        exact = render_fixture(manifest, fixture_id) if selected["kind"] == "usd" else None
        resolution = selected["resolution"]
        width, height = int(resolution["width"]), int(resolution["height"])
        if (width, height) != validation.VISUAL_DIMENSIONS[golden]:
            raise validation.ValidationError(
                f"fixture resolution {width}x{height} does not match golden {golden}"
            )
        blender = blender_executable()
        if blender is None:
            raise validation.ValidationError("Blender is unavailable")
        if result["golden_evidence"].get("status") != "pass":
            raise validation.ValidationError(
                str(result["golden_evidence"].get("reason", "golden is unavailable"))
            )
        expression = _golden_setup_expression(
            selected,
            exact,
            runtime,
            width,
            height,
            validation.VISUAL_SAMPLES[golden],
        )
        command = _golden_blender_command(str(blender), expression, output)
        env = {
            "OV_BLENDER_EXAMPLE_WORKER_COMMAND": validation._worker_command(runtime),
            "OV_BLENDER_EXAMPLE_NATIVE_CLIENT_MODULE": "ovrtx_bridge_client",
            "OV_BLENDER_EXAMPLE_NATIVE_CLIENT_PATH": runtime["ovrtx_blender_client"],
            "OV_BLENDER_EXAMPLE_RENDER_ARTIFACT": str(output / "render-artifact.json"),
            "OV_BLENDER_EXAMPLE_WORKER_LOG": str(output / "worker.log"),
        }
        with tempfile.TemporaryDirectory(prefix="ovrtx-blender-user-") as user_root:
            config_root = Path(user_root) / "config"
            scripts_root = Path(user_root) / "scripts"
            config_root.mkdir()
            scripts_root.mkdir()
            env["BLENDER_USER_CONFIG"] = str(config_root)
            env["BLENDER_USER_SCRIPTS"] = str(scripts_root)
            completed = executor.run(
                command,
                cwd=ROOT,
                output_dir=output,
                label="render",
                env=env,
            )
        result["command"] = completed.evidence()
        if completed.exit_status != 0:
            result["outcome"] = "render-failed"
            result["error"] = f"Blender exited {completed.exit_status}"
        else:
            artifact = output / "render-artifact.json"
            render_artifact = json.loads(artifact.read_text(encoding="utf-8"))
            color = (
                render_artifact.get("color_presentation")
                if isinstance(render_artifact, Mapping)
                else None
            )
            samples = validation.VISUAL_SAMPLES[golden]
            if not (
                isinstance(render_artifact, Mapping)
                and render_artifact.get("status") == "pass"
                and render_artifact.get("addon_path")
                == str((ROOT / "addon" / "ovrtx_blender_example").resolve())
                and render_artifact.get("width") == width
                and render_artifact.get("height") == height
                and render_artifact.get("min_samples") == samples
                and render_artifact.get("max_samples") == samples
                and render_artifact.get("completed_samples") == samples
                and render_artifact.get("render_var") == "HdrColor"
                and render_artifact.get("frame_format") == "rgba16f"
                and (
                    exact is None
                    or (
                        render_artifact.get("input_usd_path")
                        == exact["fixture_usd_path"]
                        and render_artifact.get("render_product_path")
                        == exact["render_product_path"]
                    )
                )
                and isinstance(color, Mapping)
                and color.get("requested_mode") == validation.COLOR_PRESENTATION
                and color.get("active_mode") == validation.COLOR_PRESENTATION
                and color.get("status") == "current_behavior"
                and color.get("render_var") == "HdrColor"
                and color.get("frame_format") == "rgba16f"
                and color.get("frame_color_mode") == "scene_linear"
                and color.get("display_transform_application_count") == 1
                and color.get("display_transform_applied_by") == "consumer"
                and color.get("display_transform_consistent") is True
            ):
                raise validation.ValidationError("render artifact does not match golden contract")
            result["render_artifact"] = render_artifact
            comparison = visual_comparison.compare(
                golden_root / "frame.png",
                candidate,
                expected_width=width,
                expected_height=height,
                expected_presentation=validation.COLOR_PRESENTATION,
                control_presentation=validation.COLOR_PRESENTATION,
                contender_presentation=str(color["active_mode"]),
            )
            result["comparison"] = comparison
            result["outcome"] = comparison["outcome"]
            if candidate.is_file():
                data = candidate.read_bytes()
                result["image_artifact"] = {
                    "path": str(candidate),
                    "size_bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
    except (OSError, ValueError, json.JSONDecodeError, validation.ValidationError) as error:
        result["outcome"] = "unavailable"
        result["error"] = str(error)
    validation._write_json(result_path, result)
    return result


def _golden_blender_command(
    blender: str,
    expression: str,
    output: Path,
    *,
    system: str | None = None,
) -> list[str]:
    command = [blender]
    if (system or platform.system()).lower() == "windows":
        command.extend(("--gpu-backend", "vulkan"))
    command.extend(("--background", "--disable-autoexec", "--factory-startup"))
    command.extend(
        (
            "--python-exit-code",
            "1",
            "--python-expr",
            expression,
            "--engine",
            "OVRTX_EXAMPLE",
            "--render-output",
            str(output / "frame-####"),
            "--render-format",
            "PNG",
            "--use-extension",
            "1",
            "--render-frame",
            "1",
        )
    )
    return command


def _golden_setup_expression(
    selected: Mapping[str, Any],
    exact: Mapping[str, Any] | None,
    runtime: Mapping[str, str],
    width: int,
    height: int,
    samples: int,
) -> str:
    document_setup = (
        f"bpy.ops.wm.open_mainfile(filepath={selected['path']!r})"
        if selected["kind"] == "blend"
        else ""
    )
    exact_setup = ""
    if exact is not None:
        exact_setup = f"""
from ovrtx_blender_example import engine as ovrtx_engine
ovrtx_engine.configure_exact_stage(
    input_usd_path={exact["fixture_usd_path"]!r},
    camera_prim_path={exact["camera_prim_path"]!r},
    render_product_path={exact["render_product_path"]!r},
)
bpy.ops.object.camera_add(location=(0.0, -4.0, 2.5), rotation=(1.05, 0.0, 0.0))
bpy.context.active_object.name = {exact["camera_prim_path"]!r}
bpy.context.scene.camera = bpy.context.active_object
"""
    return f"""
import sys
sys.path.insert(0, {str(ROOT / "addon")!r})
sys.path.insert(0, {runtime["ovrtx_blender_client"]!r})
import bpy
{document_setup}
import ovrtx_blender_example
ovrtx_blender_example.register()
{exact_setup}
scene = bpy.context.scene
scene.render.resolution_x = {width}
scene.render.resolution_y = {height}
scene.render.resolution_percentage = 100
scene.render.image_settings.color_mode = "RGBA"
scene.render.image_settings.color_depth = "8"
scene.ovrtx_example.min_samples = {samples}
scene.ovrtx_example.max_samples = {samples}
scene.ovrtx_example.color_presentation_mode = {validation.COLOR_PRESENTATION!r}
"""


def _performance(
    suite: str,
    runtime: Mapping[str, str],
    output: Path | None,
    selected: Sequence[str] | None = None,
) -> int:
    measurements = inventory.PERFORMANCE[suite]
    selected_names = tuple(measurements) if selected is None else tuple(selected)
    if not selected_names:
        print(json.dumps({"suite": suite, "measurements": []}))
        return 0
    if output is None:
        raise validation.ValidationError("--output-dir is required for this suite")
    blender = blender_executable()
    if blender is None:
        raise validation.ValidationError("Blender is unavailable")
    output.mkdir(parents=True, exist_ok=False)
    completed: list[str] = []

    def run(name: str) -> bool:
        script, presentation, reporter = measurements[name]
        interpreter = sys.executable
        result = output / f"{name}.json"
        command = [
            interpreter,
            str(ROOT / "scripts" / script),
            "--source-root",
            str(ROOT),
            "--output",
            str(result),
            "--blender-command",
            str(blender),
            "--native-client-path",
            runtime["ovrtx_blender_client"],
            "--worker-command",
            validation._worker_command(runtime),
        ]
        if presentation is not None:
            command.extend(("--color-presentation", presentation))
        succeeded = subprocess.run(command, cwd=ROOT, check=False).returncode == 0
        if succeeded and reporter is not None:
            print(f"{name}:", flush=True)
            succeeded = (
                subprocess.run(
                    [interpreter, str(ROOT / "scripts" / reporter), str(result)],
                    cwd=ROOT,
                    check=False,
                ).returncode
                == 0
            )
        if succeeded:
            completed.append(name)
        return succeeded

    failed = [name for name in selected_names if not run(name)]
    # ??? CI evidence indicates first-run RTX cache priming; retry a
    # failed member once only after a sibling proves this runner can render.
    if completed:
        for name in failed:
            run(name)
    completed = [name for name in selected_names if name in completed]
    print(
        json.dumps(
            {
                "suite": suite,
                "selected": selected_names,
                "completed": completed,
                "measurements": completed,
            }
        )
    )
    return 0 if len(completed) == len(selected_names) else 2


def _performance_selection(suite: str, requested: Sequence[str]) -> tuple[str, ...]:
    if suite != "performance-large":
        raise validation.ValidationError(
            "measurement selection is only supported for performance-large"
        )
    measurements = inventory.PERFORMANCE[suite]
    unknown = sorted(set(requested) - measurements.keys())
    if unknown:
        raise validation.ValidationError(
            "measurements do not belong to performance-large: " + ", ".join(unknown)
        )
    requested_set = set(requested)
    return tuple(name for name in measurements if name in requested_set)


def _inventory() -> dict[str, object]:
    unit = inventory.unit_tests(ROOT)
    return {
        "unit": {
            "pytest_files": unit,
            "deselect": (*inventory.BLENDER_TESTS, *inventory.EXCLUDED_TESTS),
        },
        "golden-small": {
            "driver": "Blender CLI",
            "cases": inventory.GOLDENS["golden-small"],
        },
        "golden-large": {
            "driver": "Blender CLI",
            "cases": inventory.GOLDENS["golden-large"],
        },
        "ov-integration": inventory.OV_INTEGRATION_TESTS,
        "blender-integration": {
            "pytest": inventory.BLENDER_TESTS,
            "probes": inventory.INTEGRATION_PROBES,
        },
        "performance-small": inventory.PERFORMANCE["performance-small"],
        "performance-large": inventory.PERFORMANCE["performance-large"],
        "excluded": inventory.EXCLUDED_PROBES,
        "excluded-tests": inventory.EXCLUDED_TESTS,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=inventory.SUITES)
    parser.add_argument("--list", action="store_true", help="print the complete inventory")
    parser.add_argument("--addon-root", type=Path, default=ROOT / "addon")
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--measurement",
        action="append",
        help="run a named performance-large measurement (repeatable)",
    )
    args = parser.parse_args(argv)
    if args.list:
        print(json.dumps(_inventory(), indent=2, sort_keys=True))
        return 0
    try:
        selected = (
            _performance_selection(args.suite, args.measurement)
            if args.measurement
            else None
        )
        addon_root = args.addon_root.resolve()
        if addon_root != (ROOT / "addon").resolve():
            raise validation.ValidationError(
                "--addon-root must be prepared as the add-on beside this test tree"
            )
        runtime = _runtime(args.runtime_root, addon_root)
    except (OSError, RuntimeError, ImportError, validation.ValidationError) as error:
        print(json.dumps({"suite": args.suite, "error": str(error)}))
        return 2
    if args.suite == "unit":
        return _pytest(
            inventory.unit_tests(ROOT),
            (*inventory.BLENDER_TESTS, *inventory.EXCLUDED_TESTS),
        )
    if args.suite == "ov-integration":
        return _pytest(inventory.OV_INTEGRATION_TESTS)
    if args.suite == "blender-integration":
        try:
            blender = blender_executable()
        except ValueError as error:
            print(
                json.dumps(
                    {"suite": args.suite, "result": "fail", "error": str(error)}
                )
            )
            return 2
        if blender is None:
            print(
                json.dumps(
                    {
                        "suite": args.suite,
                        "result": "fail",
                        "error": "Blender is unavailable",
                    }
                )
            )
            return 2
        if _pytest(inventory.BLENDER_TESTS) != 0:
            return 1
        try:
            if args.output_dir is None:
                raise validation.ValidationError(
                    "--output-dir is required for this suite"
                )
            args.output_dir.mkdir(parents=True, exist_ok=False)
            executor = validation.SubprocessExecutor(verbose=True)
            return 0 if _probe(executor, args.output_dir, runtime, blender) else 1
        except (OSError, RuntimeError, validation.ValidationError) as error:
            print(
                json.dumps(
                    {"suite": args.suite, "result": "fail", "error": str(error)}
                )
            )
            return 2
    try:
        if args.suite.startswith("golden-"):
            return _golden(args.suite, runtime, args.output_dir)
        return _performance(args.suite, runtime, args.output_dir, selected)
    except (OSError, RuntimeError, ValueError, validation.ValidationError) as error:
        print(json.dumps({"suite": args.suite, "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

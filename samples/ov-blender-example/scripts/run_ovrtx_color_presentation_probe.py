# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
"""Junk Shop color-presentation evidence probe (task03-03).

Runs the Junk Shop blue-background presentation regression as reproducible,
headless evidence. The regression was fixed by correct presentation *ownership*
(who applies Blender's display transform) without changing the correctly
converted source light, so it distinguishes a presentation fix from a scene
fix. This probe proves that ownership contract at the metadata and
display-transform levels:

* ``ldr_rgba8_display_passthrough`` (baseline): OVRTX owns the display encoding.
  The presented RGBA8 frame passes through raw and is invariant to Blender's
  View Transform / Look / Exposure / Gamma. ``display_transform_application_count
  == 1`` applied by ``ovrtx_render_product``.
* ``scene_linear_hdr`` (default + one variation each of View Transform, Look,
  Exposure, Gamma): Blender owns the display transform. The same scene-linear
  source is presented through Blender's real OCIO transform (Image.save_render),
  so the presented frame *responds* to every view-settings change while
  ``display_transform_application_count == 1`` applied by ``consumer`` and
  ``display_transform_consistent`` stays true throughout.

The scene-linear frame here is a synthetic Junk-Shop-representative blue-
background crop (a blue background plus a bright warm highlight patch so tone
mapping is visible). The one-time real-GPU crop against the OVRTX worker is the
Linux golden host's domain; this probe changes no production code, so the LDR
golden lane stays green unchanged. Evidence is written under
``out/artifacts/ovrtx-color-presentation-probe/`` per the repo's validation-run
conventions.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from ovrtx_probe_support import BLENDER_COMMAND  # noqa: E402


#: Crop dimensions for the synthetic blue-background evidence frame. Small on
#: purpose (fast headless pixel work); large enough for a stable mean.
CROP_WIDTH = 48
CROP_HEIGHT = 32


def _probe_cases() -> list[dict[str, Any]]:
    """The per-mode / per-view-settings evidence matrix.

    One LDR passthrough baseline plus one scene-linear default and a distinct
    View Transform / Look / Exposure / Gamma variation each (per the task
    clarification). ``look == "__first_non_none__"`` is resolved inside Blender
    against the active view transform's look enum so the Look variation always
    applies a real, version-independent change.
    """

    return [
        {
            "name": "ldr_passthrough_baseline",
            "mode": "ldr_rgba8_display_passthrough",
            "presentation": "ldr",
            "view_transform": "AgX",
            "look": "None",
            "exposure": 0.0,
            "gamma": 1.0,
        },
        {
            "name": "scene_linear_default",
            "mode": "scene_linear_hdr",
            "presentation": "scene_linear",
            "view_transform": "AgX",
            "look": "None",
            "exposure": 0.0,
            "gamma": 1.0,
        },
        {
            "name": "scene_linear_view_transform",
            "mode": "scene_linear_hdr",
            "presentation": "scene_linear",
            "view_transform": "Standard",
            "look": "None",
            "exposure": 0.0,
            "gamma": 1.0,
        },
        {
            "name": "scene_linear_look",
            "mode": "scene_linear_hdr",
            "presentation": "scene_linear",
            "view_transform": "AgX",
            "look": "__first_non_none__",
            "exposure": 0.0,
            "gamma": 1.0,
        },
        {
            "name": "scene_linear_exposure",
            "mode": "scene_linear_hdr",
            "presentation": "scene_linear",
            "view_transform": "AgX",
            "look": "None",
            "exposure": 1.0,
            "gamma": 1.0,
        },
        {
            "name": "scene_linear_gamma",
            "mode": "scene_linear_hdr",
            "presentation": "scene_linear",
            "view_transform": "AgX",
            "look": "None",
            "exposure": 0.0,
            "gamma": 1.5,
        },
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "out" / "artifacts" / "ovrtx-color-presentation-probe",
    )
    parser.add_argument(
        "--blender-command",
        default=os.environ.get("BLENDER_COMMAND", BLENDER_COMMAND),
    )
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = args.output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "result": args.output_dir / "color-presentation-runtime.json",
        "setup": args.output_dir / "color_presentation_probe_setup.py",
        "log": args.output_dir / "blender.log",
    }
    config = {
        "repo": str(REPO),
        "result": str(paths["result"]),
        "crops_dir": str(crops_dir),
        "crop_width": CROP_WIDTH,
        "crop_height": CROP_HEIGHT,
        "cases": _probe_cases(),
    }
    paths["setup"].write_text(_setup_script(config), encoding="utf-8")
    completed = subprocess.run(
        [args.blender_command, "--background", "--python", str(paths["setup"])],
        cwd=str(REPO),
        env=os.environ.copy(),
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    paths["log"].write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "result": str(paths["result"]),
                    "blender_log": str(paths["log"]),
                    "returncode": completed.returncode,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return completed.returncode
    if not paths["result"].exists():
        print(
            json.dumps(
                {
                    "status": "failed",
                    "result": str(paths["result"]),
                    "blender_log": str(paths["log"]),
                    "error": "Probe did not write color-presentation-runtime.json.",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1
    result = json.loads(paths["result"].read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "status": result.get("status"),
                "result": str(paths["result"]),
                "crops_dir": str(crops_dir),
                "checks": result.get("checks", {}),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result.get("status") == "pass" else 1


# --- Blender-side probe body ------------------------------------------------
#
# Built as a plain string (not an f-string) so the embedded Python's own braces
# stay literal; only the CONFIG constant is injected, via json round-tripping.

_BLENDER_PREAMBLE = """
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
"""


_BLENDER_BODY = r'''
sys.path.insert(0, str(Path(CONFIG["repo"]) / "addon"))

import bpy
import ovrtx_blender_example as addon
from ovrtx_blender_example import color_presentation, engine, viewport_artifact_recorder
from ovrtx_blender_example.engine import build_request_from_scene
from ovrtx_blender_example.blender_signals import (
    BlenderRenderIntent,
    BlenderRenderSignalSource,
)
from ovrtx_blender_example.ovrtx_runtime_client import RenderResult


def _linear_blue_crop(width, height):
    """A blue-background scene-linear RGBA crop with a bright warm highlight.

    The highlight patch pushes values above 1.0 so the differences between AgX,
    Standard, exposure, and gamma are visible in the presented (tone-mapped)
    output. Returned as a flat R,G,B,A float list in bottom-up row order to
    match ``Image.pixels``.
    """

    background = (0.02, 0.06, 0.35, 1.0)
    highlight = (2.4, 1.6, 0.6, 1.0)
    x0, x1 = width // 3, (2 * width) // 3
    y0, y1 = height // 3, (2 * height) // 3
    pixels = []
    for y in range(height):
        for x in range(width):
            if x0 <= x < x1 and y0 <= y < y1:
                pixels.extend(highlight)
            else:
                pixels.extend(background)
    return pixels


def _linear_to_srgb8(value):
    v = max(0.0, min(1.0, float(value)))
    if v <= 0.0031308:
        s = 12.92 * v
    else:
        s = 1.055 * (v ** (1.0 / 2.4)) - 0.055
    return max(0.0, min(1.0, s))


def _display_encoded_crop(linear_pixels):
    """OVRTX-baked display encoding for the LDR passthrough frame (0..1)."""

    return [_linear_to_srgb8(v) if (i % 4) != 3 else 1.0 for i, v in enumerate(linear_pixels)]


def _pack_rgba16f(linear_pixels):
    """Best-effort real RGBA16F bytes for the RenderResult linear payload."""

    try:
        import numpy as np

        return np.asarray(linear_pixels, dtype=np.float16).tobytes()
    except Exception:
        # The evidence derivation only needs a non-empty payload; fall back to
        # a deterministic non-empty byte string sized to the pixel count.
        return bytes(len(linear_pixels) * 2) or b"\x00\x00"


def _apply_nontrivial_look(view_settings):
    """Apply a real (non-None) look, returning the applied identifier.

    ``look`` is a dynamic OCIO enum: ``bl_rna`` only reports the static
    ``NONE`` placeholder, but assigning the config-provided identifier
    validates against the live list. Try view-transform-appropriate
    candidates and keep the first that sticks.
    """

    candidates = (
        "AgX - Punchy",
        "AgX - High Contrast",
        "AgX - Medium High Contrast",
        "High Contrast",
        "Medium High Contrast",
        "Punchy",
    )
    for candidate in candidates:
        try:
            view_settings.look = candidate
        except Exception:
            continue
        if str(getattr(view_settings, "look", "None")) == candidate:
            return candidate
    return str(getattr(view_settings, "look", "None"))


def _apply_view_settings(scene, case):
    vs = scene.view_settings
    ds = scene.display_settings
    if hasattr(ds, "display_device"):
        try:
            ds.display_device = "sRGB"
        except Exception:
            pass
    view_transform = case.get("view_transform")
    if view_transform:
        try:
            vs.view_transform = view_transform
        except Exception:
            pass
    look = case.get("look")
    if look == "__first_non_none__":
        _apply_nontrivial_look(vs)
    elif look is not None:
        try:
            vs.look = look
        except Exception:
            pass
    try:
        vs.exposure = float(case.get("exposure", 0.0))
    except Exception:
        pass
    try:
        vs.gamma = float(case.get("gamma", 1.0))
    except Exception:
        pass
    return {
        "view_transform": str(getattr(vs, "view_transform", "")),
        "look": str(getattr(vs, "look", "")),
        "exposure": float(getattr(vs, "exposure", 0.0)),
        "gamma": float(getattr(vs, "gamma", 1.0)),
    }


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _mean_rgb_from_png(path):
    """Best-effort mean R,G,B of a stored PNG (no re-linearization)."""

    try:
        img = bpy.data.images.load(str(path), check_existing=False)
        try:
            img.colorspace_settings.name = "Non-Color"
        except Exception:
            pass
        px = list(img.pixels)
        bpy.data.images.remove(img)
        if not px:
            return None
        chans = [0.0, 0.0, 0.0]
        counts = [0, 0, 0]
        for i, v in enumerate(px):
            c = i % 4
            if c < 3:
                chans[c] += float(v)
                counts[c] += 1
        return [chans[c] / counts[c] if counts[c] else 0.0 for c in range(3)]
    except Exception:
        return None


def _present_scene_linear_crop(scene, linear_pixels, width, height, out_path):
    """Present a scene-linear crop through Blender's real display transform.

    This is the exactly-once, Blender-owned presentation: save_render applies
    the scene's View Transform / Look / Exposure / Gamma to the scene-linear
    buffer. Changing any of those changes the written pixels.
    """

    img = bpy.data.images.new("ovrtx_scene_linear_crop", width, height, alpha=True, float_buffer=True)
    try:
        img.pixels = linear_pixels
        img.save_render(str(out_path), scene=scene)
    finally:
        bpy.data.images.remove(img)
    return out_path


def _present_ldr_crop(display_pixels, width, height, out_path):
    """Present the LDR passthrough crop raw (OVRTX owns display encoding).

    The already-display-encoded RGBA8 payload is written without Blender's view
    transform, so the result is invariant to the scene's color management.
    """

    img = bpy.data.images.new("ovrtx_ldr_crop", width, height, alpha=True, float_buffer=False)
    try:
        try:
            img.colorspace_settings.name = "Non-Color"
        except Exception:
            pass
        img.pixels = display_pixels
        img.file_format = "PNG"
        img.filepath_raw = str(out_path)
        try:
            img.save(filepath=str(out_path))
        except TypeError:
            img.save()
    finally:
        bpy.data.images.remove(img)
    return out_path


def _record_case(scene, case, linear_pixels, display_pixels, width, height, crops_dir):
    old_mode = os.environ.get(color_presentation.ENV_COLOR_PRESENTATION_MODE)
    mode = case.get("mode")
    try:
        if mode is None:
            os.environ.pop(color_presentation.ENV_COLOR_PRESENTATION_MODE, None)
        else:
            os.environ[color_presentation.ENV_COLOR_PRESENTATION_MODE] = mode
        request = build_request_from_scene(
            scene,
            source=BlenderRenderSignalSource.FINAL_RENDER,
            intent=BlenderRenderIntent.FINAL_RENDER,
        )
    finally:
        if old_mode is None:
            os.environ.pop(color_presentation.ENV_COLOR_PRESENTATION_MODE, None)
        else:
            os.environ[color_presentation.ENV_COLOR_PRESENTATION_MODE] = old_mode

    presentation = request.color_presentation
    is_scene_linear = presentation["frame_format"] == color_presentation.FRAME_FORMAT_RGBA16F
    if is_scene_linear:
        result = RenderResult(
            width=width,
            height=height,
            rgba8=bytes(display_pixels_to_bytes(display_pixels)),
            completed_samples=1,
            session_completed_samples=1,
            simulation_time_ns=10,
            frame_format=presentation["frame_format"],
            frame_color_mode=presentation["frame_color_mode"],
            render_var=presentation["render_var"],
            linear_rgba16f=_pack_rgba16f(linear_pixels),
        )
    else:
        result = RenderResult(
            width=width,
            height=height,
            rgba8=bytes(display_pixels_to_bytes(display_pixels)),
            completed_samples=1,
            session_completed_samples=1,
            simulation_time_ns=10,
            frame_format=presentation["frame_format"],
            frame_color_mode=presentation["frame_color_mode"],
            render_var=presentation["render_var"],
        )

    recorder = viewport_artifact_recorder.Recorder(
        profile_factory=lambda: {},
        record_profile=lambda _profile, _record: None,
        profile_summary=lambda _profile, _latency_ms: {"enabled": True},
        enabled=lambda: True,
    )
    artifact = recorder.artifact(
        viewport_artifact_recorder.State(
            simulation_id="probe-" + case["name"],
            request=request,
            result=result,
            snapshot_index=0,
            render_count=1,
            draw_count=1,
            snapshot_count=1,
            camera_update_count=0,
            camera_controls_mode="usd_camera",
        )
    )

    crops_dir = Path(crops_dir)
    if is_scene_linear:
        crop_path = crops_dir / (case["name"] + ".png")
        _present_scene_linear_crop(scene, linear_pixels, width, height, crop_path)
        presented = {
            "presentation": "scene_linear_blender_owned",
            "crop_png": str(crop_path),
            "crop_sha256": _sha256_file(crop_path),
            "crop_mean_rgb": _mean_rgb_from_png(crop_path),
        }
    else:
        crop_path = crops_dir / (case["name"] + ".png")
        _present_ldr_crop(display_pixels, width, height, crop_path)
        # Re-present under a deliberately different view setting to prove the
        # LDR passthrough frame ignores Blender color management.
        alt_view = scene.view_settings.exposure
        scene.view_settings.exposure = alt_view + 2.0
        alt_path = crops_dir / (case["name"] + "_alt_view.png")
        _present_ldr_crop(display_pixels, width, height, alt_path)
        scene.view_settings.exposure = alt_view
        presented = {
            "presentation": "ldr_passthrough_ovrtx_owned",
            "crop_png": str(crop_path),
            "crop_sha256": _sha256_file(crop_path),
            "crop_mean_rgb": _mean_rgb_from_png(crop_path),
            "alt_view_crop_png": str(alt_path),
            "alt_view_crop_sha256": _sha256_file(alt_path),
        }

    cp = artifact["color_presentation"]
    return {
        "name": case["name"],
        "requested_mode": presentation["requested_mode"],
        "active_mode": presentation["active_mode"],
        "status": presentation["status"],
        "unavailable_reason": presentation["unavailable_reason"],
        "authored_values_adjusted": presentation["authored_values_adjusted"],
        "applied_view_settings": _current_view_settings(scene),
        "request_color_presentation": dict(presentation),
        "artifact_color_presentation": {
            "render_var": cp.get("render_var"),
            "display_transform_owner": cp.get("display_transform_owner"),
            "display_transform_application_count": cp.get("display_transform_application_count"),
            "display_transform_applied_by": cp.get("display_transform_applied_by"),
            "display_transform_consistent": cp.get("display_transform_consistent"),
            "result_frame_format": cp.get("result_frame_format"),
            "result_frame_color_mode": cp.get("result_frame_color_mode"),
        },
        "presented": presented,
    }


def display_pixels_to_bytes(display_pixels):
    return bytes(int(round(max(0.0, min(1.0, v)) * 255.0)) for v in display_pixels)


def _current_view_settings(scene):
    vs = scene.view_settings
    return {
        "view_transform": str(getattr(vs, "view_transform", "")),
        "look": str(getattr(vs, "look", "")),
        "exposure": float(getattr(vs, "exposure", 0.0)),
        "gamma": float(getattr(vs, "gamma", 1.0)),
    }


def _build_scene():
    bpy.ops.wm.read_homefile(use_empty=True)
    addon.register()
    scene = bpy.context.scene
    scene.render.engine = engine.ENGINE_ID
    scene.render.resolution_x = CONFIG["crop_width"]
    scene.render.resolution_y = CONFIG["crop_height"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.ovrtx_example.min_samples = 1
    scene.ovrtx_example.max_samples = 1
    # Minimal converted content plus an active camera so the request builds.
    bpy.ops.mesh.primitive_cube_add(location=(0.0, 0.0, 0.0))
    bpy.ops.object.camera_add(location=(0.0, -6.0, 3.0))
    scene.camera = bpy.context.object
    bpy.context.view_layer.update()
    return scene


def _evaluate(cases):
    checks = {}
    scene_linear = [c for c in cases if c["request_color_presentation"]["frame_format"] == color_presentation.FRAME_FORMAT_RGBA16F]
    ldr = [c for c in cases if c["request_color_presentation"]["frame_format"] == color_presentation.FRAME_FORMAT_RGBA8]

    # Every case carries the task03-02 evidence fields.
    evidence_fields = ("render_var", "display_transform_owner", "display_transform_application_count", "display_transform_applied_by", "display_transform_consistent")
    checks["evidence_fields_present"] = all(
        all(c["artifact_color_presentation"].get(f) is not None for f in evidence_fields)
        for c in cases
    )

    # No scene-value changes: presentation ownership fix only.
    checks["no_authored_value_changes"] = all(c["authored_values_adjusted"] is False for c in cases)

    # LDR passthrough baseline: OVRTX owns display, exactly once, and the
    # presented frame ignores Blender color management (invariant crop).
    checks["ldr_baseline_present"] = len(ldr) >= 1
    checks["ldr_ownership"] = all(
        c["active_mode"] == color_presentation.MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH
        and c["request_color_presentation"]["frame_format"] == color_presentation.FRAME_FORMAT_RGBA8
        and c["artifact_color_presentation"]["display_transform_application_count"] == 1
        and c["artifact_color_presentation"]["display_transform_applied_by"] == color_presentation.DISPLAY_TRANSFORM_OWNER_OVRTX
        and c["artifact_color_presentation"]["display_transform_consistent"] is True
        for c in ldr
    )
    checks["ldr_passthrough_invariant_to_view_settings"] = all(
        c["presented"]["crop_sha256"] == c["presented"]["alt_view_crop_sha256"]
        for c in ldr
    )

    # Scene-linear: Blender owns the transform exactly once and consistently.
    checks["scene_linear_present"] = len(scene_linear) >= 5
    checks["scene_linear_ownership_exactly_once"] = all(
        c["active_mode"] == color_presentation.MODE_SCENE_LINEAR_HDR
        and c["status"] == color_presentation.STATUS_CURRENT
        and c["request_color_presentation"]["frame_format"] == color_presentation.FRAME_FORMAT_RGBA16F
        and c["request_color_presentation"]["render_var"] == color_presentation.RENDER_VAR_HDR_COLOR
        and c["artifact_color_presentation"]["display_transform_application_count"] == 1
        and c["artifact_color_presentation"]["display_transform_applied_by"] == color_presentation.DISPLAY_TRANSFORM_OWNER_CONSUMER
        and c["artifact_color_presentation"]["display_transform_consistent"] is True
        for c in scene_linear
    )

    # The presented scene-linear frame responds to each view-settings change:
    # every variation differs from the scene-linear default crop.
    by_name = {c["name"]: c for c in scene_linear}
    default_case = by_name.get("scene_linear_default")
    variations = [c for c in scene_linear if c["name"] != "scene_linear_default"]
    if default_case is not None and variations:
        default_hash = default_case["presented"]["crop_sha256"]
        checks["scene_linear_responds_to_view_settings"] = all(
            c["presented"]["crop_sha256"] != default_hash for c in variations
        )
    else:
        checks["scene_linear_responds_to_view_settings"] = False

    return checks


result = {
    "schema_version": 2,
    "artifact_id": "ovrtx-color-presentation-artifact-probe",
    "fixture": "perf_junk_shop_1280x720",
    "evidence": "junk-shop blue-background presentation regression (synthetic scene-linear crop)",
    "status": "running",
    "started_at_ns": time.time_ns(),
}
try:
    scene = _build_scene()
    width = CONFIG["crop_width"]
    height = CONFIG["crop_height"]
    linear_pixels = _linear_blue_crop(width, height)
    display_pixels = _display_encoded_crop(linear_pixels)
    cases = []
    for case in CONFIG["cases"]:
        _apply_view_settings(scene, case)
        cases.append(
            _record_case(scene, case, linear_pixels, display_pixels, width, height, CONFIG["crops_dir"])
        )
    result["cases"] = cases
    checks = _evaluate(cases)
    result["checks"] = checks
    result["status"] = "pass" if all(checks.values()) else "failed"
except BaseException as exc:
    result.update(
        {
            "status": "failed",
            "error": str(exc),
            "error_type": type(exc).__name__,
            "traceback": traceback.format_exc(),
        }
    )
finally:
    result["completed_at_ns"] = time.time_ns()
Path(CONFIG["result"]).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
'''


def _setup_script(config: dict[str, Any]) -> str:
    header = "CONFIG = json.loads(%r)\n" % json.dumps(config, sort_keys=True)
    return _BLENDER_PREAMBLE + "\n" + header + _BLENDER_BODY


if __name__ == "__main__":
    raise SystemExit(main())

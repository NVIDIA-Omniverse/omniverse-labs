#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isolated Blender GPU boundary test; does not load/save/edit any scene.

Run in a fresh caller-owned GUI process (GPU APIs require a graphics context):
  blender --factory-startup --python probe_frame_buffer_gpu.py -- --output /tmp/probe.json

The asymmetric pixels are a synthetic immutable Python fixture, not renderer
output. Native exporter lifetime is tested separately with a real worker. This
probe exercises the production adapter, staging, upload/update, draw rectangle,
and GPU readback. The caller MUST inspect the JSON status, not just exit code.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))
TOP = bytes((255, 0, 0, 255, 0, 255, 0, 255))
BOTTOM = bytes((0, 0, 255, 255, 255, 255, 0, 255))
PIXELS = TOP + BOTTOM


def run_probe():
    import bpy
    import gpu
    from gpu_extras.presets import draw_texture_2d
    from mathutils import Matrix
    from ovrtx_blender_example import engine, ovrtx_runtime_client as adapter

    def decode(*_args):
        return {"frames": {"/Color": {"width": 2, "height": 2, "rgba8": memoryview(PIXELS)}}}

    frame = adapter._frame_buffer_decoder(decode, bytes)(None, None)["frames"]["/Color"]
    candidate = adapter.render_result_from_native(frame, 2, 2)
    control = replace(candidate, rgba8=BOTTOM + TOP, rgba8_origin="bottom_left")
    report = {
        "schema": "ovrtx.phase09.gpu-boundary.v1",
        "blender": bpy.app.version_string,
        "fixture": "synthetic asymmetric 2x2 immutable Python bytes; not renderer output",
        "renderer": gpu.platform.renderer_get(),
        "expected_bottom_up_rgba8": list(BOTTOM + TOP),
        "source_sha256": hashlib.sha256(PIXELS).hexdigest(),
        "cases": [],
    }
    original_update = os.environ.get("OVRTX_VIEWPORT_TEXTURE_UPDATE")
    offscreen = gpu.types.GPUOffScreen(2, 2)
    try:
        with offscreen.bind():
            gpu.state.viewport_set(0, 0, 2, 2)
            gpu.state.depth_test_set("NONE")
            gpu.state.blend_set("NONE")
            gpu.state.face_culling_set("NONE")
            for route in ("new", "update", "float"):
                os.environ["OVRTX_VIEWPORT_TEXTURE_UPDATE"] = "0" if route == "float" else "1"
                for name, result in (("control", control), ("candidate", candidate)):
                    cached = None
                    if route == "update":
                        blank = replace(control, rgba8=bytes(16))
                        initial = engine._upload_viewport_texture(
                            gpu, blank, cached_texture=None, cached_texture_size=None,
                            cached_texture_snapshot_index=-1, snapshot_index=0, accepts_rgba8=True,
                        )
                        cached = initial.texture
                    uploaded = engine._upload_viewport_texture(
                        gpu, result, cached_texture=cached,
                        cached_texture_size=(2, 2, result.frame_format) if cached else None,
                        cached_texture_snapshot_index=0, snapshot_index=1,
                        accepts_rgba8=(route != "float"),
                    )
                    if route == "update" and uploaded.diagnostics["texture_path"] != "update":
                        raise AssertionError(f"GPUTexture.update route unavailable: {uploaded.diagnostics}")
                    if route == "new":
                        # Some supported Blender builds accept only FLOAT in
                        # GPUTexture's constructor while supporting UBYTE for
                        # later updates. Exercise and report the production
                        # startup fallback; do not claim UBYTE creation passed.
                        texture_path = uploaded.diagnostics["texture_path"]
                        if texture_path == "fallback_float":
                            assert uploaded.diagnostics.get("texture_rgba8_fast_path_failed") is True
                            assert uploaded.diagnostics.get("texture_rgba8_fast_path_error")
                        elif texture_path != "new_texture":
                            raise AssertionError(f"unexpected startup upload route: {uploaded.diagnostics}")
                    if route == "float" and uploaded.diagnostics["texture_path"] != "fallback_float":
                        raise AssertionError(f"float fallback not exercised: {uploaded.diagnostics}")
                    framebuffer = gpu.state.active_framebuffer_get()
                    framebuffer.clear(color=(0, 0, 0, 0))
                    position, width, height = engine._viewport_texture_draw_rect(result, 0, 0, 2, 2)
                    with gpu.matrix.push_pop(), gpu.matrix.push_pop_projection():
                        gpu.matrix.load_matrix(Matrix.Identity(4))
                        gpu.matrix.load_projection_matrix(engine._ortho_2d(2, 2))
                        draw_texture_2d(uploaded.texture, position, width, height)
                    readback = framebuffer.read_color(0, 0, 2, 2, 4, 0, "UBYTE")
                    actual = bytes(memoryview(readback).cast("B"))
                    assert actual == BOTTOM + TOP, (name, route, list(actual))
                    assert result.rgba8_bytes(origin="top_left") == PIXELS
                    report["cases"].append({
                        "variant": name, "route": route,
                        "ubyte_constructor_available": (
                            uploaded.diagnostics["texture_path"] == "new_texture"
                            if route == "new" else None
                        ),
                        "readback_bottom_up_rgba8": list(actual),
                        "diagnostics": uploaded.diagnostics,
                        "draw": {"position": position, "width": width, "height": height},
                    })

            diagnostics = {"texture_buffer_copy_bytes": 0}
            staging_buffer = engine._rgba8_upload_buffer(gpu, candidate, diagnostics)
            staging_buffer[0] = 13
            writable = memoryview(staging_buffer)
            writable.cast("B")[1] = 17
            assert bytes(candidate.rgba8) == PIXELS
            assert diagnostics["texture_buffer_copy_bytes"] == len(PIXELS)
            report["staging_mutation"] = {
                "native_fixture_unchanged": True,
                "staging_writable": not writable.readonly,
                "copy_bytes": diagnostics["texture_buffer_copy_bytes"],
            }
            writable.release()
        report["status"] = "pass"
        return report
    finally:
        offscreen.free()
        if original_update is None:
            os.environ.pop("OVRTX_VIEWPORT_TEXTURE_UPDATE", None)
        else:
            os.environ["OVRTX_VIEWPORT_TEXTURE_UPDATE"] = original_update


def main():
    import bpy
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def callback():
        try:
            report = run_probe()
        except Exception:
            report = {"status": "failed", "error": traceback.format_exc()}
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        bpy.ops.wm.quit_blender()
        return None

    bpy.app.timers.register(callback, first_interval=0.1)


if __name__ == "__main__":
    main()

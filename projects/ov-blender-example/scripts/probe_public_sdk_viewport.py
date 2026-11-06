#!/usr/bin/env python3
"""Draw the spike scene in a real Blender window and retain presentation proof."""
import json
import hashlib
import os
from pathlib import Path
import sys
import time

import bpy

source = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(source / 'addon'))
import ovrtx_blender_example as addon
from ovrtx_blender_example import engine

output = Path(os.environ['OV_BLENDER_PUBLIC_SDK_OUTPUT']) / 'viewport-smoke'
output.mkdir(parents=True, exist_ok=True)
addon.register()
bpy.context.scene.ovrtx_example.max_samples = 16
for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        area.spaces.active.region_3d.view_perspective = 'CAMERA'
        area.spaces.active.shading.type = 'RENDERED'
        area.tag_redraw()
started = time.monotonic()
first_presented = None


def check():
    global first_presented
    presented = []
    errors = []
    for instance in engine._live_viewport_engines():
        frame = getattr(instance, '_presented_frame', None)
        error = getattr(instance, '_viewport_reported_error', '')
        if error:
            errors.append(error)
        if frame is not None and frame.render_result is not None:
            result = frame.render_result
            payload = result.rgba8_bytes(origin='top_left')
            rgb = [value for i, value in enumerate(payload) if i % 4 != 3]
            presented.append({'width': result.width, 'height': result.height,
                              'samples': result.completed_samples,
                              'publication': frame.publication_index,
                              'render_var': result.render_var,
                              'pixel_range': [min(rgb), max(rgb)],
                              'sha256': hashlib.sha256(payload).hexdigest()})
            if max(rgb) - min(rgb) < 3:
                errors.append('Presented frame pixels are blank or uniform')
    elapsed = time.monotonic() - started
    if elapsed < 180 and not errors and not presented:
        return 0.5
    if presented and first_presented is None:
        first_presented = elapsed
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=2)
        return 2.0
    report = {'ok': bool(presented) and not errors, 'seconds': elapsed,
              'presented': presented, 'errors': errors,
              'sdk_imports_in_blender': [name for name in ('ovrtx', 'ovphysx', 'ovstage', 'ovstream', 'warp') if name in sys.modules]}
    if presented:
        # Retain the actual presented pixels independently of window capture.
        frame_image = bpy.data.images.new('Public SDK presented frame', width=result.width, height=result.height)
        frame_image.pixels.foreach_set([value / 255.0 for value in result.rgba8_bytes()])
        frame_image.filepath_raw = str(output / 'presented-frame.png')
        frame_image.file_format = 'PNG'
        frame_image.save()
        bpy.data.images.remove(frame_image)
        bpy.ops.screen.screenshot(filepath=str(output / 'viewport.png'))
        screenshot = bpy.data.images.load(str(output / 'viewport.png'), check_existing=False)
        values = list(screenshot.pixels)[::4]
        report['window_capture_nonblank'] = max(values) - min(values) > 0.01
        bpy.data.images.remove(screenshot)
    (output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PUBLIC_SDK_VIEWPORT', json.dumps(report), flush=True)
    bpy.ops.wm.quit_blender()
    return None


bpy.app.timers.register(check, first_interval=1)

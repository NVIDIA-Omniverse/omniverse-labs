# SPDX-License-Identifier: Apache-2.0
"""Synchronous native HDR stills on the durable camera capture owner."""
from dataclasses import replace
from pathlib import Path
import json
import threading
import time

from . import render_sequence

PUBLIC_SDK_DURABLE_HDR_API = 1
_captures = {}
_images = {}

try:
    import bpy
except ModuleNotFoundError:
    bpy = None


def prepare_display(scene):
    """Keep native HDR scene-linear; preserve authored transform/white balance."""
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.view_settings.use_curve_mapping = False


def last_capture(scene):
    return _captures.get(int(scene.session_uid))


def last_image(scene):
    image = _images.get(int(scene.session_uid))
    try:
        return image if image is not None and image.name else None
    except ReferenceError:
        return None


def save_outputs(image, scene, png):
    """Save display PNG and scene-linear EXR, restoring format even on failure."""
    settings = scene.render.image_settings
    old = settings.file_format, settings.color_depth
    png = Path(png)
    png.parent.mkdir(parents=True, exist_ok=True)
    try:
        settings.file_format = 'PNG'
        image.save_render(str(png), scene=scene)
        settings.file_format = 'OPEN_EXR'
        settings.color_depth = '16'
        exr = png.with_suffix('.exr')
        image.save_render(str(exr), scene=scene)
    finally:
        settings.file_format, settings.color_depth = old
    if not png.is_file() or not exr.is_file() or not png.stat().st_size or not exr.stat().st_size:
        raise RuntimeError('Native HDR still did not produce both output files')
    return str(png), str(exr)


def render_hdr_still(blender, scene):
    """Return a completed display PNG path; never use F12's transient image."""
    from .render_capabilities import require_candidate_capture
    require_candidate_capture('Native HDR still convenience capture')
    from . import engine
    if scene.render.engine != 'OVRTX_EXAMPLE':
        raise ValueError('Select OVRTX first')
    if scene.ovrtx_example.color_presentation_mode != 'scene_linear_hdr':
        raise ValueError('Select scene-linear HDR first')
    if scene.render.image_settings.file_format != 'PNG':
        raise ValueError('HDR still action requires PNG output; an EXR is also saved')
    if scene.camera is None:
        raise ValueError('Select a scene camera')
    if render_sequence.active_for_scene(scene) is not None or blender.app.is_job_running('RENDER'):
        raise RuntimeError('Another capture owns this scene')
    if any(area.type == 'VIEW_3D' and area.spaces.active.shading.type == 'RENDERED'
           for window in blender.context.window_manager.windows for area in window.screen.areas):
        raise RuntimeError('Switch rendered viewports to Solid before HDR still capture')
    _captures.pop(int(scene.session_uid), None)
    started = time.perf_counter()
    prepare_display(scene)
    blender.context.view_layer.update()
    request = engine._render_callback_adapter(
        f'camera-sequence-{scene.session_uid}').final_render_from_scene(scene)
    request = replace(request, color_presentation=engine._final_render_color_presentation_from_scene(scene))
    labels = None
    if getattr(request, 'semantic_output_path', ''):
        from .semantic_identity import scene_labels
        labels = scene_labels(scene)
    owner = render_sequence.capture_owner_for_scene(scene)
    try:
        result, runtime = owner.executor.submit(owner.capture, request, threading.Event()).result()
        return _finish_capture(blender, scene, request, result, runtime, labels, started)
    except Exception:
        render_sequence.retire_capture_owner(scene)
        raise


def _finish_capture(blender, scene, request, result, runtime, labels, started):
    from . import engine
    if len(result.linear_rgba16f) != result.width * result.height * 8:
        raise RuntimeError('Native HDR still requires a complete scene-linear RGBA16F plane')
    png = scene.render.frame_path(frame=scene.frame_current)
    semantic = None
    if getattr(request, 'semantic_output_path', ''):
        from .semantic_output import save_capture
        semantic = save_capture(Path(request.semantic_output_path), result,
            labels=labels, camera=request.camera_prim_path, frame=scene.frame_current)
        if not semantic.get('verified'):
            raise RuntimeError('Native semantic capture was not verified')
    rgba = engine._rgba16f_to_float_array(result.linear_rgba16f_bytes())
    image = blender.data.images.new('OVRTX HDR Still', width=result.width,
        height=result.height, alpha=True, float_buffer=True)
    try:
        image.pixels.foreach_set(rgba)
        png, exr = save_outputs(image, scene, png)
        elapsed = round((time.perf_counter()-started)*1000, 3)
        receipt = render_sequence.write_receipt(png, frame=scene.frame_current,
            camera=scene.camera.name, camera_prim_path=request.camera_prim_path,
            result=result, runtime=runtime, elapsed_ms=elapsed, semantic=semantic,
            native_exposure=getattr(request, 'native_exposure', None))
    except Exception:
        blender.data.images.remove(image)
        raise
    previous = last_image(scene)
    if previous is not None:
        blender.data.images.remove(previous)
    _images[int(scene.session_uid)] = image
    _captures[int(scene.session_uid)] = dict(receipt, display_png=png, linear_exr=exr)
    scene['ovrtx_last_hdr_frame_time_ms'] = elapsed
    print('OVRTX_HDR_STAGE_TIMING', json.dumps({'hdr_capture_total_ms':elapsed,
        'display_png_bytes':Path(png).stat().st_size, 'linear_exr_bytes':Path(exr).stat().st_size}), flush=True)
    return png


def start_hdr_still(blender, scene):
    from .render_capabilities import require_candidate_capture
    require_candidate_capture('Native HDR still convenience capture')
    if scene.render.engine != 'OVRTX_EXAMPLE' or scene.ovrtx_example.color_presentation_mode != 'scene_linear_hdr':
        raise ValueError('Select OVRTX scene-linear HDR first')
    if scene.render.image_settings.file_format != 'PNG' or scene.camera is None:
        raise ValueError('Select PNG output and a scene camera')
    if render_sequence.active_for_scene(scene) is not None or blender.app.is_job_running('RENDER'):
        raise RuntimeError('Another capture owns this scene')
    if any(area.type == 'VIEW_3D' and area.spaces.active.shading.type == 'RENDERED'
           for window in blender.context.window_manager.windows for area in window.screen.areas):
        raise RuntimeError('Switch rendered viewports to Solid before HDR still capture')
    prepare_display(scene)
    _captures.pop(int(scene.session_uid), None)
    owner = render_sequence.capture_owner_for_scene(scene)
    sequence = render_sequence.Sequence(scene, [(scene.frame_current, scene.camera)], owner,
        [], threading.Event(), hdr_still=True)
    render_sequence.start_sequence(sequence)
    return sequence


if bpy is not None and hasattr(getattr(bpy, 'types', None), 'Operator'):
    class OVRTXEXAMPLE_OT_render_hdr_still(bpy.types.Operator):
        bl_idname = 'ovrtx_example.render_hdr_still'
        bl_label = 'Render Native HDR Still'

        @classmethod
        def poll(cls, context):
            return context.scene is not None and context.scene.render.engine == 'OVRTX_EXAMPLE' and render_sequence.active_for_scene(context.scene) is None and not bpy.app.is_job_running('RENDER')

        def execute(self, context):
            try:
                start_hdr_still(bpy, context.scene)
            except Exception as exc:
                self.report({'ERROR'}, str(exc))
                return {'CANCELLED'}
            return {'RUNNING_MODAL'}


def register():
    # Standard F12 HDR/EXR output remains registered through the render engine.
    pass


def unregister():
    _captures.clear()
    _images.clear()
    if bpy is not None and 'OVRTXEXAMPLE_OT_render_hdr_still' in globals():
        if getattr(OVRTXEXAMPLE_OT_render_hdr_still, 'is_registered', False):
            bpy.utils.unregister_class(OVRTXEXAMPLE_OT_render_hdr_still)

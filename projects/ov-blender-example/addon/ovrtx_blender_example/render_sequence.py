# SPDX-License-Identifier: Apache-2.0
"""Camera-marker captures with one durable render owner and hash-bound receipts.

Frame preparation and image saving run on Blender's main thread. Native SDK
operations run serially on a persistent owner thread; no SIL transport is added.
Requested auxiliary outputs must complete before the RGB frame is promoted.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import threading
import time

try:
    import bpy
except ModuleNotFoundError:
    bpy = None

_jobs = {}
_owners = {}
_last_result = None


def active_for_scene(scene):
    return _jobs.get(int(scene.session_uid))


def capture_owner_for_scene(scene):
    from .render_capabilities import require_candidate_capture
    require_candidate_capture('Camera-marker timeline capture')
    uid = int(scene.session_uid)
    if uid not in _owners:
        _owners[uid] = CaptureOwner(uid)
    if _owners[uid]._release_future is not None:
        raise RuntimeError('Camera capture teardown is draining')
    from . import scene_generation_sessions
    scene_generation_sessions.reserve_camera_capture(scene, _owners[uid])
    return _owners[uid]


def retire_capture_owner(scene):
    owner = _owners.pop(int(scene.session_uid), None)
    if owner is not None:
        owner.executor.submit(owner.close)
        owner.executor.shutdown(wait=True)


def last_result():
    return _last_result


def camera_frames(markers, *, selected_only=False):
    rows = [(int(marker.frame), marker.camera) for marker in markers
            if marker.camera is not None and (not selected_only or marker.select)]
    rows.sort(key=lambda row: row[0])
    if not rows or len({frame for frame, _ in rows}) != len(rows):
        raise ValueError('Select at least one camera marker and exactly one camera per frame')
    return rows


class CaptureOwner:
    """One RPC lane retained between successive camera/preset sequences."""
    def __init__(self, scene_uid, controller_factory=None):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='ovrtx-camera-sequence')
        self.controller = None
        self.scene_uid = scene_uid
        self.controller_factory = controller_factory
        self._busy = False
        self._release_future = None
        self.generation_activation = None
        self.retained_revision = None

    def capture(self, request, cancel):
        self._busy = True
        try:
            return self._capture(request, cancel)
        finally:
            self._busy = False

    def _capture(self, request, cancel):
        capture_started = time.perf_counter()
        from . import engine
        from .ovrtx_session_controller import OvrtxSessionController
        if self.controller is None:
            factory = self.controller_factory or OvrtxSessionController
            self.controller = factory(simulation_id=f'ovrtx-camera-sequence-{self.scene_uid}')
            self.controller.adopt_owning_thread()
        from . import scene_generation_sessions
        scene_identity = SimpleNamespace(session_uid=self.scene_uid)
        activation = None
        if scene_generation_sessions.owns_request(scene_identity, request):
            # Use the existing generation activation/retained-value/acceptance
            # contract. Keep our controller alive after this capture; unlike
            # the F12 lane, a camera sequence owns it across subsequent jobs.
            activation = scene_generation_sessions.activate_for_camera_capture(
                scene_identity, request, controller=self.controller, capture_owner=self)
            composition = activation.last_ensure_result.composition
        else:
            composition = self.controller.ensure(request).composition
        activation_ms = (time.perf_counter()-capture_started)*1000
        outcome = engine._run_final_render_job(request, cancel,
            controller=self.controller, composition=composition)
        if outcome['status'] != engine.FINAL_RENDER_STATUS_COMPLETED:
            raise RuntimeError('Camera sequence cancelled before completed samples')
        result = outcome['result']
        if result.completed_samples < max(1, int(request.max_samples)):
            raise RuntimeError('Incomplete native camera frame')
        runtime = dict(self.controller.diagnostics())
        generation_owner = scene_generation_sessions._owners.get(self.scene_uid)
        generation = getattr(generation_owner, 'current_generation', None)
        if generation is not None:
            runtime['scene_generation'] = dict(generation.diagnostics)
        if activation is not None:
            runtime['generation_activation_timings'] = dict(getattr(activation, 'activation_timings', {}))
        runtime['sequence_timings'] = {'activation_ms': activation_ms,
            'capture_ms': (time.perf_counter()-capture_started)*1000,
            **outcome.get('timings', {})}
        return result, runtime

    def is_busy(self):
        return self._busy or self.scene_uid in _jobs

    def release_for_viewport(self):
        if self._release_future is None:
            def release():
                self.close()
                if _owners.get(self.scene_uid) is self:
                    _owners.pop(self.scene_uid, None)
                self.executor.shutdown(wait=False)
                if bpy is not None:
                    def redraw():
                        for window in bpy.context.window_manager.windows:
                            for area in window.screen.areas:
                                if area.type == 'VIEW_3D':
                                    area.tag_redraw()
                        return None
                    bpy.app.timers.register(redraw, first_interval=.01)
            self._release_future = self.executor.submit(release)
        if not self._release_future.done():
            return False
        self._release_future.result()
        return True

    def close(self):
        if self.controller is not None:
            status = self.controller.shutdown()
            if status not in ('stopped', 'not_found'):
                raise RuntimeError('Camera capture native teardown was not confirmed')
        from . import scene_generation_sessions
        scene_generation_sessions.release_camera_capture(self.scene_uid, self)
        self.generation_activation = None
        self.retained_revision = None


def write_receipt(output, *, frame, camera, result, runtime, elapsed_ms,
                  camera_prim_path=None, semantic=None, native_exposure=None):
    path = Path(output)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError('Camera frame did not produce an output file')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    receipt = {'frame': int(frame), 'camera': str(camera), 'output': str(path),
        'sha256': digest, 'completed_samples': result.completed_samples,
        'width': result.width, 'height': result.height, 'total_ms': elapsed_ms,
        'runtime': runtime, 'semantic_available': semantic is not None,
        'camera_prim_path': str(camera_prim_path or camera),
        'native_timings': dict(result.native_timings)}
    if native_exposure is not None:
        receipt['native_exposure'] = native_exposure.diagnostics()
    if semantic is not None:
        receipt['semantic_capture'] = semantic
    path.with_suffix('.sequence.json').write_text(json.dumps(receipt, indent=2, default=str))
    if semantic is None:
        path.with_suffix('.capture.json').write_text(json.dumps(
            dict(receipt, camera=receipt['camera_prim_path']), indent=2, default=str))
    print('OVRTX_SEQUENCE_FRAME', json.dumps(receipt, default=str), flush=True)
    return receipt


@dataclass
class Sequence:
    scene: object
    frames: list
    owner: CaptureOwner
    receipts: list
    cancel_event: threading.Event
    written: int = 0
    error: str = ''
    future: object = None
    request: object = None
    started: float = 0.
    output: str = ''
    semantic_labels: object = None
    hdr_still: bool = False
    preparation_ms: float = 0.
    preparation_timings: object = None

    def runtime_diagnostics(self):
        if self.receipts:
            return self.receipts[-1]['runtime']
        return {'active': bool(self.future), 'consumer': 'camera_sequence'}

    def cancel(self):
        self.cancel_event.set()

    def tick(self):
        global _last_result
        uid = int(self.scene.session_uid)
        try:
            if self.future is None:
                if self.cancel_event.is_set():
                    raise RuntimeError('Camera sequence cancelled')
                if self.written == len(self.frames):
                    _last_result = {'status': 'completed', 'frames': self.receipts, 'error': ''}
                    _jobs.pop(uid, None)
                    print('OVRTX_SEQUENCE_COMPLETE', json.dumps(_last_result, default=str), flush=True)
                    return None
                frame, camera = self.frames[self.written]
                preparation_started = time.perf_counter()
                phase_started = time.perf_counter()
                self.preparation_timings = {}
                # Re-evaluate only when the requested frame differs. Lighting
                # and camera changes are admitted together by the update below.
                if (self.scene.frame_current != frame
                        or float(getattr(self.scene, 'frame_subframe', 0.)) != 0.):
                    self.scene.frame_set(frame)
                self.scene.camera = camera
                bpy.context.view_layer.update()
                self.preparation_timings['frame_camera_depsgraph_ms'] = (time.perf_counter()-phase_started)*1000
                from . import engine
                phase_started = time.perf_counter()
                adapter = engine._render_callback_adapter(f'camera-sequence-{uid}')
                self.request = adapter.final_render_from_scene(self.scene)
                self.preparation_timings['callback_request_ms'] = (time.perf_counter()-phase_started)*1000
                self.preparation_timings['callback'] = dict(getattr(adapter, '_preparation_timings', {}))
                translation = getattr(adapter, '_translation_timings_snapshot', None)
                if callable(translation):
                    self.preparation_timings['translation'] = dict(translation())
                phase_started = time.perf_counter()
                self.request = replace(self.request, color_presentation=
                    engine._final_render_color_presentation_from_scene(self.scene))
                if getattr(self.request, 'semantic_output_path', ''):
                    from .semantic_identity import scene_labels
                    self.semantic_labels = scene_labels(self.scene)
                self.preparation_timings['color_and_semantic_ms'] = (time.perf_counter()-phase_started)*1000
                self.started = time.perf_counter()
                self.preparation_ms = (self.started-preparation_started)*1000
                self.output = self.scene.render.frame_path(frame=frame)
                self.future = self.owner.executor.submit(self.owner.capture, self.request, self.cancel_event)
                return .1
            if not self.future.done():
                return .1
            result, runtime = self.future.result()
            runtime.setdefault('sequence_timings', {})['host_prepare_ms'] = self.preparation_ms
            runtime['host_preparation_timings'] = dict(self.preparation_timings or {})
            if self.cancel_event.is_set():
                raise RuntimeError('Camera sequence cancelled')
            frame, camera = self.frames[self.written]
            if self.hdr_still:
                from . import hdr_output
                hdr_output._finish_capture(bpy, self.scene, self.request, result, runtime,
                    self.semantic_labels, self.started)
                self.receipts.append(hdr_output.last_capture(self.scene))
                self.written += 1
                self.future = None
                return .01
            semantic = None
            if getattr(self.request, 'semantic_output_path', ''):
                from .semantic_output import save_capture
                semantic = save_capture(Path(self.output).with_suffix(''), result,
                    labels=self.semantic_labels, camera=self.request.camera_prim_path, frame=frame)
                if not semantic.get('verified'):
                    raise RuntimeError('Native semantic capture was not verified')
            from . import engine
            rgba = (engine._rgba16f_to_float_array(result.linear_rgba16f_bytes())
                    if result.linear_rgba16f else engine._rgba8_to_float_array(result.rgba8_bytes()))
            image = bpy.data.images.new('OVRTX camera sequence frame', width=result.width,
                height=result.height, alpha=True, float_buffer=bool(result.linear_rgba16f))
            try:
                image.pixels.foreach_set(rgba)
                Path(self.output).parent.mkdir(parents=True, exist_ok=True)
                image.save_render(self.output, scene=self.scene)
            finally:
                bpy.data.images.remove(image)
            self.receipts.append(write_receipt(self.output, frame=frame,
                camera=camera.name, camera_prim_path=self.request.camera_prim_path,
                result=result, runtime=runtime, semantic=semantic,
                native_exposure=getattr(self.request, 'native_exposure', None),
                elapsed_ms=round((time.perf_counter()-self.started)*1000, 3)))
            self.written += 1
            self.future = None
            return .01
        except Exception as exc:
            self.error = str(exc)
            _last_result = {'status': 'failed', 'frames': self.receipts, 'error': self.error}
            _jobs.pop(uid, None)
            print('OVRTX_SEQUENCE_COMPLETE', json.dumps(_last_result, default=str), flush=True)
            # Uncertain native failures must not be reused for the next batch.
            _owners.pop(uid, None)
            self.owner.executor.submit(self.owner.close)
            self.owner.executor.shutdown(wait=False)
            return None


def start_sequence(sequence):
    from .render_capabilities import require_candidate_capture
    require_candidate_capture('Camera-marker timeline capture')
    uid = int(sequence.scene.session_uid)
    if _jobs:
        raise RuntimeError('Another camera sequence is active')
    _jobs[uid] = sequence
    bpy.app.timers.register(sequence.tick, first_interval=.01)


def shutdown(*, wait=True):
    for sequence in list(_jobs.values()):
        sequence.cancel()
        if bpy is not None:
            timers = getattr(bpy.app, 'timers', None)
            if timers is not None and timers.is_registered(sequence.tick):
                timers.unregister(sequence.tick)
    for owner in list(_owners.values()):
        owner.executor.submit(owner.close)
        owner.executor.shutdown(wait=wait)
    _owners.clear()
    _jobs.clear()


if bpy is not None and hasattr(getattr(bpy, 'types', None), 'Operator'):
    class OVRTXEXAMPLE_OT_render_timeline(bpy.types.Operator):
        bl_idname = 'ovrtx_example.render_timeline'
        bl_label = 'Render Camera Timeline'
        __annotations__ = {'selected_only': bpy.props.BoolProperty(default=False)}

        @classmethod
        def poll(cls, context):
            return context.scene is not None and context.scene.render.engine == 'OVRTX_EXAMPLE' and not _jobs and not bpy.app.is_job_running('RENDER')

        def execute(self, context):
            scene = context.scene
            try:
                if any(area.type == 'VIEW_3D' and area.spaces.active.shading.type == 'RENDERED'
                       for window in context.window_manager.windows for area in window.screen.areas):
                    raise RuntimeError('Switch rendered viewports to Solid before camera timeline capture')
                frames = camera_frames(scene.timeline_markers, selected_only=self.selected_only)
                uid = int(scene.session_uid)
                owner = capture_owner_for_scene(scene)
                sequence = Sequence(scene, frames, owner, [], threading.Event())
                start_sequence(sequence)
                return {'RUNNING_MODAL'}
            except Exception as exc:
                self.report({'ERROR'}, str(exc))
                return {'CANCELLED'}


def _before_file_change(*_args):
    shutdown()


if bpy is not None and hasattr(getattr(bpy.app, 'handlers', None), 'persistent'):
    _before_file_change = bpy.app.handlers.persistent(_before_file_change)


def register():
    # Candidate-only marker capture is not the supported Blender animation route.
    pass


def unregister():
    shutdown()
    if bpy is not None and 'OVRTXEXAMPLE_OT_render_timeline' in globals():
        for name in ('load_pre', 'exit_pre'):
            handlers = getattr(bpy.app.handlers, name, None)
            if handlers is not None and _before_file_change in handlers:
                handlers.remove(_before_file_change)
        if getattr(OVRTXEXAMPLE_OT_render_timeline, 'is_registered', False):
            bpy.utils.unregister_class(OVRTXEXAMPLE_OT_render_timeline)

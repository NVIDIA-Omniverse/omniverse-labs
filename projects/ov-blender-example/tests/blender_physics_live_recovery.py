# SPDX-License-Identifier: Apache-2.0
"""Continuous scene-owned physics survives renderer death; uncertain physics steps never replay."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time
import threading
import traceback
import uuid

import bpy
import numpy as np

source = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(source / 'tests'))
from physics_live_recovery_checks import assert_live_progress, validate_live, assert_uncertain_not_replayed, assert_fresh_frame, RecentNativeFrames, survival_boundary, assert_renderer_not_reloaded, RENDER_OWNER_FIELDS

assert not os.environ.get('OV_BLENDER_PUBLIC_SDK_PHYSICS_ADMISSION')
os.environ['OV_BLENDER_EXAMPLE_FIXED_VIEWPORT_RESOLUTION'] = '1'
os.environ['OV_BLENDER_EXAMPLE_OVPHYSX_DEVICE'] = 'cpu'
fixture = runpy.run_path(str(source / 'scripts/probe_public_sdk_ui_edits.py'))
from ovrtx_blender_example import engine, physics_capabilities, scene_generation_sessions as sessions
from ovrtx_blender_example.physics_endpoints import initialize_owner
from ovrtx_blender_example.direct_runtime_session import RuntimeSession, RuntimeTransportError
from ovrtx_blender_example.direct_sdk_client import PublicSdkRenderClient

out = Path(os.environ['OV_BLENDER_PUBLIC_SDK_OUTPUT']) / 'recovery'
out.mkdir(parents=True, exist_ok=True)
control_only = os.environ.get('OV_BLENDER_PHYSICS_LIVE_CONTROL') == '1'
trace_lock = threading.Lock()
scene, cube = fixture['scene'], fixture['cube']
scene.render.resolution_x, scene.render.resolution_y = 320, 240
scene.ovrtx_example.color_presentation_mode = 'ldr_rgba8_display_passthrough'
scene.ovrtx_example.max_samples = 64
scene.frame_end = 100000
scene.ov.physics_scene.schema_opinion = 'APPLY'
scene.ov.physics_scene.gravity_magnitude = 0
body = bpy.data.objects.new('Continuity Body', None)
scene.collection.objects.link(body)
body.location = (100, 0, 1)
initialize_owner(body)
physics = body.ov.physics
physics.rigid_body.schema_opinion = 'APPLY'
physics.rigid_body.role = 'DYNAMIC'
physics.rigid_body.mass_kg = 1
physics.rigid_body.initial_linear_velocity = (.25, 0, 0)
physics.rigid_body.linear_damping = 0
physics.rigid_body.angular_damping = 0
collider = physics.colliders.add()
collider.name = collider.persistent_id = uuid.uuid4().hex
collider.schema_opinion, collider.shape, collider.radius_m = 'APPLY', 'ANALYTIC_SPHERE', .2
bpy.context.view_layer.update()
report = dict(status='running', cases=[], physics={}, faults=[], admissions=[], streams=[],
              callback_seconds=[], fixture_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              measurement_only=False, local_manifest_override=None,
              diagnostic_control=control_only, control_wait_seconds=3.,
              control_description='Same authored scene and Play; no edit, renderer fault or restart' if control_only else None)
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='continuity-observer')
future = None
clients = []
physics_owners = []
draws = {}
fault_armed = False
physics_fault_armed = False
original_validate = physics_capabilities.validate_fixed_runtime
original_exchange = RuntimeSession._exchange
original_copy = PublicSdkRenderClient._copy_frame
original_consume = PublicSdkRenderClient._consume_render_result
original_draw = engine.OvrtxExampleRenderEngine._draw_viewport_texture
submission_sequence = copy_sequence = 0
submissions = {}
produced_results = RecentNativeFrames()


def admission(config, import_native_client, public_sdk_identity=None, *, family='joints'):
    result = original_validate(config, import_native_client, public_sdk_identity, family=family)
    report['admissions'].append(result)
    assert result['status'] == 'production' and result['local_override'] is False, result
    assert Path(result['manifest_path']).resolve() == (source / 'addon/ovrtx_blender_example/public_sdk_physics_admission.json').resolve()
    return result


def terminate_producer(runtime):
    identity, owner = runtime.identity, runtime._owner
    assert identity is not None and owner is not None
    assert owner.worker_pid == identity.pid and owner.poll_worker() is None and owner.contains(identity.pid)
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    api.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    api.OpenProcess.restype = wintypes.HANDLE
    api.TerminateProcess.argtypes = (wintypes.HANDLE, wintypes.UINT)
    api.TerminateProcess.restype = wintypes.BOOL
    api.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    api.WaitForSingleObject.restype = wintypes.DWORD
    api.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = api.OpenProcess(0x100001, False, identity.pid)
    if not handle: raise ctypes.WinError(ctypes.get_last_error())
    try:
        assert runtime.identity == identity and owner.contains(identity.pid)
        if not api.TerminateProcess(handle, 241): raise ctypes.WinError(ctypes.get_last_error())
        assert api.WaitForSingleObject(handle, 5000) == 0 and owner.poll_worker() == 241
        return {**asdict(identity), 'producer_pid': identity.pid, 'producer_exit_code': 241}
    finally:
        api.CloseHandle(handle)


def exchange(self, op, params):
    global fault_armed, physics_fault_armed, submission_sequence
    submission = None
    if self.kind == 'renderer' and op in {'render', 'begin_render'}:
        submission_sequence += 1
        submission = dict(sequence=submission_sequence, started=time.monotonic(),
                          operation=op, pid=self.pid, epoch=self.epoch)
    if self.kind == 'physics' and op == 'load':
        # Capture the exact solver input, which may differ from the renderer's
        # metadata wrapper. Read-only USD inspection never imports an SDK.
        from pxr import Usd, UsdGeom
        stage_path = Path(params['scene_path'])
        stage = Usd.Stage.Open(str(stage_path))
        assert stage is not None
        index = len(report.setdefault('physics_stage_inputs', []))
        preserved = out / f'physics-input-{index}.usda'
        stage.Export(str(preserved))
        raw = out / f'physics-input-{index}-root{stage_path.suffix}'
        raw.write_bytes(stage_path.read_bytes())
        layers = []
        for layer_index, layer in enumerate(stage.GetUsedLayers()):
            archived_layer = out / f'physics-input-{index}-layer-{layer_index}.usda'
            layer.Export(str(archived_layer))
            layers.append(dict(identifier=layer.identifier, real_path=layer.realPath,
                               archived_path=archived_layer.name,
                               sha256=hashlib.sha256(archived_layer.read_bytes()).hexdigest()))
        report['physics_stage_inputs'].append(dict(
            source_path=str(stage_path), source_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
            flattened_path=preserved.name, root_path=raw.name,
            stage_metadata={key: dict(authored=stage.HasAuthoredMetadata(key), value=stage.GetMetadata(key))
                            for key in ('metersPerUnit', 'upAxis', 'timeCodesPerSecond', 'framesPerSecond')},
            layers=layers,
            resolved_meters_per_unit=UsdGeom.GetStageMetersPerUnit(stage),
            bodies=[dict(path=path, schemas=list(stage.GetPrimAtPath(path).GetAppliedSchemas()),
                attributes={name: stage.GetPrimAtPath(path).GetAttribute(name).Get()
                    for name in ('physics:velocity', 'physics:angularVelocity', 'physics:mass',
                                 'physxRigidBody:sleepThreshold', 'physxRigidBody:stabilizationThreshold',
                                 'physxRigidBody:linearDamping', 'physxRigidBody:angularDamping')})
                    for path in params['body_prims']]))
    started = time.monotonic()
    result = original_exchange(self, op, params)
    if self.kind == 'physics':
        with trace_lock:
            with (out / 'physics-operations.jsonl').open('a', encoding='utf-8') as log:
                log.write(json.dumps(dict(operation=op, started=started, completed=time.monotonic(),
                    pid=self.pid, epoch=self.epoch, params=params, result=result), default=str) + '\n')
    if submission is not None:
        if op == 'begin_render':
            submissions[(id(self), self.epoch, result['ticket'])] = submission
        else:
            result['_workflow_submission'] = submission
    elif self.kind == 'renderer' and op == 'poll_render' and result.get('shm'):
        result['_workflow_submission'] = submissions[(id(self), self.epoch, params['ticket'])]
    if self.kind == 'renderer' and fault_armed and op == 'update':
        fault_armed = False
        report['faults'].append(dict(kind='native_update_done_reply_not_retained',
                                    values=params['values'], native_updated_count=result.get('updated_count'),
                                    termination=terminate_producer(self)))
        raise RuntimeTransportError('Continuity fixture: renderer died after native update before ACK retention')
    if self.kind == 'physics' and physics_fault_armed and op == 'step':
        physics_fault_armed = False
        report['faults'].append(dict(kind='native_step_done_reply_not_retained',
                                    command=params, native_result=result,
                                    termination=terminate_producer(self)))
        raise RuntimeTransportError('Live recovery fixture: physics died after actual step before ACK retention')
    return result


def copied(self, result, width, height, render_var):
    if self not in clients: clients.append(self)
    if result.get('shm'):
        stream = result['shm']['stream_name']
        if stream not in report['streams']: report['streams'].append(stream)
    return original_copy(self, result, width, height, render_var)


def consumed(self, state, result, *args, **kwargs):
    global copy_sequence
    submission = result.get('_workflow_submission')
    rendered = original_consume(self, state, result, *args, **kwargs)
    copy_sequence += 1
    proof = dict(copy_sequence=copy_sequence, copied_at=time.monotonic(),
                 submission=submission, native=dict(rendered.native_timings))
    produced_results.record(rendered, proof)
    with (out / 'native-copies.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(proof) + '\n')
    return rendered


def drawn(self, context, texture, result, scene=None):
    original_draw(self, context, texture, result, scene)
    draws[context.area.as_pointer()] = (result, time.monotonic())


physics_capabilities.validate_fixed_runtime = admission
RuntimeSession._exchange = exchange
PublicSdkRenderClient._copy_frame = copied
PublicSdkRenderClient._consume_render_result = consumed
engine.OvrtxExampleRenderEngine._draw_viewport_texture = drawn
sessions.demand_physics_playback(scene)


def snapshot(runtime):
    # Telemetry runs off Blender's callback thread and uses the scheduler's
    # normal lock order. It neither starts a client nor steps/resets a solver.
    with runtime.scheduler._tick_lock:
        adapter = runtime.ovphysx
        controller = adapter.controller
        assert controller is not None and controller.started and not controller.closed
        client = controller._physics_client
        owner = client.runtime
        if owner not in physics_owners: physics_owners.append(owner)
        with owner.transaction():
            health = owner.request('health', {})
            states, raw = client.read_body_states(controller.current_simulation_time_ns)
            producer = asdict(controller._pose_producer.state())
            pose_producer = controller._pose_producer
            with pose_producer._lock:
                producer_activity = dict(playback_active=pose_producer._playback_active,
                                         thread_alive=pose_producer._thread is not None and pose_producer._thread.is_alive(),
                                         stop_requested=pose_producer._stop_event is None or pose_producer._stop_event.is_set())
            identity = owner.identity
            assert owner.returncode is None and identity is not None
            assert health['_runtime_pid'] == identity.pid and health['_runtime_epoch'] == identity.epoch
            assert health['_runtime_session'] == identity.session
            return dict(object_ids=[id(runtime), id(adapter), id(controller), id(client), id(owner)],
                        **asdict(identity), loaded=health['loaded'], stage_identity=health['stage_identity'],
                        physics_identity=health['physics_identity'], runtime_identity=health['runtime_identity'],
                        cpu_only_policy=health['cpu_only_policy'], physics_generation=adapter.physics_generation,
                        expected_paths=list(client.config.body_prims), states=states,
                        producer=producer, producer_activity=producer_activity,
                        observation_method='scheduler tick lock + owner transaction; no producer stop, adoption, reset or step',
                        timestep_ns=controller.config.timestep_ns,
                        step_count=controller.step_count, controller_time_ns=controller.current_simulation_time_ns,
                        backend_time_ns=raw['simulation_time_ns'])


steps = ['initial', 'play', 'before', 'renderer_fault', 'during', 'restart',
         'after', 'physics_fault', 'uncertain', 'renderer_survives', 'close']
if control_only:
    steps = ['initial', 'play', 'before', 'control_wait', 'during', 'close']
step, armed = 0, False
changed_at = heartbeat_at = time.monotonic()
heartbeat_max = 0.
renderer_before = renderer_after = None
last_frame = None


def persist():
    (out / 'result.json').write_text(json.dumps(report, indent=2, default=str) + '\n')


def heartbeat():
    global heartbeat_at, heartbeat_max
    now = time.monotonic()
    heartbeat_max = max(heartbeat_max, now - heartbeat_at)
    heartbeat_at = now
    report['heartbeat'] = dict(interval=.05, maximum_gap=heartbeat_max, limit=2.)
    return .05


def advance():
    global step, armed, changed_at
    step += 1
    armed = False
    changed_at = time.monotonic()
    persist()


def frame():
    available = [(r, t) for r, t in draws.values() if t > changed_at and r is not last_frame
                 and (r.width, r.height) == (320, 240)]
    if not available:
        return None
    result, _ = max(available, key=lambda item: item[1])
    rgba = np.frombuffer(result.rgba8_bytes(origin='top_left'), np.uint8).reshape(240, 320, 4).copy()
    assert rgba[..., :3].std() > 5
    return result, rgba


def uncertainty():
    owner = physics_owners[0]
    before = owner.diagnostics()
    refusals = []
    for op, params in [('step', {'dt_ns': 16666666}), ('read', {})]:
        try:
            owner.request(op, params)
        except Exception as exc:
            assert 'unknown' in str(exc), str(exc)
            refusals.append(dict(operation=op, error=str(exc)))
        else:
            raise AssertionError('Uncertain physics state silently reused')
    after = owner.diagnostics()
    assert_uncertain_not_replayed(before, after)
    return dict(before=before, after=after, refused_operations=refusals)


def tick():
    global armed, future, fault_armed, physics_fault_armed, renderer_before, renderer_after, last_frame
    try:
        name = steps[step]
        if time.monotonic() - changed_at > 150:
            raise TimeoutError(name)
        runtime = sessions.active_runtime_for_scene(scene)
        if name in {'before', 'renderer_fault', 'during', 'restart', 'after'}:
            assert bpy.context.screen.is_animation_playing, 'Ordinary timeline playback stopped during recovery'
        if not armed:
            armed = True
            started = time.monotonic()
            if name == 'play':
                assert bpy.ops.screen.animation_play() == {'FINISHED'}
            elif name in {'before', 'during', 'after'}:
                assert runtime is not None
                report.setdefault('playback_observations', {})[name] = dict(
                    before_observer=bool(bpy.context.screen.is_animation_playing), frame=scene.frame_current)
                future = executor.submit(snapshot, runtime)
            elif name == 'renderer_fault':
                fault_armed = True
                cube.location.x = 2.5
            elif name == 'restart':
                assert bpy.ops.ovrtx_example.restart_ovrtx_worker() == {'FINISHED'}
            elif name == 'physics_fault':
                physics_fault_armed = True
            elif name == 'uncertain':
                future = executor.submit(uncertainty)
            elif name == 'renderer_survives':
                # An ordinary renderer setting demands another native frame.
                baseline = produced_results.latest_proof()
                report['survival_demand'] = survival_boundary(renderer_after, baseline,
                    armed_at=time.monotonic(),
                    submission_sequence=submission_sequence, copy_sequence=baseline['copy_sequence'])
                matching = [c for c in clients if c.runtime.pid == renderer_after['runtime_pid']]
                assert matching
                report['survival_demand_diagnostics'] = matching[-1].runtime.diagnostics()
                persist()
                scene.ovrtx_example.max_samples += 32
            elif name == 'close':
                if bpy.context.screen.is_animation_playing:
                    bpy.ops.screen.animation_cancel(restore_frame=False)
                fixture['addon'].unregister()
            elapsed = time.monotonic() - started
            report['callback_seconds'].append(dict(name=name, seconds=elapsed))
            assert elapsed < 1., (name, elapsed)
            return .1
        if name in {'before', 'during', 'after'}:
            if not future.done():
                return .1
            value = future.result()
            report['playback_observations'][name]['after_observer'] = bool(bpy.context.screen.is_animation_playing)
            assert all(report['playback_observations'][name][key] for key in ('before_observer', 'after_observer'))
            validate_live(value)
            report['physics'][name] = value
            if name == 'during':
                if control_only:
                    try:
                        report['during_progression'] = assert_live_progress(report['physics']['before'], value)
                    except AssertionError:
                        report['motion_oracle'] = dict(passed=False, error=traceback.format_exc())
                    else:
                        report['motion_oracle'] = dict(passed=True)
                else:
                    report['during_progression'] = assert_live_progress(report['physics']['before'], value)
            elif name == 'after':
                report['after_progression'] = assert_live_progress(report['physics']['during'], value)
            persist()
            advance()
            return .1
        if name == 'control_wait':
            assert bpy.context.screen.is_animation_playing
            if time.monotonic() - changed_at < report['control_wait_seconds']:
                return .1
            assert not report['faults']
            advance()
            return .1
        if name == 'renderer_fault':
            failures = [str(s.failure) for c in clients for s in c.states.values() if s.failure is not None]
            if not any(f['kind'] == 'native_update_done_reply_not_retained' for f in report['faults']) or not failures:
                return .1
            if time.monotonic() - changed_at < 2:
                return .1
            report['cases'].append(dict(name=name, explicit_failures=failures))
            advance()
            return .1
        if name == 'physics_fault':
            if not any(f['kind'] == 'native_step_done_reply_not_retained' for f in report['faults']):
                return .1
            # Let the real producer surface its failed operation to the owner.
            if not physics_owners[0].diagnostics()['uncertain_operation']:
                return .1
            report['physics_error'] = dict(controller_error=runtime.ovphysx.controller.last_error,
                                          owner=physics_owners[0].diagnostics())
            advance()
            return .1
        if name == 'uncertain':
            if not future.done():
                return .1
            report['uncertainty'] = future.result()
            advance()
            return .1
        if name == 'close':
            status = sessions.close_status()
            assert status['status'] != 'failed', status
            if status['status'] != 'closed':
                return .1
            report['renderer_cleanup'] = [c.runtime.diagnostics() for c in clients]
            report['physics_cleanup'] = [owner.diagnostics() for owner in physics_owners]
            assert all(d['closed'] for d in report['renderer_cleanup'] + report['physics_cleanup'])
            assert len(physics_owners) == 1 and heartbeat_max < 2 and report['admissions']
            forbidden = {'ovrtx', 'ovphysx', 'ovstage', 'ovstream', 'warp', 'ovusd', 'ovmaterial', 'ovwarp'}
            report['vendor_sdk_imports'] = sorted(k for k in sys.modules if k.split('.')[0] in forbidden)
            assert not report['vendor_sdk_imports']
            report['status'] = 'diagnostic-complete' if control_only else 'passed'
            persist()
            executor.shutdown(wait=False)
            bpy.ops.wm.quit_blender()
            return None
        candidate = frame()
        if candidate is None:
            return .1
        result, rgba = candidate
        if name == 'initial':
            if runtime is None or runtime.preparation_status != 'ready' or runtime.ovphysx.controller is None:
                return .1
            if not runtime.ovphysx.controller.started:
                return .1
            renderer_before = dict(result.native_timings)
        elif name == 'play':
            if runtime.ovphysx.controller.step_count < 2:
                return .1
        elif name == 'restart':
            if engine.worker_restart_status()['status'] == 'restart_pending':
                return .1
            if result.native_timings['runtime_pid'] == renderer_before['runtime_pid']:
                return .1
            renderer_after = dict(result.native_timings)
        elif name == 'renderer_survives':
            proof = produced_results.proof_for(result)
            if (proof is None or proof['submission'] is None or
                    proof['submission']['started'] <= report['survival_demand']['armed_at']):
                return .1
            report['survival_newly_drawn_native_frame'] = proof
            persist()
            assert_fresh_frame(report['survival_demand'], proof)
            for key in RENDER_OWNER_FIELDS:
                assert result.native_timings[key] == renderer_after[key], ('Renderer changed after physics failure', key)
            matching = [c for c in clients if c.runtime.pid == renderer_after['runtime_pid']]
            assert matching and matching[-1].runtime.returncode is None
            report['renderer_survival_identity'] = asdict(matching[-1].runtime.identity)
            report['renderer_survival_diagnostics'] = matching[-1].runtime.diagnostics()
            assert_renderer_not_reloaded(report['survival_demand_diagnostics'], report['renderer_survival_diagnostics'])
            assert_uncertain_not_replayed(report['uncertainty']['after'], physics_owners[0].diagnostics())
        last_frame = result
        np.save(out / (name + '.npy'), rgba)
        report['cases'].append(dict(name=name, native=dict(result.native_timings), samples=result.completed_samples,
                                    sha256=hashlib.sha256(rgba.tobytes()).hexdigest()))
        advance()
        return .1
    except Exception:
        report.update(status='failed', failed_step=steps[step], error=traceback.format_exc(),
                      restart=engine.worker_restart_status(), close=sessions.close_status())
        report['native_copy_progress'] = dict(submission_sequence=submission_sequence,
                                              copy_sequence=copy_sequence)
        report['renderer_diagnostics_at_failure'] = [c.runtime.diagnostics() for c in clients]
        persist()
        executor.shutdown(wait=False, cancel_futures=True)
        bpy.ops.wm.quit_blender()
        return None


bpy.app.timers.register(heartbeat, first_interval=.05)
bpy.app.timers.register(tick, first_interval=.2)



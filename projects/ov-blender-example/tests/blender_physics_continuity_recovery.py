# SPDX-License-Identifier: Apache-2.0
"""Real scene-owned physics remains exact through renderer death and UI restart."""
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
import traceback
import uuid

import bpy
import numpy as np

source = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(source / 'tests'))
from physics_continuity_checks import assert_preserved, assert_progressed, validate_snapshot

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
scene, cube = fixture['scene'], fixture['cube']
scene.render.resolution_x, scene.render.resolution_y = 320, 240
scene.ovrtx_example.color_presentation_mode = 'ldr_rgba8_display_passthrough'
scene.ovrtx_example.max_samples = 64
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
              measurement_only=False, local_manifest_override=None)
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='continuity-observer')
future = None
clients = []
physics_owners = []
draws = {}
fault_armed = False
original_validate = physics_capabilities.validate_fixed_runtime
original_exchange = RuntimeSession._exchange
original_copy = PublicSdkRenderClient._copy_frame
original_draw = engine.OvrtxExampleRenderEngine._draw_viewport_texture


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
    global fault_armed
    result = original_exchange(self, op, params)
    if self.kind == 'renderer' and fault_armed and op == 'update':
        fault_armed = False
        report['faults'].append(dict(kind='native_update_done_reply_not_retained',
                                    values=params['values'], native_updated_count=result.get('updated_count'),
                                    termination=terminate_producer(self)))
        raise RuntimeTransportError('Continuity fixture: renderer died after native update before ACK retention')
    return result


def copied(self, result, width, height, render_var):
    if self not in clients: clients.append(self)
    if result.get('shm'):
        stream = result['shm']['stream_name']
        if stream not in report['streams']: report['streams'].append(stream)
    return original_copy(self, result, width, height, render_var)


def drawn(self, context, texture, result, scene=None):
    original_draw(self, context, texture, result, scene)
    draws[context.area.as_pointer()] = (result, time.monotonic())


physics_capabilities.validate_fixed_runtime = admission
RuntimeSession._exchange = exchange
PublicSdkRenderClient._copy_frame = copied
engine.OvrtxExampleRenderEngine._draw_viewport_texture = drawn
sessions.demand_physics_playback(scene)


def snapshot(runtime):
    # Telemetry runs off Blender's callback thread and uses the scheduler's
    # normal lock order. It neither starts a client nor steps/resets a solver.
    with runtime.scheduler._tick_lock:
        adapter = runtime.ovphysx
        controller = adapter.controller
        assert controller is not None and controller.started and not controller.closed
        controller.stop_async_playback(wait=True, invalidate=True)
        controller.adopt_async_producer_position()
        client = controller._physics_client
        owner = client.runtime
        if owner not in physics_owners: physics_owners.append(owner)
        with owner.transaction():
            health = owner.request('health', {})
            states, raw = client.read_body_states(controller.current_simulation_time_ns)
            identity = owner.identity
            assert owner.returncode is None and identity is not None
            assert health['_runtime_pid'] == identity.pid and health['_runtime_epoch'] == identity.epoch
            assert health['_runtime_session'] == identity.session
            return dict(object_ids=[id(runtime), id(adapter), id(controller), id(client), id(owner)],
                        **asdict(identity), loaded=health['loaded'], stage_identity=health['stage_identity'],
                        physics_identity=health['physics_identity'], runtime_identity=health['runtime_identity'],
                        cpu_only_policy=health['cpu_only_policy'], physics_generation=adapter.physics_generation,
                        expected_paths=list(client.config.body_prims), states=states,
                        step_count=controller.step_count, controller_time_ns=controller.current_simulation_time_ns,
                        backend_time_ns=raw['simulation_time_ns'])


steps = ['initial', 'advance', 'baseline', 'held', 'before', 'fault', 'failed_snapshot',
         'restart', 'after', 'resume', 'progressed', 'close']
step, armed = 0, False
changed_at = heartbeat_at = time.monotonic()
heartbeat_max = 0.
reference = None
last_frame = None
renderer_before = None
restore_limit = None


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
                 and r.completed_samples >= 32 and (r.width, r.height) == (320, 240)]
    if not available: return None
    result, _ = max(available, key=lambda item: item[1])
    rgba = np.frombuffer(result.rgba8_bytes(origin='top_left'), np.uint8).reshape(240, 320, 4).copy()
    assert rgba[..., :3].std() > 5
    return result, rgba


def tick():
    global armed, future, fault_armed, reference, last_frame, renderer_before, restore_limit
    try:
        name = steps[step]
        assert not bpy.context.screen.is_animation_playing
        if time.monotonic() - changed_at > 150: raise TimeoutError(name)
        runtime = sessions.active_runtime_for_scene(scene)
        if not armed:
            armed = True
            started = time.monotonic()
            if name == 'advance': scene.frame_set(scene.frame_start + 3)
            elif name in {'baseline', 'held'}: scene.ovrtx_example.max_samples += 32
            elif name in {'before', 'failed_snapshot', 'after', 'progressed'}:
                assert runtime is not None
                future = executor.submit(snapshot, runtime)
            elif name == 'fault':
                fault_armed = True
                cube.location.x = 2.5
            elif name == 'restart':
                result = bpy.ops.ovrtx_example.restart_ovrtx_worker()
                assert result == {'FINISHED'}, result
            elif name == 'resume': scene.frame_set(scene.frame_start + 6)
            elif name == 'close': fixture['addon'].unregister()
            elapsed = time.monotonic() - started
            report['callback_seconds'].append(dict(name=name, seconds=elapsed))
            assert elapsed < 1., (name, elapsed)
            return .1
        if name in {'before', 'failed_snapshot', 'after', 'progressed'}:
            if not future.done(): return .1
            result = future.result()
            report['physics'][name] = result
            persist()
            validate_snapshot(result)
            if name in {'failed_snapshot', 'after'}: assert_preserved(report['physics']['before'], result)
            if name == 'progressed':
                report['progression'] = assert_progressed(report['physics']['after'], result)
            advance()
            return .1
        if name == 'fault':
            failures = [str(s.failure) for c in clients for s in c.states.values() if s.failure is not None]
            if not report['faults'] or not failures: return .1
            report['cases'].append(dict(name=name, explicit_failures=failures))
            advance()
            return .1
        if name == 'close':
            status = sessions.close_status()
            assert status['status'] != 'failed', status
            if status['status'] != 'closed': return .1
            report['renderer_cleanup'] = [c.runtime.diagnostics() for c in clients]
            assert all(d['closed'] for d in report['renderer_cleanup'])
            report['physics_cleanup'] = [owner.diagnostics() for owner in physics_owners]
            assert len(physics_owners) == 1 and all(d['closed'] for d in report['physics_cleanup'])
            assert heartbeat_max < 2.
            assert report['admissions']
            forbidden = {'ovrtx', 'ovphysx', 'ovstage', 'ovstream', 'warp', 'ovusd', 'ovmaterial', 'ovwarp'}
            report['vendor_sdk_imports'] = sorted(k for k in sys.modules if k.split('.')[0] in forbidden)
            assert not report['vendor_sdk_imports']
            report['status'] = 'passed'
            persist()
            executor.shutdown(wait=False)
            bpy.ops.wm.quit_blender()
            return None
        candidate = frame()
        if candidate is None: return .1
        result, rgba = candidate
        if name == 'initial':
            if runtime is None or runtime.preparation_status != 'ready' or runtime.ovphysx.controller is None: return .1
            if not runtime.ovphysx.controller.started: return .1
        if name == 'restart':
            if engine.worker_restart_status()['status'] == 'restart_pending': return .1
            if result.native_timings.get('runtime_pid') == renderer_before: return .1
        if name in {'advance', 'resume'}:
            # Wait for scheduler consumption of this ordinary forward scrub.
            if runtime.scheduler._last_timeline_frame != scene.frame_current: return .1
        last_frame = result
        np.save(out / (name + '.npy'), rgba)
        if name == 'baseline':
            reference = rgba
            renderer_before = result.native_timings['runtime_pid']
        mae = None if reference is None else float(np.abs(rgba.astype(float) - reference).mean())
        report['cases'].append(dict(name=name, native=dict(result.native_timings), samples=result.completed_samples,
                                    mae=mae, sha256=hashlib.sha256(rgba.tobytes()).hexdigest()))
        persist()
        if name == 'held':
            restore_limit = max(.3, 6 * mae)
            report['restoration_limit'] = restore_limit
        if name == 'restart': assert mae <= restore_limit, (mae, restore_limit)
        advance()
        return .1
    except Exception:
        report.update(status='failed', failed_step=steps[step], error=traceback.format_exc(),
                      restart=engine.worker_restart_status(), close=sessions.close_status())
        persist()
        executor.shutdown(wait=False, cancel_futures=True)
        bpy.ops.wm.quit_blender()
        return None


bpy.app.timers.register(heartbeat, first_interval=.05)
bpy.app.timers.register(tick, first_interval=.2)

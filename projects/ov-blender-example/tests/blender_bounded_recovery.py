# SPDX-License-Identifier: Apache-2.0
"""Actual viewport/operator recovery, with raw owned-frame references and faults."""
import hashlib
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import runpy
import sys
import threading
import time
import traceback

import bpy
import numpy as np

source = Path(__file__).resolve().parents[1]
os.environ['OV_BLENDER_EXAMPLE_FIXED_VIEWPORT_RESOLUTION'] = '1'
fixture = runpy.run_path(str(source / 'scripts/probe_public_sdk_ui_edits.py'))
from ovrtx_blender_example import engine, scene_generation_sessions as sessions
from ovrtx_blender_example.direct_sdk_client import PublicSdkRenderClient
from ovrtx_blender_example.direct_runtime_session import RuntimeSession, RuntimeTransportError

out = Path(os.environ['OV_BLENDER_PUBLIC_SDK_OUTPUT']) / 'recovery'
out.mkdir(parents=True, exist_ok=True)
scene, cube = fixture['scene'], fixture['cube']
scene.ovrtx_example.color_presentation_mode = 'ldr_rgba8_display_passthrough'
scene.ovrtx_example.max_samples = 64
scene.render.resolution_x, scene.render.resolution_y = 320, 240
report = dict(status='running', cases=[], faults=[], callback_seconds=[],
              fixture_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              policy={'channel': 'raw display LDR rgba8', 'minimum_samples': 32,
                      'restoration_mae': 'max(0.3, 6 * initial held MAE)',
                      'edit_mae': 'max(1, 8 * initial held MAE)'})
original_exchange = RuntimeSession._exchange
original_copy = PublicSdkRenderClient._copy_frame
original_draw = engine.OvrtxExampleRenderEngine._draw_viewport_texture
draws = {}
clients = []
streams = set()
fault = None
blocked = threading.Event()
release = threading.Event()
heartbeat_at = time.monotonic()
heartbeat_max = 0.0


def terminate_producer(runtime):
    identity = runtime.identity
    owner = runtime._owner
    assert identity is not None and owner is not None
    assert owner.worker_pid == identity.pid and owner.poll_worker() is None
    assert owner.contains(identity.pid)
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    api.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    api.OpenProcess.restype = wintypes.HANDLE
    api.TerminateProcess.argtypes = (wintypes.HANDLE, wintypes.UINT)
    api.TerminateProcess.restype = wintypes.BOOL
    api.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    api.WaitForSingleObject.restype = wintypes.DWORD
    api.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = api.OpenProcess(0x100001, False, identity.pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        # The authenticated owner's retained handle pins the producer identity;
        # job membership checks prevent confusing its launcher with the producer.
        assert runtime.identity == identity and owner.contains(identity.pid)
        if not api.TerminateProcess(handle, 241):
            raise ctypes.WinError(ctypes.get_last_error())
        assert api.WaitForSingleObject(handle, 5000) == 0
        assert owner.poll_worker() == 241
        return dict(producer_pid=identity.pid, launcher_pid=identity.launcher_pid,
                    epoch=identity.epoch, session_nonce=identity.session,
                    producer_exit_code=241, launcher_returncode=runtime._process.poll())
    finally:
        api.CloseHandle(handle)


def heartbeat():
    global heartbeat_at, heartbeat_max
    now = time.monotonic()
    gap = now - heartbeat_at
    heartbeat_at = now
    heartbeat_max = max(heartbeat_max, gap)
    report['heartbeat'] = dict(interval=.05, maximum_gap=heartbeat_max, limit=2.0)
    return .05


def observed_draw(self, context, texture, result, scene=None):
    original_draw(self, context, texture, result, scene)
    draws[context.area.as_pointer()] = (self, result, time.monotonic())


def exchange(self, op, params):
    global fault
    if self.kind == 'renderer' and fault == 'block_render' and op == 'render':
        fault = None
        blocked.set()
        if not release.wait(15):
            raise AssertionError('Fixture did not release blocked render')
    result = original_exchange(self, op, params)
    if self.kind == 'renderer' and fault == 'lose_update_ack' and op == 'update':
        fault = None
        report['faults'].append(dict(kind='real_native_update_reply_lost_before_retention',
                                    pid=self.pid, epoch=self.epoch, values=params['values'],
                                    native_updated_count=result.get('updated_count')))
        report['faults'][-1]['termination'] = terminate_producer(self)
        raise RuntimeTransportError('Fixture producer death after native write, before accepted reply retention')
    return result


def copy_frame(self, result, width, height, render_var):
    global fault
    if self not in clients:
        clients.append(self)
    if result.get('shm'):
        streams.add(result['shm']['stream_name'])
        report['streams'] = sorted(streams)
    if fault == 'die_owned_copy':
        fault = None
        report['faults'].append(dict(kind='producer_death_while_ticket_owned_before_copy',
                                    ticket=result.get('shm'), pid=self.runtime.pid,
                                    epoch=self.runtime.epoch))
        report['faults'][-1]['termination'] = terminate_producer(self.runtime)
    return original_copy(self, result, width, height, render_var)


RuntimeSession._exchange = exchange
PublicSdkRenderClient._copy_frame = copy_frame
engine.OvrtxExampleRenderEngine._draw_viewport_texture = observed_draw
steps = ['initial', 'held', 'uncertain_update', 'restart_update', 'owned_copy_death',
         'restart_copy', 'scene_replace_blocked', 'scene_added', 'scene_removed',
         'resize_burst', 'editor_off', 'editor_restored', 'disable', 'reenable', 'close']
step = 0
armed = False
changed_at = time.monotonic()
previous_identity = None
reference = None
noise = None
restore_limit = None
edit_limit = None
added_name = 'Recovery During Render'
area = next(a for a in bpy.context.screen.areas if a.type == 'VIEW_3D')
last_fault_count = 0
last_result = None


def persist():
    (out / 'result.json').write_text(json.dumps(report, indent=2, default=str) + '\n')


def timed(action):
    started = time.monotonic()
    result = action()
    elapsed = time.monotonic() - started
    report['callback_seconds'].append(dict(step=steps[step], seconds=elapsed, result=str(result)))
    assert elapsed < 1, ('Blender callback blocked', steps[step], elapsed)
    return result


def advance():
    global step, armed, changed_at
    step += 1
    armed = False
    changed_at = time.monotonic()
    persist()


def pixels_for_step():
    candidates = [value for value in draws.values() if value[2] > changed_at]
    if not candidates:
        return None
    instance, result, stamp = max(candidates, key=lambda value: value[2])
    if result is last_result:
        return None
    if result.completed_samples < 32 or (result.width, result.height) != (320, 240):
        return None
    rgba = np.frombuffer(result.rgba8_bytes(origin='top_left'), np.uint8).reshape(240, 320, 4).copy()
    assert rgba[..., :3].std() > 5
    native = dict(result.native_timings)
    identity = (native.get('runtime_pid'), native.get('runtime_epoch'), native.get('native_stage_id'))
    return instance, result, rgba, native, identity


def record_pixels(name, frame, *, restored=False, changed=False, new_worker=False):
    global previous_identity, reference, noise, restore_limit, edit_limit, last_result
    instance, result, rgba, native, identity = frame
    np.save(out / (name + '.npy'), rgba)
    if name == 'initial':
        reference = rgba
    mae = None if reference is None else float(np.abs(rgba.astype(float) - reference).mean())
    if name == 'held':
        noise = mae
        restore_limit = max(.3, 6 * noise)
        edit_limit = max(1., 8 * noise)
        report['frozen_limits'] = dict(noise=noise, restoration=restore_limit, edit=edit_limit)
    row = dict(name=name, mae=mae, identity=identity, native=native,
               samples=result.completed_samples, sha256=hashlib.sha256(rgba.tobytes()).hexdigest())
    report['cases'].append(row)
    persist()  # Retain failing frames/metrics before any acceptance assertion.
    if restored:
        assert mae <= restore_limit, (name, mae, restore_limit)
    if changed:
        assert mae > edit_limit, (name, mae, edit_limit)
    if new_worker:
        assert identity[0] != previous_identity[0], (identity, previous_identity)
    previous_identity = identity
    last_result = result


def tick():
    global armed, changed_at, fault, last_fault_count
    try:
        name = steps[step]
        if time.monotonic() - changed_at > 150:
            raise TimeoutError(name)
        if not armed:
            armed = True
            if name == 'held':
                timed(lambda: setattr(scene.ovrtx_example, 'max_samples', scene.ovrtx_example.max_samples + 32))
            elif name == 'uncertain_update':
                fault = 'lose_update_ack'
                last_fault_count = len(report['faults'])
                timed(lambda: setattr(cube.location, 'x', 2.5))
            elif name in {'restart_update', 'restart_copy'}:
                assert timed(lambda: bpy.ops.ovrtx_example.restart_ovrtx_worker()) == {'FINISHED'}
                report['restart_requested'] = engine.worker_restart_status()
            elif name == 'owned_copy_death':
                fault = 'die_owned_copy'
                last_fault_count = len(report['faults'])
                # Ordinary sampling RNA obtains a fresh frame without changing
                # the accepted camera or image reference.
                timed(lambda: setattr(scene.ovrtx_example, 'max_samples', scene.ovrtx_example.max_samples + 32))
                area.tag_redraw()
            elif name == 'scene_replace_blocked':
                fault = 'block_render'
                release.clear()
                blocked.clear()
                timed(lambda: setattr(scene.ovrtx_example, 'max_samples', scene.ovrtx_example.max_samples + 32))
                area.tag_redraw()
            elif name == 'scene_removed':
                timed(lambda: bpy.data.objects.remove(bpy.data.objects[added_name], do_unlink=True))
            elif name == 'resize_burst':
                def resize():
                    for width in (352, 384, 416, 448, 320):
                        scene.render.resolution_x = width
                    scene.render.resolution_y = 240
                    scene.ovrtx_example.max_samples += 32
                timed(resize)
            elif name == 'editor_off':
                timed(lambda: setattr(area, 'type', 'NODE_EDITOR'))
            elif name == 'editor_restored':
                def restore():
                    area.type = 'VIEW_3D'
                    area.spaces.active.region_3d.view_perspective = 'CAMERA'
                    area.spaces.active.region_3d.view_camera_zoom = 5
                    area.spaces.active.shading.type = 'RENDERED'
                    area.spaces.active.overlay.show_overlays = False
                timed(restore)
            elif name in {'disable', 'close'}:
                timed(fixture['addon'].unregister)
            elif name == 'reenable':
                timed(fixture['addon'].register)
                scene.render.engine = 'OVRTX_EXAMPLE'
                area.spaces.active.shading.type = 'RENDERED'
            return .1
        if name in {'uncertain_update', 'owned_copy_death'}:
            if len(report['faults']) == last_fault_count:
                return .1
            failures = [s.failure for client in clients for s in client.states.values() if s.failure is not None]
            if not failures:
                return .1
            report['cases'].append(dict(name=name, explicit_failure=[str(f) for f in failures]))
            advance()
            return .1
        if name == 'scene_replace_blocked':
            if not blocked.is_set():
                return .1
            def add():
                bpy.ops.mesh.primitive_cube_add(size=2, location=(-1, -2, 1))
                bpy.context.object.name = added_name
            timed(add)
            release.set()
            advance()
            return .1
        if name == 'editor_off':
            if time.monotonic() - changed_at < .5:
                return .1
            advance()
            return .1
        if name in {'disable', 'close'}:
            status = sessions.close_status()
            assert status['status'] != 'failed', status
            if status['status'] != 'closed':
                return .1
            diagnostics = [client.runtime.diagnostics() for client in clients]
            assert all(d['closed'] for d in diagnostics), diagnostics
            report['cases'].append(dict(name=name, close=status, runtimes=diagnostics))
            if name == 'close':
                forbidden = ('ovrtx', 'ovstage', 'ovphysx', 'ovstream', 'warp', 'ovusd', 'ovmaterial', 'ovwarp')
                report['vendor_sdk_imports'] = [key for key in sys.modules if key.split('.')[0] in forbidden]
                assert not report['vendor_sdk_imports']
                assert heartbeat_max < 2.0, report['heartbeat']
                report['status'] = 'passed'
                persist()
                bpy.ops.wm.quit_blender()
                return None
            advance()
            return .1
        frame = pixels_for_step()
        if frame is None:
            return .1
        if name.startswith('restart_') and engine.worker_restart_status()['status'] == 'restart_pending':
            return .1
        restored = name in {'restart_update', 'restart_copy', 'scene_removed', 'resize_burst', 'editor_restored', 'reenable'}
        trial_mae = None if reference is None else float(np.abs(frame[2].astype(float) - reference).mean())
        if (restored and trial_mae > restore_limit) or (name == 'scene_added' and trial_mae <= edit_limit):
            np.save(out / (name + '-pending.npy'), frame[2])
            report['pending_frame'] = dict(step=name, mae=trial_mae, identity=frame[4], native=frame[3])
            persist()
            return .1
        if name in {'restart_update', 'restart_copy'} and frame[4][0] == previous_identity[0]:
            return .1
        record_pixels(name, frame, restored=restored, changed=name == 'scene_added',
                      new_worker=name in {'restart_update', 'restart_copy'})
        if name == 'restart_update':
            # Recovery has now proven the failed RNA transform was not replayed.
            # Explicitly restore authoring intent before later full populations.
            timed(lambda: setattr(cube.location, 'x', 0))
        advance()
        return .1
    except Exception:
        release.set()
        report.update(status='failed', failed_step=steps[step], error=traceback.format_exc(),
                      restart=engine.worker_restart_status(), close=sessions.close_status())
        persist()
        bpy.ops.wm.quit_blender()
        return None


bpy.app.timers.register(heartbeat, first_interval=.05)
bpy.app.timers.register(tick, first_interval=.2)

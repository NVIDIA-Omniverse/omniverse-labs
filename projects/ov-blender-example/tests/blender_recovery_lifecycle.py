# SPDX-License-Identifier: Apache-2.0
"""Real transport deadline, mode bursts, INVOKE cancellation and file-load recovery.

Reuses the first batch's scene/observers, not its already-qualified actions.
The timeout suspends only the authenticated producer, then calls the unchanged
socket transport. Cancellation clicks Blender's stock running-job Cancel button
after a real native sample; it does not replace the cancellation predicate.
"""
import ctypes
from ctypes import wintypes
import hashlib
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

base = runpy.run_path(str(Path(__file__).with_name('blender_bounded_recovery.py')))
g = base['tick'].__globals__
bpy.app.timers.unregister(g['tick'])
bpy.app.timers.unregister(g['heartbeat'])
bpy.app.timers.unregister(g['fixture']['tick'])  # Unused LDR-only telemetry.
engine, sessions = g['engine'], g['sessions']
scene, area = g['scene'], g['area']
report, out = g['report'], g['out']
report['fixture_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
report['policy']['hdr_restoration_mae'] = 'max(0.005, 6 * held raw HDR MAE)'
steps = ['initial', 'held', 'timeout', 'restart_timeout', 'hdr_initial', 'hdr_held',
         'mode_burst_hdr', 'mode_burst_ldr', 'cancel_prepare', 'cancel', 'after_cancel',
         'file_load', 'after_file_load', 'close']
if os.environ.get('OVRTX_RECOVERY_LIFECYCLE_TAIL') == '1':
    steps = ['initial', 'held', 'cancel_prepare', 'cancel', 'after_cancel', 'file_load', 'after_file_load', 'close']
if os.environ.get('OVRTX_RECOVERY_FILE_LOAD_ONLY') == '1':
    steps = ['initial', 'held', 'file_load', 'after_file_load', 'close']
report['required_steps'] = steps
g['steps'] = steps
step = 0
armed = False
changed_at = time.monotonic()
timeout_armed = False
timeout_done = False
timeout_runtime = None
timeout_seen_at = None
timeout_counts = None
file_blocked = threading.Event()
file_rpc_completed = threading.Event()
file_request_count_before = 0
file_opened = False
cancel_armed = False
cancel_native = 0
cancel_outcome = None
final_writes = 0
render_finished = False
final_thread = None
escape_sent = False
original_header_draw = bpy.types.VIEW3D_HT_header.draw
original_render_display = bpy.context.preferences.view.render_display_type
hdr_reference = None
hdr_limit = None
last_result = None
old_file_clients = []
saved_blend = out / 'recovery.blend'
cancel_output = out / 'cancel-must-not-exist.png'
original_job = engine._run_final_render_job
original_render = engine.OvrtxSessionController.render
original_write = engine.OvrtxExampleRenderEngine._write_blender_result


def exchange(self, op, params):
    global timeout_armed, timeout_done, timeout_runtime, file_request_count_before
    if not (timeout_armed and self.kind == 'renderer' and op == 'render'):
        return g['original_exchange'](self, op, params)
    fault_role = timeout_armed
    timeout_armed = False
    timeout_runtime = self
    identity, owner = self.identity, self._owner
    assert identity and owner.worker_pid == identity.pid and owner.contains(identity.pid)
    assert owner.poll_worker() is None
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    api.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    api.OpenProcess.restype = wintypes.HANDLE
    api.CloseHandle.argtypes = (wintypes.HANDLE,)
    nt = ctypes.WinDLL('ntdll')
    for name in ('NtSuspendProcess', 'NtResumeProcess'):
        getattr(nt, name).argtypes = (wintypes.HANDLE,)
        getattr(nt, name).restype = wintypes.LONG
    handle = api.OpenProcess(0x0800, False, identity.pid)  # PROCESS_SUSPEND_RESUME
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    old_timeout = self.timeout_s
    row = dict(kind='real_socket_deadline_with_owned_producer_suspended',
               producer_pid=identity.pid, launcher_pid=identity.launcher_pid,
               epoch=identity.epoch, session=identity.session,
               role='file_load' if fault_role == 'file_load' else 'restart',
               deadline_seconds=3.0 if fault_role == 'file_load' else .25,
               requests_before=dict(self._request_counts))
    suspended = False
    started = time.monotonic()
    try:
        assert owner.contains(identity.pid) and self.identity == identity
        assert nt.NtSuspendProcess(handle) == 0
        suspended = True
        self.timeout_s = row['deadline_seconds']
        if fault_role == 'file_load':
            file_request_count_before = self._request_counts.get('render', 0)
            file_blocked.set()
        try:
            g['original_exchange'](self, op, params)
        except g['RuntimeTransportError'] as error:
            row.update(error=str(error), exception_type=type(error).__name__,
                       cause_type=type(error.__cause__).__name__,
                       elapsed_seconds=time.monotonic() - started,
                       requests_after=dict(self._request_counts), producer_alive=owner.poll_worker() is None)
            assert isinstance(error.__cause__, TimeoutError), row
            assert row['deadline_seconds'] - .05 <= row['elapsed_seconds'] < row['deadline_seconds'] + 1, row
            assert row['producer_alive'], row
            assert row['requests_after']['render'] == row['requests_before'].get('render', 0) + 1
            raise
        else:
            raise AssertionError('Suspended producer unexpectedly answered native render')
    finally:
        self.timeout_s = old_timeout
        if suspended:
            row['resume_status'] = nt.NtResumeProcess(handle)
        api.CloseHandle(handle)
        report['faults'].append(row)
        timeout_done = True
        if fault_role == 'file_load':
            file_rpc_completed.set()


def render(self, *args, **kwargs):
    global cancel_native
    result = original_render(self, *args, **kwargs)
    if cancel_armed and threading.get_ident() == final_thread:
        cancel_native += 1
        report.setdefault('cancel_native_samples', []).append(dict(result.native_timings))
    return result


def job(request, *args, **kwargs):
    global cancel_outcome, final_thread
    if cancel_armed:
        final_thread = threading.get_ident()
    try:
        result = original_job(request, *args, **kwargs)
        if cancel_armed:
            cancel_outcome = result
        return result
    finally:
        final_thread = None


def write(self, result):
    global final_writes
    final_writes += 1
    return original_write(self, result)


def finished(*args):
    global render_finished
    render_finished = True
    report['render_handler'] = 'complete'


def cancelled(*args):
    global render_finished
    render_finished = True
    report['render_handler'] = 'cancel'


def cancel_header(self, context):
    # Stock UI implementation and callback; only its location is isolated so
    # the owned-window fixture can address it without desktop-global input.
    self.layout.template_running_jobs()
    report['cancel_button_layout'] = self.layout.introspect()


g['RuntimeSession']._exchange = exchange
engine.OvrtxSessionController.render = render
engine._run_final_render_job = job
engine.OvrtxExampleRenderEngine._write_blender_result = write
bpy.app.handlers.render_complete.append(finished)
bpy.app.handlers.render_cancel.append(cancelled)


def advance():
    global step, armed, changed_at
    step += 1
    g['step'] = step
    armed = False
    changed_at = time.monotonic()
    g['changed_at'] = changed_at
    g['persist']()


def fresh_frame(hdr=False):
    global last_result
    values = [v for v in g['draws'].values() if v[2] > changed_at]
    if not values:
        return None
    instance, result, stamp = max(values, key=lambda v: v[2])
    if result is last_result or result.completed_samples < 32 or (result.width, result.height) != (320, 240):
        return None
    if result.render_var != ('HdrColor' if hdr else 'LdrColor'):
        return None
    payload = result.linear_rgba16f_bytes(origin='top_left') if hdr else result.rgba8_bytes(origin='top_left')
    pixels = np.frombuffer(payload, np.float16 if hdr else np.uint8).reshape(240, 320, 4).copy()
    assert np.isfinite(pixels).all() and pixels[..., :3].astype(float).std() > (.01 if hdr else 5)
    native = dict(result.native_timings)
    return instance, result, pixels, native, (native.get('runtime_pid'), native.get('runtime_epoch'), native.get('native_stage_id'))


def tick():
    global armed, timeout_armed, cancel_armed, hdr_reference, hdr_limit, last_result, scene, area
    global timeout_seen_at, timeout_counts, file_opened, escape_sent
    if report['status'] == 'failed':
        # Let the RPC's finally resume the owned process even if file loading
        # failed; the runner's owned-job watchdog is the final hard bound.
        if file_blocked.is_set() and not file_rpc_completed.is_set():
            return .1
        g['persist']()
        bpy.ops.wm.quit_blender()
        return None
    try:
        name = steps[step]
        assert time.monotonic() - changed_at < 150, name
        if not armed:
            armed = True
            if name in {'held', 'hdr_held', 'timeout'}:
                timeout_armed = name == 'timeout'
                g['timed'](lambda: setattr(scene.ovrtx_example, 'max_samples', scene.ovrtx_example.max_samples + 32))
            elif name == 'restart_timeout':
                assert g['timed'](lambda: bpy.ops.ovrtx_example.restart_ovrtx_worker()) == {'FINISHED'}
            elif name in {'hdr_initial', 'mode_burst_hdr', 'mode_burst_ldr'}:
                def mode():
                    for value in (['scene_linear_hdr'] if name == 'hdr_initial' else
                                  ['ldr_rgba8_display_passthrough', 'scene_linear_hdr'] * 4 +
                                  (['ldr_rgba8_display_passthrough'] if name.endswith('ldr') else [])):
                        scene.ovrtx_example.color_presentation_mode = value
                    scene.ovrtx_example.max_samples += 32
                g['timed'](mode)
            elif name == 'cancel_prepare':
                bpy.context.preferences.view.show_splash = False
                bpy.context.preferences.view.render_display_type = 'NONE'
                bpy.types.VIEW3D_HT_header.draw = cancel_header
                # Dismiss a factory-startup popup, not the render (not started).
                bpy.context.window.event_simulate(type='ESC', value='PRESS')
                bpy.context.window.event_simulate(type='ESC', value='RELEASE')
            elif name == 'cancel':
                cancel_armed = True
                scene.ovrtx_example.max_samples = 4096
                scene.render.filepath = str(cancel_output)
                scene.render.image_settings.file_format = 'PNG'
                assert g['timed'](lambda: bpy.ops.render.render('INVOKE_DEFAULT', write_still=True)) == {'RUNNING_MODAL'}
            elif name == 'file_load':
                # Save happens outside the timed teardown acceptance operation.
                bpy.ops.wm.save_as_mainfile(filepath=str(saved_blend), check_existing=False)
                old_file_clients.extend(g['clients'])
                report['file_before'] = [c.runtime.diagnostics() for c in old_file_clients]
                timeout_armed = 'file_load'
                scene.ovrtx_example.max_samples += 32
                area.tag_redraw()
            elif name == 'close':
                g['timed'](g['fixture']['addon'].unregister)
            return .1
        if name == 'cancel_prepare':
            advance()
            return .2
        if name == 'timeout':
            if not timeout_done:
                return .1
            # Render RPC failure quarantines the transport. Mutation/copy-ACK
            # state.failure is a different boundary and need not be populated.
            failures = []
            for instance, result, stamp in g['draws'].values():
                loop = getattr(instance, '_render_loop', None)
                detail = getattr(loop, '_last_failure_detail', '')
                if detail:
                    failures.append(str(detail))
            if not timeout_runtime._transport_failed or not failures:
                return .1
            if timeout_seen_at is None:
                timeout_seen_at = time.monotonic()
                timeout_counts = dict(timeout_runtime._request_counts)
                return .1
            if time.monotonic() - timeout_seen_at < .5:
                return .1
            row = report['faults'][-1]
            assert timeout_runtime.pid == row['producer_pid'] and timeout_runtime.epoch == row['epoch']
            assert timeout_runtime._owner.poll_worker() is None and row['resume_status'] == 0
            assert dict(timeout_runtime._request_counts) == timeout_counts, 'Quarantined runtime issued another RPC'
            report['cases'].append(dict(name=name, failures=failures, no_automatic_epoch_change=True,
                                        quarantined_counts=timeout_counts, runtime=timeout_runtime.diagnostics()))
            advance()
            return .1
        if name == 'cancel':
            if cancel_native > 0 and not escape_sent:
                if 'Stop this job' not in json.dumps(report.get('cancel_button_layout')):
                    area.tag_redraw()
                    return .1
                region = next(r for r in area.regions if r.type == 'HEADER')
                window = bpy.context.window
                point = dict(x=region.x + round(region.height * 9.5), y=region.y + region.height // 2)
                report['cancel_button'] = dict(native_samples_before_signal=cancel_native,
                    window=window.as_pointer(), region=dict(x=region.x,y=region.y,width=region.width,height=region.height), **point)
                def click_cancel():
                    window.event_simulate(type='MOUSEMOVE', value='NOTHING', **point)
                    window.event_simulate(type='LEFTMOUSE', value='PRESS', **point)
                    window.event_simulate(type='LEFTMOUSE', value='RELEASE', **point)
                g['timed'](click_cancel)
                escape_sent = True
                return .1
            if not render_finished or cancel_outcome is None:
                return .1
            report['cancel'] = dict(outcome=cancel_outcome, native_samples=cancel_native, final_writes=final_writes,
                                    output_exists=cancel_output.exists(),
                                    signal='Stock running-job Cancel button via owned-window mouse event after actual native sample')
            g['persist']()
            assert cancel_outcome['status'] == 'cancelled' and cancel_native > 0 and final_writes == 0
            assert escape_sent and report['render_handler'] == 'cancel'
            assert not cancel_output.exists()
            cancel_armed = False
            bpy.types.VIEW3D_HT_header.draw = original_header_draw
            bpy.context.preferences.view.render_display_type = original_render_display
            advance()
            return .1
        if name == 'file_load':
            if not file_opened:
                if not file_blocked.is_set():
                    return .1
                if timeout_runtime._request_counts.get('render', 0) == file_request_count_before:
                    return .1
                # This is the actual pending native call, not a blocked Python
                # callback: the authenticated worker cannot answer its socket.
                assert not timeout_runtime._transport_failed
                report['file_load_while_rpc_pending'] = dict(pid=timeout_runtime.pid,
                    epoch=timeout_runtime.epoch, producer_alive=timeout_runtime._owner.poll_worker() is None)
                assert report['file_load_while_rpc_pending']['producer_alive']
                g['timed'](lambda: bpy.ops.wm.open_mainfile(filepath=str(saved_blend), load_ui=False))
                report['file_load_returned_before_rpc_completion'] = not file_rpc_completed.is_set()
                assert report['file_load_returned_before_rpc_completion']
                file_opened = True
                g['persist']()
                return .1
            close = sessions.close_status()
            assert close['status'] != 'failed', close
            # diagnostics() takes the RPC lock. Never ask for it on Blender's
            # thread while the deliberately suspended call still owns it.
            if not file_rpc_completed.is_set() or close['status'] != 'closed':
                return .1
            # open_mainfile invalidates the invoking timer's context. Rebind
            # from the new window roster on a subsequent tick, after cleanup.
            windows = list(bpy.context.window_manager.windows)
            candidates = [(window, candidate) for window in windows if window.screen and window.scene
                          for candidate in window.screen.areas if candidate.type == 'VIEW_3D']
            if not candidates:
                return .1
            window, area = candidates[0]
            scene = window.scene
            g['scene'], g['area'] = scene, area
            g['draws'].clear()
            area.spaces.active.region_3d.view_perspective = 'CAMERA'
            area.spaces.active.shading.type = 'RENDERED'
            report['file_rebound'] = dict(window=window.as_pointer(), area=area.as_pointer(), scene=scene.name)
            diagnostics = [c.runtime.diagnostics() for c in old_file_clients]
            if not all(d['closed'] for d in diagnostics):
                return .1
            report['file_old_runtimes'] = diagnostics
            assert report['faults'][-1]['role'] == 'file_load' and report['faults'][-1]['resume_status'] == 0
            advance()
            return .1
        if name == 'close':
            status = sessions.close_status()
            assert status['status'] != 'failed', status
            if status['status'] != 'closed':
                return .1
            diagnostics = [c.runtime.diagnostics() for c in g['clients']]
            assert all(d['closed'] for d in diagnostics), diagnostics
            report['cases'].append(dict(name=name, close=status, runtimes=diagnostics))
            forbidden = ('ovrtx', 'ovstage', 'ovphysx', 'ovstream', 'warp', 'ovusd', 'ovmaterial', 'ovwarp')
            report['vendor_sdk_imports'] = [k for k in sys.modules if k.split('.')[0] in forbidden]
            assert not report['vendor_sdk_imports'] and g['heartbeat_max'] < 2
            report['status'] = 'passed'
            g['persist']()
            bpy.ops.wm.quit_blender()
            return None
        hdr = name in {'hdr_initial', 'hdr_held', 'mode_burst_hdr'}
        frame = fresh_frame(hdr)
        if frame is None:
            return .1
        if hdr:
            pixels = frame[2]
            np.save(out / (name + '.npy'), pixels)
            if hdr_reference is None:
                hdr_reference = pixels
            mae = float(np.abs(pixels.astype(float) - hdr_reference).mean())
            if name == 'hdr_held':
                hdr_limit = max(.005, 6 * mae)
                report['frozen_hdr_limit'] = hdr_limit
            report['cases'].append(dict(name=name, mae=mae, identity=frame[4], native=frame[3],
                                       sha256=hashlib.sha256(pixels.tobytes()).hexdigest()))
            g['persist']()
            if name == 'mode_burst_hdr':
                assert mae <= hdr_limit, (mae, hdr_limit)
        else:
            restored = name not in {'initial', 'held'}
            if name == 'restart_timeout' and frame[4][0] == g['previous_identity'][0]:
                return .1
            if restored and float(np.abs(frame[2].astype(float) - g['reference']).mean()) > g['restore_limit']:
                np.save(out / (name + '-pending.npy'), frame[2])
                return .1
            g['record_pixels'](name, frame, restored=restored, new_worker=name == 'restart_timeout')
        last_result = frame[1]
        advance()
        return .1
    except Exception:
        report.update(status='failed', failed_step=steps[step], error=traceback.format_exc())
        g['persist']()
        return .1


bpy.app.timers.register(g['heartbeat'], first_interval=.05, persistent=True)
bpy.app.timers.register(tick, first_interval=.2, persistent=True)

#!/usr/bin/env python3
"""Run six ordinary engine round trips in one installed Blender UI process.

Uses a disposable unsaved startup scene, normal user profile, and read-only
runtime observations. Never calls a renderer callback, reconnect or recovery.
"""
from __future__ import annotations
import argparse
import faulthandler
import hashlib
import importlib
import json
import os
from pathlib import Path
import socket
import shutil
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
from renderer_switch_oracle import baseline_reproduced, count_write_rejections, first_runtime_error, fresh_frame, sidecar_failed, validate


def write(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + '\n', encoding='utf-8')


def process_table():
    """Read PID/parent/name without third-party packages or touching workers."""
    if os.name != 'nt':
        result = {}
        for directory in Path('/proc').glob('[0-9]*'):
            try:
                stat = (directory / 'stat').read_text()
                tail = stat[stat.rindex(')')+2:].split()
                result[int(directory.name)] = (int(tail[1]), stat[stat.index('(')+1:stat.rindex(')')])
            except (OSError, ValueError):
                continue
        return result
    import ctypes
    from ctypes import wintypes as w
    class Entry(ctypes.Structure):
        _fields_ = [('size', w.DWORD), ('usage', w.DWORD), ('pid', w.DWORD),
                    ('heap', ctypes.c_size_t), ('module', w.DWORD), ('threads', w.DWORD),
                    ('parent', w.DWORD), ('priority', w.LONG), ('flags', w.DWORD),
                    ('exe', w.WCHAR * 260)]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateToolhelp32Snapshot.restype = w.HANDLE
    kernel.Process32FirstW.argtypes = [w.HANDLE, ctypes.POINTER(Entry)]
    kernel.Process32NextW.argtypes = [w.HANDLE, ctypes.POINTER(Entry)]
    kernel.CloseHandle.argtypes = [w.HANDLE]
    handle = kernel.CreateToolhelp32Snapshot(2, 0)
    if handle == ctypes.c_void_p(-1).value:
        raise OSError(ctypes.get_last_error(), 'process snapshot failed')
    entry = Entry()
    entry.size = ctypes.sizeof(entry)
    result = {}
    try:
        more = kernel.Process32FirstW(handle, ctypes.byref(entry))
        while more:
            result[entry.pid] = (entry.parent, entry.exe)
            more = kernel.Process32NextW(handle, ctypes.byref(entry))
    finally:
        kernel.CloseHandle(handle)
    return result


def process_identity(pid):
    """Executable image and creation identity from the actual owned process."""
    if os.name != 'nt':
        directory = Path('/proc')/str(pid)
        stat = (directory/'stat').read_text()
        return {'pid': pid, 'executable': str((directory/'exe').resolve(strict=True)),
                'created': stat[stat.rindex(')')+2:].split()[19]}
    import ctypes
    from ctypes import wintypes as w
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
    kernel.OpenProcess.restype = w.HANDLE
    kernel.CloseHandle.argtypes = [w.HANDLE]
    kernel.QueryFullProcessImageNameW.argtypes = [w.HANDLE, w.DWORD, w.LPWSTR, ctypes.POINTER(w.DWORD)]
    kernel.GetProcessTimes.argtypes = [w.HANDLE] + [ctypes.POINTER(w.FILETIME)]*4
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        raise OSError(ctypes.get_last_error(), 'Cannot inspect owned process')
    try:
        size = w.DWORD(32768)
        path = ctypes.create_unicode_buffer(size.value)
        if not kernel.QueryFullProcessImageNameW(handle, 0, path, ctypes.byref(size)):
            raise OSError(ctypes.get_last_error(), 'Cannot read executable image')
        created, exited, cpu, user = (w.FILETIME() for _ in range(4))
        if not kernel.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited), ctypes.byref(cpu), ctypes.byref(user)):
            raise OSError(ctypes.get_last_error(), 'Cannot read process creation time')
        return {'pid': pid, 'executable': path.value,
                'created': (created.dwHighDateTime << 32) | created.dwLowDateTime}
    finally:
        kernel.CloseHandle(handle)


def live_owned(owned, read_identity=process_identity):
    live = {}
    for pid, expected in owned.items():
        try:
            actual = read_identity(pid)
            if actual == expected:
                live[pid] = actual
        except (OSError, ProcessLookupError):
            pass
    return live


def discover_owned(table, root, owned, read_identity=process_identity):
    """A recycled PID cannot become a parent or inherit an old ownership claim."""
    live = live_owned({root['pid']: root, **owned}, read_identity)
    for _ in range(4):
        for pid, (parent, _) in table.items():
            if parent not in live or pid in owned or pid == root['pid']:
                continue
            try:
                child = read_identity(pid)
                if read_identity(parent) == live[parent] and int(child['created']) >= int(live[parent]['created']):
                    owned[pid] = child
                    live[pid] = child
            except (OSError, ProcessLookupError):
                continue
    live.pop(root['pid'], None)
    return live


def changed_worker_identities(workers, owned, read_identity=process_identity):
    changed = []
    for pid in workers:
        try:
            actual = read_identity(pid)
            if actual != owned[pid]:
                changed.append({'expected': owned[pid], 'observed': actual})
        except (OSError, ProcessLookupError):
            pass
    return changed


def _linux_pidfd_syscall(number, *arguments):
    """Blender Python may omit pidfd wrappers even on a capable Linux kernel."""
    import ctypes
    import platform
    if sys.platform != 'linux' or platform.machine().lower() != 'x86_64':
        raise OSError('pidfd wrappers unavailable for this platform')
    # Linux x86_64 UAPI: __NR_pidfd_open=434, __NR_pidfd_send_signal=424.
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(ctypes.c_long(number), *arguments)
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


def terminate_owned(expected):
    """Bind termination to an OS handle, checking creation before signalling."""
    pid = expected['pid']
    if os.name != 'nt':
        import signal
        import ctypes
        handle = (os.pidfd_open(pid) if hasattr(os, 'pidfd_open') else
                  _linux_pidfd_syscall(434, ctypes.c_int(pid), ctypes.c_uint(0)))
        try:
            if process_identity(pid) == expected:
                if hasattr(signal, 'pidfd_send_signal'):
                    signal.pidfd_send_signal(handle, signal.SIGTERM)
                else:
                    _linux_pidfd_syscall(424, ctypes.c_int(handle), ctypes.c_int(signal.SIGTERM),
                                        ctypes.c_void_p(), ctypes.c_uint(0))
        finally:
            os.close(handle)
        return
    import ctypes
    from ctypes import wintypes as w
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
    kernel.OpenProcess.restype = w.HANDLE
    kernel.CloseHandle.argtypes = [w.HANDLE]
    kernel.GetProcessTimes.argtypes = [w.HANDLE] + [ctypes.POINTER(w.FILETIME)]*4
    kernel.TerminateProcess.argtypes = [w.HANDLE, w.UINT]
    handle = kernel.OpenProcess(0x1001, False, pid)
    if not handle:
        return
    try:
        created, exited, cpu, user = (w.FILETIME() for _ in range(4))
        if kernel.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited), ctypes.byref(cpu), ctypes.byref(user)):
            if (created.dwHighDateTime << 32) | created.dwLowDateTime == expected['created']:
                kernel.TerminateProcess(handle, 1)
    finally:
        kernel.CloseHandle(handle)


def inside(args):
    import bpy
    out = args.output
    addon = importlib.import_module(args.installed_addon + '.ovrtx_blender_example')
    engine = importlib.import_module(addon.__name__ + '.engine')
    services = importlib.import_module(addon.__name__ + '.runtime_services')
    record = {'schema_version': 1, 'steps': [], 'switches': [], 'failed_slots': 0,
              'first_failure': None, 'teardown': {}, 'blender_pid': os.getpid()}
    state = {'phase': 'startup', 'since': time.monotonic(), 'cycle': 0,
             'boundary_ns': time.perf_counter_ns(), 'name': 'initial'}
    pixels_seen = {}
    identity = {}
    dumped_phases = set()

    def emit(event, **value):
        with (out / 'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'event': event, 'at_ns': time.perf_counter_ns(), **value}, default=str) + '\n')

    def views():
        return [(w, a) for w in bpy.context.window_manager.windows for a in w.screen.areas if a.type == 'VIEW_3D']

    def redraw():
        bpy.context.view_layer.update()
        for _, area in views():
            area.tag_redraw()

    def observe():
        found, failures, engines = [], [], []
        # References last for this call only; keeping wrappers across switches
        # changes Blender's teardown and invalidates the test.
        for instance in list(engine._ACTIVE_VIEWPORT_ENGINES):
            try:
                if not instance.as_pointer():
                    continue
                slot = instance._frame_slot.peek_latest()
                if slot is not None and slot.status == 'failed':
                    failures.append(slot.detail)
                loop = instance._render_loop
                controller = instance._ovrtx_session_controller
                engines.append({'engine_id': id(instance), 'loop': loop.diagnostics() if loop else {},
                                'controller_id': id(controller) if controller else None,
                                'simulation_id': getattr(controller, '_simulation_id', None),
                                'slot': None if slot is None else {'status': slot.status, 'detail': slot.detail}})
                frame = instance._presented_frame
                if frame is None or frame.render_result is None:
                    continue
                snapshot = getattr(loop, '_snapshot', None)
                projection = getattr(snapshot, 'camera_projection', None)
                result = frame.render_result
                item = {'engine_id': id(instance), 'controller_id': id(controller) if controller else None,
                        'simulation_id': getattr(controller, '_simulation_id', None),
                        'session_revision': getattr(controller, '_session_revision', None),
                        'publication': frame.publication_index, 'published_ns': frame.published_monotonic_ns,
                        'render_started_ns': frame.timing_marks.get('render_call_started_monotonic_ns', 0),
                        'status': frame.status, 'width': result.width, 'height': result.height,
                        'simulation_time_ns': result.simulation_time_ns,
                        'output_time_ns': result.render_output_simulation_time_ns,
                        'samples': frame.completed_samples,
                        'snapshot_matches': snapshot is not None and snapshot.key == frame.snapshot_key,
                        'projection_clip': getattr(projection, 'clipping_range', None),
                        'loop': loop.diagnostics() if loop else {},
                        'snapshot_key': repr(frame.snapshot_key)}
                # Copy pixels only for accepted candidates, before releasing frame.
                if fresh_frame(item, state['boundary_ns']):
                    import numpy as np
                    raw = bytes(result.rgba8)
                    array = np.frombuffer(raw, dtype=np.uint8).reshape(result.height, result.width, 4)[:, :, :3]/255.0
                    tiles = [float(array[y*result.height//8:(y+1)*result.height//8,
                                              x*result.width//8:(x+1)*result.width//8].mean())
                             for y in range(8) for x in range(8)]
                    item['pixels'] = {'sha256': hashlib.sha256(raw).hexdigest(), 'tiles': tiles}
                    if state['name'].startswith('clip-'):
                        pixels_seen[id(instance)] = (result.width, result.height, raw)
                found.append(item)
            except ReferenceError:
                continue
        sidecars = []
        for sidecar in tuple(engine._ENGINE_RUNTIMES.values()):
            thread = sidecar.get('render_thread')
            sidecars.append({'thread': thread.diagnostics() if thread else None,
                             'teardown': sidecar.get('teardown_state'),
                             'stop_pending': sidecar.get('stop_pending'),
                             'stop_outcome': sidecar.get('stop_outcome'),
                             'stop_caller_ident': sidecar.get('stop_caller_ident'),
                             'stop_confirmed': sidecar.get('stop_confirmed')})
        failed_slots = len(failures)
        for item in sidecars:
            if sidecar_failed(item):
                failures.append('viewport_thread_failure: ' + str(item))
        # Freeze data-only diagnostics now. A later teardown must not mutate
        # the historical observation through a shared errors list/dictionary.
        return json.loads(json.dumps(
            {'frames': found, 'engines': engines, 'failures': failures,
             'failed_slots': failed_slots, 'sidecars': sidecars,
             'services': services.owner.diagnostics()}, default=str))

    def wait_frame(name):
        state.update(phase='frame', name=name, since=time.monotonic(), boundary_ns=time.perf_counter_ns())
        clips = []
        for _, area in views():
            space = area.spaces.active
            source = bpy.context.scene.camera.data if space.region_3d.view_perspective == 'CAMERA' else space
            clips.append([source.clip_start, source.clip_end])
        state['expected_clip'] = clips
        redraw()
        emit('request', name=name, cycle=state['cycle'], panes=len(views()), boundary_ns=state['boundary_ns'])

    def teardown(failure=None):
        if failure and record['first_failure'] is None:
            record['first_failure'] = failure
        state.update(phase='teardown', since=time.monotonic())
        bpy.context.scene.render.engine = 'BLENDER_WORKBENCH'
        for _, area in views():
            area.spaces.active.shading.type = 'SOLID'
        redraw()
        write(out / 'ui-result.json', record)

    def navigate():
        cycle = state['cycle']
        mode = 'CAMERA' if cycle % 2 else 'PERSP'
        camera = bpy.context.scene.camera
        camera.location.x += 0.06
        camera.data.clip_start, camera.data.clip_end = (0.125 + cycle * 0.01, 250 + cycle * 10)
        for _, area in views():
            space = area.spaces.active
            space.region_3d.view_perspective = mode
            if mode == 'PERSP':
                space.region_3d.view_rotation = camera.rotation_euler.to_quaternion()
                space.region_3d.view_location = (0.06 * cycle, 0, 0)
                space.region_3d.view_distance = 10 + cycle * 0.1
            space.clip_start, space.clip_end = (0.125 + cycle * 0.01, 250 + cycle * 10)
        emit('navigation', mode=mode, camera_clip=[camera.data.clip_start, camera.data.clip_end])
        wait_frame(f'navigate-{cycle}')

    def tick():
        nonlocal identity
        try:
            phase = state['phase']
            now = time.monotonic()
            if phase == 'done':
                return None
            if phase == 'startup':
                if bpy.app.background or args.installed_addon not in bpy.context.preferences.addons:
                    raise RuntimeError('Requires an enabled installed addon in an interactive normal profile')
                runtime = addon.runtime_bundle_status()
                if runtime.get('state') != 'ready':
                    raise RuntimeError('Installed runtime not ready: ' + str(runtime))
                if services.owner.diagnostics()['status'] != 'ready':
                    if now-state['since'] > args.frame_timeout:
                        raise TimeoutError('service startup')
                    return .5
                if bpy.data.filepath or bpy.context.scene.camera is None:
                    raise RuntimeError('Requires disposable unsaved startup scene with camera')
                identity = {'blender': bpy.app.version_string, 'binary': bpy.app.binary_path,
                            'binary_sha256': hashlib.sha256(Path(bpy.app.binary_path).read_bytes()).hexdigest(),
                            'build_hash': bpy.app.build_hash.decode(), 'runtime': runtime,
                            'harness': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                        (Path(__file__), Path(__file__).with_name('renderer_switch_oracle.py'))},
                            'source': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in Path(addon.__file__).parent.glob('*.py')},
                            'native': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in Path(runtime['current_root']).rglob('*')
                                       if p.is_file() and (p.suffix in ('.pyd', '.so') or p.name in ('ovrtx-bridge-server.exe', 'ovrtx-bridge-server'))}}
                write(out / 'identity.json', identity)
                scene = bpy.context.scene
                scene.ovrtx_example.min_samples = 1
                scene.ovrtx_example.max_samples = 1024
                scene.render.engine = 'OVRTX_EXAMPLE'
                for _, area in views():
                    area.spaces.active.region_3d.view_perspective = 'CAMERA'
                    area.spaces.active.shading.type = 'RENDERED'
                wait_frame('initial')
                return .25
            observation = observe()
            emit('observation', phase=phase, name=state['name'], **observation)
            phase_key = (phase, state['name'])
            if now-state['since'] > 30 and phase_key not in dumped_phases:
                dumped_phases.add(phase_key)
                with (out/'python-stacks.log').open('a', encoding='utf-8') as trace:
                    trace.write(f'phase={phase} step={state["name"]}\n')
                    trace.flush()
                    faulthandler.dump_traceback(file=trace, all_threads=True)
            if observation['failures']:
                record['failed_slots'] += observation['failed_slots']
                failure = {'kind': 'failed_slot' if observation['failed_slots'] else 'viewport_thread_failure',
                           'name': state['name'], 'phase': phase, 'errors': observation['failures']}
                if record['first_failure'] is None:
                    record['first_failure'] = failure
                if phase != 'teardown':
                    teardown()
                    return .25
            if phase == 'teardown':
                alive = [s for s in observation['sidecars'] if s['thread'] and s['thread'].get('alive')]
                if not alive or now-state['since'] > 60:
                    record['teardown']['threads_stopped'] = not alive
                    record['teardown']['sidecars'] = observation['sidecars']
                    write(out / 'ui-result.json', record)
                    state['phase'] = 'done'
                    bpy.ops.wm.quit_blender()
                    return None
            elif phase == 'frame':
                frames = [f for f in observation['frames'] if fresh_frame(f, state['boundary_ns'])]
                if state['name'].startswith('clip-'):
                    # Compare refined output, allowing temporal reconstruction
                    # to settle before concluding whether clipping is honored.
                    frames = [f for f in frames if f['samples'] >= 32]
                if len(frames) >= len(views()):
                    if state['name'] == 'initial':
                        loaded = []
                        for name, module in tuple(sys.modules.items()):
                            file = getattr(module, '__file__', None)
                            if file and 'ovrtx_bridge_client' in name and Path(file).suffix in ('.pyd', '.so'):
                                path = Path(file).resolve()
                                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                                if not path.is_relative_to(Path(identity['runtime']['current_root']).resolve()):
                                    raise RuntimeError('Loaded native extension is outside authenticated installed runtime')
                                if digest != identity['native'].get(path.name):
                                    raise RuntimeError('Loaded native extension differs from installed package identity')
                                loaded.append({'module': name, 'path': str(path), 'sha256': digest})
                        if not loaded:
                            raise RuntimeError('Actual loaded native client extension was not found')
                        identity['loaded_native_clients'] = loaded
                        write(out/'identity.json', identity)
                    step = {'name': state['name'], 'boundary_ns': state['boundary_ns'],
                            'expected_clip': state['expected_clip'],
                            'panes': len(views()), 'frames': frames, 'view_modes':
                            [a.spaces.active.region_3d.view_perspective for _, a in views()]}
                    record['steps'].append(step)
                    emit('accepted', **step)
                    if state['name'].startswith('clip-'):
                        from importlib import import_module
                        png = import_module(addon.__name__ + '.shared_stage_composition').write_rgba_png
                        width, height, raw = pixels_seen[frames[0]['engine_id']]
                        png(out / (state['name']+'.png'), width, height, raw)
                    name = state['name']
                    if name.startswith('return-'):
                        navigate()
                    elif name == 'navigate-1':
                        wait_frame('clip-visible')
                        bpy.context.scene.camera.location.x += .00001
                        redraw()
                    elif name == 'clip-visible':
                        bpy.context.scene.camera.data.clip_end = .5
                        wait_frame('clip-hidden')
                    elif name == 'clip-hidden':
                        bpy.context.scene.camera.data.clip_end = 260
                        wait_frame('clip-restored')
                    elif name == 'navigate-2':
                        window, area = views()[0]
                        with bpy.context.temp_override(window=window, area=area):
                            bpy.ops.screen.area_split(direction='VERTICAL', factor=.5)
                        for _, area in views():
                            area.spaces.active.shading.type = 'RENDERED'
                        wait_frame('two-panes')
                    elif name == 'navigate-6':
                        window, area = views()[-1]
                        with bpy.context.temp_override(window=window, area=area):
                            bpy.ops.screen.area_close()
                        # Require new work in the surviving pane after closure.
                        bpy.context.scene.camera.location.x += .1
                        for _, area in views():
                            area.spaces.active.region_3d.view_distance += .1
                        wait_frame('pane-closed')
                    elif name == 'pane-closed':
                        teardown()
                    else:
                        state.update(phase='active', since=now)
                elif now-state['since'] > args.frame_timeout:
                    teardown({'kind': 'frame_timeout', 'name': state['name'], 'observation': observation})
            elif phase == 'active':
                active = any(f['loop'].get('running') and not f['loop'].get('final_view_refined')
                             and f['loop'].get('completed_samples', 0) > 0 for f in observation['frames'])
                if active:
                    alternative = ('BLENDER_EEVEE', 'BLENDER_WORKBENCH')[state['cycle'] % 2]
                    record['switches'].append({'alternative': alternative, 'active_work': active,
                                               'at_ns': time.perf_counter_ns(), 'observation': observation})
                    bpy.context.scene.render.engine = alternative
                    redraw()
                    state.update(phase='away', since=time.monotonic())
                elif now-state['since'] > 5:
                    # Ordinary navigation schedules a new active refinement.
                    bpy.context.scene.camera.location.x += .001
                    for _, area in views():
                        area.spaces.active.region_3d.view_distance += .001
                    redraw()
                    if now-state['since'] > args.frame_timeout:
                        raise TimeoutError('no active refinement before switch')
            elif phase == 'away' and now-state['since'] > 5:
                state['cycle'] += 1
                bpy.context.scene.render.engine = 'OVRTX_EXAMPLE'
                wait_frame(f'return-{state["cycle"]}')
            return .25
        except Exception:
            failure = {'kind': 'harness_exception', 'traceback': traceback.format_exc()}
            emit('exception', **failure)
            teardown(failure)
            return .25
    bpy.app.timers.register(tick, first_interval=2)


def outer(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    (out/'tmp').mkdir()
    for source in (Path(__file__).resolve(), Path(__file__).with_name('renderer_switch_oracle.py').resolve()):
        shutil.copyfile(source, out/source.name)
    # Refuse an occupied default service; never attach to or kill another run.
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1', 50051)) == 0:
            raise RuntimeError('OVRTX port is occupied')
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
                       OV_BLENDER_EXAMPLE_WORKER_LOG=str(out/'worker.log'),
                       OVRTX_EXAMPLE_WORKER_LOG=str(out/'worker.log'),
                       OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR=str(out/'viewport-work'))
    environment.update(TEMP=str(out/'tmp'), TMP=str(out/'tmp'), TMPDIR=str(out/'tmp'))
    command = [args.blender_command, '--window-geometry', '30', '30', '960', '720',
               '--python', str(out/Path(__file__).name), '--', '--inside-blender',
               '--output', str(out), '--installed-addon', args.installed_addon,
               '--frame-timeout', str(args.frame_timeout)]
    startup = None
    if os.name == 'nt':
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = 0
    workers, owned = set(), {}
    worker_observations = []
    worker_identities = {}
    worker_identity_changes = []
    forced = False
    with (out/'blender.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment, startupinfo=startup)
        root_identity = process_identity(process.pid)
        write(out/'launch.json', {'command': command, 'pid': process.pid})
        start = time.monotonic()
        while process.poll() is None:
            table = process_table()
            for change in changed_worker_identities(workers, owned):
                if change not in worker_identity_changes:
                    worker_identity_changes.append(change)
            live = discover_owned(table, root_identity, owned)
            workers.update(pid for pid in live if pid in table and table[pid][1].startswith('ovrtx-bridge'))
            live_workers = sorted(workers.intersection(live))
            for pid in live_workers:
                details = dict(live[pid])
                key = (pid, details['created'])
                if key not in worker_identities:
                    details['sha256'] = hashlib.sha256(Path(details['executable']).read_bytes()).hexdigest()
                    worker_identities[key] = details
            worker_observations.append({'at_ns': time.perf_counter_ns(), 'pids': live_workers})
            if time.monotonic()-start > args.timeout:
                forced = True
                process.terminate()
                break
            time.sleep(.5)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            forced = True
            process.kill()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass  # Preserve failed exit evidence and still inspect descendants.
    # Give the worker its ordinary process-exit grace period; retain orphan
    # evidence BEFORE cleaning only descendants proven owned by this launch.
    deadline = time.monotonic()+30
    while time.monotonic() < deadline and live_owned(owned):
        time.sleep(.5)
    orphans = sorted(live_owned(owned))
    for pid in orphans:
        try:
            terminate_owned(owned[pid])
        except (OSError, ProcessLookupError):
            pass
    path = out/'ui-result.json'
    record = json.loads(path.read_text()) if path.exists() else {'schema_version': 1, 'first_failure': 'UI result missing'}
    record['workers'] = sorted(workers)
    record['worker_observations'] = worker_observations
    record['worker_identities'] = list(worker_identities.values())
    record['worker_identity_changes'] = worker_identity_changes
    record['owned_process_identities'] = list(owned.values())
    identity_path = out/'identity.json'
    identity = json.loads(identity_path.read_text()) if identity_path.exists() else {}
    record['loaded_native_clients'] = identity.get('loaded_native_clients', [])
    installed_native = identity.get('native', {})
    record['worker_images_match_installed'] = bool(worker_identities) and all(
        item['sha256'] == installed_native.get(Path(item['executable']).name)
        for item in worker_identities.values())
    record.setdefault('teardown', {}).update(ordinary_exit=not forced and process.returncode == 0,
                                            blender_exit_code=process.returncode, orphans=orphans)
    worker_log = (out/'worker.log').read_text(errors='replace') if (out/'worker.log').exists() else ''
    record['aborted_errors'] = count_write_rejections(worker_log)
    blender_log = (out/'blender.log').read_text(errors='replace')
    record['teardown']['worker_graceful'] = ('Shutdown complete.' in worker_log
                                            and 'forcing owned process termination' not in blender_log
                                            and 'worker exited abnormally' not in blender_log)
    original = first_runtime_error(blender_log)
    record['first_underlying_error'] = original
    if original and record.get('first_failure') is None:
        record['first_failure'] = {'kind': 'runtime_error', 'error': original}
    record['errors'] = validate(record)
    record['status'] = 'passed' if not record['errors'] else 'failed'
    record['baseline_reproduced'] = baseline_reproduced(record)
    write(out/'result.json', record)
    print(json.dumps({'status': record['status'], 'errors': record['errors'], 'output': str(out)}))
    return 0 if (record['baseline_reproduced'] if args.expect_baseline_failure else record['status'] == 'passed') else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blender-command', default='blender')
    parser.add_argument('--installed-addon', default='bl_ext.user_default.ovrtx_blender_example')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--frame-timeout', default=180., type=float)
    parser.add_argument('--timeout', default=1800., type=float)
    parser.add_argument('--inside-blender', action='store_true')
    parser.add_argument('--expect-baseline-failure', action='store_true')
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else None)
    args.output = args.output.resolve()
    return inside(args) if args.inside_blender else outer(args)


if __name__ == '__main__':
    if '--inside-blender' in sys.argv:
        main()
    else:
        raise SystemExit(main())

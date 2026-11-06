"""Pure acceptance oracle for the installed interactive renderer switch probe."""
from __future__ import annotations
import math


def first_runtime_error(log):
    """The slot can be overwritten between UI polls; ordered raw log wins."""
    for line in log.splitlines():
        if 'RenderClientError:' in line or 'SchedulerError:' in line:
            return line[line.find('RenderClientError:'):] if 'RenderClientError:' in line else line
    return None


def count_write_rejections(log):
    return sum('ABORTED' in line or 'already been written' in line for line in log.splitlines())


def sidecar_failed(item):
    thread = item.get('thread') or {}
    return bool(thread.get('leaked_thread') or thread.get('failure')
                or (item.get('teardown') or {}).get('errors')
                or (item.get('stop_confirmed') is False and not thread.get('alive')))


def fresh_frame(frame, boundary_ns):
    return bool(frame and frame.get('status') == 'frame'
                and frame.get('published_ns', 0) > boundary_ns
                and frame.get('render_started_ns', 0) > boundary_ns
                and frame.get('snapshot_matches', False)
                and frame.get('width', 0) > 0 and frame.get('height', 0) > 0)


def clipping_errors(visible, clipped, restored):
    """Compare coarse RGB tiles; byte hashes alone accept noise as clipping."""
    a, b, c = (item['pixels']['tiles'] for item in (visible, clipped, restored))
    if not a or len(a) != len(b) or len(a) != len(c):
        return ['clipping_pixel_shape']
    difference = lambda x, y: sum(abs(i-j) for i, j in zip(x, y))/len(x)
    removed, recovered = difference(a, b), difference(a, c)
    return [] if removed > 0.015 and recovered < removed * 0.5 else ['clipping_not_observable']


def baseline_reproduced(record):
    """Startup and harness failures are never an expected negative control."""
    failure = record.get('first_failure') or {}
    original = record.get('first_underlying_error') or ''
    steps = record.get('steps', [])
    return bool(steps and steps[0]['name'] == 'initial'
                and all(fresh_frame(f, steps[0]['boundary_ns']) for f in steps[0]['frames'])
                and failure.get('kind') == 'failed_slot' and failure.get('name') == 'return-1'
                and 'attribute_values.clippingRange.type must be' in original
                and record.get('aborted_errors', 0) > 0
                and len(record.get('workers', [])) == 1
                and record.get('teardown', {}).get('ordinary_exit')
                and not record.get('teardown', {}).get('orphans'))


def validate(record):
    errors = []
    if record.get('schema_version') != 1:
        errors.append('schema_version')
    if record.get('first_failure'):
        errors.append('first_failure')
    steps = record.get('steps', [])
    names = [s['name'] for s in steps]
    required = ['initial']
    for cycle in range(1, 7):
        required += [f'return-{cycle}', f'navigate-{cycle}']
    required += ['clip-visible', 'clip-hidden', 'clip-restored', 'two-panes', 'pane-closed']
    if set(required) - set(names):
        errors.append('missing_steps')
    native_times = {}
    for step in steps:
        name = step['name']
        expected_panes = 2 if name == 'two-panes' or name in {
            f'{action}-{cycle}' for action in ('return', 'navigate') for cycle in range(3, 7)} else 1
        if step.get('panes') != expected_panes:
            errors.append('scenario_pane_count')
        ids = [frame.get('engine_id') for frame in step.get('frames', [])]
        if len(set(ids)) != len(ids) or None in ids:
            errors.append('duplicate_or_missing_pane_identity')
        modes = step.get('view_modes', [])
        if len(modes) != expected_panes:
            errors.append('view_mode_coverage')
        if name.startswith('navigate-'):
            expected_mode = 'CAMERA' if int(name.split('-')[1]) % 2 else 'PERSP'
            if modes != [expected_mode]*expected_panes:
                errors.append('view_mode_coverage')
        if len(step.get('frames', [])) != step.get('panes'):
            errors.append('pane_frame_count')
        if any(not fresh_frame(frame, step['boundary_ns']) for frame in step.get('frames', [])):
            errors.append('stale_frame')
        # Wrapper publication time cannot turn cached native output into a
        # new render. A recreated simulation has its own time domain.
        step_times = {}
        for frame in step.get('frames', []):
            identity = (frame.get('controller_id'), frame.get('simulation_id'), frame.get('session_revision'))
            output = frame.get('output_time_ns')
            if output is None or output <= native_times.get(identity, -1):
                errors.append('stale_native_output')
            elif identity not in step_times or output > step_times[identity]:
                step_times[identity] = output
        native_times.update(step_times)
        expected = step.get('expected_clip', [])
        for frame in step.get('frames', []):
            actual = frame.get('projection_clip')
            if not actual or len(actual) != 2 or not all(math.isfinite(a) for a in actual) or not any(all(abs(a-b) <= max(1e-5, abs(b)*.005)
                                        for a, b in zip(actual, pair)) for pair in expected):
                errors.append('camera_clip_mismatch')
            if name.startswith('clip-') and frame.get('samples', 0) < 32:
                errors.append('clipping_not_refined')
        if step.get('panes', 0) > 1:
            controllers = {f.get('controller_id') for f in step['frames']}
            simulations = {f.get('simulation_id') for f in step['frames']}
            if len(controllers) != 1 or None in controllers or len(simulations) != 1 or None in simulations:
                errors.append('unshared_runtime')
    trips = record.get('switches', [])
    for engine in ('BLENDER_EEVEE', 'BLENDER_WORKBENCH'):
        if sum(s.get('alternative') == engine for s in trips) != 3:
            errors.append('round_trip_count')
    if len(trips) != 6 or not all(s.get('active_work') for s in trips):
        errors.append('switch_without_active_work')
    pixels = {s['name']: s for s in steps}
    if 'navigate-6' in pixels and 'pane-closed' in pixels:
        previous = {f.get('controller_id') for f in pixels['navigate-6']['frames']}
        if any(f.get('controller_id') not in previous for f in pixels['pane-closed']['frames']):
            errors.append('pane_close_lost_shared_controller')
    if all(n in pixels for n in ('clip-visible', 'clip-hidden', 'clip-restored')):
        errors += clipping_errors(*(pixels[n]['frames'][0] for n in ('clip-visible', 'clip-hidden', 'clip-restored')))
    workers = record.get('workers', [])
    if len(workers) != 1 or record.get('worker_identity_changes'):
        errors.append('worker_restart_or_missing')
    identities = record.get('worker_identities', [])
    if len(identities) != 1 or not identities[0].get('sha256') or identities[0].get('pid') not in workers:
        errors.append('worker_image_identity_unconfirmed')
    if not record.get('worker_images_match_installed') or not record.get('loaded_native_clients'):
        errors.append('loaded_package_identity_unconfirmed')
    if record.get('failed_slots', 0):
        errors.append('failed_slot')
    if record.get('aborted_errors', 0):
        errors.append('aborted_write')
    teardown = record.get('teardown', {})
    if not teardown.get('ordinary_exit') or teardown.get('orphans') or not teardown.get('threads_stopped'):
        errors.append('unclean_teardown')
    for sidecar in teardown.get('sidecars', []):
        thread = sidecar.get('thread') or {}
        cleanup = sidecar.get('teardown') or {}
        if (thread.get('failure') or thread.get('leaked_thread') or thread.get('alive')
                or thread.get('status') == 'failed' or cleanup.get('errors')
                or sidecar.get('stop_confirmed') is False
                or (thread and cleanup.get('ran') is False)):
            errors.append('unclean_teardown')
    if not teardown.get('worker_graceful'):
        errors.append('worker_shutdown_unconfirmed')
    return sorted(set(errors))

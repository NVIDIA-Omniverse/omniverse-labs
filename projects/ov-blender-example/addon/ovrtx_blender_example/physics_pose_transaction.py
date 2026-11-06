# SPDX-License-Identifier: Apache-2.0
"""Writable main-loop admission of one complete image's pose presentation."""
import time
from . import interactive_operator_state as operator_state
from .physics_pose_presentation import ResolutionIndex


class _MeasuredEvaluation:
    def __init__(self, view_layer):
        self.view_layer = view_layer
        self.count = 0
        self.elapsed_ns = 0

    def update(self):
        start = time.perf_counter_ns()
        try:
            self.view_layer.update()
        finally:
            self.count += 1
            self.elapsed_ns += time.perf_counter_ns() - start


def _containers(value):
    if isinstance(value, dict):
        return {k: _containers(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_containers(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_containers(v) for v in value)
    return value  # retain RNA identities; never duplicate Blender datablocks


def _snapshot(scene, pending, lock, index):
    channels = {}
    paths = pending['poses_by_path']
    targets = pending.get('presentation_targets', ())
    for obj in scene.objects:
        path = operator_state._pose_object_path(obj, pending.get('object_paths_by_session_uid'))
        joint = getattr(getattr(getattr(obj, 'ov', None), 'physics', None), 'joint', None)
        if path in paths or (joint is not None and joint.enabled and joint.persistent_id):
            channels[id(obj)] = obj
    if targets:
        from .physics_pose_presentation import resolve
        for target in targets:
            if target.body_path in paths and target.enabled and target.role == 'DYNAMIC':
                obj, channel = resolve(scene, target, index)
                channels[id(channel)] = channel
    for record in lock._presentation_records.values():
        channels[id(record['channel'])] = record['channel']
    for record in lock._records.values():
        channels[id(record['object'])] = record['object']
    for obj, basis in lock._legacy_initial_bases.values():
        channels[id(obj)] = obj
    rows = []
    names = ('location', 'rotation_euler', 'rotation_quaternion', 'rotation_axis_angle',
             'scale', 'lock_location', 'lock_rotation', 'lock_rotation_w', 'lock_rotations_4d')
    for channel in channels.values():
        matrix_name = 'matrix_basis' if hasattr(channel, 'matrix_basis') else 'matrix_world'
        matrix = getattr(channel, matrix_name).copy()
        values = {}
        for name in names:
            if hasattr(channel, name):
                value = getattr(channel, name)
                values[name] = tuple(value) if hasattr(value, '__iter__') else value
        rows.append((channel, matrix_name, matrix, values))
    return rows, _containers(lock.__dict__)


def _rollback(rows, state, lock, view_layer):
    errors = []
    with operator_state.suppress_interactive_edit_bridge():
        for channel, matrix_name, matrix, values in rows:
            try:
                setattr(channel, matrix_name, matrix)
                for name, value in values.items():
                    setattr(channel, name, value)
            except Exception as error:
                errors.append(str(error))
        lock.__dict__.clear()
        lock.__dict__.update(state)
        if view_layer is not None:
            view_layer.update()
    if errors:
        raise RuntimeError('Physics presentation rollback failed: ' + '; '.join(errors))


def apply(bpy, scene, pending, lock, *, initial, still_current):
    """All scene scans/evaluation stay in the writable preparation timer."""
    started = time.perf_counter_ns()
    view_layer = getattr(getattr(bpy, 'context', None), 'view_layer', None)
    # Dispatch queued authored edits before taking rollback preimages.
    if view_layer is not None:
        view_layer.update()
    flushed = time.perf_counter_ns()
    if not still_current():
        return {'enabled': False, 'status': 'stale', 'reason': 'stale_frame_presentation'}
    index = ResolutionIndex(scene)
    rows, state = _snapshot(scene, pending, lock, index)
    resolved = time.perf_counter_ns()
    evaluation = _MeasuredEvaluation(view_layer) if view_layer is not None else None
    pending = dict(pending, _presentation_index=index, _authored_edits_flushed=True,
                   _presentation_view_layer=evaluation)
    try:
        if initial and lock.is_active():
            lock.clear(reason='initial_condition_frame', frame1_cleared=True)
        result = operator_state.apply_pending_runtime_pose_mirror(bpy, pending, lock)
        complete = bool(result and result.get('enabled')
            and result.get('status') in {'applied', 'restored'}
            and not result.get('mirror_error_count') and not result.get('missing_object_paths'))
        if not complete or not still_current():
            _rollback(rows, state, lock, view_layer)
            return {'enabled': False, 'status': 'failed', 'reason': 'incomplete_frame_pose_mirror',
                    'mirror_result': result}
        result = dict(result, presentation_transaction_ns=time.perf_counter_ns() - started,
                      presentation_authored_evaluation_ns=flushed - started,
                      presentation_resolution_snapshot_ns=resolved - flushed,
                      presentation_index_object_count=index.object_count,
                      presentation_pose_evaluation_count=evaluation.count if evaluation else 0,
                      presentation_pose_evaluation_ns=evaluation.elapsed_ns if evaluation else 0)
        return result
    except Exception:
        _rollback(rows, state, lock, view_layer)
        raise

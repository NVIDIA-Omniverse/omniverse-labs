"""Pending artist events must dispatch before physics-owned write suppression."""
import sys
from types import SimpleNamespace
from ovrtx_blender_example import interactive_operator_state as state
from ovrtx_blender_example import scene_generation_sessions as sessions
from ovrtx_blender_example import physics_pose_presentation as presentation


def test_poll_preflush_can_release_owner_without_enforcing_stale_records(monkeypatch):
    scene = SimpleNamespace(session_uid=123)
    lock = SimpleNamespace(_presentation_records={'body': object()})
    monkeypatch.setattr(sessions, '_runtimes', {123: SimpleNamespace(playback_lock=lock)})
    observed = []
    def flush():
        observed.append(state.interactive_edit_bridge_suppressed())
        sessions._runtimes.clear()
    bpy = SimpleNamespace(data=SimpleNamespace(scenes=[scene]),
                          context=SimpleNamespace(view_layer=SimpleNamespace(update=flush)))
    monkeypatch.setattr(sessions, 'bpy', bpy)
    monkeypatch.setattr(sessions, '_bake_capture_owner', None)
    monkeypatch.setattr(sessions, '_render_only_physics_current', lambda scene: False)
    monkeypatch.setattr(sessions, 'observe_physics_edits', lambda scene: None)
    monkeypatch.setattr(presentation, 'enforce', lambda lock: (_ for _ in ()).throw(AssertionError('stale owner')))
    sessions._poll_physics_edits()
    assert observed == [False]


def test_poll_suppresses_only_owned_writes(monkeypatch):
    scene = SimpleNamespace(session_uid=123)
    lock = SimpleNamespace(_presentation_records={'body': object()})
    monkeypatch.setattr(sessions, '_runtimes', {123: SimpleNamespace(playback_lock=lock)})
    observed = []
    def flush(): observed.append(('flush', state.interactive_edit_bridge_suppressed()))
    monkeypatch.setattr(sessions, 'bpy', SimpleNamespace(data=SimpleNamespace(scenes=[scene]),
        context=SimpleNamespace(view_layer=SimpleNamespace(update=flush))))
    monkeypatch.setattr(sessions, '_bake_capture_owner', None)
    monkeypatch.setattr(sessions, '_render_only_physics_current', lambda scene: False)
    monkeypatch.setattr(sessions, 'observe_physics_edits', lambda scene: None)
    monkeypatch.setattr(presentation, 'enforce', lambda lock: observed.append(('enforce', state.interactive_edit_bridge_suppressed())))
    sessions._poll_physics_edits()
    assert observed == [('flush', False), ('enforce', True), ('flush', True)]


def test_mirror_rechecks_generation_after_pending_edit_dispatch(monkeypatch):
    monkeypatch.setitem(sys.modules, 'mathutils', SimpleNamespace(Matrix=object, Quaternion=object, Vector=object))
    current = [1]
    observed = []
    scene = SimpleNamespace()
    def flush():
        observed.append(state.interactive_edit_bridge_suppressed())
        current[0] = 2
    module = SimpleNamespace(data=SimpleNamespace(scenes={'scene': scene}),
        context=SimpleNamespace(view_layer=SimpleNamespace(update=flush)))
    monkeypatch.setattr(sessions, 'current_presentation_context', lambda scene: (current[0], 1, ()))
    pending = dict(scene_name='scene', scene_generation_number=1, presentation_epoch=1,
                   poses_by_path={'/Body': {'translate': (0, 0, 0), 'orient': (0, 0, 0, 1)}})
    result = state.apply_pending_runtime_pose_mirror(module, pending, object())
    assert result['status'] == 'stale'
    assert observed == [False]

from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace

import pytest
import ovrtx_blender_example
from test_scheduler_thread_pose_mirror import _load_engine_with_fake_bpy, _tick_result
from ovrtx_blender_example.runtime_scheduler import RuntimeScheduler, RuntimeTickRequest, RuntimeTickResult, RuntimeTickStatus


@pytest.mark.parametrize('failed,written', [(False, False), (False, True), (True, False)])
def test_baked_tick_preserves_view_result_but_does_not_admit_live_physics(failed, written):
    scheduler = RuntimeScheduler()
    scheduler.baked_playback_paused = True
    calls = []
    scheduler._controller_provider = lambda: SimpleNamespace(stop_async_playback=lambda **kw: calls.append(kw))
    original = RuntimeTickResult(status=RuntimeTickStatus.FAILED if failed else RuntimeTickStatus.NOOP,
        enabled=True, values_written=written, should_reset_refinement=written,
        update={'failed': failed, 'values_written': written}, skipped_reason='retained reason')
    scheduler._apply_pending_view_values_locked = lambda updates: original
    result = scheduler.tick_viewport(RuntimeTickRequest('/scene.usda'), ovrtx_updates=object())
    assert result == replace(original, enabled=False,
        status=RuntimeTickStatus.FAILED if failed else RuntimeTickStatus.NOT_ENABLED)
    assert calls == [{'wait': True, 'invalidate': True}]


def test_successful_baked_handoff_retires_only_own_scene_and_stale_timer(monkeypatch):
    module = _load_engine_with_fake_bpy(monkeypatch)
    monkeypatch.setattr(ovrtx_blender_example, 'engine', module, raising=False)
    scene = SimpleNamespace(session_uid=41)
    viewport = module.OvrtxExampleRenderEngine()
    other = module.OvrtxExampleRenderEngine()
    viewport._viewport_scene = scene
    other._viewport_scene = SimpleNamespace(session_uid=42)
    old = SimpleNamespace(runtime_tick_result=_tick_result(module), runtime_tick_timeline=(True, 2, 1))
    old_epoch = viewport._viewport_display_epoch
    identity = viewport._presentation_identity()
    viewport._ready_presentation_frame = (old, identity, {}, {}, False)
    viewport._mirrored_frame = viewport._presented_frame = old
    other._mirrored_frame = other._presented_frame = old
    viewport._queued_display_work = object()
    viewport._queue_frame_presentation(old)
    viewport._pending_tick_absorb = object()
    discarded = []
    monkeypatch.setattr(module._DISPLAY_OVERLAY, 'discard_owner', discarded.append)
    monkeypatch.setattr(module, '_live_viewport_engines', lambda: [viewport, other])
    restored = []
    sessions = module.scene_generation_sessions
    monkeypatch.setattr(sessions, 'restore_physics_presentation', lambda s: restored.append(s))
    future = Future()
    future._ov_baked_scene_uid = 41
    future.set_result(None)
    sessions.finish_baked_playback_pause(scene, future)
    assert restored == [scene] and discarded == [viewport]
    assert viewport._viewport_display_epoch is not old_epoch
    assert viewport._queued_display_work is None and viewport._pending_tick_absorb is None
    assert viewport._ready_presentation_frame is None and viewport._mirrored_frame is None
    assert viewport._presented_frame is None
    assert other._mirrored_frame is old and other._presented_frame is old
    # The already registered old prepare callback must not regain RNA ownership.
    prepare = viewport._prepare_frame_presentation
    monkeypatch.setattr(viewport, '_prepare_frame_presentation', lambda: pytest.fail('stale prepare executed'))
    module.bpy.app.timers.run_pending()
    # A pre-drain publication arriving afterward gets the new display epoch,
    # but baked ownership still prohibits its pose preparation and commit.
    monkeypatch.setattr(viewport, '_prepare_frame_presentation', prepare)
    monkeypatch.setattr(sessions, 'presentation_stopped', lambda s: s is scene)
    viewport._queue_frame_presentation(old)
    module.bpy.app.timers.run_pending()
    assert viewport._ready_presentation_frame is None and viewport._mirrored_frame is None
    baked = SimpleNamespace(runtime_tick_result=replace(old.runtime_tick_result,
        enabled=False, status=RuntimeTickStatus.NOT_ENABLED, physics_pose_set=()), publication_index=20)
    assert viewport._frame_for_presentation(old) is None
    assert not viewport._commit_frame_pose_mirror(old, prepared=(old, identity, {}, {}, False))
    assert viewport._frame_for_presentation(baked) is baked
    assert viewport._commit_frame_pose_mirror(baked)


@pytest.mark.parametrize('state', ['pending', 'failed', 'wrong_scene'])
def test_uncompleted_or_invalid_drain_cannot_restore_or_retire(monkeypatch, state):
    module = _load_engine_with_fake_bpy(monkeypatch)
    sessions = module.scene_generation_sessions
    monkeypatch.setattr(sessions, 'restore_physics_presentation', lambda s: pytest.fail('restored before drain'))
    monkeypatch.setattr(module, 'retire_physics_presentation', lambda s: pytest.fail('retired before drain'))
    future = Future()
    future._ov_baked_scene_uid = 99 if state == 'wrong_scene' else 41
    if state == 'failed':
        future.set_exception(RuntimeError('native drain failed'))
    elif state == 'wrong_scene':
        future.set_result(None)
    with pytest.raises(RuntimeError):
        sessions.finish_baked_playback_pause(SimpleNamespace(session_uid=41), future)

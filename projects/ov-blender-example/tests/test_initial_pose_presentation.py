from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_scheduler_thread_pose_mirror import _load_engine_with_fake_bpy, _tick_result


@pytest.mark.parametrize('playing,steps,stamp,reset,restores', [
    (True, 2, 33333332, False, False),
    (False, 2, 33333332, False, False),
    (False, 2, 33333332, True, False),
    (True, 0, 0, True, False),
    (False, 0, 0, True, True),
    (False, 0, 0, False, True),
])
def test_start_frame_presentation_restores_only_paused_zero_step(monkeypatch, playing, steps, stamp, reset, restores):
    module = _load_engine_with_fake_bpy(monkeypatch)
    engine = module.OvrtxExampleRenderEngine()
    tick = replace(_tick_result(module), step_count=steps, simulation_time_ns=stamp, timeline_reset=reset)
    frame = SimpleNamespace(runtime_tick_result=tick, runtime_tick_timeline=(True, 1, 1),
                            runtime_tick_playing=playing, publication_index=1)
    world = {'pose': 'authored'}
    def prepare(poses, *args, **kwargs):
        return {'poses_by_path': {}, 'lock_runtime_owned': kwargs['lock_runtime_owned']}, {'enabled': True}
    def apply(bpy, pending, lock):
        world['pose'] = 'submitted' if pending['lock_runtime_owned'] else 'authored'
        return {'enabled': True, 'status': 'applied' if pending['lock_runtime_owned'] else 'restored'}
    monkeypatch.setattr(module.operator_state, 'prepare_runtime_pose_mirror', prepare)
    monkeypatch.setattr(module.operator_state, 'apply_pending_runtime_pose_mirror', apply)
    engine._queue_frame_presentation(frame)
    module.bpy.app.timers.run_pending()
    assert world['pose'] == ('authored' if restores else 'submitted')
    assert engine._ready_presentation_frame[-1] is restores
    assert engine._commit_frame_pose_mirror(frame)
    assert engine._runtime_tick_result is tick

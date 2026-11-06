"""Terminal solver failures cannot replay, or starve an independent renderer."""
from dataclasses import replace

import pytest

from ovrtx_blender_example.ovphysx_stage import OvphysxStageController
from ovrtx_blender_example.ovrtx_value_updates import OvrtxSessionUpdatePort
from ovrtx_blender_example.runtime_scheduler import RuntimeScheduler, RuntimeTickStatus
from ovrtx_blender_example.viewport_handoff import FRAME_STATUS_FAILED, FRAME_STATUS_FRAME
from test_runtime_scheduler import _config, _FakePhysicsClient, _FakeRenderClient, _request, _velocity_intent
from test_latest_view_render_loop import _Harness, _snapshot, _transform_intent, _matrix


class RecordingPhysics(_FakePhysicsClient):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.uncertain = False
        self.fail_velocity = False

    def start(self):
        self.calls.append('start')
        return super().start()

    def create_simulation(self):
        self.calls.append('load')
        return super().create_simulation()

    def read_body_states(self, time_ns):
        self.calls.append('read')
        return super().read_body_states(time_ns)

    def advance_and_read_body_states(self, *args):
        self.calls.append('step')
        if self.uncertain:
            raise RuntimeError("Outcome of 'step' is unknown; explicit reset required")
        return super().advance_and_read_body_states(*args)

    def write_body_velocities(self, *args, **kwargs):
        self.calls.append('write_velocities')
        if self.fail_velocity:
            raise RuntimeError("Outcome of 'write_velocities' is unknown")
        return super().write_body_velocities(*args, **kwargs)


def fail_step(controller, physics):
    physics.uncertain = True
    controller.tick()
    assert controller.failed and controller.started
    assert 'unknown' in controller.last_error


@pytest.mark.parametrize('mutation', ['step', 'write_velocities'])
def test_first_failed_mutation_reports_physics_failure_without_replaying(mutation):
    physics = RecordingPhysics()
    controller = OvphysxStageController(_config(max_steps=100), physics_client=physics)
    scheduler = RuntimeScheduler(controller_provider=lambda: controller)
    port = OvrtxSessionUpdatePort(_FakeRenderClient(), 'same-renderer')
    try:
        controller.tick()
        scheduler.tick_viewport(_request(), ovrtx_updates=port, project_complete_pose=True)
        if mutation == 'step':
            physics.uncertain = True
        else:
            physics.fail_velocity = True
            scheduler.submit_edit(_velocity_intent())
        failed = scheduler.tick_viewport(_request(timeline_frame=2), ovrtx_updates=port)
        assert failed.status == RuntimeTickStatus.FAILED and failed.physics_only_failure
        calls = list(physics.calls)
        again = scheduler.tick_viewport(_request(timeline_frame=1), ovrtx_updates=port)
        assert again.status == RuntimeTickStatus.FAILED and again.physics_only_failure
        assert 'unknown' in again.skipped_reason and physics.calls == calls
    finally:
        scheduler.shutdown()
        controller.shutdown()


def test_explicit_reset_can_replace_failed_solver():
    clients, controllers = [], []
    def reset():
        if controllers:
            controllers[-1].shutdown()
        client = RecordingPhysics()
        controller = OvphysxStageController(_config(max_steps=100), physics_client=client)
        controller.tick()
        clients.append(client)
        controllers.append(controller)
        return True
    reset()
    scheduler = RuntimeScheduler(controller_provider=lambda: controllers[-1], controller_reset=reset)
    port = OvrtxSessionUpdatePort(_FakeRenderClient(), 'same-renderer')
    try:
        scheduler.tick_viewport(_request(), ovrtx_updates=port, project_complete_pose=True)
        fail_step(controllers[0], clients[0])
        result = scheduler.tick_viewport(_request(simulation_reset_token=1), ovrtx_updates=port)
        assert len(controllers) == 2 and controllers[0].closed
        assert result.timeline_reset and result.simulation_time_ns == 0
        assert result.status != RuntimeTickStatus.FAILED and not result.physics_only_failure
    finally:
        scheduler.shutdown()
        controllers[-1].shutdown()


@pytest.mark.parametrize('frame,playing', [(1, False), (9, False), (1, True)])
def test_failed_solver_rewind_and_pending_sim_edits_never_replay(frame, playing):
    physics = RecordingPhysics()
    controller = OvphysxStageController(_config(max_steps=100), physics_client=physics)
    resets = []
    scheduler = RuntimeScheduler(controller_provider=lambda: controller,
                                 controller_reset=lambda: resets.append(True))
    port = OvrtxSessionUpdatePort(_FakeRenderClient(), 'same-renderer')
    try:
        controller.tick()
        scheduler.tick_viewport(_request(timeline_frame=4), ovrtx_updates=port, project_complete_pose=True)
        fail_step(controller, physics)
        calls, count, stamp = list(physics.calls), controller.step_count, controller.current_simulation_time_ns
        scheduler.submit_edit(_velocity_intent())
        result = scheduler.tick_viewport(_request(timeline_frame=frame, timeline_playing=playing), ovrtx_updates=port)
        assert result.status == RuntimeTickStatus.FAILED and result.physics_only_failure
        assert 'unknown' in result.skipped_reason
        assert result.physics_pose_set == () and result.update['simulation_updates_blocked']
        assert physics.calls == calls and not physics.pose_writes and not physics.velocity_writes
        assert not resets and controller.step_count == count and controller.current_simulation_time_ns == stamp
        assert scheduler.has_pending_sim_updates
    finally:
        scheduler.shutdown()
        controller.shutdown()


@pytest.mark.parametrize('failure', [None, 'renderer_update', 'complete_projection', 'engine_sink'])
def test_real_loop_renders_after_physics_failure_but_keeps_renderer_failures_fatal(tmp_path, monkeypatch, failure):
    harness = _Harness(monkeypatch, tmp_path)
    physics = RecordingPhysics()
    config = replace(_config(max_steps=100), input_usd_path=harness.base_request.input_usd_path)
    controller = OvphysxStageController(config, physics_client=physics)
    scheduler = RuntimeScheduler(controller_provider=lambda: controller)
    harness.loop._scheduler = scheduler
    sink = []
    harness.loop._tick_result_sink = lambda result, request: sink.append(result)
    try:
        controller.tick()
        harness.loop._adopt_snapshot(_snapshot(tx=2, max_samples=4))
        harness.loop._iterate()
        assert harness.slot.frames()[-1].status == FRAME_STATUS_FRAME
        before = (harness.client.starts, harness.client.deletes, harness.client.render_calls)
        fail_step(controller, physics)
        calls = list(physics.calls)
        if failure == 'engine_sink':
            from test_scheduler_thread_pose_mirror import _load_engine_with_fake_bpy, _handoff_from_worker
            module = _load_engine_with_fake_bpy(monkeypatch)
            engine = module.OvrtxExampleRenderEngine()
            obj = module._test_scene.objects[0]
            engine._physics_playback_lock.lock_object('/World/Cube', obj, generation=4)
            engine._pending_pose_mirror = {'must_be_discarded': True}
            def handoff(result, request):
                sink.append(result)
                _handoff_from_worker(engine, result, request)
            harness.loop._tick_result_sink = handoff
        scheduler.submit_edit(_velocity_intent())
        scheduler.submit_edit(_transform_intent(_matrix(8)))
        if failure == 'renderer_update':
            harness.client.transform_failures_remaining = 1
        if failure == 'complete_projection':
            harness.controller._project_complete_pose = True
        harness.loop._adopt_snapshot(_snapshot(tx=3, max_samples=4))
        harness.loop._iterate()
        assert physics.calls == calls and not physics.pose_writes and not physics.velocity_writes
        assert controller.failed and scheduler.has_pending_sim_updates
        if failure in {'renderer_update', 'complete_projection'}:
            assert harness.slot.frames()[-1].status == FRAME_STATUS_FAILED
            assert harness.client.render_calls == before[2]
        else:
            assert harness.slot.frames()[-1].status == FRAME_STATUS_FRAME
            assert harness.client.render_calls > before[2]
            assert (harness.client.starts, harness.client.deletes) == before[:2]
            assert sink[-1].status == RuntimeTickStatus.FAILED and sink[-1].physics_only_failure
            assert 'unknown' in sink[-1].skipped_reason
            assert harness.client.camera_translation == 3
            if failure == 'engine_sink':
                assert engine._physics_playback_lock.is_active()
                module.bpy.app.timers.run_pending()
                assert engine._runtime_tick_result is sink[-1]
                assert engine._runtime_tick_result.status == RuntimeTickStatus.FAILED
                assert not engine._physics_playback_lock.is_active()
                assert engine._pending_pose_mirror == {}
                assert physics.calls == calls
    finally:
        harness.controller.shutdown()
        scheduler.shutdown()
        controller.shutdown()

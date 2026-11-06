"""GUI render preparation is dispatched without holding Blender's GPU context."""
import threading
import time
from types import SimpleNamespace
import pytest
from ovrtx_blender_example import engine,main_thread_waits
from ovrtx_blender_example.blender_signal_translation import BlenderSignalTranslationError,BlenderRenderSignalSource


@pytest.fixture(autouse=True)
def reset_final_lane(monkeypatch):
    monkeypatch.setattr(engine, '_FINAL_RENDER_ACTIVE', False)


def queued_timers(monkeypatch):
    queued=[]
    monkeypatch.setattr(engine,'bpy',SimpleNamespace(app=SimpleNamespace(timers=SimpleNamespace(register=lambda fn,first_interval:queued.append(fn)))))
    return queued


def run_worker(function):
    outcome={}
    def work():
        try:outcome['value']=function()
        except BaseException as exc:outcome['error']=exc
    thread=threading.Thread(target=work);thread.start()
    return thread,outcome


def test_job_init_defers_all_blender_preparation(monkeypatch):
    monkeypatch.setattr(engine,'_FINAL_RENDER_REQUEST',object())
    monkeypatch.setattr(engine,'_FINAL_RENDER_INIT_ERROR',RuntimeError('old'))
    monkeypatch.setattr(engine,'_prepare_final_render_request',lambda *a:pytest.fail('job thread accessed Blender'))
    monkeypatch.setattr(engine._DISPLAY_OVERLAY,'unregister',lambda *a:pytest.fail('job thread removed GPU draw handler'))
    scene=SimpleNamespace(render=SimpleNamespace(engine=engine.ENGINE_ID))
    thread,outcome=run_worker(lambda:engine._final_render_init_handler(scene));thread.join(2)
    assert not thread.is_alive() and 'error' not in outcome
    assert engine._FINAL_RENDER_REQUEST is engine._FINAL_RENDER_INIT_ERROR is None
    assert engine._FINAL_RENDER_ACTIVE


def test_preparation_refusal_reaches_waiting_render_unchanged(monkeypatch):
    queued=queued_timers(monkeypatch)
    refusal=BlenderSignalTranslationError(BlenderRenderSignalSource.FINAL_RENDER,'unsupported product')
    def prepare():raise refusal
    thread,outcome=run_worker(lambda:engine._run_on_main_thread(prepare,timeout=1))
    while not queued:thread.join(.001)
    queued[0]();thread.join(2)
    assert outcome['error'] is refusal


def test_timeout_invalidates_late_timer(monkeypatch):
    queued=queued_timers(monkeypatch);ran=[]
    thread,outcome=run_worker(lambda:engine._run_on_main_thread(lambda:ran.append(True),timeout=.001))
    thread.join(2)
    assert not thread.is_alive()
    assert isinstance(outcome.get('error'),BlenderSignalTranslationError)
    assert 'timed out' in str(outcome['error'])
    queued[0]()
    assert ran==[]


def test_cancel_queued_preparation_is_local_to_call(monkeypatch):
    queued=queued_timers(monkeypatch);ran=[]
    thread,outcome=run_worker(lambda:engine._run_on_main_thread(lambda:ran.append(True),cancel_requested=lambda:True))
    thread.join(2)
    assert not thread.is_alive()
    assert isinstance(outcome.get('error'),main_thread_waits.MainThreadWaitCancelled)
    queued[0]()
    assert ran==[] and thread not in main_thread_waits._cancelled


def test_cancel_claimed_export_waits_before_discarding_result(monkeypatch):
    queued=queued_timers(monkeypatch);entered=threading.Event();release=threading.Event();cancel=threading.Event()
    def export():entered.set();assert release.wait(2);return 'must be discarded'
    discarded=[]
    thread,outcome=run_worker(lambda:engine._run_on_main_thread(export,cancel_requested=cancel.is_set,on_discard=discarded.append))
    while not queued:thread.join(.001)
    callback=threading.Thread(target=queued[0]);callback.start();assert entered.wait(1)
    cancel.set();thread.join(.15)
    assert thread.is_alive(),'render depsgraph released while export still uses it'
    release.set();callback.join(2)
    deadline=time.monotonic()+2
    while len(queued)<2 and time.monotonic()<deadline:thread.join(.001)
    assert len(queued)==2
    queued[1]();thread.join(2)
    assert discarded==['must be discarded']
    assert not thread.is_alive()
    assert isinstance(outcome.get('error'),main_thread_waits.MainThreadWaitCancelled)
    assert 'value' not in outcome


def test_discard_preserves_predecessor_and_unrelated_candidate(monkeypatch,tmp_path):
    from ovrtx_blender_example import scene_generation_sessions as sessions
    from ovrtx_blender_example.scene_generation import SceneGenerationOwner,SceneGeneration
    from ovrtx_blender_example.render_requests import RenderRequest
    owner=SceneGenerationOwner(tmp_path)
    def generation(number):
        return SceneGeneration(number,str(number),None,'base',f'{number}.usda',{},(),None,'base.usda',(),{}, {})
    predecessor=generation(0);candidate=generation(1)
    owner._current=predecessor;owner._pending=candidate
    monkeypatch.setattr(sessions,'_owners',{91:owner})
    monkeypatch.setattr(sessions,'_dirty',{91:{'existing'}})
    monkeypatch.setattr(sessions,'_pending_affected',{91:{'captured'}})
    monkeypatch.setattr(sessions,'_blocked_reconciliations',{91:('retry',{'captured'})})
    scene=SimpleNamespace(session_uid=91)
    assert not sessions.discard_final_render_request(scene,RenderRequest(input_usd_path='elsewhere',current_scene_generation=True))
    assert not sessions.discard_final_render_request(scene,RenderRequest(input_usd_path=candidate.usd_path))
    assert owner.pending_generation is candidate
    assert sessions.discard_final_render_request(scene,RenderRequest(input_usd_path=candidate.usd_path,current_scene_generation=True))
    assert owner.current_generation is predecessor and owner.pending_generation is None
    assert sessions._dirty[91]=={'existing','captured'}
    assert 91 not in sessions._pending_affected and 91 not in sessions._blocked_reconciliations
    owner._pending=generation(2)
    assert owner.reuse().number==2


@pytest.mark.parametrize('restore', (False, True))
def test_complete_or_cancel_releases_final_lane_and_restores_viewport(monkeypatch, restore):
    monkeypatch.setattr(engine, '_FINAL_RENDER_ACTIVE', True)
    monkeypatch.setattr(engine, '_FINAL_RENDER_RESTORE_VIEWPORT', restore)
    redraw = []
    monkeypatch.setattr(engine, '_schedule_viewport_redraw', lambda: redraw.append(True))
    engine._final_render_end_handler(object())
    assert not engine._FINAL_RENDER_ACTIVE
    assert redraw == ([True] if restore else [])
    assert not engine._FINAL_RENDER_RESTORE_VIEWPORT


def test_unregister_clears_final_lane_even_without_blender_handlers(monkeypatch):
    monkeypatch.setattr(engine, '_FINAL_RENDER_ACTIVE', True)
    monkeypatch.setattr(engine, 'bpy', None)
    assert not engine.unregister_final_render_handlers()
    assert not engine._FINAL_RENDER_ACTIVE


def test_animation_holds_lane_across_frames_until_terminal_handler(monkeypatch):
    scene = SimpleNamespace(render=SimpleNamespace(engine=engine.ENGINE_ID))
    monkeypatch.setattr(engine, '_prepare_final_render_request', lambda fn: 'prepared')
    engine._final_render_init_handler(scene)
    assert engine._FINAL_RENDER_ACTIVE
    # Frame request consumption does not release the whole animation's lane.
    for _frame in (1, 2, 3):
        monkeypatch.setattr(engine, '_FINAL_RENDER_REQUEST', None)
        engine._consume_final_render_request(scene)
        assert engine._FINAL_RENDER_ACTIVE
    engine._final_render_end_handler(scene)
    assert not engine._FINAL_RENDER_ACTIVE


def test_explicit_init_refusal_is_preserved_and_terminal_hook_releases_lane(monkeypatch):
    scene = SimpleNamespace(render=SimpleNamespace(engine=engine.ENGINE_ID))
    refusal = RuntimeError('explicit unsupported input')
    def refuse(_prepare):
        raise refusal
    monkeypatch.setattr(engine, '_prepare_final_render_request', refuse)
    engine._final_render_init_handler(scene)
    assert engine._FINAL_RENDER_ACTIVE
    assert engine._consume_final_render_request(scene) == (None, refusal)
    engine._final_render_end_handler(scene)
    assert not engine._FINAL_RENDER_ACTIVE


def test_handler_registration_cancellation_and_reregistration_release_lane(monkeypatch):
    hooks = SimpleNamespace(render_init=[], render_complete=[], render_cancel=[])
    monkeypatch.setattr(engine, 'bpy', SimpleNamespace(app=SimpleNamespace(handlers=hooks)))
    assert engine.register_final_render_handlers()
    assert hooks.render_complete == hooks.render_cancel == [engine._final_render_end_handler]
    monkeypatch.setattr(engine, '_FINAL_RENDER_ACTIVE', True)
    hooks.render_cancel[0](object())
    assert not engine._FINAL_RENDER_ACTIVE
    engine._FINAL_RENDER_ACTIVE = True
    assert engine.unregister_final_render_handlers()
    assert not engine._FINAL_RENDER_ACTIVE
    assert not hooks.render_init and not hooks.render_complete and not hooks.render_cancel
    assert engine.register_final_render_handlers()
    assert not engine._FINAL_RENDER_ACTIVE


@pytest.mark.parametrize('current_scene', (False, True))
def test_preparation_preserves_exact_stage_host_but_reconnects_authoring(monkeypatch, current_scene):
    from ovrtx_blender_example.render_requests import RenderRequest
    request = RenderRequest(current_scene_generation=current_scene)
    calls = []
    monkeypatch.setattr(engine, '_FINAL_RENDER_RESTORE_VIEWPORT', False)
    monkeypatch.setattr(engine, 'reconnect_viewport_sessions', lambda: calls.append(True) or {'active_session_count': 1})
    assert engine._prepare_final_render_request(lambda: request) is request
    assert calls == ([True] if current_scene else [])


@pytest.mark.parametrize('fail_at', ('none', 'activation', 'restore'))
def test_prepared_final_lease_serializes_all_chunks_and_releases_queued_edits(monkeypatch, fail_at):
    from ovrtx_blender_example import scene_generation_sessions as sessions
    from ovrtx_blender_example.ovrtx_session_controller import OvrtxSessionController
    from ovrtx_blender_example.render_requests import RenderRequest
    controller = OvrtxSessionController()
    generation = SimpleNamespace(number=1, usd_path='accepted.usda')
    previous = RenderRequest(input_usd_path=generation.usd_path)
    desired = RenderRequest(input_usd_path=generation.usd_path, max_samples=32)
    calls = []
    class Runtime:
        _request = previous
        ovrtx = SimpleNamespace(controller=controller, last_ensure_result=SimpleNamespace(composition=object()))
        def activate(self, _generation, request, **kwargs):
            # Real controller gate must reenter without waiting behind a peer.
            with controller._serialized_transport():
                calls.append(request)
                if fail_at == ('activation' if request is desired else 'restore'):
                    raise RuntimeError(fail_at)
    runtime = Runtime()
    monkeypatch.setattr(sessions, '_owners', {91: SimpleNamespace(current_generation=generation, pending_generation=None)})
    monkeypatch.setattr(sessions, '_runtimes', {91: runtime})
    blocked = []
    monkeypatch.setattr(sessions, 'fail_closed_runtime_reuse', blocked.append)
    if fail_at == 'activation':
        with pytest.raises(RuntimeError, match='activation'):
            sessions.activate_for_final_render(SimpleNamespace(session_uid=91), desired, controller=object())
        lease = None
    else:
        lease = sessions.activate_for_final_render(SimpleNamespace(session_uid=91), desired, controller=object())
    queued = threading.Event()
    applied = threading.Event()
    def replay():
        queued.set()
        with controller._serialized_transport():
            applied.set()
    peer = threading.Thread(target=replay)
    peer.start()
    assert queued.wait(1)
    if lease is not None:
        assert not applied.wait(.05), 'queued edit entered final sampling lease'
        for _chunk in (1, 1, 2, 4, 8, 16):
            with controller._serialized_transport():
                assert not applied.is_set()
        assert lease.deactivate() == ('failed' if fail_at == 'restore' else 'stopped')
    assert applied.wait(1), 'queued edit was dropped or gate leaked after terminal result'
    peer.join(1)
    assert not peer.is_alive()
    assert blocked == ([runtime] if fail_at == 'restore' else [])


def test_prepared_final_restores_request_accepted_before_its_fifo_admission(monkeypatch):
    from ovrtx_blender_example import scene_generation_sessions as sessions
    from ovrtx_blender_example.ovrtx_session_controller import OvrtxSessionController
    from ovrtx_blender_example.render_requests import RenderRequest
    controller = OvrtxSessionController()
    generation = SimpleNamespace(number=1, usd_path='accepted.usda')
    old = RenderRequest(input_usd_path=generation.usd_path, width=100)
    latest = RenderRequest(input_usd_path=generation.usd_path, width=200)
    final = RenderRequest(input_usd_path=generation.usd_path, width=300)
    activations = []
    class Runtime:
        _request = old
        ovrtx = SimpleNamespace(controller=controller, last_ensure_result=SimpleNamespace(composition=object()))
        def activate(self, _generation, request, **kwargs):
            with controller._serialized_transport():
                self._request = request
                activations.append(request)
    runtime = Runtime()
    monkeypatch.setattr(sessions, '_owners', {91: SimpleNamespace(current_generation=generation, pending_generation=None)})
    monkeypatch.setattr(sessions, '_runtimes', {91: runtime})
    def render():
        lease = sessions.activate_for_final_render(SimpleNamespace(session_uid=91), final, controller=object())
        assert lease.deactivate() == 'stopped'
    with controller._serialized_transport():
        thread, outcome = run_worker(render)
        deadline = time.monotonic() + 1
        while not controller._transport_waiters and time.monotonic() < deadline:
            thread.join(.001)
        assert controller._transport_waiters, 'final did not queue behind older activation'
        runtime._request = latest
    thread.join(1)
    assert not thread.is_alive() and 'error' not in outcome
    assert activations == [final, latest]
    assert runtime._request is latest

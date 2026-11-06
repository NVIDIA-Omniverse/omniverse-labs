from contextlib import nullcontext
import threading
from types import SimpleNamespace as N
import pytest
from ovrtx_blender_example import scene_generation_sessions as sessions, render_sequence

@pytest.fixture
def state(monkeypatch):
    monkeypatch.setattr(sessions,'_runtimes',{})
    monkeypatch.setattr(sessions,'_owners',{})
    monkeypatch.setattr(sessions,'_camera_capture_owners',{})
    monkeypatch.setattr(sessions,'_preparation_pending',{})
    monkeypatch.setattr(sessions,'_bake_capture_owner',None)
    monkeypatch.setattr(render_sequence,'_owners',{})
    monkeypatch.setattr(render_sequence,'_jobs',{})
    return N(scene=N(session_uid=991),calls=[])


def prepared(state,*,attached=False,closed=True):
    def close():state.calls.append('close-detached');return closed
    runtime=N(viewport_ids={'pane'} if attached else set(),reuse_blocked=False,
        _preparation_lock=threading.RLock(),close=close,
        ovrtx=N(controller=N(_serialized_transport=nullcontext)))
    sessions._runtimes[state.scene.session_uid]=runtime
    return runtime


def test_attached_viewport_is_rejected_before_any_native_mutation(state):
    runtime=prepared(state,attached=True)
    with pytest.raises(RuntimeError,match='Detach rendered viewports'):
        sessions.reserve_camera_capture(state.scene,object())
    assert state.calls==[] and not sessions._camera_capture_owners
    assert sessions._runtimes[991] is runtime


def test_detached_runtime_drains_then_independent_capture_accepts_source(state,monkeypatch):
    prepared(state)
    capture=N();controller=object()
    generation=N(usd_path='candidate');previous=N(usd_path='old')
    owner=N(current_generation=previous,pending_generation=generation,retained_revision=0,
        retained_values_for=lambda gen:(('retained-transform',),('retained-value',),()),
        accept=lambda gen:state.calls.append('accept-source'))
    sessions._owners[991]=owner
    class Adapter:
        def __init__(self, supplied):assert supplied is controller;self.last_error='';self.activation_timings={}
        def update_request(self,req):pass
        def activate(self,gen,**kwargs):
            assert 'close-detached' in state.calls and 991 not in sessions._runtimes
            transforms, attributes, initial = kwargs['retained_values']()
            assert transforms==('retained-transform',)
            assert attributes==('retained-value',)
            assert not kwargs['skip_unchanged_retained']
            state.calls.append('capture-activate');return True
    monkeypatch.setattr(sessions,'OvrtxGenerationAdapter',Adapter)
    sessions.reserve_camera_capture(state.scene,capture)
    sessions.activate_for_camera_capture(state.scene,N(input_usd_path='candidate'),controller=controller,capture_owner=capture)
    assert state.calls==['close-detached','capture-activate','accept-source']
    assert sessions._camera_capture_owners[991] is capture


def test_unconfirmed_detached_teardown_retains_blocked_runtime(state):
    runtime=prepared(state,closed=False);capture=object()
    sessions.reserve_camera_capture(state.scene,capture)
    with pytest.raises(RuntimeError,match='teardown was not confirmed'):
        sessions.activate_for_camera_capture(state.scene,N(input_usd_path='candidate'),controller=object(),capture_owner=capture)
    assert runtime.reuse_blocked and sessions._runtimes[991] is runtime
    assert state.calls==['close-detached']


def test_viewport_return_does_not_block_ui_or_reuse_capture_rpc_lane(state,monkeypatch):
    entered=threading.Event();release=threading.Event();thread_ids=[]
    class Controller:
        def shutdown(self):
            thread_ids.append(threading.get_ident());entered.set();release.wait(2);return 'stopped'
    capture=render_sequence.CaptureOwner(991);capture.controller=Controller()
    render_sequence._owners[991]=capture
    sessions.reserve_camera_capture(state.scene,capture)
    try:
        with pytest.raises(RuntimeError,match='teardown is draining'):
            sessions._release_camera_capture_for_viewport(991)
        assert entered.wait(1) and 991 in sessions._camera_capture_owners
        assert thread_ids[0]!=threading.get_ident()
        release.set();capture._release_future.result(timeout=2)
        sessions._release_camera_capture_for_viewport(991)
        assert 991 not in sessions._camera_capture_owners and 991 not in render_sequence._owners
    finally:release.set();capture.executor.shutdown(wait=True)


def test_active_capture_rejects_viewport_without_teardown(state):
    capture=N(is_busy=lambda:True,release_for_viewport=lambda:state.calls.append('unsafe-release'))
    sessions.reserve_camera_capture(state.scene,capture)
    with pytest.raises(RuntimeError,match='Camera capture is active'):
        sessions._release_camera_capture_for_viewport(991)
    assert state.calls==[]


def test_ordinary_authoring_lookup_keeps_completed_capture_owner(state,monkeypatch):
    capture=N(is_busy=lambda:False,release_for_viewport=lambda:state.calls.append('retire-native'))
    sessions.reserve_camera_capture(state.scene,capture)
    owner=N(current_generation=object(),pending_generation=None)
    sessions._owners[991]=owner
    class Runtime:
        def __init__(self,controller,owner):
            self.viewport_ids=set();self.reuse_blocked=False
        def attach(self,pane):self.viewport_ids.add(pane)
    monkeypatch.setattr(sessions,'AuthoringGenerationRuntime',Runtime)
    runtime=sessions._runtime_for_owner(991,owner)
    assert not runtime.viewport_ids and state.calls==[]
    assert sessions._camera_capture_owners[991] is capture
    # Actual rendered-view admission remains the only ownership-transfer path.
    with pytest.raises(RuntimeError,match='teardown is draining'):
        sessions.runtime_for_viewport(state.scene,viewport_id='real-rendered-pane')
    assert state.calls==['retire-native']

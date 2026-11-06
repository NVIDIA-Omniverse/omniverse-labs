import threading,json
from types import SimpleNamespace as N
import pytest
from ovrtx_blender_example import render_sequence as seq

def test_marker_selection():
    camera=object();markers=[N(frame=2,camera=camera,select=True),N(frame=1,camera=camera,select=False),N(frame=9,camera=None,select=True)]
    assert seq.camera_frames(markers)==[(1,camera),(2,camera)]
    assert seq.camera_frames(markers,selected_only=True)==[(2,camera)]
    with pytest.raises(ValueError):seq.camera_frames([markers[0],markers[0]])
    with pytest.raises(ValueError):seq.camera_frames([])

def test_hash_receipt_no_fake_semantics(tmp_path,capsys):
    output=tmp_path/'frame.jpg';output.write_bytes(b'completed frame');result=N(completed_samples=16,width=400,height=334,native_timings={'completed':True})
    receipt=seq.write_receipt(output,frame=31,camera='/World/Camera',result=result,runtime={'active':True},elapsed_ms=5)
    assert not receipt['semantic_available']
    assert json.loads(output.with_suffix('.sequence.json').read_text())==receipt
    assert json.loads(output.with_suffix('.capture.json').read_text())['camera']==receipt['camera_prim_path']
    assert 'OVRTX_SEQUENCE_FRAME ' in capsys.readouterr().out
    with pytest.raises(RuntimeError):seq.write_receipt(tmp_path/'missing.jpg',frame=31,camera='/World/Camera',result=result,runtime={},elapsed_ms=5)

def test_successive_jobs_keep_owner(monkeypatch):
    from ovrtx_blender_example import engine
    instances=[];samples=[]
    class Controller:
        def __init__(self,**kwargs):instances.append(self);self.closed=False
        def adopt_owning_thread(self):self.thread=threading.get_ident()
        def ensure(self,request):assert self.thread==threading.get_ident();return N(composition='accepted')
        def diagnostics(self):return {'worker_epoch':1,'native_session_id':'retained'}
        def shutdown(self):self.closed=True;return 'stopped'
    def render(request,cancel,**kwargs):
        samples.append(request.max_samples);return {'status':engine.FINAL_RENDER_STATUS_COMPLETED,'result':N(completed_samples=request.max_samples)}
    monkeypatch.setattr(engine,'_run_final_render_job',render);owner=seq.CaptureOwner(123,Controller)
    try:
        for maximum in [1,16,1]:
            result,runtime=owner.executor.submit(owner.capture,N(max_samples=maximum),threading.Event()).result();assert runtime['native_session_id']=='retained'
        assert len(instances)==1 and not instances[0].closed and samples==[1,16,1]
        owner.executor.submit(owner.close).result();assert instances[0].closed
    finally:owner.executor.shutdown(wait=True)

def test_incomplete_cancelled_never_succeed(monkeypatch):
    from ovrtx_blender_example import engine
    class Controller:
        def __init__(self,**kwargs):pass
        def adopt_owning_thread(self):pass
        def ensure(self,request):return N(composition='accepted')
    monkeypatch.setattr(engine,'_run_final_render_job',lambda *args,**kwargs:{'status':engine.FINAL_RENDER_STATUS_COMPLETED,'result':N(completed_samples=1)})
    owner=seq.CaptureOwner(124,Controller)
    try:
        with pytest.raises(RuntimeError,match='Incomplete'):owner.capture(N(max_samples=16),threading.Event())
        monkeypatch.setattr(engine,'_run_final_render_job',lambda *args,**kwargs:{'status':engine.FINAL_RENDER_STATUS_CANCELLED,'result':None})
        with pytest.raises(RuntimeError,match='cancelled'):owner.capture(N(max_samples=16),threading.Event())
    finally:owner.executor.shutdown(wait=True)

def test_generation_requests_use_acceptance_without_per_frame_teardown(monkeypatch):
    from ovrtx_blender_example import engine, scene_generation_sessions
    calls=[]
    class Controller:
        def __init__(self, **kwargs):pass
        def adopt_owning_thread(self):pass
        def ensure(self, request):raise AssertionError('generation activation must own ensure')
        def diagnostics(self):return {'worker_epoch':7}
    monkeypatch.setattr(scene_generation_sessions,'owns_request',lambda scene,request:scene.session_uid==45)
    def activate(scene, request, *, controller, capture_owner):
        calls.append(request)
        return N(controller=controller,last_ensure_result=N(composition='accepted-source'))
    monkeypatch.setattr(scene_generation_sessions,'activate_for_camera_capture',activate)
    def render(request,cancel,**kwargs):
        assert kwargs['composition']=='accepted-source'
        return {'status':engine.FINAL_RENDER_STATUS_COMPLETED,'result':N(completed_samples=16)}
    monkeypatch.setattr(engine,'_run_final_render_job',render)
    owner=seq.CaptureOwner(45,Controller)
    try:
        for _ in range(2):owner.executor.submit(owner.capture,N(max_samples=16),threading.Event()).result()
        assert len(calls)==2
    finally:owner.executor.shutdown(wait=True)


def test_semantic_receipt_preserves_native_auxiliary_metadata(tmp_path):
    output=tmp_path/'frame.jpg';output.write_bytes(b'rgb')
    capture=output.with_suffix('.capture.json');capture.write_text('{"camera":"/World/Camera","native_id_map":{"2":"kitchen"}}')
    before=capture.read_bytes()
    receipt=seq.write_receipt(output,frame=31,camera='Kitchen overview',camera_prim_path='/World/Camera',
        result=N(completed_samples=16,width=2,height=2,native_timings={}),runtime={},elapsed_ms=5,
        semantic={'verified':True,'capture_path':str(capture)})
    assert receipt['camera']=='Kitchen overview' and receipt['camera_prim_path']=='/World/Camera'
    assert receipt['semantic_available'] and capture.read_bytes()==before


def test_semantic_failure_never_saves_rgb(monkeypatch,tmp_path):
    import sys
    from concurrent.futures import Future
    future=Future();future.set_result((N(),{}))
    module=N(save_capture=lambda *args,**kwargs:{'verified':False})
    monkeypatch.setitem(sys.modules,'ovrtx_blender_example.semantic_output',module)
    owner=N(executor=N(submit=lambda *args:None,shutdown=lambda **kwargs:None),close=lambda:None)
    scene=N(session_uid=56)
    job=seq.Sequence(scene,[(31,N(name='Kitchen overview'))],owner,[],threading.Event(),
        future=future,request=N(semantic_output_path='requested',camera_prim_path='/World/Camera'),
        output=str(tmp_path/'frame.jpg'))
    assert job.tick() is None
    assert job.error=='Native semantic capture was not verified'
    assert not (tmp_path/'frame.jpg').exists() and job.written==0

@pytest.mark.parametrize('current,subframe,expected_sets',[(31,0.,0),(31,.25,1),(110,0.,1)])
def test_frame_camera_edits_are_evaluated_once_before_request(monkeypatch,current,subframe,expected_sets):
    from dataclasses import dataclass
    from concurrent.futures import Future
    from ovrtx_blender_example import engine
    @dataclass
    class Request:
        color_presentation: object = None
        semantic_output_path: str = ''
    events=[]
    scene=N(session_uid=708,frame_current=current,frame_subframe=subframe,
            camera=None,render=N(frame_path=lambda **kwargs:'unused.jpg'))
    def frame_set(frame):
        events.append('frame');scene.frame_current=frame;scene.frame_subframe=0.
    scene.frame_set=frame_set
    camera=N(name='camera')
    def evaluate():
        assert scene.frame_current==31 and scene.frame_subframe==0.
        assert scene.camera is camera
        events.append('evaluate')
    def request(value):
        assert events[-1]=='evaluate'
        events.append('request');return Request()
    monkeypatch.setattr(seq,'bpy',N(context=N(view_layer=N(update=evaluate))))
    monkeypatch.setattr(engine,'_render_callback_adapter',lambda name:N(final_render_from_scene=request))
    monkeypatch.setattr(engine,'_final_render_color_presentation_from_scene',lambda scene:None)
    owner=N(capture=lambda *args:None,executor=N(submit=lambda *args:Future()))
    job=seq.Sequence(scene,[(31,camera)],owner,[],threading.Event())
    assert job.tick()==.1
    assert not job.error
    assert events==(['frame'] if expected_sets else [])+['evaluate','request']

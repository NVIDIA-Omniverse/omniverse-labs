from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace as N
import pytest
from ovrtx_blender_example import hdr_output as hdr, engine, render_sequence
from ovrtx_blender_example.render_requests import RenderRequest

class Scene(dict):
    pass


def setup_capture(monkeypatch,tmp_path,*,semantic=False):
    scene=Scene();scene.session_uid=79;scene.frame_current=31;scene.camera=N(name='Kitchen overview')
    scene.view_settings=N(exposure=2,gamma=2,use_curve_mapping=True)
    scene.ovrtx_example=N(color_presentation_mode='scene_linear_hdr')
    settings=N(file_format='PNG',color_depth='8')
    scene.render=N(engine='OVRTX_EXAMPLE',image_settings=settings,frame_path=lambda **kw:str(tmp_path/'frame0031.png'))
    saved=[];removed=[]
    class Image:
        name='OVRTX HDR Still'
        pixels=N(foreach_set=lambda data:None)
        def save_render(self,path,*,scene):
            saved.append((path,scene.render.image_settings.file_format))
            from pathlib import Path
            Path(path).write_bytes(b'owned native output')
    image=Image()
    blender=N(app=N(is_job_running=lambda job:False),context=N(view_layer=N(update=lambda:None),window_manager=N(windows=[])),
        data=N(images=N(new=lambda *args,**kw:image,remove=lambda im:removed.append(im))))
    request=RenderRequest(max_samples=16,camera_prim_path='/World/Camera')
    if semantic:
        # The semantic agent's request field is optional until source integration.
        from dataclasses import make_dataclass
        Kind=make_dataclass('SemanticRequest',[('semantic_output_path',str,'')],bases=(RenderRequest,),frozen=True)
        request=Kind(max_samples=16,camera_prim_path='/World/Camera',semantic_output_path=str(tmp_path/'semantic'))
    monkeypatch.setattr(engine,'_render_callback_adapter',lambda name:N(final_render_from_scene=lambda sc:request))
    monkeypatch.setattr(engine,'_final_render_color_presentation_from_scene',lambda sc:{'mode':'scene_linear_hdr'})
    result=N(width=2,height=2,linear_rgba16f=bytes(32),linear_rgba16f_bytes=lambda:bytes(32),completed_samples=16,native_timings={})
    future=Future();future.set_result((result,{'session_id':'durable'}))
    owner=N(executor=N(submit=lambda *args:future),capture=lambda *args:None)
    monkeypatch.setattr(render_sequence,'capture_owner_for_scene',lambda sc:owner)
    monkeypatch.setattr(render_sequence,'retire_capture_owner',lambda sc:None)
    hdr._captures.clear();hdr._images.clear()
    return scene,blender,saved,image


def test_retained_hdr_writer_plane_png_exr_and_receipt(monkeypatch,tmp_path):
    scene,blender,saved,image=setup_capture(monkeypatch,tmp_path)
    # Exercise retained storage primitives, not candidate operator admission.
    import time
    request = RenderRequest(camera_prim_path='/World/Camera')
    result=N(width=2,height=2,linear_rgba16f=bytes(32),linear_rgba16f_bytes=lambda:bytes(32),completed_samples=16,native_timings={})
    png=hdr._finish_capture(blender,scene,request,result,{},None,time.perf_counter())
    assert [kind for path,kind in saved]==['PNG','OPEN_EXR']
    assert scene.render.image_settings.file_format=='PNG' and scene.render.image_settings.color_depth=='8'
    assert hdr.last_image(scene) is image
    assert hdr.last_capture(scene)['display_png']==png
    assert hdr.last_capture(scene)['frame']==31 and hdr.last_capture(scene)['camera_prim_path']=='/World/Camera'
    assert scene.view_settings.exposure==2 and scene.view_settings.gamma==2 and scene.view_settings.use_curve_mapping


def test_save_failure_restores_scene_format(tmp_path):
    scene=Scene();scene.render=N(image_settings=N(file_format='PNG',color_depth='8'))
    def save(path,*,scene):raise OSError('disk full')
    with pytest.raises(OSError,match='disk full'):hdr.save_outputs(N(save_render=save),scene,tmp_path/'frame.png')
    assert scene.render.image_settings.file_format=='PNG' and scene.render.image_settings.color_depth=='8'


def test_semantic_failure_prevents_hdr_promotion(monkeypatch,tmp_path):
    import sys
    scene,blender,saved,image=setup_capture(monkeypatch,tmp_path,semantic=True)
    monkeypatch.setitem(sys.modules,'ovrtx_blender_example.semantic_identity',N(scene_labels=lambda scene:{}))
    monkeypatch.setitem(sys.modules,'ovrtx_blender_example.semantic_output',N(save_capture=lambda *args,**kw:{'verified':False}))
    with pytest.raises(ValueError,match='not admitted'):hdr.render_hdr_still(blender,scene)
    assert saved==[] and hdr.last_capture(scene) is None

def test_user_hdr_start_rejected_before_work_or_scene_mutation(monkeypatch,tmp_path):
    scene,blender,saved,image=setup_capture(monkeypatch,tmp_path)
    scheduled=[]
    monkeypatch.setattr(render_sequence,'start_sequence',lambda job:scheduled.append(job))
    with pytest.raises(ValueError,match='not admitted'):
        hdr.start_hdr_still(blender,scene)
    assert scheduled==[] and saved==[] and hdr.last_capture(scene) is None
    assert scene.view_settings.exposure==2 and scene.view_settings.gamma==2

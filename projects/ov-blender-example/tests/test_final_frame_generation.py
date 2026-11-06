"""Frame capture replaces retained Blender edits only after runtime acceptance."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
from ovrtx_blender_example.scene_generation import (
    SceneGeneration, SceneGenerationOwner, BlenderId, BlenderPrimPath)
from ovrtx_blender_example.ovrtx_value_updates import OvrtxTransformValue

IDENTITY=BlenderId('OBJECT',1)

def generation(number=0, frame=1):
    return SceneGeneration(number, str(number), None, 'base', f'{number}.usda',
        {IDENTITY:BlenderPrimPath('Cube','OBJECT','/World/Cube','/World/Cube/Mesh')},
        (), None, 'base.usda', (), {}, {'physics_initial_frame':frame})

def owner(tmp_path):
    result=SceneGenerationOwner(tmp_path)
    result._current=generation()
    value=OvrtxTransformValue('/World/Cube', [[1,0,0,0],[0,1,0,0],[0,0,1,0],[7,0,0,1]])
    result.retain_transform_values([value])
    return result,value

def test_still_reuses_same_evaluated_frame(tmp_path,monkeypatch):
    current,_=owner(tmp_path)
    monkeypatch.setattr(current,'replace',lambda *a,**kw:pytest.fail('unexpected export'))
    assert current.reconcile_final_frame(SimpleNamespace(frame_current=1,frame_subframe=0)) is current.current_generation

def test_final_frame_acceptance_discards_only_superseded_values(tmp_path,monkeypatch):
    current,old_value=owner(tmp_path)
    previous=current.current_generation
    def export(scene,*,force=False):
        assert force
        result=generation(1,scene.frame_current)
        current._current=result
        current._transform_values.clear()  # simulated reentrant export callback
        return result
    monkeypatch.setattr(current,'replace',export)
    candidate=current.reconcile_final_frame(SimpleNamespace(frame_current=2))
    assert current.current_generation is previous
    assert current.retained_values_for(previous)[0]==(old_value,)
    assert current.retained_values_for(candidate)==((),(),())
    current.reject(candidate)
    assert current.current_generation is previous
    assert current.retained_values_for(previous)[0]==(old_value,)
    candidate=current.reconcile_final_frame(SimpleNamespace(frame_current=2))
    current.accept(candidate)
    assert current.current_generation is candidate
    assert current.retained_values_for(candidate)==((),(),())
    current.retain_transform_values([old_value])
    assert current.retained_values_for(candidate)[0]==(old_value,)  # later viewport edits survive

def test_failed_export_preserves_predecessor_and_retained_values(tmp_path,monkeypatch):
    current,value=owner(tmp_path);previous=current.current_generation
    revision=current.retained_revision
    def failed(*a,**kw):
        current._transform_values.clear();current._retained_revision+=1
        raise RuntimeError('export failed')
    monkeypatch.setattr(current,'replace',failed)
    with pytest.raises(RuntimeError,match='export failed'):
        current.reconcile_final_frame(SimpleNamespace(frame_current=2))
    assert current.current_generation is previous and current.pending_generation is None
    assert current.retained_values_for(previous)[0]==(value,)
    assert current.retained_revision==revision

def test_subframe_requires_capture_and_pending_other_frame_is_refused(tmp_path,monkeypatch):
    current,_=owner(tmp_path)
    monkeypatch.setattr(current,'replace',lambda scene,**kw:generation(1,1.5))
    pending=current.reconcile_final_frame(SimpleNamespace(frame_current=1,frame_subframe=.5))
    assert current.reconcile_final_frame(SimpleNamespace(frame_current=1,frame_subframe=.5)) is pending
    with pytest.raises(RuntimeError,match='awaiting runtime handoff'):
        current.reconcile_final_frame(SimpleNamespace(frame_current=2))

@pytest.mark.parametrize('current_scene,expected',[(True,None),(False,'request')])
def test_prebuilt_request_frame_check_preserves_exact_stage(monkeypatch,current_scene,expected):
    from ovrtx_blender_example import engine
    from ovrtx_blender_example.render_requests import RenderRequest
    request=RenderRequest(current_scene_generation=current_scene,timeline_frame=3)
    monkeypatch.setattr(engine,'_FINAL_RENDER_REQUEST',request)
    monkeypatch.setattr(engine,'_FINAL_RENDER_INIT_ERROR',None)
    result,error=engine._consume_final_render_request(SimpleNamespace(frame_current=1))
    assert result is (request if expected else None) and error is None
    assert engine._FINAL_RENDER_REQUEST is None

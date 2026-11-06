from types import SimpleNamespace as N
import pytest
from ovrtx_blender_example import scene_generation
from ovrtx_blender_example.object_visibility import export_visibility


@pytest.mark.parametrize('fail',[False,True])
@pytest.mark.parametrize('driven',[False,True])
def test_hidden_locked_sparse_object_export_restores_authored_state(tmp_path,monkeypatch,fail,driven):
    collection=N(hide_render=True,hide_viewport=True,hide_select=True,children=[])
    class Object:
        def __init__(self,name,*,hidden=False,selected=False):
            self.name_full=name;self.hide_render=hidden;self.hide_viewport=hidden;self.hide_select=hidden
            self.selected=selected;self.mode='OBJECT'
        def select_set(self,value):
            assert not self.hide_select and not collection.hide_select
            assert not self.hide_viewport
            self.selected=value
    original=Object('original',selected=True);target=Object('hidden wall',hidden=True)
    curve=N(data_path='hide_render',mute=False)
    if driven:target.animation_data=N(drivers=[curve])
    objects=[original,target]
    def update():
        if driven and not curve.mute:target.hide_render=True
    layer=N(objects=N(active=original),update=update)
    scene=N(objects=objects,collection=collection)
    class Context:
        view_layer=layer
        @property
        def selected_objects(self):return tuple(obj for obj in objects if obj.selected)
    module=N(context=Context())
    module.context.scene=scene
    monkeypatch.setattr(scene_generation,'_blender_module_provider',lambda:module)
    calls=[]
    def export(scene,path,*,selected_objects_only):
        assert selected_objects_only
        assert not target.hide_render and not target.hide_viewport and not target.hide_select
        assert not collection.hide_render and not collection.hide_viewport and not collection.hide_select
        assert target.selected and not original.selected and layer.objects.active is target
        calls.append(path)
        if fail:raise RuntimeError('USD export failed')
    monkeypatch.setattr(scene_generation,'_stock_export',export)
    if fail:
        with pytest.raises(RuntimeError,match='USD export failed'):
            scene_generation._selected_stock_export(scene,(target,),tmp_path/'selected.usda')
    else:scene_generation._selected_stock_export(scene,(target,),tmp_path/'selected.usda')
    assert len(calls)==1
    assert target.hide_render and target.hide_viewport and target.hide_select and not target.selected
    assert collection.hide_render and collection.hide_viewport and collection.hide_select
    assert original.selected and layer.objects.active is original
    assert not curve.mute


def test_nested_visibility_context_keeps_outer_export_state_then_restores():
    obj=N(hide_render=True,hide_viewport=True,hide_select=True)
    child=N(hide_render=True,hide_viewport=True,hide_select=True,children=[])
    root=N(hide_render=True,hide_viewport=True,hide_select=True,children=[child])
    scene=N(objects=[obj],collection=root)
    layer=N(update=lambda:None);module=N(context=N(scene=scene,view_layer=layer))
    with export_visibility(module,scene):
        with export_visibility(module,scene):assert not obj.hide_render and not child.hide_select
        assert not obj.hide_render and not root.hide_select
    assert obj.hide_render and obj.hide_select and root.hide_select and child.hide_select


@pytest.mark.parametrize('fail',[False,True])
@pytest.mark.parametrize('initial_mute',[False,True])
@pytest.mark.parametrize('original_value',[False,True])
@pytest.mark.parametrize('property_name',['hide_render','hide_viewport','hide_select'])
def test_driven_visibility_nested_export_preserves_driver_and_value(property_name,initial_mute,original_value,fail):
    from ovrtx_blender_example.object_visibility import scene_visibility
    curve=N(data_path=property_name,mute=initial_mute)
    unrelated=N(data_path='energy',mute=False)
    collection_curve=N(data_path=property_name,mute=initial_mute)
    obj=N(type='MESH',session_uid=1,hide_render=True,hide_viewport=True,hide_select=True,
          animation_data=N(drivers=[curve,unrelated]))
    root=N(hide_render=True,hide_viewport=True,hide_select=True,objects=[obj],children=[],
           animation_data=N(drivers=[collection_curve]))
    setattr(obj,property_name,original_value)
    setattr(root,property_name,original_value)
    scene=N(objects=[obj],collection=root)
    updates=[]
    def update():
        for owner,driver in ((obj,curve),(root,collection_curve)):
            if not driver.mute:setattr(owner,property_name,original_value)
        updates.append(bool(curve.mute))
    layer=N(update=update);module=N(context=N(scene=scene,view_layer=layer))
    before=scene_visibility(scene,layer)
    def export():
        with export_visibility(module,scene):
            assert curve.mute and collection_curve.mute and not unrelated.mute
            assert not getattr(obj,property_name) and not getattr(root,property_name)
            with export_visibility(module,scene):
                update()
                assert curve.mute and not getattr(obj,property_name)
            assert curve.mute and collection_curve.mute
            if fail:raise RuntimeError('export failed')
    if fail:
        with pytest.raises(RuntimeError,match='export failed'):export()
    else:export()
    assert curve.mute is initial_mute and collection_curve.mute is initial_mute
    assert not unrelated.mute
    assert all(getattr(owner,name) == (original_value if name==property_name else True)
               for owner in (obj,root) for name in ('hide_render','hide_viewport','hide_select'))
    assert scene_visibility(scene,layer)==before
    assert updates and updates[0] is True

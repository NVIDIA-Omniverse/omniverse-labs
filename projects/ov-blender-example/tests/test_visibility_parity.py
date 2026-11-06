# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace as NS
import pytest
from ovrtx_blender_example import object_visibility as visibility
from ovrtx_blender_example import light_value_conversion as lights
from ovrtx_blender_example import scene_generation_sessions as sessions
from ovrtx_blender_example.scene_generation import BlenderId


def test_explicit_and_foreign_layer_selection():
    a,b = NS(name="A"),NS(name="B")
    scene = NS(view_layers=[a,b])
    assert visibility.scene_view_layer(scene, view_layer=b) is b
    with pytest.raises(ValueError, match="does not belong"):
        visibility.scene_view_layer(scene, view_layer=NS(name="Other"))
    with pytest.raises(ValueError, match="explicit owning"):
        visibility.scene_view_layer(scene, NS(context=NS(scene=None)))


@pytest.mark.parametrize("kind", sorted(visibility.GEOMETRY_TYPES | {"LIGHT"}))
def test_geometry_and_light_collection_path_visibility(kind):
    obj = NS(type=kind,session_uid=1,hide_render=False,hide_viewport=False)
    hidden = NS(objects=[obj],children=[],hide_render=True)
    enabled = NS(objects=[obj],children=[],hide_render=False)
    scene = NS(objects=[obj],collection=NS(objects=[],children=[hidden,enabled]))
    assert visibility.scene_visibility(scene) == ((obj,True),)
    enabled.hide_render=True
    assert visibility.scene_visibility(scene) == ((obj,False),)


def test_unreported_collection_visibility_marks_changed_generation_owner(monkeypatch):
    obj = NS(type="LIGHT",session_uid=1,hide_render=True)
    scene = NS(session_uid=99,objects=[obj])
    key = BlenderId("OBJECT",1)
    generation = NS(object_visibility={key:True},shadow_visibility={})
    monkeypatch.setattr(sessions,"_owners",{99:NS(pending_generation=None,current_generation=generation)})
    calls=[]
    monkeypatch.setattr(sessions,"mark_scene_dirty",lambda s,ids,**kw:calls.append((ids,kw)))
    sessions._check_render_visibility(scene)
    assert calls[0][0]=={key}
    assert calls[0][1]['update_provenance'][key].geometry is False


def test_normalization_and_contributions_share_typed_initial_live_policy():
    data=NS(type="AREA",energy=12,exposure=2,normalize=True,shape="RECTANGLE",size=2,size_y=3,diffuse_factor=.25,specular_factor=.75)
    normalized={a.name:a for a in lights.usd_attribute_values(data)}
    data.normalize=False
    unnormalized={a.name:a for a in lights.usd_attribute_values(data)}
    assert unnormalized["inputs:intensity"].value==pytest.approx(normalized["inputs:intensity"].value*6)
    assert unnormalized["inputs:diffuse"].value==.25
    assert unnormalized["inputs:specular"].value==.75
    for field,attr in (("normalize","inputs:intensity"),("diffuse_factor","inputs:diffuse"),("specular_factor","inputs:specular")):
        assert attr in lights.classify_field(data,field).usd_attributes


def test_observer_is_bounded_and_freed_scene_is_forgotten(monkeypatch):
    callbacks=[]
    scene=NS(session_uid=77)
    monkeypatch.setattr(sessions,"bpy",NS(app=NS(timers=NS(register=lambda callback,**kw:callbacks.append((callback,kw))))))
    monkeypatch.setattr(sessions,"_visibility_observers",{})
    monkeypatch.setattr(sessions,"_owners",{77:NS(current_generation=NS(object_visibility={}))})
    monkeypatch.setattr(sessions,"_dirty",{})
    monkeypatch.setattr(sessions,"is_authoring",lambda s:False)
    monkeypatch.setattr(sessions,"_check_render_visibility",lambda s:None)
    sessions._ensure_render_visibility_observer(scene)
    sessions._ensure_render_visibility_observer(scene)
    assert len(callbacks)==1 and callbacks[0][1]["first_interval"]==.25
    callback=callbacks[0][0]
    assert callback()==.25
    def freed(s): raise ReferenceError("Scene removed during file load")
    monkeypatch.setattr(sessions,"_check_render_visibility",freed)
    assert callback() is None
    assert 77 not in sessions._visibility_observers


def test_source_cache_uses_owning_layer_not_ambient_layer(tmp_path,monkeypatch):
    from ovrtx_blender_example import scene_generation as sg
    path=tmp_path/"saved.blend"
    path.write_bytes(b"fixture")
    own=NS(name="Own")
    scene=NS(name_full="Scene",objects=[],view_layers=[own])
    ambient=NS(name="Ambient")
    module=NS(data=NS(filepath=str(path),is_dirty=False),context=NS(scene=object(),view_layer=ambient))
    monkeypatch.setattr(sg,"_blender_module_provider",lambda:module)
    before=sg._source_lookup(scene)[0]
    ambient.name="ChangedOtherScene"
    assert sg._source_lookup(scene)[0]==before
    own.name="ChangedOwningLayer"
    assert sg._source_lookup(scene)[0]!=before


def test_stock_blender_retained_visibility_contract(tmp_path):
    import subprocess
    from pathlib import Path
    from blender_test_support import blender_executable
    blender=blender_executable()
    if blender is None:
        pytest.skip("Stock Blender unavailable")
    fixture=Path(__file__).with_name("blender_visibility_parity.py")
    completed=subprocess.run([str(blender),"--background","--factory-startup","--disable-autoexec",
        "--python-exit-code","1","--python",str(fixture)],capture_output=True,text=True,timeout=120)
    assert completed.returncode==0, completed.stdout+completed.stderr
    assert "VISIBILITY_PARITY_PASS" in completed.stdout



def test_expanded_retained_owners_do_not_cause_permanent_identity_dirty():
    owners=[NS(session_uid=uid,type=kind,library=None,override_library=None,data=NS(materials=[]))
            for uid,kind in ((1,"CURVE"),(2,"EMPTY"))]
    mapped={BlenderId("OBJECT",obj.session_uid) for obj in owners}
    assert sessions._export_reachable_current_identities(NS(objects=owners),mapped)==mapped


def test_idle_visibility_watch_never_traverses_objects_and_notifies_on_changes():
    subscriptions, cleared, handlers = [], [], []
    class NoScan:
        def __iter__(self):
            pytest.fail('Idle or camera-only visibility poll scanned the scene')
    scene = NS(objects=NoScan())
    module = NS(context=NS(scene=scene, view_layer=object()),
        types=NS(Object=object(), Collection=object(), LayerCollection=object()),
        msgbus=NS(subscribe_rna=lambda **kw: subscriptions.append(kw),
                  clear_by_owner=lambda owner: cleared.append(owner)),
        app=NS(handlers=NS(depsgraph_update_post=handlers)))
    watch = visibility.VisibilityWatch(scene, module)
    assert watch.active and len(subscriptions) == 8 and len(handlers) == 1
    assert watch.pending()
    watch.dirty = False
    camera = NS(type='CAMERA', bl_rna=NS(identifier='Object'))
    for _ in range(100):
        handlers[0](scene, NS(updates=[NS(id=camera)]))
        assert not watch.pending()
    for subscription in subscriptions:
        subscription['notify']()
        assert watch.pending()
        watch.dirty = False
    handlers[0](scene, NS(updates=[NS(id=NS(type='MESH'))]))
    assert watch.pending()
    watch.dirty = False
    module.context.view_layer = object()
    assert watch.pending()
    watch.close()
    assert not handlers and cleared == [watch.owner]


def test_visibility_watch_partial_subscription_failure_keeps_conservative_polling():
    cleared = []
    def unavailable(**kw):
        raise TypeError('Unsupported RNA property')
    module = NS(context=NS(scene=None), types=NS(Object=object()),
        msgbus=NS(subscribe_rna=unavailable, clear_by_owner=cleared.append),
        app=NS(handlers=NS(depsgraph_update_post=[])))
    watch = visibility.VisibilityWatch(NS(), module)
    watch.dirty = False
    assert watch.pending() and not watch.active and cleared == [watch.owner]

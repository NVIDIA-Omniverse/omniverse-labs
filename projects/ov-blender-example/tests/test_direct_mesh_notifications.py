"""Capture must not lose UV/attribute-only edits rejected by legacy loop proof."""
from types import SimpleNamespace as NS
import pytest
from addon.ovrtx_blender_example import engine
from addon.ovrtx_blender_example.scene_generation import BlenderId


def update(kind='Mesh', geometry=True, transform=False, uid=102):
    source=NS(bl_rna=NS(identifier=kind),session_uid=uid,library=None,type='MESH')
    return NS(id=source,is_updated_geometry=geometry,is_updated_transform=transform,is_updated_shading=True)


@pytest.mark.parametrize('geometry',[True,None])
def test_uv_or_unknown_mesh_notification_reaches_capture_despite_no_legacy_edit(monkeypatch,geometry):
    scene=NS(session_uid=101); graph=NS(updates=[update(geometry=geometry)],mode='VIEWPORT')
    sessions=engine.scene_generation_sessions
    monkeypatch.setitem(sessions._camera_capture_owners,101,object())
    monkeypatch.setattr(sessions,'observe_depsgraph_physics_edits',lambda *_:NS(restart=False,redraw=False,pending_anchor=False,body_input=False))
    monkeypatch.setattr(sessions,'topology_identity_changes',lambda *_:set())
    monkeypatch.setattr(sessions,'shader_only_generation_update',lambda *_:False)
    monkeypatch.setattr(sessions,'is_reconciling',lambda *_:False)
    monkeypatch.setattr(engine,'_interactive_edit_bridge_suppressed',lambda:False)
    monkeypatch.setattr(engine,'submit_depsgraph_interactive_edits_to_active_viewports',lambda *_,**__:[])
    calls=[]
    monkeypatch.setattr(sessions,'mark_scene_dirty',lambda *args,**kw:calls.append((args,kw)))
    engine._live_interactive_edit_depsgraph_handler(scene,graph)
    assert len(calls)==1
    assert calls[0][0][1]=={BlenderId('MESH',102)}
    assert calls[0][1]['update_provenance'][BlenderId('MESH',102)].geometry is geometry


def test_transform_only_and_shading_only_notifications_do_not_enter_geometry_lane(monkeypatch):
    scene=NS(session_uid=101)
    monkeypatch.setitem(engine.scene_generation_sessions._camera_capture_owners,101,object())
    graph=NS(updates=[update(kind='Object',geometry=False,transform=True), update(geometry=False)])
    assert engine._capture_mesh_geometry_ids(scene,graph)==set()


def test_capture_mesh_notifications_keep_legacy_viewport_scope(monkeypatch):
    scene=NS(session_uid=101)
    monkeypatch.delitem(engine.scene_generation_sessions._camera_capture_owners,101,raising=False)
    assert engine._capture_mesh_geometry_ids(scene,NS(updates=[update()]))==set()


def test_only_nominated_geometry_sources_are_retained(monkeypatch):
    scene=NS(session_uid=101)
    monkeypatch.setitem(engine.scene_generation_sessions._camera_capture_owners,101,object())
    graph=NS(updates=[update(),update(kind='Object',uid=103),update(kind='Material',uid=104)])
    assert engine._capture_mesh_geometry_ids(scene,graph)=={BlenderId('MESH',102),BlenderId('OBJECT',103)}


@pytest.mark.parametrize('object_type',['MESH','LIGHT'])
def test_visibility_return_to_exported_baseline_is_nominated_from_ack_state(monkeypatch,object_type):
    scene=NS(session_uid=201);sessions=engine.scene_generation_sessions
    key=BlenderId('OBJECT',202)
    original=NS(session_uid=202,type=object_type,bl_rna=NS(identifier='Object'),library=None,
                hide_render=False,hide_viewport=False)
    event=NS(id=original,is_updated_geometry=False,is_updated_transform=False,is_updated_shading=True)
    graph=NS(updates=[event],mode='VIEWPORT')
    # Initial USD was visible; direct hide has been ACKed. A show edit restores
    # the initial authoring value, so the legacy replacement builder emits none.
    state=NS(token='invisible',render_enabled=True)
    contract=NS(states={key:state},handles={key:original},view_layer=None)
    owner=NS(current_generation=None,_direct_state=NS(visibility=contract,data_users={}))
    monkeypatch.setitem(sessions._camera_capture_owners,201,object())
    monkeypatch.setitem(sessions._owners,201,owner)
    monkeypatch.setattr(sessions,'observe_depsgraph_physics_edits',lambda *_:NS(restart=False,redraw=False,pending_anchor=False,body_input=False))
    monkeypatch.setattr(sessions,'topology_identity_changes',lambda *_:set())
    monkeypatch.setattr(sessions,'shader_only_generation_update',lambda *_:False)
    monkeypatch.setattr(sessions,'is_reconciling',lambda *_:False)
    monkeypatch.setattr(engine,'_interactive_edit_bridge_suppressed',lambda:False)
    monkeypatch.setattr(engine,'submit_depsgraph_interactive_edits_to_active_viewports',lambda *_,**__:[])
    calls=[];monkeypatch.setattr(sessions,'mark_scene_dirty',lambda *a,**kw:calls.append((a,kw)))
    engine._live_interactive_edit_depsgraph_handler(scene,graph)
    assert len(calls)==1 and calls[0][0][1]=={key}
    assert calls[0][1]['update_provenance'][key].geometry is False
    # Once shown and ACKed, ordinary pose-only notifications stay on their path.
    state.token='inherited';event.is_updated_transform=True;calls.clear()
    engine._live_interactive_edit_depsgraph_handler(scene,graph)
    assert not calls


def test_visibility_data_notification_maps_only_indexed_owners_and_retains_flags(monkeypatch):
    from addon.ovrtx_blender_example.scene_generation import SceneUpdateProvenance
    scene=NS(session_uid=301);key=BlenderId('OBJECT',302);data_key=BlenderId('LIGHT',303)
    source=NS(type='LIGHT',hide_render=False,hide_viewport=False)
    contract=NS(states={key:NS(token='invisible',render_enabled=True)},handles={key:source},view_layer=None)
    direct=NS(visibility=contract,data_users={303:(key,)})
    monkeypatch.setitem(engine.scene_generation_sessions._camera_capture_owners,301,object())
    monkeypatch.setitem(engine.scene_generation_sessions._owners,301,NS(_direct_state=direct))
    event=update(kind='AreaLight',geometry=False,uid=303)
    provenance={data_key:SceneUpdateProvenance(False,False,True)}
    assert engine._capture_visibility_ids(scene,NS(updates=[event]),provenance)=={key}
    assert provenance[key]==provenance[data_key]

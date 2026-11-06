# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import sys
from types import SimpleNamespace as N
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'addon'))
from ovrtx_blender_example import native_scene_edit_batch as batch
from ovrtx_blender_example import scene_generation as generation
from ovrtx_blender_example import scene_generation_sessions as sessions
from ovrtx_blender_example import material_graph_fingerprint as graph
from ovrtx_blender_example import object_visibility


def setup(monkeypatch):
    mat=N(session_uid=10,signature='color-A')
    obj=N(type='MESH',session_uid=1,data=N(session_uid=2),material_slots=[N(material=mat)],
          visible_shadow=True,instance_collection=None,visible=True,inputs=('tile-A',))
    world=N(session_uid=11,signature='world-A')
    scene=N(session_uid=912000,objects=[obj],world=world)
    monkeypatch.setattr(generation,'_mesh_render_inputs',lambda o,d,include_elements=False:o.inputs)
    monkeypatch.setattr(object_visibility,'scene_visibility',lambda s:[(o,o.visible) for o in s.objects])
    monkeypatch.setattr(graph,'_graph_signature',lambda m,**kw:m.signature)
    monkeypatch.setattr(graph,'world_graph_signature',lambda w:w.signature)
    calls=[]
    monkeypatch.setattr(sessions,'mark_scene_dirty',lambda s,ids:calls.append(set(ids)) or True)
    return scene,obj,mat,world,calls


def test_native_binding_modifier_and_visibility_change_marks_only_actual_ids(monkeypatch):
    scene,obj,mat,world,calls=setup(monkeypatch)
    with batch.native_scene_edit_batch(scene):
        obj.data=N(session_uid=3);obj.inputs=('tile-B',);obj.visible=False
    assert calls==[{generation.BlenderId('OBJECT',1)}]
    assert generation.BlenderId('WORLD',11) not in calls[0]


def test_noop_batch_and_failed_caller_do_not_dirty_source(monkeypatch):
    scene,obj,mat,world,calls=setup(monkeypatch)
    with batch.native_scene_edit_batch(scene): pass
    assert calls==[]
    with pytest.raises(ValueError):
        with batch.native_scene_edit_batch(scene):
            obj.data=N(session_uid=3)
            raise ValueError('rollback')
    assert calls==[]


def test_shader_value_and_world_graph_detection_and_return(monkeypatch):
    scene,obj,mat,world,calls=setup(monkeypatch)
    with batch.native_scene_edit_batch(scene): mat.signature='color-B'
    assert calls==[{generation.BlenderId('MATERIAL',10)}]
    with batch.native_scene_edit_batch(scene): mat.signature='color-A'
    assert calls[-1]=={generation.BlenderId('MATERIAL',10)}
    with batch.native_scene_edit_batch(scene): world.signature='world-B'
    assert calls[-1]=={generation.BlenderId('WORLD',11)}


def test_deferred_native_refresh_uses_only_accepted_generation_snapshot(monkeypatch):
    scene,obj,mat,world,calls=setup(monkeypatch)
    g=N(number=1,digest='accepted')
    owner=N(current_generation=g,pending_generation=None)
    monkeypatch.setitem(sessions._owners,scene.session_uid,owner)
    batch.remember_generation_snapshot(scene,g)
    mat.signature='deferred-color'
    pending=N(number=2,digest='candidate')
    owner.pending_generation=pending
    batch.remember_generation_snapshot(scene,pending)
    affected=batch.refresh_native_scene(scene)
    assert affected=={generation.BlenderId('MATERIAL',10)}
    assert calls[-1]==affected
    owner.current_generation=pending;owner.pending_generation=None
    assert not batch.refresh_native_scene(scene)
    batch._generation_snapshots.clear()

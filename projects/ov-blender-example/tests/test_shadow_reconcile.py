# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from ovrtx_blender_example import scene_generation as sg


def test_shadow_change_uses_complete_export_pending_lifecycle(tmp_path, monkeypatch):
    key=sg.BlenderId('OBJECT',17)
    glass=SimpleNamespace(type='MESH',session_uid=17,visible_shadow=False)
    scene=SimpleNamespace(objects=[glass])
    before=SimpleNamespace(shadow_visibility={key:True})
    after=SimpleNamespace(shadow_visibility={key:False})
    owner=sg.SceneGenerationOwner(tmp_path/'generations')
    owner._current=before
    calls=[]
    def complete_export(value):
        calls.append(value)
        owner._current=after
        return after
    monkeypatch.setattr(owner,'replace',complete_export)
    result=owner.reconcile(scene,{key})
    assert calls == [scene]
    assert result is after
    assert owner.current_generation is before
    assert owner.pending_generation is after


def test_unchanged_shadow_does_not_force_full_export():
    key=sg.BlenderId('OBJECT',17)
    glass=SimpleNamespace(type='MESH',session_uid=17,visible_shadow=False)
    scene=SimpleNamespace(objects=[glass])
    generation=SimpleNamespace(shadow_visibility={key:(False,True)})
    assert not sg._shadow_visibility_changed(scene,generation,{key})
    glass.visible_shadow=True
    assert sg._shadow_visibility_changed(scene,generation,{key})
    glass.visible_shadow=False
    glass.visible_camera=False  # camera ray visibility regenerates the same way as shadows
    assert sg._shadow_visibility_changed(scene,generation,{key})
    assert not sg._shadow_visibility_changed(scene,generation,{sg.BlenderId('OBJECT',18)})

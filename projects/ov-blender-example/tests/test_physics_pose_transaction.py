# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'addon'))
from ovrtx_blender_example import physics_pose_transaction as transaction
from ovrtx_blender_example.interactive_operator_state import PhysicsPlaybackLock


class Matrix:
    def __init__(self, value): self.value=value
    def copy(self): return Matrix(self.value)


class Output:
    def __init__(self, path, reject=False):
        self.path=path;self._matrix=Matrix('old');self.reject=reject
        self.lock_location=(False,)*3;self.lock_rotation=(False,)*3
    def get(self, name, default=None): return self.path
    @property
    def matrix_world(self): return self._matrix
    @matrix_world.setter
    def matrix_world(self, value):
        if self.reject and value.value=='new': raise RuntimeError('second command failed')
        self._matrix=value


@pytest.mark.parametrize('failure', ['exception', 'partial', 'stale', 'false'])
def test_timer_batch_rolls_back_all_outputs_and_lock_on_failure(monkeypatch, failure):
    rows=[Output('/World/A'),Output('/World/B',reject=failure=='exception')]
    scene=SimpleNamespace(objects=rows)
    layer=SimpleNamespace(update=lambda:None)
    bpy=SimpleNamespace(context=SimpleNamespace(view_layer=layer))
    lock=PhysicsPlaybackLock();lock._last_reason='old'
    pending={'poses_by_path':{r.path:{} for r in rows}}
    current=[True]
    def apply(*args):
        lock._last_reason='new'
        rows[0].matrix_world=Matrix('new')
        rows[1].matrix_world=Matrix('new')
        if failure=='stale':current[0]=False
        return {'enabled':failure!='false', 'status':'applied','mirror_error_count':1 if failure=='partial' else 0}
    monkeypatch.setattr(transaction.operator_state, 'apply_pending_runtime_pose_mirror', apply)
    if failure=='exception':
        with pytest.raises(RuntimeError,match='second command'):
            transaction.apply(bpy,scene,pending,lock,initial=False,still_current=lambda:current[0])
    else:
        result=transaction.apply(bpy,scene,pending,lock,initial=False,still_current=lambda:current[0])
        assert not result['enabled']
    assert [r.matrix_world.value for r in rows]==['old','old']
    assert lock._last_reason=='old'


def test_authored_flush_invalidates_before_any_mirror_write(monkeypatch):
    current=[True]
    layer=SimpleNamespace(update=lambda:current.__setitem__(0,False))
    bpy=SimpleNamespace(context=SimpleNamespace(view_layer=layer))
    monkeypatch.setattr(transaction.operator_state, 'apply_pending_runtime_pose_mirror', lambda *args:pytest.fail('stale write'))
    result=transaction.apply(bpy,SimpleNamespace(objects=[]),{'poses_by_path':{}},PhysicsPlaybackLock(),
        initial=False,still_current=lambda:current[0])
    assert result['status']=='stale'

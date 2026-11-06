"""Narrow physics opinion validation and unchanged baseline dispatch."""
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import pytest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
from inline_physics_geometry import prepare
from incremental_physics_backend import IncrementalPhysicsBackend
from public_sdk_backend import PhysicsBackend


def read(path, name):
    return np.asarray([2.]) if path == '/Ground' and name == 'size' else None


def test_current_geometry_opinion_replaces_value_without_history():
    first, text = prepare([dict(prim_path='/Ground', attribute_name='size', value_type='Double', value=3.)], {}, read)
    second, next_text = prepare([dict(prim_path='/Ground', attribute_name='size', value_type='Double', value=4.)], first, read)
    assert len(second) == 1 and second[('/Ground', 'size')] == [4.]
    assert 'double size = 4.0' in next_text and '3.0' not in next_text


@pytest.mark.parametrize('row', [
    dict(prim_path='/Ground', attribute_name='size', value_type='Double', value=-1),
    dict(prim_path='/Ground', attribute_name='physics:mass', value_type='Float', value=1),
    dict(prim_path='/Missing', attribute_name='size', value_type='Double', value=1),
])
def test_unsupported_geometry_fails_before_population(row):
    with pytest.raises(ValueError):
        prepare([row], {}, read)


def test_dynamic_collider_geometry_not_admitted():
    with pytest.raises(ValueError, match='static'):
        prepare([dict(prim_path='/Ground', attribute_name='size', value_type='Double', value=3)], {},
                lambda p,n: np.asarray([True]) if n == 'physics:rigidBodyEnabled' else read(p,n))


def test_old_operations_delegate_to_unchanged_backend(monkeypatch):
    seen=[]
    monkeypatch.setattr(PhysicsBackend, 'dispatch', lambda self,op,p: (seen.append((op,p)) or 'original'))
    obj=IncrementalPhysicsBackend.__new__(IncrementalPhysicsBackend)
    assert obj.dispatch('step', {'dt_ns':1}) == 'original'
    assert seen == [('step', {'dt_ns':1})]


def test_new_geometry_dispatch_is_cpu_only():
    obj=IncrementalPhysicsBackend.__new__(IncrementalPhysicsBackend)
    obj._geometry_device='gpu'
    with pytest.raises(ValueError, match='CPU only'):
        obj.dispatch('write_geometry', {'values':[]})

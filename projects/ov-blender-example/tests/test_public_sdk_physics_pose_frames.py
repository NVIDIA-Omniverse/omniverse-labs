"""Stage pose edits carry world poses; omni:xform must hold the parent-local matrix.

OVPhysX 0.7 composes a control-edited ``omni:xform`` with its ancestors (and
stops at ``omni:resetXformStack``). Writing the world matrix there applied a
transformed parent twice (measured: requested (8, 0, 6) under a parent rotated
90 degrees about Z at (5, 1, 0.5) landed at (5, 9, 6.5)).
"""
from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
import public_sdk_backend as backend

# Row-vector convention (v' = v @ M): +90 degrees about Z, then translate.
HOLDER = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [5, 1, .5, 1]], dtype=np.float64)
OFFSET = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 10, 1]], dtype=np.float64)


def worker(values):
    writes = {}

    class Result:
        def wait(self):
            pass

        def groups(self):
            return iter(())

    stage = SimpleNamespace(
        query_from_path_list=lambda plist: nullcontext(plist),
        read_attributes=lambda query, attributes, ordinals: nullcontext(Result()),
        write_attribute=lambda query, attribute, ordinal, value, **kw: (
            writes.__setitem__((query, attribute), np.asarray(value)[0]), SimpleNamespace(wait=lambda: None))[1],
        advance_write_floor=lambda ordinal: SimpleNamespace(wait=lambda: None))
    paths = SimpleNamespace(create_path_list_from_strings=lambda items: nullcontext(items[0]),
                            intern_token=lambda name: name)
    instance = backend.PhysicsBackend.__new__(backend.PhysicsBackend)
    instance.np = np
    instance.stage = stage
    instance.ordinal = 4
    instance.ovstage = SimpleNamespace(PathDictionary=lambda stage: nullcontext(paths),
                                       OrdinalRange=SimpleNamespace(latest=lambda ordinal: ordinal),
                                       AttributeSemantic=SimpleNamespace(MATRIX='matrix'))
    instance.physx = SimpleNamespace(update_from_ovstage=lambda a, b: None, wait_all=lambda: None)
    instance._stage_value = lambda path, attribute: values.get((path, attribute))
    return instance, writes


def pose(translate=(8.0, 0.0, 6.0), orient=(0.0, 0.0, 0.0, 1.0)):
    return {'prim_path': '/World/Holder/Child', 'translate': list(translate), 'orient': list(orient)}


def test_parented_world_pose_is_written_parent_local():
    values = {('/World', 'omni:xform'): OFFSET, ('/World/Holder', 'omni:xform'): HOLDER}
    instance, writes = worker(values)
    instance._write_stage_poses({'/World/Holder/Child': pose()})
    world = writes[('/World/Holder/Child', 'omni:fabric:worldMatrix')]
    local = writes[('/World/Holder/Child', 'omni:xform')]
    assert np.allclose(world[3, :3], (8, 0, 6)) and np.allclose(world[:3, :3], np.eye(3))
    # The consumer composition reproduces the requested world pose, not the parent twice.
    assert np.allclose(local @ HOLDER @ OFFSET, world)
    assert not np.allclose(local, world)
    assert instance.ordinal == 5


def test_reset_xform_stack_on_body_writes_world_directly():
    values = {('/World/Holder', 'omni:xform'): HOLDER,
              ('/World/Holder/Child', 'omni:resetXformStack'): np.array([True])}
    instance, writes = worker(values)
    instance._write_stage_poses({'/World/Holder/Child': pose()})
    assert np.allclose(writes[('/World/Holder/Child', 'omni:xform')],
                       writes[('/World/Holder/Child', 'omni:fabric:worldMatrix')])


def test_ancestor_reset_stops_composition_and_missing_rows_are_identity():
    values = {('/World', 'omni:xform'): OFFSET, ('/World/Holder', 'omni:xform'): HOLDER,
              ('/World/Holder', 'omni:resetXformStack'): np.array([True])}
    instance, _ = worker(values)
    assert np.allclose(instance._parent_world('/World/Holder/Child'), HOLDER)
    assert np.allclose(instance._parent_world('/Root'), np.eye(4))
    assert np.allclose(worker({})[0]._parent_world('/Scope/Child'), np.eye(4))


@pytest.mark.parametrize('orient', [(0.0, 0.0, 0.7071068, 0.7071068), (0.3, -0.2, 0.1, 0.9)])
def test_unparented_body_keeps_world_matrix(orient):
    instance, writes = worker({('/World', 'omni:xform'): np.eye(4)})
    row = dict(pose(orient=orient), prim_path='/World/Body')
    instance._write_stage_poses({'/World/Body': row})
    assert np.allclose(writes[('/World/Body', 'omni:xform')], writes[('/World/Body', 'omni:fabric:worldMatrix')])

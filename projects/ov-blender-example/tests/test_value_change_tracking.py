# SPDX-License-Identifier: Apache-2.0
"""Change tracking certificates require acknowledged, dependency-complete state."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'addon'))
from ovrtx_blender_example.value_change_tracking import AcknowledgedValueLedger
import pytest


def pose(path, x):
    return {'prim_path': path, 'attribute_name': 'omni:xform', 'value_type': 'Matrix4d',
            'value': [[1,0,0,0], [0,1,0,0], [0,0,1,0], [x,0,0,1]]}


def test_parent_change_keeps_explicit_unchanged_child_override():
    ledger = AcknowledgedValueLedger()
    parent = pose('/World/Parent', 0); child = pose('/World/Parent/Child', 5)
    unrelated = pose('/World/Other', 10)
    ledger.acknowledge([parent, child, unrelated], epoch=1)
    moved = pose('/World/Parent', 20)
    assert ledger.changed([moved, child, unrelated], epoch=1) == [moved, child]


def test_parent_change_invalidates_child_until_accepted_receipt():
    ledger = AcknowledgedValueLedger()
    parent = pose('/World/Parent', 0); child = pose('/World/Parent/Child', 5)
    ledger.acknowledge([parent, child], epoch=1)
    moved = pose('/World/Parent', 20)
    ledger.acknowledge([moved], epoch=1)
    assert ledger.changed([child], epoch=1) == [child]
    inherited = pose('/World/Parent/Child', 25)
    ledger.acknowledge([moved, inherited], epoch=1)
    assert ledger.changed([moved, inherited], epoch=1) == []


def test_mutable_values_are_snapshotted_only_on_ack():
    ledger = AcknowledgedValueLedger(); row = pose('/World/A', 1)
    assert ledger.changed([row], epoch=1) == [row]
    assert ledger.changed([row], epoch=1) == [row]
    ledger.acknowledge([row], epoch=1)
    row['value'][3][0] = 2
    assert ledger.changed([row], epoch=1) == [row]
    assert ledger.changed([pose('/World/A', 1)], epoch=1) == []


def test_repeated_property_sequence_preserves_last_requested_value():
    ledger = AcknowledgedValueLedger()
    a, b = pose('/World/A', 1), pose('/World/A', 2)
    ledger.acknowledge([a], epoch=1)
    assert ledger.changed([a, b, a], epoch=1) == [b, a]


def test_scoped_graph_invalidation_preserves_unrelated_certificates():
    ledger = AcknowledgedValueLedger()
    material = dict(prim_path='/Materials/A/Shader', attribute_name='inputs:roughness', value_type='Float', value=.5)
    other = pose('/World/Camera', 2)
    similarly_named = dict(material, prim_path='/Materials/AB/Shader')
    ledger.acknowledge([material, other, similarly_named], epoch=1)
    assert ledger.invalidate_subtrees(['/Materials/A'], epoch=1) == 1
    assert ledger.changed([material, other, similarly_named], epoch=1) == [material]


@pytest.mark.parametrize('reset', [True, False])
def test_moving_world_pose_retains_requested_reset_boundary(reset):
    ledger = AcknowledgedValueLedger()
    initial = pose('/World/Camera', 1)
    boundary = dict(prim_path='/World/Camera', attribute_name='omni:resetXformStack',
                    value_type='Bool', value=reset)
    ledger.acknowledge([initial, boundary], epoch=1)
    moved = pose('/World/Camera', 2)
    assert ledger.changed([moved, boundary], epoch=1) == [moved, boundary]
    assert ledger.changed([boundary, moved], epoch=1) == [boundary, moved]
    ledger.acknowledge([moved, boundary], epoch=1)
    assert ledger.changed([moved, boundary], epoch=1) == []


def test_pose_update_does_not_invent_an_unsupplied_boundary():
    ledger = AcknowledgedValueLedger()
    boundary = dict(prim_path='/World/Camera', attribute_name='omni:resetXformStack',
                    value_type='Bool', value=True)
    ledger.acknowledge([pose('/World/Camera', 1), boundary], epoch=1)
    moved = pose('/World/Camera', 2)
    assert ledger.changed([moved], epoch=1) == [moved]


def test_pose_boundary_keeps_sequential_changes_and_filters_duplicates():
    ledger = AcknowledgedValueLedger()
    boundary = dict(prim_path='/World/Camera', attribute_name='omni:resetXformStack',
                    value_type='Bool', value=True)
    ledger.acknowledge([pose('/World/Camera', 1), boundary], epoch=1)
    moved = pose('/World/Camera', 2)
    local = dict(boundary, value=False)
    assert ledger.changed([moved, boundary, boundary, local, boundary], epoch=1) == [moved, boundary, local, boundary]

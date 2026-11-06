"""Configuration fingerprints preserve edits but ignore exact zero's sign."""
from types import SimpleNamespace
import math

import pytest

from ovrtx_blender_example.physics_bake_configuration import (
    channel_configuration, TRANSFORMS, DELTAS,
)
from ovrtx_blender_example.physics_bake_capture import BakeCaptureError
from ovrtx_blender_example import physics_pose_presentation


def owner():
    values = dict(location=[0.0]*3, rotation_euler=[0.0]*3,
        rotation_quaternion=[1.0, 0.0, 0.0, 0.0],
        rotation_axis_angle=[0.0, 0.0, 1.0, 0.0], scale=[1.0]*3,
        delta_location=[0.0]*3, delta_rotation_euler=[0.0]*3,
        delta_rotation_quaternion=[1.0, 0.0, 0.0, 0.0], delta_scale=[1.0]*3)
    return SimpleNamespace(animation_data=None, **values)


def fingerprint(obj):
    return channel_configuration(obj, obj, TRANSFORMS + DELTAS)


@pytest.mark.parametrize('name', TRANSFORMS + DELTAS)
def test_signed_zero_equivalence_for_every_static_channel(name):
    obj = owner()
    getattr(obj, name)[0] = 0.0
    expected = fingerprint(obj)
    getattr(obj, name)[0] = -0.0
    assert math.copysign(1.0, getattr(obj, name)[0]) == -1.0
    assert fingerprint(obj) == expected


@pytest.mark.parametrize('value', [1e-300, -1e-300, 1e-7, -1e-7, 1.0, -1.0])
def test_every_nonzero_edit_remains_distinct(value):
    obj = owner()
    expected = fingerprint(obj)
    obj.rotation_euler[1] = value
    assert fingerprint(obj) != expected


def test_observed_unselected_striker_signed_zero_roundtrip():
    obj = owner()
    obj.location[0] = -1.0
    obj.rotation_euler[1] = -0.0
    captured = fingerprint(obj)
    obj.rotation_euler[1] = 0.0
    assert fingerprint(obj) == captured
    obj.location[1] = 3.0
    assert fingerprint(obj) != captured


def test_animation_definitions_and_binding_still_change_fingerprint(monkeypatch):
    obj = owner()
    prop = SimpleNamespace(identifier='control_value', type='FLOAT',
                           is_readonly=False, is_array=False)
    curve = SimpleNamespace(data_path='location', array_index=0, mute=False,
        driver=None, control_value=1.0,
        bl_rna=SimpleNamespace(properties=[prop]))
    obj.animation_data = SimpleNamespace(action=SimpleNamespace(name_full='Action'),
        action_slot=SimpleNamespace(identifier='slot'), action_blend_type='REPLACE',
        action_extrapolation='HOLD', action_influence=1.0, use_nla=True, nla_tracks=[])
    monkeypatch.setattr(physics_pose_presentation, '_curves', lambda _: [curve])
    expected = fingerprint(obj)
    obj.location[0] = 3.0  # Evaluated animated value is deliberately not a definition.
    assert fingerprint(obj) == expected
    curve.control_value = 2.0
    assert fingerprint(obj) != expected
    curve.control_value = 1.0
    obj.animation_data.action_influence = 0.5
    assert fingerprint(obj) != expected


def test_external_driver_refusal_is_unchanged(monkeypatch):
    obj = owner()
    curve = SimpleNamespace(data_path='location', array_index=0, mute=False,
                            driver=object())
    monkeypatch.setattr(physics_pose_presentation, '_curves', lambda _: [curve])
    with pytest.raises(BakeCaptureError, match='external drivers'):
        fingerprint(obj)

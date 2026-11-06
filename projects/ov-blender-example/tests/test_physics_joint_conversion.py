import math
from types import SimpleNamespace as N
import pytest
from ovrtx_blender_example.physics_frames import PhysicsUnits
from ovrtx_blender_example.physics_joint_conversion import JointError, limits_for_axis


def test_explicit_free_and_locked_limits():
    units = PhysicsUnits(.01)
    assert limits_for_axis(N(state="FREE"), units, angular=True) is None
    assert limits_for_axis(N(state="LOCKED"), units, angular=False) == (1, -1)


def test_concrete_degree_and_meter_limits():
    units = PhysicsUnits(.01)
    assert limits_for_axis(N(state="LIMITED", lower=-100, upper=200), units, angular=False) == (-1, 2)
    low, high = limits_for_axis(N(state="LIMITED", lower=-math.pi/2, upper=math.pi), units, angular=True)
    assert low == pytest.approx(-90)
    assert high == pytest.approx(180)


@pytest.mark.parametrize("low,high", [(2, 1), (math.nan, 1), (0, math.inf), (-math.inf, 2)])
def test_invalid_limited_bounds_fail(low, high):
    with pytest.raises(JointError):
        limits_for_axis(N(state="LIMITED", lower=low, upper=high), PhysicsUnits(), angular=False)

from types import SimpleNamespace
from ovrtx_blender_example.physics_pose_presentation import _curves


def test_legacy_actions_without_slots_and_owner_drivers_remain_visible():
    curve, driver = object(), object()
    action = SimpleNamespace(fcurves=(curve,))
    animation = SimpleNamespace(action=action, drivers=(driver,), nla_tracks=())
    assert tuple(_curves(animation)) == (driver, curve)


def test_legacy_action_on_slot_capable_blender_uses_legacy_fcurves():
    curve = object()
    action = SimpleNamespace(slots=(), is_action_legacy=True, fcurves=(curve,))
    animation = SimpleNamespace(action=action, action_slot=None, drivers=(), nla_tracks=())
    assert tuple(_curves(animation)) == (curve,)


def test_unbound_modern_action_has_no_output_channels():
    action = SimpleNamespace(slots=(object(),), is_action_legacy=False, layers=(object(),))
    animation = SimpleNamespace(action=action, action_slot=None, drivers=(), nla_tracks=())
    assert tuple(_curves(animation)) == ()

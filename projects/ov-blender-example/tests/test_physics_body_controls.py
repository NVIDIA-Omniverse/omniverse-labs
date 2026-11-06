import math
import pytest
from ovrtx_blender_example.physics_body_controls import positive, FIELD_MATRIX
from ovrtx_blender_example.physics_capabilities import body_control_capability, _body_admitted_devices

@pytest.mark.parametrize("value", [0, -1, math.inf, -math.inf, math.nan])
def test_nonphysical_mass_and_dimensions_are_rejected(value):
    with pytest.raises(ValueError):
        positive(value, "mass")

def test_damping_zero_is_admitted_but_negative_is_not():
    assert positive(0, "damping", zero=True) == 0
    with pytest.raises(ValueError):
        positive(-1, "damping", zero=True)

@pytest.mark.parametrize("name", ["CCD", "sleep", "solver", "combine_mode", "filters", "telemetry"])
def test_unvalidated_controls_are_not_authorable(name):
    assert not body_control_capability(name).authorable

def test_body_control_admission_never_claims_live_transport():
    _body_admitted_devices.clear()
    assert not body_control_capability("BOX", "cpu").restart_simulation
    _body_admitted_devices.add("cpu")
    try:
        assert body_control_capability("BOX", "cpu").restart_simulation
        assert not body_control_capability("BOX", "cpu").live
        assert not body_control_capability("BOX", "gpu").restart_simulation
    finally:
        _body_admitted_devices.clear()

@pytest.mark.parametrize("field", ["mass_mode", "mass_kg", "density", "center_of_mass", "diagonal_inertia", "principal_axes", "initial_linear_velocity", "initial_angular_velocity", "linear_damping", "angular_damping", "static_friction", "dynamic_friction", "restitution", "gravity_direction", "gravity_magnitude"])
def test_each_exposed_field_has_explicit_authoring_admission(field):
    assert body_control_capability(field).authorable

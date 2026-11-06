# SPDX-License-Identifier: Apache-2.0
"""Shared initial/live UsdLux light semantics."""
import math
from types import SimpleNamespace

import pytest

from ovrtx_blender_example import light_value_conversion as lights
from ovrtx_blender_example.light_scene_layer import _LightRecord, _layer_from_records


@pytest.mark.parametrize("family", ["POINT", "SPOT", "AREA", "SUN"])
@pytest.mark.parametrize("exposure", [-3.0, 0.0, 4.0])
def test_light_exposure_remains_native_stops_in_initial_and_live_layer(family, exposure):
    data = SimpleNamespace(type=family, energy=10.0, exposure=exposure,
                           shadow_soft_size=0.5, size=2.0, shape="SQUARE")
    obj = SimpleNamespace(data=data, scale=(1.0, 1.0, 1.0))
    classification = lights.classify_field(data, "exposure")
    assert classification.status == lights.STATUS_SUPPORTED
    attrs = lights.usd_attribute_values(obj)
    exposure_attr = next(attr for attr in attrs if attr.name == "inputs:exposure")
    assert exposure_attr.value_type == "Float"
    assert exposure_attr.value == exposure
    assert exposure_attr.blender_property_path == "exposure"
    # Energy is authored independently: do not bake stops twice into intensity.
    data.exposure = 0.0
    assert next(attr.value for attr in attrs if attr.name == "inputs:intensity") == next(
        attr.value for attr in lights.usd_attribute_values(obj) if attr.name == "inputs:intensity")
    layer = _layer_from_records((_LightRecord("Key", "/World/Key", lights.exported_light_family(family, "SQUARE"), attrs),))
    assert ("/World/Key", "inputs:exposure") in layer.authored_properties
    assert f"float inputs:exposure = {exposure:.9g}" in layer.layer_body


def test_unsupported_field_snapshot_reports_only_real_field_changes():
    data = SimpleNamespace(type="AREA", normalize=True, use_shadow=True, exposure=0.0,
                           diffuse_factor=1.0, energy=10.0)
    baseline = lights.unsupported_field_values(data)
    assert "exposure" not in baseline
    assert lights.unsupported_field_changes(data, baseline) == ()
    data.exposure = 2.0
    assert lights.unsupported_field_changes(data, baseline) == ()
    data.use_shadow = False
    change, = lights.unsupported_field_changes(data, baseline)
    assert change == {"field": "use_shadow", "value": False, "previous_value": True,
                      "reason": "unsupported_light_shadow_toggle"}


def _sun(energy=3.0, angle=0.1):
    data = SimpleNamespace(type="SUN", energy=energy, exposure=0.0, angle=angle, color=(1.0, 1.0, 1.0),
                           use_temperature=False, normalize=True, node_tree=None)
    return SimpleNamespace(data=data, scale=(1.0, 1.0, 1.0))


def test_sun_intensity_matches_cycles_irradiance_scale():
    # Measured on a 0.8-albedo receiver: Cycles S W/m^2 gives 0.8*S/pi radiance; the
    # sun uses the same 120*pi scale as other lights (it was 120*pi^2, 3.19x too bright).
    attrs = {a.name: a for a in lights.usd_attribute_values(_sun(energy=3.0))}
    assert attrs["inputs:intensity"].value == pytest.approx(3.0 * 120.0 * math.pi)
    assert attrs["inputs:intensity"].metadata["formula"] == "120*pi*energy"


def test_sun_angle_is_the_full_angular_diameter_in_degrees():
    # Measured: OVRTX treats inputs:angle as the full diameter, like Blender's sun angle.
    attrs = {a.name: a for a in lights.usd_attribute_values(_sun(angle=math.radians(10.0)))}
    assert attrs["inputs:angle"].value == pytest.approx(10.0)


def _spot(size_deg, blend):
    data = SimpleNamespace(type="SPOT", energy=100.0, exposure=0.0, shadow_soft_size=0.1,
                           spot_size=math.radians(size_deg), spot_blend=blend, color=(1.0, 1.0, 1.0),
                           use_temperature=False, normalize=True, node_tree=None)
    return SimpleNamespace(data=data, scale=(1.0, 1.0, 1.0))


def test_spot_blend_becomes_a_hard_cone_at_the_blend_midpoint_with_zero_softness():
    attrs = {a.name: a for a in lights.usd_attribute_values(_spot(90.0, 0.0))}
    assert attrs["inputs:shaping:cone:angle"].value == pytest.approx(45.0)
    assert attrs["inputs:shaping:cone:softness"].value == 0.0
    # Cycles' smoothstep edge spans cos(45deg) .. cos(45deg) + 0.2*(1 - cos(45deg)).
    cos_half = math.cos(math.radians(45.0))
    expected = math.degrees(math.acos(cos_half + 0.5 * (1.0 - cos_half) * 0.2))
    attrs = {a.name: a for a in lights.usd_attribute_values(_spot(90.0, 0.2))}
    assert attrs["inputs:shaping:cone:angle"].value == pytest.approx(expected)
    assert 40.0 < expected < 45.0
    assert attrs["inputs:shaping:cone:softness"].value == 0.0


def test_spot_blend_edits_update_both_cone_attributes():
    spot = _spot(60.0, 0.15).data
    for field in ("spot_size", "spot_blend"):
        classification = lights.classify_field(spot, field)
        assert classification.status == lights.STATUS_SUPPORTED
        assert set(classification.usd_attributes) == {"inputs:shaping:cone:angle", "inputs:shaping:cone:softness"}

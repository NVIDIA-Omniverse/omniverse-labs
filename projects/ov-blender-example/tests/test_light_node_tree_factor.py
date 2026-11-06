# SPDX-License-Identifier: Apache-2.0
"""Constant light node trees scale intensity and colour the way Cycles does."""
from pathlib import Path
from types import SimpleNamespace as N
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from ovrtx_blender_example import light_value_conversion as lights  # noqa: E402


class _Inputs(dict):
    def get(self, name, default=None):
        return super().get(name, default)


def _socket(name, value=None):
    return N(name=name, default_value=value, links=[])


def _node(kind, **inputs):
    return N(type=kind, is_active_output=True, inputs=_Inputs({k: _socket(k, v) for k, v in inputs.items()}))


def _link(target_node, socket_name, source, output):
    target_node.inputs[socket_name].links.append(N(from_node=source, from_socket=N(name=output)))


def _light(*, strength=1.0, color=(1.0, 1.0, 1.0, 1.0), falloff=None, temperature=None, energy=10.0,
           light_type="POINT", extra_strength_node=None):
    emission = _node("EMISSION", Color=color, Strength=strength, Weight=0.0)
    output = _node("OUTPUT_LIGHT", Surface=None)
    _link(output, "Surface", emission, "Emission")
    nodes = [output, emission]
    if falloff is not None:
        output_name, value = falloff
        node = _node("LIGHT_FALLOFF", Strength=value, Smooth=0.0)
        _link(emission, "Strength", node, output_name)
        nodes.append(node)
    if extra_strength_node is not None:
        _link(emission, "Strength", extra_strength_node, "Value")
        nodes.append(extra_strength_node)
    if temperature is not None:
        node = _node("BLACKBODY", Temperature=temperature)
        _link(emission, "Color", node, "Color")
        nodes.append(node)
    data = N(type=light_type, energy=energy, exposure=0.0, shadow_soft_size=0.1, size=1.0, shape="SQUARE",
             color=(1.0, 1.0, 1.0), use_temperature=False, normalize=True, node_tree=N(nodes=nodes))
    return N(data=data, scale=(1.0, 1.0, 1.0))


def _values(obj):
    attrs = {attr.name: attr for attr in lights.usd_attribute_values(obj)}
    return attrs["inputs:intensity"].value, tuple(attrs["inputs:color"].value), attrs


def _no_nodes(obj):
    obj.data.node_tree = None
    return obj


def test_light_without_a_node_tree_is_unchanged():
    plain = _no_nodes(_light())
    intensity, color, attrs = _values(plain)
    assert color == (1.0, 1.0, 1.0)
    assert attrs["inputs:intensity"].metadata["light_node_tree"]["status"] == "absent"
    assert intensity == _values(_light())[0]  # strength 1, white: identical


def test_constant_emission_strength_and_color_scale_the_light():
    base, _, _ = _values(_no_nodes(_light()))
    intensity, color, _ = _values(_light(strength=15.0, color=(0.5, 1.0, 1.0, 1.0)))
    assert intensity == pytest.approx(base * 15.0)
    assert color == pytest.approx((0.5, 1.0, 1.0))


def test_quadratic_falloff_with_blackbody_matches_the_classroom_lamp():
    base, _, _ = _values(_no_nodes(_light()))
    intensity, color, attrs = _values(_light(falloff=("Quadratic", 20.0), temperature=5000.0))
    assert intensity == pytest.approx(base * 20.0)
    # Blender 5.2.1 Light.temperature_color at 5000 K.
    assert color == pytest.approx((1.2131, 0.9608, 0.7622), abs=2e-4)
    assert attrs["inputs:color"].metadata["light_node_tree"]["status"] == "applied"


@pytest.mark.parametrize("kelvin,expected", [
    (1500, (3.2643, 0.434, 0.0)), (3000, (1.7708, 0.8443, 0.2722)), (5800, (1.1093, 0.9759, 0.9179)),
    (6500, (1.0426, 0.9841, 1.0353)), (10000, (0.8714, 0.9949, 1.4305))])
def test_blackbody_matches_blender_temperature_color(kelvin, expected):
    assert lights.blackbody_rgb(kelvin) == pytest.approx(expected, abs=2e-4)


@pytest.mark.parametrize("case,reason", [
    ("linear_falloff", "unsupported_light_falloff:Linear"),
    ("constant_falloff", "unsupported_light_falloff:Constant"),
    ("math_strength", "unsupported_light_node:MATH"),
])
def test_other_strength_graphs_keep_energy_only(case, reason):
    base, _, _ = _values(_no_nodes(_light()))
    if case == "math_strength":
        obj = _light(extra_strength_node=_node("MATH", Value=3.0))
    else:
        obj = _light(falloff=("Linear" if case == "linear_falloff" else "Constant", 20.0))
    intensity, color, attrs = _values(obj)
    assert intensity == pytest.approx(base)
    assert color == (1.0, 1.0, 1.0)
    assert attrs["inputs:intensity"].metadata["light_node_tree"] == {
        "status": "unsupported", "reason": reason, "strength": 1.0, "color": (1.0, 1.0, 1.0)}


def test_linked_blackbody_temperature_keeps_energy_only():
    obj = _light(temperature=5000.0)
    blackbody = obj.data.node_tree.nodes[-1]
    _link(blackbody, "Temperature", _node("VALUE"), "Value")
    _, color, attrs = _values(obj)
    assert color == (1.0, 1.0, 1.0)
    assert attrs["inputs:color"].metadata["light_node_tree"]["reason"] == "linked_blackbody_temperature"


def test_sun_strength_multiplies_irradiance():
    base, _, _ = _values(_no_nodes(_light(light_type="SUN", energy=1.0)))
    intensity, _, _ = _values(_light(light_type="SUN", energy=1.0, strength=15.0, temperature=5800.0))
    assert intensity == pytest.approx(base * 15.0)


def test_falloff_smooth_is_reported_as_an_approximation():
    obj = _light(falloff=("Quadratic", 20.0))
    falloff = obj.data.node_tree.nodes[-1]
    falloff.inputs["Smooth"].default_value = 1.0
    intensity, _, attrs = _values(obj)
    assert intensity == pytest.approx(_values(_no_nodes(_light()))[0] * 20.0)
    info = attrs["inputs:intensity"].metadata["light_node_tree"]
    assert info["status"] == "approximated" and info["reason"] == "light_falloff_smooth:1"


@pytest.mark.parametrize("muted", ["node", "link"])
def test_muted_nodes_and_links_are_not_evaluated(muted):
    obj = _light(falloff=("Quadratic", 20.0))
    emission = obj.data.node_tree.nodes[1]
    link = emission.inputs["Strength"].links[0]
    if muted == "node":
        link.from_node.mute = True
    else:
        link.is_muted = True
    intensity, _, attrs = _values(obj)
    assert intensity == pytest.approx(_values(_no_nodes(_light()))[0])
    assert attrs["inputs:intensity"].metadata["light_node_tree"]["reason"] == "unsupported_light_node:MUTED"

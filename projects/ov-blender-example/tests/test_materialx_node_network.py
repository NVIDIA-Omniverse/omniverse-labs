# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MaterialX node-network translation of Blender colour/scalar graphs (n-04-05, n-04-06, n-04-08)."""
import colorsys
import math
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from test_materialx_openpbr_conversion import (  # noqa: E402
    _Binding,
    _Material,
    _Node,
    _Socket,
    _convert_bindings,
    _diffuse_material,
    _link,
    _mapping,
    _principled_material,
    _tex_coord_into,
    _texture_node,
)
sys.path.insert(0, str(ROOT))
from ovrtx_blender_example import material_graph_updates as graph_updates  # noqa: E402
from ovrtx_blender_example import materialx_node_network as network  # noqa: E402
from ovrtx_blender_example import materialx_openpbr_conversion as conversion  # noqa: E402

# --------------------------------------------------------------------------
# A small interpreter for the emitted MaterialX nodes, to check numerics.
# --------------------------------------------------------------------------


def _vec(value):
    if isinstance(value, (tuple, list)):
        return value[0] if len(value) == 1 else tuple(value)
    return value


def _lift(function):
    def apply(*args):
        if any(isinstance(arg, tuple) for arg in args):
            size = max(len(arg) for arg in args if isinstance(arg, tuple))
            expanded = [arg if isinstance(arg, tuple) else (arg,) * size for arg in args]
            return tuple(function(*items) for items in zip(*expanded))
        return function(*args)
    return apply


def _hsv_adjust(args):
    r, g, b = args["in"]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    shift, sat, val = args["amount"]
    return colorsys.hsv_to_rgb((h + shift) % 1.0, s * sat, v * val)


def _range(args):
    def one(x, low, high, out_low, out_high):
        t = (x - low) / (high - low)
        t = min(max(t, 0.0), 1.0) if args.get("doclamp") else t
        return out_low + t * (out_high - out_low)
    return _lift(one)(args["in"], args["inlow"], args["inhigh"], args["outlow"], args["outhigh"])


def _overlay(args):
    def one(fg, bg, mix):
        blended = 2.0 * fg * bg if bg < 0.5 else 1.0 - 2.0 * (1.0 - fg) * (1.0 - bg)
        return bg * (1.0 - mix) + blended * mix
    return _lift(one)(args["fg"], args["bg"], args["mix"])


def _screen(args):
    def one(fg, bg, mix):
        blended = 1.0 - (1.0 - fg) * (1.0 - bg)
        return bg * (1.0 - mix) + blended * mix
    return _lift(one)(args["fg"], args["bg"], args["mix"])


_OPS = {
    "ND_range_float": _range,
    "ND_range_color3": _range,
    "ND_mix_float": lambda a: _lift(lambda f, b, m: b * (1 - m) + f * m)(a["fg"], a["bg"], a["mix"]),
    "ND_mix_color3": lambda a: _lift(lambda f, b, m: b * (1 - m) + f * m)(a["fg"], a["bg"], a["mix"]),
    "ND_mix_color3_color3": lambda a: _lift(lambda f, b, m: b * (1 - m) + f * m)(a["fg"], a["bg"], a["mix"]),
    "ND_multiply_color3": lambda a: _lift(lambda x, y: x * y)(a["in1"], a["in2"]),
    "ND_multiply_color3FA": lambda a: _lift(lambda x, y: x * y)(a["in1"], a["in2"]),
    "ND_multiply_float": lambda a: a["in1"] * a["in2"],
    "ND_add_color3": lambda a: _lift(lambda x, y: x + y)(a["in1"], a["in2"]),
    "ND_add_color3FA": lambda a: _lift(lambda x, y: x + y)(a["in1"], a["in2"]),
    "ND_add_float": lambda a: a["in1"] + a["in2"],
    "ND_subtract_color3": lambda a: _lift(lambda x, y: x - y)(a["in1"], a["in2"]),
    "ND_subtract_float": lambda a: a["in1"] - a["in2"],
    "ND_power_float": lambda a: a["in1"] ** a["in2"],
    "ND_power_color3FA": lambda a: _lift(lambda x, y: x ** y)(a["in1"], a["in2"]),
    "ND_absval_color3": lambda a: _lift(abs)(a["in"]),
    "ND_min_color3": lambda a: _lift(min)(a["in1"], a["in2"]),
    "ND_max_color3": lambda a: _lift(max)(a["in1"], a["in2"]),
    "ND_max_float": lambda a: max(a["in1"], a["in2"]),
    "ND_clamp_float": lambda a: min(max(a["in"], a["low"]), a["high"]),
    "ND_clamp_color3": lambda a: _lift(lambda x, lo, hi: min(max(x, lo), hi))(a["in"], a["low"], a["high"]),
    "ND_invert_color3": lambda a: _lift(lambda x: 1.0 - x)(a["in"]),
    "ND_hsvadjust_color3": _hsv_adjust,
    "ND_rgbtohsv_color3": lambda a: colorsys.rgb_to_hsv(*a["in"]),
    "ND_hsvtorgb_color3": lambda a: colorsys.hsv_to_rgb(a["in"][0] % 1.0, a["in"][1], a["in"][2]),
    "ND_overlay_color3": _overlay,
    "ND_screen_color3": _screen,
    "ND_convert_color3_vector3": lambda a: a["in"],
    "ND_texcoord_vector2": lambda a: (0.3, 0.6),
    "ND_convert_vector2_vector3": lambda a: (a["in"][0], a["in"][1], 0.0),
    "ND_multiply_vector3": lambda a: tuple(x * y for x, y in zip(a["in1"], a["in2"])),
    "ND_add_vector3": lambda a: tuple(x + y for x, y in zip(a["in1"], a["in2"])),
    "ND_subtract_vector3": lambda a: tuple(x - y for x, y in zip(a["in1"], a["in2"])),
    "ND_rotate3d_vector3": lambda a: _rotate3d(a["in"], a["amount"], a["axis"]),
    "ND_dotproduct_vector3": lambda a: sum(x * y for x, y in zip(a["in1"], a["in2"])),
    "ND_convert_float_color3": lambda a: (a["in"],) * 3,
    "ND_extract_color3": lambda a: a["in"][a["index"]],
    "ND_combine3_color3": lambda a: (a["in1"], a["in2"], a["in3"]),
    "ND_ifgreater_float": lambda a: a["in1"] if a["value1"] > a["value2"] else a["in2"],
}


def _evaluate(net, value, image=(0.5, 0.5, 0.5)):
    """Value of ``value`` with every image node returning ``image`` (a colour or a function of the path)."""
    if value.is_const:
        return _vec(value.const)
    by_name = {node["name"]: node for node in net.nodes}
    cache = {}

    def resolve(name):
        if name in cache:
            return cache[name]
        node = by_name[name]
        if node.get("image"):
            result = image(node["image"]["asset_path"]) if callable(image) else image
        else:
            args = {
                row["name"]: resolve(row["connect"]) if row["connect"] else _vec(row["const"])
                for row in node["inputs"]
            }
            result = _OPS[node["info_id"]](args)
        cache[name] = result
        return result

    return resolve(value.node)


def _rotate3d(vector, degrees, axis):
    """Rodrigues rotation of ``vector`` about a unit ``axis`` (counter-clockwise for positive angles)."""
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    dot = sum(v * a for v, a in zip(vector, axis))
    cross = (axis[1] * vector[2] - axis[2] * vector[1], axis[2] * vector[0] - axis[0] * vector[2],
             axis[0] * vector[1] - axis[1] * vector[0])
    return tuple(v * c + x * s + a * dot * (1 - c) for v, x, a in zip(vector, cross, axis))


def _approx(a, b, tolerance=1.0e-6):
    if isinstance(a, tuple):
        return all(abs(x - y) <= tolerance for x, y in zip(a, b))
    return abs(a - b) <= tolerance


# --------------------------------------------------------------------------
# Node builders
# --------------------------------------------------------------------------


def _image(tmp_path, name="Image Texture"):
    return _texture_node(name, tmp_path / (name.replace(" ", "_") + ".png"))


def _hue_sat(color_source, *, hue=0.5, saturation=1.0, value=1.0, fac=1.0):
    node = _Node("HUE_SAT", "Hue Saturation Value", [
        _Socket("Hue", hue), _Socket("Saturation", saturation), _Socket("Value", value),
        _Socket("Fac", fac), _Socket("Color", (0.8, 0.8, 0.8, 1.0), socket_type="RGBA")])
    _link(node.inputs["Color"], color_source, from_socket_name="Color")
    return node


def _mix(a=None, b=None, *, blend="MIX", fac=0.5, fac_source=None, a_value=(0.2, 0.4, 0.6, 1.0),
         b_value=(0.8, 0.2, 0.4, 1.0), clamp_factor=True):
    sockets = [_Socket("Factor", fac), _Socket("A", a_value, socket_type="RGBA"),
               _Socket("B", b_value, socket_type="RGBA")]
    for socket, identifier in zip(sockets, ("Factor_Float", "A_Color", "B_Color")):
        socket.identifier = identifier
    node = _Node("MIX", "Mix", sockets)
    node.blend_type, node.data_type, node.clamp_factor = blend, "RGBA", clamp_factor
    if fac_source is not None:
        _link(node.inputs["Factor"], fac_source, from_socket_name="Color")
    if a is not None:
        _link(node.inputs["A"], a, from_socket_name="Color")
    if b is not None:
        _link(node.inputs["B"], b, from_socket_name="Color")
    return node


def _curve_node(color_source, functions, *, fac=1.0):
    """RGB Curves with python callables (R, G, B, combined) standing in for the mapping."""
    curves = [object() for _ in range(4)]
    table = dict(zip(map(id, curves), functions))
    mapping = SimpleNamespace(curves=curves, evaluate=lambda curve, x: table[id(curve)](x),
                              black_level=(0.0, 0.0, 0.0), white_level=(1.0, 1.0, 1.0))
    node = _Node("CURVE_RGB", "RGB Curves", [_Socket("Fac", fac), _Socket("Color", (1.0, 1.0, 1.0, 1.0), socket_type="RGBA")])
    node.mapping = mapping
    _link(node.inputs["Color"], color_source, from_socket_name="Color")
    return node


def _ramp(source, stops, *, interpolation="LINEAR"):
    node = _Node("VALTORGB", "ColorRamp", [_Socket("Fac", 0.5)])
    elements = [SimpleNamespace(position=p, color=c) for p, c in stops]
    node.color_ramp = SimpleNamespace(elements=elements, interpolation=interpolation, color_mode="RGB",
                                      evaluate=lambda x: _ramp_reference(stops, x))
    _link(node.inputs["Fac"], source, from_socket_name="Color")
    return node


def _ramp_reference(stops, x):
    if x <= stops[0][0]:
        return stops[0][1]
    for (x0, c0), (x1, c1) in zip(stops, stops[1:]):
        if x <= x1:
            t = (x - x0) / (x1 - x0)
            return tuple(a + (b - a) * t for a, b in zip(c0, c1))
    return stops[-1][1]


def _math(operation, first, second=None, *, first_value=0.5, second_value=0.5, clamp=False):
    node = _Node("MATH", "Math", [_Socket("Value", first_value), _Socket("Value", second_value)])
    node.operation, node.use_clamp = operation, clamp
    if first is not None:
        _link(node.inputs.values()[0], first, from_socket_name="Color")
    if second is not None:
        _link(node.inputs.values()[1], second, from_socket_name="Color")
    return node


def _translate(socket, kind="color3"):
    return network.translate_socket("ND_t", socket, kind)


def _linked_socket(node, name="Color", socket_type="RGBA", output="Color"):
    socket = _Socket("Target", (0.0, 0.0, 0.0, 1.0), socket_type=socket_type)
    return _link(socket, node, from_socket_name=output)


# --------------------------------------------------------------------------
# Hue/Saturation, Mix, Color Ramp, RGB Curves (n-04-06)
# --------------------------------------------------------------------------


def test_hue_saturation_becomes_hsvadjust_with_blender_hue_centred_on_half(tmp_path) -> None:
    image = _image(tmp_path)
    node = _hue_sat(image, hue=0.65, saturation=0.8)

    net, value = _translate(_linked_socket(node))

    adjust = next(entry for entry in net.nodes if entry["info_id"] == "ND_hsvadjust_color3")
    amount = next(row for row in adjust["inputs"] if row["name"] == "amount")
    assert amount["const"] == pytest.approx((0.15, 0.8, 1.0))
    # Matches Blender's HSV: hue shifts by (hue - 0.5), saturation scales.
    expected = _hsv_adjust({"in": (0.8, 0.4, 0.2), "amount": (0.15, 0.8, 1.0)})
    assert _approx(_evaluate(net, value, image=(0.8, 0.4, 0.2)), expected)


def test_neutral_hue_saturation_adds_no_nodes(tmp_path) -> None:
    net, value = _translate(_linked_socket(_hue_sat(_image(tmp_path))))

    assert not any(entry["info_id"] == "ND_hsvadjust_color3" for entry in net.nodes)


def test_hue_saturation_factor_blends_with_the_input(tmp_path) -> None:
    node = _hue_sat(_image(tmp_path), hue=0.9, saturation=0.7, fac=0.25)

    net, value = _translate(_linked_socket(node))

    color = (0.8, 0.4, 0.2)
    adjusted = _hsv_adjust({"in": color, "amount": (0.4, 0.7, 1.0)})
    expected = tuple(c * 0.75 + a * 0.25 for c, a in zip(color, adjusted))
    assert _approx(_evaluate(net, value, image=color), expected)


@pytest.mark.parametrize("blend", ["MIX", "MULTIPLY", "ADD", "SUBTRACT", "DIFFERENCE", "DARKEN", "LIGHTEN", "SCREEN", "OVERLAY"])
def test_mix_blend_modes_match_blenders_formulas(tmp_path, blend) -> None:
    image = _image(tmp_path)
    mix = _mix(a=image, blend=blend, fac=0.6, b_value=(0.3, 0.7, 0.55, 1.0))
    a, b, fac = (0.6, 0.25, 0.8), (0.3, 0.7, 0.55), 0.6

    net, value = _translate(_linked_socket(mix))

    def blend_one(x, y):
        return {
            "MIX": y, "MULTIPLY": x * y, "ADD": x + y, "SUBTRACT": x - y, "DIFFERENCE": abs(x - y),
            "DARKEN": min(x, y), "LIGHTEN": max(x, y), "SCREEN": 1.0 - (1.0 - x) * (1.0 - y),
            "OVERLAY": 2.0 * x * y if x < 0.5 else 1.0 - 2.0 * (1.0 - x) * (1.0 - y),
        }[blend]

    expected = tuple(x * (1.0 - fac) + blend_one(x, y) * fac for x, y in zip(a, b))
    assert _approx(_evaluate(net, value, image=a), expected, 1.0e-6)


def test_mix_with_linked_factor_uses_the_factor_image(tmp_path) -> None:
    mask, base = _image(tmp_path, "Mask"), _image(tmp_path, "Base")
    mix = _mix(a=base, fac_source=mask, b_value=(1.0, 0.0, 0.0, 1.0))

    net, value = _translate(_linked_socket(mix))

    paths = [entry["image"]["asset_path"] for entry in net.nodes if entry.get("image")]
    assert len(paths) == 2
    result = _evaluate(net, value, image=lambda path: (0.5, 0.5, 0.5) if "Base" in path else (0.25, 0.25, 0.25))
    luminance = 0.25
    assert _approx(result, tuple(0.5 * (1 - luminance) + b * luminance for b in (1.0, 0.0, 0.0)), 1.0e-5)


def test_constant_mix_folds_without_nodes() -> None:
    rgb = _Node("RGB", "RGB", [], outputs=[_Socket("Color", (0.4, 0.4, 0.4, 1.0))])
    mix = _mix(a=rgb, blend="MULTIPLY", fac=1.0, b_value=(0.5, 0.25, 1.0, 1.0))

    net, value = _translate(_linked_socket(mix))

    assert net.nodes == [] and value.const == pytest.approx((0.2, 0.1, 0.4))


def test_mix_unsupported_blend_is_untranslatable(tmp_path) -> None:
    mix = _mix(a=_image(tmp_path), blend="COLOR_DODGE")

    with pytest.raises(network.Untranslatable, match="COLOR_DODGE"):
        _translate(_linked_socket(mix))


def test_mix_factor_is_clamped_when_blender_clamps_it(tmp_path) -> None:
    mask = _image(tmp_path, "Mask")
    mix = _mix(a=_image(tmp_path), fac_source=mask)

    net, _ = _translate(_linked_socket(mix))

    assert any(entry["info_id"] == "ND_clamp_float" for entry in net.nodes)


@pytest.mark.parametrize("interpolation", ["LINEAR", "CONSTANT"])
def test_color_ramp_matches_blender_stops(tmp_path, interpolation) -> None:
    stops = [(0.2, (0.9, 0.1, 0.1)), (0.5, (0.1, 0.9, 0.1)), (0.8, (0.1, 0.2, 0.9))]
    ramp = _ramp(_image(tmp_path), stops, interpolation=interpolation)
    net, value = _translate(_linked_socket(ramp))

    for fac in (0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 1.0):
        gray = (fac, fac, fac)
        result = _evaluate(net, value, image=gray)
        if interpolation == "LINEAR":
            expected = _ramp_reference(stops, fac)
        else:
            expected = [c for p, c in stops if p <= fac + 1.0e-4][-1] if fac >= stops[0][0] else stops[0][1]
        assert _approx(result, expected, 2.0e-3), (fac, result, expected)


def test_smooth_color_ramp_states_its_error_bound(tmp_path) -> None:
    stops = [(0.1, (0.0, 0.0, 0.0)), (0.9, (1.0, 1.0, 1.0))]
    ramp = _ramp(_image(tmp_path), stops, interpolation="EASE")
    ramp.color_ramp.evaluate = lambda x: (lambda t: (t * t * (3 - 2 * t),) * 3)(min(max((x - 0.1) / 0.8, 0.0), 1.0))

    net, value = _translate(_linked_socket(ramp))

    assert net.error_bounds and net.error_bounds[0].startswith("VALTORGB:EASE:max_abs_error<=")
    bound = float(net.error_bounds[0].rsplit("<=", 1)[1])
    assert bound <= network.CURVE_ERROR_BOUND
    for fac in [i / 40 for i in range(41)]:
        truth = ramp.color_ramp.evaluate(fac)[0]
        assert abs(_evaluate(net, value, image=(fac,) * 3)[0] - truth) <= bound + 1.0e-6


def test_constant_color_ramp_input_folds(tmp_path) -> None:
    value_node = _Node("VALUE", "Value", [], outputs=[_Socket("Value", 0.6)])
    ramp = _ramp(value_node, [(0.0, (0.0, 0.0, 0.0)), (1.0, (1.0, 0.5, 0.0))])
    ramp.inputs["Fac"].links[0].from_socket = SimpleNamespace(name="Value")

    net, value = _translate(_linked_socket(ramp))

    assert net.nodes == [] and value.const == pytest.approx((0.6, 0.3, 0.0))


def test_rgb_curves_use_the_curve_with_a_stated_error_bound(tmp_path) -> None:
    def curve(x):  # a smooth S-curve with overshoot-free shoulders
        return x * x * (3.0 - 2.0 * x)

    node = _curve_node(_image(tmp_path), [lambda x: x, lambda x: x, lambda x: x, curve])
    net, value = _translate(_linked_socket(node))

    assert net.error_bounds and net.error_bounds[0].startswith("CURVE_RGB:max_abs_error<=")
    bound = float(net.error_bounds[0].rsplit("<=", 1)[1])
    assert bound <= network.CURVE_ERROR_BOUND
    for x in [i / 50 for i in range(51)]:
        assert abs(_evaluate(net, value, image=(x,) * 3)[1] - curve(x)) <= bound + 1.0e-6


def test_rgb_curves_per_channel_curves_apply_after_the_combined_curve(tmp_path) -> None:
    node = _curve_node(_image(tmp_path), [lambda x: 1.0 - x, lambda x: x, lambda x: x * x, lambda x: min(1.0, x * 1.5)])

    net, value = _translate(_linked_socket(node))

    r, g, b = _evaluate(net, value, image=(0.4, 0.4, 0.4))
    assert (r, g, b) == pytest.approx((1.0 - 0.6, 0.6, 0.36), abs=network.CURVE_ERROR_BOUND)


def test_identity_rgb_curves_add_nothing(tmp_path) -> None:
    node = _curve_node(_image(tmp_path), [lambda x: x] * 4)

    net, _ = _translate(_linked_socket(node))

    assert not any("range" in entry["info_id"] for entry in net.nodes) and net.error_bounds == []


def test_rgb_curves_factor_blends_the_result(tmp_path) -> None:
    node = _curve_node(_image(tmp_path), [lambda x: x] * 3 + [lambda x: 0.0], fac=0.5)

    net, value = _translate(_linked_socket(node))

    assert _evaluate(net, value, image=(0.6,) * 3) == pytest.approx((0.3,) * 3, abs=network.CURVE_ERROR_BOUND)


def test_curve_without_a_readable_mapping_is_untranslatable(tmp_path) -> None:
    node = _curve_node(_image(tmp_path), [lambda x: x] * 4)
    node.mapping = None

    with pytest.raises(network.Untranslatable):
        _translate(_linked_socket(node))


# --------------------------------------------------------------------------
# Math, Object Info, Attribute, procedural textures (n-04-05)
# --------------------------------------------------------------------------


def test_math_folds_constants_and_clamps() -> None:
    value = _Node("VALUE", "Value", [], outputs=[_Socket("Value", 0.9)])
    node = _math("MULTIPLY", value, second_value=3.0, clamp=True)
    node.inputs.values()[0].links[0].from_socket = SimpleNamespace(name="Value")

    net, result = _translate(_linked_socket(node, output="Value"), "float")

    assert net.nodes == [] and result.const == (1.0,)


def test_math_comparison_uses_ifgreater(tmp_path) -> None:
    image = _image(tmp_path)
    node = _math("LESS_THAN", image, second_value=0.33)

    net, value = _translate(_linked_socket(node, output="Value"), "float")

    assert any(entry["info_id"] == "ND_ifgreater_float" for entry in net.nodes)
    assert _evaluate(net, value, image=(0.1, 0.1, 0.1)) == 1.0
    assert _evaluate(net, value, image=(0.9, 0.9, 0.9)) == 0.0


def test_object_info_random_is_frozen_and_reported() -> None:
    info = _Node("OBJECT_INFO", "Object Info", [], outputs=[_Socket("Random", 0.0)])
    node = _math("LESS_THAN", info, second_value=0.33)
    node.inputs.values()[0].links[0].from_socket = SimpleNamespace(name="Random")

    net, value = _translate(_linked_socket(node, output="Value"), "float")

    assert value.const == (0.0,) and "OBJECT_INFO.Random=0.5" in net.approximations


def test_attribute_reads_the_exported_colour_primvar() -> None:
    attribute = _Node("ATTRIBUTE", "Attribute", [], outputs=[_Socket("Color", (0, 0, 0, 1))])
    attribute.attribute_name, attribute.attribute_type = "Col", "GEOMETRY"

    net, value = _translate(_linked_socket(attribute))

    assert [entry["info_id"] for entry in net.nodes] == ["ND_geompropvalue_color4", "ND_convert_color4_color3"]
    assert net.nodes[0]["inputs"][0]["const"] == "Col"
    assert value.node == net.nodes[1]["name"]


def test_attribute_other_than_named_geometry_is_untranslatable() -> None:
    attribute = _Node("ATTRIBUTE", "Attribute", [], outputs=[_Socket("Color", (0, 0, 0, 1))])
    attribute.attribute_name, attribute.attribute_type = "Col", "OBJECT"

    with pytest.raises(network.Untranslatable):
        _translate(_linked_socket(attribute))


def _noise(vector_source=None, *, detail=2.0, roughness=0.5, scale=4.0, distortion=0.0):
    node = _Node("TEX_NOISE", "Noise Texture", [
        _Socket("Vector", (0.0, 0.0, 0.0)), _Socket("Scale", scale), _Socket("Detail", detail),
        _Socket("Roughness", roughness), _Socket("Lacunarity", 2.0), _Socket("Distortion", distortion)])
    node.noise_dimensions = "3D"
    if vector_source is not None:
        _link(node.inputs["Vector"], vector_source, from_socket_name="Vector")
    return node


def test_noise_becomes_normalised_fractal3d_and_is_reported() -> None:
    mapping = _mapping("TEXTURE", scale=(2.0, 2.0, 1.0))
    _tex_coord_into(mapping.inputs["Vector"])
    node = _noise(mapping, detail=2.0, roughness=0.5)

    net, value = _translate(_linked_socket(node, output="Fac"), "float")

    fractal = next(entry for entry in net.nodes if entry["info_id"] == "ND_fractal3d_float")
    rows = {row["name"]: row["const"] for row in fractal["inputs"]}
    assert rows["octaves"] == 3 and _vec(rows["diminish"]) == 0.5 and _vec(rows["lacunarity"]) == 2.0
    assert _vec(rows["amplitude"]) == pytest.approx(0.5 / 1.75)
    assert "TEX_NOISE" in net.approximations


def test_noise_without_coordinates_uses_its_mean_and_says_so() -> None:
    net, value = _translate(_linked_socket(_noise(), output="Fac"), "float")

    assert value.const == (0.5,) and net.nodes == []
    assert any("coordinates_unavailable" in text for text in net.approximations)


def test_noise_distortion_is_reported() -> None:
    mapping = _mapping("POINT")
    _tex_coord_into(mapping.inputs["Vector"])

    net, _ = _translate(_linked_socket(_noise(mapping, distortion=0.4), output="Fac"), "float")

    assert "TEX_NOISE.distortion_ignored" in net.approximations


def test_wave_bands_use_a_sine_and_are_reported() -> None:
    mapping = _mapping("POINT")
    _tex_coord_into(mapping.inputs["Vector"])
    wave = _Node("TEX_WAVE", "Wave Texture", [
        _Socket("Vector", (0.0, 0.0, 0.0)), _Socket("Scale", 5.0), _Socket("Distortion", 0.0),
        _Socket("Detail", 0.0), _Socket("Phase Offset", 0.0)])
    wave.wave_type, wave.wave_profile, wave.bands_direction = "BANDS", "SIN", "X"
    _link(wave.inputs["Vector"], mapping, from_socket_name="Vector")

    net, _ = _translate(_linked_socket(wave, output="Fac"), "float")

    assert any(entry["info_id"] == "ND_sin_float" for entry in net.nodes)
    assert "TEX_WAVE" in net.approximations


def test_unknown_nodes_and_image_alpha_are_untranslatable(tmp_path) -> None:
    with pytest.raises(network.Untranslatable, match="unsupported_node:BEVEL"):
        _translate(_linked_socket(_Node("BEVEL", "Bevel", [])))
    with pytest.raises(network.Untranslatable, match="alpha|Alpha|output"):
        _translate(_linked_socket(_image(tmp_path), output="Alpha"), "float")


# --------------------------------------------------------------------------
# Image extension modes and named UV maps (n-04-08)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("extension,mode", [("EXTEND", "clamp"), ("CLIP", "constant"), ("MIRROR", "mirror")])
def test_image_extension_modes_author_address_modes(tmp_path, extension, mode) -> None:
    texture = _image(tmp_path)
    texture.extension = extension
    material = _diffuse_material("Edge", texture)

    result = _convert_bindings([_Binding(material, ("/World/Geom/Edge",))])

    body = result.value.layer_body
    assert f'string inputs:uaddressmode = "{mode}"' in body
    assert f'string inputs:vaddressmode = "{mode}"' in body


def test_repeat_extension_authors_no_address_mode(tmp_path) -> None:
    texture = _image(tmp_path)
    texture.extension = "REPEAT"

    result = _convert_bindings([_Binding(_diffuse_material("Tile", texture), ("/World/Geom/Tile",))])

    assert "addressmode" not in result.value.layer_body


def test_unknown_extension_is_rejected_with_a_diagnostic(tmp_path) -> None:
    texture = _image(tmp_path)
    texture.extension = "SPIRAL"

    result = _convert_bindings([_Binding(_diffuse_material("Odd", texture), ("/World/Geom/Odd",))])

    assert result.status.value == "error"
    assert "image extension mode is not translated: Image Texture" in str(result.diagnostics)


def test_extension_reaches_networked_images_too(tmp_path) -> None:
    texture = _image(tmp_path)
    texture.extension = "EXTEND"
    material = _diffuse_material("Net", _hue_sat(texture, hue=0.7), extra=(texture,))

    result = _convert_bindings([_Binding(material, ("/World/Geom/Net",))])

    assert 'string inputs:uaddressmode = "clamp"' in result.value.layer_body
    assert "ND_hsvadjust_color3" in result.value.layer_body


def _uv_map_material(tmp_path, uv_name):
    texture = _image(tmp_path)
    uv = _Node("UVMAP", "UV Map", [])
    uv.uv_map = uv_name
    _link(texture.inputs["Vector"], uv, from_socket_name="UV")
    return _diffuse_material("Named", texture, extra=(uv,))


@pytest.mark.parametrize(
    "layouts,expected_primvar",
    [
        ([("UVMap", ("UVMap", "Lightmap"))], ""),  # the exported render UV: st
        ([("UVMap", ("UVMap", "Lightmap")), ("UVMap", ("UVMap", "Lightmap"))], ""),
    ],
)
def test_named_uv_map_that_is_every_meshs_render_uv_reads_the_default_uv(tmp_path, monkeypatch, layouts, expected_primvar) -> None:
    monkeypatch.setattr(conversion, "_material_uv_layouts", lambda material: layouts)

    result = _convert_bindings([_Binding(_uv_map_material(tmp_path, "UVMap"), ("/World/Geom/Named",))])

    assert result.status.value == "ok"
    assert "geompropvalue" not in result.value.layer_body
    assert "ND_texcoord_vector2" not in result.value.layer_body


def test_named_uv_map_that_is_no_meshs_render_uv_reads_its_primvar(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(conversion, "_material_uv_layouts", lambda material: [("UVMap", ("UVMap", "Light map"))])

    result = _convert_bindings([_Binding(_uv_map_material(tmp_path, "Light map"), ("/World/Geom/Named",))])

    body = result.value.layer_body
    assert 'info:id = "ND_geompropvalue_vector2"' in body
    assert 'string inputs:geomprop = "Light_map"' in body
    assert "inputs:texcoord.connect = </OVRTX_Materials/Named/ND_image_color3_base_color_uv.outputs:out>" in body


@pytest.mark.parametrize(
    "layouts,reason",
    [
        ([("UVMap", ("UVMap", "Lightmap")), ("Lightmap", ("UVMap", "Lightmap"))], "meshes disagree on the render UV"),
        ([("UVMap", ("UVMap",))], "a bound mesh lacks the map"),
        ([], "no mesh bound"),
        (None, "meshes unknown"),
    ],
)
def test_named_uv_map_is_rejected_when_meshes_cannot_agree(tmp_path, monkeypatch, layouts, reason) -> None:
    monkeypatch.setattr(conversion, "_material_uv_layouts", lambda material: layouts)

    result = _convert_bindings([_Binding(_uv_map_material(tmp_path, "Lightmap"), ("/World/Geom/Named",))])

    assert result.status.value == "error", reason
    assert "image coordinate graph is not translated: Image Texture" in str(result.diagnostics)


def test_named_uv_map_with_a_mapping_keeps_the_scale(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(conversion, "_material_uv_layouts", lambda material: [("UVMap", ("UVMap", "Lightmap"))])
    texture = _image(tmp_path)
    mapping = _mapping("POINT", scale=(2.0, 3.0, 1.0))
    uv = _Node("UVMAP", "UV Map", [])
    uv.uv_map = "Lightmap"
    _link(mapping.inputs["Vector"], uv, from_socket_name="UV")
    _link(texture.inputs["Vector"], mapping, from_socket_name="Vector")

    result = _convert_bindings([_Binding(_diffuse_material("Mapped", texture, extra=(mapping, uv)), ("/World/Geom/M",))])

    body = result.value.layer_body
    assert "ND_geompropvalue_vector2" in body and "float2 inputs:in2 = (2, 3)" in body


# --------------------------------------------------------------------------
# Remaining Classroom shader nodes (n-04-05)
# --------------------------------------------------------------------------


def _surface_material(name, surface, *extra):
    output = _Node("OUTPUT_MATERIAL", "Material Output", [_Socket("Surface")])
    _link(output.inputs["Surface"], surface)
    return _Material(name, [surface, output, *extra], source_usd_path="/World/Looks/" + name)


def _glossy(color=(0.8, 0.8, 0.8, 1.0), roughness=0.4):
    return _Node("BSDF_GLOSSY", "Glossy BSDF", [
        _Socket("Color", color, socket_type="RGBA"), _Socket("Roughness", roughness),
        _Socket("Anisotropy", 0.0), _Socket("Normal", (0.0, 0.0, 0.0))])


def _diffuse(color=(0.1, 0.3, 0.8, 1.0)):
    return _Node("BSDF_DIFFUSE", "Diffuse BSDF", [
        _Socket("Color", color, socket_type="RGBA"), _Socket("Roughness", 0.0), _Socket("Normal", (0.0, 0.0, 0.0))])


def _mix_shader(first, second, factor=0.5):
    node = _Node("MIX_SHADER", "Mix Shader", [
        _Socket("Factor", factor), _Socket("Shader", None, socket_type="SHADER"),
        _Socket("Shader", None, socket_type="SHADER")])
    values = node.inputs.values()
    for socket, source in zip(values[1:], (first, second)):
        if source is not None:
            _link(socket, source)
    return node


def _converted(material, target="/World/Geom/X"):
    result = _convert_bindings([_Binding(material, (target,))])
    assert result.status.value == "ok", result.diagnostics
    return result.diagnostics["materials"][0]


def test_glossy_bsdf_is_a_rough_metal() -> None:
    record = _converted(_surface_material("Metal", _glossy((0.9, 0.6, 0.2, 1.0), 0.25)))

    values = record["openpbr_values"]
    assert values["base_metalness"] == 1.0 and values["specular_roughness"] == 0.25
    assert values["base_color"] == pytest.approx((0.9, 0.6, 0.2))


def test_glass_bsdf_becomes_transmission_with_its_ior() -> None:
    glass = _Node("BSDF_GLASS", "Glass BSDF", [
        _Socket("Color", (0.8, 1.0, 0.9, 1.0), socket_type="RGBA"), _Socket("Roughness", 0.05),
        _Socket("IOR", 1.45), _Socket("Normal", (0.0, 0.0, 0.0))])

    record = _converted(_surface_material("Glass", glass))

    values = record["openpbr_values"]
    assert values["transmission_weight"] == 1.0 and values["specular_ior"] == pytest.approx(1.45)
    assert values["transmission_color"] == pytest.approx((0.8, 1.0, 0.9))
    assert values["specular_roughness"] == pytest.approx(0.05)


def test_translucent_bsdf_is_reported_as_a_diffuse_approximation() -> None:
    translucent = _Node("BSDF_TRANSLUCENT", "Translucent BSDF", [
        _Socket("Color", (0.9, 0.8, 0.5, 1.0), socket_type="RGBA"), _Socket("Normal", (0.0, 0.0, 0.0))])

    record = _converted(_surface_material("Paper", translucent))

    assert "BSDF_TRANSLUCENT:modelled_as_diffuse" in record["approximations"]
    assert record["openpbr_values"]["specular_weight"] == 0.0


@pytest.mark.parametrize("glossy_first", [False, True])
def test_diffuse_glossy_mix_matches_reflectance_at_normal_incidence(glossy_first) -> None:
    glossy, diffuse = _glossy((0.8, 0.8, 0.8, 1.0), 0.3), _diffuse()
    # Mix Shader weights its first shader by (1 - factor).
    mix = _mix_shader(glossy, diffuse, 0.94) if glossy_first else _mix_shader(diffuse, glossy, 0.06)

    record = _converted(_surface_material("Plastic", mix))

    values = record["openpbr_values"]
    f0 = ((values["specular_ior"] - 1.0) / (values["specular_ior"] + 1.0)) ** 2
    assert f0 == pytest.approx(0.06, rel=1.0e-6)
    assert values["specular_color"] == pytest.approx((0.8, 0.8, 0.8))
    assert values["specular_roughness"] == pytest.approx(0.3) and "specular_weight" not in values
    assert values["base_color"] == pytest.approx((0.1, 0.3, 0.8))
    assert any("glossy_specular_layer" in note for note in record["approximations"])


def test_strong_glossy_mix_becomes_partial_metalness() -> None:
    mix = _mix_shader(_diffuse((0.2, 0.2, 0.2, 1.0)), _glossy((1.0, 1.0, 1.0, 1.0)), 0.6)

    record = _converted(_surface_material("Chrome", mix))

    values = record["openpbr_values"]
    assert values["base_metalness"] == pytest.approx(0.6)
    assert any("partial_metalness" in note for note in record["approximations"])


def test_glossy_mix_with_non_constant_factor_is_rejected(tmp_path) -> None:
    factor = _image(tmp_path, "Factor")
    mix = _mix_shader(_diffuse(), _glossy(), 0.5)
    _link(mix.inputs["Factor"], factor, from_socket_name="Color")
    material = _surface_material("Linked", mix, factor)

    # the factor is an image: the mix cannot be reduced to one specular layer
    result = _convert_bindings([_Binding(material, ("/World/Geom/X",))])

    assert result.status.value == "error"


def test_glossy_roughness_from_constant_math_is_folded() -> None:
    value = _Node("VALUE", "Value", [], outputs=[_Socket("Value", 0.04)])
    sqrt = _math("POWER", value, second_value=0.5)
    sqrt.inputs.values()[0].links[0].from_socket = SimpleNamespace(name="Value")
    glossy = _glossy()
    _link(glossy.inputs["Roughness"], sqrt, from_socket_name="Value")

    record = _converted(_surface_material("Folded", glossy, value, sqrt))

    assert record["openpbr_values"]["specular_roughness"] == pytest.approx(0.2)


def test_glossy_texture_colour_is_a_network_texture(tmp_path) -> None:
    texture = _image(tmp_path)
    tinted = _hue_sat(texture, hue=0.6)
    glossy = _glossy()
    _link(glossy.inputs["Color"], tinted, from_socket_name="Color")

    record = _converted(_surface_material("Tinted", glossy, texture, tinted))

    assert record["openpbr_values"]["textures"]["base_color"]["network"]["nodes"]


def test_untranslatable_bump_height_is_dropped_and_reported() -> None:
    noise = _noise()
    bump = _Node("BUMP", "Bump", [_Socket("Strength", 0.1), _Socket("Height", 0.0), _Socket("Normal", (0.0, 0.0, 0.0))])
    _link(bump.inputs["Height"], noise, from_socket_name="Fac")
    diffuse = _diffuse()
    _link(diffuse.inputs["Normal"], bump, from_socket_name="Normal")

    record = _converted(_surface_material("Rough", diffuse, noise, bump))

    assert "geometry_normal" not in record["openpbr_values"]["textures"]
    assert "geometry_normal:BUMP(TEX_NOISE)" in record["approximations"]


def test_linked_displacement_is_reported() -> None:
    material = _surface_material("Displaced", _diffuse())
    output = material.node_tree.nodes[1]
    output.inputs["Displacement"] = _link(_Socket("Displacement", (0.0, 0.0, 0.0)), _Node("DISPLACEMENT", "Displacement", []))

    assert "material_output:DISPLACEMENT_not_translated" in _converted(material)["approximations"]


def test_network_texture_reports_noise_approximations_and_error_bounds(tmp_path) -> None:
    mapping = _mapping("TEXTURE", scale=(2.0, 2.0, 1.0))
    _tex_coord_into(mapping.inputs["Vector"])
    noise = _noise(mapping)
    curve = _curve_node(noise, [lambda x: x] * 3 + [lambda x: x * x])
    _link(curve.inputs["Color"], noise, from_socket_name="Color")
    material = _diffuse_material("Dirt", curve, extra=(noise, mapping))

    result = _convert_bindings([_Binding(material, ("/World/Geom/Dirt",))])

    record = result.diagnostics["materials"][0]
    assert record["status"] == "generated"
    assert "base_color:TEX_NOISE" in record["approximations"]
    assert record["error_bounds"][0].startswith("base_color:CURVE_RGB:max_abs_error<=")
    assert 'info:id = "ND_fractal3d_float"' in result.value.layer_body


def test_network_shader_names_are_unique_and_connected(tmp_path) -> None:
    texture = _image(tmp_path)
    material = _diffuse_material("Chain", _hue_sat(texture, hue=0.7, saturation=1.2), extra=(texture,))

    body = _convert_bindings([_Binding(material, ("/World/Geom/Chain",))]).value.layer_body

    names = [line.split('"')[1] for line in body.splitlines() if line.strip().startswith("def Shader ")]
    assert len(names) == len(set(names))
    for line in body.splitlines():
        if ".connect = <" in line:
            target = line.split("<", 1)[1].split(".outputs", 1)[0].rsplit("/", 1)[1]
            assert target in names


# --------------------------------------------------------------------------
# Node groups
# --------------------------------------------------------------------------


class _GroupSocket(_Socket):
    def __init__(self, name, default=None, socket_type="VALUE", identifier=None):
        super().__init__(name, default, socket_type=socket_type)
        self.identifier = identifier or name

    @property
    def is_linked(self):
        return getattr(self, "_linked", False)


def _node_with_sockets(node_type, name, inputs, outputs=()):
    node = _Node(node_type, name, list(inputs), outputs=list(outputs))
    return node


def _flat_link(from_node, from_socket, to_socket):
    link = SimpleNamespace(from_node=from_node, from_socket=from_socket, to_socket=to_socket)
    to_socket._linked = True
    return link


def _plastic_group_material(outer_color=(0.2, 0.4, 0.6, 1.0), outer_roughness=0.25):
    """Material Output <- Group(Color, Roughness) whose inner tree is Mix(Glossy, Diffuse)."""

    group_color = _GroupSocket("Color", outer_color, "RGBA")
    group_roughness = _GroupSocket("Roughness", outer_roughness, "VALUE")
    group_out = _GroupSocket("Shader", None, "SHADER")
    group = _node_with_sockets("GROUP", "Group", [group_color, group_roughness], [group_out])

    in_color = _GroupSocket("Color", None, "RGBA")
    in_roughness = _GroupSocket("Roughness", None, "VALUE")
    group_input = _node_with_sockets("GROUP_INPUT", "Group Input", [], [in_color, in_roughness])
    diffuse = _diffuse()
    glossy = _glossy()
    mix_inputs = [_GroupSocket("Factor", 0.94), _GroupSocket("Shader", None, "SHADER"), _GroupSocket("Shader", None, "SHADER")]
    mix = _node_with_sockets("MIX_SHADER", "Mix Shader", mix_inputs, [_GroupSocket("Shader", None, "SHADER")])
    sqrt = _node_with_sockets("MATH", "Math", [_GroupSocket("Value", 0.5), _GroupSocket("Value", 0.5)], [_GroupSocket("Value", 0.0)])
    sqrt.operation = "POWER"
    group_output = _node_with_sockets("GROUP_OUTPUT", "Group Output", [_GroupSocket("Shader", None, "SHADER")])
    group_output.is_active_output = True
    # make the Blender-style sockets linkable
    for node in (diffuse, glossy):
        node.inputs = type(node.inputs)([_GroupSocket(s.name, s.default_value, s.type) for s in node.inputs])
    links = [
        _flat_link(group_input, in_color, diffuse.inputs["Color"]),
        _flat_link(group_input, in_roughness, sqrt.inputs.values()[0]),
        _flat_link(sqrt, sqrt.outputs.values()[0], glossy.inputs["Roughness"]),
        _flat_link(glossy, SimpleNamespace(name="BSDF"), mix_inputs[1]),
        _flat_link(diffuse, SimpleNamespace(name="BSDF"), mix_inputs[2]),
        _flat_link(mix, mix.outputs.values()[0], group_output.inputs.values()[0]),
    ]
    group.node_tree = SimpleNamespace(
        nodes=[group_input, diffuse, glossy, mix, sqrt, group_output], links=links)
    out_surface = _GroupSocket("Surface", None, "SHADER")
    output = _node_with_sockets("OUTPUT_MATERIAL", "Material Output", [out_surface])
    outer_links = [_flat_link(group, group_out, out_surface)]
    material = _Material("Plastique", [group, output])
    material.node_tree = SimpleNamespace(nodes=[group, output], links=outer_links)
    return material, group


def test_node_group_is_flattened_with_constant_group_inputs() -> None:
    material, _ = _plastic_group_material()

    record = _converted(material)

    values = record["openpbr_values"]
    assert values["base_color"] == pytest.approx((0.2, 0.4, 0.6))
    assert values["specular_roughness"] == pytest.approx(0.5)  # sqrt(0.25)
    assert values["specular_ior"] != 1.5


def test_node_group_inputs_follow_outer_links(tmp_path) -> None:
    material, group = _plastic_group_material()
    rgb = _Node("RGB", "RGB", [], outputs=[_Socket("Color", (0.9, 0.1, 0.1, 1.0))])
    outer_color = group.inputs["Color"]
    material.node_tree.links.append(_flat_link(rgb, rgb.outputs["Color"], outer_color))
    material.node_tree.nodes.append(rgb)

    record = _converted(material)

    assert record["openpbr_values"]["base_color"] == pytest.approx((0.9, 0.1, 0.1))


def test_group_tree_instanced_twice_keeps_the_group_rejected() -> None:
    material, group = _plastic_group_material()
    second = _node_with_sockets("GROUP", "Group.001", [], [_GroupSocket("Shader", None, "SHADER")])
    second.node_tree = group.node_tree
    material.node_tree.nodes.append(second)

    nodes, links = conversion._flattened_node_tree(material.node_tree)

    assert any(node.type == "GROUP" for node in nodes)
    assert links is not None


def _bump_over(height_source, strength=0.5):
    bump = _Node("BUMP", "Bump", [_Socket("Strength", strength), _Socket("Height", 0.0), _Socket("Normal", (0.0, 0.0, 0.0))])
    _link(bump.inputs["Height"], height_source, from_socket_name="Color")
    return bump


def test_bump_height_through_a_colour_ramp_keeps_the_image_and_reports_the_ramp(tmp_path) -> None:
    """heighttonormal only differentiates an image node in the installed renderer, so a bump height
    that is a network of an image falls back to the image itself and the omitted ramp is reported."""
    texture = _image(tmp_path)
    ramp = _ramp(texture, [(0.3, (0.0, 0.0, 0.0)), (0.45, (1.0, 1.0, 1.0))])
    bump = _bump_over(ramp)
    diffuse = _diffuse()
    _link(diffuse.inputs["Normal"], bump, from_socket_name="Normal")
    material = _surface_material("Bumpy", diffuse, texture, ramp, bump)

    result = _convert_bindings([_Binding(material, ("/World/Geom/Bumpy",))])

    body = result.value.layer_body
    record = result.diagnostics["materials"][0]
    assert record["status"] == "generated"
    assert "geometry_normal:VALTORGB" in record["approximations"]
    assert "network" not in record["openpbr_values"]["textures"]["geometry_normal"]
    assert "float inputs:in.connect = </OVRTX_Materials/Bumpy/ND_image_float_geometry_normal.outputs:out>" in body
    assert "float inputs:scale = 0.5" in body


def test_bump_with_only_a_direct_image_keeps_the_compact_record(tmp_path) -> None:
    texture = _image(tmp_path)
    diffuse = _diffuse()
    _link(diffuse.inputs["Normal"], _bump_over(texture), from_socket_name="Normal")
    material = _surface_material("Plain", diffuse, texture, diffuse.inputs["Normal"].links[0].from_node)

    record = _converted(material)

    assert "network" not in record["openpbr_values"]["textures"]["geometry_normal"]


def _apply_uv_steps(steps, uv):
    u, v = uv
    for kind, value in steps:
        if kind == "scale":
            u, v = u * value[0], v * value[1]
        elif kind == "offset":
            u, v = u + value[0], v + value[1]
        else:
            c, s = math.cos(math.radians(value)), math.sin(math.radians(value))
            u, v = c * u - s * v, s * u + c * v
    return u, v


@pytest.mark.parametrize("vector_type", ["POINT", "TEXTURE", "VECTOR"])
def test_z_rotated_mapping_is_ordered_steps_matching_blenders_mapping(vector_type) -> None:
    location, scale, angle = (0.3, -0.2, 0.0), (2.0, 4.0, 1.0), 0.6
    mapping = _mapping(vector_type, location=location, rotation=(0.0, 0.0, angle), scale=scale)
    _tex_coord_into(mapping.inputs["Vector"])
    link = SimpleNamespace(from_node=mapping, from_socket=SimpleNamespace(name="Vector"))

    chain = conversion._uv_chain(link)

    assert chain is not None and chain[3] is not None
    assert conversion._uv_coordinate_transform(link) is None
    u, v = 0.35, 0.8
    c, s = math.cos(angle), math.sin(angle)
    if vector_type == "TEXTURE":  # inverse of point: R^-1 (p - L) / S
        x, y = u - location[0], v - location[1]
        expected = ((c * x + s * y) / scale[0], (-s * x + c * y) / scale[1])
    else:
        x, y = u * scale[0], v * scale[1]
        shift = location[:2] if vector_type == "POINT" else (0.0, 0.0)
        expected = (c * x - s * y + shift[0], s * x + c * y + shift[1])
    assert _apply_uv_steps(chain[3], (u, v)) == pytest.approx(expected)


def test_rotated_mapping_authors_rotate2d_between_scale_and_offset(tmp_path) -> None:
    texture = _image(tmp_path)
    mapping = _mapping("POINT", location=(0.1, 0.2, 0.0), rotation=(0.0, 0.0, math.radians(2.0)), scale=(0.9, 0.9, 1.0))
    _link(texture.inputs["Vector"], mapping, from_socket_name="Vector")
    material = _diffuse_material("Tilted", texture, extra=(mapping, _tex_coord_into(mapping.inputs["Vector"])))

    body = _convert_bindings([_Binding(material, ("/World/Geom/Tilted",))]).value.layer_body

    assert 'info:id = "ND_rotate2d_vector2"' in body
    assert "float inputs:amount = 2" in body
    order = [line for line in body.splitlines() if 'info:id = "ND_texcoord_vector2"' in line
             or 'info:id = "ND_multiply_vector2"' in line or 'info:id = "ND_rotate2d_vector2"' in line
             or 'info:id = "ND_add_vector2"' in line]
    assert [line.split('"')[1] for line in order] == [
        "ND_texcoord_vector2", "ND_multiply_vector2", "ND_rotate2d_vector2", "ND_add_vector2"]
    assert "inputs:texcoord.connect = </OVRTX_Materials/Tilted/ND_image_color3_base_color_uv_2_offset.outputs:out>" in body


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("vector_type", ["POINT", "TEXTURE"])
def test_single_axis_tilt_shortens_one_uv_axis_like_blenders_euler_rotation(vector_type, axis) -> None:
    angle = 0.0349
    rotation = [0.0, 0.0, 0.0]
    rotation[axis] = angle
    mapping = _mapping(vector_type, location=(0.7, 0.0, 0.0), rotation=tuple(rotation), scale=(0.85, 0.85, 1.0))
    _tex_coord_into(mapping.inputs["Vector"])
    link = SimpleNamespace(from_node=mapping, from_socket=SimpleNamespace(name="Vector"))

    chain = conversion._uv_chain(link)

    assert chain is not None and chain[3] is None  # no Z rotation: still a plain affine transform
    u, v = 0.4, 0.9
    # Blender: rotate about X (or Y) the vector (u, v, 0) scaled; only x and y are read afterwards.
    c = math.cos(angle)
    if vector_type == "POINT":
        sx, sy = u * 0.85, v * 0.85
        expected = (sx * (c if axis == 1 else 1.0) + 0.7, sy * (c if axis == 0 else 1.0))
    else:
        px, py = u - 0.7, v
        expected = ((px * (c if axis == 1 else 1.0)) / 0.85, (py * (c if axis == 0 else 1.0)) / 0.85)
    (scale, offset) = chain[0], chain[1]
    assert (u * scale[0] + offset[0], v * scale[1] + offset[1]) == pytest.approx(expected)


def test_tilt_after_another_mapping_is_rejected() -> None:
    outer = _mapping("POINT", rotation=(0.3, 0.0, 0.0))
    inner = _mapping("POINT", location=(0.0, 0.0, 0.5))
    _link(outer.inputs["Vector"], inner, from_socket_name="Vector")
    _tex_coord_into(inner.inputs["Vector"])
    link = SimpleNamespace(from_node=outer, from_socket=SimpleNamespace(name="Vector"))

    assert conversion._uv_chain(link) is None


def test_mix_shader_reports_the_connected_branch_it_does_not_convert() -> None:
    translucent = _Node("BSDF_TRANSLUCENT", "Translucent BSDF", [
        _Socket("Color", (0.9, 0.8, 0.5, 1.0), socket_type="RGBA"), _Socket("Normal", (0.0, 0.0, 0.0))])
    mix = _mix_shader(_diffuse(), translucent, 0.3)

    record = _converted(_surface_material("Paper", mix))

    assert "MIX_SHADER:omitted_BSDF_TRANSLUCENT" in record["approximations"]


def _euler_xyz(rotation, vector):
    """Blender's XYZ Euler: rotate about X, then Y, then Z (all counter-clockwise)."""
    for axis, angle in enumerate(rotation):
        unit = tuple(1.0 if i == axis else 0.0 for i in range(3))
        vector = _rotate3d(vector, math.degrees(angle), unit)
    return vector


@pytest.mark.parametrize("vector_type", ["POINT", "TEXTURE", "VECTOR"])
def test_rotated_mapping_feeding_a_texture_matches_blenders_three_axis_mapping(vector_type) -> None:
    location, scale, rotation = (0.2, -0.1, 0.23), (1.7, 1.3, 2.0), (0.3, -0.4, 0.7854)
    mapping = _mapping(vector_type, location=location, rotation=rotation, scale=scale)
    _tex_coord_into(mapping.inputs["Vector"])
    noise = _noise(mapping, scale=1.0)
    net, _ = _translate(_linked_socket(noise, output="Fac"), "float")
    vector_net = network.Network("v")
    value = network._vector_value(vector_net, noise.inputs["Vector"])

    result = _evaluate(vector_net, value)

    uv = (0.3, 0.6, 0.0)
    if vector_type == "TEXTURE":
        inverse = (uv[0] - location[0], uv[1] - location[1], uv[2] - location[2])
        for axis in (2, 1, 0):
            unit = tuple(1.0 if i == axis else 0.0 for i in range(3))
            inverse = _rotate3d(inverse, -math.degrees(rotation[axis]), unit)
        expected = tuple(x / s for x, s in zip(inverse, scale))
    else:
        rotated = _euler_xyz(rotation, tuple(x * s for x, s in zip(uv, scale)))
        expected = tuple(x + (l if vector_type == "POINT" else 0.0) for x, l in zip(rotated, location))
    assert result == pytest.approx(expected, abs=1.0e-6)


def _kitchen_sink_material(tmp_path):
    """Hue/Saturation + RGB Curves + Color Ramp + Mix + Noise + Attribute feeding one Diffuse and a Bump."""
    mapping = _mapping("TEXTURE", scale=(2.0, 2.0, 1.0), rotation=(0.0, 0.0, 0.4))
    _tex_coord_into(mapping.inputs["Vector"])
    noise = _noise(mapping, detail=3.0)
    texture = _image(tmp_path)
    tinted = _hue_sat(texture, hue=0.6, saturation=1.4)
    curve = _curve_node(tinted, [lambda x: x, lambda x: x, lambda x: x, lambda x: x * x])
    ramp = _ramp(noise, [(0.2, (0.9, 0.1, 0.1)), (0.7, (0.1, 0.2, 0.9))])
    ramp.inputs["Fac"].links[0].from_socket = SimpleNamespace(name="Fac")
    attribute = _Node("ATTRIBUTE", "Attribute", [], outputs=[_Socket("Color", (0, 0, 0, 1))])
    attribute.attribute_name, attribute.attribute_type = "Col", "GEOMETRY"
    overlay = _mix(a=curve, b=attribute, blend="OVERLAY", fac=0.5)
    final = _mix(a=overlay, b=ramp, blend="MULTIPLY", fac=0.7)
    bump = _bump_over(_ramp(texture, [(0.3, (0.0, 0.0, 0.0)), (0.45, (1.0, 1.0, 1.0))]))
    diffuse = _diffuse()
    _link(diffuse.inputs["Color"], final, from_socket_name="Result")
    _link(diffuse.inputs["Normal"], bump, from_socket_name="Normal")
    return _surface_material("Sink", diffuse, mapping, noise, texture, tinted, curve, ramp, attribute, overlay, final, bump)


def test_network_material_authors_a_valid_stage_with_every_connection_resolved(tmp_path) -> None:
    pytest.importorskip("pxr")
    from pxr import Sdf, Usd

    material = _kitchen_sink_material(tmp_path)

    body = _convert_bindings([_Binding(material, ("/World/Geom/Sink",))]).value.layer_body

    layer = Sdf.Layer.CreateAnonymous("sink.usda")
    assert layer.ImportFromString("#usda 1.0" + chr(10) + body)
    stage = Usd.Stage.Open(layer)
    shaders = {str(prim.GetPath()) for prim in stage.Traverse() if prim.GetTypeName() == "Shader"}
    assert len(shaders) > 20
    for prim in stage.Traverse():
        for attribute in prim.GetAttributes():
            for source in attribute.GetConnections():
                assert str(source.GetPrimPath()) in shaders, (prim.GetPath(), source)


def test_network_material_compiles_as_an_inline_graph_layer(tmp_path) -> None:
    pytest.importorskip("pxr")
    material = _kitchen_sink_material(tmp_path)

    text, errors = graph_updates.material_layer(material, "/World/_materials/Sink")

    assert not errors and text
    assert "/OVRTX_Materials/" not in text
    assert "ND_fractal3d_float" in text and "ND_heighttonormal_vector3" in text
    assert graph_updates.validate_material_layer("/World/_materials/Sink", text)


_USD_TO_MATERIALX = {
    "float": "float", "color3f": "color3", "color4f": "color4", "float2": "vector2", "float3": "vector3",
    "int": "integer", "bool": "boolean", "string": "string", "asset": "filename",
}


def _material_documents_with_every_shape(tmp_path):
    from_sink = _kitchen_sink_material(tmp_path)
    texture = _image(tmp_path, "Edge")
    texture.extension = "MIRROR"
    mapping = _mapping("POINT", location=(0.1, 0.2, 0.0), rotation=(0.0, 0.0, 0.3), scale=(2.0, 2.0, 1.0))
    _link(texture.inputs["Vector"], mapping, from_socket_name="Vector")
    rotated = _diffuse_material("Rot", texture, extra=(mapping, _tex_coord_into(mapping.inputs["Vector"])))
    glass = _Node("BSDF_GLASS", "Glass BSDF", [
        _Socket("Color", (0.8, 1.0, 0.9, 1.0), socket_type="RGBA"), _Socket("Roughness", 0.05),
        _Socket("IOR", 1.45), _Socket("Normal", (0.0, 0.0, 0.0))])
    plastic = _mix_shader(_diffuse(), _glossy(), 0.06)
    return [from_sink, rotated, _surface_material("Glass", glass), _surface_material("Plastic", plastic)]


def test_every_authored_node_matches_the_materialx_standard_library_signatures(tmp_path) -> None:
    """Static check of the emitted shapes: node ids, input names/types, connection types."""
    mx = pytest.importorskip("MaterialX")
    pytest.importorskip("pxr")
    from pxr import Sdf, Usd

    library = mx.createDocument()
    folder = mx.FilePath(str(Path(mx.__file__).parent))
    mx.loadLibraries([mx.FilePath("libraries")], mx.FileSearchPath(folder.asString()), library)
    checked = 0
    for index, material in enumerate(_material_documents_with_every_shape(tmp_path)):
        body = _convert_bindings([_Binding(material, (f"/World/Geom/M{index}",))]).value.layer_body
        layer = Sdf.Layer.CreateAnonymous("check.usda")
        assert layer.ImportFromString("#usda 1.0" + chr(10) + body)
        stage = Usd.Stage.Open(layer)
        output_types = {}
        for prim in stage.Traverse():
            if prim.GetTypeName() == "Shader":
                info = prim.GetAttribute("info:id").Get()
                nodedef = library.getNodeDef(info)
                assert nodedef is not None, f"unknown MaterialX node {info}"
                out = prim.GetAttribute("outputs:out")
                if out:
                    output_types[str(prim.GetPath())] = str(out.GetTypeName())
        for prim in stage.Traverse():
            if prim.GetTypeName() != "Shader":
                continue
            nodedef = library.getNodeDef(prim.GetAttribute("info:id").Get())
            for attribute in prim.GetAttributes():
                name = attribute.GetName()
                if not name.startswith("inputs:"):
                    continue
                port = nodedef.getActiveInput(name[len("inputs:"):])
                assert port is not None, f"{nodedef.getName()} has no input {name}"
                expected = port.getType()
                got = _USD_TO_MATERIALX.get(str(attribute.GetTypeName()), str(attribute.GetTypeName()))
                assert got == expected, f"{prim.GetPath()}.{name}: authored {got}, nodedef wants {expected}"
                for source in attribute.GetConnections():
                    source_type = _USD_TO_MATERIALX.get(output_types.get(str(source.GetPrimPath()), ""), "")
                    assert source_type == expected, f"{prim.GetPath()}.{name} <- {source}: {source_type} != {expected}"
                checked += 1
            out = prim.GetAttribute("outputs:out")
            expected_out = nodedef.getActiveOutputs()[0].getType() if nodedef.getActiveOutputs() else ""
            if out and expected_out != "surfaceshader":
                assert _USD_TO_MATERIALX.get(str(out.GetTypeName())) == expected_out, (prim.GetPath(), expected_out)
    assert checked > 150


# --------------------------------------------------------------------------
# Parameter-name collisions in the installed renderer
# --------------------------------------------------------------------------


def _gamma_node(source, gamma=2.2):
    node = _Node("GAMMA", "Gamma", [_Socket("Color", (1.0, 1.0, 1.0, 1.0), socket_type="RGBA"), _Socket("Gamma", gamma)])
    _link(node.inputs["Color"], source, from_socket_name="Color")
    return node


def _order_material(tmp_path, name, order, *, hue=0.7, gamma=2.2):
    texture = _image(tmp_path, name + " image")
    if order == "hg":
        inner = _hue_sat(texture, hue=hue, saturation=1.8)
        outer = _gamma_node(inner, gamma)
    else:
        inner = _gamma_node(texture, gamma)
        outer = _hue_sat(inner, hue=hue, saturation=1.8)
    return _diffuse_material(name, outer, extra=(texture, inner))


def _layout_of(material):
    values = conversion._classify_material(material, ("/World/Geom/X",), (), {})["openpbr_values"]
    return conversion._parameter_layout(conversion._material_block_lines("M", values))


def test_same_parameter_names_with_different_wiring_is_what_the_renderer_cannot_compile(tmp_path) -> None:
    first = _layout_of(_order_material(tmp_path, "A", "hg"))
    second = _layout_of(_order_material(tmp_path, "B", "gh"))
    repeated = _layout_of(_order_material(tmp_path, "C", "hg", hue=0.3, gamma=0.6))

    assert first[0] == second[0] and first[1] != second[1]  # the collision the renderer rejects
    assert first == repeated  # only constants differ: one wiring, shared safely


def test_colliding_wiring_gets_neutral_pad_nodes_so_both_materials_stay_converted(tmp_path) -> None:
    first = _order_material(tmp_path, "A", "hg")
    second = _order_material(tmp_path, "B", "gh")

    result = _convert_bindings([_Binding(first, ("/World/Geom/A",)), _Binding(second, ("/World/Geom/B",))])

    assert [record["status"] for record in result.diagnostics["materials"]] == ["generated", "generated"]
    body = result.value.layer_body
    assert body.count("_pad_01") == 2  # one pad shader: its definition and the surface connection
    pad_blocks = [block for block in body.split("def Shader ")[1:] if "_pad_01" in block.split(chr(10))[0]]
    assert len(pad_blocks) == 1
    assert 'uniform token info:id = "ND_add_color3FA"' in pad_blocks[0] and "float inputs:in2 = 0" in pad_blocks[0]
    layouts = []
    for record in result.diagnostics["materials"]:
        values = record["openpbr_values"]
        layouts.append(conversion._parameter_layout(conversion._material_block_lines("M", values))[0])
    assert layouts[0] != layouts[1]


def test_identical_wiring_needs_no_padding(tmp_path) -> None:
    result = _convert_bindings([
        _Binding(_order_material(tmp_path, "A", "hg"), ("/World/Geom/A",)),
        _Binding(_order_material(tmp_path, "B", "hg", hue=0.2, gamma=0.5), ("/World/Geom/B",)),
    ])

    assert "_pad_" not in result.value.layer_body


def test_colliding_material_that_cannot_be_padded_keeps_stock_shading() -> None:
    values = {"base_color": (0.5, 0.5, 0.5), "specular_roughness": 0.5, "base_metalness": 0.0,
              "geometry_opacity": 1.0, "textures": {}}
    names, wiring = conversion._parameter_layout(conversion._material_block_lines("M", values))

    assert conversion._deconflicted_values("M", values, {names: ("another wiring",)}) is None
    assert conversion._deconflicted_values("M", values, {names: wiring}) is values


def test_hue_saturation_that_boosts_saturation_clamps_it_like_blender(tmp_path) -> None:
    """hsvadjust would drive a channel negative above saturation 1; Blender clamps saturation to 1."""
    node = _hue_sat(_image(tmp_path), hue=0.7, saturation=1.8, value=0.9)
    color = (1.0, 0.0817, 0.0006)

    net, value = _translate(_linked_socket(node))

    assert not any(entry["info_id"] == "ND_hsvadjust_color3" for entry in net.nodes)
    h, s, v = colorsys.rgb_to_hsv(*color)
    expected = colorsys.hsv_to_rgb((h + 0.2) % 1.0, min(1.0, s * 1.8), v * 0.9)
    result = _evaluate(net, value, image=color)
    assert _approx(result, expected, 1.0e-6) and min(result) >= 0.0


def test_gamma_and_power_floor_negative_bases_at_zero(tmp_path) -> None:
    """Blender's safe power gives 0 for a negative base; a raw negative pow would be NaN (black)."""
    texture = _image(tmp_path)
    gamma = _gamma_node(texture, 2.2)
    net, value = _translate(_linked_socket(gamma))
    assert _evaluate(net, value, image=(-0.5, 0.5, 0.25)) == pytest.approx((0.0, 0.5 ** 2.2, 0.25 ** 2.2))

    power = _math("POWER", texture, second_value=0.5)
    net, value = _translate(_linked_socket(power, output="Value"), "float")
    assert _evaluate(net, value, image=(-0.5, -0.5, -0.5)) == pytest.approx(0.0)

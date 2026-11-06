# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Translate Blender colour/scalar node networks into MaterialX standard nodes.

The conversion module feeds one OpenPBR input from a Blender socket. When the
socket's upstream graph is more than a single image (RGB Curves, Hue/Saturation,
Mix, Color Ramp, Math, procedural noise, ...) this module walks it and builds a
list of MaterialX shader node descriptions that the conversion module authors
as USDA. Everything is translated exactly except where ``Network.approximations``
(approximate or fixed operations) or ``Network.error_bounds`` (numerically
bounded curve fits) say otherwise; an untranslatable node raises
``Untranslatable`` so the caller can keep its conservative fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Iterable

_HOST: Any = None

# Rec.709 luminance, the weights Blender's colour to float socket conversion uses.
_LUMA = (0.2126, 0.7152, 0.0722)
# Largest absolute error a piecewise-linear curve or ramp fit may have.
CURVE_ERROR_BOUND = 0.004
_MAX_CURVE_POINTS = 48
_EPS = 1.0e-6
_USDA_TYPES = {
    "float": "float",
    "color3": "color3f",
    "vector2": "float2",
    "vector3": "float3",
    "integer": "int",
    "string": "string",
    "boolean": "bool",
}
_SUFFIX = {"float": "float", "color3": "color3", "vector2": "vector2", "vector3": "vector3"}


class Untranslatable(Exception):
    """The node graph has a shape this translator does not reproduce."""


def _host() -> Any:
    global _HOST
    if _HOST is None:
        from . import materialx_openpbr_conversion as host

        _HOST = host
    return _HOST


@dataclass(frozen=True)
class Value:
    """A typed network value: a literal (``const``) or a shader node output."""

    kind: str
    const: tuple[float, ...] | None = None
    node: str = ""

    @property
    def is_const(self) -> bool:
        return self.const is not None


def const_float(value: float) -> Value:
    return Value("float", (float(value),))


def const_color(value: Iterable[float]) -> Value:
    channels = tuple(float(channel) for channel in list(value)[:3])
    if len(channels) != 3:
        raise Untranslatable("colour needs three channels")
    return Value("color3", channels)


class Network:
    """Accumulates the MaterialX nodes one OpenPBR input needs."""

    def __init__(self, prefix: str, *, fallback_color_space: str = "sRGB") -> None:
        self.prefix = prefix
        # Colour space of an image whose own colour space is not a recognised sRGB/linear name.
        self.fallback_color_space = fallback_color_space
        self.nodes: list[dict[str, Any]] = []
        self.approximations: list[str] = []
        self.error_bounds: list[str] = []
        self._memo: dict[Any, Value] = {}
        self._used_nodes: set[str] = set()

    def approximate(self, text: str) -> None:
        if text not in self.approximations:
            self.approximations.append(text)

    def bound(self, text: str) -> None:
        if text not in self.error_bounds:
            self.error_bounds.append(text)

    def _name(self, hint: str) -> str:
        return f"{self.prefix}_{len(self.nodes):02d}_{hint}"

    def emit(
        self,
        info_id: str,
        kind: str,
        inputs: dict[str, Any],
        hint: str,
    ) -> Value:
        """Append a standard node. ``inputs`` maps names to ``Value`` or typed literals.

        A typed literal is ``(kind, value)`` with kind one of ``integer``,
        ``string``, ``boolean`` (uniform settings that are never connected).
        """

        name = self._name(hint)
        rows = []
        for input_name, spec in inputs.items():
            if isinstance(spec, Value):
                if spec.is_const:
                    rows.append({"type": _USDA_TYPES[spec.kind], "name": input_name,
                                 "const": spec.const, "connect": ""})
                else:
                    rows.append({"type": _USDA_TYPES[spec.kind], "name": input_name,
                                 "const": None, "connect": spec.node})
            else:
                literal_kind, literal = spec
                rows.append({"type": _USDA_TYPES[literal_kind], "name": input_name,
                             "const": literal, "connect": ""})
        self.nodes.append({"name": name, "info_id": info_id, "out_type": _USDA_TYPES[kind],
                           "inputs": rows})
        return Value(kind, None, name)

    def emit_geomprop_color(self, name: str) -> Value:
        """Read a colour primvar (Blender writes colour attributes as color4f) as colour3."""

        source = self._name("attribute4")
        self.nodes.append({"name": source, "info_id": "ND_geompropvalue_color4", "out_type": "color4f",
                           "inputs": [{"type": "string", "name": "geomprop", "const": name, "connect": ""}]})
        converted = self._name("attribute")
        self.nodes.append({"name": converted, "info_id": "ND_convert_color4_color3", "out_type": "color3f",
                           "inputs": [{"type": "color4f", "name": "in", "const": None, "connect": source}]})
        return Value("color3", None, converted)

    def emit_image(self, image: dict[str, Any]) -> Value:
        name = self._name("image")
        self.nodes.append({"name": name, "info_id": "ND_image_color3", "out_type": "color3f",
                           "image": image, "inputs": []})
        return Value("color3", None, name)


# ---------------------------------------------------------------------------
# Constant helpers and type coercion
# ---------------------------------------------------------------------------

def _luma(color: tuple[float, ...]) -> float:
    return sum(weight * channel for weight, channel in zip(_LUMA, color))


def as_float(net: Network, value: Value) -> Value:
    if value.kind == "float":
        return value
    if value.kind == "color3":
        if value.is_const:
            return const_float(_luma(value.const))
        # ND_luminance_color3 evaluates to a constant in the installed renderer (validated with
        # an image-driven Color Ramp), so take the Rec.709 dot product explicitly.
        vector = net.emit("ND_convert_color3_vector3", "vector3", {"in": value}, "to_vector")
        return net.emit("ND_dotproduct_vector3", "float",
                        {"in1": vector, "in2": Value("vector3", _LUMA)}, "luma")
    raise Untranslatable(f"cannot read {value.kind} as float")


def as_color(net: Network, value: Value) -> Value:
    if value.kind == "color3":
        return value
    if value.kind == "float":
        if value.is_const:
            return const_color((value.const[0],) * 3)
        return net.emit("ND_convert_float_color3", "color3", {"in": value}, "tocolor")
    if value.kind == "vector3":
        if value.is_const:
            return const_color(value.const)
        return net.emit("ND_convert_vector3_color3", "color3", {"in": value}, "tocolor")
    raise Untranslatable(f"cannot read {value.kind} as colour")


def _coerce(net: Network, value: Value, kind: str) -> Value:
    if kind == "float":
        return as_float(net, value)
    if kind == "color3":
        return as_color(net, value)
    raise Untranslatable(f"unsupported target type {kind}")


def _info(base: str, kind: str, variant: str = "") -> str:
    return f"ND_{base}_{_SUFFIX[kind]}{variant}"


def _binary(net: Network, base: str, kind: str, left: Value, right: Value, hint: str) -> Value:
    """Component-wise binary op on two values of one kind (``in1``, ``in2``)."""

    return net.emit(_info(base, kind), kind, {"in1": left, "in2": right}, hint)


def _clamp01(net: Network, value: Value) -> Value:
    if value.is_const:
        return Value(value.kind, tuple(min(1.0, max(0.0, channel)) for channel in value.const))
    zero, one = (const_float(0.0), const_float(1.0)) if value.kind == "float" else (
        const_color((0.0,) * 3), const_color((1.0,) * 3))
    return net.emit(_info("clamp", value.kind), value.kind,
                    {"in": value, "low": zero, "high": one}, "clamp")


def _mix(net: Network, background: Value, foreground: Value, amount: Value, hint: str) -> Value:
    """``background * (1 - amount) + foreground * amount`` with constant folding."""

    kind = foreground.kind
    if amount.is_const and amount.kind == "float":
        weight = amount.const[0]
        if weight <= 0.0:
            return background
        if weight >= 1.0:
            return foreground
        if background.is_const and foreground.is_const:
            return Value(kind, tuple(
                b * (1.0 - weight) + f * weight for b, f in zip(background.const, foreground.const)))
    return net.emit(_info("mix", kind), kind,
                    {"fg": foreground, "bg": background, "mix": amount}, hint)


# ---------------------------------------------------------------------------
# Socket helpers
# ---------------------------------------------------------------------------

def _socket_by_identifier(node: Any, identifiers: tuple[str, ...], names: tuple[str, ...]) -> Any | None:
    host = _host()
    sockets = host._input_sockets(node)
    for identifier in identifiers:
        for socket in sockets:
            if str(getattr(socket, "identifier", "")) == identifier:
                return socket
    for name in names:
        for socket in sockets:
            if str(getattr(socket, "name", "")) == name and getattr(socket, "enabled", True):
                return socket
    return None


def _need(node: Any, name: str) -> Any:
    socket = _host()._input_socket(node, name)
    if socket is None:
        raise Untranslatable(f"{_host()._node_type(node)} lacks input {name}")
    return socket


def _literal(socket: Any, kind: str) -> Value:
    raw = getattr(socket, "default_value", None)
    try:
        if kind == "float":
            if isinstance(raw, (int, float)):
                return const_float(raw)
            channels = list(raw)
            return const_float(channels[0] if len(channels) == 1 else _luma(tuple(channels[:3])))
        channels = tuple(float(channel) for channel in list(raw)[:3])
        if len(channels) != 3:
            raise ValueError
        return Value(kind, channels)
    except (TypeError, ValueError):
        raise Untranslatable("socket has no readable default value") from None


def value_from_socket(net: Network, socket: Any, kind: str) -> Value:
    """The value feeding ``socket``, coerced to ``kind`` (float or color3)."""

    host = _host()
    link = host._socket_link(socket) if host._socket_linked(socket) else None
    if link is None:
        return _literal(socket, kind)
    source = getattr(link, "from_node", None)
    output = str(getattr(getattr(link, "from_socket", None), "name", ""))
    value = _from_output(net, source, output, kind)
    return _coerce(net, value, kind)


def _from_output(net: Network, node: Any, output: str, want: str, depth: int = 0) -> Value:
    if depth > 24 or node is None:
        raise Untranslatable("node graph too deep")
    host = _host()
    key = (host._blender_identity(node), output)
    cached = net._memo.get(key)
    if cached is not None:
        return cached
    node_type = host._node_type(node)
    handler = _HANDLERS.get(node_type)
    if handler is None:
        raise Untranslatable(f"unsupported_node:{node_type}")
    value = handler(net, node, output, want)
    net._memo[key] = value
    return value


# ---------------------------------------------------------------------------
# Image textures
# ---------------------------------------------------------------------------

def _tex_image(net: Network, node: Any, output: str, want: str) -> Value:
    host = _host()
    if output not in {"Color", ""}:
        raise Untranslatable(f"image output {output} is not translated")
    if str(getattr(node, "projection", "FLAT")) != "FLAT":
        raise Untranslatable("image projection is not translated")
    vector = host._input_socket(node, "Vector")
    if host._socket_linked(vector) and host._uv_chain(host._socket_link(vector)) is None:
        raise Untranslatable("image coordinate graph is not translated")
    image = getattr(node, "image", None)
    asset_path = host._image_asset_path(image)
    if not asset_path:
        raise Untranslatable("image has no readable file")
    return net.emit_image({
        "asset_path": asset_path,
        "color_space": host._image_color_space(image, net.fallback_color_space),
        "uv_transform": host._image_uv_transform(node),
        **host._image_address_modes(node),
    })


# ---------------------------------------------------------------------------
# Constants and placeholders
# ---------------------------------------------------------------------------

def _rgb_node(net: Network, node: Any, output: str, want: str) -> Value:
    color = _host()._output_color3(node, "Color")
    if color is None:
        raise Untranslatable("RGB node without a colour")
    return const_color(color)


def _value_node(net: Network, node: Any, output: str, want: str) -> Value:
    value = _host()._output_float(node, "Value")
    if value is None:
        raise Untranslatable("Value node without a value")
    return const_float(value)


_OBJECT_INFO_FIXED = {
    "Random": 0.5,
    "Object Index": 0.0,
    "Material Index": 0.0,
    "Alpha": 1.0,
}


def _object_info(net: Network, node: Any, output: str, want: str) -> Value:
    """Per-object values are not knowable from a shared material.

    A material is shared by every object that uses it, while Random and the
    pass indices differ per object, so the converter freezes them at one value
    and reports it; the result is a single plausible variant, not an average.
    """

    if output in _OBJECT_INFO_FIXED:
        fixed = _OBJECT_INFO_FIXED[output]
        net.approximate(f"OBJECT_INFO.{output}={fixed:g}")
        return const_float(fixed)
    if output == "Color":
        net.approximate("OBJECT_INFO.Color=white")
        return const_color((1.0, 1.0, 1.0))
    raise Untranslatable(f"Object Info output {output} is not translated")


def _attribute(net: Network, node: Any, output: str, want: str) -> Value:
    """Vertex colour layers authored as ``primvars:<name>`` reach the node through ``geompropvalue``."""

    attribute_type = str(getattr(node, "attribute_type", "GEOMETRY"))
    name = str(getattr(node, "attribute_name", "") or "")
    if attribute_type != "GEOMETRY" or not name:
        raise Untranslatable("only named geometry attributes are translated")
    if output in {"Color", ""}:
        return net.emit_geomprop_color(name)
    if output in {"Fac", "Factor"}:
        return net.emit("ND_geompropvalue_float", "float", {"geomprop": ("string", name)}, "attribute")
    raise Untranslatable(f"Attribute output {output} is not translated")


# ---------------------------------------------------------------------------
# Mix
# ---------------------------------------------------------------------------

_SUPPORTED_BLENDS = {"MIX", "ADD", "MULTIPLY", "SUBTRACT", "DIFFERENCE", "DARKEN", "LIGHTEN",
                     "SCREEN", "OVERLAY"}


def _fold_blend(blend: str, base: float, top: float) -> float:
    if blend == "ADD":
        return base + top
    if blend == "MULTIPLY":
        return base * top
    if blend == "SUBTRACT":
        return base - top
    if blend == "DIFFERENCE":
        return abs(base - top)
    if blend == "DARKEN":
        return min(base, top)
    return max(base, top)


def _mix_node(net: Network, node: Any, output: str, want: str) -> Value:
    host = _host()
    node_type = host._node_type(node)
    data_type = str(getattr(node, "data_type", "RGBA"))
    if node_type == "MIX" and data_type not in {"RGBA", "FLOAT"}:
        raise Untranslatable("vector Mix is not translated")
    kind = "float" if data_type == "FLOAT" else "color3"
    blend = str(getattr(node, "blend_type", "MIX"))
    if kind == "float":
        blend = "MIX"
    if blend not in _SUPPORTED_BLENDS:
        raise Untranslatable(f"Mix blend {blend} is not translated")
    if kind == "color3":
        factor = _socket_by_identifier(node, ("Factor_Float",), ("Factor", "Fac"))
        base = _socket_by_identifier(node, ("A_Color",), ("A", "Color1"))
        top = _socket_by_identifier(node, ("B_Color",), ("B", "Color2"))
    else:
        factor = _socket_by_identifier(node, ("Factor_Float",), ("Factor", "Fac"))
        base = _socket_by_identifier(node, ("A_Float",), ("A",))
        top = _socket_by_identifier(node, ("B_Float",), ("B",))
    if factor is None or base is None or top is None:
        raise Untranslatable("Mix node lacks its sockets")
    clamp_factor = bool(getattr(node, "clamp_factor", True)) if node_type == "MIX" else True
    amount = value_from_socket(net, factor, "float")
    if clamp_factor:
        amount = _clamp01(net, amount)
    a = value_from_socket(net, base, kind)
    b = value_from_socket(net, top, kind)
    if blend == "MIX":
        result = _mix(net, a, b, amount, "mix")
    elif blend in {"SCREEN", "OVERLAY"}:
        result = net.emit(f"ND_{blend.lower()}_color3", "color3",
                          {"fg": b, "bg": a, "mix": amount}, blend.lower())
    else:
        if a.is_const and b.is_const:
            blended = Value(kind, tuple(_fold_blend(blend, x, y) for x, y in zip(a.const, b.const)))
        elif blend in {"ADD", "MULTIPLY", "SUBTRACT"}:
            blended = _binary(net, blend.lower(), kind, a, b, blend.lower())
        elif blend == "DIFFERENCE":
            delta = _binary(net, "subtract", kind, a, b, "diff")
            blended = net.emit(_info("absval", kind), kind, {"in": delta}, "absval")
        else:
            op = "min" if blend == "DARKEN" else "max"
            blended = _binary(net, op, kind, a, b, blend.lower())
        result = _mix(net, a, blended, amount, "blend")
    clamp_result = bool(getattr(node, "clamp_result", False) or getattr(node, "use_clamp", False))
    return _clamp01(net, result) if clamp_result else result


# ---------------------------------------------------------------------------
# Colour adjustments
# ---------------------------------------------------------------------------

def _hue_sat(net: Network, node: Any, output: str, want: str) -> Value:
    host = _host()
    color = value_from_socket(net, _need(node, "Color"), "color3")
    hue = value_from_socket(net, _need(node, "Hue"), "float")
    saturation = value_from_socket(net, _need(node, "Saturation"), "float")
    value = value_from_socket(net, _need(node, "Value"), "float")
    fac = value_from_socket(net, _need(node, "Fac" if host._input_socket(node, "Fac") else "Factor"), "float")
    if hue.is_const and hue.const[0] == 0.5 and saturation.is_const and saturation.const[0] == 1.0 \
            and value.is_const and value.const[0] == 1.0:
        return color
    if hue.is_const:
        shift = const_float(hue.const[0] - 0.5)
    else:
        shift = net.emit("ND_subtract_float", "float", {"in1": hue, "in2": const_float(0.5)}, "hue")
    if hue.is_const and saturation.is_const and value.is_const and saturation.const[0] <= 1.0:
        # Saturation cannot exceed 1, so the standard node is exact.
        amount = Value("vector3", (shift.const[0], saturation.const[0], value.const[0]))
        adjusted = net.emit("ND_hsvadjust_color3", "color3", {"in": color, "amount": amount}, "hsvadjust")
    else:
        adjusted = _hsv_adjust_clamped(net, color, shift, saturation, value)
    return _mix(net, color, adjusted, _clamp01(net, fac), "hsv_fac")


def _hsv_adjust_clamped(net: Network, color: Value, shift: Value, saturation: Value, value: Value) -> Value:
    """Blender's Hue/Saturation/Value, which clamps saturation to 1.

    ``ND_hsvadjust_color3`` does not clamp: boosted saturation drives a channel negative
    (``v * (1 - s)``), which the installed renderer shades differently from Cycles. The
    adjustment is written out with the saturation product clamped.
    """

    hsv = net.emit("ND_rgbtohsv_color3", "color3", {"in": color}, "to_hsv")
    channels = [
        net.emit("ND_extract_color3", "float", {"in": hsv, "index": ("integer", index)}, name)
        for index, name in enumerate(("h", "s", "v"))
    ]
    hue = channels[0]
    if not (shift.is_const and abs(shift.const[0]) <= _EPS):
        hue = net.emit("ND_add_float", "float", {"in1": hue, "in2": shift}, "hue_shift")
    scaled = net.emit("ND_multiply_float", "float", {"in1": channels[1], "in2": saturation}, "sat_scaled")
    sat = _clamp01(net, scaled)
    brightness = channels[2]
    if not (value.is_const and abs(value.const[0] - 1.0) <= _EPS):
        brightness = net.emit("ND_multiply_float", "float", {"in1": brightness, "in2": value}, "val_scaled")
    combined = net.emit("ND_combine3_color3", "color3", {"in1": hue, "in2": sat, "in3": brightness}, "hsv")
    return net.emit("ND_hsvtorgb_color3", "color3", {"in": combined}, "from_hsv")


def _bright_contrast(net: Network, node: Any, output: str, want: str) -> Value:
    color = value_from_socket(net, _need(node, "Color"), "color3")
    bright = value_from_socket(net, _need(node, "Bright"), "float")
    contrast = value_from_socket(net, _need(node, "Contrast"), "float")
    if bright.is_const and contrast.is_const:
        scale = const_float(1.0 + contrast.const[0])
        offset = const_float(bright.const[0] - contrast.const[0] * 0.5)
    else:
        scale = net.emit("ND_add_float", "float", {"in1": contrast, "in2": const_float(1.0)}, "bc_scale")
        half = net.emit("ND_multiply_float", "float", {"in1": contrast, "in2": const_float(0.5)}, "bc_half")
        offset = net.emit("ND_subtract_float", "float", {"in1": bright, "in2": half}, "bc_offset")
    scaled = net.emit("ND_multiply_color3FA", "color3", {"in1": color, "in2": scale}, "bc_mul")
    shifted = net.emit("ND_add_color3FA", "color3", {"in1": scaled, "in2": offset}, "bc_add")
    return net.emit("ND_max_color3", "color3",
                    {"in1": shifted, "in2": const_color((0.0, 0.0, 0.0))}, "bc_floor")


def _gamma(net: Network, node: Any, output: str, want: str) -> Value:
    color = value_from_socket(net, _need(node, "Color"), "color3")
    gamma = value_from_socket(net, _need(node, "Gamma"), "float")
    if gamma.is_const and abs(gamma.const[0] - 1.0) <= _EPS:
        return color
    if not color.is_const:
        # Blender's safe power returns 0 for a negative base; a negative in MDL pow is NaN (black).
        color = net.emit("ND_max_color3", "color3", {"in1": color, "in2": const_color((0.0, 0.0, 0.0))}, "gamma_floor")
    return net.emit("ND_power_color3FA", "color3", {"in1": color, "in2": gamma}, "gamma")


def _invert(net: Network, node: Any, output: str, want: str) -> Value:
    color = value_from_socket(net, _need(node, "Color"), "color3")
    fac = value_from_socket(net, _need(node, "Fac" if _host()._input_socket(node, "Fac") else "Factor"), "float")
    if color.is_const:
        inverted = const_color(tuple(1.0 - channel for channel in color.const))
    else:
        inverted = net.emit("ND_invert_color3", "color3", {"in": color}, "invert")
    return _mix(net, color, inverted, _clamp01(net, fac), "invert_fac")


def _separate_color(net: Network, node: Any, output: str, want: str) -> Value:
    host = _host()
    if str(getattr(node, "mode", "RGB")) != "RGB":
        raise Untranslatable("Separate Color mode is not translated")
    channel = host._CHANNEL_INDEX.get(output)
    if channel is None:
        raise Untranslatable(f"Separate Color output {output} is not translated")
    source = _need(node, "Color" if host._input_socket(node, "Color") else "Image")
    color = value_from_socket(net, source, "color3")
    if color.is_const:
        return const_float(color.const[channel])
    return net.emit("ND_extract_color3", "float", {"in": color, "index": ("integer", channel)}, "channel")


# ---------------------------------------------------------------------------
# Piecewise-linear functions (RGB Curves, Color Ramp)
# ---------------------------------------------------------------------------

def _simplify(
    samples: list[tuple[float, tuple[float, ...]]], tolerance: float
) -> list[tuple[float, tuple[float, ...]]]:
    """Douglas-Peucker reduction of sampled points under a max per-channel error."""

    keep = {0, len(samples) - 1}
    stack = [(0, len(samples) - 1)]
    while stack:
        start, end = stack.pop()
        x0, y0 = samples[start]
        x1, y1 = samples[end]
        worst, index = 0.0, -1
        for i in range(start + 1, end):
            x, y = samples[i]
            t = (x - x0) / (x1 - x0) if x1 > x0 else 0.0
            error = max(abs(c - (a + (b - a) * t)) for c, a, b in zip(y, y0, y1))
            if error > worst:
                worst, index = error, i
        if index >= 0 and worst > tolerance:
            keep.add(index)
            stack.extend([(start, index), (index, end)])
    return [samples[i] for i in sorted(keep)]


def _interpolate(points: list[tuple[float, tuple[float, ...]]], x: float) -> tuple[float, ...]:
    if x <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x <= x1:
            t = (x - x0) / (x1 - x0) if x1 > x0 else 1.0
            return tuple(a + (b - a) * t for a, b in zip(y0, y1))
    return points[-1][1]


def fit_piecewise(
    function: Callable[[float], tuple[float, ...]],
    *,
    tolerance: float = CURVE_ERROR_BOUND,
) -> tuple[list[tuple[float, tuple[float, ...]]], float]:
    """Piecewise-linear points on [0, 1] and the measured max error against ``function``.

    The error is measured on 512 midpoints against the true function, not assumed from the
    reduction tolerance; the reduction tightens until the measured error is within ``tolerance``
    (or the point budget is hit, in which case the larger measured error is what gets reported).
    """

    dense = [(i / 256.0, tuple(function(i / 256.0))) for i in range(257)]
    probe = [((i + 0.5) / 512.0, tuple(function((i + 0.5) / 512.0))) for i in range(512)]

    def measured(points: list[tuple[float, tuple[float, ...]]]) -> float:
        return max(max(abs(a - b) for a, b in zip(truth, _interpolate(points, x))) for x, truth in probe)

    tol = tolerance
    points = _simplify(dense, tol)
    error = measured(points)
    for _ in range(12):
        if error <= tolerance or tol < 1.0e-5:
            break
        tol *= 0.75
        points = _simplify(dense, tol)
        error = measured(points)
    while len(points) > _MAX_CURVE_POINTS and tol < 0.05:
        tol *= 1.3
        points = _simplify(dense, tol)
        error = measured(points)
    return points, error


def _emit_piecewise(
    net: Network, source: Value, points: list[tuple[float, tuple[float, ...]]], kind: str, hint: str
) -> Value:
    """Evaluate the piecewise-linear function through ``range`` + ``mix`` pairs.

    ``f(x) = mix(f_prev, y_i, t_i)`` with ``t_i = clamp((x - x_{i-1}) / (x_i - x_{i-1}))``: for x in
    segment j every earlier step saturates at its own y and every later step is zero.
    ``source`` is a float (a ramp factor, one function applied to a scalar) or a colour (one curve
    applied to every channel); ``kind`` is the output type (float or colour).
    """

    def literal(channels: tuple[float, ...]) -> Value:
        if kind == "float":
            return const_float(channels[0])
        return const_color(channels * 3 if len(channels) == 1 else channels)

    if all(point[1] == points[0][1] for point in points):
        return literal(points[0][1])
    if source.is_const:
        if source.kind == "float":
            return literal(_interpolate(points, source.const[0]))
        return const_color(tuple(_interpolate(points, channel)[0] for channel in source.const))
    current = literal(points[0][1])
    ratio_color = source.kind == "color3"
    for (x0, _), (x1, y1) in zip(points, points[1:]):
        width = max(x1 - x0, _EPS)
        if ratio_color:
            ratio = net.emit("ND_range_color3", "color3", {
                "in": source, "inlow": const_color((x0,) * 3), "inhigh": const_color((x0 + width,) * 3),
                "outlow": const_color((0.0,) * 3), "outhigh": const_color((1.0,) * 3),
                "doclamp": ("boolean", True)}, "step")
        else:
            ratio = net.emit("ND_range_float", "float", {
                "in": source, "inlow": const_float(x0), "inhigh": const_float(x0 + width),
                "outlow": const_float(0.0), "outhigh": const_float(1.0),
                "doclamp": ("boolean", True)}, "step")
        if kind == "float":
            info = "ND_mix_float"
        else:
            info = "ND_mix_color3_color3" if ratio_color else "ND_mix_color3"
        current = net.emit(info, kind, {"fg": literal(y1), "bg": current, "mix": ratio}, "seg")
    return current


def _bound_text(error: float) -> str:
    """Error bound rounded up to four decimals, so the stated bound is never below the measured error."""

    return f"{math.ceil(max(error, 0.0) * 1.0e4) / 1.0e4:.4f}"


def _curve_functions(node: Any) -> list[Callable[[float], float]]:
    mapping = getattr(node, "mapping", None)
    curves = list(getattr(mapping, "curves", ()) or ())
    evaluate = getattr(mapping, "evaluate", None)
    if len(curves) < 4 or not callable(evaluate):
        raise Untranslatable("RGB Curves mapping is not readable")
    black = tuple(getattr(mapping, "black_level", (0.0, 0.0, 0.0)))
    white = tuple(getattr(mapping, "white_level", (1.0, 1.0, 1.0)))
    if any(abs(v) > _EPS for v in black[:3]) or any(abs(v - 1.0) > _EPS for v in white[:3]):
        raise Untranslatable("RGB Curves black/white levels are not translated")
    update = getattr(mapping, "update", None)
    if callable(update):
        update()
    combined = curves[3]
    return [
        (lambda x, c=channel, k=combined: float(evaluate(c, float(evaluate(k, x)))))
        for channel in curves[:3]
    ]


def _curve_rgb(net: Network, node: Any, output: str, want: str) -> Value:
    color = value_from_socket(net, _need(node, "Color"), "color3")
    fac_socket = _host()._input_socket(node, "Fac") or _host()._input_socket(node, "Factor")
    fac = value_from_socket(net, fac_socket, "float") if fac_socket is not None else const_float(1.0)
    functions = _curve_functions(node)
    identity = all(
        abs(function(i / 16.0) - i / 16.0) <= 1.0e-5 for function in functions for i in range(17)
    )
    if identity:
        return color
    same = all(
        abs(functions[0](i / 32.0) - function(i / 32.0)) <= 1.0e-5
        for function in functions[1:] for i in range(33)
    )
    if same:
        points, error = fit_piecewise(lambda x: (functions[0](x),))
        net.bound(f"CURVE_RGB:max_abs_error<={_bound_text(error)}")
        curved = _emit_piecewise(net, color, points, "color3", "curve")
    else:
        worst = 0.0
        channels = []
        for index, function in enumerate(functions):
            points, error = fit_piecewise(lambda x, f=function: (f(x),))
            worst = max(worst, error)
            channel = (const_float(color.const[index]) if color.is_const else net.emit(
                "ND_extract_color3", "float", {"in": color, "index": ("integer", index)}, "chan"))
            channels.append(_emit_piecewise(net, channel, points, "float", f"curve{index}"))
        net.bound(f"CURVE_RGB:max_abs_error<={_bound_text(worst)}")
        curved = net.emit("ND_combine3_color3", "color3",
                          {"in1": channels[0], "in2": channels[1], "in3": channels[2]}, "curve")
    return _mix(net, color, curved, _clamp01(net, fac), "curve_fac")


def _color_ramp(net: Network, node: Any, output: str, want: str) -> Value:
    if output not in {"Color", ""}:
        raise Untranslatable("Color Ramp alpha is not translated")
    ramp = getattr(node, "color_ramp", None)
    elements = list(getattr(ramp, "elements", ()) or ())
    if not elements:
        raise Untranslatable("Color Ramp has no stops")
    if str(getattr(ramp, "color_mode", "RGB")) != "RGB":
        raise Untranslatable("Color Ramp hue interpolation is not translated")
    interpolation = str(getattr(ramp, "interpolation", "LINEAR"))
    factor = value_from_socket(net, _need(node, "Fac" if _host()._input_socket(node, "Fac") else "Factor"), "float")
    stops = sorted(((float(e.position), tuple(float(c) for c in e.color[:3])) for e in elements),
                   key=lambda item: item[0])
    if interpolation == "LINEAR":
        points = stops
        if points[0][0] > 0.0:
            points = [(0.0, points[0][1]), *points]
        if points[-1][0] < 1.0:
            points = [*points, (1.0, points[-1][1])]
    elif interpolation == "CONSTANT":
        points = [(0.0, stops[0][1])]
        for index, (position, color) in enumerate(stops):
            if index:
                previous = stops[index - 1][1]
                points.append((max(position - 1.0e-5, points[-1][0]), previous))
                points.append((position, color))
        points.append((1.0, stops[-1][1]))
    else:
        evaluate = getattr(ramp, "evaluate", None)
        if not callable(evaluate):
            raise Untranslatable("Color Ramp cannot be sampled")
        points, error = fit_piecewise(lambda x: tuple(float(c) for c in evaluate(x)[:3]))
        net.bound(f"VALTORGB:{interpolation}:max_abs_error<={_bound_text(error)}")
    return _emit_piecewise(net, as_float(net, factor), points, "color3", "ramp")


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------

_MATH_FOLD: dict[str, Callable[[float, float], float]] = {
    "ADD": lambda a, b: a + b,
    "SUBTRACT": lambda a, b: a - b,
    "MULTIPLY": lambda a, b: a * b,
    "DIVIDE": lambda a, b: a / b if b != 0.0 else 0.0,
    "MINIMUM": min,
    "MAXIMUM": max,
    "LESS_THAN": lambda a, b: 1.0 if a < b else 0.0,
    "GREATER_THAN": lambda a, b: 1.0 if a > b else 0.0,
    "POWER": lambda a, b: a ** b if a > 0.0 or float(b).is_integer() else 0.0,
}
_MATH_NODE = {"ADD": "add", "SUBTRACT": "subtract", "MULTIPLY": "multiply", "DIVIDE": "divide",
              "MINIMUM": "min", "MAXIMUM": "max", "POWER": "power"}


def _math(net: Network, node: Any, output: str, want: str) -> Value:
    host = _host()
    operation = str(getattr(node, "operation", ""))
    sockets = host._input_sockets(node)
    unary = {"SQRT", "ABSOLUTE"}
    if operation not in _MATH_FOLD and operation not in unary:
        raise Untranslatable(f"Math {operation} is not translated")
    if len(sockets) < (1 if operation in unary else 2):
        raise Untranslatable("Math node lacks its inputs")
    first = value_from_socket(net, sockets[0], "float")
    if operation in unary:
        if first.is_const:
            result = const_float(math.sqrt(max(first.const[0], 0.0)) if operation == "SQRT" else abs(first.const[0]))
        else:
            result = net.emit("ND_sqrt_float" if operation == "SQRT" else "ND_absval_float", "float",
                              {"in": first}, operation.lower())
    else:
        second = value_from_socket(net, sockets[1], "float")
        if first.is_const and second.is_const:
            result = const_float(_MATH_FOLD[operation](first.const[0], second.const[0]))
        elif operation == "LESS_THAN":
            result = net.emit("ND_ifgreater_float", "float", {
                "value1": second, "value2": first, "in1": const_float(1.0), "in2": const_float(0.0)}, "less")
        elif operation == "GREATER_THAN":
            result = net.emit("ND_ifgreater_float", "float", {
                "value1": first, "value2": second, "in1": const_float(1.0), "in2": const_float(0.0)}, "greater")
        else:
            if operation == "POWER" and not first.is_const:
                first = net.emit("ND_max_float", "float", {"in1": first, "in2": const_float(0.0)}, "power_floor")
            result = _binary(net, _MATH_NODE[operation], "float", first, second, operation.lower())
    if bool(getattr(node, "use_clamp", False)):
        result = _clamp01(net, result)
    return result


# ---------------------------------------------------------------------------
# Vectors and procedural textures
# ---------------------------------------------------------------------------

def _vector_value(net: Network, socket: Any) -> Value:
    """A vector3 for a Noise/Wave input: UV or a rotation-free Mapping of it."""

    host = _host()
    link = host._socket_link(socket) if host._socket_linked(socket) else None
    if link is None:
        raise Untranslatable("procedural texture without a vector input uses generated coordinates")
    source = getattr(link, "from_node", None)
    output = str(getattr(getattr(link, "from_socket", None), "name", ""))
    node_type = host._node_type(source)
    if node_type == "TEX_COORD" and output == "UV":
        return _uv_vector(net)
    if node_type == "UVMAP" and output == "UV" and not str(getattr(source, "uv_map", "") or ""):
        return _uv_vector(net)
    if node_type != "MAPPING" or output != "Vector":
        raise Untranslatable(f"procedural vector from {node_type}.{output} is not translated")
    vector_type = str(getattr(source, "vector_type", "POINT"))
    if vector_type not in {"POINT", "TEXTURE", "VECTOR"}:
        raise Untranslatable("Mapping type is not translated")
    parts = {}
    for name in ("Location", "Rotation", "Scale"):
        mapping_socket = _need(source, name)
        if host._socket_linked(mapping_socket):
            raise Untranslatable("linked Mapping values are not translated")
        parts[name] = tuple(float(v) for v in mapping_socket.default_value)
    inner = _vector_value(net, _need(source, "Vector"))
    location, scale, rotation = parts["Location"], parts["Scale"], parts["Rotation"]
    if vector_type == "TEXTURE":
        if any(abs(s) <= _EPS for s in scale):
            raise Untranslatable("Mapping with zero scale")
        result = inner
        if any(abs(l) > _EPS for l in location):
            result = net.emit("ND_subtract_vector3", "vector3",
                              {"in1": result, "in2": Value("vector3", location)}, "map_offset")
        # Inverse of Blender's XYZ Euler rotation: undo Z, then Y, then X.
        result = _rotate_vector(net, result, rotation, inverse=True)
        return net.emit("ND_multiply_vector3", "vector3",
                        {"in1": result, "in2": Value("vector3", tuple(1.0 / s for s in scale))}, "map_scale")
    result = net.emit("ND_multiply_vector3", "vector3",
                      {"in1": inner, "in2": Value("vector3", scale)}, "map_scale")
    result = _rotate_vector(net, result, rotation, inverse=False)
    if vector_type == "POINT" and any(abs(l) > _EPS for l in location):
        result = net.emit("ND_add_vector3", "vector3",
                          {"in1": result, "in2": Value("vector3", location)}, "map_offset")
    return result


_AXES = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _rotate_vector(net: Network, vector: Value, rotation: tuple[float, ...], *, inverse: bool) -> Value:
    """Blender's XYZ Euler rotation (X applied first), or its inverse (Z undone first)."""

    order = (2, 1, 0) if inverse else (0, 1, 2)
    for axis in order:
        angle = math.degrees(rotation[axis])
        if abs(angle) <= _EPS:
            continue
        vector = net.emit("ND_rotate3d_vector3", "vector3", {
            "in": vector, "amount": const_float(-angle if inverse else angle),
            "axis": Value("vector3", _AXES[axis])}, "map_rotate")
    return vector


def _uv_vector(net: Network) -> Value:
    key = ("uv3", 0)
    cached = net._memo.get(key)
    if cached is None:
        uv = net.emit("ND_texcoord_vector2", "vector2", {"index": ("integer", 0)}, "uv")
        cached = net.emit("ND_convert_vector2_vector3", "vector3", {"in": uv}, "uv3")
        net._memo[key] = cached
    return cached


def _const_float_input(node: Any, name: str) -> float:
    socket = _need(node, name)
    if _host()._socket_linked(socket):
        raise Untranslatable(f"linked {name} is not translated")
    return float(socket.default_value)


def _scaled_vector(net: Network, node: Any, hint: str) -> Value | None:
    """The scaled coordinate vector, or None when the coordinates are not available.

    Generated and object coordinates need the mesh's bounding box, which a shared
    material does not have. Missing sockets and linked scales are still
    untranslatable (they raise).
    """

    vector_socket = _need(node, "Vector")
    scale = _const_float_input(node, "Scale")
    try:
        vector = _vector_value(net, vector_socket)
    except Untranslatable:
        return None
    if abs(scale - 1.0) <= _EPS:
        return vector
    return net.emit("ND_multiply_vector3FA", "vector3", {"in1": vector, "in2": const_float(scale)}, hint)


def _tex_noise(net: Network, node: Any, output: str, want: str) -> Value:
    """Perlin fBm through MaterialX ``fractal3d``, normalised to Blender's 0..1 range.

    The lattice frequency matches (unit cells at Scale 1) but the gradient hash
    differs, so the pattern is statistically equivalent, not identical.
    """

    if str(getattr(node, "noise_dimensions", "3D")) != "3D":
        raise Untranslatable("only 3D noise is translated")
    if str(getattr(node, "noise_type", "FBM")) != "FBM":
        raise Untranslatable("only fBm noise is translated")
    position = _scaled_vector(net, node, "noise_p")
    if position is None:
        # Use the texture's mean (0.5) rather than invent a frequency.
        net.approximate("TEX_NOISE.coordinates_unavailable_mean_0.5_used")
        return const_color((0.5, 0.5, 0.5)) if output == "Color" else const_float(0.5)
    detail = max(_const_float_input(node, "Detail"), 0.0)
    roughness = _const_float_input(node, "Roughness")
    lacunarity = _const_float_input(node, "Lacunarity")
    octaves = min(int(math.floor(detail)) + 1, 8)
    if abs(_const_float_input(node, "Distortion")) > _EPS:
        net.approximate("TEX_NOISE.distortion_ignored")
    if str(getattr(node, "normalize", True)) == "False":
        raise Untranslatable("unnormalised noise is not translated")
    max_amp = sum(roughness ** i for i in range(octaves)) or 1.0
    fractal = net.emit("ND_fractal3d_float", "float", {
        "amplitude": const_float(0.5 / max_amp), "octaves": ("integer", octaves),
        "lacunarity": const_float(lacunarity), "diminish": const_float(roughness),
        "position": position}, "noise")
    fac = net.emit("ND_add_float", "float", {"in1": fractal, "in2": const_float(0.5)}, "noise_fac")
    net.approximate("TEX_NOISE")
    if output in {"Fac", "Factor"}:
        return fac
    if output == "Color":
        net.approximate("TEX_NOISE.Color=gray")
        return as_color(net, fac)
    raise Untranslatable(f"Noise output {output} is not translated")


def _tex_wave(net: Network, node: Any, output: str, want: str) -> Value:
    """Sine bands along an axis; distortion and detail are omitted and reported."""

    if str(getattr(node, "wave_type", "BANDS")) != "BANDS":
        raise Untranslatable("ring waves are not translated")
    if str(getattr(node, "wave_profile", "SIN")) != "SIN":
        raise Untranslatable("wave profile is not translated")
    direction = str(getattr(node, "bands_direction", "X"))
    vector = _scaled_vector(net, node, "wave_p")
    if vector is None:
        net.approximate("TEX_WAVE.coordinates_unavailable_mean_0.5_used")
        return const_color((0.5, 0.5, 0.5)) if output == "Color" else const_float(0.5)
    if direction == "DIAGONAL":
        axes = (0, 1, 2)
    else:
        axes = ({"X": 0, "Y": 1, "Z": 2}.get(direction, 0),)
    parts = []
    for index in axes:
        parts.append(_extract_vector3(net, vector, index))
    coordinate = parts[0]
    for extra in parts[1:]:
        coordinate = net.emit("ND_add_float", "float", {"in1": coordinate, "in2": extra}, "wave_sum")
    frequency = 10.0 if direction == "DIAGONAL" else 20.0
    phase = _const_float_input(node, "Phase Offset")
    arg = net.emit("ND_multiply_float", "float", {"in1": coordinate, "in2": const_float(frequency)}, "wave_n")
    arg = net.emit("ND_add_float", "float", {"in1": arg, "in2": const_float(phase - math.pi / 2.0)}, "wave_phase")
    sine = net.emit("ND_sin_float", "float", {"in": arg}, "wave_sin")
    half = net.emit("ND_multiply_float", "float", {"in1": sine, "in2": const_float(0.5)}, "wave_half")
    fac = net.emit("ND_add_float", "float", {"in1": half, "in2": const_float(0.5)}, "wave_fac")
    if _const_float_input(node, "Distortion") > _EPS:
        net.approximate("TEX_WAVE.distortion_ignored")
    net.approximate("TEX_WAVE.detail_ignored" if _const_float_input(node, "Detail") > _EPS else "TEX_WAVE")
    return as_color(net, fac) if output == "Color" else fac


def _extract_vector3(net: Network, vector: Value, index: int) -> Value:
    return net.emit("ND_extract_vector3", "float", {"in": vector, "index": ("integer", index)}, "axis")


_HANDLERS: dict[str, Callable[[Network, Any, str, str], Value]] = {
    "TEX_IMAGE": _tex_image,
    "RGB": _rgb_node,
    "VALUE": _value_node,
    "OBJECT_INFO": _object_info,
    "ATTRIBUTE": _attribute,
    "MIX": _mix_node,
    "MIX_RGB": _mix_node,
    "HUE_SAT": _hue_sat,
    "BRIGHTCONTRAST": _bright_contrast,
    "GAMMA": _gamma,
    "INVERT": _invert,
    "SEPARATE_COLOR": _separate_color,
    "SEPRGB": _separate_color,
    "CURVE_RGB": _curve_rgb,
    "VALTORGB": _color_ramp,
    "MATH": _math,
    "TEX_NOISE": _tex_noise,
    "TEX_WAVE": _tex_wave,
}


def translate_socket(prefix: str, socket: Any, kind: str) -> tuple[Network, Value]:
    """Build the network for ``socket``; raises ``Untranslatable`` when it cannot."""

    net = Network(prefix, fallback_color_space="sRGB" if kind == "color3" else "raw")
    value = value_from_socket(net, socket, kind)
    return net, value

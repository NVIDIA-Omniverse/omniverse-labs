# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Light proxies for emissive meshes (GH #44 lane 2) and their camera visibility.

OVRTX delivers only about 41% of the illumination a Cycles emissive mesh does
(measured on a grey receiver, GH #44). Each emissive mesh therefore gets a
UsdLux proxy of Cycles-equivalent power: radiance L over emitting area A is a
Blender light of power ``pi * L * A``, converted with the same policy as every
other light. The mesh keeps its camera-visible emission and is made
``invisibleToSecondaryRays`` so its own (weak) transport is not counted twice.
Proxy lights are not camera visible (UsdLux lights are not seen by primary
rays unless ``visibleInPrimaryRay`` is authored).

The analysis is scene agnostic: it reads the stock-exported USD composition
(mesh geometry, material bindings) and the Blender material node trees named
by the exported materials. Nothing here knows a scene or object name.
"""

from __future__ import annotations

import colorsys
import math
from typing import Any, Callable, Iterable, Mapping, Sequence

from . import light_value_conversion as _lights
from .sky_texture_radiance import sky_texture_radiance

PROXY_PRIM_NAME = "OvrtxEmissiveProxy"
INVISIBLE_TO_SECONDARY_PRIMVAR = "invisibleToSecondaryRays"
MATERIAL_NAME_ATTRIBUTE = "userProperties:blender:data_name"
MIN_RADIANCE = 1.0e-4
MIN_AREA = 1.0e-8
PLANAR_NORMAL_COHERENCE = 0.97

SIDE_BOTH = "both"
SIDE_FRONT = "front"
SIDE_BACK = "back"

Color = tuple[float, float, float]


class EmissionSpec:
    """Constant radiance a material emits to each side of its surface."""

    __slots__ = ("front", "back", "approximations", "unsupported")

    def __init__(self, front: Color = (0.0, 0.0, 0.0), back: Color = (0.0, 0.0, 0.0),
                 approximations: Sequence[str] = (), unsupported: Sequence[str] = ()) -> None:
        self.front = front
        self.back = back
        self.approximations = tuple(approximations)
        self.unsupported = tuple(unsupported)

    @property
    def emits(self) -> bool:
        return max(self.front) > MIN_RADIANCE or max(self.back) > MIN_RADIANCE

    def diagnostics(self) -> dict[str, Any]:
        return {"front": self.front, "back": self.back, "approximations": list(self.approximations),
                "unsupported": list(self.unsupported)}


# --- Blender material analysis --------------------------------------------------------------

def material_emission(material: Any) -> EmissionSpec:
    """Constant emission radiance of a Blender material's active surface."""

    tree = getattr(material, "node_tree", None)
    nodes = list(getattr(tree, "nodes", ()) or ()) if tree is not None else []
    if not nodes or not bool(getattr(material, "use_nodes", True)):
        return EmissionSpec()
    outputs = [node for node in nodes if _lights._node_kind(node) == "OUTPUT_MATERIAL"]
    active = [node for node in outputs if bool(getattr(node, "is_active_output", True))
              and str(getattr(node, "target", "ALL")) in {"ALL", "CYCLES"}] or outputs
    if not active:
        return EmissionSpec()
    link = _lights._linked_from(_lights._socket(active[0], "Surface"))
    if link is None:
        return EmissionSpec()
    notes: list[str] = []
    unsupported: list[str] = []
    terms = _surface_terms(link[0], notes, unsupported, depth=6)
    front = (0.0, 0.0, 0.0)
    back = (0.0, 0.0, 0.0)
    for side, color in terms:
        if side in {SIDE_BOTH, SIDE_FRONT}:
            front = _add(front, color)
        if side in {SIDE_BOTH, SIDE_BACK}:
            back = _add(back, color)
    return EmissionSpec(front, back, tuple(dict.fromkeys(notes)), tuple(dict.fromkeys(unsupported)))


def _surface_terms(node: Any, notes: list[str], unsupported: list[str], *, depth: int) -> list[tuple[str, Color]]:
    kind = _lights._node_kind(node)
    if depth <= 0 or kind in {"MUTED", ""}:
        return []
    if kind == "EMISSION":
        return _emission_term(node, "Color", "Strength", notes, unsupported)
    if kind == "BSDF_PRINCIPLED":
        return _emission_term(node, "Emission Color", "Emission Strength", notes, unsupported)
    if kind == "ADD_SHADER":
        terms: list[tuple[str, Color]] = []
        for socket in _shader_inputs(node):
            terms += _terms_from_socket(socket, notes, unsupported, depth)
        return terms
    if kind == "MIX_SHADER":
        return _mix_terms(node, notes, unsupported, depth)
    if kind in {"BSDF_TRANSPARENT", "BSDF_DIFFUSE", "BSDF_GLOSSY", "BSDF_GLASS", "BSDF_REFRACTION", "BSDF_TRANSLUCENT",
                "BSDF_SHEEN", "BSDF_TOON", "BSDF_HAIR", "BSDF_HAIR_PRINCIPLED", "BSDF_METALLIC", "SUBSURFACE_SCATTERING",
                "HOLDOUT", "VOLUME_ABSORPTION", "VOLUME_SCATTER", "BACKGROUND", "EEVEE_SPECULAR"}:
        return []
    unsupported.append("surface_node:" + kind)
    return []


def _shader_inputs(node: Any) -> list[Any]:
    return [socket for socket in (getattr(node, "inputs", ()) or ()) if str(getattr(socket, "type", "")) == "SHADER"]


def _terms_from_socket(socket: Any, notes: list[str], unsupported: list[str], depth: int) -> list[tuple[str, Color]]:
    link = _lights._linked_from(socket)
    return _surface_terms(link[0], notes, unsupported, depth=depth - 1) if link is not None else []


def _mix_terms(node: Any, notes: list[str], unsupported: list[str], depth: int) -> list[tuple[str, Color]]:
    shaders = _shader_inputs(node)
    if len(shaders) < 2:
        return []
    first = _terms_from_socket(shaders[0], notes, unsupported, depth)
    second = _terms_from_socket(shaders[1], notes, unsupported, depth)
    factor_socket = _lights._socket(node, "Fac") or _lights._socket(node, "Factor")
    link = _lights._linked_from(factor_socket)
    if link is None:
        factor = min(1.0, max(0.0, _lights._socket_float(factor_socket, 0.5)))
        return _scaled(first, 1.0 - factor) + _scaled(second, factor)
    source, output = link
    kind = _lights._node_kind(source)
    if kind == "NEW_GEOMETRY" and output == "Backfacing":
        # Fac = 1 on back faces: the first shader is seen from the front, the second from behind.
        notes.append("mix_backfacing_one_sided_emission")
        return _sided(first, SIDE_FRONT) + _sided(second, SIDE_BACK)
    if kind == "LIGHT_PATH" and output == "Is Camera Ray":
        notes.append("mix_is_camera_ray_light_transport_branch")
        return first
    if kind == "LIGHT_PATH" and output in {"Is Shadow Ray", "Is Diffuse Ray", "Is Glossy Ray", "Is Reflection Ray",
                                           "Is Transmission Ray", "Is Singular Ray"}:
        notes.append("mix_light_path_branch:" + output)
        return first
    factor = min(1.0, max(0.0, _lights._socket_float(factor_socket, 0.5)))
    notes.append("mix_linked_factor_constant:" + kind)
    return _scaled(first, 1.0 - factor) + _scaled(second, factor)


def _scaled(terms: list[tuple[str, Color]], weight: float) -> list[tuple[str, Color]]:
    return [(side, tuple(channel * weight for channel in color)) for side, color in terms]  # type: ignore[misc]


def _sided(terms: list[tuple[str, Color]], side: str) -> list[tuple[str, Color]]:
    result = []
    for term_side, color in terms:
        if term_side == SIDE_BOTH or term_side == side:
            result.append((side, color))
    return result


def _emission_term(node: Any, color_name: str, strength_name: str, notes: list[str],
                   unsupported: list[str]) -> list[tuple[str, Color]]:
    strength_socket = _lights._socket(node, strength_name)
    if _lights._linked_from(strength_socket) is not None:
        unsupported.append("linked_emission_strength")
        return []
    strength = _lights._socket_float(strength_socket, 1.0 if strength_name == "Strength" else 0.0)
    if strength <= MIN_RADIANCE:
        return []
    color = _static_color(_lights._socket(node, color_name), notes, unsupported, depth=8)
    if color is None:
        return []
    radiance = tuple(max(0.0, channel) * strength for channel in color)
    return [(SIDE_BOTH, radiance)] if max(radiance) > MIN_RADIANCE else []  # type: ignore[list-item]


def _static_color(socket: Any | None, notes: list[str], unsupported: list[str], *, depth: int) -> Color | None:
    """Constant colour a socket evaluates to; ``None`` when it cannot be reduced."""

    link = _lights._linked_from(socket)
    if link is None:
        return _lights._socket_color(socket)
    node = link[0]
    if depth <= 0:
        unsupported.append("color_chain_too_deep")
        return None
    kind = _lights._node_kind(node)
    if kind == "RGB":
        return _lights._socket_color(getattr(node, "outputs", {}).get("Color"))  # type: ignore[union-attr]
    if kind == "VALUE":
        value = _lights._socket_float(getattr(node, "outputs", {}).get("Value"), 1.0)  # type: ignore[union-attr]
        return (value, value, value)
    if kind == "BLACKBODY":
        if _lights._linked_from(_lights._socket(node, "Temperature")) is not None:
            unsupported.append("linked_blackbody_temperature")
            return None
        return _lights.blackbody_rgb(_lights._socket_float(_lights._socket(node, "Temperature"), 1500.0))
    if kind == "TEX_SKY":
        color, note = sky_texture_radiance(node)
        if note:
            notes.append(note)
        return color
    if kind in {"TEX_IMAGE", "TEX_ENVIRONMENT", "TEX_NOISE", "TEX_VORONOI", "TEX_WAVE", "TEX_CHECKER", "TEX_GRADIENT"}:
        unsupported.append("textured_emission_color:" + kind)
        return None
    inner = _static_color(_lights._socket(node, "Color"), notes, unsupported, depth=depth - 1) \
        if kind in {"HUE_SAT", "CURVE_RGB", "GAMMA", "BRIGHTCONTRAST", "INVERT"} else None
    if kind == "HUE_SAT" and inner is not None:
        return _hue_sat(node, inner)
    if kind == "CURVE_RGB" and inner is not None:
        return _curve_rgb(node, inner, notes)
    if kind == "GAMMA" and inner is not None:
        gamma = _lights._socket_float(_lights._socket(node, "Gamma"), 1.0)
        return tuple(max(0.0, channel) ** gamma for channel in inner)  # type: ignore[return-value]
    if kind == "BRIGHTCONTRAST" and inner is not None:
        bright = _lights._socket_float(_lights._socket(node, "Bright"), 0.0)
        contrast = _lights._socket_float(_lights._socket(node, "Contrast"), 0.0)
        slope = 1.0 + contrast if contrast >= 0 else 1.0 / (1.0 - contrast)
        return tuple(max(0.0, slope * (channel - 0.5) + 0.5 + bright) for channel in inner)  # type: ignore[return-value]
    if kind == "INVERT" and inner is not None:
        fac = _lights._socket_float(_lights._socket(node, "Fac"), 1.0)
        return tuple(channel + fac * (1.0 - 2.0 * channel) for channel in inner)  # type: ignore[return-value]
    if kind in {"MIX", "MIX_RGB"}:
        a = _static_color(_lights._socket(node, "A") or _lights._socket(node, "Color1"), notes, unsupported, depth=depth - 1)
        b = _static_color(_lights._socket(node, "B") or _lights._socket(node, "Color2"), notes, unsupported, depth=depth - 1)
        if a is not None and b is not None:
            fac_socket = _lights._socket(node, "Factor") or _lights._socket(node, "Fac")
            fac = min(1.0, max(0.0, _lights._socket_float(fac_socket, 0.5)))
            notes.append("mix_color_node_approximated")
            return tuple(x + fac * (y - x) for x, y in zip(a, b))  # type: ignore[return-value]
        return None
    unsupported.append("emission_color_node:" + kind)
    return None


def _hue_sat(node: Any, color: Color) -> Color:
    hue = _lights._socket_float(_lights._socket(node, "Hue"), 0.5)
    saturation = _lights._socket_float(_lights._socket(node, "Saturation"), 1.0)
    value = _lights._socket_float(_lights._socket(node, "Value"), 1.0)
    fac = min(1.0, max(0.0, _lights._socket_float(_lights._socket(node, "Fac") or _lights._socket(node, "Factor"), 1.0)))
    peak = max(color)
    scale = peak if peak > 1.0 else 1.0
    h, s, v = colorsys.rgb_to_hsv(*(min(1.0, channel / scale) for channel in color))
    h = (h + hue - 0.5) % 1.0
    s = min(1.0, max(0.0, s * saturation))
    r, g, b = colorsys.hsv_to_rgb(h, s, v * value)
    shifted = (r * scale, g * scale, b * scale)
    return tuple(c + fac * (n - c) for c, n in zip(color, shifted))  # type: ignore[return-value]


def _curve_rgb(node: Any, color: Color, notes: list[str]) -> Color:
    mapping = getattr(node, "mapping", None)
    curves = list(getattr(mapping, "curves", ()) or ())
    evaluate = getattr(mapping, "evaluate", None)
    if len(curves) < 4 or not callable(evaluate):
        notes.append("rgb_curves_not_evaluated")
        return color
    fac = min(1.0, max(0.0, _lights._socket_float(_lights._socket(node, "Fac") or _lights._socket(node, "Factor"), 1.0)))
    try:
        curved = tuple(float(evaluate(curves[3], float(evaluate(curves[index], channel)))) for index, channel in enumerate(color))
    except (TypeError, ValueError, RuntimeError):
        notes.append("rgb_curves_not_evaluated")
        return color
    return tuple(c + fac * (n - c) for c, n in zip(color, curved))  # type: ignore[return-value]


def _add(a: Color, b: Color) -> Color:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


# --- Mesh geometry ---------------------------------------------------------------------------

def _sub(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _length(a: Sequence[float]) -> float:
    return math.sqrt(_dot(a, a))


def _unit(a: Sequence[float]) -> tuple[float, float, float]:
    length = _length(a)
    return (a[0] / length, a[1] / length, a[2] / length) if length > 1.0e-20 else (0.0, 0.0, 1.0)


class EmitterGeometry:
    """World-space area, mean normal and bounds of a set of emitting faces."""

    __slots__ = ("area", "normal", "coherence", "center", "points", "face_count")

    def __init__(self, area: float, normal: tuple[float, float, float], coherence: float,
                 center: tuple[float, float, float], points: list[tuple[float, float, float]], face_count: int) -> None:
        self.area, self.normal, self.coherence, self.center, self.points, self.face_count = (
            area, normal, coherence, center, points, face_count)


def emitter_geometry(points: Sequence[Sequence[float]], counts: Sequence[int], indices: Sequence[int],
                     faces: Iterable[int] | None, *, left_handed: bool = False) -> EmitterGeometry | None:
    """Area, area-weighted normal and centroid of ``faces`` (all faces when ``None``)."""

    starts: list[int] = []
    offset = 0
    for count in counts:
        starts.append(offset)
        offset += int(count)
    selected = range(len(counts)) if faces is None else faces
    total = 0.0
    weighted = [0.0, 0.0, 0.0]
    centroid = [0.0, 0.0, 0.0]
    used: dict[int, tuple[float, float, float]] = {}
    face_count = 0
    sign = -1.0 if left_handed else 1.0
    for face in selected:
        count = int(counts[face])
        if count < 3:
            continue
        ring = [points[indices[starts[face] + k]] for k in range(count)]
        origin = ring[0]
        normal_sum = [0.0, 0.0, 0.0]
        for k in range(1, count - 1):
            c = _cross(_sub(ring[k], origin), _sub(ring[k + 1], origin))
            normal_sum[0] += c[0]
            normal_sum[1] += c[1]
            normal_sum[2] += c[2]
        twice_area = _length(normal_sum)
        area = 0.5 * twice_area
        if area <= 0.0:
            continue
        total += area
        for axis in range(3):
            weighted[axis] += sign * 0.5 * normal_sum[axis]
            centroid[axis] += area * (sum(p[axis] for p in ring) / count)
        for k in range(count):
            vertex = indices[starts[face] + k]
            used[vertex] = tuple(points[vertex])  # type: ignore[assignment]
        face_count += 1
    if total <= MIN_AREA:
        return None
    mean_normal = _unit(weighted)
    coherence = _length(weighted) / total
    center = (centroid[0] / total, centroid[1] / total, centroid[2] / total)
    return EmitterGeometry(total, mean_normal, coherence, center, list(used.values()), face_count)


class ProxyShape:
    """A proxy light: kind, world placement and Cycles-equivalent UsdLux values."""

    __slots__ = ("kind", "side", "center", "axes", "width", "height", "radius", "intensity", "color", "emit_area",
                 "shape_area")

    def __init__(self, **values: Any) -> None:
        for name in self.__slots__:
            setattr(self, name, values.get(name))


def plan_proxies(geometry: EmitterGeometry, spec: EmissionSpec) -> list[ProxyShape]:
    """Proxy lights reproducing ``spec`` radiance over ``geometry`` (world space)."""

    result: list[ProxyShape] = []
    if geometry.coherence >= PLANAR_NORMAL_COHERENCE:
        normal = geometry.normal
        u, v, width, height, center = _plane_frame(geometry, normal)
        shape_area = max(width * height, MIN_AREA)
        for side, radiance in ((SIDE_FRONT, spec.front), (SIDE_BACK, spec.back)):
            if max(radiance) <= MIN_RADIANCE:
                continue
            direction = normal if side == SIDE_FRONT else (-normal[0], -normal[1], -normal[2])
            z_axis = (-direction[0], -direction[1], -direction[2])
            x_axis = u
            y_axis = _cross(z_axis, x_axis)
            if side == SIDE_BACK:
                # Keep a right-handed light frame that still faces away from the surface.
                x_axis = (-u[0], -u[1], -u[2])
                y_axis = _cross(z_axis, x_axis)
            result.append(ProxyShape(kind="RectLight", side=side, center=center, axes=(x_axis, y_axis, z_axis),
                                     width=width, height=height, radius=0.0, color=radiance,
                                     intensity=_proxy_intensity(radiance, geometry.area, shape_area),
                                     emit_area=geometry.area, shape_area=shape_area))
        return result
    radiance = tuple(max(a, b) for a, b in zip(spec.front, spec.back))
    if max(radiance) <= MIN_RADIANCE:
        return result
    radius = math.sqrt(geometry.area / (4.0 * math.pi))
    shape_area = 4.0 * math.pi * radius * radius
    result.append(ProxyShape(kind="SphereLight", side=SIDE_BOTH, center=geometry.center,
                             axes=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), width=0.0, height=0.0,
                             radius=radius, color=radiance,
                             intensity=_proxy_intensity(radiance, geometry.area, shape_area),
                             emit_area=geometry.area, shape_area=shape_area))
    return result


def _proxy_intensity(radiance: Color, emit_area: float, shape_area: float) -> float:
    """Blender power ``pi*L*A`` through the shared Blender-to-UsdLux area policy.

    The area policy gives ``120*pi*P / shape_area``; with ``P = pi*L*A`` the
    shape area only matters when the proxy shape is larger than the emitter.
    """

    peak = max(radiance)
    power = math.pi * peak * emit_area
    return _lights.blender_energy_to_intensity(power) / max(shape_area, MIN_AREA)


def _plane_frame(geometry: EmitterGeometry, normal: tuple[float, float, float]):
    reference = (0.0, 0.0, 1.0) if abs(normal[2]) < 0.9 else (1.0, 0.0, 0.0)
    u0 = _unit(_cross(reference, normal))
    v0 = _cross(normal, u0)
    xs = [_dot(_sub(p, geometry.center), u0) for p in geometry.points]
    ys = [_dot(_sub(p, geometry.center), v0) for p in geometry.points]
    sxx = sum(x * x for x in xs)
    syy = sum(y * y for y in ys)
    sxy = sum(x * y for x, y in zip(xs, ys))
    angle = 0.5 * math.atan2(2.0 * sxy, sxx - syy)
    c, s = math.cos(angle), math.sin(angle)
    u = (u0[0] * c + v0[0] * s, u0[1] * c + v0[1] * s, u0[2] * c + v0[2] * s)
    v = _cross(normal, u)
    pu = [_dot(_sub(p, geometry.center), u) for p in geometry.points]
    pv = [_dot(_sub(p, geometry.center), v) for p in geometry.points]
    width = max(pu) - min(pu)
    height = max(pv) - min(pv)
    mid_u = 0.5 * (max(pu) + min(pu))
    mid_v = 0.5 * (max(pv) + min(pv))
    center = (geometry.center[0] + u[0] * mid_u + v[0] * mid_v,
              geometry.center[1] + u[1] * mid_u + v[1] * mid_v,
              geometry.center[2] + u[2] * mid_u + v[2] * mid_v)
    return u, v, max(width, 1.0e-4), max(height, 1.0e-4), center


# --- USD authoring ----------------------------------------------------------------------------

def opinion_records(stock_stage: Any, material_lookup: Callable[[str], Any | None]) -> tuple[Any, ...]:
    """Add-on opinion records for emissive-mesh proxies and the matching mesh overrides."""

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade
    from .usd_opinion_records import AddOnUsdOpinionRecord

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    specs: dict[str, tuple[EmissionSpec | None, str]] = {}

    def spec_for(material: Any) -> EmissionSpec | None:
        if material is None or not material:
            return None
        path = str(material.GetPath())
        if path not in specs:
            attribute = material.GetPrim().GetAttribute(MATERIAL_NAME_ATTRIBUTE)
            name = str(attribute.Get()) if attribute and attribute.HasValue() else ""
            source = material_lookup(name) if name else None
            specs[path] = (material_emission(source) if source is not None else None, name)
        return specs[path][0]

    records: list[Any] = []
    for prim in stock_stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        faces_total = len(mesh.GetFaceVertexCountsAttr().Get() or ())
        if not faces_total:
            continue
        bindings: list[tuple[Any, list[int] | None]] = []
        subsets = UsdGeom.Subset.GetGeomSubsets(mesh, elementType=UsdGeom.Tokens.face, familyName="materialBind")
        covered = 0
        for subset in subsets:
            material = UsdShade.MaterialBindingAPI(subset.GetPrim()).ComputeBoundMaterial()[0]
            indices = list(subset.GetIndicesAttr().Get() or ())
            covered += len(indices)
            bindings.append((material, indices))
        if not subsets or covered < faces_total:
            material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
            if not subsets:
                bindings = [(material, None)]
            else:
                taken = {i for _, idx in bindings for i in (idx or ())}
                bindings.append((material, [i for i in range(faces_total) if i not in taken]))
        emitting = [(material, faces) for material, faces in bindings
                    if (spec := spec_for(material)) is not None and spec.emits]
        if not emitting:
            continue
        mixed = len(emitting) != len(bindings)
        points = mesh.GetPointsAttr().Get() or ()
        counts = mesh.GetFaceVertexCountsAttr().Get() or ()
        indices = mesh.GetFaceVertexIndicesAttr().Get() or ()
        orientation = mesh.GetOrientationAttr().Get()
        left = str(orientation) == "leftHanded"
        world = cache.GetLocalToWorldTransform(prim)
        world_points = [tuple(world.Transform(Gf.Vec3d(*map(float, p)))) for p in points]
        mesh_path = prim.GetPath()
        proxies: list[tuple[ProxyShape, str, EmissionSpec]] = []
        notes: list[str] = []
        for slot, (material, faces) in enumerate(emitting):
            spec = specs[str(material.GetPath())][0]
            geometry = emitter_geometry(world_points, counts, indices, faces, left_handed=left)
            if geometry is None:
                continue
            planned = plan_proxies(geometry, spec)
            proxies += [(shape, f"{slot}", spec) for shape in planned]
            notes += list(spec.approximations) + list(spec.unsupported)
        if not proxies:
            continue
        if mixed:
            # A mesh with both emitting and non-emitting faces cannot be hidden from
            # secondary rays without hiding its body, so it would count its own
            # emission twice. Leave it on mesh emission and report why.
            continue
        inverse = world.GetInverse()
        names = set()
        for shape, slot, spec in proxies:
            suffix = f"_{slot}_{shape.side}" if shape.kind == "RectLight" else f"_{slot}"
            name = PROXY_PRIM_NAME + suffix
            if name in names:
                continue
            names.add(name)
            layer = Sdf.Layer.CreateAnonymous(".usda")
            stage = Usd.Stage.Open(layer)
            stage.OverridePrim(mesh_path)
            light_path = mesh_path.AppendChild(name)
            if shape.kind == "RectLight":
                light = UsdLux.RectLight.Define(stage, light_path)
                light.CreateWidthAttr().Set(float(shape.width))
                light.CreateHeightAttr().Set(float(shape.height))
            else:
                light = UsdLux.SphereLight.Define(stage, light_path)
                light.CreateRadiusAttr().Set(float(shape.radius))
            peak = max(shape.color)
            light.CreateIntensityAttr().Set(float(shape.intensity))
            light.CreateColorAttr().Set(Gf.Vec3f(*(float(c / peak) for c in shape.color)))
            light.CreateNormalizeAttr().Set(False)
            light.CreateEnableColorTemperatureAttr().Set(False)
            matrix = Gf.Matrix4d(1.0)
            for row, axis in enumerate(shape.axes):
                matrix.SetRow3(row, Gf.Vec3d(*axis))
            matrix.SetTranslateOnly(Gf.Vec3d(*shape.center))
            local = matrix * inverse
            UsdGeom.Xformable(light.GetPrim()).AddTransformOp().Set(local)
            records.append(AddOnUsdOpinionRecord(
                str(light_path), layer,
                {"source": "emissive_light_proxy", "kind": shape.kind, "side": shape.side,
                 "emit_area": shape.emit_area, "shape_area": shape.shape_area, "intensity": shape.intensity,
                 "mesh": str(mesh_path), "notes": tuple(dict.fromkeys(notes))}))
        layer = Sdf.Layer.CreateAnonymous(".usda")
        stage = Usd.Stage.Open(layer)
        override = stage.OverridePrim(mesh_path)
        UsdGeom.PrimvarsAPI(override).CreatePrimvar(
            INVISIBLE_TO_SECONDARY_PRIMVAR, Sdf.ValueTypeNames.Bool, UsdGeom.Tokens.constant).Set(True)
        records.append(AddOnUsdOpinionRecord(
            str(mesh_path), layer, {"source": "emissive_light_proxy_mesh", "mesh": str(mesh_path)}))
    return tuple(records)


__all__ = [
    "EmissionSpec",
    "emitter_geometry",
    "material_emission",
    "opinion_records",
    "plan_proxies",
]

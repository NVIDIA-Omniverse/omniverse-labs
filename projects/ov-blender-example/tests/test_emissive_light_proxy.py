# SPDX-License-Identifier: Apache-2.0
"""Emissive-mesh light proxies (GH #44 lane 2) and camera ray visibility."""
import math
from types import SimpleNamespace as N

import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

from ovrtx_blender_example import emissive_light_proxy as proxy
from ovrtx_blender_example import light_value_conversion as lights
from ovrtx_blender_example.sky_texture_radiance import sky_texture_radiance


class _Inputs(dict):
    """Blender's socket collection: lookup by name, iteration over sockets."""

    def __iter__(self):
        return iter(list(self.values()))

    def get(self, name, default=None):
        return super().get(name, default)


def _socket(name, value=None, kind="VALUE"):
    return N(name=name, default_value=value, links=[], type=kind)


def _node(kind, outputs=(), **inputs):
    return N(type=kind, is_active_output=True, target="ALL", mute=False,
             inputs=_Inputs({k: _socket(k, v, "SHADER" if k.startswith("Shader") else "VALUE") for k, v in inputs.items()}),
             outputs=_Inputs({o: _socket(o, (1.0, 1.0, 1.0, 1.0)) for o in outputs}))


def _link(target, socket_name, source, output):
    target.inputs[socket_name].links.append(N(from_node=source, from_socket=N(name=output), is_muted=False))


def _material(surface, *extra):
    out = _node("OUTPUT_MATERIAL", Surface=None)
    _link(out, "Surface", surface, "BSDF")
    return N(use_nodes=True, node_tree=N(nodes=[out, surface, *extra]))


def test_emission_shader_radiance_is_color_times_strength():
    emission = _node("EMISSION", Color=(1.0, 0.5, 0.25, 1.0), Strength=4.0)
    spec = proxy.material_emission(_material(emission))
    assert spec.front == pytest.approx((4.0, 2.0, 1.0)) and spec.back == spec.front
    assert spec.emits and not spec.unsupported


def test_principled_emission_and_add_shader():
    principled = _node("BSDF_PRINCIPLED", **{"Emission Color": (1.0, 1.0, 1.0, 1.0), "Emission Strength": 3.0})
    assert proxy.material_emission(_material(principled)).front == pytest.approx((3.0, 3.0, 3.0))
    emission = _node("EMISSION", Color=(1.0, 1.0, 1.0, 1.0), Strength=5.0)
    diffuse = _node("BSDF_DIFFUSE", Color=(0.8, 0.8, 0.8, 1.0))
    add = _node("ADD_SHADER", Shader=None, Shader_001=None)
    _link(add, "Shader", diffuse, "BSDF")
    _link(add, "Shader_001", emission, "Emission")
    spec = proxy.material_emission(_material(add, diffuse, emission))
    assert spec.front == pytest.approx((5.0, 5.0, 5.0))


def test_mix_with_transparent_by_backfacing_emits_to_one_side_only():
    transparent = _node("BSDF_TRANSPARENT", Color=(1.0, 1.0, 1.0, 1.0))
    emission = _node("EMISSION", Color=(1.0, 1.0, 1.0, 1.0), Strength=20.0)
    geometry = _node("NEW_GEOMETRY", outputs=("Backfacing",))
    mix = _node("MIX_SHADER", Fac=0.0, Shader=None, Shader_001=None)
    _link(mix, "Fac", geometry, "Backfacing")
    _link(mix, "Shader", transparent, "BSDF")
    _link(mix, "Shader_001", emission, "Emission")
    spec = proxy.material_emission(_material(mix, transparent, emission, geometry))
    assert max(spec.front) == 0.0
    assert spec.back == pytest.approx((20.0, 20.0, 20.0))
    assert "mix_backfacing_one_sided_emission" in spec.approximations


def test_constant_mix_factor_weights_emission():
    transparent = _node("BSDF_TRANSPARENT", Color=(1.0, 1.0, 1.0, 1.0))
    emission = _node("EMISSION", Color=(1.0, 1.0, 1.0, 1.0), Strength=10.0)
    mix = _node("MIX_SHADER", Fac=0.25, Shader=None, Shader_001=None)
    _link(mix, "Shader", transparent, "BSDF")
    _link(mix, "Shader_001", emission, "Emission")
    assert proxy.material_emission(_material(mix, transparent, emission)).front == pytest.approx((2.5, 2.5, 2.5))


def test_sky_texture_colour_uses_hemisphere_mean_and_blackbody_is_evaluated():
    sky = _node("TEX_SKY", outputs=("Color",))
    emission = _node("EMISSION", Color=(1.0, 1.0, 1.0, 1.0), Strength=2.0)
    _link(emission, "Color", sky, "Color")
    spec = proxy.material_emission(_material(emission, sky))
    assert spec.front == pytest.approx(tuple(2.0 * c for c in sky_texture_radiance(sky)[0]))
    assert "sky_texture_horizon_band_mean" in spec.approximations
    blackbody = _node("BLACKBODY", Temperature=5000.0)
    emission = _node("EMISSION", Color=(1.0, 1.0, 1.0, 1.0), Strength=1.0)
    _link(emission, "Color", blackbody, "Color")
    assert proxy.material_emission(_material(emission, blackbody)).front == pytest.approx(lights.blackbody_rgb(5000.0))


def test_textured_or_linked_strength_emission_is_reported_not_guessed():
    image = _node("TEX_IMAGE", outputs=("Color",))
    emission = _node("EMISSION", Color=(1.0, 1.0, 1.0, 1.0), Strength=2.0)
    _link(emission, "Color", image, "Color")
    spec = proxy.material_emission(_material(emission, image))
    assert not spec.emits and "textured_emission_color:TEX_IMAGE" in spec.unsupported
    value = _node("VALUE", outputs=("Value",))
    emission = _node("EMISSION", Color=(1.0, 1.0, 1.0, 1.0), Strength=2.0)
    _link(emission, "Strength", value, "Value")
    spec = proxy.material_emission(_material(emission, value))
    assert not spec.emits and "linked_emission_strength" in spec.unsupported


def test_planar_proxy_matches_cycles_power_and_faces_the_emitting_side():
    # A 2 x 3 m quad in the XY plane, normal +Z.
    points = [(0, 0, 0), (2, 0, 0), (2, 3, 0), (0, 3, 0)]
    geometry = proxy.emitter_geometry(points, [4], [0, 1, 2, 3], None)
    assert geometry.area == pytest.approx(6.0) and geometry.normal == pytest.approx((0, 0, 1))
    spec = proxy.EmissionSpec(front=(5.0, 5.0, 5.0), back=(0.0, 0.0, 0.0))
    (front,) = proxy.plan_proxies(geometry, spec)
    assert front.kind == "RectLight" and front.side == "front"
    assert sorted((front.width, front.height)) == pytest.approx([2.0, 3.0])
    # Light emits along its local -Z, so its Z axis opposes the emitting normal.
    assert front.axes[2] == pytest.approx((0, 0, -1))
    # Blender power pi*L*A through the area policy: 120*pi*P/A = 120*pi^2*L.
    assert front.intensity == pytest.approx(lights.MEASURED_LIGHT_SCALE * 5.0)
    both = proxy.plan_proxies(geometry, proxy.EmissionSpec(front=(5.0, 5.0, 5.0), back=(5.0, 5.0, 5.0)))
    assert {shape.side for shape in both} == {"front", "back"}
    back_only, = proxy.plan_proxies(geometry, proxy.EmissionSpec(back=(2.0, 2.0, 2.0)))
    assert back_only.side == "back" and back_only.axes[2] == pytest.approx((0, 0, 1))


def test_non_rectangular_emitter_scales_intensity_by_fill_ratio():
    # Right triangle covering half of its 2 x 2 bounding rectangle.
    geometry = proxy.emitter_geometry([(0, 0, 0), (2, 0, 0), (0, 2, 0)], [3], [0, 1, 2], None)
    shape, = proxy.plan_proxies(geometry, proxy.EmissionSpec(front=(1.0, 1.0, 1.0)))
    assert shape.emit_area == pytest.approx(2.0) and shape.shape_area == pytest.approx(4.0)
    assert shape.intensity == pytest.approx(lights.MEASURED_LIGHT_SCALE * 0.5)


def test_curved_emitter_becomes_an_equal_area_sphere():
    # Closed octahedron: area-weighted normals cancel.
    v = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    faces = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4), (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
    indices = [i for f in faces for i in f]
    geometry = proxy.emitter_geometry(v, [3] * len(faces), indices, None)
    shape, = proxy.plan_proxies(geometry, proxy.EmissionSpec(front=(2.0, 2.0, 2.0), back=(2.0, 2.0, 2.0)))
    assert shape.kind == "SphereLight"
    assert 4.0 * math.pi * shape.radius ** 2 == pytest.approx(geometry.area)
    assert shape.intensity == pytest.approx(lights.MEASURED_LIGHT_SCALE * 2.0)


def _stage_with_quad(*, extra_face=False):
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    material = UsdShade.Material.Define(stage, "/World/_materials/Glow")
    material.GetPrim().CreateAttribute(proxy.MATERIAL_NAME_ATTRIBUTE, Sdf.ValueTypeNames.String).Set("Glow")
    plain = UsdShade.Material.Define(stage, "/World/_materials/Plain")
    plain.GetPrim().CreateAttribute(proxy.MATERIAL_NAME_ATTRIBUTE, Sdf.ValueTypeNames.String).Set("Plain")
    owner = UsdGeom.Xform.Define(stage, "/World/Panel")
    owner.AddTranslateOp().Set(Gf.Vec3d(10, 0, 0))
    owner.AddScaleOp().Set(Gf.Vec3f(2, 2, 2))
    mesh = UsdGeom.Mesh.Define(stage, "/World/Panel/Quad")
    points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    counts, indices = [4], [0, 1, 2, 3]
    if extra_face:
        points += [(2, 0, 0), (3, 0, 0), (3, 1, 0)]
        counts.append(3)
        indices += [4, 5, 6]
    mesh.GetPointsAttr().Set(points)
    mesh.GetFaceVertexCountsAttr().Set(counts)
    mesh.GetFaceVertexIndicesAttr().Set(indices)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    if extra_face:
        subset = UsdGeom.Subset.CreateGeomSubset(mesh, "other", UsdGeom.Tokens.face, [1], familyName="materialBind")
        UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(plain)
    return stage


def _materials():
    glow = _material(_node("EMISSION", Color=(1.0, 1.0, 1.0, 1.0), Strength=5.0))
    plain = _material(_node("BSDF_DIFFUSE", Color=(0.8, 0.8, 0.8, 1.0)))
    return {"Glow": glow, "Plain": plain}.get


def test_records_author_a_scaled_rect_proxy_and_hide_mesh_from_secondary_rays_only():
    stage = _stage_with_quad()
    records = proxy.opinion_records(stage, _materials())
    by_path = {record.usd_prim_path: record for record in records}
    assert set(by_path) == {"/World/Panel/Quad", "/World/Panel/Quad/OvrtxEmissiveProxy_0_front",
                            "/World/Panel/Quad/OvrtxEmissiveProxy_0_back"}
    layer = Sdf.Layer.CreateAnonymous(".usda")
    layer.ImportFromString(by_path["/World/Panel/Quad/OvrtxEmissiveProxy_0_front"].layer_text)
    light = layer.GetPrimAtPath("/World/Panel/Quad/OvrtxEmissiveProxy_0_front")
    assert light.typeName == "RectLight"
    # World size 2 x 2 m (unit quad under a 2x scale); intensity is Cycles-equivalent.
    assert light.properties["inputs:width"].default == pytest.approx(2.0)
    assert light.properties["inputs:intensity"].default == pytest.approx(lights.MEASURED_LIGHT_SCALE * 5.0)
    assert "visibleInPrimaryRay" not in "".join(light.properties.keys())
    mesh_layer = Sdf.Layer.CreateAnonymous(".usda")
    mesh_layer.ImportFromString(by_path["/World/Panel/Quad"].layer_text)
    attribute = mesh_layer.GetAttributeAtPath("/World/Panel/Quad.primvars:invisibleToSecondaryRays")
    assert attribute.default is True
    assert not mesh_layer.GetAttributeAtPath("/World/Panel/Quad.primvars:hideForCamera")


def test_proxy_world_transform_places_the_light_on_the_emitting_plane():
    stage = _stage_with_quad()
    records = proxy.opinion_records(stage, _materials())
    overlay = Sdf.Layer.CreateAnonymous(".usda")
    overlay.subLayerPaths.append(stage.GetRootLayer().identifier)
    keep = []
    for record in records:
        extra = Sdf.Layer.CreateAnonymous(".usda")
        extra.ImportFromString(record.layer_text)
        keep.append(extra)
        overlay.subLayerPaths.insert(0, extra.identifier)
    composed = Usd.Stage.Open(overlay)
    light = composed.GetPrimAtPath("/World/Panel/Quad/OvrtxEmissiveProxy_0_front")
    world = UsdGeom.Xformable(light).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    assert world.ExtractTranslation() == pytest.approx(Gf.Vec3d(11.0, 1.0, 0.0), abs=1e-6)
    # The light emits along its local -Z: toward the quad normal (+Z).
    assert world.TransformDir(Gf.Vec3d(0, 0, -1)) == pytest.approx(Gf.Vec3d(0, 0, 1), abs=1e-6)


def test_mixed_emitting_and_plain_faces_are_left_on_mesh_emission():
    stage = _stage_with_quad(extra_face=True)
    assert proxy.opinion_records(stage, _materials()) == ()


def test_non_emissive_and_unresolved_materials_author_nothing():
    stage = _stage_with_quad()
    assert proxy.opinion_records(stage, lambda name: _material(_node("BSDF_DIFFUSE", Color=(1, 1, 1, 1)))) == ()
    assert proxy.opinion_records(stage, lambda name: None) == ()

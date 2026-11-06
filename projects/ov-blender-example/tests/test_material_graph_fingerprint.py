# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import sys
from types import SimpleNamespace as N

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from ovrtx_blender_example.material_graph_fingerprint import material_graph_signature, world_graph_signature


def socket(name, value=None):
    return N(name=name, identifier=name, default_value=value, links=[])


def node(kind, inputs=(), outputs=()):
    return N(type=kind, name=kind, inputs=list(inputs), outputs=list(outputs), mute=False)


def connect(upstream, downstream):
    downstream.links.append(N(from_node=upstream, from_socket=upstream.outputs[0]))


def graph():
    surface = node("BSDF_PRINCIPLED", [socket("Base Color", (0.2, 0.3, 0.4, 1)), socket("Roughness", 0.3)])
    surface.outputs = [socket("BSDF")]
    output = node("OUTPUT_MATERIAL", [socket("Surface")])
    output.is_active_output = True
    connect(surface, output.inputs[0])
    material = N(use_nodes=True, node_tree=N(nodes=[surface, output]))
    return material, surface


def test_native_constants_and_graph_editor_changes_do_not_regenerate():
    material, surface = graph()
    initial = material_graph_signature(material)
    surface.inputs[0].default_value = (1, 0, 0, 1)
    surface.inputs[1].default_value = 0.8
    surface.name = "Artist renamed shader"
    surface.location = (10, 20)
    surface.bl_rna = N(properties=[N(identifier="location_absolute", type="FLOAT")])
    surface.location_absolute = (10, 20)
    material.node_tree.nodes.append(node("VALUE", outputs=[socket("Value", 123)]))
    material.node_tree.nodes.reverse()
    assert material_graph_signature(material) == initial


def test_linked_value_and_rewiring_regenerate_even_when_socket_stays_connected():
    material, surface = graph()
    value = node("VALUE", outputs=[socket("Value", 0.4)])
    connect(value, surface.inputs[1])
    initial = material_graph_signature(material)
    value.outputs[0].default_value = 0.6
    assert material_graph_signature(material) != initial
    value.outputs[0].default_value = 0.4
    replacement = node("MATH", [socket("Value", 0.4)], [socket("Value")])
    replacement.operation = "MULTIPLY"
    surface.inputs[1].links[0].from_node = replacement
    assert material_graph_signature(material) != initial


def test_image_identity_colorspace_mapping_and_ramp_changes_regenerate():
    material, surface = graph()
    texture = node("TEX_IMAGE", [socket("Vector", (0, 0, 0))], [socket("Color")])
    texture.image = N(name="A", filepath="a.png", source="FILE", colorspace_settings=N(name="sRGB"))
    connect(texture, surface.inputs[0])
    initial = material_graph_signature(material)
    texture.image.filepath = "b.png"
    assert material_graph_signature(material) != initial
    texture.image.filepath = "a.png"
    texture.image.colorspace_settings.name = "Non-Color"
    assert material_graph_signature(material) != initial
    texture.image.colorspace_settings.name = "sRGB"
    texture.inputs[0].default_value = (0.5, 0, 0)
    assert material_graph_signature(material) != initial
    ramp = node("VALTORGB", [socket("Fac", 0.2)], [socket("Color")])
    ramp.color_ramp = N(elements=[N(position=0.3, color=(1, 0, 0, 1))], interpolation="LINEAR")
    surface.inputs[0].links[0].from_node = ramp
    initial = material_graph_signature(material)
    ramp.color_ramp.elements[0].position = 0.6
    assert material_graph_signature(material) != initial


def test_nested_groups_and_cycles_are_bounded():
    material, surface = graph()
    value = node("VALUE", outputs=[socket("Value", 0.3)])
    output = node("GROUP_OUTPUT", [socket("Value")])
    connect(value, output.inputs[0])
    group = node("GROUP", outputs=[socket("Value")])
    group.node_tree = N(nodes=[value, output])
    connect(group, surface.inputs[1])
    initial = material_graph_signature(material)
    value.outputs[0].default_value = 0.7
    assert material_graph_signature(material) != initial
    connect(group, output.inputs[0])
    assert len(material_graph_signature(material)) == 64


def test_world_background_values_are_part_of_graph_identity():
    background = node("BACKGROUND", [socket("Strength", 1.0)], [socket("Background")])
    output = node("OUTPUT_WORLD", [socket("Surface")])
    connect(background, output.inputs[0])
    world = N(use_nodes=True, node_tree=N(nodes=[background, output]))
    initial = world_graph_signature(world)
    background.inputs[0].default_value = 2.0
    assert world_graph_signature(world) != initial


def test_accepted_generation_detects_upstream_edit_without_native_constant_false_positive():
    from ovrtx_blender_example.scene_generation import BlenderId, BlenderPrimPath
    from ovrtx_blender_example.scene_generation_sessions import _material_graph_change
    material, surface = graph()
    material.session_uid = 37
    identity = BlenderId("MATERIAL", 37)
    generation = N(material_graph_fingerprints={identity: material_graph_signature(material)},
                   blender_prim_paths={identity: BlenderPrimPath("Mat", "MATERIAL", "/World/Mat", "/World/Mat")})
    surface.inputs[1].default_value = 0.9
    assert _material_graph_change(generation, material) is None
    value = node("VALUE", outputs=[socket("Value", 0.4)])
    connect(value, surface.inputs[1])
    change = _material_graph_change(generation, material)
    assert change["usd_prim_path"] == "/World/Mat"
    assert change["previous_fingerprint"] != change["current_fingerprint"]
    generation.material_graph_fingerprints[identity] = material_graph_signature(material)
    assert _material_graph_change(generation, material) is None


def test_composite_principled_constants_require_authored_generation():
    material, surface = graph()
    mix = node('MIX_SHADER', [socket('Shader')], [socket('Shader')])
    connect(surface, mix.inputs[0])
    material.node_tree.nodes[-1].inputs[0].links[0].from_node = mix
    initial = material_graph_signature(material)
    surface.inputs[1].default_value = .8
    assert material_graph_signature(material) != initial


class RewrappedLink:
    """Represent Blender returning a fresh Python wrapper for one RNA node."""
    def __init__(self, underlying):
        self.underlying = underlying
        self.from_socket = underlying.outputs[0]

    @property
    def from_node(self):
        return N(**vars(self.underlying))


def test_fresh_rna_wrappers_preserve_native_constants_but_detect_graph_edits():
    material, surface = graph()
    surface.as_pointer = lambda: 12345
    material.node_tree.nodes[-1].inputs[0].links = [RewrappedLink(surface)]
    assert material.node_tree.nodes[-1].inputs[0].links[0].from_node is not surface
    initial = material_graph_signature(material)
    surface.inputs[0].default_value = (1, 0, 0, 1)
    surface.inputs[1].default_value = 0.95
    assert material_graph_signature(material) == initial
    value = node('VALUE', outputs=[socket('Value', 0.4)])
    value.as_pointer = lambda: 56789
    surface.inputs[1].links = [RewrappedLink(value)]
    linked = material_graph_signature(material)
    assert linked != initial
    value.outputs[0].default_value = 0.9
    assert material_graph_signature(material) != linked


def test_rna_cycle_with_fresh_wrappers_is_bounded():
    material, surface = graph()
    surface.as_pointer = lambda: 12345
    surface.inputs[1].links = [RewrappedLink(surface)]
    material.node_tree.nodes[-1].inputs[0].links = [RewrappedLink(surface)]
    assert len(material_graph_signature(material)) == 64


def test_rgb_curve_point_edits_regenerate():
    """RGB Curves are translated, so moving a curve point changes the generated graph."""
    material, surface = graph()
    point = N(location=(0.5, 0.5), handle_type="AUTO")
    curve_node = node("CURVE_RGB", [socket("Fac", 1.0), socket("Color")], [socket("Color")])
    curve_node.mapping = N(curves=[N(points=[point])], black_level=(0, 0, 0), white_level=(1, 1, 1), extend="HORIZONTAL")
    connect(curve_node, surface.inputs[0])
    initial = material_graph_signature(material)
    point.location = (0.5, 0.7)
    assert material_graph_signature(material) != initial

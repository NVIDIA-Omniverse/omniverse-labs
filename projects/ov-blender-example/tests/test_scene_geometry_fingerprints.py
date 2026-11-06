"""Dirty geometry must not be mistaken for export callback noise."""
from types import SimpleNamespace as NS

import pytest

from addon.ovrtx_blender_example import scene_generation as generation


def mesh_scene():
    mesh = NS(session_uid=31, name_full="Mesh", vertices=[NS(co=(0, 0, 0))],
              polygons=[NS(vertices=(0,), material_index=0, use_smooth=False)],
              materials=[], corner_normals=[NS(vector=(0, 0, 1))], shape_keys=None)
    obj = NS(session_uid=11, name_full="Object", type="MESH", data=mesh,
             parent=None, hide_render=False, modifiers=[], material_slots=[],
             matrix_world=(1, 0, 0, 1))
    return NS(objects=[obj]), obj, mesh


def fingerprint(scene):
    return generation._topology_fingerprints(scene)[generation.BlenderId("OBJECT", 11)]


@pytest.mark.parametrize("change", [
    lambda obj, mesh: setattr(mesh.polygons[0], "material_index", 1),
    lambda obj, mesh: setattr(mesh.polygons[0], "use_smooth", True),
    lambda obj, mesh: setattr(mesh.corner_normals[0], "vector", (0, 1, 0)),
    lambda obj, mesh: obj.material_slots.append(NS(link="OBJECT", material=NS(session_uid=41))),
])
def test_mesh_render_input_changes_require_generation(change):
    scene, obj, mesh = mesh_scene()
    before = fingerprint(scene)
    change(obj, mesh)
    assert fingerprint(scene) != before


def test_transform_and_selection_noise_preserves_geometry_signature():
    scene, obj, _ = mesh_scene()
    before = fingerprint(scene)
    obj.matrix_world = (1, 2, 3, 1)
    obj.select = True
    assert fingerprint(scene) == before


class Modifier(dict):
    def __init__(self):
        super().__init__()
        self.type = "SUBSURF"
        self.levels = 1
        self.show_viewport = self.show_render = True
        self.show_expanded = False
        self.bl_rna = NS(properties=[
            NS(identifier="levels", type="INT", is_readonly=False),
            NS(identifier="show_expanded", type="BOOLEAN", is_readonly=False),
        ])


def test_modifier_settings_visibility_and_order_invalidate_but_ui_does_not():
    scene, obj, _ = mesh_scene()
    modifier = Modifier()
    obj.modifiers.append(modifier)
    before = fingerprint(scene)
    modifier.show_expanded = True
    assert fingerprint(scene) == before
    modifier.levels = 2
    assert fingerprint(scene) != before
    before = fingerprint(scene)
    modifier.show_viewport = False
    assert fingerprint(scene) != before
    obj.modifiers.append(Modifier())
    before = fingerprint(scene)
    obj.modifiers.reverse()
    assert fingerprint(scene) != before


def test_geometry_nodes_socket_values_and_group_defaults_are_render_inputs():
    scene, obj, _ = mesh_scene()
    modifier = Modifier()
    modifier.type = "NODES"
    modifier["Socket_2"] = 1.0
    socket = NS(name="Size", identifier="Size", default_value=(1., 1., 1.))
    modifier.node_group = NS(session_uid=70, nodes=[
        NS(name="Cube", bl_idname="GeometryNodeMeshCube", inputs=[socket])
    ], links=[])
    obj.modifiers.append(modifier)
    before = fingerprint(scene)
    modifier["Socket_2"] = 2.0
    assert fingerprint(scene) != before
    before = fingerprint(scene)
    socket.default_value = (2., 1., 1.)
    assert fingerprint(scene) != before


def test_shape_key_weight_mute_and_authored_positions_invalidate():
    scene, _, mesh = mesh_scene()
    key = NS(name="Smile", value=0., mute=False, data=[NS(co=(0., 0., 0.))])
    mesh.shape_keys = NS(use_relative=True, eval_time=0., key_blocks=[key])
    before = fingerprint(scene)
    key.value = 0.5
    assert fingerprint(scene) != before
    before = fingerprint(scene)
    key.mute = True
    assert fingerprint(scene) != before
    before = fingerprint(scene)
    key.data[0].co = (1., 0., 0.)
    assert fingerprint(scene) != before


def test_filtered_dirty_ids_do_not_inspect_unrelated_geometry():
    scene, _, _ = mesh_scene()
    other = NS(session_uid=12, type="MESH")  # Reading other.data would fail.
    scene.objects.append(other)
    assert generation._topology_fingerprints(scene, {generation.BlenderId("OBJECT", 11)})


def test_visibility_does_not_change_retained_geometry_signature():
    scene,obj,_ = mesh_scene()
    before = fingerprint(scene)
    obj.hide_render = True
    assert fingerprint(scene) == before
    assert generation._object_visibility(scene)[generation.BlenderId("OBJECT",11)] is False

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pytest

import nanousd

FIXTURE = Path(__file__).with_name("fixtures") / "root.usda"


def test_path_listop_and_math_helpers():
    root = nanousd.Path("/")
    child = root.append_child("World").append_child("Cube")
    prop = child.append_property("points")

    assert str(child) == "/World/Cube"
    assert prop.is_property
    assert prop.parent() == child
    assert prop.name == "points"

    explicit = nanousd.ListOp.create_explicit(["/A", "/B"])
    assert explicit.is_explicit
    assert explicit.items == ["/A", "/B"]

    op = nanousd.ListOp.create(prepended=["/Strong"], appended=["/Weak"], deleted=["/Gone"])
    assert op.prepended_items == ["/Strong"]
    assert op.appended_items == ["/Weak"]
    assert op.deleted_items == ["/Gone"]

    assert nanousd.dot3d([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) == pytest.approx(32.0)
    assert nanousd.cross3d([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]) == [0.0, 0.0, 1.0]
    assert nanousd.transform_point3d(
        [1.0, 0.0, 0.0, 10.0, 0.0, 1.0, 0.0, 20.0, 0.0, 0.0, 1.0, 30.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 2.0, 3.0],
    ) == pytest.approx([11.0, 22.0, 33.0])


def test_native_sdf_path_validation_helper():
    from nanousd import _nanousd

    assert _nanousd.sdf_path_is_valid("/Robot/Body_1")
    assert _nanousd.sdf_path_validated_text("/Robot/Body_1") == "/Robot/Body_1"
    assert _nanousd.sdf_path_validated_text("/Robot/Joint.rel[/Robot/Body]") == (
        "/Robot/Joint.rel[/Robot/Body]"
    )

    for bad_path in ("/Robot//Body", "/Robot/1Body", "/Robot:Bad", "<bad>"):
        assert not _nanousd.sdf_path_is_valid(bad_path)
        assert _nanousd.sdf_path_validated_text(bad_path) == ""


def test_native_attribute_namespace_filtering():
    stage = nanousd.Stage.create()
    prim = stage.define_prim("/World", "Xform")
    assert prim.create_attribute("newton:kp", "float")
    assert prim.create_attribute("newton:kd", "float")
    assert prim.create_attribute("newton:vec2", "float2")
    assert prim.create_attribute("physics:mass", "float")

    assert set(prim.attribute_names_in_namespace("newton")) == {"newton:kp", "newton:kd", "newton:vec2"}
    assert set(prim.attribute_names_in_namespace("newton:")) == {"newton:kp", "newton:kd", "newton:vec2"}
    assert prim.attribute_names_in_namespace("physics") == ["physics:mass"]
    assert not prim.is_attribute_authored("newton:kp")
    assert prim.authored_attribute_names_in_namespace("newton") == []
    assert prim.authored_attribute_values_in_namespace("newton") == {}

    assert prim.set_float("newton:kp", 10.0)
    assert prim.set_vec2f("newton:vec2", [1.0, 2.0])
    assert prim.is_attribute_authored("newton:kp")
    assert set(prim.authored_attribute_names_in_namespace("newton")) == {"newton:kp", "newton:vec2"}
    assert prim.authored_attribute_values_in_namespace("newton") == {
        "newton:kp": 10.0,
        "newton:vec2": [1.0, 2.0],
    }
    assert prim.authored_attribute_values_in_namespaces(["newton", "physics"]) == {
        "newton:kp": 10.0,
        "newton:vec2": [1.0, 2.0],
    }
    assert prim.authored_attribute_values(["newton:kp", "newton:vec2", "newton:missing"]) == {
        "newton:kp": 10.0,
        "newton:vec2": [1.0, 2.0],
    }

    assert prim.set_sample_float("newton:kd", 1.0, 2.0)
    assert prim.is_attribute_authored("newton:kd")
    assert set(prim.authored_attribute_names_in_namespace("newton")) == {
        "newton:kp",
        "newton:kd",
        "newton:vec2",
    }


def test_native_numeric_array_numpy_helpers():
    stage = nanousd.Stage.create()
    prim = stage.define_prim("/World/Mesh", "Mesh")
    assert prim.create_attribute("points", "point3f[]")
    assert prim.set_vec3f_array("points", [0, 0, 0, 1, 0, 0])
    assert prim.create_attribute("faceVertexIndices", "int[]")
    assert prim.set_int_array("faceVertexIndices", [0, 1, 0])

    points = prim.read_float_array_numpy("points", 3)
    indices = prim.read_int_array_numpy("faceVertexIndices")

    assert isinstance(points, np.ndarray)
    assert points.dtype == np.float32
    assert points.shape == (2, 3)
    np.testing.assert_allclose(points, [[0, 0, 0], [1, 0, 0]])
    assert isinstance(indices, np.ndarray)
    assert indices.dtype == np.int32
    np.testing.assert_array_equal(indices, [0, 1, 0])


def test_native_fan_triangulate_indices_helper():
    from nanousd import _nanousd

    counts = np.array([3, 4, 2], dtype=np.int32)
    indices = np.array([0, 1, 2, 3, 4, 5, 6, 8, 9], dtype=np.int32)

    tris = _nanousd.fan_triangulate_indices(counts, indices)
    assert tris.dtype == np.int32
    np.testing.assert_array_equal(
        tris,
        np.array(
            [
                [0, 1, 2],
                [3, 4, 5],
                [3, 5, 6],
            ],
            dtype=np.int32,
        ),
    )

    flipped = _nanousd.fan_triangulate_indices(counts, indices, flip_winding=True)
    np.testing.assert_array_equal(flipped, tris[:, ::-1])


def test_native_layer_inspection_on_fixture():
    stage = nanousd.Stage.open(str(FIXTURE))

    assert stage.num_layers == 2
    assert any("sublayer.usda" in p for p in stage.layer_sublayer_paths(0))
    assert stage.layer_has_prim_spec(0, "/Prim")
    assert stage.layer_has_prim_spec(1, "/Prim")
    assert stage.layer_has_attr_opinion(0, "/Prim", "rootAttr")
    assert stage.layer_has_attr_opinion(1, "/Prim", "subAttr")
    assert stage.layer_offset(0) == pytest.approx([0.0, 1.0])
    assert stage.diagnostics_json().startswith("[")


def test_native_authoring_export_and_reopen(tmp_path):
    stage = nanousd.Stage.create()
    assert stage.set_metadata_double("metersPerUnit", 0.01)
    assert stage.set_metadata_token("upAxis", "Z")
    assert stage.metadata_string("upAxis") == "Z"

    cube = stage.define_prim("/World/Cube", "Xform")
    assert cube.create_attribute("size", "double")
    assert cube.set_double("size", 2.5)
    assert cube.read_double("size") == pytest.approx(2.5)

    assert cube.create_attribute("label", "token")
    assert cube.set_token("label", "crate")
    assert cube.read_token("label") == "crate"
    assert cube.get("label") == "crate"

    assert cube.create_attribute("translate", "double3")
    assert cube.set_vec3d("translate", [1.0, 2.0, 3.0])
    assert cube.read_vec3d("translate") == pytest.approx([1.0, 2.0, 3.0])

    assert cube.create_attribute("weights", "float[]")
    assert cube.set_float_array("weights", [0.25, 0.5, 1.0])
    assert cube.read_float_array("weights") == pytest.approx([0.25, 0.5, 1.0])

    assert cube.set_sample_double("size", 1.0, 4.0)
    assert cube.has_samples("size")
    assert cube.sample_keys("size") == pytest.approx([1.0])
    assert cube.read_sample_double("size", 1.0) == pytest.approx(4.0)

    assert cube.create_relationship("material:binding")
    assert cube.add_relationship_target("material:binding", "/World/Looks/Mat")
    assert cube.relationship_names() == ["material:binding"]
    assert cube.relationship_targets("material:binding") == ["/World/Looks/Mat"]

    assert cube.set_metadata_token("kind", "component")
    assert cube.metadata_string("kind") == "component"

    usda_text = stage.to_usda_string()
    assert "Cube" in usda_text
    assert "size" in usda_text

    out = tmp_path / "authored.usda"
    assert stage.write_usda(str(out))
    reopened = nanousd.Stage.open(str(out))
    assert reopened.metadata_string("upAxis") == "Z"
    reopened_cube = reopened.get_prim_at_path("/World/Cube")
    assert reopened_cube is not None
    assert reopened_cube.read_double("size") == pytest.approx(2.5)
    assert reopened_cube.read_token("label") == "crate"


def test_native_variants_and_composition_arcs_are_exposed():
    stage = nanousd.Stage.create()
    model = stage.define_prim("/Model", "Xform")
    instance = stage.define_prim("/Instance", "Xform")

    assert model.create_variant_set("shape")
    assert model.create_variant("shape", "box")
    assert model.has_variant_set("shape")
    assert "shape" in model.variant_set_names()
    assert "box" in model.variant_names("shape")
    assert model.set_variant_selection("shape", "box")

    assert instance.add_reference("", "/Model")
    refs = instance.listop("references")
    assert refs is not None
    assert "/Model" in refs.prepended_items or any("/Model" in item for item in refs.prepended_items)


def test_invalid_vector_sizes_raise():
    stage = nanousd.Stage.create()
    prim = stage.define_prim("/Prim", "Xform")
    prim.create_attribute("v", "double3")

    with pytest.raises(RuntimeError):
        prim.set_vec3d("v", [1.0, 2.0])

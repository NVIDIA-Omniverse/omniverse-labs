# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys
import types

import numpy as np

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

FIXTURE = Path(__file__).with_name("fixtures") / "root.usda"


def test_pxr_find_spec_works_after_import():
    import importlib.util

    pxr_spec = importlib.util.find_spec("pxr")
    usd_spec = importlib.util.find_spec("pxr.Usd")

    assert pxr_spec is not None
    assert pxr_spec.name == "pxr"
    assert pxr_spec.submodule_search_locations is not None
    assert usd_spec is not None
    assert usd_spec.name == "pxr.Usd"


def test_pxr_stage_layer_and_attr_workflow(tmp_path):
    stage = Usd.Stage.CreateInMemory()
    stage.SetTimeCodesPerSecond(60)
    stage.SetFramesPerSecond(60)

    prim = stage.DefinePrim("/World/Cube", "Xform")
    assert prim
    assert prim.GetPath() == Sdf.Path("/World/Cube")
    assert prim.GetStage() is stage

    attr = prim.CreateAttribute("size", "double")
    assert attr.Set(3.0)
    assert attr.Set(5.0, 12.0)
    assert attr.Get() == 3.0
    assert attr.Get(12.0) == 5.0
    assert attr.GetTimeSamples() == [12.0]

    rel = prim.CreateRelationship("material:binding")
    assert rel.AddTarget(Sdf.Path("/World/Looks/Mat"))
    assert rel.GetTargets() == [Sdf.Path("/World/Looks/Mat")]

    stage.SetDefaultPrim(prim)
    text = stage.GetRootLayer().ExportToString()
    assert "Cube" in text
    assert "size" in text

    out = tmp_path / "pxr_workflow.usda"
    stage.GetRootLayer().realPath = str(out)
    stage.GetRootLayer().Save()
    reopened = Usd.Stage.Open(str(out))
    assert reopened.GetRootLayer().anonymous is False
    flattened = tmp_path / "flattened.usda"
    assert reopened.Export(str(flattened))
    assert flattened.exists()
    session = tmp_path / "session.usda"
    assert reopened.GetSessionLayer().Export(str(session))
    assert session.exists()
    assert reopened.GetPrimAtPath("/World/Cube").GetAttribute("size").Get() == 3.0


def test_pxr_fixture_and_geom_xformable():
    stage = Usd.Stage.Open(str(FIXTURE))
    prim = stage.GetPrimAtPath("/Prim")
    assert prim.IsValid()
    assert prim.GetName() == "Prim"
    assert prim.GetTypeName() == "Xform"
    assert prim.GetAttribute("rootAttr").Get() == "root"

    xformable = UsdGeom.Xformable(prim)
    assert xformable.GetPrim() == prim
    assert xformable.GetLocalTransformation() is not None


def test_sdf_value_type_name_is_public_type():
    value_type = Sdf.GetValueTypeNameForValue("material")

    assert isinstance(value_type, Sdf.ValueTypeName)
    assert value_type.scalarType == "string"
    assert value_type.isArray is False
    assert isinstance(Sdf.ValueTypeNames.String, Sdf.ValueTypeName)


def test_sdf_list_op_compat_accessors():
    token_op = Sdf.TokenListOp.Create(
        prependedItems=["PhysicsRigidBodyAPI"],
        appendedItems=["PhysicsMassAPI"],
    )

    assert token_op.GetAppliedItems() == [
        "PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    assert token_op.GetAddedOrExplicitItems() == token_op.GetAppliedItems()
    assert token_op.GetComposedItems() == token_op.GetAppliedItems()

    stage = Usd.Stage.CreateInMemory()
    prim = stage.DefinePrim("/Body", "Xform")
    prim.ApplyAPI("PhysicsRigidBodyAPI")
    schemas = prim.GetMetadata("apiSchemas")

    assert schemas.GetComposedItems() == ["PhysicsRigidBodyAPI"]
    assert schemas.GetAppliedItems() == schemas.GetComposedItems()
    assert schemas.GetAddedOrExplicitItems() == schemas.GetComposedItems()


def test_bbox_cache_extents_hint_accessors():
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"], True)

    assert cache.GetTime() == Usd.TimeCode.Default()
    assert cache.GetIncludedPurposes() == ["default"]
    assert cache.GetUseExtentsHint() is True

    cache.SetIncludedPurposes(["render"])
    cache.SetUseExtentsHint(False)

    assert cache.GetIncludedPurposes() == ["render"]
    assert cache.GetUseExtentsHint() is False


def test_pxr_compat_exports_all_public_names():
    import nanousd.pxr_compat as pxr_compat
    from pxr import Glf, Ndr, Plug, Sdr, UsdImagingGL

    for name in pxr_compat.__all__:
        assert hasattr(pxr_compat, name), name

    assert pxr_compat.Sdr is Sdr
    assert pxr_compat.Ndr is Ndr
    assert pxr_compat.Plug is Plug
    assert pxr_compat.Glf is Glf
    assert pxr_compat.UsdImagingGL is UsdImagingGL


def test_refinement_complexities_accept_usdview_menu_names():
    from pxr.UsdAppUtils.complexityArgs import RefinementComplexities

    assert RefinementComplexities.LOW.name == "Low"
    assert RefinementComplexities.MEDIUM.name == "Medium"
    assert RefinementComplexities.HIGH.name == "High"
    assert RefinementComplexities.VERY_HIGH.name == "Very High"
    assert RefinementComplexities.fromName("Low") is RefinementComplexities.LOW
    assert RefinementComplexities.fromName("low") is RefinementComplexities.LOW
    assert RefinementComplexities.fromName("Very High") is RefinementComplexities.VERY_HIGH
    assert RefinementComplexities.fromName("veryhigh") is RefinementComplexities.VERY_HIGH
    assert RefinementComplexities.fromName("very_high") is RefinementComplexities.VERY_HIGH
    assert RefinementComplexities.names() == ("Low", "Medium", "High", "Very High")


def test_pxr_compat_uses_nanobind_native_layer():
    import nanousd.pxr_compat as pxr_compat
    from nanousd.pxr_compat import _native_compat

    stage = pxr_compat.Usd.Stage.CreateInMemory()
    prim = stage.DefinePrim("/World", "Xform")

    assert not hasattr(pxr_compat, "_install_defensive_cdll")
    assert not hasattr(_native_compat, "_lib")
    assert prim._prim._h.__class__.__module__ == "nanousd._nanousd"
    assert hasattr(prim._prim._h, "local_transform_info_at")
    assert hasattr(prim._prim._h, "read_float_array_flat")


def test_sdf_path_fast_common_prim_path_validation_preserves_edge_cases():
    assert Sdf.Path("/Robot/Body_1").GetString() == "/Robot/Body_1"
    assert Sdf.Path("Robot/Body_1").GetString() == "Robot/Body_1"
    assert Sdf.Path("/Robot/Joint.rel[/Robot/Body]").targetPath == Sdf.Path("/Robot/Body")

    for bad_path in (
        "/Robot//Body",
        "/Robot/1Body",
        "/Robot:Bad",
        "/Robot/Body/",
        "<bad>",
        "/Bad$Name",
    ):
        assert Sdf.Path(bad_path).IsEmpty()


def test_sdf_path_common_prefix_uses_path_elements():
    assert Sdf.Path("/World/Floor").GetCommonPrefix(
        Sdf.Path("/World/Cube")) == Sdf.Path("/World")
    assert Sdf.Path("/World/Cube").GetCommonPrefix(
        Sdf.Path("/Other/Cube")) == Sdf.Path("/")
    assert Sdf.Path("/World/Cube.points").GetCommonPrefix(
        Sdf.Path("/World/Cube.normals")) == Sdf.Path("/World/Cube")
    assert Sdf.Path("World/Floor").GetCommonPrefix(
        Sdf.Path("World/Cube")) == Sdf.Path("World")


def test_prim_get_property_returns_attributes_and_relationships():
    stage = Usd.Stage.CreateInMemory()
    prim = stage.DefinePrim("/World/Cube", "Mesh")
    attr = prim.CreateAttribute("points", Sdf.ValueTypeNames.Point3fArray)
    rel = prim.CreateRelationship("material:binding")
    visibility = prim.CreateAttribute("visibility", Sdf.ValueTypeNames.Token)
    assert visibility.Set("invisible")

    assert prim.GetProperty("points").GetPath() == attr.GetPath()
    assert prim.GetProperty("material:binding").GetPath() == rel.GetPath()
    assert prim.GetProperty("points").GetPrimPath() == Sdf.Path("/World/Cube")
    assert prim.GetProperty("material:binding").GetPrimPath() == Sdf.Path("/World/Cube")
    assert attr.GetTypeName().isArray
    assert str(attr.GetTypeName()) == "point3f[]"
    assert attr.GetPropertyStackWithLayerOffsets(Usd.TimeCode.Default()) == []
    assert stage.GetPropertyAtPath("/World/Cube.points").GetPath() == attr.GetPath()
    assert stage.GetPropertyAtPath(
        "/World/Cube.material:binding").GetPath() == rel.GetPath()
    assert visibility.Clear()
    assert visibility.Get() != "invisible"
    assert UsdGeom.Tokens.visibility == "visibility"
    clip_range = Gf.Range1f(0.25, 100.0)
    assert clip_range.min == 0.25
    assert clip_range.max == 100.0


def test_pseudo_root_children_can_be_mutated_after_visibility_edits():
    stage = Usd.Stage.CreateInMemory()
    world = stage.DefinePrim("/World", "Xform")
    cube = stage.DefinePrim("/World/Cube", "Mesh")

    cube_img = UsdGeom.Imageable(cube)
    assert cube_img.MakeInvisible() is None
    assert cube_img.MakeVisible() is None
    assert cube_img.GetVisibilityAttr().Clear()

    root_children = stage.GetPseudoRoot().GetChildren()
    assert [str(child.GetPath()) for child in root_children] == ["/World"]
    for child in root_children:
        UsdGeom.Imageable(child).MakeInvisible()

    assert UsdGeom.Imageable(world).GetVisibilityAttr().Get() == "invisible"


def test_child_traversal_preserves_authored_namespace_order(tmp_path):
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/World", "Xform")
    stage.DefinePrim("/World/B", "Xform")
    stage.DefinePrim("/World/A", "Xform")
    stage.DefinePrim("/World/B", "Xform")

    assert [child.GetName() for child in stage.GetPrimAtPath("/World").GetChildren()] == [
        "B",
        "A",
    ]
    assert [str(prim.GetPath()) for prim in Usd.PrimRange(stage.GetPrimAtPath("/World"))] == [
        "/World",
        "/World/B",
        "/World/A",
    ]

    root = tmp_path / "root.usda"
    physics = tmp_path / "Physics.usda"
    root.write_text(
        """#usda 1.0
(
    subLayers = [
        @./Physics.usda@
    ]
)
""",
        encoding="utf-8",
    )
    physics.write_text(
        """#usda 1.0

def Xform "h1"
{
    def Scope "Physics"
    {
        def PhysicsJoint "left_hip_yaw" {}
        def PhysicsJoint "left_hip_roll" {}
        def PhysicsJoint "left_hip_pitch" {}
        def PhysicsJoint "left_knee" {}
        def PhysicsJoint "left_ankle" {}
    }
}
""",
        encoding="utf-8",
    )

    reopened = Usd.Stage.Open(str(root))
    assert [
        child.GetName() for child in reopened.GetPrimAtPath("/h1/Physics").GetChildren()
    ] == [
        "left_hip_yaw",
        "left_hip_roll",
        "left_hip_pitch",
        "left_knee",
        "left_ankle",
    ]


def test_articulation_desc_paths_match_openusd_sorted_order():
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/Robot").GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(root)
    body_b = UsdGeom.Xform.Define(stage, "/Robot/BodyB").GetPrim()
    body_a = UsdGeom.Xform.Define(stage, "/Robot/BodyA").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.RigidBodyAPI.Apply(body_b)
    UsdPhysics.RigidBodyAPI.Apply(body_a)

    joint_z = UsdPhysics.FixedJoint.Define(stage, "/Robot/Joints/zJoint")
    joint_z.CreateBody0Rel().SetTargets([root.GetPath()])
    joint_z.CreateBody1Rel().SetTargets([body_b.GetPath()])
    joint_a = UsdPhysics.FixedJoint.Define(stage, "/Robot/Joints/aJoint")
    joint_a.CreateBody0Rel().SetTargets([root.GetPath()])
    joint_a.CreateBody1Rel().SetTargets([body_a.GetPath()])

    physics = UsdPhysics.LoadUsdPhysicsFromRange(stage, ["/Robot"])
    _, descs = physics[UsdPhysics.ObjectType.Articulation]

    assert [str(path) for path in descs[0].articulatedBodies] == [
        "/Robot",
        "/Robot/BodyA",
        "/Robot/BodyB",
    ]
    assert [str(path) for path in descs[0].articulatedJoints] == [
        "/Robot/Joints/aJoint",
        "/Robot/Joints/zJoint",
    ]


def test_stage_up_axis_token_metadata_round_trips_in_memory_and_file(tmp_path):
    stage = Usd.Stage.CreateInMemory()

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    assert stage.GetUpAxis() == "Z"

    out = tmp_path / "z_up.usda"
    stage.GetRootLayer().realPath = str(out)
    stage.GetRootLayer().Save()
    assert out.exists()

    reopened = Usd.Stage.Open(str(out))
    assert UsdGeom.GetStageUpAxis(reopened) == UsdGeom.Tokens.z
    assert 'upAxis = "Z"' in reopened.GetRootLayer().ExportToString()


def test_xform_translate_op_matrix_uses_authored_xyz_order():
    stage = Usd.Stage.CreateInMemory()
    prim = stage.DefinePrim("/World/Body", "Xform")
    xform = UsdGeom.Xformable(prim)
    xform.AddTranslateOp().Set((1.0, 2.0, 3.0))

    matrix = xform.GetLocalTransformation()

    np.testing.assert_allclose(matrix[3, :3], [1.0, 2.0, 3.0])


def test_get_reset_xform_stack_reflects_authored_marker():
    stage = Usd.Stage.CreateInMemory()
    prim = stage.DefinePrim("/World/Body", "Xform")
    xform = UsdGeom.Xformable(prim)

    # Default: no xformOpOrder authored at all.
    assert xform.GetResetXformStack() is False

    # Author ops without the reset marker.
    translate = xform.AddTranslateOp()
    xform.SetXformOpOrder([translate], resetXformStack=False)
    assert xform.GetResetXformStack() is False

    # Author the reset marker via SetXformOpOrder.
    xform.SetXformOpOrder([translate], resetXformStack=True)
    assert xform.GetResetXformStack() is True


def test_material_binding_physics_purpose_round_trips_for_newton():
    stage = Usd.Stage.CreateInMemory()
    body = stage.DefinePrim("/World/Body", "Mesh")
    material = UsdShade.Material.Define(stage, "/World/Looks/PhysMat")

    binding = UsdShade.MaterialBindingAPI.Apply(body)
    assert binding.Bind(material, "physics")

    bound, rel = binding.ComputeBoundMaterial("physics")

    assert body.HasAPI(UsdShade.MaterialBindingAPI)
    assert rel is None
    assert bound is not None
    assert bound.GetPrim().GetPath() == material.GetPrim().GetPath()
    assert binding.GetDirectBindingRel("physics").GetTargets() == [
        Sdf.Path("/World/Looks/PhysMat")
    ]
    assert "material:binding:physics" in stage.GetRootLayer().ExportToString()


def test_instance_proxy_traversal_uses_instance_paths_and_reads_authored_arrays():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/Prototypes/TetProto")
    tetmesh = stage.DefinePrim("/Prototypes/TetProto/SoftBody", "TetMesh")
    tetmesh.CreateAttribute("points", Sdf.ValueTypeNames.Point3fArray).Set(
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    )
    tetmesh.CreateAttribute("tetVertexIndices", Sdf.ValueTypeNames.Int4Array).Set(
        [(0, 1, 2, 3)]
    )
    child = UsdGeom.Xform.Define(stage, "/Prototypes/TetProto/CustomChild").GetPrim()
    child.CreateAttribute("newton:test:child_value", Sdf.ValueTypeNames.Float).Set(99.0)

    instance = UsdGeom.Xform.Define(stage, "/World/Instance0").GetPrim()
    assert instance.GetReferences().AddInternalReference("/Prototypes/TetProto")
    assert instance.SetInstanceable(True)

    traversed = list(Usd.PrimRange(stage.GetPrimAtPath("/World"), Usd.TraverseInstanceProxies()))
    paths = [str(prim.GetPath()) for prim in traversed]

    assert "/World/Instance0/SoftBody" in paths
    assert "/__Prototype_0/SoftBody" not in paths

    proxy = stage.GetPrimAtPath("/World/Instance0/SoftBody")
    assert proxy.IsValid()
    assert proxy.IsInstanceProxy()
    assert proxy.GetPrimInPrototype().GetPath() == Sdf.Path("/__Prototype_0/SoftBody")
    np.testing.assert_allclose(
        proxy.GetAttribute("points").Get(),
        np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        proxy.GetAttribute("tetVertexIndices").Get(),
        np.array([(0, 1, 2, 3)], dtype=np.int32),
    )

    child_proxy = stage.GetPrimAtPath("/World/Instance0/CustomChild")
    child_attr = child_proxy.GetAttribute("newton:test:child_value")
    assert child_attr.HasAuthoredValue()
    assert child_attr.Get() == 99.0


def test_newton_actuator_api_definitions_filter_schema_properties():
    registry = Usd.SchemaRegistry()

    scene_def = registry.FindConcretePrimDefinition("PhysicsScene")
    pd_def = registry.FindAppliedAPIPrimDefinition("NewtonPDControlAPI")
    delay_def = registry.FindAppliedAPIPrimDefinition("NewtonActuatorDelayAPI")
    effort_def = registry.FindAppliedAPIPrimDefinition("NewtonMaxEffortClampingAPI")

    assert "newton:maxSolverIterations" in scene_def.GetPropertyNames()
    assert "newton:timeStepsPerSecond" in scene_def.GetPropertyNames()
    assert pd_def.GetPropertyNames() == ["newton:kp", "newton:kd"]
    assert delay_def.GetPropertyNames() == ["newton:delaySteps"]
    assert effort_def.GetPropertyNames() == ["newton:maxEffort"]
    assert "newton:delaySteps" not in pd_def.GetPropertyNames()


def test_authored_properties_in_namespace_uses_native_namespace_filter():
    stage = Usd.Stage.CreateInMemory()
    prim = stage.DefinePrim("/World/Controller", "Xform")
    prim.CreateAttribute("newton:kp", Sdf.ValueTypeNames.Float).Set(10.0)
    prim.CreateAttribute("newton:kd", Sdf.ValueTypeNames.Float).Set(1.0)
    prim.CreateAttribute("newton:unset", Sdf.ValueTypeNames.Float)
    prim.CreateAttribute("physics:mass", Sdf.ValueTypeNames.Float).Set(2.0)

    assert hasattr(prim._prim._h, "attribute_names_in_namespace")
    assert hasattr(prim._prim._h, "authored_attribute_values_in_namespace")
    props = {prop.GetName(): prop for prop in prim.GetAuthoredPropertiesInNamespace("newton")}

    assert set(props) == {"newton:kp", "newton:kd", "newton:unset"}
    assert props["newton:kp"].HasAuthoredValue()
    assert props["newton:kd"].HasAuthoredValue()
    assert not props["newton:unset"].HasAuthoredValue()
    assert prim.GetAuthoredAttributeValuesInNamespace("newton") == {
        "newton:kp": 10.0,
        "newton:kd": 1.0,
    }
    assert prim.GetAuthoredAttributeValues(["newton:kp", "newton:missing"]) == {
        "newton:kp": 10.0,
    }
    assert prim.GetAuthoredAttributeValuesInNamespaces(["newton", "physics"]) == {
        "newton:kp": 10.0,
        "newton:kd": 1.0,
        "physics:mass": 2.0,
    }


def test_newton_runtime_accelerators(monkeypatch):
    fake_utils = types.ModuleType("newton._src.usd.utils")
    fake_schema = types.ModuleType("newton._src.usd.schema_resolver")

    def slow_fan(_counts, _indices):
        raise AssertionError("native accelerator should replace this helper")

    def slow_namespace(_prim, _namespace):
        raise AssertionError("native namespace accelerator should replace this helper")

    def slow_collect_by_name(_prim, _names):
        raise AssertionError("native name-batch accelerator should replace this helper")

    def slow_collect_by_namespace(_prim, _namespaces):
        raise AssertionError("native namespace-batch accelerator should replace this helper")

    fake_utils.fan_triangulate_faces = slow_fan
    fake_utils.get_attributes_in_namespace = slow_namespace
    fake_schema._collect_attrs_by_name = slow_collect_by_name
    fake_schema._collect_attrs_by_namespace = slow_collect_by_namespace
    monkeypatch.setitem(sys.modules, "newton._src.usd.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "newton._src.usd.schema_resolver", fake_schema)

    stage = Usd.Stage.CreateInMemory()
    prim = stage.DefinePrim("/World/Controller", "Xform")
    prim.CreateAttribute("newton:kp", Sdf.ValueTypeNames.Float).Set(10.0)
    prim.CreateAttribute("physics:mass", Sdf.ValueTypeNames.Float).Set(2.0)
    UsdGeom.Mesh(stage.DefinePrim("/World/Mesh", "Mesh"))

    counts = np.array([3, 4, 2], dtype=np.int32)
    indices = np.array([0, 1, 2, 3, 4, 5, 6, 8, 9], dtype=np.int32)
    tris = fake_utils.fan_triangulate_faces(counts, indices)

    assert getattr(fake_utils.fan_triangulate_faces, "_nanousd_accelerated", False)
    assert getattr(fake_utils.get_attributes_in_namespace, "_nanousd_accelerated", False)
    assert getattr(fake_schema._collect_attrs_by_name, "_nanousd_accelerated", False)
    assert getattr(fake_schema._collect_attrs_by_namespace, "_nanousd_accelerated", False)
    np.testing.assert_array_equal(
        tris,
        np.array([[0, 1, 2], [3, 4, 5], [3, 5, 6]], dtype=np.int32),
    )
    assert fake_utils.get_attributes_in_namespace(prim, "newton") == {"newton:kp": 10.0}
    assert fake_schema._collect_attrs_by_name(prim, ["newton:kp", "newton:missing"]) == {
        "newton:kp": 10.0,
    }
    assert fake_schema._collect_attrs_by_namespace(prim, ["newton", "physics"]) == {
        "newton:kp": 10.0,
        "physics:mass": 2.0,
    }


def test_mass_api_schema_fallback_values_are_not_authored():
    stage = Usd.Stage.CreateInMemory()
    body = UsdGeom.Cube.Define(stage, "/World/Body").GetPrim()
    mass_api = UsdPhysics.MassAPI.Apply(body)

    assert mass_api.GetMassAttr().Get() == 0.0
    assert not mass_api.GetMassAttr().HasAuthoredValue()
    assert not mass_api.GetDiagonalInertiaAttr().HasAuthoredValue()
    assert not mass_api.GetCenterOfMassAttr().HasAuthoredValue()
    assert not mass_api.GetDensityAttr().HasAuthoredValue()


def test_mass_api_authored_values_from_sublayer_are_authored(tmp_path):
    root = tmp_path / "root.usda"
    physics = tmp_path / "Physics.usda"
    root.write_text(
        """#usda 1.0
(
    subLayers = [
        @./Physics.usda@
    ]
)
""",
        encoding="utf-8",
    )
    physics.write_text(
        """#usda 1.0

def Xform "World"
{
    def Xform "Body" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {
        float physics:mass = 3.5
        float3 physics:diagonalInertia = (1, 2, 3)
        point3f physics:centerOfMass = (0.1, 0.2, 0.3)
    }
}
""",
        encoding="utf-8",
    )

    stage = Usd.Stage.Open(str(root))
    mass_api = UsdPhysics.MassAPI(stage.GetPrimAtPath("/World/Body"))

    assert mass_api.GetMassAttr().HasAuthoredValue()
    assert mass_api.GetMassAttr().Get() == 3.5
    assert mass_api.GetDiagonalInertiaAttr().HasAuthoredValue()
    assert mass_api.GetCenterOfMassAttr().HasAuthoredValue()


def test_attribute_custom_data_by_key_reads_usda_metadata(tmp_path):
    asset = tmp_path / "custom_data.usda"
    asset.write_text(
        """#usda 1.0

def PhysicsScene "physicsScene"
{
    custom float newton:testBodyScalar = 0 (
        customData = {
            string assignment = "model"
            string frequency = "body"
        }
    )
}
""",
        encoding="utf-8",
    )
    stage = Usd.Stage.Open(str(asset))
    attr = stage.GetPrimAtPath("/physicsScene").GetAttribute("newton:testBodyScalar")

    assert attr.GetCustomDataByKey("assignment") == "model"
    assert attr.GetCustomDataByKey("frequency") == "body"
    assert attr.GetCustomDataByKey("missing") is None


def test_usdview_editor_metadata_and_stack_api_parity(tmp_path):
    root = tmp_path / "root.usda"
    sublayer = tmp_path / "sublayer.usda"
    ref = tmp_path / "ref.usda"
    payload = tmp_path / "payload.usda"

    root.write_text(
        """#usda 1.0
(
    defaultPrim = "World"
    subLayers = [
        @sublayer.usda@
    ]
)

def Xform "World" (
    active = true
    prepend apiSchemas = ["PhysicsCollisionAPI"]
    assetInfo = {
        asset identifier = @assetIdentifier.usd@
        string name = "WorldAsset"
    }
    customData = {
        string note = "root-world"
    }
    prepend inherits = </ClassPrim>
    prepend payload = @payload.usda@</PayloadRoot>
    prepend references = @ref.usda@</RefRoot>
    prepend specializes = </SpecializedPrim>
    variants = {
        string shape = "cube"
    }
    prepend variantSets = "shape"
)
{
    custom rel targetRel
    prepend rel targetRel = </ClassPrim>

    def Xform "Child"
    {
        custom float childAttr = 1.25
    }

    variantSet "shape" = {
        "cube" {
            def Cube "Geom"
            {
                double size = 2
            }
        }
    }
}
""",
        encoding="utf-8",
    )
    sublayer.write_text(
        """#usda 1.0

class "ClassPrim" (
    customData = {
        string classCustom = "class-layer"
    }
)
{
    custom int classAttr = 42
}

def Xform "SpecializedPrim"
{
    custom string specialAttr = "special"
}

over "World" (
    customData = {
        string weakOpinion = "from-sublayer"
    }
)
{
    custom string weakAttr = "weak"
}
""",
        encoding="utf-8",
    )
    ref.write_text(
        """#usda 1.0

def Xform "RefRoot" (
    customData = {
        string source = "reference-layer"
    }
)
{
    custom double refAttr = 7
}
""",
        encoding="utf-8",
    )
    payload.write_text(
        """#usda 1.0

def Xform "PayloadRoot"
{
    custom string payloadAttr = "payload-value"
}
""",
        encoding="utf-8",
    )

    stage = Usd.Stage.Open(str(root))
    world = stage.GetPrimAtPath("/World")
    class_prim = stage.GetPrimAtPath("/ClassPrim")
    child = stage.GetPrimAtPath("/World/Child")
    geom = stage.GetPrimAtPath("/World/Geom")

    assert class_prim.GetAllMetadata()["specifier"] == "Sdf.SpecifierClass"
    assert "active" not in child.GetAllMetadata()
    assert world.GetMetadata("customData")["weakOpinion"] == "from-sublayer"
    assert str(world.GetMetadata("assetInfo")["identifier"]) == "@assetIdentifier.usd@"

    rel_names = {rel.GetName() for rel in world.GetRelationships()}
    assert {"targetRel", "proxyPrim", "physics:simulationOwner"}.issubset(rel_names)
    assert world.GetRelationship("targetRel").GetPropertyStackWithLayerOffsets(Usd.TimeCode.Default())

    class_attr_stack = world.GetAttribute("classAttr").GetPropertyStackWithLayerOffsets(Usd.TimeCode.Default())
    ref_attr_stack = world.GetAttribute("refAttr").GetPropertyStackWithLayerOffsets(Usd.TimeCode.Default())
    payload_attr_stack = world.GetAttribute("payloadAttr").GetPropertyStackWithLayerOffsets(Usd.TimeCode.Default())
    assert str(class_attr_stack[0][0].path) == "/ClassPrim.classAttr"
    assert str(ref_attr_stack[0][0].path) == "/RefRoot.refAttr"
    assert str(payload_attr_stack[0][0].path) == "/PayloadRoot.payloadAttr"

    geom_stack = geom.GetPrimStackWithLayerOffsets()
    assert str(geom_stack[0][0].path) == "/World{shape=cube}Geom"
    assert geom.GetAttribute("extent").GetMetadata("default") == "[(-1, -1, -1), (1, 1, 1)]"


def test_primvar_interpolation_round_trips_for_points_attrs():
    stage = Usd.Stage.CreateInMemory()
    points = UsdGeom.Points.Define(stage, "/World/Points")

    UsdGeom.Primvar(points.GetDisplayColorAttr()).SetInterpolation(UsdGeom.Tokens.vertex)
    UsdGeom.Primvar(points.GetWidthsAttr()).SetInterpolation(UsdGeom.Tokens.constant)

    reopened_points = UsdGeom.Points.Get(stage, "/World/Points")
    assert (
        UsdGeom.Primvar(reopened_points.GetDisplayColorAttr()).GetInterpolation()
        == UsdGeom.Tokens.vertex
    )
    assert (
        UsdGeom.Primvar(reopened_points.GetWidthsAttr()).GetInterpolation()
        == UsdGeom.Tokens.constant
    )


def test_usd_physics_joint_local_pose_applies_body_scale():
    stage = Usd.Stage.CreateInMemory()
    cart = UsdGeom.Xform.Define(stage, "/World/Cart")
    cart_xform = UsdGeom.Xformable(cart.GetPrim())
    cart_xform.AddTranslateOp().Set((0.0, 0.0, 0.0))
    cart_xform.AddScaleOp().Set((0.2, 0.25, 0.2))
    UsdPhysics.RigidBodyAPI.Apply(cart.GetPrim())

    pole = UsdGeom.Xform.Define(stage, "/World/Pole")
    UsdPhysics.RigidBodyAPI.Apply(pole.GetPrim())

    joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/Joint")
    joint.CreateBody0Rel().SetTargets([cart.GetPrim().GetPath()])
    joint.CreateBody1Rel().SetTargets([pole.GetPrim().GetPath()])
    joint.CreateLocalPos0Attr().Set((0.55, 0.0, 0.0))
    joint.CreateLocalPos1Attr().Set((0.0, 0.0, 0.0))

    physics = UsdPhysics.LoadUsdPhysicsFromRange(stage, ["/World"])
    _, descs = physics[UsdPhysics.ObjectType.RevoluteJoint]

    np.testing.assert_allclose(descs[0].localPose0Position, [0.11, 0.0, 0.0])


def test_usd_physics_joint_local_pose_folds_child_target_transform():
    stage = Usd.Stage.CreateInMemory()
    body0 = UsdGeom.Xform.Define(stage, "/World/Body0")
    UsdPhysics.RigidBodyAPI.Apply(body0.GetPrim())

    body1 = UsdGeom.Xform.Define(stage, "/World/Body1")
    UsdPhysics.RigidBodyAPI.Apply(body1.GetPrim())
    site = UsdGeom.Xform.Define(stage, "/World/Body1/Site")
    UsdGeom.Xformable(site.GetPrim()).AddTranslateOp().Set((0.2, -0.1, 0.3))

    joint = UsdPhysics.FixedJoint.Define(stage, "/World/Weld")
    joint.CreateBody0Rel().SetTargets([body0.GetPrim().GetPath()])
    joint.CreateBody1Rel().SetTargets([site.GetPrim().GetPath()])
    joint.CreateLocalPos0Attr().Set((0.25, -0.2, 0.1))
    joint.CreateLocalPos1Attr().Set((0.1, 0.3, -0.2))

    physics = UsdPhysics.LoadUsdPhysicsFromRange(stage, ["/World"])
    _, descs = physics[UsdPhysics.ObjectType.FixedJoint]

    np.testing.assert_allclose(descs[0].localPose1Position, [0.3, 0.2, 0.1])


def test_usd_physics_joint_body_targets_resolve_to_rigid_body_ancestors():
    stage = Usd.Stage.CreateInMemory()
    world = UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xformable(world.GetPrim()).AddTranslateOp().Set((1.0, 2.0, 0.0))
    UsdGeom.Xformable(world.GetPrim()).AddOrientOp().Set(
        Gf.Quatf(0.70710677, 0.0, 0.0, 0.70710677)
    )
    rigid = UsdGeom.Xform.Define(stage, "/World/Rigid")
    UsdPhysics.RigidBodyAPI.Apply(rigid.GetPrim())
    site = UsdGeom.Xform.Define(stage, "/World/Rigid/Site")
    UsdGeom.Xformable(site.GetPrim()).AddTranslateOp().Set((0.2, 0.0, 0.0))
    UsdGeom.Xformable(site.GetPrim()).AddOrientOp().Set(
        Gf.Quatf(0.70710677, 0.0, 0.0, 0.70710677)
    )

    joint = UsdPhysics.FixedJoint.Define(stage, "/World/Weld")
    joint.CreateBody0Rel().SetTargets([world.GetPrim().GetPath()])
    joint.CreateBody1Rel().SetTargets([site.GetPrim().GetPath()])
    joint.CreateLocalPos0Attr().Set((1.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr().Set((0.3, 0.0, 0.0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    physics = UsdPhysics.LoadUsdPhysicsFromRange(stage, ["/World"])
    _, descs = physics[UsdPhysics.ObjectType.FixedJoint]
    pose0_rot = descs[0].localPose0Orientation
    pose1_rot = descs[0].localPose1Orientation

    assert descs[0].body0 == Sdf.Path("")
    assert descs[0].body1 == Sdf.Path("/World/Rigid")
    np.testing.assert_allclose(descs[0].localPose0Position, [1.0, 3.0, 0.0])
    np.testing.assert_allclose(descs[0].localPose1Position, [0.2, 0.3, 0.0])
    np.testing.assert_allclose(
        [pose0_rot.GetReal(), *pose0_rot.GetImaginary()],
        [0.70710677, 0.0, 0.0, 0.70710677],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        [pose1_rot.GetReal(), *pose1_rot.GetImaginary()],
        [0.70710677, 0.0, 0.0, 0.70710677],
        atol=1e-6,
    )


def test_xform_translate_op_accepts_gf_vec3f_for_double3_ops():
    stage = Usd.Stage.CreateInMemory()
    xform = UsdGeom.Xform.Define(stage, "/World/Site")

    xform.AddTranslateOp().Set(Gf.Vec3f(0.2, -0.1, 0.3))

    prim = xform.GetPrim()
    attr = prim.GetAttribute("xformOp:translate")
    assert attr.HasAuthoredValue()
    np.testing.assert_allclose(attr.Get(), [0.2, -0.1, 0.3])

    cache = UsdGeom.XformCache()
    np.testing.assert_allclose(
        cache.GetLocalToWorldTransform(prim)[3, :3],
        [0.2, -0.1, 0.3],
    )


def test_xformop_set_invalidates_hasauthoredvalue():
    """XformOp.Set must update the HasAuthoredValue cache (editor-write path)."""
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    xform = UsdGeom.Xform.Define(stage, "/X")
    prim = xform.GetPrim()
    op = xform.AddTranslateOp()
    # Prime the (path, name) attr cache with a not-yet-authored read.
    assert prim.GetAttribute("xformOp:translate").HasAuthoredValue() is False
    op.Set(Gf.Vec3d(1.0, 2.0, 3.0))
    # The direct XformOp.Set previously skipped cache invalidation -> stale False.
    assert prim.GetAttribute("xformOp:translate").HasAuthoredValue() is True

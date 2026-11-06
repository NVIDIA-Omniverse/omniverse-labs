# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from pxr import Sdf, Usd, UsdGeom
from ovrtx_blender_example.shadow_visibility import author_mesh_shadow_visibility


def test_mesh_boolean_on_off_and_nested_owners(tmp_path):
    path = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(path))
    for name, uid in (("Glass", 1), ("Frame", 2), ("Glass/Nested", 3)):
        owner = UsdGeom.Xform.Define(stage, "/World/" + name).GetPrim()
        owner.CreateAttribute("ovrtx:object_session_uid", Sdf.ValueTypeNames.Int64).Set(uid)
        UsdGeom.Mesh.Define(stage, owner.GetPath().AppendChild("Mesh"))
    UsdGeom.Mesh.Define(stage, "/World/Unmapped")
    stage.GetRootLayer().Save()
    objects = [SimpleNamespace(type="MESH", session_uid=i, visible_shadow=v)
               for i, v in ((1, False), (2, True), (3, True))]
    scene = SimpleNamespace(objects=objects)
    assert author_mesh_shadow_visibility(path, scene) == 3
    for name, value in (("Glass", True), ("Frame", False), ("Glass/Nested", False)):
        pv = UsdGeom.PrimvarsAPI(stage.GetPrimAtPath("/World/" + name + "/Mesh")).GetPrimvar("doNotCastShadows")
        assert pv.Get() is value
        assert pv.GetTypeName() == Sdf.ValueTypeNames.Bool
        assert pv.GetInterpolation() == "constant"
    assert not stage.GetPrimAtPath("/World/Unmapped").HasAttribute("primvars:doNotCastShadows")
    objects[0].visible_shadow = True
    author_mesh_shadow_visibility(path, scene)
    assert stage.GetPrimAtPath("/World/Glass/Mesh").GetAttribute("primvars:doNotCastShadows").Get() is False


def test_same_mesh_data_can_have_different_object_visibility(tmp_path):
    path = tmp_path / "linked.usda"
    stage = Usd.Stage.CreateNew(str(path))
    for i in (1, 2):
        mesh = UsdGeom.Mesh.Define(stage, f"/World/Mesh{i}").GetPrim()
        mesh.CreateAttribute("ovrtx:object_session_uid", Sdf.ValueTypeNames.Int64).Set(i)
        mesh.CreateAttribute("ovrtx:mesh_session_uid", Sdf.ValueTypeNames.Int64).Set(42)
    stage.GetRootLayer().Save()
    scene = SimpleNamespace(objects=[SimpleNamespace(type="MESH", session_uid=i, visible_shadow=i==2) for i in (1, 2)])
    author_mesh_shadow_visibility(path, scene)
    assert stage.GetPrimAtPath("/World/Mesh1").GetAttribute("primvars:doNotCastShadows").Get() is True
    assert stage.GetPrimAtPath("/World/Mesh2").GetAttribute("primvars:doNotCastShadows").Get() is False


def test_objects_inside_instanced_collections_are_matched_by_exported_name(tmp_path):
    path = tmp_path / "instanced.usda"
    stage = Usd.Stage.CreateNew(str(path))
    for name, object_name in (("Globe_0", "Globe"), ("Globe_1", "Globe"), ("Shared_0", "Twin"), ("Local_0", "Local")):
        owner = UsdGeom.Xform.Define(stage, "/World/" + name).GetPrim()
        owner.CreateAttribute("userProperties:blender:object_name", Sdf.ValueTypeNames.String).Set(object_name)
        UsdGeom.Mesh.Define(stage, owner.GetPath().AppendChild("Mesh"))
    local = UsdGeom.Xform.Define(stage, "/World/LocalOwner").GetPrim()
    local.CreateAttribute("ovrtx:object_session_uid", Sdf.ValueTypeNames.Int64).Set(7)
    local.CreateAttribute("userProperties:blender:object_name", Sdf.ValueTypeNames.String).Set("Local")
    UsdGeom.Mesh.Define(stage, "/World/LocalOwner/Mesh")
    stage.GetRootLayer().Save()
    globe = SimpleNamespace(type="MESH", name="Globe", session_uid=11, visible_shadow=False, visible_camera=False)
    twin_a = SimpleNamespace(type="MESH", name="Twin", session_uid=12, visible_shadow=False, visible_camera=True)
    twin_b = SimpleNamespace(type="MESH", name="Twin", session_uid=13, visible_shadow=True, visible_camera=True)
    collection = SimpleNamespace(objects=[globe, twin_a, twin_b], children=[])
    holder = SimpleNamespace(type="EMPTY", session_uid=1, instance_collection=collection)
    scene_mesh = SimpleNamespace(type="MESH", session_uid=7, visible_shadow=True, visible_camera=True)
    scene = SimpleNamespace(objects=[holder, scene_mesh])
    assert author_mesh_shadow_visibility(path, scene) == 3  # Globe_0, Globe_1 and the scene mesh
    for owner in ("Globe_0", "Globe_1"):
        prim = stage.GetPrimAtPath(f"/World/{owner}/Mesh")
        assert prim.GetAttribute("primvars:doNotCastShadows").Get() is True
        assert prim.GetAttribute("primvars:hideForCamera").Get() is True
    # A name shared by two instanced objects cannot be attributed, so it is left alone.
    assert not stage.GetPrimAtPath("/World/Shared_0/Mesh").HasAttribute("primvars:doNotCastShadows")
    local_prim = stage.GetPrimAtPath("/World/LocalOwner/Mesh")
    assert local_prim.GetAttribute("primvars:doNotCastShadows").Get() is False
    assert not local_prim.HasAttribute("primvars:hideForCamera")

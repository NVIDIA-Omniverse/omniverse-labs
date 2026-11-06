"""Visual changes may reuse physics only when the full physics closure is equal."""
from pathlib import Path
import sys

import pytest
pytest.importorskip("pxr")
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from ovrtx_blender_example.physics_generation_fingerprint import physics_generation_fingerprint
from ovrtx_blender_example.physics_generation_fingerprint import physics_mass_delta


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    parent = UsdGeom.Xform.Define(stage, "/World")
    parent.AddTranslateOp().Set((3, 0, 0))
    scene = UsdPhysics.Scene.Define(stage, "/World/Physics")
    scene.CreateGravityMagnitudeAttr(9.81)
    mesh = UsdGeom.Mesh.Define(stage, "/World/Body")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3]); mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    UsdPhysics.RigidBodyAPI.Apply(mesh.GetPrim())
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MassAPI.Apply(mesh.GetPrim()).CreateMassAttr(1)
    stage.GetRootLayer().Save()
    return path, stage


def test_visual_topology_is_irrelevant(source):
    path, stage = source
    before = physics_generation_fingerprint(str(path))
    UsdGeom.Cube.Define(stage, "/World/VisibleOnly").CreateSizeAttr(12)
    stage.GetRootLayer().Save()
    assert before and physics_generation_fingerprint(str(path)) == before
    stage.RemovePrim("/World/VisibleOnly"); stage.GetRootLayer().Save()
    assert physics_generation_fingerprint(str(path)) == before


@pytest.mark.parametrize("edit", ["mass", "points", "parent", "gravity", "units", "roster", "relationship"])
def test_every_physics_input_change_prevents_reuse(source, edit):
    path, stage = source
    before = physics_generation_fingerprint(str(path))
    body = stage.GetPrimAtPath("/World/Body")
    if edit == "mass":
        body.GetAttribute("physics:mass").Set(2)
    elif edit == "points":
        body.GetAttribute("points").Set([(0, 0, 0), (2, 0, 0), (0, 1, 0)])
    elif edit == "parent":
        stage.GetPrimAtPath("/World").GetAttribute("xformOp:translate").Set((4, 0, 0))
    elif edit == "gravity":
        stage.GetPrimAtPath("/World/Physics").GetAttribute("physics:gravityMagnitude").Set(0)
    elif edit == "units":
        UsdGeom.SetStageMetersPerUnit(stage, .01)
    elif edit == "roster":
        UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Cube.Define(stage, "/World/Added").GetPrim())
    else:
        body.CreateRelationship("physics:simulationOwner").SetTargets(["/AnotherScene"])
    stage.GetRootLayer().Save()
    assert before and physics_generation_fingerprint(str(path)) != before


def test_unmeasured_animated_physics_disables_reuse(source):
    path, stage = source
    attr = stage.GetPrimAtPath("/World/Body").GetAttribute("physics:mass")
    attr.Set(1, 0); attr.Set(2, 1)
    stage.GetRootLayer().Save()
    assert physics_generation_fingerprint(str(path)) is None


def test_untagged_child_geometry_under_body_is_included(source):
    path, stage = source
    child = UsdGeom.Cube.Define(stage, "/World/Body/Child")
    child.CreateSizeAttr(1)
    stage.GetRootLayer().Save()
    before = physics_generation_fingerprint(str(path))
    child.GetSizeAttr().Set(2); stage.GetRootLayer().Save()
    assert before and physics_generation_fingerprint(str(path)) != before


def test_transitive_relationship_dependencies_are_included(source):
    path, stage = source
    collection = stage.DefinePrim("/Collection")
    dependency = UsdGeom.Cube.Define(stage, "/Collection/Shape")
    dependency.CreateSizeAttr(1)
    stage.GetPrimAtPath("/World/Physics").CreateRelationship("physics:filteredGroups").SetTargets(["/Collection"])
    stage.GetRootLayer().Save()
    before = physics_generation_fingerprint(str(path))
    dependency.GetSizeAttr().Set(2); stage.GetRootLayer().Save()
    assert before and physics_generation_fingerprint(str(path)) != before


def test_failed_optional_inspection_disables_reuse(tmp_path):
    assert physics_generation_fingerprint(str(tmp_path / "missing.usda")) is None


@pytest.mark.parametrize("other_change", [None, "geometry", "gravity", "automatic_mass", "wrong_roster"])
def test_live_mass_delta_accepts_only_explicit_mass_on_existing_roster(source, other_change):
    old_path, old = source
    new_path = old_path.with_name("next.usda")
    new_path.write_text(old.GetRootLayer().ExportToString())
    new = Usd.Stage.Open(str(new_path))
    new.GetPrimAtPath("/World/Body").GetAttribute("physics:mass").Set(2)
    if other_change == "geometry":
        new.GetPrimAtPath("/World/Body").GetAttribute("points").Set([(0, 0, 0), (4, 0, 0), (0, 1, 0)])
    elif other_change == "gravity":
        new.GetPrimAtPath("/World/Physics").GetAttribute("physics:gravityMagnitude").Set(0)
    elif other_change == "automatic_mass":
        new.GetPrimAtPath("/World/Body").GetAttribute("physics:mass").Set(0)
    new.GetRootLayer().Save()
    roster = () if other_change == "wrong_roster" else ("/World/Body",)
    result = physics_mass_delta(str(old_path), str(new_path), roster)
    assert result == ([{"prim_path": "/World/Body", "mass": 2.0}] if other_change is None else None)


@pytest.mark.parametrize('other_change', [None, 'gravity', 'dynamic_geometry', 'transform'])
def test_static_collider_delta_preserves_every_other_physics_input(source, other_change):
    from ovrtx_blender_example.physics_generation_fingerprint import physics_static_geometry_delta
    old_path, old = source
    ground = UsdGeom.Cube.Define(old, '/World/Ground')
    ground.CreateSizeAttr(2)
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    old.GetRootLayer().Save()
    new_path = old_path.with_name('next.usda')
    new_path.write_text(old.GetRootLayer().ExportToString())
    new = Usd.Stage.Open(str(new_path))
    new.GetPrimAtPath('/World/Ground').GetAttribute('size').Set(3)
    if other_change == 'gravity':
        new.GetPrimAtPath('/World/Physics').GetAttribute('physics:gravityMagnitude').Set(0)
    elif other_change == 'dynamic_geometry':
        new.GetPrimAtPath('/World/Body').GetAttribute('points').Set([(0,0,0),(2,0,0),(0,1,0)])
    elif other_change == 'transform':
        UsdGeom.Xformable(new.GetPrimAtPath('/World/Ground')).AddTranslateOp().Set((1,0,0))
    new.GetRootLayer().Save()
    result = physics_static_geometry_delta(old_path, new_path)
    assert result == ([dict(prim_path='/World/Ground', attribute_name='size', value_type='Double', value=3.)] if other_change is None else None)


@pytest.mark.parametrize('mechanism', [None, 'kinematic', 'joint'])
def test_standalone_dynamic_geometry_delta_has_explicit_mechanism_guards(source, mechanism):
    from ovrtx_blender_example.physics_generation_fingerprint import physics_geometry_delta
    old_path, old = source
    if mechanism == 'kinematic':
        old.GetPrimAtPath('/World/Body').GetAttribute('physics:kinematicEnabled').Set(True)
    elif mechanism == 'joint':
        joint=UsdPhysics.FixedJoint.Define(old, '/World/Joint')
        joint.CreateBody0Rel().SetTargets(['/World/Body'])
    old.GetRootLayer().Save()
    new_path=old_path.with_name('next.usda');new_path.write_text(old.GetRootLayer().ExportToString())
    new=Usd.Stage.Open(str(new_path))
    new.GetPrimAtPath('/World/Body').GetAttribute('points').Set([(0,0,0),(2,0,0),(0,1,0)])
    new.GetRootLayer().Save()
    result=physics_geometry_delta(old_path,new_path,allow_dynamic=True)
    assert bool(result) is (mechanism is None)

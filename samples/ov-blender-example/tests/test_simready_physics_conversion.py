# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import authoring_properties  # noqa: E402
from ovrtx_blender_example import scene_generation  # noqa: E402
from ovrtx_blender_example import simready_physics_conversion  # noqa: E402


class _FakeId(dict):
    def __init__(self, name: str, session_uid: int, **values: object) -> None:
        super().__init__()
        self.name = name
        self.name_full = name
        self.session_uid = session_uid
        for key, value in values.items():
            setattr(self, key, value)


class _Collection:
    def __init__(self, name: str, objects: tuple[object, ...] = ()) -> None:
        self.name = name
        self.objects = objects
        self.children: list[_Collection] = []


def _ov_object(
    *,
    rigid: str = authoring_properties.INHERIT,
    collision: str = authoring_properties.INHERIT,
) -> object:
    return SimpleNamespace(
        physics=SimpleNamespace(
            rigid_body=SimpleNamespace(schema_opinion=rigid),
            collision=SimpleNamespace(schema_opinion=collision),
        )
    )


def _ov_material(opinion: str = authoring_properties.INHERIT) -> object:
    return SimpleNamespace(physics=SimpleNamespace(schema_opinion=opinion))


def _material(name: str = "Orange", uid: int = 20) -> _FakeId:
    material = _FakeId(name, uid, ov=_ov_material())
    material.update(
        {
            "pxr:usd:physics_density": 650.0,
            "pxr:usd:physics_dynamicFriction": 0.35,
            "pxr:usd:physics_staticFriction": 0.4,
            "pxr:usd:physics_restitution": 0.5,
            "pxr:usd:physics_type": "orange",
        }
    )
    return material


def _scene(
    *,
    body: _FakeId | None = None,
    reference: _FakeId | None = None,
    colliders: tuple[object, ...] = (),
    include_export: bool = True,
) -> SimpleNamespace:
    material = _material()
    mesh = _FakeId(
        "OrangeMesh",
        10,
        name_full="OrangeMesh",
        materials=(material,),
        vertices=(
            SimpleNamespace(co=(0.0, 0.0, 0.0)),
            SimpleNamespace(co=(1.0, 0.0, 0.0)),
            SimpleNamespace(co=(0.0, 1.0, 0.0)),
        ),
        polygons=(SimpleNamespace(vertices=(0, 1, 2)),),
    )
    reference = reference or _FakeId("Empty_joint_00", 3, type="EMPTY")
    body = body or _FakeId(
        "obs_orange_01_obj_00",
        1,
        type="MESH",
        data=mesh,
        parent=None,
        ov=_ov_object(),
        material_slots=(SimpleNamespace(material=material),),
        constraints=(
            SimpleNamespace(type="CHILD_OF", target=reference, name="ChildOf"),
        ),
        users_collection=(),
    )
    body.update(
        {
            "pxr:usd:physics_mass": 0.4666406379946932,
            "pxr:usd:physics_centerofmass": (
                -0.00025447955704294145,
                0.00018116607679985464,
                0.014395528472959995,
            ),
            "pxr:usd:physics_inertia": (
                0.0007005843744643366,
                0.0007012929457005791,
                0.0006700477519417232,
            ),
            "pxr:physics:principalAxes": (1.0, 0.0, 0.0, 0.0),
        }
    )
    root = _Collection("Scene Collection")
    objects = (body, reference, *colliders)
    if include_export:
        export = _Collection("Export")
        geometry = _Collection("Geometry", (body,))
        reference_prims = _Collection("ReferencePrims", (reference,))
        collider_collection = _Collection("Colliders", colliders)
        export.children.extend((geometry, reference_prims, collider_collection))
        root.children.append(export)
        body.users_collection = (geometry,)
    return SimpleNamespace(collection=root, objects=objects, ov=object())


def test_reads_valid_simready_unibody_values() -> None:
    unibody = simready_physics_conversion.read_scene_unibodies(_scene())[0]

    assert unibody.mass == pytest.approx(0.4666406379946932)
    assert unibody.center_of_mass == pytest.approx(
        (-0.00025447955704294145, 0.00018116607679985464, 0.014395528472959995)
    )
    assert unibody.diagonal_inertia == pytest.approx(
        (0.0007005843744643366, 0.0007012929457005791, 0.0006700477519417232)
    )
    assert unibody.principal_axes == pytest.approx((1.0, 0.0, 0.0, 0.0))
    assert unibody.density == 650.0
    assert unibody.dynamic_friction == 0.35
    assert unibody.static_friction == 0.4
    assert unibody.restitution == 0.5


def test_raw_simready_properties_without_structure_are_not_recognized() -> None:
    assert simready_physics_conversion.read_scene_unibodies(
        _scene(include_export=False)
    ) == ()


def test_rejects_separate_simready_collider_geometry() -> None:
    collider = _FakeId("Collider", 4, type="MESH")

    with pytest.raises(scene_generation.SceneGenerationError) as raised:
        simready_physics_conversion.read_scene_unibodies(
            _scene(colliders=(collider,))
        )

    assert raised.value.diagnostics[0]["reason"] == (
        "separate_simready_colliders_unsupported"
    )


def test_rejects_conflicting_simready_physics_materials() -> None:
    scene = _scene()
    body = scene.objects[0]
    second = _material("Rubber", 21)
    body.material_slots = (
        SimpleNamespace(material=body.material_slots[0].material),
        SimpleNamespace(material=second),
    )

    with pytest.raises(scene_generation.SceneGenerationError) as raised:
        simready_physics_conversion.read_scene_unibodies(scene)

    assert raised.value.diagnostics[0]["reason"] == (
        "simready_unibody_requires_one_physics_material"
    )


def test_explicit_object_physics_precedence_skips_simready_conversion() -> None:
    scene = _scene()
    body = scene.objects[0]
    body.ov = _ov_object(rigid=authoring_properties.APPLY)
    unibodies = simready_physics_conversion.read_scene_unibodies(scene)
    mapping = scene_generation.BlenderPrimPath(
        "Orange",
        "MESH",
        "/World/Orange",
        "/World/Orange/Orange",
    )

    assert (
        simready_physics_conversion.convert_scene_unibodies(
            unibodies,
            {scene_generation.BlenderId("OBJECT", 1): mapping},
            set(),
            {},
        )
        == ()
    )


def test_convert_scene_unibody_authors_usd_physics_surface() -> None:
    pytest.importorskip("pxr", reason="OpenUSD Python bindings unavailable")
    from pxr import Sdf, Usd, UsdPhysics

    scene = _scene()
    mapping = scene_generation.BlenderPrimPath(
        "Orange",
        "MESH",
        "/World/Orange",
        "/World/Orange/OrangeMesh",
    )
    records = simready_physics_conversion.convert_scene_unibodies(
        simready_physics_conversion.read_scene_unibodies(scene),
        {scene_generation.BlenderId("OBJECT", 1): mapping},
        set(),
        {},
    )
    root = Sdf.Layer.CreateAnonymous(".usda")
    layers = []
    for record in records:
        layer = Sdf.Layer.CreateAnonymous(".usda")
        assert layer.ImportFromString(record.layer_text)
        layers.append(layer)
    root.subLayerPaths = [layer.identifier for layer in layers]
    stage = Usd.Stage.Open(root)

    body_prim = stage.GetPrimAtPath(mapping.object_path)
    mesh_prim = stage.GetPrimAtPath(mapping.schema_path)
    material_targets = mesh_prim.GetRelationship(
        "material:binding:physics"
    ).GetTargets()
    material_api = UsdPhysics.MaterialAPI(stage.GetPrimAtPath(material_targets[0]))

    assert len(records) == 2
    assert "PhysicsRigidBodyAPI" in body_prim.GetAppliedSchemas()
    assert "PhysicsMassAPI" in body_prim.GetAppliedSchemas()
    assert "PhysicsCollisionAPI" in mesh_prim.GetAppliedSchemas()
    assert "PhysicsMeshCollisionAPI" in mesh_prim.GetAppliedSchemas()
    assert UsdPhysics.MeshCollisionAPI(mesh_prim).GetApproximationAttr().Get() == (
        "convexDecomposition"
    )
    assert len(material_targets) == 1
    assert UsdPhysics.MassAPI(body_prim).GetMassAttr().Get() == pytest.approx(
        0.4666406379946932
    )
    assert material_api.GetDensityAttr().Get() == pytest.approx(650.0)
    assert material_api.GetDynamicFrictionAttr().Get() == pytest.approx(0.35)


def test_simready_authoring_changes_topology_fingerprint() -> None:
    scene = _scene()
    identity = scene_generation.BlenderId("OBJECT", 1)
    first = scene_generation._topology_fingerprints(scene)[identity]  # noqa: SLF001

    scene.objects[0]["pxr:usd:physics_mass"] = 0.7
    second = scene_generation._topology_fingerprints(scene)[identity]  # noqa: SLF001

    scene.objects[0]["pxr:usd:physics_mass"] = 0.4666406379946932
    scene.objects[0].material_slots[0].material["pxr:usd:physics_restitution"] = 0.1
    third = scene_generation._topology_fingerprints(scene)[identity]  # noqa: SLF001

    assert second != first
    assert third != first

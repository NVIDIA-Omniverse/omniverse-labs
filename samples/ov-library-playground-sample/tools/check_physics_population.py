# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify that the simulation cache preserves physics and excludes visual shells."""
from pathlib import Path
import sys
import unittest
from pxr import Gf, Tf, Usd, UsdGeom, UsdPhysics, UsdShade
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from nvidia.ov_library_lab.native_runtime import NativeRuntime


class PhysicsPopulation(unittest.TestCase):
    def setUp(self):
        self.rt = object.__new__(NativeRuntime)
        self.rt.usd = Usd.Stage.CreateInMemory()
        self.rt.dirty = set()
        self.rt._physics_source_stage = None
        self.stage = self.rt.usd
        self.scene = UsdPhysics.Scene.Define(self.stage, '/World/Physics')
        self.scene.CreateGravityMagnitudeAttr(9.81)
        self.body = UsdGeom.Xform.Define(self.stage, '/World/Carton')
        self.position = self.body.AddTranslateOp()
        UsdPhysics.RigidBodyAPI.Apply(self.body.GetPrim())
        UsdPhysics.MassAPI.Apply(self.body.GetPrim()).CreateMassAttr(2.)
        self.collider = UsdGeom.Cube.Define(self.stage, '/World/Carton/Collider')
        UsdPhysics.CollisionAPI.Apply(self.collider.GetPrim())
        UsdGeom.Mesh.Define(self.stage, '/World/Carton/Visual/Shell')
        self.material = UsdShade.Material.Define(self.stage, '/World/Looks/Cardboard')
        self.friction = UsdPhysics.MaterialAPI.Apply(self.material.GetPrim()).CreateDynamicFrictionAttr(.38)
        UsdShade.MaterialBindingAPI.Apply(self.collider.GetPrim()).Bind(self.material, materialPurpose='physics')
        self.notice = Tf.Notice.Register(Usd.Notice.ObjectsChanged, self.rt._changed, self.stage)

    def tearDown(self):
        self.notice.Revoke()

    def test_parent_physics_and_material_remain_but_visual_shell_does_not(self):
        source = self.rt.physics_source()
        self.assertTrue(source.GetPrimAtPath(self.body.GetPath()).HasAPI(UsdPhysics.RigidBodyAPI))
        self.assertEqual(source.GetAttributeAtPath('/World/Carton.physics:mass').Get(), 2.)
        self.assertTrue(source.GetPrimAtPath(self.collider.GetPath()).HasAPI(UsdPhysics.CollisionAPI))
        self.assertTrue(source.GetPrimAtPath(self.material.GetPath()).HasAPI(UsdPhysics.MaterialAPI))
        self.assertFalse(source.GetPrimAtPath('/World/Carton/Visual'))
        self.assertTrue(source.GetPrimAtPath('/World/Physics'))

    def test_cached_stage_observes_live_pose_gravity_and_friction(self):
        source = self.rt.physics_source()
        self.position.Set(Gf.Vec3d(1, 4, 2))
        self.scene.GetGravityMagnitudeAttr().Set(1.62)
        self.friction.Set(.9)
        self.assertIs(self.rt.physics_source(), source)
        self.assertEqual(source.GetAttributeAtPath('/World/Carton.xformOp:translate').Get(), Gf.Vec3d(1,4,2))
        self.assertAlmostEqual(source.GetAttributeAtPath('/World/Physics.physics:gravityMagnitude').Get(), 1.62, places=5)
        self.assertAlmostEqual(source.GetAttributeAtPath('/World/Looks/Cardboard.physics:dynamicFriction').Get(), .9, places=5)

    def test_new_collider_invalidates_population(self):
        before = self.rt.physics_source()
        other = UsdGeom.Cube.Define(self.stage, '/World/NewCollider')
        UsdPhysics.CollisionAPI.Apply(other.GetPrim())
        source = self.rt.physics_source()
        self.assertIsNot(before, source)
        self.assertTrue(source.GetPrimAtPath('/World/NewCollider').HasAPI(UsdPhysics.CollisionAPI))

    def test_sibling_colliders_and_joints_are_preserved(self):
        other = UsdGeom.Cube.Define(self.stage, '/World/Carton/SecondCollider')
        UsdPhysics.CollisionAPI.Apply(other.GetPrim())
        joint = UsdPhysics.FixedJoint.Define(self.stage, '/World/Joint')
        joint.CreateBody0Rel().SetTargets([self.body.GetPath()])
        source = self.rt.physics_source()
        self.assertTrue(source.GetPrimAtPath('/World/Carton/SecondCollider'))
        self.assertTrue(source.GetPrimAtPath('/World/Joint'))


if __name__ == '__main__':unittest.main()

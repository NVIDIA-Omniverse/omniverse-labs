# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""USD bridge regressions: inherited poses, topology changes and prototypes."""
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
import numpy as np
from pxr import Gf, Tf, Usd, UsdGeom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'source'))
from nvidia.ov_library_lab.native_runtime import NativeRuntime


class TransformFlush(unittest.TestCase):
    def setUp(self):
        self.usd = Usd.Stage.CreateInMemory()
        self.parent = UsdGeom.Xform.Define(self.usd, '/Parent')
        self.parent_op = self.parent.AddTranslateOp()
        self.parent_op.Set(Gf.Vec3d(1, 2, 3))
        self.child = UsdGeom.Cube.Define(self.usd, '/Parent/Child')
        self.child.AddTranslateOp().Set(Gf.Vec3d(0, 1, 0))
        rt = self.rt = object.__new__(NativeRuntime)
        rt.usd = self.usd
        rt.dirty = set()
        rt.ordinal = 1
        rt._transform_members = {}
        rt._physics_source_stage = None
        rt.stage = NS(advance_write_floor=lambda ordinal: NS(wait=lambda: None))
        rt.ovstage = NS(AttributeSemantic=NS(MATRIX=1))
        self.writes = {}
        rt.write = lambda paths, name, values, **kw: self.writes.update(
            {(path, name): value for path, value in zip(paths, values)})
        self.notice = Tf.Notice.Register(Usd.Notice.ObjectsChanged, rt._changed, self.usd)
        self.addCleanup(self.notice.Revoke)

    def move(self, x):
        self.writes.clear()
        self.parent_op.Set(Gf.Vec3d(x, 2, 3))
        # This regression exercises transform publication; material and schema
        # values are covered by the native integration checks.
        self.rt.dirty = {p for p in self.rt.dirty if '.xformOp' in p}
        self.rt.flush()

    def test_inherited_child_pose_and_reset_flag_are_still_published(self):
        self.move(2)
        for x in (3, 4):
            self.move(x)
            np.testing.assert_array_equal(self.writes['/Parent/Child', 'omni:xform'],
                                          UsdGeom.Xformable(self.child).GetLocalTransformation())
            self.assertFalse(self.writes['/Parent/Child', 'omni:resetXformStack'])
        UsdGeom.Xformable(self.child).SetResetXformStack(True)
        self.move(5)
        self.assertTrue(self.writes['/Parent/Child', 'omni:resetXformStack'])

    def test_added_and_removed_descendants_invalidate_membership(self):
        self.move(2)
        added = UsdGeom.Cube.Define(self.usd, '/Parent/NewChild')
        added.AddTranslateOp().Set(Gf.Vec3d(7, 8, 9))
        self.move(3)
        self.assertIn(('/Parent/NewChild', 'omni:xform'), self.writes)
        self.usd.RemovePrim('/Parent/NewChild')
        self.move(4)
        self.assertNotIn(('/Parent/NewChild', 'omni:xform'), self.writes)

    def test_instance_prototype_is_refreshed_after_parent_changes(self):
        UsdGeom.Xform.Define(self.usd, '/Asset')
        mesh = UsdGeom.Cube.Define(self.usd, '/Asset/Body')
        mesh.AddTranslateOp().Set(Gf.Vec3d(0, 2, 0))
        instance = self.usd.DefinePrim('/Parent/Instance')
        instance.GetReferences().AddInternalReference('/Asset')
        instance.SetInstanceable(True)
        prototype = str(instance.GetPrototype().GetPath())+'/Body'
        self.move(2)
        self.assertIn((prototype, 'omni:xform'), self.writes)
        self.move(3)
        self.assertIn((prototype, 'omni:xform'), self.writes)


if __name__ == '__main__':
    unittest.main(verbosity=2)

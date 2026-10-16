# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run with OpenUSD Python (or inside Kit); core-only environments skip these tests."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

try:
    from pxr import Gf, Sdf, Tf, Usd, UsdGeom
except ImportError:
    Usd = None


@unittest.skipUnless(Usd, 'Requires OpenUSD Python; also run by the Kit verification script')
class SceneColorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        path = root / 'source/clients/extensions/utils/omni.example.reference_client/omni/example/reference_client/scene_colors.py'
        spec = importlib.util.spec_from_file_location('tested_scene_colors', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.compositor_type = module.SceneColors

    def test_composition_never_changes_asset_and_skips_unchanged_colors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'asset.usda'
            stage = Usd.Stage.CreateNew(str(path))
            shape = UsdGeom.Cylinder.Define(stage, '/Machine/Cylinder')
            original = [Gf.Vec3f(.1, .2, .3)]
            shape.CreateDisplayColorAttr(original)
            stage.GetRootLayer().Save()
            asset = path.read_bytes()
            edit_target = stage.GetEditTarget()
            unrelated = Sdf.Layer.CreateAnonymous('unrelated-session-edits.usda')
            stage.GetSessionLayer().subLayerPaths.append(unrelated.identifier)
            colors = self.compositor_type(stage)
            identity = {'source': 'gateway', 'id': 'machine-1'}
            colors.configure({('gateway', 'machine-1'): ['/Machine']}, [dict(identity, display_color={})])
            notices = []
            listener = Tf.Notice.Register(Usd.Notice.ObjectsChanged, lambda notice, sender: notices.append(1), stage)
            row = {'binding': identity, 'display_color': {'rgb': [1, 0, 0], 'unknown_rgb': [.3, .3, .3]}}
            try:
                colors.update([row])
                self.assertEqual(list(shape.GetDisplayColorAttr().Get()), [Gf.Vec3f(1, 0, 0)])
                self.assertEqual(shape.GetDisplayColorAttr().GetPropertyStack()[0].layer, colors.layer)
                count = len(notices)
                for _ in range(100):
                    colors.update([row])
                self.assertEqual(colors.write_count, 1)
                self.assertEqual(len(notices), count)
                colors.update([row], stale=True)
                self.assertEqual(list(shape.GetDisplayColorAttr().Get()), [Gf.Vec3f(.3, .3, .3)])
                self.assertEqual(colors.write_count, 2)
                self.assertEqual(stage.GetEditTarget(), edit_target)
                self.assertFalse(stage.GetRootLayer().dirty)
                self.assertEqual(path.read_bytes(), asset)
            finally:
                listener.Revoke()
                colors.close()
            self.assertEqual(list(shape.GetDisplayColorAttr().Get()), original)
            self.assertEqual(list(stage.GetSessionLayer().subLayerPaths), [unrelated.identifier])

    def test_nested_ownership_conflicts_and_configuration_removal(self):
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Cube.Define(stage, '/Line/Base')
        shape = UsdGeom.Cylinder.Define(stage, '/Line/Machine/Cylinder')
        shape.CreateDisplayColorAttr([Gf.Vec3f(.1, .2, .3)])
        shape.GetDisplayColorPrimvar().SetIndices([0])
        stage.GetPrimAtPath('/Line/Machine').SetAssetInfoByKey('sourceIds:gateway:id', 'machine')
        paths = {('gateway', 'line'): ['/Line'], ('gateway', 'machine'): ['/Line/Machine']}
        scene = [dict(source=source, id=identity, display_color={}) for source, identity in paths]
        colors = self.compositor_type(stage)
        try:
            colors.configure(paths, scene)
            self.assertEqual(colors.targets, {'/Line/Base': ('gateway', 'line'), '/Line/Machine/Cylinder': ('gateway', 'machine')})
            colors.update([{'binding': {'source': 'gateway', 'id': 'machine'},
                            'display_color': {'rgb': [1, 0, 0], 'unknown_rgb': [0, 0, 0]}}])
            self.assertEqual(list(shape.GetDisplayColorPrimvar().ComputeFlattened()), [Gf.Vec3f(1, 0, 0)])
            colors.configure(paths, [])
            self.assertEqual(list(shape.GetDisplayColorAttr().Get()), [Gf.Vec3f(.1, .2, .3)])
            paths[('other', 'machine')] = ['/Line/Machine']
            scene.append(dict(source='other', id='machine', display_color={}))
            colors.configure(paths, scene)
            self.assertNotIn('/Line/Machine/Cylinder', colors.targets)
            self.assertEqual(len(colors.diagnostics), 1)
        finally:
            colors.close()

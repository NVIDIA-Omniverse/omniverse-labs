# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transient display-color opinions; never save, reload or edit the asset layer."""
from pxr import Gf, Sdf, Usd, UsdGeom, Vt
from composable_bindings.reference.config import source_key
from composable_bindings.reference.scene_bindings import source_ids


class SceneColors:
    def __init__(self, stage):
        self.stage = stage
        self.layer = Sdf.Layer.CreateAnonymous('composable-display-colors.usda')
        self.targets = {}
        self.attributes = {}
        self.last_colors = {}
        self.diagnostics = []
        self.writing = False
        self.write_count = 0
        self.stage.GetSessionLayer().subLayerPaths.insert(0, self.layer.identifier)

    def configure(self, paths, scene):
        """Resolve renderable descendants only when bindings or scene structure change."""
        configured = {source_key(entry) for entry in scene if 'display_color' in entry}
        targets, conflicts, indexed = {}, set(), set()
        for identity in sorted(configured):
            for path in paths.get(identity, []):
                root = self.stage.GetPrimAtPath(path)
                if not root:
                    continue
                iterator = iter(Usd.PrimRange(root))
                for prim in iterator:
                    # A nested bound assembly owns its geometry independently.
                    if prim != root and source_ids(prim.GetAssetInfo()):
                        iterator.PruneChildren()
                        continue
                    if not prim.IsA(UsdGeom.Gprim) or prim.IsInstanceProxy():
                        continue
                    target = str(prim.GetPath())
                    if prim.HasAttribute('primvars:displayColor:indices'):
                        indexed.add(target)
                    if target in targets and targets[target] != identity:
                        conflicts.add(target)
                    targets[target] = identity
        self.diagnostics = ['Multiple color owners: ' + path for path in sorted(conflicts)]
        for path in conflicts:
            targets.pop(path)
        if targets == self.targets:
            return
        self.writing = True
        try:
            # Only Sdf operations inside the block: USD queries above have finished.
            with Sdf.ChangeBlock():
                self.layer.Clear()
                self.attributes.clear()
                for path in targets:
                    prim = Sdf.CreatePrimInLayer(self.layer, path)
                    attr = Sdf.AttributeSpec(prim, 'primvars:displayColor', Sdf.ValueTypeNames.Color3fArray)
                    attr.SetInfo('interpolation', 'constant')
                    if path in indexed:
                        indices = Sdf.AttributeSpec(prim, 'primvars:displayColor:indices', Sdf.ValueTypeNames.IntArray)
                        indices.default = Sdf.ValueBlock()
                    self.attributes[path] = attr
        finally:
            self.writing = False
        self.targets = targets
        self.last_colors.clear()

    def update(self, rows, stale=False):
        """One batched default-value edit per changed target; unchanged RGB costs no write."""
        # Compare the float32 values actually authored, including gradient rounding.
        colors = {source_key(row['binding']): tuple(Gf.Vec3f(*row['display_color']['unknown_rgb' if stale else 'rgb']))
                  for row in rows if 'display_color' in row}
        changes = [(path, colors[identity]) for path, identity in self.targets.items()
                   if identity in colors and self.last_colors.get(path) != colors[identity]]
        if not changes:
            return
        self.writing = True
        try:
            with Sdf.ChangeBlock():
                for path, rgb in changes:
                    self.attributes[path].default = Vt.Vec3fArray([Gf.Vec3f(*rgb)])
        finally:
            self.writing = False
        self.last_colors.update(changes)
        self.write_count += len(changes)

    def close(self):
        """Remove only our layer, revealing the original authored colors again."""
        self.writing = True
        try:
            sublayers = self.stage.GetSessionLayer().subLayerPaths
            if self.layer.identifier in sublayers:
                sublayers.remove(self.layer.identifier)
        finally:
            self.writing = False
        self.targets.clear()
        self.attributes.clear()
        self.last_colors.clear()

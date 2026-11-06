# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Factory planning exhibits: conveyor fit, rack revisions and a pallet lift."""
import math

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
from .extensions import call

QC_COLORS = ((.025, .24, .65), (.95, .30, .015), (.025, .55, .07))


class ExtraExhibits(call('exhibit_base') or object):
    def __init__(self, gallery):
        self.g = gallery
        self.stage = gallery.stage
        self.storage_variant = 0
        self.saved_variant = None
        self.storage_revision = 0
        self.storage_status = "Working: empty rack  |  no saved baseline"
        self.snapshot_path = gallery.data_dir.parent / "artifacts" / "storage" / "saved-asset.usda"
        self.revisions = []
        self.selected_revision = len(self.revisions)-1
        self.undo_stack, self.redo_stack = [], []
        self.lift_value = .25
        self.part_color = 0
        self.work_light_on = True
        gallery.materials['pink'] = gallery.material('pink', (.85, .18, .45), .25, .3)
        # Keep a populated material bound to the sticker. Rebinding to materials
        # unused at initial RTX population otherwise falls back to neutral gray.
        self.qc_material = gallery.material('qc_label', QC_COLORS[0], 0., .65)
        self.qc_color = UsdShade.Shader(self.stage.GetPrimAtPath(
            self.qc_material.GetPath().AppendChild('Surface'))).GetInput('diffuseColor')
        call('build_extra', self)
        self._storage()
        self._interface()
        gallery.active_room = None

    def _asset(self, name, x):
        g = self.g
        path = "/World/Storage/"+name
        with g.factory.assembly(path, (x, 0, -5.4)) as root:
            root.GetPrim().CreateAttribute("lab:variant", Sdf.ValueTypeNames.Int).Set(0)
            blocks = []
            for i in range(3):
                blocks.append(g.factory.rack(path+f"/Configuration{i}", (0, 0, 0), 3.0, 2.8, 2, stock_levels=i))
        self._set_asset(root, blocks, 0)
        return root, blocks

    def _storage(self):
        g = self.g
        g.active_room = "storage"
        for name, x in (("Working", -2.3), ("Saved", 2.3)):
            g.factory.boundary(f"/World/Storage/{name}Bay", x, -5.4, 3.5, 2.2)
            a = g.room_position("storage", (x-1.7, 0, -7.1))
            b = g.room_position("storage", (x+1.7, 0, -3.7))
            g.obstacles.append((min(a[0], b[0]), max(a[0], b[0]), min(a[2], b[2]), max(a[2], b[2])))
        self.working, self.working_blocks = self._asset("WorkingAsset", -2.3)
        self.saved, self.saved_blocks = self._asset("SavedAsset", 2.3)
        self.saved.MakeInvisible()
        if not g.campus:
            g.sign("StorageLabel", (0, .37, -3.65), 7.5, .6, "storage_label.png")

    def _set_asset(self, root, blocks, variant):
        root.GetPrim().GetAttribute("lab:variant").Set(variant)
        for i, block in enumerate(blocks):
            (block.MakeVisible if i == variant else block.MakeInvisible)()

    def change_asset(self):
        self.set_asset_variant((self.storage_variant+1) % 3)

    def set_asset_variant(self, variant, remember=True):
        if remember and variant != self.storage_variant:
            self.undo_stack.append(self.storage_variant)
            self.redo_stack.clear()
        self.storage_variant = int(variant)
        self._set_asset(self.working, self.working_blocks, self.storage_variant)
        self.storage_status = f"Working: {self.storage_variant*3} cartons\nSaved baseline unchanged"

    def undo_asset(self, redo=False):
        source, target = (self.redo_stack, self.undo_stack) if redo else (self.undo_stack, self.redo_stack)
        if source:
            target.append(self.storage_variant)
            self.set_asset_variant(source.pop(), remember=False)

    def _interface(self):
        g = self.g
        g.active_room = "interface"
        g.box("/World/Interface/Base", (0, .15, -5.5), (6.8, .3, 4.5), "black", obstacle=True)
        g.factory.boundary("/World/Interface/Safety", 0, -5.5, 7.2, 4.8)
        for n, x in enumerate((-2.5, 2.5)):
            g.factory.bollard(f"/World/Interface/Guard{n}", (x, 0, -3.5))
        if g.factory.simready.assets:
            from .pallet_stacker import PalletStacker
            self.lift_table=PalletStacker(g)
        else:
            from .lift_table import LiftTable
            self.lift_table=LiftTable(g)
        self.lift,self.part=self.lift_table.platform,self.lift_table.tag
        g.bind(self.part.GetPrim(), self.qc_material)
        g.factory.stacklight("/World/Interface/Stacklight", (3., 1.82, -3.7))
        g.box("/World/Interface/HMIStand", (3.0, .65, -3.7), (.1, 1.3, .1), "steel")
        g.box("/World/Interface/HMI", (3.0, 1.5, -3.7), (.8, .62, .13), "black")
        g.sign("HMI", (3., 1.5, -3.62), .72, .48, "hmi.png")
        self.work_light = g.rect_light("InterfaceWork", (0, 4.6, -4.0), (0, 1.8, -5.5), (1., .63, .38), 5000., 2., 2.)
        g.placard("InterfaceLabel", (-2.8, 1.05, -4.2), 2., .70, "interface_label.png", floor_y=.3)
        self.set_lift(self.lift_value)

    def set_lift(self, value):
        self.lift_value = max(0., min(1., float(value)))
        self.lift_table.set_height(self.lift_value if self.g.factory.simready.assets else .75+self.lift_value*2.1)

    def change_part_color(self):
        self.set_quality((self.part_color + 1) % 3)

    def set_quality(self, value):
        value = int(value)
        if value not in range(len(QC_COLORS)):
            raise ValueError('Unknown QC status')
        self.part_color = value
        self.qc_color.Set(Gf.Vec3f(*QC_COLORS[value]))

    def toggle_work_light(self):
        self.work_light_on = not self.work_light_on
        self.work_light.GetIntensityAttr().Set(5000. if self.work_light_on else 0.)

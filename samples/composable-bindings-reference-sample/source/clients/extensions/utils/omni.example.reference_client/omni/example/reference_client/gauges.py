# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metric-independent vector gauges, adapted from the advanced demo's ring widget.

One SceneView for the viewport; no texture-backed sc.Widget per metric. Only
changed text/arc properties are assigned. Network/formatting stay in shared Client.
"""
import math
import omni.ui as ui
from omni.ui import scene as sc
from pxr import Gf, Usd, UsdGeom

BEGIN = -math.pi * 1.25
SWEEP = math.pi * 1.5
TRACK = (.12, .22, .30, 1.)
TEXT = (.06, .12, .17, 1.)
COLORS = {'normal': (.015, .48, .36, 1.), 'alarm': (.9, .08, .04, 1.), 'unknown': (.55, .6, .65, 1.)}
SCALE = .009
STEP = 1.32


def label(text, y, size, color=TEXT):
    with sc.Transform(transform=sc.Matrix44.get_translation_matrix(0, y, .02)):
        return sc.Label(text, size=size, color=color, alignment=ui.Alignment.CENTER)


class MetricGauge:
    """The same renderer for any configured numeric metric; no metric-name branches."""
    def __init__(self, spec, index):
        self.spec = spec
        self.previous = None
        with sc.Transform(transform=sc.Matrix44.get_translation_matrix(0, index * STEP, 0)):
            with sc.Transform(transform=sc.Matrix44.get_scale_matrix(SCALE, SCALE, SCALE)):
                self.title = label(spec['title'], 61, 15)
                self.track = sc.Arc(45, begin=BEGIN, end=BEGIN+SWEEP, axis=2,
                    thickness=5, color=TRACK, wireframe=True, sector=False, tesselation=48)
                self.arc = sc.Arc(45, begin=BEGIN, end=BEGIN+.001, axis=2,
                    thickness=5, color=TEXT, wireframe=True, sector=False, tesselation=48)
                self.value = label('Waiting', 3, 20)
                self.state = label('WAITING', -22, 10)
                limits = spec.get('range')
                self.limits = label(f'{limits[0]:g} .. {limits[1]:g}' if limits else '', -53, 10)
                self.track.visible = self.arc.visible = spec['kind'] == 'gauge'
                if limits:
                    for i in range(9):
                        angle = BEGIN + SWEEP * i / 8
                        sc.Line([49*math.cos(angle),49*math.sin(angle),0],
                                [53*math.cos(angle),53*math.sin(angle),0], color=TRACK, thickness=1)

    def update(self, display):
        state = (display['text'], display['state'], display['fraction'])
        if state == self.previous: return
        self.previous = state
        self.value.text = display['text']
        self.state.text = display['state'].upper()
        self.value.color = self.state.color = self.arc.color = COLORS[display['state']]
        self.arc.end = BEGIN + max(.001, SWEEP * display['fraction'])
        self.arc.visible = self.spec['kind'] == 'gauge' and display['sample'] is not None

    def resize(self, pixels):
        factor = max(.2, min(1.4, pixels / 100))
        for item, size in ((self.title,15),(self.value,20),(self.state,10),(self.limits,10)):
            item.size = size * factor


class MachineGauges:
    def __init__(self, scene_view, prim, displays, offset=(0,0,0), scale=1.):
        self.prim = prim
        self.selected = False
        self.stage_visible = False
        self.offset = Gf.Vec3d(*offset)
        self.scale = scale
        with scene_view.scene:
            self.root = sc.Transform()
            with self.root:
                with sc.Transform(look_at=sc.Transform.LookAt.CAMERA):
                    with sc.Transform(transform=sc.Matrix44.get_scale_matrix(scale,scale,scale)):
                        self.gauges = {d['spec']['id']: MetricGauge(d['spec'], i)
                                       for i,d in enumerate(displays)}
        self.anchor = Gf.Vec3d(0)
        self.update_anchor()

    def update_anchor(self):
        if not self.prim.IsValid():
            self.stage_visible = False
            self.root.visible = False
            return
        bound = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]).ComputeWorldBound(self.prim).ComputeAlignedRange()
        if bound.IsEmpty():
            self.stage_visible = False
            self.root.visible = False
            return
        self.stage_visible = UsdGeom.Imageable(self.prim).ComputeVisibility() != UsdGeom.Tokens.invisible
        self.root.visible = self.selected and self.stage_visible
        center = bound.GetMidpoint()
        self.anchor = Gf.Vec3d(center[0], center[1], bound.GetMax()[2] + .85) + self.offset
        self.root.transform = sc.Matrix44.get_translation_matrix(*self.anchor)

    def set_selected(self, selected):
        self.selected = selected
        self.root.visible = selected and self.stage_visible

    def resize(self, viewport):
        # Scene label fonts are in pixels, while ring geometry is in world units.
        # Match them using the viewport's conformed projection after camera changes.
        matrix = viewport.world_to_ndc
        camera = viewport.view.GetInverse()
        up = camera.TransformDir(Gf.Vec3d(0,1,0)).GetNormalized()
        a, b = matrix.Transform(self.anchor), matrix.Transform(self.anchor + up * .9 * self.scale)
        pixels = abs(b[1]-a[1]) * viewport.resolution[1] / 2
        for gauge in self.gauges.values(): gauge.resize(pixels)


class ViewportHud:
    """Fixed UI cards with independent orthographic cameras; no USD camera binding."""
    def __init__(self, frame, items):
        self.gauges = {}
        self.views = []
        frame.clear()
        if not items:
            return
        identity = [1.,0.,0.,0., 0.,1.,0.,0., 0.,0.,1.,0., 0.,0.,0.,1.]
        with frame:
            with ui.VStack():
                ui.Spacer(height=16)
                with ui.HStack(height=0):
                    ui.Spacer(width=16)
                    with ui.VStack(width=200, height=0, spacing=8):
                        for binding, display in items:
                            with ui.ZStack(width=200, height=200):
                                ui.Rectangle(style={'background_color':0xffeadbc3,'border_radius':10})
                                view = sc.SceneView(sc.CameraModel(identity, identity), width=200, height=200)
                                self.views.append(view)
                                with view.scene:
                                    with sc.Transform(transform=sc.Matrix44.get_scale_matrix(1.5,1.5,1.5)):
                                        gauge = MetricGauge(display['spec'],0)
                                        gauge.value.size = 16
                                        self.gauges[binding,display['spec']['id']] = gauge
                    ui.Spacer()
                ui.Spacer()

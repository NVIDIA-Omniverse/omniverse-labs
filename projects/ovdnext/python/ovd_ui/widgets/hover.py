# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Passive plot hover using OVUI's public scene gesture API.

Widget.mouse_moved intentionally reports only pressed-pointer motion. This
transparent, nondrawing scene supplies normalized passive pointer coordinates
without polling OS input or reaching into Inspector internals. Compose inside a
ZStack matching the target surface; put its persistent input button above this
view. Keep the HoverOverlay alive and call destroy() when the panel closes.
"""
from omni.ui_scene import scene as sc
from omni.ui import color as cl


class HoverOverlay:
    def __init__(self, on_move, on_leave=None):
        self.on_move, self.on_leave = on_move, on_leave
        self.position = None
        self.destroyed = False
        projection = [1,0,0,0, 0,1,0,0, 0,0,2e-7,0, 0,0,1,1]
        self.widget = sc.SceneView(sc.CameraModel(projection, sc.Matrix44.get_translation_matrix(0,0,-1)),
            aspect_ratio_policy=sc.AspectRatioPolicy.STRETCH, explicit_hover=True, opaque_for_mouse_events=False)
        self.gesture = sc.HoverGesture(on_began_fn=self._move, on_changed_fn=self._move, on_ended_fn=self._leave)
        with self.widget.scene:
            sc.Rectangle(2,2,color=cl.transparent,gesture=self.gesture)

    def _move(self, sender):
        if self.destroyed:return
        point = sender.gesture_payload.ray_closest_point
        self.position = ((point[0]+1)*.5, (1-point[1])*.5)
        self.on_move(*self.position)

    def _leave(self, sender):
        if self.destroyed:return
        self.position = None
        if self.on_leave:self.on_leave()

    def destroy(self):
        self.destroyed = True
        self.on_move = self.on_leave = None
        self.widget.scene.clear()
        self.widget.visible = False

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The existing Qt visual action's measured, explicitly accepted values.

These are the effective values from the original controls, not the nominal
QSettings defaults. Independent joint scale is accepted by the application
transaction alongside these camera/rendering values.
"""
from dataclasses import dataclass, replace
import numpy as np

from .camera import MovementMode
from .render_settings import DisplayMode, RenderingSettings


NEAR = 0.9111621975898743
NEAR_SLIDER = 57
GIZMO_SCALE = 3.1517279148101807
SPEED_SCALE = 0.025118865072727203
RENDER_FIELDS = ("wireframe", "sleeping_objects", "trigger_shapes", "particle_size")
SELECTION_CATEGORIES = (1, 2, 4, 32)
RESET_OVERLAY_MASK = 1 | 2 | 4 | 32 | 64


@dataclass(frozen=True)
class VisualReset:
    rendering: RenderingSettings
    position: tuple[float, float, float]
    overlay_mask: int

    @classmethod
    def prepare(cls, rendering, camera, overlay_mask):
        # Camera.set_world_up replaces its arrays, so staging through a
        # dataclass copy leaves the shared renderer/controller camera intact.
        candidate = replace(camera)
        if not np.array_equal(candidate.world_up,(0,1,0)):
            candidate.set_world_up((0, 1, 0))
        candidate.left_handed = False
        return cls(replace(rendering, wireframe=DisplayMode.NONE,
                           sleeping_objects=DisplayMode.ALL,
                           trigger_shapes=DisplayMode.ALL, particle_size=5.),
                   tuple(float(v) for v in candidate.position),
                   overlay_mask & ~RESET_OVERLAY_MASK)

    def apply_camera(self, controller):
        camera = controller.camera
        # Use owned numeric arrays without replacing the shared Camera object.
        camera.position = np.array(self.position)
        camera.world_up = np.array((0., 1., 0.))
        camera.left_handed = False
        camera.near = NEAR
        controller.mode = MovementMode.FIRST_PERSON
        controller.speed_scale = SPEED_SCALE
        # The original renderer retains its tracking offset across up-axis
        # changes. Keep both the target and offset: the next owned frame can
        # resume that same tracked view. Captured-camera intent also survives.

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small deterministic, floor-constrained walking controller, independent of Kit."""
import math
from .layout import LOBBY_POSE, contains, room_at


class Walker:
    def __init__(self, obstacles=()):
        self.obstacles = list(obstacles)
        self.dynamic_blockers = []
        self.x, self.z, self.yaw, self.pitch = LOBBY_POSE
        self.radius = 0.24

    def valid(self, x, z):
        r = self.radius
        if not contains(x, z, r):
            return False
        return (not any(a-r < x < b+r and c-r < z < d+r for a, b, c, d in self.obstacles)
                and not any(block(x,z,r) for block in self.dynamic_blockers))

    def move(self, forward, right, dt, fast=False):
        length = math.hypot(forward, right)
        if not length:
            return
        distance = min(max(dt, 0.0), 0.1) * (5.4 if fast else 3.0)
        dx = (math.sin(self.yaw)*forward + math.cos(self.yaw)*right) / length * distance
        dz = (-math.cos(self.yaw)*forward + math.sin(self.yaw)*right) / length * distance
        # Substeps prevent tunnelling through thin walls even after a slow frame.
        steps = max(1, math.ceil(distance / 0.08))
        for _ in range(steps):
            if self.valid(self.x + dx/steps, self.z):
                self.x += dx/steps
            if self.valid(self.x, self.z + dz/steps):
                self.z += dz/steps

    def room(self):
        return room_at(self.x, self.z)

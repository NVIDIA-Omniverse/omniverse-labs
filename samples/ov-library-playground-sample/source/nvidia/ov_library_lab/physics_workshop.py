# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Repeatable, actual PhysX experiments on the receiving conveyor."""
from pxr import Gf, UsdPhysics


def configure(lab, mode=None, friction=None):
    lab.reset_physics()
    if mode is not None:
        lab.physics_mode = mode
    if friction is not None:
        lab.friction = float(friction)
    g = lab.gallery
    material = UsdPhysics.MaterialAPI(g.bounce_material.GetPrim())
    material.GetStaticFrictionAttr().Set(lab.friction*1.2)
    material.GetDynamicFrictionAttr().Set(lab.friction)
    for path, initial in g.initial_positions.items():
        i = int(path[-1])
        x = -8.85 if '/Simulated/' in path else -13.15
        tilt = ((4, -5, 3), (-3, 7, -5), (6, 2, -3), (-5, -6, 4), (3, 4, 6), (-4, 3, -6))[i]
        if lab.physics_mode == 'stack':
            pos = (x+(i%2-.5)*.85, 1.25+(i//2)*.60, -5.5)
            tilt = (0, 0, 0)
        elif lab.physics_mode == 'slide':
            pos = (x-1.1+(i%3)*.8, 1.21, -6.15+(i//3)*1.3)
            tilt = (0, 0, 0)
        else:
            pos = (initial[0], 3.35 if lab.high_drop else 2.25, initial[2])
        prim = lab.stage.GetPrimAtPath(path)
        prim.GetAttribute('xformOp:translate').Set(Gf.Vec3d(*g.room_position('physics', pos)))
        prim.GetAttribute('xformOp:rotateXYZ').Set(Gf.Vec3f(*tilt))
        if path in g.dynamic_paths:
            body = UsdPhysics.RigidBodyAPI(prim)
            body.CreateVelocityAttr().Set(Gf.Vec3f(0))
            body.CreateAngularVelocityAttr().Set(Gf.Vec3f(0))


def nudge(lab):
    """Wake sleeping bodies and apply a real velocity pulse at their centres."""
    if not lab.sim_started:
        lab.toggle_physics()
    pulse(lab, lab.gallery.dynamic_paths[-2:], .8)
    lab.running = True


def pulse(lab, paths, velocity):
    lab.native.pulse(paths, velocity)

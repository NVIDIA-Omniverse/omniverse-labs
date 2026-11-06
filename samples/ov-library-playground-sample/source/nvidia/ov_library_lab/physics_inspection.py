# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inspect the authored native collision shapes and physical test-load mass."""
from pxr import Gf, UsdGeom, UsdPhysics


def build(g):
    g.collider_outlines = []
    g.show_colliders = False
    colliders = [p for p in g.stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)
                 and str(p.GetPath()).startswith('/World/Physics/') and p.IsA(UsdGeom.Mesh)]
    for prim in colliders:
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        edges, offset = set(), 0
        for count in mesh.GetFaceVertexCountsAttr().Get():
            face = list(indices[offset:offset+count]); offset += count
            for a,b in zip(face, face[1:]+face[:1]):
                edges.add(tuple(sorted((tuple(points[a]),tuple(points[b])))))
        curve = UsdGeom.BasisCurves.Define(g.stage, str(prim.GetPath())+'Outline')
        curve.CreateTypeAttr('linear'); curve.CreateWrapAttr('nonperiodic')
        curve.CreateCurveVertexCountsAttr([2]*len(edges))
        curve.CreatePointsAttr([Gf.Vec3f(*p) for edge in edges for p in edge])
        curve.CreateWidthsAttr([.008]); curve.SetWidthsInterpolation('constant')
        curve.AddTransformOp().Set(UsdGeom.Xformable(prim).GetLocalTransformation())
        g.bind(curve.GetPrim(), 'physics_glow')
        curve.MakeInvisible(); g.collider_outlines.append(curve)


def show(g, enabled):
    g.show_colliders = enabled
    for curve in g.collider_outlines:
        (curve.MakeVisible if enabled else curve.MakeInvisible)()


def set_mass(lab, mass):
    from .physics_workshop import configure
    configure(lab)
    lab.carton_mass = float(mass)
    for path in lab.gallery.dynamic_paths:
        api = UsdPhysics.MassAPI(lab.stage.GetPrimAtPath(path))
        api.GetMassAttr().Set(mass)
        mesh = UsdGeom.Mesh(lab.stage.GetPrimAtPath(path+'/CorrugatedBox'))
        transform = UsdGeom.Xformable(mesh).GetLocalTransformation()
        points = [transform.Transform(Gf.Vec3d(p)) for p in mesh.GetPointsAttr().Get()]
        sides = [max(p[i] for p in points)-min(p[i] for p in points) for i in range(3)]
        w,h,d = sides
        api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(mass*(h*h+d*d)/12, mass*(w*w+d*d)/12, mass*(w*w+h*h)/12))

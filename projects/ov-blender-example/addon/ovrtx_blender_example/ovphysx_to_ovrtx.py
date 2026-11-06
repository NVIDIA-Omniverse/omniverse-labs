# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Translate authoritative OVPhysX poses into OVRTX transform values."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .shared_stage_composition import BodyPose
from .ovrtx_value_updates import OvrtxTransformValue


@dataclass(frozen=True)
class BodyShapeTransform:
    """Authored geometry transform relative to a body's world rigid pose."""
    prim_path: str
    matrix: tuple[tuple[float, ...], ...]
    body_path: str = ""


def read_rigid_body_view_paths(usd_path: str, object_paths: Sequence[str]) -> frozenset[str]:
    """Index compiler-owned rigid roots whose visual scale lives on children."""
    if not object_paths:
        return frozenset()
    from pxr import Usd
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        raise ValueError(f"Cannot inspect compiled body transforms: {usd_path}")
    return frozenset(path for path in object_paths
                     if stage.GetPrimAtPath(path)
                     and stage.GetPrimAtPath(path).GetAttribute("xformOp:transform:ovPhysics"))


def rigid_body_view_matrix(matrix):
    """Strip owner scale already carried by compiled collider/visual geometry."""
    from .physics_frames import matrix_to_transform
    column_matrix = tuple(zip(*matrix))
    return matrix_to_transform(column_matrix, allow_nonuniform=True).frame.gf_matrix()


def read_body_shape_transforms(usd_path: str, body_paths: Sequence[str]) -> tuple[BodyShapeTransform, ...]:
    # Read the immutable generation once on activation, never on viewport draw.
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.Open(usd_path)
    if not stage:
        raise ValueError(f"Cannot inspect authored body transforms: {usd_path}")
    cache = UsdGeom.XformCache()
    shapes = []
    bodies = set(body_paths)
    for path in body_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim or not UsdGeom.Xformable(prim):
            raise ValueError(f"Missing authored body transform: {path}")
        world = cache.GetLocalToWorldTransform(prim)
        rigid = Gf.Matrix4d(1).SetRotate(Gf.Transform(world).GetRotation())
        rigid.SetTranslateOnly(world.ExtractTranslation())
        inverse_rigid = rigid.GetInverse()
        descendants = iter(Usd.PrimRange(prim))
        for child in descendants:
            child_path = str(child.GetPath())
            xformable = UsdGeom.Xformable(child)
            if child != prim and (child_path in bodies or (xformable and xformable.GetResetXformStack())):
                descendants.PruneChildren()
                continue
            if not xformable:
                continue
            # OVRTX consumes per-prim world transforms. Updating only the body
            # Xform does not move its stock-exported child Mesh in the renderer.
            shape = cache.GetLocalToWorldTransform(child) * inverse_rigid
            shapes.append(BodyShapeTransform(child_path, tuple(tuple(float(v) for v in row) for row in shape), path))
    return tuple(shapes)


def translate_values(
    poses: Sequence[BodyPose], body_scale: float,
    body_shapes: Sequence[BodyShapeTransform] = (),
) -> list[OvrtxTransformValue]:
    scale = float(body_scale)
    if not math.isfinite(scale):
        raise ValueError("body_scale must be finite")
    paths: set[str] = set()
    for pose in poses:
        if pose.prim_path in paths:
            raise ValueError(f"duplicate body pose path: {pose.prim_path}")
        paths.add(pose.prim_path)
    values: list[OvrtxTransformValue] = []
    shapes: dict[str, list[BodyShapeTransform]] = {}
    for shape in body_shapes:
        shapes.setdefault(shape.body_path or shape.prim_path, []).append(shape)
    for pose in poses:
        rigid = _matrix4d_rows(pose.translate, pose.orient, scale)
        targets = shapes[pose.prim_path] if body_shapes else (None,)
        for target in targets:
            matrix = rigid if target is None else [
                [sum(target.matrix[row][k] * rigid[k][column] for k in range(4))
                 for column in range(4)] for row in range(4)]
            if not all(math.isfinite(value) for row in matrix for value in row):
                raise ValueError(f"body pose matrix must be finite: {pose.prim_path}")
            values.append(OvrtxTransformValue(pose.prim_path if target is None else target.prim_path, matrix))
    return values


def _matrix4d_rows(
    translate: tuple[float, float, float],
    orient: tuple[float, float, float, float],
    scale: float,
) -> list[list[float]]:
    tx, ty, tz = translate
    x, y, z, w = orient
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    flat = [
        scale * (1.0 - 2.0 * (yy + zz)), scale * 2.0 * (xy + wz), scale * 2.0 * (xz - wy), 0.0,
        scale * 2.0 * (xy - wz), scale * (1.0 - 2.0 * (xx + zz)), scale * 2.0 * (yz + wx), 0.0,
        scale * 2.0 * (xz + wy), scale * 2.0 * (yz - wx), scale * (1.0 - 2.0 * (xx + yy)), 0.0,
        tx, ty, tz, 1.0,
    ]
    return [flat[index : index + 4] for index in range(0, 16, 4)]

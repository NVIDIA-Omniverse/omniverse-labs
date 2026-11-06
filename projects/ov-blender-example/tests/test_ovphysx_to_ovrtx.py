# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import math
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example.ovphysx_to_ovrtx import BodyShapeTransform, read_body_shape_transforms, translate_values
from ovrtx_blender_example.ovrtx_value_updates import OvrtxTransformValue
from ovrtx_blender_example.shared_stage_composition import BodyPose


def _pose(path: str = "/World/Body") -> BodyPose:
    return BodyPose(path, (1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0))


def test_compiled_body_view_matrix_preserves_parent_rotation_translation_and_rejects_shear():
    from ovrtx_blender_example.ovphysx_to_ovrtx import rigid_body_view_matrix
    matrix=((0.,2.,0.,0.),(-3.,0.,0.,0.),(0.,0.,4.,0.),(5.,6.,7.,1.))
    expected=((0.,1.,0.,0.),(-1.,0.,0.,0.),(0.,0.,1.,0.),(5.,6.,7.,1.))
    assert [v for row in rigid_body_view_matrix(matrix) for v in row] == pytest.approx([v for row in expected for v in row])
    with pytest.raises(ValueError,match='shear'):
        rigid_body_view_matrix(((1.,0.,0.,0.),(.2,1.,0.,0.),(0.,0.,1.,0.),(0.,0.,0.,1.)))


def test_translate_values_emits_complete_scaled_row_vector_matrix() -> None:
    assert translate_values([_pose()], 2.0) == [
        OvrtxTransformValue(
            "/World/Body",
            [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [1.0, 2.0, 3.0, 1.0],
            ],
        )
    ]


def test_translate_values_preserves_input_order() -> None:
    poses = [_pose("/World/B"), _pose("/World/A")]
    values = translate_values(poses, 1.0)
    assert [value.prim_path for value in values] == ["/World/B", "/World/A"]


def test_translate_values_accepts_empty_input() -> None:
    assert translate_values([], 1.0) == []


def test_per_body_shape_scale_survives_rotating_and_falling():
    shape = BodyShapeTransform("/World/Body", (
        (0.5, 0.0, 0.0, 0.0), (0.0, 0.75, 0.0, 0.0),
        (0.0, 0.0, 0.25, 0.0), (0.0, 0.0, 0.0, 1.0)))
    pose = BodyPose("/World/Body", (2.0, 3.0, 0.25), (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)))
    matrix = translate_values((pose,), 1.0, (shape,))[0].matrix
    assert matrix[0] == pytest.approx((0.0, 0.5, 0.0, 0.0))
    assert matrix[1] == pytest.approx((-0.75, 0.0, 0.0, 0.0))
    assert matrix[2] == pytest.approx((0.0, 0.0, 0.25, 0.0))
    assert matrix[3] == pytest.approx((2.0, 3.0, 0.25, 1.0))


def test_authored_shape_preserves_nested_rotated_nonuniform_geometry(tmp_path):
    from pxr import Gf, Usd, UsdGeom
    source = tmp_path / "nested.usda"
    stage = Usd.Stage.CreateNew(str(source))
    parent = UsdGeom.Xform.Define(stage, "/Parent")
    parent.AddTranslateOp().Set((4, 5, 6))
    parent.AddRotateZOp().Set(30)
    parent.AddScaleOp().Set((2, 3, 4))
    body = UsdGeom.Xform.Define(stage, "/Parent/Body")
    body.AddTranslateOp().Set((1, 2, 3))
    body.AddRotateZOp().Set(15)
    body.AddScaleOp().Set((0.5, 0.75, 0.25))
    mesh = UsdGeom.Cube.Define(stage, "/Parent/Body/Visual")
    mesh.AddTranslateOp().Set((1, 0, 0))
    stage.GetRootLayer().Save()
    world = UsdGeom.XformCache().GetLocalToWorldTransform(body.GetPrim())
    quat = Gf.Transform(world).GetRotation().GetQuat()
    initial = BodyPose(str(body.GetPath()), tuple(world.ExtractTranslation()), (*quat.GetImaginary(), quat.GetReal()))
    shape = read_body_shape_transforms(str(source), (initial.prim_path,))
    matrix = translate_values((initial,), 1.0, shape)[0].matrix
    for actual, expected in zip(matrix, world):
        assert actual == pytest.approx(tuple(expected), abs=1e-6)
    moved = BodyPose(initial.prim_path, (initial.translate[0], initial.translate[1], initial.translate[2] - 2), initial.orient)
    projected = translate_values((moved,), 1.0, shape)
    assert [value.prim_path for value in projected] == ["/Parent/Body", "/Parent/Body/Visual"]
    mesh_world = UsdGeom.XformCache().GetLocalToWorldTransform(mesh.GetPrim())
    mesh_world.SetTranslateOnly(mesh_world.ExtractTranslation() - Gf.Vec3d(0, 0, 2))
    for actual, expected in zip(projected[1].matrix, mesh_world):
        assert actual == pytest.approx(tuple(expected), abs=1e-6)


@pytest.mark.parametrize(
    "poses,scale,match",
    [
        ([_pose(), _pose()], 1.0, "duplicate"),
        ([BodyPose("/World/Body", (0.0, 0.0, 0.0), (1e154, 0.0, 0.0, 1.0))], 1.0, "matrix"),
        ([BodyPose("/World/Body", (0.0, 0.0, 0.0), (1e155, 0.0, 0.0, 1.0))], 1.0, "matrix"),
        ([_pose()], math.nan, "body_scale"),
    ],
)
def test_translate_values_rejects_invalid_batches(poses, scale, match) -> None:
    with pytest.raises(ValueError, match=match):
        translate_values(poses, scale)

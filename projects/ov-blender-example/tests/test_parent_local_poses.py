# SPDX-License-Identifier: Apache-2.0
"""World body poses are written in their composed parent's space."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

pxr = pytest.importorskip("pxr")

from ovrtx_blender_example.ovphysx_stage import parent_local_poses  # noqa: E402
from ovrtx_blender_example.shared_stage_composition import BodyPose  # noqa: E402

STAGE = """#usda 1.0
def Xform "World"
{
    def Xform "Moved"
    {
        double3 xformOp:translate = (3, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def Xform "Body" {}
    }
    def Xform "Scaled"
    {
        double3 xformOp:scale = (2, 2, 2)
        uniform token[] xformOpOrder = ["xformOp:scale"]
        def Xform "Body" {}
    }
    def Xform "Body" {}
}
"""


def controller(tmp_path):
    path = tmp_path / "stage.usda"
    path.write_text(STAGE)
    return SimpleNamespace(config=SimpleNamespace(input_usd_path=str(path)))


def test_identity_parent_keeps_world_pose(tmp_path):
    pose = BodyPose("/World/Body", (1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0))
    assert parent_local_poses(controller(tmp_path), (pose,)) == (pose,)


def test_translated_parent_converts_to_local(tmp_path):
    (local,) = parent_local_poses(controller(tmp_path), (BodyPose("/World/Moved/Body", (3.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0)),))
    assert local.translate == pytest.approx((0.0, 0.0, 1.0))


def test_scaled_parent_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="scaled parent"):
        parent_local_poses(controller(tmp_path), (BodyPose("/World/Scaled/Body", (0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0)),))


@pytest.mark.parametrize("parent", ["Moved", "Scaled"])
def test_public_sdk_world_pose_bypasses_legacy_parent_compensation(tmp_path, parent):
    owner = controller(tmp_path)
    owner._physics_client = SimpleNamespace(pose_write_space="world")
    pose = BodyPose(f"/World/{parent}/Body", (3.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0))
    assert parent_local_poses(owner, (pose,)) == (pose,)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example.shared_stage_composition import (
    BodyPose,
    BodyVelocity,
    RuntimeStageHost,
    pose_from_ovphysx_state,
    write_rgba_png,
)


def test_stage_host_records_only_changed_pose() -> None:
    host = RuntimeStageHost(scene_id="fixture.usda")
    pose = BodyPose("/World/PhysicsIsland/DynamicBodies/Cube_00", (0.0, 5.0, 0.0), (0.0, 0.0, 0.0, 1.0))

    first = host.publish_ovphysx_poses([pose], simulation_time_ns=0)
    second = host.publish_ovphysx_poses([pose], simulation_time_ns=10)
    third = host.publish_ovphysx_poses(
        [BodyPose("/World/PhysicsIsland/DynamicBodies/Cube_00", (0.0, 4.0, 0.0), (0.0, 0.0, 0.0, 1.0))],
        simulation_time_ns=20,
    )

    assert first.revision == 1
    assert first.dirty_paths == ("/World/PhysicsIsland/DynamicBodies/Cube_00",)
    assert second.revision == 1
    assert second.dirty_paths == ()
    assert third.revision == 2
    assert third.dirty_paths == ("/World/PhysicsIsland/DynamicBodies/Cube_00",)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prim_path": "", "linear": (0.0, 0.0, 0.0), "angular": (0.0, 0.0, 0.0)},
        {"prim_path": "/World/Body", "linear": (float("nan"), 0.0, 0.0), "angular": (0.0, 0.0, 0.0)},
        {"prim_path": "/World/Body", "linear": (0.0, 0.0), "angular": (0.0, 0.0, 0.0)},
    ],
)
def test_body_velocity_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        BodyVelocity(**kwargs)  # type: ignore[arg-type]


def test_body_velocity_defaults_angular_to_zero() -> None:
    assert BodyVelocity("/World/Body", (1.0, 2.0, 3.0)).angular == (0.0, 0.0, 0.0)


def test_pose_from_ovphysx_state_and_authoritative_read() -> None:
    state = {
        "prim_path": "/World/PhysicsIsland/DynamicBodies/Cube_00",
        "translate": {"found": True, "x": 1, "y": 2, "z": 3},
        "orient": {"found": True, "i": 0, "j": 0, "k": 0, "r": 1},
    }
    pose = pose_from_ovphysx_state(state)
    host = RuntimeStageHost(scene_id="fixture.usda")
    mutation = host.publish_ovphysx_poses([pose], simulation_time_ns=30)

    assert host.body_poses_for(mutation.dirty_paths) == (pose,)
    assert host.body_poses_for(()) == ()


@pytest.mark.parametrize(
    ("prim_path", "translate", "orient", "message"),
    [
        ("", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), "prim_path"),
        ("/World/Body", (0.0, 0.0), (0.0, 0.0, 0.0, 1.0), "three values"),
        ("/World/Body", (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), "four values"),
        ("/World/Body", (0.0, "bad", 0.0), (0.0, 0.0, 0.0, 1.0), "numeric finite"),
        ("/World/Body", (0.0, float("inf"), 0.0), (0.0, 0.0, 0.0, 1.0), "numeric finite"),
        ("/World/Body", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0), "nonzero"),
    ],
)
def test_body_pose_enforces_shared_pose_invariants(
    prim_path: str,
    translate: tuple[object, ...],
    orient: tuple[object, ...],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        BodyPose(prim_path, translate, orient)  # type: ignore[arg-type]


def test_pose_from_ovphysx_state_rejects_zero_quaternion() -> None:
    with pytest.raises(ValueError, match="nonzero"):
        pose_from_ovphysx_state({
            "prim_path": "/World/Body",
            "translate": {"found": True, "x": 0, "y": 0, "z": 0},
            "orient": {"found": True, "i": 0, "j": 0, "k": 0, "r": 0},
        })


def test_stage_host_rejects_duplicate_and_missing_pose_reads() -> None:
    host = RuntimeStageHost(scene_id="fixture.usda")
    pose = BodyPose("/World/Body", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    host.publish_ovphysx_poses([pose], simulation_time_ns=0)

    with pytest.raises(ValueError, match="unique"):
        host.body_poses_for((pose.prim_path, pose.prim_path))
    with pytest.raises(KeyError, match="/World/Missing"):
        host.body_poses_for(("/World/Missing",))


def test_write_rgba_png(tmp_path: Path) -> None:
    output = tmp_path / "frame.png"

    artifact = write_rgba_png(output, 1, 1, bytes([255, 0, 0, 255]))

    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifact["width"] == 1
    assert artifact["height"] == 1
    assert artifact["size_bytes"] == output.stat().st_size

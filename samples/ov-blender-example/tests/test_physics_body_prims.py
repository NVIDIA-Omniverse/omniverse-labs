# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example.physics_body_prims import discover_dynamic_body_prims  # noqa: E402


def test_discovers_dynamic_rigid_bodies_under_physics_island(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture.usda"
    fixture.write_text(
        """#usda 1.0
(
    defaultPrim = "World"
)

def Xform "World"
{
    def Xform "PhysicsIsland"
    {
        def Xform "DynamicBodies"
        {
            def Cube "Cube_00" (
                prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI", "PhysicsMassAPI"]
            )
            {
                double3 xformOp:translate = (0, 0, 1)
            }
            def Cube "KinematicCube" (
                prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]
            )
            {
                bool physics:kinematicEnabled = true
            }
        }
        def Cube "StaticStep" (
            prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]
        )
        {
            bool physics:kinematicEnabled = true
        }
    }
}
""",
        encoding="utf-8",
    )

    assert discover_dynamic_body_prims(str(fixture)) == ("/World/PhysicsIsland/DynamicBodies/Cube_00",)


def test_discover_dynamic_body_prims_returns_empty_for_missing_file(tmp_path: Path) -> None:
    assert discover_dynamic_body_prims(str(tmp_path / "missing.usda")) == ()

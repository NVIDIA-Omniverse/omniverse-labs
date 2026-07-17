# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest

from blender_test_support import blender_executable


ROOT = Path(__file__).resolve().parents[1]


def test_real_blender_scene_generation_contract(tmp_path: Path) -> None:
    blender = blender_executable()
    if blender is None:
        pytest.skip("Blender is unavailable")
    output = tmp_path / "result.json"
    completed = subprocess.run(
        (
            str(blender),
            "--background",
            "--factory-startup",
            "--python",
            str(ROOT / "scripts" / "run_scene_generation_contract.py"),
            "--",
            str(output),
        ),
        cwd=ROOT,
        env=dict(os.environ),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert output.is_file(), completed.stdout + completed.stderr
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "passed"
    assert result["generation_numbers"] == list(range(13))
    assert result["reused_generation"] == 0
    assert result["first_generation_retained"] is True
    assert result["first_generation_opinion_records"] == 2
    assert result["first_generation_sparse_change_replacements"] == 0
    assert result["first_generation_sparse_record"] is True
    assert result["first_generation_add_on_physics_material"] is True
    assert result["first_generation_mass_kg"] == 12.5
    assert result["first_generation_collision_invisible"] is True
    assert result["second_generation_analytic_sphere"] is True
    assert result["complete_export_modes"] == [True] + [False] * 12
    assert result["topology_delta_counts"] == list(range(13))
    assert result["recreated_object_has_new_session_uid"] is True
    assert result["recreated_mesh_has_new_session_uid"] is True
    assert result["recreated_material_has_new_session_uid"] is True
    assert result["edit_mode_preserved"] is True
    assert result["current_scene_callback_contract"] is True
    assert result["predecessor_retained_value_reuse"] is True
    assert result["retained_value_final_handoff"] is True
    assert result["first_generation_mesh_count"] == 1
    assert result["second_generation_paths"]["MESH:Sphere"]["schema_path"].endswith(
        "/Sphere"
    )
    assert "MESH:Cube" not in result["third_generation_paths"]
    assert [item["count"] for item in result["point_light_generations"]] == [
        2,
        3,
        4,
        5,
        6,
        6,
        5,
    ]
    assert all(result["point_light_identities"].values())
    assert result["parented_light_hierarchy"] is True
    assert result["parented_light_followed_parent"] is True
    assert result["deleted_light_inactive"] is True
    assert result["final_complete_export"] == {
        "complete_export": True,
        "light_count": 5,
        "work_directory_exists_after_close": False,
    }
    assert result["blocked_callbacks_before_undo"] == 2
    assert result["blocked_affected_before_undo"]
    assert result["undo_reused_accepted_generation"] is True
    assert result["blocked_after_undo"] == {}
    assert result["point_light_runtime"]["status"] in {
        "not_requested",
        "pass-real",
    }
    if result["point_light_runtime"]["status"] == "pass-real":
        activations = result["point_light_runtime"]["activations"]
        assert [item["operation"] for item in activations] == [
            "create",
            "copy_paste",
            "unlinked_duplicate",
            "linked_duplicate",
            "delete",
        ]
        assert [item["predecessor_generation"] for item in activations] == list(
            range(5)
        )
        assert [item["activated_generation"] for item in activations] == list(
            range(1, 6)
        )
        assert [item["light_count"] for item in activations] == [2, 3, 4, 5, 4]
        assert all(item["candidate_accepted"] for item in activations)
        assert result["point_light_runtime"]["deleted_predecessor_inactive"] is True
    assert result["closed_work_directory_exists"] is False

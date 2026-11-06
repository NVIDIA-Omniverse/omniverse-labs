# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-Blender invalidation matrix for the physics-free viewport reuse path.

One Blender run (``blender_physics_viewport_invalidation.py``) exercises every
lifecycle path; each case below asserts one row. Known gaps are strict xfails
tagged with their gap ID, so fixing a gap turns its row into an XPASS failure
until the mark is removed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest

from blender_test_support import blender_executable

FIXTURE = Path(__file__).with_name("blender_physics_viewport_invalidation.py")


def _gap(gap: str, reason: str):
    return pytest.mark.xfail(strict=True, reason=f"{gap}: {reason}")


# Every gap found by n-02-01 (GAP-R2, GAP-R2-ANIM, GAP-NOTIFY, GAP-R3, GAP-R5,
# GAP-R7) is repaired by n-02-02; no row is expected to fail. The R2 rows
# (C01-C03, S02) have no Blender hook and pass as latency-bounded rows: the
# idle discovery timer must detect them within its worst-case tick bound.
# Keep ``_gap`` for any future known gap.

CASES = [
    "E01-object-scalar-setattr",
    "E02-object-vector-setattr",
    "E03-object-vector-element",
    "E04-object-pointer-setattr",
    "E05-object-enum-setattr",
    "E06-material-scalar-setattr",
    "E07-unused-material-scalar",
    "E08-posebone-scalar-setattr",
    "E09-posebone-vector-setattr",
    "E10-posebone-pointer-setattr",
    "E11-bone-identity-setattr",
    "E12-scene-control-scalar",
    "E13-scene-control-vector",
    "E14-scene-indicator-redraw",
    "E15-scene-namespace-string",
    "E16-unit-scale",
    "E20-latch-only-object-setattr",
    "E21-latch-only-material-setattr",
    "E22-latch-only-posebone-setattr",
    "E17-operator-wm-context-set",
    "E18-operator-add-body",
    "E19-operator-scene-opinion",
    "R01-unset-last-participant",
    "R02-unset-scene-control",
    "R03-unset-material",
    "R04-reset-to-default-value",
    "R05-reset-default-button-operator",
    "C01-raw-collider-add-nonparticipant",
    "C02-raw-joint-axes-add-nonparticipant",
    "C03-raw-posebone-collider-add",
    "C04-raw-add-then-configure",
    "C05-raw-remove-participant",
    "C06-raw-clear-participant",
    "C07-operator-collider-remove",
    "C09-raw-add-with-mark-dirty-notification",
    "C08-initialize-joint-notifies",
    "R1a-material-slot-assign-data",
    "R1b-material-slot-assign-object-link",
    "R1c-material-slot-add-operator",
    "R3a-link-raw-participant-during-kinematics",
    "R3b-link-raw-participant-normal",
    "R5-empty-depsgraph-updates",
    "S01-shared-object-other-scene-setattr",
    "S02-shared-object-other-scene-raw-add",
    "S03-shared-material-two-objects",
    "A01-fcurve-on-nonparticipant",
    "A02-driver-on-nonparticipant",
    "A03-frame-change-animated-participant",
    "L01-link-authored-library-object",
    "L02-library-reload-adds-physics",
    "L03-library-override-setattr",
    "X01-explicit-restart-after-raw-edit",
    "FR01-final-render-admission-after-raw-edit",
    "U01-undo-restores-first-participant",
    "U02-redo-removes-participant",
    "R7a-undo-schedule-after-session-close",
    "R7b-history-handler-noop-when-accepted",
    "R7c-history-handler-noop-when-viewport-built",
    "U03-undo-physics-free-edit",
    "F01-open-file",
    "F02-revert-file",
    "F03-read-homefile",
    "F04-addon-reload-with-raw-edit",
    "F05-R4-load-during-pending-close",
]


@pytest.fixture(scope="module")
def matrix(tmp_path_factory):
    blender = blender_executable()
    if blender is None:
        pytest.skip("set BLENDER_COMMAND to run the Blender invalidation matrix")
    out = tmp_path_factory.mktemp("physics-viewport-invalidation")
    result = subprocess.run(
        [str(blender), "--background", "--factory-startup", "--python-exit-code", "17",
         "--python", str(FIXTURE), "--", str(out)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stdout[-8000:]
    assert "PHYSICS_VIEWPORT_INVALIDATION_DONE" in result.stdout, result.stdout[-8000:]
    results = json.loads((out / "results.json").read_text())
    return {"cases": {case["id"]: case for case in results["cases"]},
            "close_failures": results["close_failures"]}


LIFECYCLE_PREFIXES = ("U0", "R7", "F0")


def test_scene_close_never_failed(matrix):
    # GAP-CLOSE guard (fixed by n-02-06): owner cleanup retries transient
    # Windows sharing errors (WinError 5/32/145) and defers a residual work
    # directory, so no session close may end 'failed'. Lifecycle rows after a
    # failed close are skipped below, so this guard is what fails the run.
    assert not matrix["close_failures"], matrix["close_failures"]


def test_matrix_rows_are_all_declared(matrix):
    declared = {getattr(case, "values", (case,))[0] for case in CASES}
    assert set(matrix["cases"]) == declared


@pytest.mark.parametrize("case_id", CASES)
def test_viewport_reuse_observes_change(matrix, case_id):
    case = matrix["cases"][case_id]
    if matrix["close_failures"] and case_id.startswith(LIFECYCLE_PREFIXES):
        pytest.skip(f"scene close failed earlier in the run (GAP-CLOSE): {matrix['close_failures'][0]}")
    assert case["ok"], json.dumps(case, indent=2, default=str)[:4000]


@pytest.mark.parametrize("count", [1000, 10000])
def test_discovery_latency_and_participant_cost(tmp_path, count):
    """R2/R6 measurements; slow (10k builds for minutes), so opt-in."""
    if os.environ.get("OVRTX_PHYSICS_INVALIDATION_PERF") != "1":
        pytest.skip("set OVRTX_PHYSICS_INVALIDATION_PERF=1 to measure R2/R6")
    blender = blender_executable()
    if blender is None:
        pytest.skip("set BLENDER_COMMAND to run the Blender invalidation measurements")
    result = subprocess.run(
        [str(blender), "--background", "--factory-startup", "--python-exit-code", "17",
         "--python", str(FIXTURE), "--", str(tmp_path), "--perf", str(count)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=1800,
    )
    assert result.returncode == 0, result.stdout[-8000:]
    perf = json.loads((tmp_path / f"perf-{count}.json").read_text())
    r2 = perf["R2_worst_case"]
    assert r2["detected"] and r2["restart_queued"]
    # Hard bound: one pass at MIN_BATCH Objects per tick (+1 for the cursor position).
    assert r2["ticks_to_detection"] <= r2["hard_bound_ticks"] + 1
    # Time-budgeted passes (n-02-02): about 1,000 Objects per 4 ms tick on the
    # reference host, so 10k Objects are covered in ~12 ticks (measured 3.3 s
    # worst case). Generous host margin: 5 s.
    assert r2["latency_s_at_0.2s_interval"] <= 5.0
    r3 = perf["R3_unobserved_link"]
    assert r3["detected"] and r3["ticks_to_detection"] <= 2 * r2["hard_bound_ticks"] + 1
    frames = perf["frame_change_handler"]
    assert not frames["latched"]
    assert frames["all_objects_animated"]["median_ms"] < 5.0  # 1 ms probe budget per frame

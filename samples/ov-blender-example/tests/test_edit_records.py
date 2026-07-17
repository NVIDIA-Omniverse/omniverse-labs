# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import edit_records  # noqa: E402


def test_selection_observation_record_keeps_unmapped_selection_inspection_only() -> None:
    selection_resolution = {
        "status": "unsupported_selection_group",
        "group_rejected": True,
        "unresolved_reasons": ["unmapped_selection_source"],
        "sources": [
            {
                "source_name": "LooseCube",
                "status": "unresolved",
                "owner_category": "inspection_only",
                "preview_only": True,
            }
        ],
    }

    record = edit_records.selection_observation_record(
        edit_id="edit-000001",
        timestamp_ns=123,
        selection_resolution=selection_resolution,
    )

    assert record["schema_version"] == edit_records.SCHEMA_VERSION
    assert record["artifact_id"] == edit_records.ARTIFACT_ID
    assert record["timestamp_ns"] == 123
    assert record["action"] == "observation"
    assert record["accepted"] is False
    assert record["result"] == "unsupported"
    assert record["fail_reason"] == "unmapped_selection_source"
    assert record["mechanism"] == "none"
    assert record["persistence"] == "none"
    assert record["values_written"] is False
    assert record["rendered_effect_observed"] is False
    assert record["operator_workflow_observed"] is False
    assert record["selection_resolution"]["sources"][0]["owner_category"] == "inspection_only"

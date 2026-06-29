# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ENGINE = Path(__file__).resolve().parents[1] / "engine"
sys.path.insert(0, str(ENGINE))

from session import Session


def _write_anchor(path: Path) -> None:
    sketch = {
        "schemaVersion": 1,
        "seed": 7,
        "assetPack": "",
        "templateZones": [
            {
                "id": "zone_a",
                "purpose": "storage",
                "allowedArchetypes": ["thing_a", "thing_b"],
                "boundsM": {"widthM": 10.0, "depthM": 10.0, "heightM": 8.0},
                "originWorldM": [10.0, 20.0, 0.0],
            }
        ],
        "tree": {
            "type": "site",
            "id": "fixture",
            "children": [
                {
                    "type": "zone",
                    "id": "zone_a",
                    "zoneType": "storage",
                    "transform": {"translateM": [10.0, 20.0, 0.0], "yawDeg": 0.0},
                    "boundsM": {"widthM": 10.0, "depthM": 10.0, "heightM": 8.0},
                    "children": [
                        {
                            "type": "placement",
                            "id": "thing_a_01",
                            "archetype": "thing_a",
                            "parentZoneId": "zone_a",
                            "slotM": {"widthM": 1.0, "depthM": 1.0, "heightM": 1.0},
                            "transform": {"translateM": [1.0, 1.0, 0.5], "yawDeg": 0.0},
                        }
                    ],
                }
            ],
        },
    }
    path.write_text(json.dumps(sketch, indent=2), encoding="utf-8")


def _placements(sketch: dict) -> list[dict]:
    out: list[dict] = []

    def walk(node: dict) -> None:
        if node.get("type") == "placement":
            out.append(node)
        for child in node.get("children", []):
            walk(child)

    walk(sketch["tree"])
    return out


class SessionZoneSemanticsTests(unittest.TestCase):
    def test_snapshot_preserves_parent_zone_for_loaded_and_new_placements(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            anchor = root / "anchor.sketch.json"
            _write_anchor(anchor)

            session = Session(anchor, None, root / "run")
            placed = session.place(
                "thing_b",
                [13.0, 23.0, 0.5],
                [1.0, 1.0, 1.0],
                id="thing_b_01",
                parentZoneId="zone_a",
            )
            self.assertEqual(placed, {"ok": True, "id": "thing_b_01"})
            by_id_live = {p["id"]: p for p in session.query_stage_graph()}
            self.assertEqual(by_id_live["thing_a_01"]["posM"], [11.0, 21.0, 0.5])

            snapshot = session.snapshot_to_sketch(root / "snapshot.sketch.json")
            sketch = json.loads(snapshot.read_text(encoding="utf-8"))
            by_id = {p["id"]: p for p in _placements(sketch)}

            self.assertEqual(by_id["thing_a_01"]["parentZoneId"], "zone_a")
            self.assertEqual(by_id["thing_b_01"]["parentZoneId"], "zone_a")
            self.assertEqual(by_id["thing_a_01"]["transform"]["translateM"], [1.0, 1.0, 0.5])
            self.assertEqual(by_id["thing_b_01"]["transform"]["translateM"], [3.0, 3.0, 0.5])

    def test_query_zones_merges_template_metadata_by_exact_zone_id(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            anchor = root / "anchor.sketch.json"
            _write_anchor(anchor)

            session = Session(anchor, None, root / "run")
            zones = session.query_zones()

            self.assertEqual(zones, [
                {
                    "id": "zone_a",
                    "boundsM": {"widthM": 10.0, "depthM": 10.0, "heightM": 8.0},
                    "originWorldM": [10.0, 20.0, 0.0],
                    "type": "storage",
                    "purpose": "storage",
                    "allowedArchetypes": ["thing_a", "thing_b"],
                }
            ])


if __name__ == "__main__":
    unittest.main()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pxr import Usd


ENGINE = Path(__file__).resolve().parents[1] / "engine"
sys.path.insert(0, str(ENGINE))

from sketch_realize import _create_crate_layer, realize


class SketchRealizeSemanticsTests(unittest.TestCase):
    def test_realize_writes_binary_usdc_for_usd_extension_by_default(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pack = root / "pack"
            pack.mkdir()
            (pack / "pack.json").write_text(
                json.dumps({"metersPerUnit": 1.0, "upAxis": "Z", "archetypes": {}}),
                encoding="utf-8",
            )
            sketch_path = root / "binary.sketch.json"
            sketch_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "seed": 1,
                        "assetPack": str(pack),
                        "defaults": {"defaultOnMiss": "synth"},
                        "tree": {
                            "type": "site",
                            "id": "binary_fixture",
                            "children": [
                                {
                                    "type": "placement",
                                    "id": "box_01",
                                    "archetype": "box",
                                    "slotM": {"widthM": 1.0, "depthM": 1.0, "heightM": 1.0},
                                    "transform": {"translateM": [0.0, 0.0, 0.0], "yawDeg": 0.0},
                                }
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )

            realize(sketch_path, root / "out")

            self.assertEqual((root / "out" / "root.usd").read_bytes()[:8], b"PXR-USDC")
            self.assertTrue(_create_crate_layer().identifier.endswith(".usdc"))

    def test_realize_can_skip_usdcore_composed_count_for_large_benchmarks(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pack = root / "pack"
            pack.mkdir()
            (pack / "pack.json").write_text(
                json.dumps({"metersPerUnit": 1.0, "upAxis": "Z", "archetypes": {}}),
                encoding="utf-8",
            )
            sketch_path = root / "skip_count.sketch.json"
            sketch_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "seed": 1,
                        "assetPack": str(pack),
                        "defaults": {"defaultOnMiss": "synth"},
                        "tree": {
                            "type": "site",
                            "id": "skip_count_fixture",
                            "children": [
                                {
                                    "type": "placement",
                                    "id": "box_01",
                                    "archetype": "box",
                                    "slotM": {"widthM": 1.0, "depthM": 1.0, "heightM": 1.0},
                                    "transform": {"translateM": [0.0, 0.0, 0.0], "yawDeg": 0.0},
                                }
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"SKETCH_STAGE_SKIP_USDCORE_COUNT": "1"}):
                manifest = realize(sketch_path, root / "out")

            self.assertIsNone(manifest["composedPrimCount_usdcore"])
            self.assertEqual(manifest["timings"]["count_s"], 0.0)

    def test_realize_authors_semantics_for_synth_placements(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pack = root / "pack"
            pack.mkdir()
            (pack / "pack.json").write_text(
                json.dumps({"metersPerUnit": 1.0, "upAxis": "Z", "archetypes": {}}),
                encoding="utf-8",
            )
            sketch_path = root / "semantic.sketch.json"
            sketch_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "seed": 1,
                        "assetPack": str(pack),
                        "defaults": {"defaultOnMiss": "synth"},
                        "tree": {
                            "type": "site",
                            "id": "semantic_fixture",
                            "children": [
                                {
                                    "type": "zone",
                                    "id": "storage_main",
                                    "zoneType": "storage",
                                    "transform": {"translateM": [0.0, 0.0, 0.0], "yawDeg": 0.0},
                                    "boundsM": {"widthM": 5.0, "depthM": 5.0, "heightM": 4.0},
                                    "children": [
                                        {
                                            "type": "placement",
                                            "id": "rack_01",
                                            "archetype": "rack",
                                            "slotM": {"widthM": 1.0, "depthM": 1.0, "heightM": 1.0},
                                            "transform": {"translateM": [1.0, 1.0, 0.0], "yawDeg": 0.0},
                                        }
                                    ],
                                }
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )

            manifest = realize(sketch_path, root / "out")

            self.assertEqual(manifest["summary"]["filled"], 1)
            stage = Usd.Stage.Open(str(root / "out" / "root.usd"))
            prim = stage.GetPrimAtPath("/World/storage_main/rack_01")
            api_schemas = prim.GetMetadata("apiSchemas")
            self.assertIn("SemanticsAPI", api_schemas.explicitItems)
            self.assertEqual(
                prim.GetAttribute("semantics:Semantics:params:semanticType").Get(),
                "class",
            )
            self.assertEqual(
                prim.GetAttribute("semantics:Semantics:params:semanticData").Get(),
                "rack",
            )


if __name__ == "__main__":
    unittest.main()

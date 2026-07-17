#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from argparse import ArgumentParser
import json
from pathlib import Path
import sys

fixtures = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(fixtures / "prep"))
from download_stair_drop_pbr_assets import prepare_assets

parser = ArgumentParser()
parser.add_argument("--force", action="store_true")
args = parser.parse_args()
spec = json.loads(Path(__file__).with_name("spec.json").read_text(encoding="utf-8"))
archives = {
    item["id"]: item["archive_sha256"]
    for source in spec["provenance"]["sources"]
    for item in source.get("assets", [])
}
prepare_assets(fixtures / "data" / spec["id"], force=args.force, expected_archives=archives)

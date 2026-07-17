#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from argparse import ArgumentParser
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "prep"))
from download_fixtures import prepare_spec

parser = ArgumentParser()
parser.add_argument("--force", action="store_true")
args = parser.parse_args()
prepare_spec(Path(__file__).with_name("spec.json"), force=args.force)

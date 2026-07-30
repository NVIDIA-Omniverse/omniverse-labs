#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility entry point for ovfmi's isolated USD schema parser."""

import runpy
from pathlib import Path

_PARSER = (
    Path(__file__).resolve().parent.parent
    / "python"
    / "_parse_fmi_schema.py"
)


if __name__ == "__main__":
    runpy.run_path(str(_PARSER), run_name="__main__")

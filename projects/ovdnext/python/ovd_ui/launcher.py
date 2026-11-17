# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Installed command entry point; the source launcher remains available."""
import os
from pathlib import Path
import runpy


def main():
    if "__GLX_VENDOR_LIBRARY_NAME" not in os.environ and Path("/proc/driver/nvidia/gpus").is_dir():
        os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
    runpy.run_module("ovd_ui", run_name="__main__")

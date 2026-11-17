# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve prebuilt native test writers supplied by the regression harness."""
import os
from pathlib import Path
import subprocess


def executable(name, variable):
    configured = os.environ.get(variable)
    if configured:
        path = Path(configured)
        if path.is_file():
            return path
        raise FileNotFoundError(f"Missing configured {name}: {path}")
    root_value = os.environ.get("PVD_FIXTURE_ROOT")
    if root_value:
        root = Path(root_value)
        paths = (root / name, root / f"{name}.exe")
    else:
        import _pvd_native
        native = Path(_pvd_native.__file__).resolve()
        build = next((parent for parent in native.parents if (parent / "CMakeCache.txt").is_file()), None)
        if build is None:
            raise FileNotFoundError(f"Set PVD_FIXTURE_ROOT or run ci/test_ovui.py for {name}")
        config = native.parent.name if native.parent.name in ("Debug", "Release", "RelWithDebInfo") else None
        command = ["cmake", "--build", str(build), "--target", name, "--parallel", "2"]
        if config:
            command += ["--config", config]
        subprocess.run(command, check=True, capture_output=True, timeout=120)
        root = build / "src/PVDSession"
        roots = (root / config, root) if config else (root,)
        paths = tuple(path for directory in roots for path in
                      (directory / name, directory / f"{name}.exe"))
    for path in paths:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing {name} in {root}")

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Launch the existing OVUI source app with process-local native DLL paths.

This is a developer trial, not a redistributable Windows package or a license
clearance. Build _pvd_native and install the documented dependencies first.
"""
import argparse
from contextlib import ExitStack
import importlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import runpy
import sys
import sysconfig


def main(argv=None):
    parser = argparse.ArgumentParser(description="OvdNext: Windows OVUI source launcher")
    parser.add_argument("--native-root", required=True, type=Path,
                        help="Directory containing the built _pvd_native CPython 3.12 .pyd")
    parser.add_argument("--cuda-bin", type=Path,
                        default=Path(sysconfig.get_path("platlib")) / "nvidia/cuda_runtime/bin",
                        help="CUDA 12 runtime DLL directory; defaults to the venv runtime package")
    parser.add_argument("--check", action="store_true",
                        help="Verify native/application imports without opening a window")
    parser.add_argument("application_args", nargs=argparse.REMAINDER,
                        help="Arguments after -- go to the existing application")
    args = parser.parse_args(argv)
    if sys.platform != "win32":
        parser.error("Use ci/run_ovui.sh on Linux; this launcher uses the Windows DLL loader")
    native_root, cuda_bin = (path.resolve() for path in (args.native_root, args.cuda_bin))
    if not any(native_root.glob("_pvd_native*.pyd")):
        parser.error(f"No built _pvd_native .pyd in {native_root}")
    for directory, filename in ((cuda_bin, "cudart64_12.dll"),):
        if not (directory / filename).is_file():
            parser.error(f"Missing {directory / filename}; see the Windows local-trial instructions")

    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root / "python"), str(native_root)]
    from ovd_ui.runtime_security import configure_runtime_security

    configure_runtime_security()
    # No inspector endpoint unless the operator explicitly opts in for QA.
    os.environ.setdefault("OVUIINSPECT_ENABLED", "0")
    with ExitStack() as directories:
        for directory in (cuda_bin,):
            directories.enter_context(os.add_dll_directory(str(directory)))
        try:
            native = importlib.import_module("_pvd_native")
            importlib.import_module("ovd_ui.application")
        except (ImportError, OSError) as error:
            parser.error(f"OvdNext runtime import failed: {error}")
        if args.check:
            print(json.dumps({"application_imported": True, "ovui_version": version("ovui"),
                              "native_module": native.__file__}))
            return 0
        application_args = args.application_args
        if application_args[:1] == ["--"]:
            application_args = application_args[1:]
        # Opening a known file avoids automatically starting a live listener.
        if not application_args:
            application_args = [str(root / "test/data/test_scene.ovd")]
        sys.argv = ["ovd_ui", *application_args]
        runpy.run_module("ovd_ui", run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

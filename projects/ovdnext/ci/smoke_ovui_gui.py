# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the existing isolated Display/renderer regression against a portable bundle."""
import argparse
import ctypes
import os
from pathlib import Path
import runpy
import sys
import tempfile


def ci_graphics(library):
    """Preload CI Mesa for GLFW and ModernGL without modifying the ZIP/runtime."""
    library = library.resolve(strict=True)
    handle = ctypes.WinDLL(str(library), winmode=8)  # DLL dependencies beside Mesa.
    os.environ["GLCONTEXT_WIN_LIBGL"] = str(library)
    return handle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--screenshot", required=True, type=Path)
    parser.add_argument("--opengl-library", type=Path, help="Windows CI-only Mesa DLL, outside the bundle")
    args = parser.parse_args()
    screenshot = args.screenshot.resolve()
    root = Path(__file__).resolve().parents[1]
    graphics = ci_graphics(args.opengl_library) if args.opengl_library else None
    # The real launcher verifies isolated Python and every runtime import path.
    sys.argv = ["OvdNext", "--check", str(root / "test/data/multi_geom.ovd")]
    runpy.run_path(str(args.bundle.resolve() / "portable_launch.py"), run_name="__main__")
    with tempfile.TemporaryDirectory(prefix="ovdnext-gui-") as preferences:
        os.environ["XDG_CONFIG_HOME"] = preferences
        os.environ["XDG_DATA_HOME"] = preferences
        os.environ["XDG_CACHE_HOME"] = preferences
        os.environ["LOCALAPPDATA"] = preferences
        regression = runpy.run_path(str(root / "python/tests/display_panel_gui.py"))
        regression["main"](str(screenshot))
    if not screenshot.is_file():
        raise RuntimeError("GUI smoke produced no screenshot")
    print("Extracted-package GUI rendering and Display controls passed.")


if __name__ == "__main__":
    main()

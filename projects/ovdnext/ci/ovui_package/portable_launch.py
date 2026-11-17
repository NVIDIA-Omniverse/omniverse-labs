# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Relative entry point; run with the bundled Python's -I -B options."""
from contextlib import ExitStack
import ctypes
import importlib
import json
import locale
import os
from pathlib import Path
import runpy
import sys
import tempfile


def preflight_graphics_driver():
    if sys.platform == "win32":
        library = "vulkan-1.dll"
        try:
            ctypes.WinDLL(library, winmode=0x00000800)
        except OSError as error:
            raise RuntimeError(
                f"Required Vulkan loader {library} is unavailable. Install or update "
                "the current 64-bit graphics driver from your GPU vendor, then start "
                "OvdNext again."
            ) from error
    elif sys.platform == "linux":
        library = "libcuda.so.1"
        try:
            ctypes.CDLL(library)
        except OSError as error:
            raise RuntimeError(
                f"Required NVIDIA driver library {library} is unavailable. Install or "
                "update the current NVIDIA driver, then start OvdNext again."
            ) from error


def main():
    root = Path(__file__).resolve().parent
    if not sys.flags.isolated or not Path(sys.executable).resolve().is_relative_to(root / "python"):
        raise RuntimeError("Use the bundled OvdNext launcher")
    sys.path[:] = [str(root / "app"), *[path for path in sys.path if Path(path).resolve().is_relative_to(root / "python")]]
    from ovd_ui.runtime_security import configure_runtime_security

    configure_runtime_security()
    os.environ["OVUIINSPECT_ENABLED"] = "0"
    for key in ("CUDA_PATH", "CUDAToolkit_ROOT", "CUDA_HOME"):
        os.environ.pop(key, None)
    if sys.platform == "win32":
        # OvUI's native filesystem paths use the CRT character encoding.
        locale.setlocale(locale.LC_CTYPE, ".UTF8")
        system = os.environ.get("SystemRoot", r"C:\Windows")
        os.environ["PATH"] = system + r"\System32;" + system
    else:
        os.environ["PATH"] = "/usr/bin:/bin"
    preflight_graphics_driver()
    with ExitStack() as dlls:
        if sys.platform == "win32":
            for directory in (root / "lib", root / "python", root / "app/omni/ui", root / "app/omni/ui_scene"):
                if directory.is_dir():
                    dlls.enter_context(os.add_dll_directory(str(directory)))
        if sys.argv[1:2] != ["--check"]:
            sys.argv[0] = "ovd_ui"
            runpy.run_module("ovd_ui", run_name="__main__")
            return
        modules = {}
        for name in ("_pvd_native", "ovd_ui", "ovd_ui.application", "ovd_ui.renderer", "omni.ui",
                     "ovui_widgets.common.style", "ovui_data_adapters.services", "numpy", "moderngl", "PIL"):
            module = importlib.import_module(name)
            path = Path(module.__file__).resolve()
            if not path.is_relative_to(root / "app"):
                raise RuntimeError(f"Import escaped bundle: {name}: {path}")
            modules[name] = path.relative_to(root).as_posix()
        import ovd_ui
        result = {"version": ovd_ui.__version__, "isolated": bool(sys.flags.isolated), "modules": modules,
                  "native_provenance": json.loads((root / "app/ovd_ui/runtime-build.json").read_text())}
        if len(sys.argv) > 2:
            from ovd_ui.ingestion import load_capture
            with tempfile.TemporaryDirectory(prefix="ovdnext-probe-") as temp:
                opened = load_capture(Path(sys.argv[2]).resolve(), temporary_directory=temp)
                try:
                    session = opened.session
                    scenes = session.scenes()
                    if not scenes or not session.seek(scenes[0].id, 0):
                        raise RuntimeError("Capture has no seekable scene")
                    presentation = session.presentation(scenes[0].id, 0)
                    result["capture"] = {"scenes": len(scenes), "instances": session.info().instance_count,
                                         "presentation": type(presentation).__name__}
                finally:
                    opened.close()
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run source-tree OVUI tests against an explicit native build and fixtures."""
import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


FIXTURES = {
    "PVD_GEOMETRY_FIXTURE": "pvd_geometry_fixture",
    "PVD_JOINT_FIXTURE": "pvd_geometry_fixture",
    "PVD_INGESTION_FIXTURE": "pvd_ingestion_fixture",
    "PVD_TOLERANCE_FIXTURE": "pvd_tolerance_fixture",
    "PVD_AVAILABILITY_FIXTURE": "pvd_availability_fixture",
    "PVD_JOINT_CALIBRATION_WRITER": "pvd_joint_calibration_fixture",
}


def _pytest_arguments(arguments, root):
    defaults = [str(root / "python/tests"), *(str(root / f"ci/tests/{name}") for name in
        ("test_ovui_build.py", "test_ovui_python.py", "test_ovui_expat.py", "test_ovui_pillow.py", "test_ovui_pillow_probe.py", "test_ovui_package.py", "test_ovui_portable.py",
         "test_ovui_licenses.py", "test_ovui_release.py", "test_ovui_version.py", "test_ovui_workflow.py", "test_ovui_launcher.py", "test_ovui_runtime_security.py",
         "test_ovui_gui_smoke.py", "test_ovui_package_verification.py", "test_test_ovui.py", "test_tcp_selection_sender.py"))]
    return [*defaults, *arguments] if not arguments or arguments[0].startswith("-") else arguments


def _fixture(root, name):
    for candidate in (root / name, root / f"{name}.exe"):
        if candidate.is_file():
            return candidate
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the OVUI native regression suite")
    parser.add_argument("--native-root", required=True, type=Path)
    parser.add_argument("--fixture-root", required=True, type=Path)
    parser.add_argument("--cuda-bin", type=Path)
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER,
                        help="pytest arguments after --; defaults to the complete suite")
    args = parser.parse_args(argv)
    native_root, fixture_root = args.native_root.resolve(), args.fixture_root.resolve()
    native_suffix = ".pyd" if sys.platform == "win32" else ".so"
    if not any(native_root.glob(f"_pvd_native*{native_suffix}")):
        parser.error(f"No built _pvd_native{native_suffix} in {native_root}")
    resolved = {}
    for variable, name in FIXTURES.items():
        executable = _fixture(fixture_root, name)
        if executable is None:
            parser.error(f"Missing {name} in {fixture_root}")
        resolved[variable] = str(executable.resolve())

    dll_paths = []
    for label, directory in (("CUDA", args.cuda_bin),):
        if directory is not None:
            directory = directory.resolve()
            if not directory.is_dir():
                parser.error(f"Missing {label} DLL directory {directory}")
            dll_paths.append(directory)
    if sys.platform == "win32":
        if mesa := os.environ.get("GLCONTEXT_WIN_LIBGL"):
            mesa = Path(mesa).resolve()
            if not mesa.is_file():
                parser.error(f"Missing configured Mesa OpenGL library {mesa}")
            dll_paths.append(mesa.parent)
        import ctypes
        ctypes.windll.kernel32.SetErrorMode(3)

    root = Path(__file__).resolve().parents[1]
    python_paths = [native_root, root / "python"]
    sys.path[:0] = [str(path) for path in python_paths]
    os.environ["OVDNEXT_TEST_NATIVE_ROOT"] = str(native_root)
    os.environ["PVD_FIXTURE_ROOT"] = str(fixture_root)
    os.environ.update(resolved)
    child_paths = [root / "ci/ovui_test_bootstrap", *python_paths]
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [*(str(path) for path in child_paths), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    if dll_paths:
        os.environ["OVUI_TEST_DLL_DIRECTORIES"] = os.pathsep.join(map(str, dll_paths))
        os.environ["PATH"] = os.pathsep.join(
            [*(str(path) for path in dll_paths), os.environ.get("PATH", "")])

    pytest_args = args.pytest_args
    if pytest_args[:1] == ["--"]:
        pytest_args = pytest_args[1:]
    pytest_args = _pytest_arguments(pytest_args, root)
    import pytest
    with ExitStack() as handles:
        preferences = Path(handles.enter_context(TemporaryDirectory(prefix="ovui-test-settings-")))
        local_data, xdg_data = preferences / "local", preferences / "xdg"
        local_data.mkdir(); xdg_data.mkdir()
        os.environ.update(LOCALAPPDATA=str(local_data), XDG_DATA_HOME=str(xdg_data))
        if sys.platform == "win32":
            for directory in dll_paths:
                handles.enter_context(os.add_dll_directory(str(directory)))
        return int(pytest.main(["-rfEs", *pytest_args]))


if __name__ == "__main__":
    raise SystemExit(main())

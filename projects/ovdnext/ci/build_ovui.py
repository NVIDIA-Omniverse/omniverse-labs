# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build, test and package OvUI using a prepared pinned CPython 3.12 environment."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "RelWithDebInfo"


def configure_command(build, python, physics, *, windows):
    return ["cmake", "-S", str(ROOT), "-B", str(build), "-G",
            "Visual Studio 17 2022" if windows else "Ninja",
            *(["-A", "x64"] if windows else []),
            f"-DCMAKE_BUILD_TYPE={CONFIG}", "-DCMAKE_DISABLE_FIND_PACKAGE_Qt5=ON",
            f"-DPython_EXECUTABLE={python.as_posix()}",
            f"-DPVDDOM_ROOT={(physics / 'physx/pvddom').as_posix()}",
            f"-DPVDRUNTIME_ROOT={(physics / 'physx/pvdruntime').as_posix()}",
            "-DPVD3_BUILD_PYTHON=ON",
            "-DPVD3_BUILD_TESTS=ON", "-DPVD3_REQUIRE_OVD_SUPPORT=ON"]


def native_targets():
    return ["_pvd_native", "pvdcore_tests", "checked_output_tests", "ovd_capture_control_tests", "ovd_capture_catalog_tests", "ovddata_tests", "ovddata_ingest_tests",
            "ovddata_omnipvd_tests", "ovdnext_data_probe_tests", *["pvd_" + name for name in (
        "geometry_fixture", "ingestion_fixture", "tolerance_fixture", "availability_fixture",
        "joint_calibration_fixture")]]


def application_environment(source):
    return {key: value for key, value in source.items() if key.upper() not in
           {"QT5_DIR", "QTDIR", "QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH"}}


def msvc_runtime(redist_root):
    candidates = [path for path in Path(redist_root).glob("*/x64/Microsoft.VC*.CRT")
                  if all((path / name).is_file() for name in
                         ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"))
                  and all(part.isdigit() for part in path.parents[1].name.split("."))]
    if not candidates:
        raise ValueError("No complete x64 MSVC CRT redistributable")
    return max(candidates, key=lambda path: tuple(map(int, path.parents[1].name.split("."))))


def run(command, *, env, check=True, capture=False):
    return subprocess.run(list(map(str, command)), cwd=ROOT, env=env, check=check,
                          text=True, stdout=subprocess.PIPE if capture else None)


def extracted_probe(package, destination, capture, env):
    """Extract only our freshly built archive to a new path, then use its launcher."""
    destination.mkdir(parents=True, exist_ok=False)
    report = json.loads((package / "package-report.json").read_text())
    archive = package / report["archive"]
    if sys.platform == "win32":
        with zipfile.ZipFile(archive) as source:
            source.extractall(destination)
        launcher = destination / "OvdNext-windows-x86_64/OvdNext.cmd"
    else:
        # Match the documented customer extraction; ZipFile drops POSIX modes.
        subprocess.run(["unzip", "-q", str(archive), "-d", str(destination)], check=True)
        launcher = destination / "OvdNext-linux-x86_64/OvdNext.sh"
    clean = {key: value for key, value in env.items() if not key.startswith("PYTHON")
             and key not in {"LD_LIBRARY_PATH", "VIRTUAL_ENV", "GLCONTEXT_WIN_LIBGL", "GALLIUM_DRIVER"}}
    clean["PATH"] = str(Path(os.environ["SystemRoot"]) / "System32") if sys.platform == "win32" else "/usr/bin:/bin"
    result = subprocess.run([str(launcher), "--check", str(capture)], cwd=destination,
                            env=clean, check=True, text=True, capture_output=True)
    probe = json.loads(result.stdout)
    if probe["version"] != report["version"] or not probe.get("capture") or not probe["isolated"]:
        raise ValueError("Extracted-package capture probe did not verify the release")
    (package / "extracted-probe.json").write_text(json.dumps(probe, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("work-root", "physics-root", "ovui-root"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--msvc-runtime", type=Path)
    parser.add_argument("--gui-smoke", action="store_true", help="Require an actual desktop/OpenGL renderer")
    parser.add_argument("--gui-opengl-library", type=Path, help="Windows CI-only Mesa DLL; never packaged")
    parser.add_argument("--allow-dirty", action="store_true", help="Local development artifacts only")
    args = parser.parse_args(argv)
    if sys.version_info[:3] != (3, 12, 16) or sys.platform not in {"win32", "linux"}:
        parser.error("Use the pinned Windows/Linux CPython 3.12.16 venv")
    if sys.platform == "linux" and shutil.which("unzip") is None:
        parser.error("Info-ZIP unzip is required; install it with: sudo apt-get install unzip")
    work = args.work_root.resolve()
    work.mkdir(parents=True, exist_ok=False)
    build, package = work / "native", work / "package"
    windows = sys.platform == "win32"
    import pybind11
    import nvidia.cuda_runtime
    cuda = Path(next(iter(nvidia.cuda_runtime.__path__))) / ("bin" if windows else "lib")
    env = {**application_environment(os.environ), "PYTHONDONTWRITEBYTECODE": "1", "OVUIINSPECT_ENABLED": "0",
           "XDG_CONFIG_HOME": str(work / "preferences"), "XDG_CACHE_HOME": str(work / "cache"),
           "XDG_DATA_HOME": str(work / "data"), "LOCALAPPDATA": str(work / "preferences")}
    env["PATH"] = os.pathsep.join((str(Path(sys.executable).parent),
                                   str(cuda), env.get("PATH", "")))
    if not windows:
        env["LD_LIBRARY_PATH"] = str(cuda) + ":" + env.get("LD_LIBRARY_PATH", "")
    command = configure_command(build, Path(sys.executable), args.physics_root.resolve(), windows=windows)
    command += ["-Dpybind11_DIR=" + pybind11.get_cmake_dir()]
    run(command, env=env)
    run(["cmake", "--build", build, "--config", CONFIG, "--parallel", "4", "--target", *native_targets()], env=env)
    ctest = run(["ctest", "--test-dir", build, "-C", CONFIG, "--output-on-failure",
                 "--output-junit", work / "ctest.xml"], env=env, check=False)
    native_root, fixture_root = build / "python", build / "src/PVDSession"
    if windows:
        native_root, fixture_root = native_root / CONFIG, fixture_root / CONFIG
    pytest = run([sys.executable, "-B", ROOT / "ci/test_ovui.py", "--native-root", native_root,
                  "--fixture-root", fixture_root, "--cuda-bin", cuda,
                  "--", "-q", "--tb=short", "-p", "no:cacheprovider", "--junitxml=" + str(work / "pytest.xml"),
                  "--basetemp=" + str(work / "pytest-temp")], env=env, check=False)
    # Failed regressions still yield inspectable preview bytes, never a successful release job.
    command = [sys.executable, "-B", ROOT / "ci/package_ovui_portable.py", "--output", package,
               "--native", native_root / ("_pvd_native" + sysconfig.get_config_var("EXT_SUFFIX")),
               "--physics-root", args.physics_root, "--ovui-root", args.ovui_root]
    if windows:
        crt = args.msvc_runtime
        if crt is None:
            vswhere = Path(os.environ["ProgramFiles(x86)"]) / "Microsoft Visual Studio/Installer/vswhere.exe"
            visual_studio = run([vswhere, "-latest", "-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                                 "-property", "installationPath"], env=env, capture=True).stdout.strip()
            crt = msvc_runtime(Path(visual_studio) / "VC/Redist/MSVC")
        command += ["--msvc-runtime", crt]
    if args.allow_dirty:
        command += ["--allow-dirty"]
    run(command, env=env)
    report = extracted_probe(package, work / "relocated package", ROOT / "test/data/multi_geom.ovd", env)
    results = {"ctest_exit": ctest.returncode, "pytest_exit": pytest.returncode,
               "extracted_probe": "passed", "gui_smoke": "not performed; desktop validation remains separate"}
    if args.gui_smoke:
        bundle = work / "relocated package" / f"OvdNext-{report['platform']}-x86_64"
        python = bundle / ("python/python.exe" if windows else "python/bin/python3.12")
        gui_env = {key: value for key, value in env.items() if not key.startswith("PYTHON")
                   and key not in {"VIRTUAL_ENV", "GLCONTEXT_WIN_LIBGL"}}
        gui_env["PATH"] = str(Path(os.environ["SystemRoot"]) / "System32") if windows else "/usr/bin:/bin"
        gui_env["LD_LIBRARY_PATH"] = str(bundle / "lib") + ":" + str(bundle / "python/lib")
        command = [str(python), "-I", "-B", str(ROOT / "ci/smoke_ovui_gui.py"), "--bundle", str(bundle),
                   "--screenshot", str(work / "gui-smoke.png")]
        if args.gui_opengl_library:
            command += ["--opengl-library", str(args.gui_opengl_library.resolve())]
        subprocess.run(command, env=gui_env, check=True, timeout=120)
        results["gui_smoke"] = "passed: rendered Display, contacts, family scales and master controls"
    (work / "verification.json").write_text(json.dumps(results, indent=2) + "\n")
    if summary := env.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a", encoding="utf-8") as out:
            out.write(f"\n### OvdNext {report['platform']} {report['version']}\n\n"
                      f"Archive: {report['archive_bytes']:,} bytes; extracted: {report['expanded_bytes']:,} bytes.\n\n"
                      f"CTest exit: {ctest.returncode}; pytest exit: {pytest.returncode}; relocated capture probe passed.\n\n"
                      f"GUI smoke: {results['gui_smoke']}.\n")
    return int(bool(ctest.returncode or pytest.returncode))


if __name__ == "__main__":
    raise SystemExit(main())

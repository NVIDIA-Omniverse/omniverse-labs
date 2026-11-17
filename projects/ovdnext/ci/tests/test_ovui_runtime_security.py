# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real MIME initialization at the viewer's process entry points."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
APPLICATION = '''import mimetypes

observed_mime = mimetypes.guess_type("capture.ovd_poison")[0]
observed_text = mimetypes.guess_type("capture.txt")[0]
observed_registry = mimetypes.guess_type("capture.ovd_registry")[0]

class Application:
    def __init__(self, *_args):
        pass
    def run(self):
        pass
'''
PROBE = '''import contextlib
import importlib.util
import json
import mimetypes
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

root, mode, platform, initialized = sys.argv[1:]
root = Path(root)
poison = root / "poison.mime.types"
mimetypes.knownfiles = [str(poison)]
mimetypes._db = None
mimetypes.inited = False

# Keep registry lookup deterministic without changing its initialization flow.
def registry_types(self, strict=True):
    self.add_type("application/x-ovd-registry", ".ovd_registry", strict)
mimetypes.MimeTypes.read_windows_registry = registry_types
if initialized == "yes":
    mimetypes.init()
sys.platform = platform
sys.path.insert(0, str(root / "app"))
sys.argv = ["ovd_ui"]

if mode == "module":
    runpy.run_module("ovd_ui", run_name="__main__")
else:
    entry = root / ("ci/run_ovui_windows.py" if mode == "source-check"
                    else "portable_launch.py")
    spec = importlib.util.spec_from_file_location("test_launcher", entry)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    if mode == "source-check":
        launcher.os.add_dll_directory = lambda _path: contextlib.nullcontext()
        launcher.version = lambda _name: "0.2.0"
        launcher.main(["--native-root", str(root / "app"),
                       "--cuda-bin", str(root / "cuda"), "--check"])
    else:
        launcher.sys = SimpleNamespace(
            executable=str(root / "python/python.exe"),
            flags=SimpleNamespace(isolated=True),
            path=[str(root / "python/Lib")], platform=platform,
            argv=[str(entry), *(["--check"] if mode == "portable-check" else [])])
        launcher.preflight_graphics_driver = lambda: None
        launcher.os.add_dll_directory = lambda _path: contextlib.nullcontext()
        launcher.main()

application = sys.modules["ovd_ui.application"]
mimetypes.init(None)
print(json.dumps({"at_application_import": application.observed_mime,
                  "after_reinitialize": mimetypes.guess_type("capture.ovd_poison")[0],
                  "text": application.observed_text,
                  "registry": application.observed_registry}))
'''


def stage_entry_points(bundle):
    for destination in (bundle / "app/ovd_ui", bundle / "python/ovd_ui"):
        destination.mkdir(parents=True)
        for filename in ("__init__.py", "__main__.py", "runtime_security.py"):
            source = ROOT / "python/ovd_ui" / filename
            if source.exists():
                shutil.copyfile(source, destination / filename)
        (destination / "application.py").write_text(APPLICATION, encoding="utf-8")
        (destination / "renderer.py").write_text("", encoding="utf-8")
        (destination / "runtime-build.json").write_text("{}", encoding="utf-8")
    # These are import-boundary fixtures; the real entry point checks their paths.
    for name in ("_pvd_native", "omni", "omni.ui", "ovui_widgets",
                 "ovui_widgets.common", "ovui_widgets.common.style",
                 "ovui_data_adapters", "ovui_data_adapters.services",
                 "numpy", "moderngl", "PIL"):
        path = bundle / "app" / Path(*name.split(".")) / "__init__.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    (bundle / "python/Lib").mkdir()
    (bundle / "cuda").mkdir()
    (bundle / "cuda/cudart64_12.dll").write_bytes(b"")
    (bundle / "app/_pvd_native-fixture.pyd").write_bytes(b"")
    (bundle / "ci").mkdir()
    shutil.copyfile(ROOT / "ci/run_ovui_windows.py", bundle / "ci/run_ovui_windows.py")
    shutil.copyfile(ROOT / "ci/ovui_package/portable_launch.py", bundle / "portable_launch.py")
    (bundle / "poison.mime.types").write_text(
        "application/x-ovd-poison ovd_poison txt\n", encoding="utf-8")


def invoke_entry_point(tmp_path, mode, platform, initialized):
    bundle = tmp_path / "viewer"
    stage_entry_points(bundle)
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", PROBE, str(bundle), mode, platform,
         "yes" if initialized else "no"],
        capture_output=True, text=True, timeout=30, cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout.splitlines()[-1])


@pytest.mark.parametrize("mode", ["module", "portable", "portable-check", "source-check"])
@pytest.mark.parametrize("initialized", [False, True], ids=["fresh", "already-initialized"])
@pytest.mark.skipif(sys.platform != "win32", reason="Real Windows MIME startup")
def test_windows_entry_point_excludes_default_mime_files_before_application_import(
        tmp_path, mode, initialized):
    result = invoke_entry_point(tmp_path, mode, "win32", initialized)
    assert result["at_application_import"] is None
    assert result["after_reinitialize"] is None
    assert result["text"] == "text/plain"
    assert result["registry"] == "application/x-ovd-registry"


@pytest.mark.parametrize("initialized", [False, True], ids=["fresh", "already-initialized"])
@pytest.mark.parametrize("mode", ["module", "portable", "portable-check"])
@pytest.mark.skipif(sys.platform != "linux", reason="Real Linux MIME startup")
def test_linux_startup_retains_normal_default_mime_files(tmp_path, mode, initialized):
    result = invoke_entry_point(tmp_path, mode, "linux", initialized)
    assert result["at_application_import"] == "application/x-ovd-poison"
    assert result["after_reinitialize"] == "application/x-ovd-poison"

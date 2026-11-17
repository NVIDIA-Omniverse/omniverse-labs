# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native build inputs must keep test targets and runtime ABI aligned."""
import importlib.util
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("build_ovui", ROOT / "ci/build_ovui.py")
build = importlib.util.module_from_spec(SPEC)
if Path(SPEC.origin).exists():
    SPEC.loader.exec_module(build)

PINNED_BUILD_PYTHON = pytest.mark.skipif(sys.version_info[:3] != (3, 12, 16),
    reason="ci/build_ovui.py rejects non-pinned interpreters before the unzip preflight")

@PINNED_BUILD_PYTHON
@pytest.mark.skipif(sys.platform != "linux", reason="Linux requires Info-ZIP unzip")
def test_build_without_unzip_fails_before_creating_work(tmp_path):
    work = tmp_path / "work"
    args = [item for name in ("work-root", "physics-root", "ovui-root")
            for item in ("--" + name, str(work if name == "work-root" else tmp_path / name))]
    result = subprocess.run([sys.executable, "-B", str(ROOT / "ci/build_ovui.py"), *args],
                            env={**os.environ, "PATH": str(tmp_path)}, text=True, capture_output=True)
    assert not work.exists(), result.stdout + result.stderr
    assert result.returncode == 2, result.stderr
    assert "Info-ZIP unzip" in result.stderr
    assert "sudo apt-get install unzip" in result.stderr
    assert "Traceback" not in result.stderr


def test_build_help_does_not_need_unzip(tmp_path):
    result = subprocess.run([sys.executable, "-B", str(ROOT / "ci/build_ovui.py"), "--help"],
                            env={**os.environ, "PATH": str(tmp_path)}, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "--work-root" in result.stdout


@PINNED_BUILD_PYTHON
def test_build_preflight_accepts_linux_unzip_or_windows_without_it(tmp_path, monkeypatch):
    commands = tmp_path / "commands"
    commands.mkdir()
    if sys.platform == "linux":
        (commands / "unzip").symlink_to(shutil.which("unzip"))
    monkeypatch.setenv("PATH", str(commands))
    args = [item for name in ("work-root", "physics-root", "ovui-root")
            for item in ("--" + name, str(tmp_path))]
    # An existing work directory stops the real CLI before dependency imports/builds.
    with pytest.raises(FileExistsError):
        build.main(args)


@pytest.mark.parametrize("windows", [False, True])
def test_configure_enables_native_regressions_with_exact_python_and_physics(tmp_path, windows):
    assert hasattr(build, "configure_command"), "OvUI build driver is not implemented"
    python, physics = (tmp_path / name for name in ("Python runtime/python", "Physics source"))
    command = build.configure_command(tmp_path / "build", python, physics, windows=windows)
    assert "-DPVD3_BUILD_TESTS=ON" in command and "-DPVD3_BUILD_PYTHON=ON" in command
    assert "-DPVD3_REQUIRE_OVD_SUPPORT=ON" in command
    assert f"-DPython_EXECUTABLE={python.as_posix()}" in command
    assert f"-DPVDDOM_ROOT={(physics / 'physx/pvddom').as_posix()}" in command
    assert f"-DPVDRUNTIME_ROOT={(physics / 'physx/pvdruntime').as_posix()}" in command
    assert "-DCMAKE_DISABLE_FIND_PACKAGE_Qt5=ON" in command
    assert not any("Qt runtime" in item for item in command)
    assert command[command.index("-G") + 1] == ("Visual Studio 17 2022" if windows else "Ninja")


def test_required_targets_include_native_regressions_and_writers():
    assert hasattr(build, "native_targets"), "OvUI build driver is not implemented"
    targets = build.native_targets()
    assert set(targets) >= {"_pvd_native", "pvdcore_tests", "checked_output_tests", "pvd_geometry_fixture",
                            "pvd_ingestion_fixture", "pvd_tolerance_fixture", "pvd_availability_fixture"}


def test_crt_resolution_picks_complete_newest_redistributable(tmp_path):
    assert hasattr(build, "msvc_runtime"), "CRT resolution is not implemented"
    for version in ("14.29.30133", "14.44.35112", "14.99.1"):
        crt = tmp_path / version / "x64/Microsoft.VC143.CRT"
        crt.mkdir(parents=True)
        for name in ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"):
            if version != "14.99.1":
                (crt / name).write_bytes(b"CRT fixture")
    assert build.msvc_runtime(tmp_path) == tmp_path / "14.44.35112/x64/Microsoft.VC143.CRT"
    with pytest.raises(ValueError, match="CRT"):
        build.msvc_runtime(tmp_path / "missing")

def test_app_environment_drops_obsolete_framework_overrides_without_an_sdk(tmp_path):
    source = {"PATH": "/toolchain", "LD_LIBRARY_PATH": "/cuda", "Qt5_DIR": "/obsolete-sdk",
              "QT_PLUGIN_PATH": "/obsolete-plugins"}
    env = build.application_environment(source)
    assert env["PATH"] == "/toolchain"
    assert env["LD_LIBRARY_PATH"] == "/cuda"
    assert "Qt5_DIR" not in env and "QT_PLUGIN_PATH" not in env

@pytest.mark.skipif(sys.platform != "linux", reason="POSIX source setup command contract")
def test_source_setup_configures_without_qt_and_runs_import_smoke(tmp_path):
    root, commands = tmp_path / "source", tmp_path / "commands"
    (root / "ci").mkdir(parents=True)
    commands.mkdir()
    script = root / "ci/setup_ovui.sh"
    script.write_bytes((ROOT / "ci/setup_ovui.sh").read_bytes())
    for dependency in ("physics", "ovui"):
        (root / "_ci/deps" / dependency).mkdir(parents=True)
    python = root / "_ci/venv/bin/python"
    python.parent.mkdir(parents=True)
    log = tmp_path / "commands.log"
    python.write_text('#!/bin/sh\ncase "$*" in\n *platform*) echo 3.12.16;;\n *pybind11*) echo /fake/pybind11;;\n *) printf "python %s\\n" "$*" >> "$TEST_LOG";;\nesac\n')
    python.chmod(0o755)
    for name, body in (
        ("uv", 'printf "uv %s\\n" "$*" >> "$TEST_LOG"'),
        ("cmake", 'printf "cmake %s\\n" "$*" >> "$TEST_LOG"'),
        ("git", 'case "$2" in *physics*) echo a12f79ce585ffb79e5ffa4848cb45efc3de2f632;; *) echo a81db103a4d99ff2d75368a444e167889536ff42;; esac')):
        path = commands / name
        path.write_text("#!/bin/sh\n" + body + "\n")
        path.chmod(0o755)
    env = {**os.environ, "PATH": str(commands) + ":/usr/bin:/bin", "TEST_LOG": str(log)}
    result = subprocess.run(["bash", str(script)], cwd=tmp_path, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    observed = log.read_text()
    assert "read_runtime_provenance(Path(sys.base_prefix), False)" in observed
    assert observed.index("read_runtime_provenance") < observed.index("-r python/requirements.txt")
    assert "python -B ci/install_ovui_pillow.py --work-root " in observed
    assert observed.index("-r python/requirements.txt") < observed.index("ci/install_ovui_pillow.py") < observed.index("cmake -S")
    assert "--source-cache " + str(root / "_ci/pillow-sources") in observed
    assert "-DCMAKE_DISABLE_FIND_PACKAGE_Qt5=ON" in observed
    assert "PVD3_BUILD_LEGACY_UI" not in observed
    assert "reference-qt-root" not in result.stdout
    assert "import _pvd_native" in observed
    assert "-m pytest" not in observed
    assert "import smoke only" in result.stdout


def source_setup_fixture(tmp_path):
    """Run real setup and runtime installer; replace network/build boundaries only."""
    root, commands = tmp_path / "source", tmp_path / "commands"
    (root / "ci").mkdir(parents=True)
    commands.mkdir()
    for name in ("setup_ovui.sh", "install_ovui_python.py"):
        (root / "ci" / name).write_bytes((ROOT / "ci" / name).read_bytes())
    for dependency in ("physics", "ovui"):
        (root / "_ci/deps" / dependency).mkdir(parents=True)
    log = tmp_path / "commands.log"
    # This process boundary represents the pinned interpreter's measured identity.
    # Provenance reads, hashes and setup branching execute production code.
    runtime_script = f'''#!{sys.executable}
import json, os, sys
from pathlib import Path
with open(os.environ['TEST_LOG'], 'a') as output:
    output.write('python ' + ' '.join(sys.argv[1:]) + '\\n')
if '-c' in sys.argv:
    code = sys.argv[sys.argv.index('-c') + 1]
    if "'implementation'" in code:
        print(json.dumps({{'python':'3.12.16','implementation':'cpython','bits':64,
            'expat':os.environ.get('TEST_RUNTIME_EXPAT', 'expat_2.9.0'),
            'openssl':'OpenSSL test','zlib':'test'}}))
    elif 'platform.python_version' in code:
        print(os.environ.get('TEST_RUNTIME_VERSION', '3.12.16'))
    elif 'read_runtime_provenance' in code:
        sys.base_prefix = str(Path(__file__).resolve().parents[1])
        sys.path.insert(0, os.getcwd())
        exec(code)
    elif 'import _pvd_native' in code:
        print('Headless native import smoke passed')
    else:
        raise SystemExit('unexpected interpreter command: ' + code)
elif sys.argv[1:3] == ['-m', 'pybind11']:
    print('/fake/pybind11')
elif 'ci/install_ovui_pillow.py' not in sys.argv:
    raise SystemExit('unexpected interpreter command')
'''.encode()
    manifest = json.loads((ROOT / "ci/ovui-python-3.12.16.json").read_text())
    entry = manifest["platforms"]["linux"]
    payload = {"bin/python3.12": runtime_script, "lib/libpython3.12.so": b"library",
               "lib/libpython3.12.so.1.0": b"library"}
    input_root = root / "_ci/python-inputs-fa506e08"
    cache = input_root / "cache"
    cache.mkdir(parents=True)
    source = cache / entry["filename"]
    with tarfile.open(source, "w:gz") as stream:
        for name, content in payload.items():
            member = tarfile.TarInfo("python/" + name)
            member.mode = 0o755 if name.startswith("bin/") else 0o644
            member.size = len(content)
            stream.addfile(member, io.BytesIO(content))
    entry.update(bytes=source.stat().st_size, sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                 core_sha256={name: hashlib.sha256(content).hexdigest() for name, content in payload.items()})
    (root / "ci/ovui-python-3.12.16.json").write_text(json.dumps(manifest))
    uv = commands / "uv"
    uv.write_text(f'''#!{sys.executable}
import os, shlex, sys
from pathlib import Path
with open(os.environ['TEST_LOG'], 'a') as output:
    output.write('uv ' + ' '.join(sys.argv[1:]) + '\\n')
if sys.argv[1:3] == ['python', 'find']:
    print({sys.executable!r})
elif sys.argv[1] == 'venv':
    python = Path(sys.argv[-1]) / 'bin/python'
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text('#!/bin/sh\\nexec ' + shlex.quote(sys.argv[3]) + ' "$@"\\n')
    python.chmod(0o755)
''')
    uv.chmod(0o755)
    for name, body in (
        ("cmake", 'printf "cmake %s\\n" "$*" >> "$TEST_LOG"'),
        ("git", 'case "$2" in *physics*) echo a12f79ce585ffb79e5ffa4848cb45efc3de2f632;; *) echo a81db103a4d99ff2d75368a444e167889536ff42;; esac')):
        path = commands / name
        path.write_text("#!/bin/sh\n" + body + "\n")
        path.chmod(0o755)
    env = {**os.environ, "PATH": str(commands) + ":/usr/bin:/bin", "TEST_LOG": str(log)}
    env.pop("OVUI_PYTHON_INPUT_ROOT", None)
    return root, input_root, log, env


@pytest.mark.skipif(sys.platform != "linux", reason="POSIX source setup runtime lifecycle")
def test_source_setup_recreates_venv_using_verified_retained_runtime(tmp_path):
    root, input_root, log, env = source_setup_fixture(tmp_path)
    script = root / "ci/setup_ovui.sh"
    first = subprocess.run(["bash", str(script)], env=env, text=True, capture_output=True)
    assert first.returncode == 0, first.stdout + first.stderr
    prefix = input_root / "runtime/python"
    before = {name: (prefix / name).read_bytes() for name in
              ("bin/python3.12", "lib/libpython3.12.so", "lib/libpython3.12.so.1.0", "runtime-input-provenance.json")}
    shutil.rmtree(root / "_ci/venv")
    second_start = len(log.read_text())
    second = subprocess.run(["bash", str(script)], env=env, text=True, capture_output=True)
    assert second.returncode == 0, second.stdout + second.stderr
    assert (root / "_ci/venv/bin/python").is_file()
    assert before == {name: (prefix / name).read_bytes() for name in before}
    observed = log.read_text()[second_start:]
    assert observed.index("'implementation'") < observed.index("uv venv")
    assert observed.index("read_runtime_provenance") < observed.index("-r python/requirements.txt")


@pytest.mark.skipif(sys.platform != "linux", reason="POSIX source setup runtime lifecycle")
@pytest.mark.parametrize("damage", ["old-prefix", "stale-identity", "stale-python", "incomplete-runtime"])
def test_source_setup_rejects_unverified_runtime_before_dependency_installs(tmp_path, damage):
    root, input_root, log, env = source_setup_fixture(tmp_path)
    script = root / "ci/setup_ovui.sh"
    first = subprocess.run(["bash", str(script)], env=env, text=True, capture_output=True)
    assert first.returncode == 0, first.stdout + first.stderr
    if damage == "old-prefix":
        (input_root / "runtime/python/runtime-input-provenance.json").unlink()
        remedy = "_ci/venv"
    elif damage == "stale-identity":
        env["TEST_RUNTIME_EXPAT"] = "expat_2.8.5"
        remedy = "_ci/venv"
    elif damage == "stale-python":
        env["TEST_RUNTIME_VERSION"] = "3.12.13"
        remedy = "_ci/venv"
    else:
        shutil.rmtree(root / "_ci/venv")
        (input_root / "runtime/python/runtime-input-provenance.json").unlink()
        remedy = str(input_root / "runtime")
    second_start = len(log.read_text())
    second = subprocess.run(["bash", str(script)], env=env, text=True, capture_output=True)
    assert second.returncode != 0
    assert remedy in second.stderr and "move" in second.stderr and "rerun ci/setup_ovui.sh" in second.stderr
    observed = log.read_text()[second_start:]
    assert "uv pip install" not in observed and "ci/install_ovui_pillow.py" not in observed and "cmake" not in observed
    assert (input_root / "runtime/python/bin/python3.12").exists()

def test_native_build_includes_every_registered_data_ctest_executable():
    assert set(build.native_targets()) >= {"ovddata_tests", "ovddata_ingest_tests",
        "ovddata_omnipvd_tests", "ovdnext_data_probe_tests", "checked_output_tests"}

@pytest.mark.skipif(sys.platform != "linux", reason="Real shared Linux executable directories")
@pytest.mark.parametrize(("path", "tools", "command", "expected_output"), [
    ("/usr/local/bin:/usr/bin:/bin", ("cmake", "git", "readelf"), ("cmake", "--version"), "cmake version"),
    ("/bin", ("sh",), ("sh", "-c", "printf shared-system-path"), "shared-system-path"),
])
def test_app_environment_keeps_shared_system_tools_available(path, tools, command, expected_output):
    env = build.application_environment({"PATH": path, "Qt5_DIR": "/usr/lib/cmake/Qt5"})
    assert env["PATH"] == path
    assert "Qt5_DIR" not in env
    for tool in tools:
        assert shutil.which(tool, path=env["PATH"]), tool
    result = subprocess.run(command, env=env, text=True, capture_output=True)
    assert result.returncode == 0 and expected_output in result.stdout


@pytest.mark.skipif(sys.platform != "linux", reason="POSIX ZIP executable mode and launcher contract")
def test_linux_zip_relocated_probe_uses_executable_launcher(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    archive = package / "OvdNext-linux-x86_64-0.2.1.zip"
    # A real process at a relocated path proves unzip preserved executable mode.
    with zipfile.ZipFile(archive, "w") as source:
        item = zipfile.ZipInfo("OvdNext-linux-x86_64/OvdNext.sh")
        item.create_system = 3
        item.external_attr = 0o100755 << 16
        source.writestr(item, '#!/bin/sh\n[ "$1" = "--check" ] && [ -f "$2" ] || exit 7\n'
            'printf \'{"version":"0.2.1","capture":{"scenes":1},"isolated":true}\\n\'\n')
    report = {"archive": archive.name, "version": "0.2.1"}
    (package / "package-report.json").write_text(json.dumps(report))
    capture = tmp_path / "test capture.ovd"
    capture.touch()
    assert build.extracted_probe(package, tmp_path / "relocated ZIP", capture, os.environ) == report
    assert json.loads((package / "extracted-probe.json").read_text())["capture"]["scenes"] == 1

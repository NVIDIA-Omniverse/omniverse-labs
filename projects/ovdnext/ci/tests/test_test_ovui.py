# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Behavioral contract for the source-tree native pytest bootstrap."""
from pathlib import Path
import importlib.util
import os
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "ci/test_ovui.py"
FIXTURES = ("pvd_geometry_fixture",
            "pvd_ingestion_fixture", "pvd_tolerance_fixture",
            "pvd_availability_fixture", "pvd_joint_calibration_fixture")


def load_runner():
    spec = importlib.util.spec_from_file_location("test_ovui_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_options_augment_default_gate_but_explicit_test_paths_select_a_focus():
    runner = load_runner()
    defaults = [str(ROOT / "python/tests"), *(str(ROOT / f"ci/tests/{name}") for name in
        ("test_ovui_build.py", "test_ovui_python.py", "test_ovui_expat.py", "test_ovui_pillow.py", "test_ovui_pillow_probe.py", "test_ovui_package.py", "test_ovui_portable.py",
         "test_ovui_licenses.py", "test_ovui_release.py", "test_ovui_version.py", "test_ovui_workflow.py", "test_ovui_launcher.py", "test_ovui_runtime_security.py",
         "test_ovui_gui_smoke.py", "test_ovui_package_verification.py", "test_test_ovui.py", "test_tcp_selection_sender.py"))]
    assert runner._pytest_arguments(["-q", "--junitxml=results.xml"], ROOT) == [
        *defaults, "-q", "--junitxml=results.xml"]
    assert runner._pytest_arguments(["python/tests/test_about_info.py", "-q"], ROOT) == [
        "python/tests/test_about_info.py", "-q"]


@pytest.fixture
def runtime(tmp_path):
    native, fixtures, cuda = (tmp_path / name for name in
                            ("native", "fixtures", "cuda"))
    for directory in (native, fixtures, cuda):
        directory.mkdir()
    native_name = "_pvd_native.cp312-win_amd64.pyd" if sys.platform == "win32" else "_pvd_native.so"
    (native / native_name).touch()
    suffix = ".exe" if sys.platform == "win32" else ""
    for name in FIXTURES:
        (fixtures / f"{name}{suffix}").touch()
    if sys.platform == "win32":
        (cuda / "cudart64_12.dll").touch()
    return native, fixtures, cuda


def invoke(runtime, pytest_file, environment=None, extra_args=()):
    native, fixtures, cuda = runtime
    command = [sys.executable, str(RUNNER), "--native-root", str(native),
               "--fixture-root", str(fixtures)]
    command += ["--cuda-bin", str(cuda)]
    return subprocess.run([*command, "--", str(pytest_file), "-q", "-p", "no:cacheprovider",
                           "--rootdir", str(pytest_file.parent), *extra_args], cwd=ROOT,
                          env=environment, capture_output=True, text=True, timeout=30)


def test_focused_run_exports_real_paths_to_pytest_and_child_processes(runtime, tmp_path):
    test = tmp_path / "test_environment.py"
    expected_native, expected_fixtures = map(str, runtime[:2])
    test.write_text(
        "from pathlib import Path\n"
        "import os, sys\n"
        "def test_environment():\n"
        f" assert {expected_native!r} in sys.path\n"
        f" assert os.environ['PVD_FIXTURE_ROOT'] == {expected_fixtures!r}\n"
        " expected = Path(os.environ['PVD_GEOMETRY_FIXTURE']).parent\n"
        " assert expected == Path(os.environ['PVD_FIXTURE_ROOT'])\n"
        f" assert {{Path(p) for p in {str(os.pathsep.join(map(str, runtime[2:])) )!r}.split(os.pathsep)}} <= "
        "{Path(p) for p in os.environ['PATH'].split(os.pathsep)}\n")
    result = invoke(runtime, test)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("inherited_native", [None, "unrelated-native-build"])
def test_runner_exports_validated_native_root_to_nested_process(runtime, tmp_path, inherited_native):
    test = tmp_path / "test_native_environment.py"
    test.write_text(
        "import os, subprocess, sys\n"
        "def test_native_environment():\n"
        " child = subprocess.check_output([sys.executable, '-c', "
        "\"import os; print(os.environ.get('OVDNEXT_TEST_NATIVE_ROOT', 'missing'))\"], text=True)\n"
        f" assert child.strip() == {str(runtime[0])!r}\n")
    environment = dict(os.environ)
    environment.pop("OVDNEXT_TEST_NATIVE_ROOT", None)
    if inherited_native is not None:
        environment["OVDNEXT_TEST_NATIVE_ROOT"] = inherited_native
    result = invoke(runtime, test, environment)
    assert result.returncode == 0, result.stdout + result.stderr


def test_runner_reports_skipped_test_reason(runtime, tmp_path):
    test = tmp_path / "test_skipped.py"
    test.write_text("import pytest\ndef test_skip(): pytest.skip('isolated desktop opt-in is absent')\n")
    result = invoke(runtime, test)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "isolated desktop opt-in is absent" in result.stdout


def test_runner_reports_failed_node_error_node_and_skip_reason_without_tracebacks(runtime, tmp_path):
    test = tmp_path / "test_summary.py"
    test.write_text(
        "import pytest\n"
        "@pytest.fixture\n"
        "def failing_setup(): raise RuntimeError('intentional fixture setup error')\n"
        "def test_setup_error(failing_setup): pass\n"
        "def test_failure(): assert False\n"
        "def test_skip(): pytest.skip('isolated desktop opt-in is absent')\n")
    result = invoke(runtime, test, extra_args=("--tb=no",))
    assert result.returncode == 1, result.stdout + result.stderr
    assert any(line.startswith("FAILED ") and "test_summary.py::test_failure" in line
               for line in result.stdout.splitlines()), result.stdout
    assert any(line.startswith("ERROR ") and "test_summary.py::test_setup_error" in line
               for line in result.stdout.splitlines()), result.stdout
    assert "isolated desktop opt-in is absent" in result.stdout


@pytest.mark.skipif(sys.version_info[:3] != (3, 12, 16), reason="CI build requires the pinned interpreter")
def test_ci_build_keeps_failure_summary_and_short_tracebacks(runtime, tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("ci_build_reporting", ROOT / "ci/build_ovui.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)

    class PytestCommandReady(Exception):
        pass

    def skip_native_build(command, **kwargs):
        # Stop at the real CI pytest boundary; no native build or package is needed.
        command = list(map(str, command))
        if str(RUNNER) in command:
            raise PytestCommandReady(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(build, "run", skip_native_build)
    work = tmp_path / "ci-build"
    with pytest.raises(PytestCommandReady) as captured:
        build.main([item for name in ("work-root", "physics-root", "ovui-root")
                    for item in ("--" + name, str(work if name == "work-root" else tmp_path / name))])
    command = captured.value.args[0]
    assert "--qt-bin" not in command and "--reference-qt-root" not in command
    options = command[command.index("--") + 1:]
    assert options[0].startswith("-"), options
    assert str(ROOT / "ci/tests/test_ovui_runtime_security.py") in load_runner()._pytest_arguments(options, ROOT)
    assert "--tb=short" in options, options
    # CI appends these after the runner's -rfEs; later overrides would hide diagnostics.
    assert not [option for option in options if option.startswith("-r")
                or option == "--reportchars" or option.startswith("--reportchars=")
                or option.startswith("--tb") and option != "--tb=short"], options

    test = tmp_path / "test_ci_reporting.py"
    test.write_text(
        "import pytest\n"
        "@pytest.fixture\n"
        "def failing_setup():\n"
        " raise RuntimeError('intentional CI setup error')\n"
        "def test_setup_error(failing_setup): pass\n"
        "def test_failure(): assert 1 == 2\n"
        "def test_skip(): pytest.skip('intentional CI skip reason')\n")
    result = invoke(runtime, test, extra_args=options)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "RuntimeError: intentional CI setup error" in result.stdout, result.stdout
    assert "assert 1 == 2" in result.stdout, result.stdout
    assert any(line.startswith("FAILED ") and "test_ci_reporting.py::test_failure" in line
               for line in result.stdout.splitlines()), result.stdout
    assert any(line.startswith("ERROR ") and "test_ci_reporting.py::test_setup_error" in line
               for line in result.stdout.splitlines()), result.stdout
    assert "intentional CI skip reason" in result.stdout, result.stdout


@pytest.mark.parametrize("variable", ["OVD_CLIPBOARD_TEST_DISPLAY", "OVD_TEST_XCLIP"])
def test_partial_clipboard_opt_in_fails_without_starting_desktop(runtime, tmp_path, variable):
    environment = dict(os.environ)
    environment.pop("OVD_CLIPBOARD_TEST_DISPLAY", None)
    environment.pop("OVD_TEST_XCLIP", None)
    environment[variable] = ":reserved-test-display" if variable.endswith("DISPLAY") else str(sys.executable)
    result = invoke(runtime, ROOT / "python/tests/test_clipboard_desktop.py", environment)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "requires both OVD_CLIPBOARD_TEST_DISPLAY and OVD_TEST_XCLIP" in result.stdout


def load_clipboard_desktop():
    spec = importlib.util.spec_from_file_location("clipboard_desktop_contract", ROOT / "python/tests/test_clipboard_desktop.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("vendor", [None, "mesa"])
def test_clipboard_desktop_preserves_caller_graphics_selection(tmp_path, monkeypatch, vendor):
    desktop = load_clipboard_desktop()
    monkeypatch.setattr(desktop, "DISPLAY", ":reserved-test-display")
    monkeypatch.setattr(desktop, "XCLIP", sys.executable)
    monkeypatch.delenv("__GLX_VENDOR_LIBRARY_NAME", raising=False)
    if vendor is not None:
        monkeypatch.setenv("__GLX_VENDOR_LIBRARY_NAME", vendor)
    # Only the external GUI launch is replaced; inspect the real startup environment.
    monkeypatch.setattr(desktop.subprocess, "Popen", lambda *args, **kwargs: object())
    monkeypatch.setattr(desktop.Desktop, "wait", lambda self, identity: None)
    app = desktop.Desktop(tmp_path)
    try:
        actual_vendor = app.env.get("__GLX_VENDOR_LIBRARY_NAME")
        assert actual_vendor == vendor
        assert app.env["DISPLAY"] == ":reserved-test-display"
        assert Path(app.env["XDG_RUNTIME_DIR"]).parent == tmp_path
    finally:
        app.log.close()


def test_missing_explicit_xclip_fails_before_gui_startup(tmp_path, monkeypatch):
    desktop = load_clipboard_desktop()
    monkeypatch.setattr(desktop, "DISPLAY", ":reserved-test-display")
    monkeypatch.setattr(desktop, "XCLIP", str(tmp_path / "missing-xclip"))

    def forbidden_gui(*args, **kwargs):
        raise AssertionError("GUI must not start without the requested clipboard prerequisites")

    monkeypatch.setattr(desktop.subprocess, "Popen", forbidden_gui)
    with pytest.raises(pytest.fail.Exception, match="OVD_TEST_XCLIP must name an executable file"):
        desktop.Desktop(tmp_path)


def test_clipboard_desktop_launch_is_confined_to_temporary_directory(tmp_path, monkeypatch):
    desktop = load_clipboard_desktop()
    monkeypatch.setattr(desktop, "DISPLAY", ":reserved-test-display")
    monkeypatch.setattr(desktop, "XCLIP", sys.executable)
    launches = []

    def record_launch(command, **kwargs):
        launches.append((command, kwargs.get("cwd")))
        return object()

    monkeypatch.setattr(desktop.subprocess, "Popen", record_launch)
    monkeypatch.setattr(desktop.Desktop, "wait", lambda self, identity: None)
    app = desktop.Desktop(tmp_path)
    try:
        command, working_directory = launches[0]
        assert working_directory == tmp_path
        assert Path(command[1]).is_absolute()
        assert Path(command[1]).is_file()
    finally:
        app.log.close()


def test_pytest_failure_exit_is_preserved(runtime, tmp_path):
    test = tmp_path / "test_failure.py"
    test.write_text("def test_failure(): assert False, 'intentional runner contract failure'\n")
    result = invoke(runtime, test)
    assert result.returncode == 1
    assert "intentional runner contract failure" in result.stdout


def test_runner_replaces_inherited_application_preference_directories(runtime, tmp_path):
    inherited = tmp_path / "real-user-preferences"
    test = tmp_path / "test_preferences.py"
    test.write_text(
        "from pathlib import Path\n"
        "import os\n"
        "def test_preferences():\n"
        f" assert Path(os.environ['LOCALAPPDATA']) != Path({str(inherited)!r})\n"
        f" assert Path(os.environ['XDG_DATA_HOME']) != Path({str(inherited)!r})\n"
        " assert Path(os.environ['LOCALAPPDATA']).is_dir()\n"
        " assert Path(os.environ['XDG_DATA_HOME']).is_dir()\n")
    environment = dict(os.environ, LOCALAPPDATA=str(inherited), XDG_DATA_HOME=str(inherited))
    result = invoke(runtime, test, environment)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(sys.platform != "win32", reason="Windows glcontext contract")
def test_configured_mesa_library_directory_is_available_to_child(runtime, tmp_path):
    mesa = tmp_path / "mesa" / "opengl32.dll"
    mesa.parent.mkdir()
    mesa.touch()
    test = tmp_path / "test_mesa.py"
    test.write_text(
        "from pathlib import Path\n"
        "import os\n"
        "def test_mesa():\n"
        f" assert Path({str(mesa.parent)!r}) in map(Path, os.environ['PATH'].split(os.pathsep))\n")
    environment = dict(os.environ, GLCONTEXT_WIN_LIBGL=str(mesa))
    environment.pop("OVD_UI_HIDDEN_GUI_TESTS", None)  # Directory wiring only; no fake DLL load.
    result = invoke(runtime, test, environment)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(sys.platform != "win32", reason="Windows glcontext contract")
def test_missing_configured_mesa_library_fails_before_pytest(runtime, tmp_path):
    test = tmp_path / "test_never_runs.py"
    test.write_text("raise AssertionError('pytest must not start')\n")
    missing = tmp_path / "missing-mesa" / "opengl32.dll"
    environment = dict(os.environ, GLCONTEXT_WIN_LIBGL=str(missing))
    environment.pop("OVD_UI_HIDDEN_GUI_TESTS", None)  # Exercise the runner's path-validation boundary.
    result = invoke(runtime, test, environment)
    assert result.returncode == 2
    assert str(missing) in result.stderr
    assert "pytest must not start" not in result.stderr


@pytest.mark.skipif(sys.platform != "win32", reason="Windows GUI bootstrap contract")
@pytest.mark.parametrize("exists", [False, True], ids=("missing", "invalid"))
def test_hidden_gui_child_rejects_unloadable_configured_opengl(tmp_path, exists):
    library = tmp_path / "opengl32.dll"
    if exists:
        library.write_bytes(b"not a DLL")
    environment = dict(os.environ, OVD_UI_HIDDEN_GUI_TESTS="1", GLCONTEXT_WIN_LIBGL=str(library),
                       OVUI_TEST_DLL_DIRECTORIES="",
                       PYTHONPATH=str(ROOT / "ci/ovui_test_bootstrap"))
    result = subprocess.run([sys.executable, "-c", "print('GUI child started')"],
                            env=environment, capture_output=True, text=True, timeout=10)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "GUI child started" not in result.stdout
    assert "Cannot load configured CI OpenGL library" in result.stderr


def test_missing_required_fixture_fails_before_pytest(runtime, tmp_path):
    suffix = ".exe" if sys.platform == "win32" else ""
    (runtime[1] / f"pvd_ingestion_fixture{suffix}").unlink()
    test = tmp_path / "test_never_runs.py"
    test.write_text("raise AssertionError('pytest must not start')\n")
    result = invoke(runtime, test)
    assert result.returncode == 2
    assert "pvd_ingestion_fixture" in result.stderr
    assert "pytest must not start" not in result.stderr

def test_runner_needs_only_native_writers_not_reference_executables(runtime, tmp_path):
    test = tmp_path / "test_writer_only.py"
    test.write_text("def test_runs(): assert True\n")
    result = invoke(runtime, test)
    assert result.returncode == 0, result.stdout + result.stderr

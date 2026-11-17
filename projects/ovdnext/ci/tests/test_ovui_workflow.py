# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail closed on the security boundary of the internal release workflow."""
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import textwrap

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows workflow shell contract")
@pytest.mark.parametrize("exit_code", [0, 23])
def test_windows_workflow_executes_gui_gate_and_preserves_failure(tmp_path, exit_code):
    workflow = (ROOT / ".github/workflows/ovui.yml").read_text()
    step = workflow.split("      - name: Native/Python regressions and relocated portable package\n", 1)[1]
    body = textwrap.dedent(step.split("        run: |\n", 1)[1].split("      - name:", 1)[0])
    # Execute the actual workflow shell; substitute only the expensive builder.
    # Its child loads the real GUI eligibility module, without opening a window.
    (tmp_path / "ci").mkdir()
    (tmp_path / "ci/build_ovui.py").write_text(
        "import json, os, runpy, sys\n"
        f"gate = runpy.run_path({str(ROOT / 'python/tests/gui_runtime.py')!r})\n"
        "print(json.dumps({'enabled': gate['GUI_AVAILABLE'], 'hidden': gate['HIDDEN_WINDOWS'], "
        "'display': os.environ.get('DISPLAY'), 'gl': os.environ.get('GLCONTEXT_WIN_LIBGL'), 'args': sys.argv[1:]}))\n"
        f"raise SystemExit({exit_code})\n")
    environment = dict(os.environ, OVUI_PYTHON=sys.executable, RUNNER_TEMP=str(tmp_path),
                       GITHUB_WORKSPACE=str(tmp_path), OVUI_REFERENCE_QT=str(tmp_path / "qt"),
                       GLCONTEXT_WIN_LIBGL=str(tmp_path / "ci-only/opengl32.dll"))
    environment.pop("DISPLAY", None)
    environment.pop("OVD_UI_HIDDEN_GUI_TESTS", None)
    environment.pop("PYTHONPATH", None)  # The workflow starts before pytest's child bootstrap exists.
    shell = shutil.which("pwsh") or shutil.which("powershell")
    assert shell, "Windows CI contract requires PowerShell"
    result = subprocess.run([shell, "-NoProfile", "-NonInteractive", "-Command", body],
                            cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=20)
    assert result.returncode == exit_code, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["enabled"] and report["hidden"], report
    assert report["display"] is None  # Never borrow or fabricate a Windows DISPLAY.
    assert report["gl"] == environment["GLCONTEXT_WIN_LIBGL"]
    assert report["args"][-2:] == ["--gui-opengl-library", environment["GLCONTEXT_WIN_LIBGL"]]


def test_private_runtime_token_isolated_from_all_dependency_preparation():
    workflow = (ROOT / ".github/workflows/ovui.yml").read_text()
    preamble, jobs = workflow.split("\njobs:\n", 1)
    build = jobs.split("\n  verify-package:\n", 1)[0]
    build_header, step_body = build.split("\n    steps:\n", 1)
    token = re.compile(r"GH_TOKEN|GITHUB_TOKEN|github\.token")
    assert not token.search(preamble + build_header)
    steps = re.split(r"(?m)^      - ", step_body)[1:]
    token_steps = [step for step in steps if token.search(step)]
    assert len(token_steps) == 1
    runtime_input = token_steps[0]
    assert "ci/install_ovui_python.py" in runtime_input
    assert not re.search(r"\buv\s+(?:venv|pip\s+install|python\s+(?:install|find))\b", runtime_input)
    assert "OVUI_RUNTIME_PYTHON=$runtimePython" in runtime_input
    exports = [line for line in runtime_input.splitlines() if "$env:GITHUB_ENV" in line]
    assert len(exports) == 1 and "OVUI_RUNTIME_PYTHON=" in exports[0]
    dependencies = next(step for step in steps if "uv venv" in step)
    assert "uv venv --python $env:OVUI_RUNTIME_PYTHON" in dependencies
    assert "uv pip install" in dependencies and not token.search(dependencies)
    bootstrap = next(step for step in steps if "uv python install" in step)
    assert not token.search(bootstrap)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux workflow shell contract")
@pytest.mark.parametrize("has_xclip", [True, False])
def test_linux_clipboard_opt_in_uses_inner_display_and_requires_xclip(tmp_path, has_xclip):
    workflow = (ROOT / ".github/workflows/ovui.yml").read_text()
    # Exercise the actual shell body run inside xvfb-run, using a probe instead
    # of the build/GUI. No X connection or clipboard operation is performed.
    wrapper = re.search(r"xvfb-run [^\n]*? sh -ec '([^']+)' sh ", workflow).group(1)
    xclip = tmp_path / "xclip"
    if has_xclip:
        xclip.write_text("#!/bin/sh\nexit 99\n")
        xclip.chmod(0o700)
    environment = {"PATH": str(tmp_path), "DISPLAY": ":owned-display-probe",
                   "OVD_CLIPBOARD_TEST_DISPLAY": ":inherited-desktop-probe",
                   "OVD_TEST_XCLIP": "/inherited/xclip"}
    probe = "import json, os; print(json.dumps([os.environ['OVD_CLIPBOARD_TEST_DISPLAY'], os.environ['OVD_TEST_XCLIP']]))"
    result = subprocess.run(["/bin/sh", "-ec", wrapper, "sh", sys.executable, "-c", probe],
                            env=environment, text=True, capture_output=True, timeout=10)
    if has_xclip:
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout) == [":owned-display-probe", str(xclip)]
    else:
        assert result.returncode == 1
        assert "xclip is required for the Linux clipboard CI gate" in result.stderr
        assert result.stdout == ""  # The build must never start with a disabled gate.


@pytest.mark.skipif(sys.platform != "linux", reason="Linux workflow shell contract")
@pytest.mark.parametrize("stage", [0, 1], ids=("build", "fresh-package"))
def test_linux_workflow_desktops_fit_the_gui_test_windows(stage):
    workflow = (ROOT / ".github/workflows/ovui.yml").read_text()
    commands = re.findall(r"& (xvfb-run [^\n]+?) (?:sh -ec |\$env:OVUI_PYTHON)", workflow)
    # Use each real workflow's Xvfb options; query its X server without a GUI.
    probe = """
import ctypes,json
x=ctypes.CDLL('libX11.so.6')
x.XOpenDisplay.argtypes=[ctypes.c_char_p]
x.XOpenDisplay.restype=ctypes.c_void_p
x.XDefaultScreen.argtypes=[ctypes.c_void_p]
for name in ('XDisplayWidth','XDisplayHeight','XDefaultDepth'):
    getattr(x,name).argtypes=[ctypes.c_void_p,ctypes.c_int]
x.XCloseDisplay.argtypes=[ctypes.c_void_p]
display=x.XOpenDisplay(None)
assert display,'The workflow Xvfb must expose its own display'
screen=x.XDefaultScreen(display)
print(json.dumps([x.XDisplayWidth(display,screen),x.XDisplayHeight(display,screen),x.XDefaultDepth(display,screen)]))
x.XCloseDisplay(display)
"""
    result = subprocess.run([*shlex.split(commands[stage]), sys.executable, "-c", probe],
                            text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    width, height, depth = json.loads(result.stdout)
    assert width >= 1600 and height >= 1000 and depth >= 24, (width, height, depth)


def test_version_gate_checks_explicit_pr_base_without_privileged_events():
    workflow = (ROOT / ".github/workflows/release-version.yml").read_text()
    assert "verify-release-version:" in workflow
    assert "pull_request:" in workflow
    assert "branches:" not in workflow and "branches-ignore:" not in workflow
    assert "types: [opened, synchronize, reopened, edited]" in workflow
    assert "push:" not in workflow and "pull_request_target" not in workflow
    assert "contents: read" in workflow and "contents: write" not in workflow
    assert "secrets." not in workflow and "persist-credentials: false" in workflow
    assert "fetch-depth: 0" in workflow
    assert "OVDNEXT_VERSION_BASE: ${{ github.event.pull_request.base.sha }}" in workflow
    assert 'ci/check_ovui_version.py --base "$OVDNEXT_VERSION_BASE"' in workflow


def test_workflow_separates_read_only_previews_and_protected_publication():
    workflow = (ROOT / ".github/workflows/ovui.yml").read_text()
    assert "pull_request_target" not in workflow and "workflow_run" not in workflow
    assert "contents: read" in workflow and workflow.count("contents: write") == 1
    assert "environment: release" in workflow
    assert "needs.build.result == 'success'" in workflow and "needs.verify-package.result == 'success'" in workflow
    assert "needs: [build, verify-package]" in workflow
    assert "OVDNEXT_PACKAGE_RESULT: ${{ needs.verify-package.result }}" in workflow
    assert "github.ref_protected" in workflow and "github.event.repository.private" in workflow
    assert "github.event.pull_request.head.repo.full_name == github.repository" in workflow
    assert "OVDNEXT_RELEASE_PUBLISH_ENABLED == 'true'" in workflow
    assert "retention-days: 30" in workflow
    graphics = (ROOT / "ci/setup_ovui_graphics_windows.ps1").read_text()
    assert workflow.count("run: ./ci/setup_ovui_graphics_windows.ps1") == 2
    assert "GLCONTEXT_WIN_LIBGL=" in graphics and "GALLIUM_DRIVER=llvmpipe" in workflow
    assert "GALLIUM_DRIVER=llvmpipe" in graphics and "LIBGL_ALWAYS_SOFTWARE=1" in graphics
    assert "ef71c6907d424439f158e5ec102c7915bbf446fb00e04f62daa575b4780c596a" in graphics
    assert "--publish" in workflow and "continue-on-error" not in workflow
    assert "${{ runner.temp }}/ovui-build/package/native-dependencies.txt" in workflow
    for step in workflow.split("uses: actions/checkout@")[1:]:
        assert "persist-credentials: false" in step.split("\n      - ", 1)[0]

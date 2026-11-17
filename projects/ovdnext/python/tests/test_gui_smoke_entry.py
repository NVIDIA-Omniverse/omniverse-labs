# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real source-smoke entry and caller-owned output lifetime regressions."""
import os
from pathlib import Path
import subprocess
import sys

import pytest
from gui_runtime import GUI_AVAILABLE

ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / 'python/tests/display_panel_gui.py'


def child(arguments, directory):
    return subprocess.run([sys.executable, '-B', *arguments], cwd=directory,
        env=os.environ.copy(), capture_output=True, text=True, timeout=60)


def test_display_failure_restores_caller_directory(tmp_path):
    # Real driver setup/cleanup, with initialization stopped before any window.
    code = f'''import os, runpy, sys
sys.path.insert(0, {str(DRIVER.parent)!r})
import omni.ui as ui
output = []
def fail(*args, **kwargs):
    output.append(os.getcwd())
    raise RuntimeError("controlled init failure")
ui.init = fail
driver = runpy.run_path({str(DRIVER)!r})
before = os.getcwd()
preferences = {{name: os.environ.get(name) for name in ('LOCALAPPDATA', 'XDG_DATA_HOME', 'XDG_CONFIG_HOME')}}
try:
    driver["main"]("failure.png")
except RuntimeError as error:
    assert str(error) == "controlled init failure"
else:
    raise AssertionError("initialization failure was swallowed")
assert os.getcwd() == before, (before, os.getcwd())
assert not os.path.exists(output[0])
assert preferences == {{name: os.environ.get(name) for name in preferences}}
assert not os.path.exists("failure.png")
'''
    result = child(['-c', code], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not GUI_AVAILABLE, reason='Requires an owned graphical test runtime')
def test_display_cli_relative_screenshot_survives_process_exit(tmp_path):
    result = child([str(DRIVER), '--screenshot', 'relative.png'], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    if sys.platform == 'win32':
        assert 'Display regression DPI:' in result.stdout
        print(result.stdout)
    for suffix in ('', '-light', '-controls', '-contacts', '-picker'):
        assert (tmp_path / f'relative{suffix}.png').is_file()
    assert not (tmp_path / 'imgui.ini').exists()


@pytest.mark.skipif(not GUI_AVAILABLE, reason='Requires an owned graphical test runtime')
def test_source_smoke_entry_resolves_driver_helper_and_restores_cwd(tmp_path):
    # Only the not-yet-packaged launcher boundary is substituted. The source
    # smoke entry and full Display/native/renderer regression run unchanged.
    bundle = tmp_path / 'source-runtime'; bundle.mkdir()
    (bundle / 'portable_launch.py').write_text(
        'import sys, _pvd_native, ovd_ui.application\nassert sys.argv[1] == "--check"\n',
        encoding='utf-8')
    code = f'''import os, runpy, sys
before = os.getcwd()
sys.argv = ["smoke", "--bundle", {str(bundle)!r}, "--screenshot", "smoke.png"]
runpy.run_path({str(ROOT / 'ci/smoke_ovui_gui.py')!r}, run_name="__main__")
assert os.getcwd() == before, (before, os.getcwd())
'''
    result = child(['-c', code], tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    if sys.platform == 'win32':
        assert 'Display regression DPI:' in result.stdout
        print(result.stdout)
    assert 'Extracted-package GUI rendering and Display controls passed.' in result.stdout
    for suffix in ('', '-light', '-controls', '-contacts', '-picker'):
        assert (tmp_path / f'smoke{suffix}.png').is_file()
    assert not (tmp_path / 'imgui.ini').exists()

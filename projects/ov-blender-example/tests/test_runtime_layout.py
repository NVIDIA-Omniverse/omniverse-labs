from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'runtime'))
from runtime_layout import addon_helper_path


@pytest.mark.parametrize('package', ['ovrtx_blender_example', 'addon/ovrtx_blender_example'])
def test_runtime_helper_resolves_installed_and_source_layout_without_import(tmp_path, package):
    runtime = tmp_path/'runtime'/'ovstream_frames.py'
    runtime.parent.mkdir()
    runtime.write_text('')
    helper = tmp_path/package/'win32_shm.py'
    helper.parent.mkdir(parents=True)
    helper.write_text('raise AssertionError("resolution must not import")')
    assert addon_helper_path(runtime, 'win32_shm.py') == helper


def test_missing_packaged_helper_fails_without_searching_unrelated_sys_path(tmp_path):
    with pytest.raises(FileNotFoundError, match='Missing packaged runtime helper'):
        addon_helper_path(tmp_path/'runtime'/'worker.py', 'win32_shm.py')


@pytest.mark.parametrize('filename', ['../win32_shm.py', '/tmp/foreign.py', 'not_python.dll'])
def test_helper_resolution_cannot_escape_package(tmp_path, filename):
    with pytest.raises(ValueError):
        addon_helper_path(tmp_path/'runtime'/'worker.py', filename)

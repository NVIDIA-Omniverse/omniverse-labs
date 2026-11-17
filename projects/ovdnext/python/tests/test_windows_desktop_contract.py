# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Windows handoff tests simulate the OS boundary; no real handler is opened."""
from types import SimpleNamespace
import os

import pytest

from ovd_ui import desktop


@pytest.mark.parametrize('kind', ['folder', 'support'])
def test_windows_handoff_submits_exact_target_and_owns_no_process(tmp_path, monkeypatch, kind):
    calls = []
    monkeypatch.setattr(desktop, 'sys', SimpleNamespace(platform='win32'))
    monkeypatch.setattr(os, 'startfile', calls.append, raising=False)
    folder = tmp_path / 'folder with spaces ; $value'
    folder.mkdir()
    request = desktop.OpenFolderRequest(folder) if kind == 'folder' else desktop.SupportComposeRequest()
    assert calls == [str(folder.resolve()) if kind == 'folder' else desktop.SUPPORT_URI]
    assert request._process is None and request.poll()
    request.close()
    assert request.poll() and len(calls) == 1


def test_windows_handoff_launch_failure_is_immediate(tmp_path, monkeypatch):
    monkeypatch.setattr(desktop, 'sys', SimpleNamespace(platform='win32'))
    def fail(target):
        raise OSError('handler unavailable')
    monkeypatch.setattr(os, 'startfile', fail, raising=False)
    with pytest.raises(desktop.DesktopError, match='No desktop file manager handler'):
        desktop.OpenFolderRequest(tmp_path)

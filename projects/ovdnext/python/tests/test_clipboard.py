# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public clipboard boundary; real X11 cross-process checks have an explicit display gate."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from ovd_ui import clipboard


@pytest.mark.parametrize("text,error", [(None, TypeError), (b"bytes", TypeError),
                                        ("first\0hidden", ValueError), ("\ud800", ValueError)])
def test_invalid_text_never_loads_or_changes_the_os_clipboard(monkeypatch, text, error):
    monkeypatch.setattr(clipboard, "_backend", None)
    monkeypatch.setattr(clipboard, "_load_backend", lambda: pytest.fail("invalid text touched the OS"))
    with pytest.raises(error):
        clipboard.copy_text(text)


def test_worker_call_rejected_before_loading_or_mutating_backend(monkeypatch):
    monkeypatch.setattr(clipboard, "_backend", None)
    monkeypatch.setattr(clipboard, "_load_backend", lambda: pytest.fail("worker touched the OS"))
    with ThreadPoolExecutor(max_workers=1) as executor:
        with pytest.raises(clipboard.ClipboardError, match="main thread"):
            executor.submit(clipboard.copy_text, "property value").result()


@pytest.mark.parametrize("text", ["", "line one\nline two\t 1.25", "物理 μ 🙂", "x" * 200000],
    ids=['empty', 'whitespace', 'unicode', 'large'])
def test_exact_utf8_text_is_owned_by_backend_without_normalization(monkeypatch, text):
    values = []
    monkeypatch.setattr(clipboard, "_backend", SimpleNamespace(copy=values.append))
    assert clipboard.copy_text(text) is None
    assert values == [text.encode("utf-8")]


def test_real_public_glfw_exports_reject_uninitialized_runtime():
    # Importing OVUI loads its library but creates no window/GPU context.
    backend = clipboard._load_backend()
    with pytest.raises(clipboard.ClipboardError, match="active OVUI desktop"):
        backend.copy(b"uninitialized")


def test_missing_exports_fail_instead_of_falling_back_to_private_test_clipboard():
    with pytest.raises(clipboard.ClipboardError, match="public desktop clipboard API"):
        clipboard._GlfwClipboard(SimpleNamespace())

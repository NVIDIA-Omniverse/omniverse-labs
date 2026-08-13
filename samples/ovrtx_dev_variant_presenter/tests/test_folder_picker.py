# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the native folder-dialog picker.

These import ONLY ``dev_variant_presenter.api.folder_picker`` — a stdlib-only module — so the
crash-safety of Browse is verified in isolation, without standing up the FastAPI app,
the render runtime, or USD. Tkinter itself is stubbed via ``sys.modules`` so no real
dialog opens.
"""
import os
import sys
import threading
import types

import dev_variant_presenter.api.folder_picker as FP


def _install_fake_tk(monkeypatch, askdirectory, on_create=None):
    class _FakeTk:
        def __init__(self):
            if on_create:
                on_create()

        def withdraw(self):
            pass

        def attributes(self, *a, **k):
            pass

    fake_tk = types.ModuleType("tkinter")
    fake_tk.Tk = _FakeTk
    fake_filedialog = types.ModuleType("tkinter.filedialog")
    fake_filedialog.askdirectory = askdirectory
    fake_tk.filedialog = fake_filedialog
    monkeypatch.setitem(sys.modules, "tkinter", fake_tk)
    monkeypatch.setitem(sys.modules, "tkinter.filedialog", fake_filedialog)


def test_pins_every_dialog_to_one_thread(monkeypatch):
    """Regression: Browse must not run Tkinter on transient executor threads.

    The old handler created a fresh ``tk.Tk()`` per request on a ``run_in_executor``
    worker. A second Browse landing on a different worker finalized a Tcl interpreter
    on the wrong thread, tripping the native ``Tcl_AsyncDelete`` abort that killed the
    process and dropped the viewport. Every dialog must run on ONE dedicated thread
    with a single reused root.
    """
    creator_threads = []
    dialog_threads = []

    def _askdirectory(title=None):
        dialog_threads.append(threading.get_ident())
        return ""

    _install_fake_tk(
        monkeypatch, _askdirectory,
        on_create=lambda: creator_threads.append(threading.get_ident()))

    picker = FP.FolderPicker()
    for _ in range(5):
        assert picker.pick() == ""

    # exactly one Tk root, created once, and every dialog ran on its thread
    assert creator_threads == [creator_threads[0]] and len(creator_threads) == 1
    assert set(dialog_threads) == {creator_threads[0]}


def test_single_flight_drops_concurrent_request(monkeypatch):
    """A second Browse while a dialog is open is a no-op, not a queued dialog."""
    dialog_calls = []
    in_dialog = threading.Event()
    release = threading.Event()

    def _askdirectory(title=None):
        dialog_calls.append(1)
        in_dialog.set()
        release.wait(2.0)   # hold the dialog "open" until the test releases it
        return "C:/picked"

    _install_fake_tk(monkeypatch, _askdirectory)
    monkeypatch.setattr(FP, "_picker", None)
    monkeypatch.setattr(FP, "_busy", threading.Lock())

    first_result = {}
    t = threading.Thread(target=lambda: first_result.setdefault("v", FP.pick_folder()))
    t.start()
    assert in_dialog.wait(2.0)             # first dialog is open
    assert FP.pick_folder() == ""          # concurrent Browse is dropped
    release.set()
    t.join(2.0)

    assert dialog_calls == [1]             # only ONE dialog ever opened
    assert first_result["v"] == os.path.normpath("C:/picked")

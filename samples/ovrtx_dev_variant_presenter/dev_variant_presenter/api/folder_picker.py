# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native folder-dialog picker — deliberately stdlib-only so it unit-tests in isolation.

Tkinter/Tcl is not thread-safe. Creating a fresh ``tk.Tk()`` per request on a transient
``run_in_executor`` worker thread is the obvious implementation and it is unsafe: when a
second dialog lands on a different worker (Browse clicked twice), Python can finalize the
first Tcl interpreter on a thread other than the one that created it, tripping the native
``Tcl_AsyncDelete: async handler deleted by the wrong thread`` abort. That kills the whole
process and drops the ovstream viewport.

This module avoids that class of crash two ways: every dialog runs on ONE dedicated,
long-lived thread with a single reused, never-destroyed root (no cross-thread create /
teardown), and the pick is single-flight (a concurrent request returns "" instead of
queuing a second dialog). It imports nothing from the app/render/USD stack on purpose —
the picker is a self-contained concern and its tests must not have to stand up the app.
"""
from __future__ import annotations

import queue
import threading

# Never wait on the picker thread forever: __init__ runs inside the HTTP request while
# holding the module locks, so a wedged thread would hang Browse permanently.
_READY_TIMEOUT_S = 20.0

# Every no-dialog failure carries the same actionable hint. Tk is an OS package, not a pip
# dependency: it is simply absent on a stock Linux python, and absent-by-design headless.
_NO_DIALOG_HINT = ("no native folder dialog on this host — on Linux install Tk "
                   "(`sudo apt install python3-tk`) and run with a desktop session; "
                   "headless hosts have no dialog. Type the folder path instead.")


class FolderPicker:
    """Serializes every native folder dialog onto ONE dedicated, long-lived thread."""

    def __init__(self) -> None:
        self._requests: "queue.Queue" = queue.Queue()
        self._ready = threading.Event()
        self._init_error: str | None = None
        self._thread = threading.Thread(
            target=self._serve, name="folder-picker", daemon=True)
        self._thread.start()
        if not self._ready.wait(_READY_TIMEOUT_S):
            # the thread died or stalled without reporting — fail the ctor rather than
            # block the caller (and the single-flight lock it holds) forever.
            self._init_error = (
                f"folder dialog did not start within {_READY_TIMEOUT_S:.0f}s — "
                f"{_NO_DIALOG_HINT}")
        if self._init_error:
            raise RuntimeError(self._init_error)

    def _serve(self) -> None:
        # tkinter is imported INSIDE the try: on Linux it is a separate OS package, so a
        # missing-module error must land in _init_error like any other Tk failure —
        # escaping the thread target would leave __init__ waiting on an event nobody sets.
        try:
            import os
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
        except Exception as exc:  # no Tk module / no display — fail the ctor
            self._init_error = f"{exc} — {_NO_DIALOG_HINT}"
            self._ready.set()
            return
        self._ready.set()
        while True:
            reply: "queue.Queue" = self._requests.get()
            try:
                root.attributes("-topmost", True)
                path = filedialog.askdirectory(title="Choose a folder")
                reply.put(("ok", os.path.normpath(path) if path else ""))
            except Exception as exc:  # never let a dialog error kill the thread
                reply.put(("err", str(exc)))

    def pick(self) -> str:
        reply: "queue.Queue" = queue.Queue()
        self._requests.put(reply)
        status, value = reply.get()
        if status == "err":
            raise RuntimeError(value)
        return value


_picker: "FolderPicker | None" = None
_picker_lock = threading.Lock()
_busy = threading.Lock()


def pick_folder() -> str:
    """Lazily start the dedicated picker thread, then run one folder dialog on it.

    Single-flight: if a dialog is already open, a concurrent request returns "" (treated
    as a cancel) instead of queuing a second dialog behind the first.

    Raises RuntimeError when the host has no usable dialog (no Tk module, no display);
    callers should degrade to manual path entry rather than error out.
    """
    global _picker
    if not _busy.acquire(blocking=False):
        return ""
    try:
        with _picker_lock:
            if _picker is None:
                _picker = FolderPicker()
        return _picker.pick()
    finally:
        _busy.release()

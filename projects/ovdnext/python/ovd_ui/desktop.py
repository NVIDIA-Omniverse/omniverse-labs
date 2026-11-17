# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Nonblocking desktop handoffs; never send mail or own the user's application."""
from pathlib import Path
import os
import subprocess
import sys
from threading import Thread


SUPPORT_EMAIL = "pvdsupport@nvidia.com"
SUPPORT_URI = "mailto:pvdsupport@nvidia.com?subject=PVD%20Issue"
SUPPORT_FALLBACK = "Email pvdsupport@nvidia.com with subject PVD Issue."


class DesktopError(RuntimeError):
    pass


def _launch(target):
    """Submit one validated target; Windows transfers ownership immediately."""
    if sys.platform == "win32":
        os.startfile(target)
        return None
    return subprocess.Popen(["xdg-open", target], stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)


class _DesktopRequest:
    def __init__(self, target, *, unavailable, missing, failed):
        self._failed = failed
        if sys.platform not in ("linux", "win32"):
            raise DesktopError(unavailable)
        try:
            # One absolute path or fixed URI argument; no shell interpretation.
            self._process = _launch(target)
        except OSError as error:
            raise DesktopError(missing) from error

    def poll(self):
        """Return False while pending, True on handoff; raise on launcher failure."""
        if self._process is None:
            return True
        result = self._process.poll()
        if result is None:
            return False
        self._process = None
        if result != 0:
            raise DesktopError(self._failed(result))
        return True

    def close(self):
        # A desktop handler can remain the user's composer/file manager. Never
        # terminate it when this viewer closes; reap without retaining the UI.
        process, self._process = self._process, None
        if process is not None and process.poll() is None:
            Thread(target=process.wait, name="pvd-desktop-handler", daemon=True).start()


class SupportComposeRequest(_DesktopRequest):
    """Poll one support compose request; this does not send a message."""
    def __init__(self):
        super().__init__(SUPPORT_URI,
            unavailable="Opening an email composer is unavailable on this platform. " + SUPPORT_FALLBACK,
            missing="No desktop email handler could be started. " + SUPPORT_FALLBACK,
            failed=lambda code: f"The desktop email handler failed (exit {code}). " + SUPPORT_FALLBACK)


class OpenFolderRequest(_DesktopRequest):
    """Open an existing resolved directory; its creation belongs to the caller.

    ``path`` is the exact absolute filesystem path passed as one argument to the
    desktop handler. Successful polling confirms the handoff, not the presence
    or continued lifetime of a visible file-manager window. Windows uses the
    documented ShellExecute handoff; Linux polls and reaps its xdg-open process.
    """
    def __init__(self, path):
        try:
            self.path = Path(path).expanduser().resolve(strict=True)
            if not self.path.is_dir():
                raise ValueError("Not a directory")
        except (OSError, RuntimeError, ValueError) as error:
            raise DesktopError(f"Could not open recordings folder: {path}. {error}") from error
        fallback = f"Open this folder in your file manager: {self.path}"
        super().__init__(str(self.path),
            unavailable="Opening a folder is unavailable on this platform. " + fallback,
            missing="No desktop file manager handler could be started. " + fallback,
            failed=lambda code: f"The desktop file manager handler failed (exit {code}). " + fallback)

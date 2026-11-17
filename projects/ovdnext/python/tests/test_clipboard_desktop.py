# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real X11 clipboard interoperability, opt-in and separate from GPU benchmarks.

OVD_CLIPBOARD_TEST_DISPLAY=:104 OVD_TEST_XCLIP=/path/to/xclip pytest ...
The caller owns that X display. No product dependency on xclip is introduced.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

DISPLAY = os.environ.get("OVD_CLIPBOARD_TEST_DISPLAY")
XCLIP = os.environ.get("OVD_TEST_XCLIP")
pytestmark = pytest.mark.skipif(not DISPLAY and not XCLIP, reason="Requires an explicitly reserved X11 display and external xclip")


class Desktop:
    def __init__(self, directory):
        if not DISPLAY or not XCLIP:
            pytest.fail("Clipboard opt-in requires both OVD_CLIPBOARD_TEST_DISPLAY and OVD_TEST_XCLIP")
        if not Path(XCLIP).is_file() or not os.access(XCLIP, os.X_OK):
            pytest.fail("OVD_TEST_XCLIP must name an executable file")
        self.directory = directory
        runtime = directory / "xdg-runtime"
        runtime.mkdir(mode=0o700)
        self.env = {**os.environ, "DISPLAY": DISPLAY, "XDG_RUNTIME_DIR": str(runtime)}
        self.counter = 0
        self.log = (directory / "desktop.log").open("w")
        self.process = subprocess.Popen([sys.executable, str(Path(__file__).resolve().with_name("clipboard_desktop_app.py")), str(directory)],
                                        cwd=directory, env=self.env, stdin=subprocess.PIPE, stdout=self.log, stderr=subprocess.STDOUT, text=True)
        try:
            self.wait("ready")
        except BaseException:
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait(timeout=5)
            self.log.close()
            raise

    def wait(self, identity):
        path = self.directory / f"{identity}.json"
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if path.is_file():
                try:
                    return json.loads(path.read_text())
                except json.JSONDecodeError:
                    pass  # A different process is completing the bounded response write.
            if self.process.poll() is not None:
                pytest.fail((self.directory / "desktop.log").read_text())
            time.sleep(.01)
        pytest.fail(f"Clipboard GUI did not reply: {identity}")

    def command(self, operation, text=""):
        self.counter += 1
        self.process.stdin.write(json.dumps({"id": str(self.counter), "operation": operation, "text": text}) + "\n")
        self.process.stdin.flush()
        return self.wait(str(self.counter))

    def paste(self):
        return subprocess.run([XCLIP, "-selection", "clipboard", "-target", "UTF8_STRING", "-out"],
                              env=self.env, capture_output=True, timeout=5)

    def close(self):
        if self.process.poll() is None:
            try:
                self.command("stop")
                self.process.wait(timeout=5)
            finally:
                if self.process.poll() is None:
                    self.process.kill()
                    self.process.wait(timeout=5)
        self.log.close()


@pytest.fixture
def desktop(tmp_path):
    app = Desktop(tmp_path)
    try:
        yield app
    finally:
        app.close()


def test_external_unicode_large_empty_paste_and_owner_shutdown(desktop):
    for text in ("native μ 物理 🙂\nproperty\t1.25", "large Δ " * 40000, "", "keep existing"):
        assert desktop.command("copy", text) == {"ok": True}
        pasted = desktop.paste()
        assert pasted.returncode == 0, pasted.stderr
        assert pasted.stdout == text.encode("utf-8")
    result = desktop.command("copy", "visible\0silently truncated")
    assert result["error_type"] == "ValueError"
    assert desktop.paste().stdout == b"keep existing"
    desktop.close()
    # Reserved Xvfb has no clipboard manager. GLFW's documented ownership ends
    # with its window/event loop; do not pretend the process-local bytes persist.
    assert desktop.paste().returncode != 0


def test_external_copy_replaces_ownership_and_survives_pvd_exit(desktop):
    assert desktop.command("copy", "original PVD property")["ok"]
    expected = "external application owns μ clipboard".encode("utf-8")
    owner = subprocess.Popen([XCLIP, "-selection", "clipboard", "-target", "UTF8_STRING", "-in", "-quiet"],
                             env=desktop.env, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        owner.stdin.write(expected)
        owner.stdin.close()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if desktop.paste().stdout == expected:
                break
            time.sleep(.01)
        else:
            pytest.fail("External application did not acquire the OS clipboard")
        desktop.close()
        pasted = desktop.paste()
        assert pasted.returncode == 0
        assert pasted.stdout == expected  # PVD never restores or clears another owner.
    finally:
        owner.terminate()
        owner.wait(timeout=5)

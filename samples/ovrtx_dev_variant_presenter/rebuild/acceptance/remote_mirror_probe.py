# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cheap, offline probe for the "remote stage mirroring" feature (area 9).

Serves a tiny 2-file USD closure (a.usda references b.usda, b defines a variant
set) over a localhost HTTP server, hands the app the root URL, and lets the caller
assert — through the PUBLIC surface only — that the app:
  (a) fetched + composed the whole closure (the scan reports b.usda's variant set), and
  (b) never wrote the served source (sha256 of both files identical before/after).

No S3, no GPU dependency beyond the app already running. Importable by grade_http.py
or runnable standalone for a smoke of the probe itself.
"""
from __future__ import annotations

import contextlib
import hashlib
import http.server
import socket
import tempfile
import threading
from pathlib import Path

A_USDA = """#usda 1.0
(
    defaultPrim = "World"
)

def Xform "World"
{
    def "Ref" (
        prepend references = @./b.usda@</Thing>
    )
    {
    }
}
"""

# b defines a variant set "probe" with two variants -> if the app's scan reports it,
# the reference (and thus the closure) was fetched and composed from the mirror.
B_USDA = """#usda 1.0
(
    defaultPrim = "Thing"
)

def Xform "Thing" (
    variantSets = "probe"
    variants = {
        string probe = "a"
    }
)
{
    variantSet "probe" = {
        "a" {
            over "marker_a" {}
        }
        "b" {
            over "marker_b" {}
        }
    }
}
"""


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class MirrorFixture:
    """Context manager: temp dir with a.usda+b.usda served over localhost HTTP."""

    def __init__(self):
        self.dir = Path(tempfile.mkdtemp(prefix="vs_mirror_probe_"))
        (self.dir / "a.usda").write_text(A_USDA, encoding="utf-8")
        (self.dir / "b.usda").write_text(B_USDA, encoding="utf-8")
        self._sha0 = {n: _sha(self.dir / n) for n in ("a.usda", "b.usda")}
        self.port = _free_port()
        self._httpd = None
        self._thread = None

    @property
    def root_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/a.usda"

    def start(self):
        directory = str(self.dir)

        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *a, **kw):
                super().__init__(*a, directory=directory, **kw)

            def log_message(self, *a):  # quiet
                pass

        self._httpd = http.server.ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def source_unchanged(self) -> bool:
        return all(_sha(self.dir / n) == self._sha0[n] for n in self._sha0)

    def stop(self):
        if self._httpd:
            with contextlib.suppress(Exception):
                self._httpd.shutdown()
                self._httpd.server_close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


if __name__ == "__main__":
    with MirrorFixture() as fx:
        print("serving:", fx.root_url)
        print("source_unchanged:", fx.source_unchanged())
        input("Open the URL in the app, then press Enter to re-check...")
        print("source_unchanged after:", fx.source_unchanged())

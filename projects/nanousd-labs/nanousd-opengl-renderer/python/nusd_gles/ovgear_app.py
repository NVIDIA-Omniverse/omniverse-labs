# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility launcher for the retired ovgear/GLES viewer path.

The public OpenGL viewer contract is now the shared nanousdview frontend
driving ``ovrtx.Renderer`` with ``--backend opengl``. Keep this module so
older commands such as ``python -m nusd_gles.ovgear_app scene.usda`` keep
working, but route them through the OVRTX facade instead of the old direct
``GlesRendererAdapter`` monkey patches.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence


def _workspace_from_here() -> Path:
    return Path(__file__).resolve().parents[3]


def _prepend_import_path(path: Path) -> None:
    if path.is_dir():
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def _strip_backend_override(args: Sequence[str]) -> list[str]:
    """This compatibility module is specifically the OpenGL launcher."""
    stripped: list[str] = []
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg == "--backend":
            skip_next = True
            continue
        if arg.startswith("--backend="):
            continue
        stripped.append(arg)
    return stripped


def main(argv: Sequence[str] | None = None) -> int:
    workspace = _workspace_from_here()
    _prepend_import_path(workspace / "nanousdview" / "python")

    from nanousdview.__main__ import main as nanousdview_main

    forwarded = [
        "--backend",
        "opengl",
        *_strip_backend_override(sys.argv[1:] if argv is None else argv),
    ]
    old_argv = sys.argv[:]
    sys.argv = ["nanousdview", *forwarded]
    try:
        return nanousdview_main(forwarded)
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())

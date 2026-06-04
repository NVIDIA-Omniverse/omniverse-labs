# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the ov-libaries-livestream ContactBinding tutorial from the viewer example tree.

The real sample lives next to the livestream tutorials; this wrapper keeps the viewer
spawning a path under ``usd-viewer-example/`` (same pattern as ``physx_subprocess_sim.py``).
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    here = Path(__file__).resolve().parent
    target = here.parent / "ov-libaries-livestream" / "contact_binding.py"
    if not target.is_file():
        print(f"error: ContactBinding script not found: {target}", file=sys.stderr)
        raise SystemExit(2)
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Clone ``/World`` (or another subtree) then place the clone root with scale (1,1,1) and a +X offset.

Flow:

1. ``clone_usd(source, [target])`` — see ``skills/cloning-prims/SKILLS.md``.
2. ``clone_offset_layer_usd.py`` (usd-core subprocess) returns the source prim’s **local** matrix
   composed with **parent +X** translation ``CLONE_OFFSET_X`` (layout: “to the right” of the source).
3. The 3×3 linear part is **orthonormalized** (SVD) so ``omni:xform`` has uniform scale **(1,1,1)**;
   translation row is unchanged.

Requires ``uv`` on PATH with ``uv run --with usd-core``, or ``OVRTX_PXR_PYTHON`` pointing at a Python
with ``usd-core`` (no ``pxr`` in the ovrtx process).

**USD viewer** calls ``run_clone_robot_on_renderer(renderer, same_string_as_open_usd, ...)``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from ovrtx import Device, PrimMode, Renderer, Semantic
from PIL import Image

# -----------------------------------------------------------------------------
# Viewer notes (what this script demonstrates)
# -----------------------------------------------------------------------------
# - Clone a USD subtree (here ``/World``) to a new prim path via ``clone_usd``.
# - Position the clone by writing ``omni:xform`` on the clone root: same orientation as the
#   source, shifted +X in parent space, with uniform scale (1,1,1).
# - The offset matrix is computed in a **separate** Python process that has ``usd-core`` /
#   ``pxr``, because the ovrtx app may not bundle OpenUSD in-process.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Stage, paths, offset, subprocess helper
# -----------------------------------------------------------------------------
USD_URL = (
    "https://omniverse-content-production.s3.us-west-2.amazonaws.com/"
    "Samples/Robot-OVRTX/robot-ovrtx.usda"
)

DEFAULT_SOURCE_PRIM = "/World"
CLONE_TARGET_PRIM = "/World_clone"

# Parent-space +X translation in stage units (see ``clone_offset_layer_usd.py`` third argument).
CLONE_OFFSET_X = 5.0

_ENV_SOURCE = "OVRTX_CLONE_ROBOT_SOURCE_PRIM"
_ENV_PXR_PYTHON = "OVRTX_PXR_PYTHON"

OUTPUT_PNG = Path(__file__).resolve().parent / "clone_example.png"
_PXR_SCRIPT = Path(__file__).resolve().parent / "clone_offset_layer_usd.py"


def _run_usd_core_python(script: Path, script_argv: list[str]) -> str:
    """Run ``clone_offset_layer_usd.py`` (or similar) and return its stdout (16 floats)."""
    if not script.is_file():
        raise FileNotFoundError(f"Missing helper script: {script}")
    pxr_py = os.environ.get(_ENV_PXR_PYTHON, "").strip()
    if pxr_py:
        args = [pxr_py, str(script), *script_argv]
    else:
        uv = shutil.which("uv")
        if not uv:
            raise RuntimeError(
                "Need OpenUSD in a separate process: put `uv` on PATH or set "
                f"{_ENV_PXR_PYTHON} to a Python with `usd-core`."
            )
        args = [uv, "run", "--with", "usd-core", "python", str(script), *script_argv]
    child_env = os.environ.copy()
    # Strip env that could pull the wrong USD/Omniverse libs into the child.
    for key in ("PYTHONPATH", "PXR_PLUGINPATH", "USD_PATH", "OMNIVERSE_PATH"):
        child_env.pop(key, None)
    proc = subprocess.run(args, check=False, capture_output=True, text=True, env=child_env)
    if proc.returncode != 0:
        raise RuntimeError(f"{script.name} failed:\n  stdout: {proc.stdout!r}\n  stderr: {proc.stderr!r}")
    return proc.stdout


def _matrix_from_pxr_line(stdout: str) -> np.ndarray:
    """Parse one line of 16 space-separated floats into a 4×4 ``numpy`` matrix."""
    parts = stdout.strip().split()
    if len(parts) != 16:
        raise RuntimeError(f"expected 16 matrix coefficients, got {len(parts)}: {stdout!r}")
    return np.array([float(x) for x in parts], dtype=np.float64).reshape(4, 4)


def _clone_root_xform_local_times_offset_x(
    usd_url_or_path: str,
    source_prim_path: str,
    offset_x: float,
) -> np.ndarray:
    # Helper script: source prim’s local matrix × parent +X translation (see its argv).
    stdout = _run_usd_core_python(
        _PXR_SCRIPT,
        [usd_url_or_path, source_prim_path, str(offset_x)],
    )
    return _matrix_from_pxr_line(stdout)


def _linear_part_scale_one(m: np.ndarray) -> np.ndarray:
    """Closest rotation matrix in the upper 3×3 (scale 1,1,1); keep translation row."""
    # Composition can leave non-uniform scale in the 3×3; SVD gives a proper rotation for omni:xform.
    out = m.astype(np.float64, copy=True)
    r = out[:3, :3]
    u, _, vt = np.linalg.svd(r)
    rot = u @ vt
    if np.linalg.det(rot) < 0.0:
        u = u.copy()
        u[:, 2] *= -1.0
        rot = u @ vt
    out[:3, :3] = rot
    return out


def apply_clone(renderer: Renderer, source: str, target: str, usd_url_or_path: str) -> None:
    """``clone_usd`` then ``omni:xform`` on ``target``: +X offset, uniform scale 1."""
    renderer.clone_usd(source, [target])
    m = _clone_root_xform_local_times_offset_x(usd_url_or_path, source, CLONE_OFFSET_X)
    m = _linear_part_scale_one(m)
    # map_attribute: one row per prim → shape (1, 4, 4) for a single clone root.
    mat = np.ascontiguousarray(m[np.newaxis, ...])
    with renderer.map_attribute(
        prim_paths=[target],
        attribute_name="omni:xform",
        semantic=Semantic.XFORM_MAT4x4,
        prim_mode=PrimMode.EXISTING_ONLY,
        device=Device.CPU,
    ) as mapping:
        dest = np.from_dlpack(mapping.tensor)
        if dest.shape != mat.shape:
            raise RuntimeError(f"map_attribute shape {dest.shape} != {mat.shape}")
        dest[:] = mat


def run_clone_robot_on_renderer(
    renderer: Renderer,
    usd_url_or_path: str,
    *,
    robot_source_override: str | None = None,
) -> str:
    # Entry point for embedding (e.g. usd-viewer-example): returns resolved source prim path.
    if robot_source_override and robot_source_override.strip():
        source = robot_source_override.strip()
    elif os.environ.get(_ENV_SOURCE, "").strip():
        source = os.environ[_ENV_SOURCE].strip()
    else:
        source = DEFAULT_SOURCE_PRIM

    apply_clone(renderer, source, CLONE_TARGET_PRIM, usd_url_or_path)
    return source


def main() -> None:
    # Standalone demo: renderer → load stage → clone + place → one camera step → save LDR PNG.
    print("Creating renderer…", file=sys.stderr)
    renderer = Renderer()
    print(f"Loading {USD_URL}…", file=sys.stderr)
    renderer.open_usd(USD_URL)

    src = os.environ.get(_ENV_SOURCE, "").strip() or DEFAULT_SOURCE_PRIM
    if os.environ.get(_ENV_SOURCE, "").strip():
        print(f"Source prim (from {_ENV_SOURCE}): {src}", file=sys.stderr)
    print(
        f"clone + omni:xform (+X={CLONE_OFFSET_X}, scale 1,1,1) {src!r} → {CLONE_TARGET_PRIM!r}…",
        file=sys.stderr,
    )
    apply_clone(renderer, src, CLONE_TARGET_PRIM, USD_URL)

    # Stage must expose this RenderProduct (see ovrtx USD / skills for valid paths).
    print("Stepping /Render/Camera…", file=sys.stderr)
    products = renderer.step(
        render_products={"/Render/Camera"},
        delta_time=1.0 / 60.0,
    )
    if products is None or "/Render/Camera" not in products:
        raise RuntimeError("step() did not return /Render/Camera")
    frame = products["/Render/Camera"].frames[0]
    # LdrColor: display-referred RGB for saving as a normal PNG.
    with frame.render_vars["LdrColor"].map(device=Device.CPU) as var:
        Image.fromarray(var.tensor.numpy()).save(OUTPUT_PNG)
    print(f"Saved {OUTPUT_PNG}", file=sys.stderr)


if __name__ == "__main__":
    main()

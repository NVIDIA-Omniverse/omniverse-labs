# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline texture-loading debug harness.

Runs three load passes through `GlesViewer` against:

  1. tests/textures_debug/textured_cube.usda — synthetic UsdPreviewSurface
     reference to a local PNG. If THIS reports `0 textures`, the loader
     is broken on the simplest case.
  2. test_pbr_materials.usda — repo's bundled PBR test scene.
  3. Isaac Simple_Warehouse Cardbox_A1.usd (or warehouse `full_warehouse.usd`)
     — the real-world Omniverse asset that prompted this debug session.

For each, captures stderr from the C side, extracts the
`material: loaded N materials, M textures` line, and renders one frame
to a PPM dump so we can eyeball whether texturing is on or off.

Usage:
    PYTHONPATH=python python tests/textures_debug/run_debug.py
"""

from __future__ import annotations

import os
import re
import sys
import contextlib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent

# Fallback test asset paths
SYNTHETIC = REPO / "tests/textures_debug/textured_cube.usda"
BUNDLED   = REPO / "test_pbr_materials.usda"
ASSET_ROOT = Path(os.environ.get("NUSD_ASSET_ROOT", str(Path.home() / "assets")))
WAREHOUSE_PROP = Path(
    os.environ.get(
        "NUSD_WAREHOUSE_PROP",
        str(ASSET_ROOT / "nvidia-simready-warehouse/Props/general/Cardbox_A1/Cardbox_A1.usd"),
    )
)
WAREHOUSE_FULL = Path(
    os.environ.get(
        "NUSD_ISAAC_WAREHOUSE",
        str(ASSET_ROOT / "isaac/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"),
    )
)

CASES = [
    ("synthetic-textured-cube", SYNTHETIC),
    ("test_pbr_materials",      BUNDLED),
    ("simready-cardbox",        WAREHOUSE_PROP),
    ("isaac-full-warehouse",    WAREHOUSE_FULL),
]

# ---------------------------------------------------------------------------
# stderr capture: redirect at FD level so C-side fprintf(stderr, ...) lands
# in our buffer.
# ---------------------------------------------------------------------------

class _StderrTap:
    def __init__(self):
        self._real_fd = os.dup(2)
        self._read_fd, self._write_fd = os.pipe()
        os.dup2(self._write_fd, 2)
        os.close(self._write_fd)

    def collect(self) -> str:
        sys.stderr.flush()
        os.dup2(self._real_fd, 2)
        os.close(self._real_fd)
        # Drain pipe non-blockingly.
        import fcntl
        fcntl.fcntl(self._read_fd, fcntl.F_SETFL, os.O_NONBLOCK)
        chunks = []
        while True:
            try:
                chunk = os.read(self._read_fd, 4096)
            except BlockingIOError:
                break
            if not chunk:
                break
            chunks.append(chunk)
        os.close(self._read_fd)
        return b"".join(chunks).decode(errors="replace")


@contextlib.contextmanager
def capture_stderr():
    tap = _StderrTap()
    try:
        yield tap
    finally:
        pass


def run_case(label: str, usd_path: Path) -> dict:
    print(f"\n=== {label}: {usd_path}")
    if not usd_path.exists():
        print(f"  SKIP — file missing")
        return {"label": label, "skipped": True}

    # Import lazily so the PYTHONPATH set by the caller takes effect.
    sys.path.insert(0, str(REPO / "python"))
    import numpy as np
    from nusd_gles._bindings import GlesViewer

    tap = _StderrTap()
    try:
        viewer = GlesViewer(str(usd_path), 256, 256)
        bounds = viewer.get_scene_bounds()
        out = np.empty((256, 256, 4), dtype=np.uint8)
        # Identity-ish view + perspective; whatever the bounds, we just
        # need *some* render so glReadPixels fires and material binding
        # actually runs through gpu_cmd_bind_material.
        view = np.eye(4, dtype=np.float32)
        view[2, 3] = -max(0.5, np.linalg.norm(np.asarray(bounds[1]) - np.asarray(bounds[0])))
        proj = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, -1.001, -0.001],
                         [0, 0, -1, 0]], dtype=np.float32)
        eye = np.array([0, 0, 5], dtype=np.float32)
        ok = viewer.render(256, 256, view, proj, eye, out=out)
        viewer.close()
    finally:
        log = tap.collect()

    # Parse the C-side log lines we care about.
    materials = textures = meshes = vertices = None
    m = re.search(r"material: loaded (\d+) materials, (\d+) textures", log)
    if m:
        materials, textures = int(m.group(1)), int(m.group(2))
    m = re.search(r"\s+(\d+) meshes,\s+(\d+) vertices", log)
    if m:
        meshes, vertices = int(m.group(1)), int(m.group(2))

    # Save a PPM of the render.
    dump_dir = REPO / "tests/textures_debug/dumps"
    dump_dir.mkdir(exist_ok=True)
    ppm_path = dump_dir / f"{label}.ppm"
    h, w, _ = out.shape
    with open(ppm_path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode())
        f.write(out[..., :3].tobytes())

    print(f"  meshes={meshes}  vertices={vertices}  materials={materials}  textures={textures}  render_ok={ok}")
    print(f"  pixel_mean={out[..., :3].mean():.1f}  unique_colors_thumbnail={len(np.unique(out[::8, ::8, :3].reshape(-1, 3), axis=0))}")
    print(f"  dump: {ppm_path}")

    # Surface any "asset path resolve failed" or "texture file not found" lines.
    interesting = [ln for ln in log.splitlines()
                   if any(k in ln.lower() for k in (
                       "texture", "material", "asset", "resolve",
                       "missing", "failed", "not found", "skip"))]
    if interesting:
        print("  notable lines:")
        for ln in interesting[:20]:
            print(f"    | {ln}")

    return {"label": label, "materials": materials, "textures": textures,
            "meshes": meshes, "vertices": vertices, "render_ok": ok}


def main() -> int:
    print(f"REPO = {REPO}")
    results = [run_case(lbl, p) for lbl, p in CASES]

    print("\n=== summary ===")
    for r in results:
        if r.get("skipped"):
            print(f"  {r['label']}: SKIPPED")
        else:
            verdict = "OK" if (r.get("textures") or 0) > 0 else "NO TEXTURES"
            print(f"  {r['label']}: meshes={r['meshes']} mats={r['materials']} "
                  f"texs={r['textures']} → {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

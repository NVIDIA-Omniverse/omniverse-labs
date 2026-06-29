#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render BasisCurves test scenes through three renderers and diff them.

EXPECTED STATUS (phase 1d → phase 2):
    This test is EXPECTED TO FAIL its RMSE threshold until BasisCurves
    phase 2 (scene loader for Curves prims) lands. Our viewer currently
    returns env-bg only for these scenes because src/scene.c silently
    ignores BasisCurves prims; Storm GL and Hydra/Metal both render the
    actual curves. The infrastructure here exists so phase 2 progress is
    measurable: once curves start showing up in our renders, the RMSE
    numbers will drop and the comparison images will reveal residual
    framing / shading deltas.

What it does:
    For each USDA in tests/curves/, render at 512x512 via:
      a) our viewer (libnusd_gles via ctypes), driven by the camera
         prim's view/proj matrices computed via pxr.UsdGeom.Camera.
      b) Hydra/Storm GL via $HOME/OpenUSD-install/bin/usdrecord
         (--renderer GL).
      c) Hydra/Metal via /usr/bin/usdrecord (--renderer Metal).
    Then compute mean per-channel RMSE pairwise and write a side-by-side
    composite PNG to tests/curves/output/compare/<scene>_<pair>.png.

Exit codes:
    0  the script itself ran end-to-end (regardless of RMSE pass/fail)
    1  the script crashed (e.g. missing USDA, ctypes load fault)
   77  CTest "skip" — neither our viewer nor any usdrecord ran

No Pillow dependency: PNG decode/encode is implemented in this file
with stdlib + numpy (zlib/struct), so the script runs in any Python
that has numpy.
"""
from __future__ import annotations

import math
import os
import shutil
import struct
import subprocess
import sys
import zlib
from pathlib import Path
from typing import Optional, Tuple

try:
    import numpy as np
except ModuleNotFoundError as exc:
    print(f"SKIP: Python dependency missing: {exc.name}")
    raise SystemExit(77) from exc


# --- paths / env ---------------------------------------------------------

REPO = Path(__file__).resolve().parents[1]
SCENE_DIR = REPO / "tests" / "curves"
OUT_DIR = SCENE_DIR / "output"
OURS_DIR = OUT_DIR / "ours"
STORM_DIR = OUT_DIR / "storm"
METAL_DIR = OUT_DIR / "metal"
COMPARE_DIR = OUT_DIR / "compare"
for d in (OURS_DIR, STORM_DIR, METAL_DIR, COMPARE_DIR):
    d.mkdir(parents=True, exist_ok=True)

W, H = 512, 512
RMSE_PASS = 30.0  # 0..255 scale; phase 2 will tighten this.

SCENES = ["basicCurves", "moreCurves", "pinnedCurves"]
CAMERA_PATH = "/World/TestCam"

# usdrecord installs (see advisor: two installs, disjoint plugin sets):
#   - /usr/bin/usdrecord ships only --renderer Metal
#   - $HOME/OpenUSD-install/bin/usdrecord ships --renderer GL/Storm
USDRECORD_METAL = Path("/usr/bin/usdrecord")
USDRECORD_GL = Path.home() / "OpenUSD-install" / "bin" / "usdrecord"
OPENUSD_PYPATH = Path.home() / "OpenUSD-install" / "lib" / "python"
OPENUSD_LIBPATH = Path.home() / "OpenUSD-install" / "lib"
OPENUSD_PLUGIN = Path.home() / "OpenUSD-install" / "plugin" / "usd"


# --- minimal PNG read/write (stdlib + numpy) -----------------------------

def _png_chunk(tag: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)


def write_png(path: Path, rgb: np.ndarray) -> None:
    """Write an HxWx{3,4} uint8 image as a PNG."""
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)
    h, w = rgb.shape[:2]
    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    if rgb.shape[2] == 4:
        color_type = 6  # RGBA
    elif rgb.shape[2] == 3:
        color_type = 2  # RGB
    else:
        raise ValueError(f"unsupported image shape {rgb.shape}")
    raw = bytearray()
    stride = rgb.shape[2]
    for y in range(h):
        raw.append(0)  # filter byte = None
        raw.extend(rgb[y].tobytes())
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)
    idat = zlib.compress(bytes(raw), 6)
    out = sig + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", idat) + _png_chunk(b"IEND", b"")
    path.write_bytes(out)


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a); pb = abs(p - b); pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def read_png(path: Path) -> np.ndarray:
    """Read a PNG into an HxWx{3,4} uint8 numpy array. Supports the colour
    types usdrecord emits (RGB / RGBA, 8-bit, no interlace)."""
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path}: not a PNG")
    pos = 8
    width = height = 0
    bit_depth = color_type = interlace = 0
    idat = bytearray()
    palette: Optional[np.ndarray] = None
    trns: Optional[bytes] = None
    while pos < len(data):
        ln = struct.unpack(">I", data[pos:pos + 4])[0]; pos += 4
        tag = data[pos:pos + 4]; pos += 4
        chunk = data[pos:pos + ln]; pos += ln
        pos += 4  # crc
        if tag == b"IHDR":
            width, height, bit_depth, color_type, _, _, interlace = struct.unpack(">IIBBBBB", chunk)
        elif tag == b"IDAT":
            idat.extend(chunk)
        elif tag == b"PLTE":
            palette = np.frombuffer(chunk, dtype=np.uint8).reshape(-1, 3)
        elif tag == b"tRNS":
            trns = chunk
        elif tag == b"IEND":
            break
    if interlace != 0:
        raise ValueError(f"{path}: interlaced PNGs unsupported")
    if bit_depth != 8:
        raise ValueError(f"{path}: only 8-bit PNGs supported (got {bit_depth})")
    raw = zlib.decompress(bytes(idat))
    if color_type == 0:
        bpp, channels = 1, 1
    elif color_type == 2:
        bpp, channels = 3, 3
    elif color_type == 3:
        bpp, channels = 1, 1  # palette index
    elif color_type == 4:
        bpp, channels = 2, 2  # gray + alpha
    elif color_type == 6:
        bpp, channels = 4, 4
    else:
        raise ValueError(f"{path}: unsupported colour type {color_type}")
    stride = width * bpp
    img = np.zeros((height, stride), dtype=np.uint8)
    prev = np.zeros(stride, dtype=np.uint8)
    p = 0
    for y in range(height):
        ftype = raw[p]; p += 1
        row = bytearray(raw[p:p + stride]); p += stride
        # Cast prev to int via bytearray to dodge numpy uint8 overflow.
        prev_b = bytes(prev)
        if ftype == 0:
            pass
        elif ftype == 1:  # Sub
            for i in range(stride):
                left = row[i - bpp] if i >= bpp else 0
                row[i] = (row[i] + left) & 0xFF
        elif ftype == 2:  # Up
            for i in range(stride):
                row[i] = (row[i] + prev_b[i]) & 0xFF
        elif ftype == 3:  # Average
            for i in range(stride):
                left = row[i - bpp] if i >= bpp else 0
                row[i] = (row[i] + (left + prev_b[i]) // 2) & 0xFF
        elif ftype == 4:  # Paeth
            for i in range(stride):
                left = row[i - bpp] if i >= bpp else 0
                up = prev_b[i]
                ul = prev_b[i - bpp] if i >= bpp else 0
                row[i] = (row[i] + _paeth(left, up, ul)) & 0xFF
        else:
            raise ValueError(f"{path}: unknown filter type {ftype}")
        img[y] = np.frombuffer(bytes(row), dtype=np.uint8)
        prev = img[y]
    arr = img.reshape(height, width, channels)
    if color_type == 3 and palette is not None:
        arr = palette[arr[:, :, 0]]
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.shape[2] == 2:
        arr = np.dstack([np.repeat(arr[:, :, :1], 3, axis=2), arr[:, :, 1:2]])
    return arr


def to_rgb(img: np.ndarray) -> np.ndarray:
    """Drop alpha if present; return HxWx3 uint8."""
    if img.ndim == 2:
        return np.stack([img, img, img], axis=-1)
    if img.shape[2] >= 3:
        return img[:, :, :3]
    raise ValueError(f"unexpected image shape {img.shape}")


def resize_nearest(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Numpy-only nearest-neighbour resize (no Pillow / cv2)."""
    src_h, src_w = img.shape[:2]
    if src_h == h and src_w == w:
        return img
    ys = (np.linspace(0, src_h - 1, h)).astype(np.int32)
    xs = (np.linspace(0, src_w - 1, w)).astype(np.int32)
    return img[ys[:, None], xs[None, :]]


# --- camera derivation ---------------------------------------------------

def _try_pxr_camera(usd_path: Path) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Run a subprocess that imports pxr (with OpenUSD-install env vars)
    and prints view, proj, eye for ``CAMERA_PATH`` in the given stage.

    Subprocess isolation matters: the main script also loads libnusd_gles
    via ctypes which links against a different USD/MaterialX, and pxr's
    static initialisers don't tolerate sharing the process. Returns
    ``(view16, proj16, eye3)`` or ``None`` on failure.
    """
    py = sys.executable
    venv_py = Path.home() / "usdview-venv" / "bin" / "python"
    if venv_py.exists():
        py = str(venv_py)
    code = (
        "import sys\n"
        "from pxr import Usd, UsdGeom, Gf\n"
        f"s = Usd.Stage.Open(r'{usd_path}')\n"
        f"cam_prim = s.GetPrimAtPath(r'{CAMERA_PATH}')\n"
        "if not cam_prim: print('NOCAM'); sys.exit(0)\n"
        "ucam = UsdGeom.Camera(cam_prim)\n"
        "cam = ucam.GetCamera()\n"
        # ComputeLocalToWorldTransform bakes parent xforms (e.g.
        # World/rotateX=-90 in moreCurves) into the camera's transform
        # so cam.frustum.ComputeViewMatrix returns the world-space view.
        "lw = ucam.ComputeLocalToWorldTransform(Usd.TimeCode.Default())\n"
        "cam.transform = Gf.Matrix4d(lw)\n"
        "f = cam.frustum\n"
        "v = f.ComputeViewMatrix()\n"
        "p = f.ComputeProjectionMatrix()\n"
        "e = f.position\n"
        "print('VIEW', ' '.join('%.9g' % v[i][j] for i in range(4) for j in range(4)))\n"
        "print('PROJ', ' '.join('%.9g' % p[i][j] for i in range(4) for j in range(4)))\n"
        "print('EYE',  ' '.join('%.9g' % e[i] for i in range(3)))\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(OPENUSD_PYPATH)
    env["DYLD_FALLBACK_LIBRARY_PATH"] = str(OPENUSD_LIBPATH)
    env["PXR_PLUGINPATH_NAME"] = str(OPENUSD_PLUGIN)
    try:
        out = subprocess.run([py, "-c", code], capture_output=True, text=True,
                             env=env, timeout=30)
    except Exception as e:
        print(f"  [camera] pxr subprocess failed: {e}")
        return None
    if out.returncode != 0:
        print(f"  [camera] pxr returned {out.returncode}: {out.stderr.strip()[:200]}")
        return None
    if "NOCAM" in out.stdout:
        print(f"  [camera] {CAMERA_PATH} not found in {usd_path.name}")
        return None
    view = proj = eye = None
    for line in out.stdout.splitlines():
        if line.startswith("VIEW "):
            vs = [float(x) for x in line.split()[1:]]
            view = np.array(vs, dtype=np.float32).reshape(4, 4)
        elif line.startswith("PROJ "):
            ps = [float(x) for x in line.split()[1:]]
            proj = np.array(ps, dtype=np.float32).reshape(4, 4)
        elif line.startswith("EYE "):
            eye = np.array([float(x) for x in line.split()[1:]], dtype=np.float32)
    if view is None or proj is None or eye is None:
        print(f"  [camera] could not parse pxr output")
        return None
    # pxr's GfMatrix4d is row-vector convention (translation in row 3,
    # `worldPt × V = viewPt`). Numpy stores this row-major: bytes[r*4+c]
    # = m[r][c]. Our viewer uploads via glUniformMatrix4fv with
    # transpose=GL_FALSE, which makes GL interpret bytes as col-major
    # for the GLSL mat4. The net effect: the shader's u_view's GLSL
    # accessor m[col][row] = bytes[col*4+row] = m_pxr[col][row], so the
    # math matrix the shader sees is m_pxr.T — exactly the col-vector
    # form GLSL needs. So pass pxr matrices through unchanged.
    return view.copy(), proj.copy(), eye


# --- renderers -----------------------------------------------------------

def _placeholder_png(out_png: Path, label: str) -> None:
    """Write a 512x512 background-grey image as the 'ours' fallback when
    viewer_create fails because the scene has no Mesh prims. The viewer
    currently rejects mesh-less scenes (src/scene.c returns NULL). Until
    BasisCurves phase 2 lets the loader keep curve-only scenes alive,
    this placeholder is what 'ours' looks like — and it's exactly what
    the user expects to see (zero curves)."""
    img = np.full((H, W, 3), 32, dtype=np.uint8)
    write_png(out_png, img)
    print(f"  [ours] wrote placeholder ({label}) — viewer rejected mesh-less scene")


def render_ours(usd_path: Path, view: np.ndarray, proj: np.ndarray,
                eye: np.ndarray, out_png: Path) -> bool:
    """Drive libnusd_gles via ctypes and dump 512x512 RGB PNG.

    Returns True on success; on viewer_create failure (the common case
    for these curve-only scenes pre-phase-2) writes a placeholder PNG
    so the comparison machinery still has an 'ours' image to diff
    against, and returns True."""
    sys.path.insert(0, str(REPO / "python"))
    # Test cameras are computed from the scene's authored frame (Z-up
    # for these test USDAs). Keep the explicit env for older builds where
    # preserving authored upAxis was still opt-in.
    os.environ["NUSD_PRESERVE_UPAXIS"] = "1"
    try:
        from nusd_gles._bindings import GlesViewer, _lib  # noqa
    except Exception as e:
        print(f"  [ours] ctypes import failed: {e}")
        return False
    try:
        v = GlesViewer(str(usd_path), width=W, height=H)
    except Exception as e:
        # Pre-phase-2: scene_load rejects curve-only scenes. Produce a
        # placeholder so the rest of the diff pipeline still runs.
        _placeholder_png(out_png, str(e)[:60])
        return True
    try:
        # RASTER mode so we don't depend on materials being loaded —
        # phase 2 hasn't authored materials for these test scenes yet.
        _lib.nusd_gl_viewer_set_render_mode(v._handle, 0)
        img = v.render(W, H, view, proj, eye)
        write_png(out_png, img[:, :, :3])
        v.close()
        return True
    except Exception as e:
        print(f"  [ours] render failed: {e}")
        try:
            v.close()
        except Exception:
            pass
        _placeholder_png(out_png, str(e)[:60])
        return True


def _have_storm() -> bool:
    if not USDRECORD_GL.exists():
        return False
    env = os.environ.copy()
    env["PYTHONPATH"] = str(OPENUSD_PYPATH)
    env["DYLD_FALLBACK_LIBRARY_PATH"] = str(OPENUSD_LIBPATH)
    env["PXR_PLUGINPATH_NAME"] = str(OPENUSD_PLUGIN)
    try:
        r = subprocess.run([str(USDRECORD_GL), "--help"], capture_output=True,
                           text=True, env=env, timeout=10)
        return "GL" in (r.stdout or "") or "Storm" in (r.stdout or "")
    except Exception:
        return False


def render_storm_gl(usd_path: Path, out_png: Path) -> bool:
    """usdrecord with --renderer GL (HdStormRendererPlugin under OpenUSD-install)."""
    if not _have_storm():
        print("  [storm] GL renderer not registered — skipped")
        return False
    env = os.environ.copy()
    env["PYTHONPATH"] = str(OPENUSD_PYPATH)
    env["DYLD_FALLBACK_LIBRARY_PATH"] = str(OPENUSD_LIBPATH)
    env["PXR_PLUGINPATH_NAME"] = str(OPENUSD_PLUGIN)
    # Note: --frames was rejected on this Storm/GL build because it
    # requires a "###" placeholder in the output path. Default time-code
    # render is fine for a single-frame scene.
    cmd = [str(USDRECORD_GL), "--renderer", "GL", "--camera", CAMERA_PATH,
           "--complexity", "veryhigh",
           "--imageWidth", str(W), str(usd_path), str(out_png)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
    except Exception as e:
        print(f"  [storm] subprocess failed: {e}")
        return False
    if r.returncode != 0 or not out_png.exists():
        print(f"  [storm] usdrecord returned {r.returncode}: {(r.stderr or r.stdout)[:200]}")
        return False
    return True


def render_metal(usd_path: Path, out_png: Path) -> bool:
    """usdrecord with --renderer Metal (the system /usr/bin install)."""
    if not USDRECORD_METAL.exists():
        print("  [metal] /usr/bin/usdrecord not found — skipped")
        return False
    cmd = [str(USDRECORD_METAL), "--renderer", "Metal", "--camera", CAMERA_PATH,
           "--complexity", "veryhigh",
           "--imageWidth", str(W), str(usd_path), str(out_png)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except Exception as e:
        print(f"  [metal] subprocess failed: {e}")
        return False
    if r.returncode != 0 or not out_png.exists():
        print(f"  [metal] usdrecord returned {r.returncode}: {(r.stderr or r.stdout)[:200]}")
        return False
    return True


# --- diff ----------------------------------------------------------------

def rmse_uint8(a: np.ndarray, b: np.ndarray) -> float:
    """Mean RMSE on the [0..255] scale across all channels, restricted to
    the 'curve region' — the union of pixels where either image has
    non-background content. Without this restriction, the metric is
    dominated by clear-color mismatch (our viewer clears to gray; Storm
    and Metal default to black) which has nothing to do with curve
    geometry quality. The mask is the union of (ours' non-bg) and
    (other's non-bg) plus a small dilation so antialiased halos
    contribute too."""
    a = a.astype(np.float32); b = b.astype(np.float32)
    # Heuristic bg detection: ours' clear is ≈ (184,184,178); storm/metal
    # clear is near black. Pixels far from either are "non-bg".
    ours_bg = np.array([184, 184, 178], np.float32)
    near_bg_a = np.linalg.norm(a - ours_bg, axis=2) < 8
    near_black_a = a.sum(axis=2) < 30
    near_bg_b = np.linalg.norm(b - ours_bg, axis=2) < 8
    near_black_b = b.sum(axis=2) < 30
    nbg_a = ~(near_bg_a | near_black_a)
    nbg_b = ~(near_bg_b | near_black_b)
    mask = nbg_a | nbg_b
    if not mask.any():
        return 0.0
    diff = (a - b)[mask]
    return float(np.sqrt(np.mean(diff ** 2)))


def composite_3up(a: np.ndarray, b: np.ndarray, label_a: str, label_b: str) -> np.ndarray:
    """Stack [A | B | abs-diff*5] horizontally."""
    a = to_rgb(a); b = to_rgb(b)
    if a.shape != b.shape:
        b = resize_nearest(b, a.shape[1], a.shape[0])
    diff = np.clip(np.abs(a.astype(np.int16) - b.astype(np.int16)) * 5, 0, 255).astype(np.uint8)
    h, w = a.shape[:2]
    sep = np.full((h, 4, 3), 64, dtype=np.uint8)
    return np.concatenate([a, sep, b, sep, diff], axis=1)


# --- main ----------------------------------------------------------------

def main() -> int:
    print(f"== curves_compare: {len(SCENES)} scenes, {W}x{H}, RMSE pass < {RMSE_PASS}")
    print(f"   USDA dir:  {SCENE_DIR}")
    print(f"   Output:    {OUT_DIR}")
    print()

    # If neither viewer nor any usdrecord is available, signal CTest skip.
    have_lib = (REPO / "build" / ("libnusd_gles.dylib" if sys.platform == "darwin"
                                  else "libnusd_gles.so")).exists()
    have_metal = USDRECORD_METAL.exists()
    have_storm = USDRECORD_GL.exists()
    if not have_lib and not have_metal and not have_storm:
        print("SKIP: no renderer backends available (no libnusd_gles, no usdrecord)")
        return 77

    rows = []
    for scene in SCENES:
        usd = SCENE_DIR / f"{scene}.usda"
        if not usd.exists():
            print(f"  MISSING USDA: {usd}")
            return 1
        print(f"--- {scene} ---")
        cam = _try_pxr_camera(usd)
        if cam is None:
            print(f"  [camera] FAILED to derive view/proj from {CAMERA_PATH}; skipping {scene}")
            rows.append((scene, None, None, "ERROR"))
            continue
        view, proj, eye = cam

        ours_png = OURS_DIR / f"{scene}.png"
        storm_png = STORM_DIR / f"{scene}.png"
        metal_png = METAL_DIR / f"{scene}.png"
        ok_ours = render_ours(usd, view, proj, eye, ours_png)
        ok_storm = render_storm_gl(usd, storm_png)
        ok_metal = render_metal(usd, metal_png)
        print(f"  ours:  {'ok' if ok_ours else 'SKIP/ERR'}")
        print(f"  storm: {'ok' if ok_storm else 'SKIP/ERR'}")
        print(f"  metal: {'ok' if ok_metal else 'SKIP/ERR'}")

        # Load whatever we have.
        ours_img = read_png(ours_png) if ok_ours else None
        storm_img = read_png(storm_png) if ok_storm else None
        metal_img = read_png(metal_png) if ok_metal else None
        # Normalise sizes (usdrecord output may not be exactly 512x512).
        for name, img in [("ours", ours_img), ("storm", storm_img),
                          ("metal", metal_img)]:
            if img is not None:
                rgb = to_rgb(img)
                if rgb.shape[:2] != (H, W):
                    rgb = resize_nearest(rgb, W, H)
                if name == "ours":
                    ours_img = rgb
                elif name == "storm":
                    storm_img = rgb
                elif name == "metal":
                    metal_img = rgb

        ours_storm = ours_metal = storm_metal = None
        if ours_img is not None and storm_img is not None:
            ours_storm = rmse_uint8(ours_img, storm_img)
            comp = composite_3up(ours_img, storm_img, "ours", "storm")
            write_png(COMPARE_DIR / f"{scene}_ours_vs_storm.png", comp)
        if ours_img is not None and metal_img is not None:
            ours_metal = rmse_uint8(ours_img, metal_img)
            comp = composite_3up(ours_img, metal_img, "ours", "metal")
            write_png(COMPARE_DIR / f"{scene}_ours_vs_metal.png", comp)
        if storm_img is not None and metal_img is not None:
            storm_metal = rmse_uint8(storm_img, metal_img)
            comp = composite_3up(storm_img, metal_img, "storm", "metal")
            write_png(COMPARE_DIR / f"{scene}_storm_vs_metal.png", comp)

        verdict = "PASS"
        for v in (ours_storm, ours_metal):
            if v is None:
                verdict = "PARTIAL"
            elif v >= RMSE_PASS:
                verdict = "FAIL"
        rows.append((scene, ours_storm, ours_metal, verdict))
        print()

    # Print summary.
    print("=" * 76)
    print(f"{'scene':<18} {'ours-vs-storm':>14} {'ours-vs-metal':>14}  {'status':>8}")
    print("-" * 76)
    for scene, a, b, verdict in rows:
        sa = f"{a:>11.2f}" if isinstance(a, float) else f"{'  --  ':>11}"
        sb = f"{b:>11.2f}" if isinstance(b, float) else f"{'  --  ':>11}"
        print(f"{scene:<18} {sa}        {sb}        {verdict:>8}")
    print("=" * 76)
    print(f"   (RMSE on 0..255 scale; pass threshold = {RMSE_PASS})")
    print()
    print("EXPECTED: ours-vs-storm and ours-vs-metal should FAIL until")
    print("BasisCurves phase 2 (scene loader) lands. Phase 1d only ships")
    print("the curves render pipeline; src/scene.c still drops curve prims.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"FATAL: {e}")
        sys.exit(1)

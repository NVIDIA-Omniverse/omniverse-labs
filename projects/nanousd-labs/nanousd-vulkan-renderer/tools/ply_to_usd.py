#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""3D Gaussian Splatting PLY → AOUSD ParticleField3DGaussianSplat USDA.

Standalone converter. No dependency on pxr / OpenUSD — the output USDA
is a plain text file that any USD parser (including nanousd, once it
recognizes the schema) can consume.

Documents the input contract per plan §1:

  PLY input (3DGS convention, output of original `train.py` / Inria
                                  Gaussian Splatting + 3dgrut):
    - x, y, z                 :: float — object-space center
    - nx, ny, nz              :: float — per-vertex normal (unused; splats
                                          use covariance, not surface normal)
    - f_dc_0, f_dc_1, f_dc_2  :: float — SH band 0 (DC) coefficients,
                                          one per channel; reconstructed
                                          radiance = 0.5 + C0 · sh0
                                          (where C0 = 1 / (2·sqrt(pi)))
    - f_rest_0..f_rest_44     :: float — SH bands 1..3, layout
                                          [degree (1, 2, 3)][coefficient]
                                          [channel (R, G, B)]; only present
                                          when sh_degree > 0
    - opacity                 :: float — *raw* (pre-sigmoid)
    - scale_0, scale_1, scale_2 :: float — *log* sigma; linear σ = exp(scale)
    - rot_0, rot_1, rot_2, rot_3 :: float — quaternion *wxyz*, normalized

  USDA output (AOUSD ParticleField3DGaussianSplat, USD 26.03):
    - positions       :: point3f[]  — object-space centers
    - scales          :: float3[]   — linear σ (post-exp from PLY)
    - orientations    :: quatf[]    — wxyz, normalized
    - opacities       :: float[]    — post-sigmoid, in [0, 1]
    - radianceSphericalHarmonicsCoefficients :: float[]
                                       — flat, length N · (deg+1)² · 3,
                                          ordered [particle][coeff][channel]
    - radianceSphericalHarmonicsDegree :: int — 0..3

This module also exposes a `synthesize_ply(out_path, N, seed)` utility
that writes a small deterministic 3DGS-format PLY for testing the
round-trip without requiring real production data. The renderer's
visual regression suite uses it (tests/gs_visual/run_tests.py).

CLI:
    python tools/ply_to_usd.py INPUT.ply OUTPUT.usda
    python tools/ply_to_usd.py --synth /tmp/synth.ply  # generate synth
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np


# ---------- 3DGS PLY parser ----------------------------------------------

class GsScene(NamedTuple):
    positions:    np.ndarray  # (N, 3) float32
    scales:       np.ndarray  # (N, 3) float32 (linear σ, post-exp)
    orientations: np.ndarray  # (N, 4) float32 (wxyz, normalized)
    opacities:    np.ndarray  # (N,) float32 in [0, 1] (post-sigmoid)
    sh:           np.ndarray  # (N, (deg+1)**2, 3) float32
    sh_degree:    int


def _parse_ply_header(f) -> tuple[int, list[str]]:
    """Returns (vertex_count, list of property names) for a binary-little-endian
    or ASCII 3DGS PLY. Raises ValueError on malformed input."""
    line = f.readline()
    if line != b"ply\n":
        raise ValueError(f"Not a PLY file (header was {line!r})")

    line = f.readline()
    if line == b"format binary_little_endian 1.0\n":
        binary = True
    elif line == b"format ascii 1.0\n":
        binary = False
    else:
        raise ValueError(f"Unsupported PLY format line: {line!r}")

    nvert = -1
    props: list[str] = []
    while True:
        line = f.readline()
        if not line:
            raise ValueError("PLY header truncated (no end_header)")
        if line == b"end_header\n":
            break
        if line.startswith(b"element vertex "):
            nvert = int(line.split()[2])
        elif line.startswith(b"property float "):
            props.append(line.split()[2].decode("ascii"))
        # silently ignore comments + other element kinds (faces etc).
    if nvert < 0:
        raise ValueError("PLY missing `element vertex N` line")

    return nvert, props, binary


def parse_3dgs_ply(path: str | Path) -> GsScene:
    """Parse a 3DGS-convention PLY into a `GsScene`. Applies sigmoid to
    opacity, exp to scale, wxyz-quaternion-normalize to rotation, and SH
    layout shuffle to the radianceSphericalHarmonicsCoefficients array."""
    p = Path(path)
    with p.open("rb") as f:
        nvert, props, binary = _parse_ply_header(f)

        # The 3DGS PLY puts each vertex as a packed float32 block in the
        # property order from the header; we read it as a structured
        # ndarray and slice it.
        if binary:
            dtype = np.dtype([(name, "<f4") for name in props])
            raw = np.frombuffer(f.read(nvert * dtype.itemsize), dtype=dtype)
        else:
            raw = np.empty(nvert, dtype=[(name, "<f4") for name in props])
            for i in range(nvert):
                line = f.readline().split()
                if len(line) != len(props):
                    raise ValueError(
                        f"Vertex {i} has {len(line)} cols, expected {len(props)}")
                for j, name in enumerate(props):
                    raw[name][i] = float(line[j])

    def col(name: str) -> np.ndarray:
        if name not in raw.dtype.names:
            raise ValueError(f"PLY missing property '{name}'")
        return raw[name].astype(np.float32)

    positions = np.stack([col("x"), col("y"), col("z")], axis=1)

    # Scale is log-σ in PLY → linear σ in USDA.
    scales_log = np.stack([col("scale_0"), col("scale_1"), col("scale_2")], axis=1)
    scales = np.exp(scales_log).astype(np.float32)

    # Quaternion: PLY stores wxyz; normalize defensively.
    quats = np.stack([col("rot_0"), col("rot_1"), col("rot_2"), col("rot_3")], axis=1)
    quats = quats / np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-12)

    # Opacity: PLY stores raw (pre-sigmoid).
    opacities = 1.0 / (1.0 + np.exp(-col("opacity")))

    # SH coefficients. Determine sh_degree from rest count.
    rest_props = [n for n in raw.dtype.names if n.startswith("f_rest_")]
    rest_count = len(rest_props)            # = 3 * (deg² + 2·deg)
    sh_degree = {0: 0, 9: 1, 24: 2, 45: 3}.get(rest_count, -1)
    if sh_degree < 0:
        raise ValueError(
            f"Unrecognized PLY SH layout: {rest_count} f_rest_* properties "
            "(expected 0/9/24/45 for sh_degree 0/1/2/3)")

    n_coeff = (sh_degree + 1) ** 2
    sh = np.zeros((nvert, n_coeff, 3), dtype=np.float32)
    sh[:, 0, 0] = col("f_dc_0")
    sh[:, 0, 1] = col("f_dc_1")
    sh[:, 0, 2] = col("f_dc_2")
    if sh_degree > 0:
        # 3DGS PLY layout: f_rest_[ch * (n_coeff-1) + (i-1)] holds the
        # i-th coefficient (i in 1..n_coeff-1) for channel ch (0=R,1=G,2=B).
        rest_per_channel = n_coeff - 1
        for ch in range(3):
            for i in range(1, n_coeff):
                sh[:, i, ch] = col(f"f_rest_{ch * rest_per_channel + (i - 1)}")

    return GsScene(positions=positions, scales=scales, orientations=quats,
                   opacities=opacities, sh=sh, sh_degree=sh_degree)


# ---------- USDA writer --------------------------------------------------


def _fmt_vec(values: np.ndarray, brace_open: str, brace_close: str,
             per_line: int = 4) -> str:
    """Pretty-print a (N, K) array as `(v1,v2,...), (v1,v2,...), ...` with
    soft line wraps at `per_line` items. Float printing fixed at 7 sig
    digits (round-trips through float32)."""
    out: list[str] = []
    n = values.shape[0]
    for i in range(n):
        row = ", ".join(f"{v:.7g}" for v in values[i])
        out.append(f"{brace_open}{row}{brace_close}")
    chunks = []
    for i in range(0, n, per_line):
        chunks.append(", ".join(out[i:i + per_line]))
    return ",\n        ".join(chunks)


def write_usda(scene: GsScene, out_path: str | Path,
               prim_path: str = "/World/Splats") -> None:
    """Emit an AOUSD-compatible USDA representation. The output is plain
    text — no pxr dependency — and any conformant USD parser (or nanousd
    once it recognizes the schema) can consume it."""
    p = Path(out_path)
    n = scene.positions.shape[0]
    sh_per = (scene.sh_degree + 1) ** 2

    parent_path = "/" + prim_path.lstrip("/").split("/")[0]
    leaf_name   = prim_path.rstrip("/").split("/")[-1]

    sh_flat = scene.sh.reshape(-1).astype(np.float32)

    lines: list[str] = []
    lines.append("#usda 1.0")
    lines.append("(")
    lines.append('    defaultPrim = "' + parent_path.lstrip("/") + '"')
    lines.append('    upAxis = "Y"')
    lines.append(")")
    lines.append("")
    lines.append(f'def Xform "{parent_path.lstrip("/")}"')
    lines.append("{")
    lines.append(f'    def ParticleField3DGaussianSplat "{leaf_name}"')
    lines.append("    {")
    lines.append("        point3f[] positions = [")
    lines.append("        " + _fmt_vec(scene.positions,    "(", ")"))
    lines.append("        ]")
    lines.append("        float3[] scales = [")
    lines.append("        " + _fmt_vec(scene.scales,        "(", ")"))
    lines.append("        ]")
    lines.append("        quatf[] orientations = [")
    lines.append("        " + _fmt_vec(scene.orientations,  "(", ")"))
    lines.append("        ]")
    lines.append("        float[] opacities = [")
    lines.append("            " + ", ".join(f"{v:.7g}" for v in scene.opacities))
    lines.append("        ]")
    # AOUSD spec uses namespaced names `radiance:spherical*` and stores
    # coefficients as vec3f[] with elementSize=(deg+1)^2. The flat-float
    # form our older converter wrote works for our loader but is
    # rejected by ovrtx ("ParticleField schema missing required
    # 'radiance:sphericalHarmonicsCoefficients' primvar"). Emit the
    # spec-compliant form so both loaders accept it.
    sh_vec3 = scene.sh.reshape(-1, 3).astype(np.float32)  # (N*sh_per, 3)
    lines.append("        float3[] radiance:sphericalHarmonicsCoefficients = [")
    lines.append("        " + _fmt_vec(sh_vec3, "(", ")"))
    lines.append("        ] (")
    lines.append(f"            elementSize = {sh_per}")
    lines.append("            interpolation = \"vertex\"")
    lines.append("        )")
    lines.append(
        f"        uniform int radiance:sphericalHarmonicsDegree = {scene.sh_degree}"
    )
    lines.append("    }")
    lines.append("}")
    lines.append("")

    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines))


# ---------- Synthetic PLY generator (for tests) --------------------------


def synthesize_ply(out_path: str | Path, N: int = 16,
                    sh_degree: int = 1, seed: int = 42) -> None:
    """Write a deterministic synthetic 3DGS PLY at `out_path`. Used by
    the visual regression suite to round-trip the converter without
    requiring a real production splat scene."""
    rng = np.random.default_rng(seed)

    positions = (rng.random((N, 3), dtype=np.float32) - 0.5).astype(np.float32)
    # PLY scale is log(σ); pick σ ∈ [0.05, 0.15] linear.
    scales_lin = 0.05 + 0.10 * rng.random((N, 3), dtype=np.float32)
    scales_log = np.log(scales_lin).astype(np.float32)
    # PLY rotation: random unit quaternion (wxyz).
    rot = rng.standard_normal((N, 4)).astype(np.float32)
    rot = rot / np.linalg.norm(rot, axis=1, keepdims=True)
    # PLY opacity: raw (pre-sigmoid). Pick post-sigmoid in [0.5, 0.95].
    op_post = (0.5 + 0.45 * rng.random(N, dtype=np.float32)).astype(np.float32)
    op_raw = -np.log(1.0 / op_post - 1.0)
    # SH DC: small random; band 1+ also small.
    f_dc = (rng.random((N, 3), dtype=np.float32) * 0.6 - 0.3).astype(np.float32)
    n_coeff = (sh_degree + 1) ** 2
    rest_per_ch = n_coeff - 1
    f_rest = (rng.standard_normal((N, 3 * rest_per_ch), dtype=np.float32)
              * 0.1).astype(np.float32)

    # Header.
    lines = ["ply", "format binary_little_endian 1.0",
             f"element vertex {N}",
             "property float x", "property float y", "property float z",
             "property float nx", "property float ny", "property float nz",
             "property float f_dc_0", "property float f_dc_1", "property float f_dc_2"]
    for c in range(3):
        for i in range(rest_per_ch):
            lines.append(f"property float f_rest_{c * rest_per_ch + i}")
    lines += ["property float opacity",
              "property float scale_0", "property float scale_1", "property float scale_2",
              "property float rot_0", "property float rot_1", "property float rot_2",
              "property float rot_3", "end_header"]
    header = ("\n".join(lines) + "\n").encode("ascii")

    # Pack each vertex into the property order above.
    n_props = 3 + 3 + 3 + 3 * rest_per_ch + 1 + 3 + 4
    body = np.zeros((N, n_props), dtype=np.float32)
    body[:, 0:3]   = positions
    body[:, 3:6]   = 0.0  # nx, ny, nz unused
    body[:, 6:9]   = f_dc
    body[:, 9:9 + 3 * rest_per_ch] = f_rest
    body[:, 9 + 3 * rest_per_ch] = op_raw
    body[:, 9 + 3 * rest_per_ch + 1: 9 + 3 * rest_per_ch + 4] = scales_log
    body[:, 9 + 3 * rest_per_ch + 4:] = rot

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        f.write(header)
        f.write(body.astype("<f4").tobytes())


# ---------- CLI ----------------------------------------------------------


def _main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("input",  nargs="?", help="input .ply (3DGS convention)")
    p.add_argument("output", nargs="?", help="output .usda")
    p.add_argument("--synth", metavar="OUT.ply",
                   help="emit a synthetic 16-particle PLY at OUT.ply (skips conversion)")
    p.add_argument("--n", type=int, default=16, help="synth particle count")
    p.add_argument("--sh-degree", type=int, default=1, help="synth SH degree")
    p.add_argument("--seed", type=int, default=42, help="synth RNG seed")
    args = p.parse_args(argv)

    if args.synth:
        synthesize_ply(args.synth, N=args.n, sh_degree=args.sh_degree, seed=args.seed)
        print(f"wrote synthetic PLY: {args.synth} (N={args.n}, deg={args.sh_degree})")
        return 0

    if not args.input or not args.output:
        p.error("input and output are required (or use --synth)")

    scene = parse_3dgs_ply(args.input)
    write_usda(scene, args.output)
    print(f"wrote {args.output}: N={scene.positions.shape[0]}, "
          f"sh_degree={scene.sh_degree}")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))

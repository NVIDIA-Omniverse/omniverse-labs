# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared pose math and version-1 JSON payloads for PhysX subprocess + live worker."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def pose_to_matrix4d_rowvector(
    px: float, py: float, pz: float, qx: float, qy: float, qz: float, qw: float
) -> list[list[float]]:
    """PhysX pose (px,py,pz, qx,qy,qz,qw) with imaginary-first quat → 4x4 row-vector matrix."""
    x, y, z, w = qx, qy, qz, qw
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n > 1e-8:
        w, x, y, z = w / n, x / n, y / n, z / n
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    r00 = 1.0 - 2.0 * (yy + zz)
    r01 = 2.0 * (xy - wz)
    r02 = 2.0 * (xz + wy)
    r10 = 2.0 * (xy + wz)
    r11 = 1.0 - 2.0 * (xx + zz)
    r12 = 2.0 * (yz - wx)
    r20 = 2.0 * (xz - wy)
    r21 = 2.0 * (yz + wx)
    r22 = 1.0 - 2.0 * (xx + yy)
    return [
        [r00, r01, r02, 0.0],
        [r10, r11, r12, 0.0],
        [r20, r21, r22, 0.0],
        [px, py, pz, 1.0],
    ]


def version1_articulation_payload(
    buf: np.ndarray,
    body_names: list[str],
    articulation_root: str,
) -> dict[str, Any]:
    """Build viewer JSON version 1 from ARTICULATION_LINK_POSE tensor (one articulation row)."""
    n_art = int(buf.shape[0])
    n_links = int(buf.shape[1])
    root = articulation_root.rstrip("/")
    if n_art != 1:
        raise ValueError(
            f"JSON exporter supports one articulation tensor row (got n_art={n_art})."
        )
    if len(body_names) != n_links:
        raise ValueError(f"body_names length {len(body_names)} != n_links {n_links}")
    prims: list[dict[str, object]] = []
    for i, name in enumerate(body_names):
        px, py, pz, qx, qy, qz, qw = (float(x) for x in buf[0, i].tolist())
        path = f"{root}/{name}"
        prims.append(
            {
                "path": path,
                "matrix4d": pose_to_matrix4d_rowvector(px, py, pz, qx, qy, qz, qw),
            }
        )
    articulation_world = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return {
        "version": 1,
        "articulation_world_matrix4d": articulation_world,
        "prims": prims,
    }


def version1_rigid_body_payload(buf: np.ndarray, rigid_paths: list[str]) -> dict[str, Any]:
    """Build viewer JSON version 1 from RIGID_BODY_POSE tensor (N, 7) poses."""
    if buf.ndim != 2 or buf.shape[1] != 7:
        raise ValueError(f"expected RIGID_BODY_POSE shape (N, 7), got {buf.shape}")
    n = int(buf.shape[0])
    if n != len(rigid_paths):
        raise ValueError(f"tensor N={n} vs {len(rigid_paths)} rigid paths")
    prims: list[dict[str, object]] = []
    for i, path in enumerate(rigid_paths):
        px, py, pz, qx, qy, qz, qw = (float(x) for x in buf[i].tolist())
        prims.append(
            {
                "path": path,
                "matrix4d": pose_to_matrix4d_rowvector(px, py, pz, qx, qy, qz, qw),
            }
        )
    articulation_world = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return {
        "version": 1,
        "articulation_world_matrix4d": articulation_world,
        "prims": prims,
    }

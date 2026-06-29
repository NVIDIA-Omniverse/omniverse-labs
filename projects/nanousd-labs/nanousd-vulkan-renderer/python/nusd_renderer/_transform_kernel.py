# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared warp kernels for the renderer transform pipeline.

Originally lived in an internal IsaacLab renderer fork (`.../renderers/vulkan/inner.py`)
lines 22-130. Extracted so that both the in-process renderer (`inner.py`)
and the subprocess renderer host (`_remote_host.py`) import the *same*
kernel definition. Kernel-definition divergence between contexts would
silently produce different transforms.

The kernel composes Newton's body_q with per-shape transforms to write
world-space 4x4 row-major matrices. See the original docstring on
`_compute_transforms_kernel` for details.
"""

from __future__ import annotations

import warp as wp


@wp.func
def _quat_mul(ax: float, ay: float, az: float, aw: float, bx: float, by: float, bz: float, bw: float):
    """Hamilton product of two quaternions (x,y,z,w)."""
    rx = aw * bx + ax * bw + ay * bz - az * by
    ry = aw * by - ax * bz + ay * bw + az * bx
    rz = aw * bz + ax * by - ay * bx + az * bw
    rw = aw * bw - ax * bx - ay * by - az * bz
    return rx, ry, rz, rw


@wp.func
def _quat_rotate(qx: float, qy: float, qz: float, qw: float, vx: float, vy: float, vz: float):
    """Rotate vector (vx,vy,vz) by quaternion (qx,qy,qz,qw)."""
    # t = 2 * cross(q.xyz, v)
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    # result = v + qw * t + cross(q.xyz, t)
    rx = vx + qw * tx + (qy * tz - qz * ty)
    ry = vy + qw * ty + (qz * tx - qx * tz)
    rz = vz + qw * tz + (qx * ty - qy * tx)
    return rx, ry, rz


@wp.kernel
def compute_transforms_kernel(
    body_q: wp.array2d(dtype=float),
    shape_body: wp.array(dtype=int),
    shape_transforms: wp.array2d(dtype=float),
    valid_indices: wp.array(dtype=int),
    out_transforms: wp.array2d(dtype=float),
):
    """Compute world-space 4x4 matrices for valid shapes on GPU.

    For each valid shape, composes ``body_q[body]`` with the static shape
    transform. Static shapes (body < 0) use the shape transform directly.
    Output is row-major 4x4 written as (N_valid, 16).
    """
    tid = wp.tid()
    shape_idx = valid_indices[tid]
    body_idx = shape_body[shape_idx]

    # Read shape-local transform (pos + quat xyzw)
    spx = shape_transforms[shape_idx, 0]
    spy = shape_transforms[shape_idx, 1]
    spz = shape_transforms[shape_idx, 2]
    sqx = shape_transforms[shape_idx, 3]
    sqy = shape_transforms[shape_idx, 4]
    sqz = shape_transforms[shape_idx, 5]
    sqw = shape_transforms[shape_idx, 6]

    # Final composed transform
    px = spx
    py = spy
    pz = spz
    qx = sqx
    qy = sqy
    qz = sqz
    qw = sqw

    if body_idx >= 0:
        # Read body transform
        bpx = body_q[body_idx, 0]
        bpy = body_q[body_idx, 1]
        bpz = body_q[body_idx, 2]
        bqx = body_q[body_idx, 3]
        bqy = body_q[body_idx, 4]
        bqz = body_q[body_idx, 5]
        bqw = body_q[body_idx, 6]

        # Composed rotation: body_q * shape_q
        qx, qy, qz, qw = _quat_mul(bqx, bqy, bqz, bqw, sqx, sqy, sqz, sqw)
        # Composed position: body_pos + body_rot * shape_pos
        rpx, rpy, rpz = _quat_rotate(bqx, bqy, bqz, bqw, spx, spy, spz)
        px = bpx + rpx
        py = bpy + rpy
        pz = bpz + rpz

    # Convert quaternion to 3x3 rotation matrix
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    # Row-major 4x4: indices [row*4+col]
    out_transforms[tid, 0] = 1.0 - 2.0 * (yy + zz)
    out_transforms[tid, 1] = 2.0 * (xy - wz)
    out_transforms[tid, 2] = 2.0 * (xz + wy)
    out_transforms[tid, 3] = px

    out_transforms[tid, 4] = 2.0 * (xy + wz)
    out_transforms[tid, 5] = 1.0 - 2.0 * (xx + zz)
    out_transforms[tid, 6] = 2.0 * (yz - wx)
    out_transforms[tid, 7] = py

    out_transforms[tid, 8] = 2.0 * (xz - wy)
    out_transforms[tid, 9] = 2.0 * (yz + wx)
    out_transforms[tid, 10] = 1.0 - 2.0 * (xx + yy)
    out_transforms[tid, 11] = pz

    out_transforms[tid, 12] = 0.0
    out_transforms[tid, 13] = 0.0
    out_transforms[tid, 14] = 0.0
    out_transforms[tid, 15] = 1.0

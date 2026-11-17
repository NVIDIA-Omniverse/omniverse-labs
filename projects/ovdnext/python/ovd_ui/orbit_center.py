# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Camera-only sphere topology and source-space placement from the Qt viewer."""
from functools import lru_cache
import math

import numpy as np


@lru_cache(maxsize=1)
def orbit_sphere():
    """The two unshifted N=16 capsule hemispheres, including triangle diagonals.

    See OpenGLWidget::addCapsuleGeometry and submitCameraOrbitCenter. Keep the
    seam/pole vertices and triangle order: an analytic high-resolution sphere
    or three rings changes the visible wire topology.
    """
    vertices, indices = [], []
    for first_band in (4, 0):  # TOP_ID, then BOTTOM_ID as submitted by Qt.
        base = len(vertices)
        for band in range(first_band, first_band + 5):
            phi = np.float32(-np.float32(math.pi / 2) +
                             np.float32(np.float32(math.pi) * band / 8))
            for longitude in range(17):
                theta = np.float32(np.float32(math.tau) * longitude / 16)
                vertices.append((math.sin(phi), math.cos(phi) * math.sin(theta),
                                 math.cos(phi) * math.cos(theta)))
        for band in range(4):
            for longitude in range(16):
                a = base + band * 17 + longitude
                indices.extend((a, a + 1, a + 18, a, a + 18, a + 17))
    points = np.asarray(vertices, dtype="f4")
    topology = np.asarray(indices, dtype="u4")
    points.flags.writeable = topology.flags.writeable = False
    return points, topology


def orbit_transform(camera, scene_length, gizmo_scale, geometry_scale, *, pivot=None):
    """Native-world radius and the actual camera pivot under renderer scaling.

    Current camera calibration and the gizmo default are deliberately retained.
    A missing calibration uses the owned capture tolerance length. The renderer
    scales geometry, but not its eye: undo that scale for the pivot only.
    """
    values = (scene_length, gizmo_scale, geometry_scale, camera.tolerance_length)
    if any(not math.isfinite(value) for value in values) or scene_length < 0 or min(values[1:]) <= 0:
        raise ValueError("Orbit marker scales must be finite and positive (scene length may be zero)")
    radius = .25 * (scene_length or camera.tolerance_length) * gizmo_scale
    result = np.eye(4, dtype="f4")
    result[:3, :3] *= radius
    result[:3, 3] = camera.mirror(camera.target if pivot is None else pivot) / geometry_scale
    if not np.isfinite(result).all():
        raise ValueError("Orbit marker transform exceeds the supported finite range")
    return result

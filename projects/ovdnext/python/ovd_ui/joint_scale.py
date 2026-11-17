# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture-owned joint calibration from the historical solid geometry queue.

Debug arrows are deliberately excluded: Qt's queue membership there depends
on its previous update's simplified-quality flag and on the gizmo's own size.
No camera, native Session, GPU resource or preference is mutated here.
"""
from dataclasses import dataclass
import math

import numpy as np

from .geometry_visibility import geometry_visibility


JOINT_SCALE_DEFAULT = 1.
TRIANGLE_CHUNK = 4096


def validate_joint_scale(value, *, master_scale=1.):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 2:
        raise ValueError("Joint scale must be a finite number between 0 and 2")
    with np.errstate(over="ignore"):
        arm=np.float32(value)*np.float32(1.5)*np.float32(master_scale)
    if not np.isfinite(arm):
        raise ValueError("Master-scaled joint arm exceeds the native float range")
    return float(np.float32(value))


def joint_scale_from_preferences(preferences):
    try:
        return validate_joint_scale(preferences.get("joint_scale", JOINT_SCALE_DEFAULT))
    except (ValueError, TypeError, OverflowError):
        return JOINT_SCALE_DEFAULT


def _extent_votes(bounds):
    """Match PxBounds3 extent arithmetic; invalid/overflow bounds cannot vote."""
    bounds = np.asarray(bounds, dtype="f4").reshape(-1, 2, 3)
    valid = np.isfinite(bounds).all(axis=(1, 2)) & (bounds[:, 1] >= bounds[:, 0]).all(axis=1)
    with np.errstate(over="ignore", invalid="ignore"):
        extents = (bounds[:, 1] - bounds[:, 0]) * np.float32(.5)
        sizes = np.max(extents, axis=1) * np.float32(2)
    valid &= np.isfinite(sizes)
    # -0 has a different sortable float bit pattern, but the same extent.
    sizes[sizes == 0] = np.float32(0)
    return sizes, valid


def _upper_median(chunks):
    """Exact nonnegative F32 weighted rank, with fixed extra storage.

    The generator is restarted four times. Each pass retains one 256-counter
    histogram and one input chunk; no triangle-sized vote list is allocated.
    """
    prefix = 0
    rank = None
    total = 0
    for shift in (24, 16, 8, 0):
        histogram = np.zeros(256, dtype="u8")
        for values, weights in chunks():
            bits = np.asarray(values, dtype="f4").view("u4")
            if shift != 24:
                keep = (bits >> (shift + 8)) == prefix
                bits, weights = bits[keep], weights[keep]
            np.add.at(histogram, (bits >> shift) & 255, weights)
        if rank is None:
            total = int(histogram.sum())
            if not total:
                return None, 0
            rank = total // 2
        cumulative = histogram.cumsum()
        digit = int(np.searchsorted(cumulative, rank, side="right"))
        rank -= int(cumulative[digit - 1]) if digit else 0
        prefix = (prefix << 8) | digit
    return float(np.array([prefix], dtype="u4").view("f4")[0]), total


@dataclass(frozen=True)
class JointCalibration:
    capture_key: int
    segment_epoch: int
    scene_id: int
    frame: int
    base_length: float
    source: str
    vote_count: int

    def matches(self, view):
        return (self.capture_key, self.segment_epoch, self.scene_id) == (
            view.capture_key, view.segment_epoch, view.scene_id)

    def request(self, scale, *, ovd=False, master_scale=1.):
        scale = validate_joint_scale(scale, master_scale=master_scale)
        with np.errstate(over="ignore"):
            arm = np.float32(self.base_length) * np.float32(max(scale, .5)) * np.float32(1.5) * np.float32(master_scale)
        if ovd and not np.isfinite(arm):
            raise ValueError("Joint scale exceeds the finite range for this capture")
        return {"joint_scale": scale, "joint_base_length": self.base_length}


def _first_surface_geometry(adapter, actor):
    # Qt stops at the first positive triangleMesh reference, even if that
    # mesh later has no vertices. Do not silently choose a subsequent shape.
    for shape in adapter.references(actor, "shapes"):
        for geometry in adapter.references(shape, "Geometry")[:1]:
            meshes = adapter.references(geometry, "triangleMesh")
            if meshes and meshes[0] > 0:
                return shape, geometry, meshes[0]
    return None


def resolve_joint_base(view, adapter, settings=None, *, selected=(), hidden=()):
    if not view.ready or not view.request_applied or adapter.session is not view:
        raise ValueError("Joint calibration requires the same complete owned presentation")
    snapshot = view.geometry_snapshot()
    scene = next(scene for scene in view.scenes() if scene.id == view.scene_id)
    plans, primitive_rows = [], []
    if snapshot is not None:
        visible, _ = geometry_visibility(snapshot, settings, scene_id=scene.instance_id,
                                          selected=selected, hidden=hidden)
        resources = {int(mesh.id): mesh for mesh in snapshot.meshes}
        first_surfaces = {}
        for row in np.flatnonzero(visible):
            kind = int(snapshot.kinds[row])
            if kind not in (1, 2, 3, 5) or snapshot.double_sided[row]:
                continue  # separate plane/heightfield/line/particle queues
            actor = int(snapshot.object_ids[row])
            cls = adapter.class_name(actor)
            if cls in ("PxDeformableVolume", "PxDeformableSurface"):
                mesh = resources.get(int(snapshot.resource_ids[row]))
                if cls == "PxDeformableSurface":
                    if actor not in first_surfaces:
                        first_surfaces[actor] = _first_surface_geometry(adapter, actor)
                    actual = (int(snapshot.shape_ids[row]), int(snapshot.geometry_ids[row]),
                              int(mesh.source_id) if mesh is not None else None)
                    if actual != first_surfaces[actor]:
                        continue
                if mesh is not None:
                    plans.append(mesh)
            else:
                primitive_rows.append(row)

    def chunks():
        for start in range(0, len(primitive_rows), TRIANGLE_CHUNK):
            rows = primitive_rows[start:start + TRIANGLE_CHUNK]
            sizes, valid = _extent_votes(snapshot.bounds[rows])
            kinds = snapshot.kinds[rows]
            weights = np.where(kinds == 1, 2, np.where(kinds == 3, 3, 1)).astype("u8")
            yield sizes[valid], weights[valid]
        for source in plans:
            vertices = source.vertices
            indices = source.indices.reshape(-1)
            # Native indices are complete triangles; reject a malformed tail
            # defensively instead of reading beyond the immutable resource.
            stop = len(indices) - len(indices) % 3
            for start in range(0, stop, TRIANGLE_CHUNK * 3):
                triangles = indices[start:min(stop, start + TRIANGLE_CHUNK * 3)].reshape(-1, 3)
                triangles = triangles[(triangles < len(vertices)).all(axis=1)]
                points = vertices[triangles]
                with np.errstate(over="ignore", invalid="ignore"):
                    normal = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
                    squared = np.sum(normal * normal, axis=1, dtype="f4")
                valid = np.isfinite(points).all(axis=(1, 2)) & (squared >= np.float32(1e-20))
                points = points[valid]
                bounds = np.stack((points.min(axis=1), points.max(axis=1)), axis=1)
                sizes, valid = _extent_votes(bounds)
                yield sizes[valid], np.ones(int(valid.sum()), dtype="u8")

    median, count = _upper_median(chunks)
    if median is not None and median > 0:
        base, source = median, "solid-geometry"
    else:
        base, source = float(view.tolerance_length), "captured-tolerance"
        if not math.isfinite(base) or base <= 0:
            raise ValueError("Owned capture tolerance length must be finite and positive")
    return JointCalibration(view.capture_key, view.segment_epoch, view.scene_id,
                            view.frame, base, source, count)

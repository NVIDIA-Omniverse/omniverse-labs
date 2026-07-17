# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared-stage composition helpers for the OVRTX + OVPhysX diagnostic path."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from numbers import Real
from pathlib import Path
from typing import Any, Mapping, Sequence
import struct
import zlib


MUTATION_AUTHORITY_OVPHYSX = "OVPhysX"


@dataclass(frozen=True)
class BodyPose:
    prim_path: str
    translate: tuple[float, float, float]
    orient: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.prim_path, str) or not self.prim_path:
            raise ValueError("body pose prim_path must be non-empty")
        if not isinstance(self.translate, tuple) or len(self.translate) != 3:
            raise ValueError("body pose translate must have exactly three values")
        if not isinstance(self.orient, tuple) or len(self.orient) != 4:
            raise ValueError("body pose orient must have exactly four values")
        components = (*self.translate, *self.orient)
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            for value in components
        ):
            raise ValueError("body pose components must be numeric finite values")
        if math.hypot(*self.orient) == 0.0:
            raise ValueError("body pose quaternion must be nonzero")


@dataclass(frozen=True)
class BodyVelocity:
    prim_path: str
    linear: tuple[float, float, float]
    angular: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if not isinstance(self.prim_path, str) or not self.prim_path:
            raise ValueError("body velocity prim_path must be non-empty")
        if not isinstance(self.linear, tuple) or len(self.linear) != 3:
            raise ValueError("body velocity linear must have exactly three values")
        if not isinstance(self.angular, tuple) or len(self.angular) != 3:
            raise ValueError("body velocity angular must have exactly three values")
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            for value in (*self.linear, *self.angular)
        ):
            raise ValueError("body velocity components must be numeric finite values")


@dataclass(frozen=True)
class StageMutation:
    authority: str
    simulation_time_ns: int
    revision: int
    dirty_paths: tuple[str, ...]


class RuntimeStageHost:
    """Minimal demo-local scene authority for the first composition validation."""

    def __init__(self, scene_id: str) -> None:
        self.scene_id = scene_id
        self.revision = 0
        self._body_poses: dict[str, BodyPose] = {}
        self.last_mutation: StageMutation | None = None

    def publish_ovphysx_poses(self, poses: Sequence[BodyPose], simulation_time_ns: int) -> StageMutation:
        dirty_paths: list[str] = []
        for pose in poses:
            previous = self._body_poses.get(pose.prim_path)
            if previous == pose:
                continue
            self._body_poses[pose.prim_path] = pose
            dirty_paths.append(pose.prim_path)

        if dirty_paths:
            self.revision += 1
        mutation = StageMutation(
            authority=MUTATION_AUTHORITY_OVPHYSX,
            simulation_time_ns=int(simulation_time_ns),
            revision=self.revision,
            dirty_paths=tuple(dirty_paths),
        )
        self.last_mutation = mutation
        return mutation

    def body_poses_for(self, prim_paths: Sequence[str]) -> tuple[BodyPose, ...]:
        paths = tuple(prim_paths)
        if len(set(paths)) != len(paths):
            raise ValueError("body pose paths must be unique")
        missing = [path for path in paths if path not in self._body_poses]
        if missing:
            raise KeyError(f"body poses are missing: {', '.join(missing)}")
        return tuple(self._body_poses[path] for path in paths)

    def diagnostics(self) -> dict[str, Any]:
        mutation = self.last_mutation
        return {
            "scene_id": self.scene_id,
            "revision": self.revision,
            "body_paths": sorted(self._body_poses),
            "last_mutation": None
            if mutation is None
            else {
                "authority": mutation.authority,
                "simulation_time_ns": mutation.simulation_time_ns,
                "revision": mutation.revision,
                "dirty_paths": list(mutation.dirty_paths),
            },
        }


def pose_from_ovphysx_state(state: Mapping[str, Any]) -> BodyPose:
    prim_path = str(state.get("prim_path", ""))
    if not prim_path:
        raise ValueError("OVPhysX body state is missing prim_path")
    return BodyPose(
        prim_path=prim_path,
        translate=_float3(state.get("translate"), "translate"),
        orient=_quatf(state.get("orient"), "orient"),
    )

def write_rgba_png(path: Path, width: int, height: int, rgba8: bytes) -> dict[str, Any]:
    expected_size = int(width) * int(height) * 4
    if len(rgba8) != expected_size:
        raise ValueError(f"RGBA payload has {len(rgba8)} bytes, expected {expected_size}")

    path.parent.mkdir(parents=True, exist_ok=True)
    scanlines = bytearray()
    row_size = int(width) * 4
    for row in range(int(height)):
        scanlines.append(0)
        start = row * row_size
        scanlines.extend(rgba8[start : start + row_size])

    def chunk(kind: bytes, payload: bytes) -> bytes:
        checksum = zlib.crc32(kind)
        checksum = zlib.crc32(payload, checksum)
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum & 0xFFFFFFFF)

    png = b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            chunk(b"IHDR", struct.pack(">IIBBBBB", int(width), int(height), 8, 6, 0, 0, 0)),
            chunk(b"IDAT", zlib.compress(bytes(scanlines))),
            chunk(b"IEND", b""),
        ]
    )
    path.write_bytes(png)
    return {
        "path": str(path),
        "width": int(width),
        "height": int(height),
        "size_bytes": len(png),
        "sha256": hashlib.sha256(png).hexdigest(),
    }


def _float3(value: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(value, Mapping) or not value.get("found"):
        raise ValueError(f"OVPhysX body state is missing {label}")
    return (float(value["x"]), float(value["y"]), float(value["z"]))


def _quatf(value: Any, label: str) -> tuple[float, float, float, float]:
    if not isinstance(value, Mapping) or not value.get("found"):
        raise ValueError(f"OVPhysX body state is missing {label}")
    return (float(value["i"]), float(value["j"]), float(value["k"]), float(value["r"]))

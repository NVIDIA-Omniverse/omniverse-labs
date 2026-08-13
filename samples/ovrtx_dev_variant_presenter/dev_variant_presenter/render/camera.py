# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Orbit/pan/dolly camera controller. Pure numpy — no ovrtx import.

Produces a row-vector (USD) 4x4 float64 transform for omni:xform:
rows = [right, up, -forward, eye]. Supports Y-up and Z-up. WebRTC mouse coords
are already render-pixel space (NVST-mapped) — pass deltas straight in.
"""
from __future__ import annotations

import math

import numpy as np

_PITCH_LIMIT = math.pi / 2 - 0.01
_MIN_DISTANCE = 0.01


class CameraController:
    def __init__(self, target=(0.0, 0.0, 0.0), distance: float = 10.0,
                 up_axis: str = "Y", azimuth: float = 0.0, elevation: float = 0.0):
        self.target = np.asarray(target, dtype=np.float64)
        self.distance = max(_MIN_DISTANCE, float(distance))
        self.up_axis = up_axis.upper()
        self.azimuth = float(azimuth)
        self.elevation = max(-_PITCH_LIMIT, min(_PITCH_LIMIT, float(elevation)))

    # --- spherical <-> direction (unit offset from target to eye) ---
    def _dir(self) -> np.ndarray:
        ce, se = math.cos(self.elevation), math.sin(self.elevation)
        ca, sa = math.cos(self.azimuth), math.sin(self.azimuth)
        if self.up_axis == "Z":
            return np.array([ce * ca, ce * sa, se], dtype=np.float64)
        return np.array([ce * sa, se, ce * ca], dtype=np.float64)  # Y-up

    def _world_up(self) -> np.ndarray:
        return np.array([0.0, 0.0, 1.0] if self.up_axis == "Z" else [0.0, 1.0, 0.0])

    def _set_from_offset(self, offset: np.ndarray) -> None:
        self.distance = max(_MIN_DISTANCE, float(np.linalg.norm(offset)))
        d = offset / self.distance
        if self.up_axis == "Z":
            self.elevation = math.asin(max(-1.0, min(1.0, d[2])))
            self.azimuth = math.atan2(d[1], d[0])
        else:
            self.elevation = math.asin(max(-1.0, min(1.0, d[1])))
            self.azimuth = math.atan2(d[0], d[2])
        self.elevation = max(-_PITCH_LIMIT, min(_PITCH_LIMIT, self.elevation))

    # --- interaction ---
    def orbit(self, d_az: float, d_el: float) -> None:
        self.azimuth += d_az
        self.elevation = max(-_PITCH_LIMIT, min(_PITCH_LIMIT, self.elevation + d_el))

    def dolly(self, dz: float) -> None:
        self.distance = max(_MIN_DISTANCE, self.distance + dz)

    def pan(self, dx: float, dy: float) -> None:
        eye = self.target + self.distance * self._dir()
        forward = (self.target - eye)
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, self._world_up())
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        self.target = self.target + right * dx + up * dy

    def snap_to(self, matrix: np.ndarray, *, focus_distance: float | None = None,
                bbox_center=None) -> None:
        """Seed the orbit state from an authored camera world transform (row-vector 4x4)."""
        m = np.asarray(matrix, dtype=np.float64)
        eye = m[3, :3].copy()
        forward = -m[2, :3].copy()
        forward /= np.linalg.norm(forward)
        if focus_distance and focus_distance > _MIN_DISTANCE:
            self.target = eye + forward * float(focus_distance)
        elif bbox_center is not None:
            c = np.asarray(bbox_center, dtype=np.float64)
            self.target = eye + forward * max(_MIN_DISTANCE, float(np.dot(c - eye, forward)))
        else:
            t = float(np.dot(-eye, forward))  # project world origin onto the ray
            self.target = eye + forward * (t if t > _MIN_DISTANCE else self.distance)
        self._set_from_offset(eye - self.target)

    def to_xform(self) -> np.ndarray:
        eye = self.target + self.distance * self._dir()
        forward = self.target - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, self._world_up())
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        m = np.eye(4, dtype=np.float64)
        m[0, :3] = right
        m[1, :3] = up
        m[2, :3] = -forward
        m[3, :3] = eye
        return m

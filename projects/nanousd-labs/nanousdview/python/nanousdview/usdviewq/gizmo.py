# SPDX-License-Identifier: Apache-2.0
"""
3D transform manipulator for nanousdview.

Translated from ddiakopoulos/tinygizmo (Unlicense / public domain) into
pure Python + numpy. Drawn on top of the renderer's pixel blit via Qt
QPainter — no GL context required. Mutates a RigidTransform in WORLD
space; the host (StageView) writes it back to the selected prim.

Matrix convention is column-vector (apply via M @ v_col), matching the
existing `_drawSelectionHighlight` math in stageView.py: vertices are
row-stored, projected with `pts_row @ vp.T`.
"""
from __future__ import annotations

import enum
import math

import numpy as np
from .qt import QtCore, QtGui

_TAU = 2.0 * math.pi
_EPS = 1e-9
_INF = float("inf")


class Mode(enum.IntEnum):
    TRANSLATE = 0
    ROTATE = 1
    SCALE = 2


class _Interact(enum.IntEnum):
    NONE = 0
    TRANSLATE_X = 1
    TRANSLATE_Y = 2
    TRANSLATE_Z = 3
    TRANSLATE_YZ = 4
    TRANSLATE_ZX = 5
    TRANSLATE_XY = 6
    TRANSLATE_XYZ = 7
    ROTATE_X = 8
    ROTATE_Y = 9
    ROTATE_Z = 10
    SCALE_X = 11
    SCALE_Y = 12
    SCALE_Z = 13


# ---- math helpers ---------------------------------------------------------

def _norm(v):
    n = float(np.linalg.norm(v))
    return v / n if n > _EPS else np.asarray(v, dtype=np.float64).copy()


def _qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        ax * bw + aw * bx + ay * bz - az * by,
        ay * bw + aw * by + az * bx - ax * bz,
        az * bw + aw * bz + ax * by - ay * bx,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dtype=np.float64)


def _qconj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def _qrot(q, v):
    qv = np.asarray(q[:3], dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    t = 2.0 * np.cross(qv, v)
    return v + q[3] * t + np.cross(qv, t)


def _qaxis_angle(axis, angle):
    s = math.sin(angle * 0.5)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s,
                     math.cos(angle * 0.5)], dtype=np.float64)


def _qmat3(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _quat_from_matrix(R):
    """3x3 rotation matrix → unit quaternion (xyzw)."""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (m[2, 1] - m[1, 2]) / S
        y = (m[0, 2] - m[2, 0]) / S
        z = (m[1, 0] - m[0, 1]) / S
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        S = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / S
        x = 0.25 * S
        y = (m[0, 1] + m[1, 0]) / S
        z = (m[0, 2] + m[2, 0]) / S
    elif m[1, 1] > m[2, 2]:
        S = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / S
        x = (m[0, 1] + m[1, 0]) / S
        y = 0.25 * S
        z = (m[1, 2] + m[2, 1]) / S
    else:
        S = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / S
        x = (m[0, 2] + m[2, 0]) / S
        y = (m[1, 2] + m[2, 1]) / S
        z = 0.25 * S
    q = np.array([x, y, z, w], dtype=np.float64)
    n = float(np.linalg.norm(q))
    return q / n if n > _EPS else np.array([0, 0, 0, 1], dtype=np.float64)


def decompose_trs(M_col):
    """Decompose a 4x4 column-vector matrix into (position, quat_xyzw, scale).
    Discards mirror info — scale is taken absolute. Suitable for the
    'simple xform' authoring path in nanousdview."""
    M = np.asarray(M_col, dtype=np.float64)
    pos = M[:3, 3].copy()
    R = M[:3, :3].copy()
    sx = float(np.linalg.norm(R[:, 0]))
    sy = float(np.linalg.norm(R[:, 1]))
    sz = float(np.linalg.norm(R[:, 2]))
    if sx > _EPS:
        R[:, 0] /= sx
    if sy > _EPS:
        R[:, 1] /= sy
    if sz > _EPS:
        R[:, 2] /= sz
    if float(np.linalg.det(R)) < 0.0:
        R[:, 0] = -R[:, 0]
        sx = -sx
    quat = _quat_from_matrix(R)
    scale = np.array([abs(sx) if sx != 0 else 1.0,
                      sy if sy > _EPS else 1.0,
                      sz if sz > _EPS else 1.0], dtype=np.float64)
    return pos, quat, scale


# ---- RigidTransform -------------------------------------------------------

class RigidTransform:
    __slots__ = ("position", "orientation", "scale")

    def __init__(self, position=None, orientation=None, scale=None):
        self.position = (np.zeros(3, dtype=np.float64) if position is None
                         else np.asarray(position, dtype=np.float64).copy())
        self.orientation = (np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
                            if orientation is None
                            else np.asarray(orientation, dtype=np.float64).copy())
        self.scale = (np.ones(3, dtype=np.float64) if scale is None
                      else np.asarray(scale, dtype=np.float64).copy())

    def matrix(self):
        R = _qmat3(self.orientation)
        M = np.eye(4, dtype=np.float64)
        M[:3, 0] = R[:, 0] * self.scale[0]
        M[:3, 1] = R[:, 1] * self.scale[1]
        M[:3, 2] = R[:, 2] * self.scale[2]
        M[:3, 3] = self.position
        return M

    def transform_vector(self, v):
        return _qrot(self.orientation, np.asarray(v, dtype=np.float64) * self.scale)

    def transform_point(self, p):
        return self.position + self.transform_vector(p)

    def detransform_vector(self, v):
        return _qrot(_qconj(self.orientation), np.asarray(v, dtype=np.float64)) / self.scale

    def detransform_point(self, p):
        return self.detransform_vector(np.asarray(p, dtype=np.float64) - self.position)


# ---- mesh generation ------------------------------------------------------

def _make_box(min_b, max_b):
    a = np.asarray(min_b, dtype=np.float64)
    b = np.asarray(max_b, dtype=np.float64)
    V = np.array([
        [a[0], a[1], a[2]], [a[0], a[1], b[2]], [a[0], b[1], b[2]], [a[0], b[1], a[2]],
        [b[0], a[1], a[2]], [b[0], b[1], a[2]], [b[0], b[1], b[2]], [b[0], a[1], b[2]],
        [a[0], a[1], a[2]], [b[0], a[1], a[2]], [b[0], a[1], b[2]], [a[0], a[1], b[2]],
        [a[0], b[1], a[2]], [a[0], b[1], b[2]], [b[0], b[1], b[2]], [b[0], b[1], a[2]],
        [a[0], a[1], a[2]], [a[0], b[1], a[2]], [b[0], b[1], a[2]], [b[0], a[1], a[2]],
        [a[0], a[1], b[2]], [b[0], a[1], b[2]], [b[0], b[1], b[2]], [a[0], b[1], b[2]],
    ], dtype=np.float64)
    T = np.array([
        [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], [8, 9, 10], [8, 10, 11],
        [12, 13, 14], [12, 14, 15], [16, 17, 18], [16, 18, 19], [20, 21, 22], [20, 22, 23],
    ], dtype=np.int32)
    return V, T


def _make_lathe(axis, arm1, arm2, slices, points, eps=0.0):
    """Revolve a 2D profile (M x 2: along-axis, radial) around `axis`."""
    axis = np.asarray(axis, dtype=np.float64)
    arm1 = np.asarray(arm1, dtype=np.float64)
    arm2 = np.asarray(arm2, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    M = len(points)
    V = np.empty(((slices + 1) * M, 3), dtype=np.float64)
    Tlist = []
    for i in range(slices + 1):
        ang = ((i % slices) * _TAU / slices) + (_TAU / 8.0)
        c, s = math.cos(ang), math.sin(ang)
        radial = arm1 * c + arm2 * s
        for j in range(M):
            V[i * M + j] = axis * points[j, 0] + radial * points[j, 1] + eps
        if i > 0:
            for j in range(1, M):
                i0 = (i - 1) * M + (j - 1)
                i1 = (i - 0) * M + (j - 1)
                i2 = (i - 0) * M + (j - 0)
                i3 = (i - 1) * M + (j - 0)
                Tlist.append([i0, i1, i2])
                Tlist.append([i0, i2, i3])
    T = np.array(Tlist, dtype=np.int32)
    return V, T


# ---- ray / mesh intersection ---------------------------------------------

def _intersect_ray_mesh(ro, rd, V, T):
    """Vectorized Möller-Trumbore over all triangles. Returns nearest t, or inf."""
    if len(T) == 0:
        return _INF
    v0 = V[T[:, 0]]
    v1 = V[T[:, 1]]
    v2 = V[T[:, 2]]
    e1 = v1 - v0
    e2 = v2 - v0
    h = np.cross(rd, e2)
    a = np.einsum("ij,ij->i", e1, h)
    valid = np.abs(a) > _EPS
    if not np.any(valid):
        return _INF
    a_safe = np.where(valid, a, 1.0)
    f = 1.0 / a_safe
    s = ro - v0
    u = f * np.einsum("ij,ij->i", s, h)
    valid &= (u >= 0.0) & (u <= 1.0)
    if not np.any(valid):
        return _INF
    q = np.cross(s, e1)
    v = f * np.einsum("j,ij->i", rd, q)
    valid &= (v >= 0.0) & (u + v <= 1.0)
    if not np.any(valid):
        return _INF
    t = f * np.einsum("ij,ij->i", e2, q)
    valid &= t >= 0.0
    if not np.any(valid):
        return _INF
    return float(np.min(t[valid]))


# ---- pick-ray helper for the host -----------------------------------------

def make_pick_ray(mx, my, vp_col, w, h):
    """Convert pixel (mx, my) → world-space ray (origin, direction).
    `vp_col` is the column-vector view-projection matrix used by
    stageView's projection path."""
    inv_vp = np.linalg.inv(vp_col)
    nx = (2.0 * mx / max(w, 1)) - 1.0
    ny = 1.0 - (2.0 * my / max(h, 1))
    near = inv_vp @ np.array([nx, ny, -1.0, 1.0], dtype=np.float64)
    far_ = inv_vp @ np.array([nx, ny, 1.0, 1.0], dtype=np.float64)
    if abs(near[3]) < _EPS or abs(far_[3]) < _EPS:
        return np.zeros(3), np.array([0.0, 0.0, -1.0])
    near = near[:3] / near[3]
    far_ = far_[:3] / far_[3]
    direction = far_ - near
    n = float(np.linalg.norm(direction))
    if n > _EPS:
        direction /= n
    return near, direction


# ---- application state ----------------------------------------------------

class AppState:
    """Per-frame input snapshot, fed by the host on every event."""
    __slots__ = (
        "mouse_left", "viewport_size", "ray_origin", "ray_direction",
        "cam_eye", "cam_yfov", "screenspace_scale",
        "snap_translation", "snap_scale", "snap_rotation",
    )

    def __init__(self):
        self.mouse_left = False
        self.viewport_size = (0, 0)
        self.ray_origin = np.zeros(3, dtype=np.float64)
        self.ray_direction = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        self.cam_eye = np.zeros(3, dtype=np.float64)
        self.cam_yfov = math.radians(60.0)
        self.screenspace_scale = 80.0
        self.snap_translation = 0.0
        self.snap_scale = 0.0
        self.snap_rotation = 0.0


# ---- internal gizmo state ------------------------------------------------

class _State:
    __slots__ = ("active", "hover", "interaction_mode",
                 "original_position", "original_orientation", "original_scale",
                 "click_offset")

    def __init__(self):
        self.active = False
        self.hover = False
        self.interaction_mode = _Interact.NONE
        self.original_position = np.zeros(3, dtype=np.float64)
        self.original_orientation = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self.original_scale = np.ones(3, dtype=np.float64)
        self.click_offset = np.zeros(3, dtype=np.float64)


# ---- transform gizmo -----------------------------------------------------

class TransformGizmo:
    """Single-target transform manipulator. Translate / Rotate / Scale.
    Public API:
      - update(state, passive=False): latch input, compute click/release edges.
      - tick(transform): run state machine, mutate transform in-place.
      - draw(painter, vp_col, w, h): render the per-frame drawlist.
      - set_mode(Mode): change between T/R/S (no-op while dragging).
      - is_dragging: True while the user holds a handle.
    `transform` is a RigidTransform in WORLD space."""

    def __init__(self):
        self.mode = Mode.TRANSLATE
        self.local_toggle = True
        self._prev_mouse_left = False
        self._has_clicked = False
        self._has_released = False
        self._passive = False
        self._state = _State()
        self._app: AppState = AppState()
        self._drawlist: list[tuple[np.ndarray, np.ndarray, tuple]] = []
        self._templates = self._build_templates()

    @staticmethod
    def _build_templates():
        arrow_pts = [(0.25, 0.0), (0.25, 0.05), (1.0, 0.05), (1.0, 0.10), (1.2, 0.0)]
        mace_pts = [(0.25, 0.0), (0.25, 0.05), (1.0, 0.05), (1.0, 0.10),
                    (1.25, 0.10), (1.25, 0.0)]
        ring_pts = [(0.025, 1.0), (-0.025, 1.0), (-0.025, 1.0), (-0.025, 1.1),
                    (-0.025, 1.1), (0.025, 1.1), (0.025, 1.1), (0.025, 1.0)]
        x = np.array([1.0, 0.0, 0.0])
        y = np.array([0.0, 1.0, 0.0])
        z = np.array([0.0, 0.0, 1.0])
        return {
            _Interact.TRANSLATE_X: (*_make_lathe(x, y, z, 16, arrow_pts),
                                    (1.0, 0.5, 0.5, 1.0), (1.0, 0.0, 0.0, 1.0)),
            _Interact.TRANSLATE_Y: (*_make_lathe(y, z, x, 16, arrow_pts),
                                    (0.5, 1.0, 0.5, 1.0), (0.0, 1.0, 0.0, 1.0)),
            _Interact.TRANSLATE_Z: (*_make_lathe(z, x, y, 16, arrow_pts),
                                    (0.5, 0.5, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0)),
            _Interact.TRANSLATE_YZ: (*_make_box((-0.01, 0.25, 0.25), (0.01, 0.75, 0.75)),
                                     (0.5, 1.0, 1.0, 0.5), (0.0, 1.0, 1.0, 0.6)),
            _Interact.TRANSLATE_ZX: (*_make_box((0.25, -0.01, 0.25), (0.75, 0.01, 0.75)),
                                     (1.0, 0.5, 1.0, 0.5), (1.0, 0.0, 1.0, 0.6)),
            _Interact.TRANSLATE_XY: (*_make_box((0.25, 0.25, -0.01), (0.75, 0.75, 0.01)),
                                     (1.0, 1.0, 0.5, 0.5), (1.0, 1.0, 0.0, 0.6)),
            _Interact.TRANSLATE_XYZ: (*_make_box((-0.05, -0.05, -0.05), (0.05, 0.05, 0.05)),
                                      (0.9, 0.9, 0.9, 0.5), (1.0, 1.0, 1.0, 0.6)),
            _Interact.ROTATE_X: (*_make_lathe(x, y, z, 32, ring_pts, 0.003),
                                 (1.0, 0.5, 0.5, 1.0), (1.0, 0.0, 0.0, 1.0)),
            _Interact.ROTATE_Y: (*_make_lathe(y, z, x, 32, ring_pts, -0.003),
                                 (0.5, 1.0, 0.5, 1.0), (0.0, 1.0, 0.0, 1.0)),
            _Interact.ROTATE_Z: (*_make_lathe(z, x, y, 32, ring_pts, 0.0),
                                 (0.5, 0.5, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0)),
            _Interact.SCALE_X: (*_make_lathe(x, y, z, 16, mace_pts),
                                (1.0, 0.5, 0.5, 1.0), (1.0, 0.0, 0.0, 1.0)),
            _Interact.SCALE_Y: (*_make_lathe(y, z, x, 16, mace_pts),
                                (0.5, 1.0, 0.5, 1.0), (0.0, 1.0, 0.0, 1.0)),
            _Interact.SCALE_Z: (*_make_lathe(z, x, y, 16, mace_pts),
                                (0.5, 0.5, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0)),
        }

    # ---- public API -------------------------------------------------
    def update(self, state: AppState, passive: bool = False):
        self._passive = passive
        self._app = state
        if passive:
            self._has_clicked = False
            self._has_released = False
        else:
            self._has_clicked = (not self._prev_mouse_left) and state.mouse_left
            self._has_released = self._prev_mouse_left and (not state.mouse_left)
        self._drawlist.clear()

    def tick(self, t: RigidTransform) -> bool:
        if self.mode == Mode.TRANSLATE:
            self._position_gizmo(t)
        elif self.mode == Mode.ROTATE:
            self._orientation_gizmo(t)
        else:
            self._scale_gizmo(t)
        if not self._passive:
            self._prev_mouse_left = self._app.mouse_left
            if self._has_released:
                self._state.interaction_mode = _Interact.NONE
                self._state.active = False
        return self._state.hover or self._state.active

    @property
    def is_dragging(self) -> bool:
        return self._state.active

    def set_mode(self, mode: Mode):
        if not self._state.active:
            self.mode = mode

    # ---- internal helpers -------------------------------------------
    def _scale_screenspace(self, position):
        if self._app.screenspace_scale <= 0.0:
            return 1.0
        dist = float(np.linalg.norm(position - self._app.cam_eye))
        half_h = math.tan(self._app.cam_yfov * 0.5) * dist
        wpp = (2.0 * half_h) / max(self._app.viewport_size[1], 1)
        return wpp * self._app.screenspace_scale

    @staticmethod
    def _detransform_ray_scaled(scale, ro, rd):
        return ro / scale, rd / scale

    def _intersect_components(self, ro, rd, components):
        best_t = _INF
        best_kind = _Interact.NONE
        for kind in components:
            V, T, _, _ = self._templates[kind]
            t = _intersect_ray_mesh(ro, rd, V, T)
            if t < best_t:
                best_t = t
                best_kind = kind
        return best_kind, best_t

    def _push_drawlist(self, kinds, model_col, active_kind):
        for kind in kinds:
            V_local, T, base_col, hi_col = self._templates[kind]
            V4 = np.hstack([V_local, np.ones((V_local.shape[0], 1), dtype=np.float64)])
            V_world = (V4 @ model_col.T)[:, :3]
            color = base_col if kind == active_kind else hi_col
            self._drawlist.append((V_world, T, color))

    def _model_matrix(self, p: RigidTransform, draw_scale: float):
        M_rot = np.eye(4, dtype=np.float64)
        M_rot[:3, :3] = _qmat3(p.orientation)
        M_t = np.eye(4, dtype=np.float64)
        M_t[:3, 3] = p.position
        M_s = np.diag([draw_scale, draw_scale, draw_scale, 1.0])
        return M_t @ M_rot @ M_s

    # ---- translate gizmo --------------------------------------------
    def _position_gizmo(self, t: RigidTransform):
        identity_q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        p = RigidTransform(t.position, t.orientation if self.local_toggle else identity_q)
        draw_scale = self._scale_screenspace(p.position)

        if self._has_clicked:
            self._state.interaction_mode = _Interact.NONE

        ro = p.detransform_point(self._app.ray_origin)
        rd = p.detransform_vector(self._app.ray_direction)
        ro, rd = self._detransform_ray_scaled(draw_scale, ro, rd)
        components = [_Interact.TRANSLATE_X, _Interact.TRANSLATE_Y, _Interact.TRANSLATE_Z,
                      _Interact.TRANSLATE_YZ, _Interact.TRANSLATE_ZX, _Interact.TRANSLATE_XY,
                      _Interact.TRANSLATE_XYZ]
        hit_kind, hit_t = self._intersect_components(ro, rd, components)
        self._state.hover = hit_t != _INF

        if self._has_clicked:
            self._state.interaction_mode = hit_kind
            if hit_kind != _Interact.NONE:
                hit_local = ro + rd * hit_t
                hit_world = p.transform_point(hit_local * draw_scale)
                self._state.click_offset = hit_world - p.position
                self._state.original_position = p.position.copy()
                self._state.active = True
            else:
                self._state.active = False

        if self.local_toggle:
            R = _qmat3(t.orientation)
            axes = (R[:, 0], R[:, 1], R[:, 2])
        else:
            axes = (np.array([1.0, 0.0, 0.0]),
                    np.array([0.0, 1.0, 0.0]),
                    np.array([0.0, 0.0, 1.0]))

        if self._state.active and self._app.mouse_left:
            mode = self._state.interaction_mode
            anchor0 = self._state.original_position + self._state.click_offset
            new_anchor = anchor0
            if mode == _Interact.TRANSLATE_X:
                new_anchor = self._axis_drag(axes[0], anchor0)
            elif mode == _Interact.TRANSLATE_Y:
                new_anchor = self._axis_drag(axes[1], anchor0)
            elif mode == _Interact.TRANSLATE_Z:
                new_anchor = self._axis_drag(axes[2], anchor0)
            elif mode == _Interact.TRANSLATE_YZ:
                new_anchor = self._plane_drag(axes[0], anchor0)
            elif mode == _Interact.TRANSLATE_ZX:
                new_anchor = self._plane_drag(axes[1], anchor0)
            elif mode == _Interact.TRANSLATE_XY:
                new_anchor = self._plane_drag(axes[2], anchor0)
            elif mode == _Interact.TRANSLATE_XYZ:
                view_fwd = _norm(self._app.cam_eye - p.position)
                new_anchor = self._plane_drag(view_fwd, anchor0)
            t.position[:] = new_anchor - self._state.click_offset

        # Rebuild p with the (possibly updated) position so the drawn
        # gizmo follows the prim during the same frame.
        p.position[:] = t.position
        model = self._model_matrix(p, draw_scale)
        self._push_drawlist(components, model, self._state.interaction_mode)

    def _axis_drag(self, axis, anchor0):
        view_to = anchor0 - self._app.cam_eye
        plane_tan = np.cross(axis, view_to)
        plane_n = np.cross(axis, plane_tan)
        new_pt = self._plane_intersect(plane_n, anchor0)
        if new_pt is None:
            return anchor0
        return anchor0 + axis * float(np.dot(new_pt - anchor0, axis))

    def _plane_drag(self, plane_n, anchor0):
        new_pt = self._plane_intersect(plane_n, anchor0)
        if new_pt is None:
            return anchor0
        if self._app.snap_translation > 0.0:
            new_pt = np.floor(new_pt / self._app.snap_translation) * self._app.snap_translation
        return new_pt

    def _plane_intersect(self, plane_n, plane_pt):
        denom = float(np.dot(self._app.ray_direction, plane_n))
        if abs(denom) < _EPS:
            return None
        t = float(np.dot(plane_pt - self._app.ray_origin, plane_n)) / denom
        if t < 0.0:
            return None
        return self._app.ray_origin + self._app.ray_direction * t

    # ---- rotate gizmo -----------------------------------------------
    def _orientation_gizmo(self, t: RigidTransform):
        identity_q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        p = RigidTransform(t.position, t.orientation if self.local_toggle else identity_q)
        draw_scale = self._scale_screenspace(p.position)

        if self._has_clicked:
            self._state.interaction_mode = _Interact.NONE

        ro = p.detransform_point(self._app.ray_origin)
        rd = p.detransform_vector(self._app.ray_direction)
        ro, rd = self._detransform_ray_scaled(draw_scale, ro, rd)
        components = [_Interact.ROTATE_X, _Interact.ROTATE_Y, _Interact.ROTATE_Z]
        hit_kind, hit_t = self._intersect_components(ro, rd, components)
        self._state.hover = hit_t != _INF

        if self._has_clicked:
            self._state.interaction_mode = hit_kind
            if hit_kind != _Interact.NONE:
                self._state.original_position = t.position.copy()
                self._state.original_orientation = t.orientation.copy()
                hit_local = ro + rd * hit_t
                self._state.click_offset = p.transform_point(hit_local * draw_scale)
                self._state.active = True
            else:
                self._state.active = False

        if self._state.active and self._app.mouse_left:
            mode = self._state.interaction_mode
            axis_local = None
            if mode == _Interact.ROTATE_X:
                axis_local = np.array([1.0, 0.0, 0.0])
            elif mode == _Interact.ROTATE_Y:
                axis_local = np.array([0.0, 1.0, 0.0])
            elif mode == _Interact.ROTATE_Z:
                axis_local = np.array([0.0, 0.0, 1.0])
            if axis_local is not None:
                start_orient = (self._state.original_orientation if self.local_toggle
                                else identity_q)
                self._axis_rotation_drag(axis_local, start_orient, t)

        p.orientation[:] = t.orientation if self.local_toggle else identity_q
        model = self._model_matrix(p, draw_scale)
        self._push_drawlist(components, model, self._state.interaction_mode)

    def _axis_rotation_drag(self, axis_local, start_orient, t: RigidTransform):
        the_axis = _qrot(start_orient, axis_local)
        # Plane through click_offset with normal = the_axis.
        denom = float(np.dot(self._app.ray_direction, the_axis))
        if abs(denom) < _EPS:
            return
        t_hit = (float(np.dot(self._state.click_offset - self._app.ray_origin, the_axis))
                 / denom)
        if t_hit < 0.0:
            return
        hit_pt = self._app.ray_origin + self._app.ray_direction * t_hit
        center_of_rot = (self._state.original_position
                         + the_axis * float(np.dot(the_axis,
                                                   self._state.click_offset
                                                   - self._state.original_position)))
        arm1 = _norm(self._state.click_offset - center_of_rot)
        arm2 = _norm(hit_pt - center_of_rot)
        d = float(np.dot(arm1, arm2))
        if d > 0.999:
            t.orientation[:] = self._state.original_orientation
            return
        angle = math.acos(max(-1.0, min(1.0, d)))
        if angle < 1e-3:
            t.orientation[:] = self._state.original_orientation
            return
        if self._app.snap_rotation > 0.0:
            angle = math.floor(angle / self._app.snap_rotation) * self._app.snap_rotation
        a = _norm(np.cross(arm1, arm2))
        dq = _qaxis_angle(a, angle)
        t.orientation[:] = _qmul(dq, self._state.original_orientation)

    # ---- scale gizmo -------------------------------------------------
    def _scale_gizmo(self, t: RigidTransform):
        p = RigidTransform(t.position, t.orientation)
        draw_scale = self._scale_screenspace(p.position)

        if self._has_clicked:
            self._state.interaction_mode = _Interact.NONE

        ro = p.detransform_point(self._app.ray_origin)
        rd = p.detransform_vector(self._app.ray_direction)
        ro, rd = self._detransform_ray_scaled(draw_scale, ro, rd)
        components = [_Interact.SCALE_X, _Interact.SCALE_Y, _Interact.SCALE_Z]
        hit_kind, hit_t = self._intersect_components(ro, rd, components)
        self._state.hover = hit_t != _INF

        if self._has_clicked:
            self._state.interaction_mode = hit_kind
            if hit_kind != _Interact.NONE:
                self._state.original_scale = t.scale.copy()
                hit_local = ro + rd * hit_t
                self._state.click_offset = p.transform_point(hit_local * draw_scale)
                self._state.active = True
            else:
                self._state.active = False

        if self._state.active and self._app.mouse_left:
            mode = self._state.interaction_mode
            if mode == _Interact.SCALE_X:
                self._axis_scale_drag(np.array([1.0, 0.0, 0.0]), t)
            elif mode == _Interact.SCALE_Y:
                self._axis_scale_drag(np.array([0.0, 1.0, 0.0]), t)
            elif mode == _Interact.SCALE_Z:
                self._axis_scale_drag(np.array([0.0, 0.0, 1.0]), t)

        model = self._model_matrix(p, draw_scale)
        self._push_drawlist(components, model, self._state.interaction_mode)

    def _axis_scale_drag(self, axis_local, t: RigidTransform):
        axis_world = _qrot(t.orientation, axis_local)
        view_to = t.position - self._app.cam_eye
        plane_tan = np.cross(axis_world, view_to)
        plane_n = np.cross(axis_world, plane_tan)
        denom = float(np.dot(self._app.ray_direction, plane_n))
        if abs(denom) < _EPS:
            return
        t_hit = float(np.dot(t.position - self._app.ray_origin, plane_n)) / denom
        if t_hit < 0.0:
            return
        hit_world = self._app.ray_origin + self._app.ray_direction * t_hit
        offset_axis_mag = float(np.dot(hit_world - self._state.click_offset, axis_world))
        new_scale = self._state.original_scale.copy()
        idx = int(np.argmax(np.abs(axis_local)))
        new_scale[idx] = max(0.01, min(1000.0,
                                       self._state.original_scale[idx] + offset_axis_mag))
        if self._app.snap_scale > 0.0:
            new_scale = np.floor(new_scale / self._app.snap_scale) * self._app.snap_scale
        t.scale[:] = new_scale

    # ---- drawing ----------------------------------------------------
    def draw(self, painter: QtGui.QPainter, vp_col: np.ndarray, w: int, h: int):
        """Project drawlist triangles through `vp_col` (4x4 column-vector
        matrix) and stroke each as a filled QPolygonF with an outline.
        Triangles are depth-sorted globally so transparent plane handles
        composite correctly with the opaque axis arrows behind them."""
        items = []
        for V_world, T, color in self._drawlist:
            V4 = np.hstack([V_world, np.ones((V_world.shape[0], 1), dtype=np.float64)])
            clip = V4 @ vp_col.T
            ww = np.where(np.abs(clip[:, 3]) < _EPS, 1.0, clip[:, 3])
            ndc = clip[:, :3] / ww[:, None]
            sx = (ndc[:, 0] + 1.0) * 0.5 * w
            sy = (1.0 - (ndc[:, 1] + 1.0) * 0.5) * h
            screen = np.stack([sx, sy], axis=1)
            for tri in T:
                a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
                if clip[a, 3] <= 0 or clip[b, 3] <= 0 or clip[c, 3] <= 0:
                    continue
                z_mean = float((ndc[a, 2] + ndc[b, 2] + ndc[c, 2]) / 3.0)
                items.append((z_mean, screen[a], screen[b], screen[c], color))
        items.sort(key=lambda it: -it[0])
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        for _z, p0, p1, p2, color in items:
            r, g, b, alpha = color
            qfill = QtGui.QColor.fromRgbF(r, g, b, alpha)
            qedge = QtGui.QColor.fromRgbF(r * 0.4, g * 0.4, b * 0.4,
                                          min(1.0, alpha + 0.2))
            painter.setBrush(QtGui.QBrush(qfill))
            painter.setPen(QtGui.QPen(qedge, 1))
            poly = QtGui.QPolygonF([
                QtCore.QPointF(float(p0[0]), float(p0[1])),
                QtCore.QPointF(float(p1[0]), float(p1[1])),
                QtCore.QPointF(float(p2[0]), float(p2[1])),
            ])
            painter.drawPolygon(poly)


# ---- USD writeback -------------------------------------------------------

def write_xform_to_prim(prim, position, orientation_xyzw, scale):
    """Author a canonical Translate / Orient / Scale stack on `prim`,
    replacing any existing xformOpOrder. Per the 'simple xform' rule:
    we don't preserve arbitrary op stacks — gizmo edits always land
    as this fixed three-op canonical form."""
    from pxr import UsdGeom, Gf
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(float(position[0]),
                                     float(position[1]),
                                     float(position[2])))
    qx, qy, qz, qw = orientation_xyzw
    xf.AddOrientOp().Set(Gf.Quatf(float(qw),
                                  Gf.Vec3f(float(qx), float(qy), float(qz))))
    xf.AddScaleOp().Set(Gf.Vec3f(float(scale[0]),
                                 float(scale[1]),
                                 float(scale[2])))

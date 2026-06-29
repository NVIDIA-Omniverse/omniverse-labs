# SPDX-License-Identifier: Apache-2.0
"""
nanousd/pxr_compat — `from pxr import …` resolves here, never to OpenUSD.

Layered on top of the native nanobind pxr_compat shim (_native_compat.py),
which contains the bulk of the pxr-style schema compatibility surface.

This package adds:
  - the gaps Phase-0 inventory identified (Gf.Camera, Gf.Frustum,
    Gf.Ray, Gf.Range1f, Gf.Cross, Tf.Notice, missing namespaces)
  - signature fixes (MaterialBindingAPI.ComputeBoundMaterial)
  - sys.modules registration so `from pxr import Usd, Sdf, ...` works

Phase 1 of NUVIEW_PLAN.md.
"""
from __future__ import annotations

import importlib as _importlib
import importlib.util as _ilu
import sys as _sys
import types as _types
from pathlib import Path as _Path

# ---------------------------------------------------------------- load native compatibility shim
_shim_path = _Path(__file__).resolve().parent / "_native_compat.py"
_spec = _ilu.spec_from_file_location("nanousd.pxr_compat._native_compat", _shim_path)
_native_compat = _ilu.module_from_spec(_spec)
_sys.modules["nanousd.pxr_compat._native_compat"] = _native_compat
_spec.loader.exec_module(_native_compat)

_NANOUSD_REFERENCE_OFFSETS = {}


# ---------------------------------------------------------------- signature fix
# The native shim's MaterialBindingAPI.ComputeBoundMaterial(self) — usdview calls
# it with materialPurpose=. Patch.
if hasattr(_native_compat, "UsdShade") and hasattr(_native_compat.UsdShade, "MaterialBindingAPI"):
    _orig_cbm = _native_compat.UsdShade.MaterialBindingAPI.ComputeBoundMaterial

    def _cbm(self, materialPurpose=None):  # noqa: N803
        try:
            return _orig_cbm(self, materialPurpose)
        except TypeError:
            return _orig_cbm(self)

    _native_compat.UsdShade.MaterialBindingAPI.ComputeBoundMaterial = _cbm


# ---------------------------------------------------------------- Gf extensions
import math as _math

_Gf = _native_compat.Gf

# --- Gf.Cross (free function on the module)
def _gf_cross(a, b):
    return _Gf.Vec3d(
        float(a[1]) * float(b[2]) - float(a[2]) * float(b[1]),
        float(a[2]) * float(b[0]) - float(a[0]) * float(b[2]),
        float(a[0]) * float(b[1]) - float(a[1]) * float(b[0]),
    )

if not hasattr(_Gf, "Cross"):
    _Gf.Cross = staticmethod(_gf_cross) if isinstance(_Gf, type) else _gf_cross


# --- Gf.Range1f
class _Range1f:
    __slots__ = ("_min", "_max")

    def __init__(self, mn=float("inf"), mx=float("-inf")):
        self._min = float(mn)
        self._max = float(mx)

    def GetMin(self): return self._min
    def GetMax(self): return self._max
    min = property(GetMin, lambda self, v: self.SetMin(v))
    max = property(GetMax, lambda self, v: self.SetMax(v))
    def SetMin(self, v): self._min = float(v)
    def SetMax(self, v): self._max = float(v)
    def IsEmpty(self): return self._max < self._min
    def UnionWith(self, v): self._min = min(self._min, v); self._max = max(self._max, v)

if not hasattr(_Gf, "Range1f"):
    _Gf.Range1f = _Range1f


# --- Gf.Matrix4d.SetRotate(rot)
def _m4d_set_rotate(self, rot):
    # rot is a Gf.Rotation (axis + angle in degrees) or Quatd. Our shim's
    # Rotation/Quatd both expose .GetAxis() / .GetAngle() OR .GetReal()/Imaginary().
    # Normalise to a quaternion (x,y,z,w) then set the upper-left 3×3.
    if hasattr(rot, "GetReal"):  # Quat-like
        w = float(rot.GetReal())
        x, y, z = (float(v) for v in rot.GetImaginary())
    else:  # Rotation-like: axis+angle
        axis = rot.GetAxis()
        ang = _math.radians(float(rot.GetAngle()))
        ax, ay, az = float(axis[0]), float(axis[1]), float(axis[2])
        n = _math.sqrt(ax * ax + ay * ay + az * az) or 1.0
        ax, ay, az = ax / n, ay / n, az / n
        s = _math.sin(ang * 0.5)
        x, y, z, w = ax * s, ay * s, az * s, _math.cos(ang * 0.5)
    # Build rotation matrix
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    # The native shim's Matrix4d stores row-major 4x4 in self.data (verify by inspection)
    if hasattr(self, "data") and len(self.data) == 16:
        m = self.data
    else:
        m = [0.0] * 16
        self.data = m
    m[0]  = 1 - 2 * (yy + zz); m[1]  = 2 * (xy - wz);     m[2]  = 2 * (xz + wy);     m[3]  = 0
    m[4]  = 2 * (xy + wz);     m[5]  = 1 - 2 * (xx + zz); m[6]  = 2 * (yz - wx);     m[7]  = 0
    m[8]  = 2 * (xz - wy);     m[9]  = 2 * (yz + wx);     m[10] = 1 - 2 * (xx + yy); m[11] = 0
    m[12] = 0; m[13] = 0; m[14] = 0; m[15] = 1
    return self

if hasattr(_Gf, "Matrix4d") and not hasattr(_Gf.Matrix4d, "SetRotate"):
    _Gf.Matrix4d.SetRotate = _m4d_set_rotate


# The native shim's _GfModule.Matrix4d is a *function* that returns a plain
# numpy.ndarray, on which we cannot stamp instance methods like SetRotate.
# Wrap it to return a subclass instance instead.
import numpy as _np


class _Matrix4dArr(_np.ndarray):
    """numpy.ndarray subclass with Pixar Gf.Matrix4d methods bolted on."""
    def __new__(cls, *args):
        if len(args) == 0:
            arr = _np.eye(4, dtype=_np.float64)
        elif len(args) == 16:
            arr = _np.array(args, dtype=_np.float64).reshape(4, 4)
        elif len(args) == 1:
            a = args[0]
            if isinstance(a, (int, float)):  # Gf.Matrix4d(s) -> s * identity
                arr = _np.eye(4, dtype=_np.float64) * float(a)
            else:
                m = _np.asarray(a, dtype=_np.float64)
                if m.shape == (4, 4):
                    arr = m.copy()
                elif m.size == 16:
                    arr = m.reshape(4, 4).copy()
                elif m.size == 4:  # Gf.Matrix4d(Vec4) -> diagonal (was: crash 4->4x4)
                    arr = _np.diag(m.ravel())
                else:
                    arr = _np.eye(4, dtype=_np.float64)
        elif len(args) == 2:  # Gf.Matrix4d(rotation|matrix3, translation)
            out = _np.eye(4, dtype=_np.float64).view(cls)
            rot = args[0]
            if hasattr(rot, "GetReal") or hasattr(rot, "GetAxis") or hasattr(rot, "_axis"):
                out.SetRotate(rot)  # Gf.Rotation / Gf.Quat* — let unsupported types raise
            else:  # a 3x3 rotation matrix (Gf.Matrix3d/ndarray): embed in the upper-left block
                m3 = _np.asarray(rot, dtype=_np.float64)
                if m3.shape == (3, 3):
                    out[:3, :3] = m3
                else:
                    raise TypeError(f"Gf.Matrix4d: unsupported rotation arg of shape {m3.shape}")
            out.SetTranslate(args[1])
            return out
        else:
            arr = _np.eye(4, dtype=_np.float64)
        return arr.view(cls)

    def SetRotate(self, rot):
        if hasattr(rot, "GetReal"):
            w = float(rot.GetReal())
            x, y, z = (float(v) for v in rot.GetImaginary())
        else:
            axis = rot.GetAxis()
            ang = _math.radians(float(rot.GetAngle()))
            ax, ay, az = float(axis[0]), float(axis[1]), float(axis[2])
            n = _math.sqrt(ax * ax + ay * ay + az * az) or 1.0
            ax, ay, az = ax / n, ay / n, az / n
            s = _math.sin(ang * 0.5)
            x, y, z, w = ax * s, ay * s, az * s, _math.cos(ang * 0.5)
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        # Row-vector convention (p' = p * M), matching OpenUSD's GfMatrix4d and
        # the native ComputeLocalToWorldTransform path. This is the TRANSPOSE of
        # the textbook column-vector (M * p) quaternion->matrix; getting it wrong
        # silently inverts every rotation that round-trips through ExtractRotation.
        self[0, 0] = 1 - 2 * (yy + zz); self[0, 1] = 2 * (xy + wz); self[0, 2] = 2 * (xz - wy); self[0, 3] = 0
        self[1, 0] = 2 * (xy - wz); self[1, 1] = 1 - 2 * (xx + zz); self[1, 2] = 2 * (yz + wx); self[1, 3] = 0
        self[2, 0] = 2 * (xz + wy); self[2, 1] = 2 * (yz - wx); self[2, 2] = 1 - 2 * (xx + yy); self[2, 3] = 0
        self[3, 0] = 0; self[3, 1] = 0; self[3, 2] = 0; self[3, 3] = 1
        return self

    def SetTranslate(self, t):
        self[3, 0], self[3, 1], self[3, 2] = float(t[0]), float(t[1]), float(t[2])
        return self

    def SetIdentity(self):
        self[:] = _np.eye(4, dtype=_np.float64)
        return self

    # OpenUSD Gf.Matrix4d '*' is the matrix PRODUCT (row-vector compose:
    # p' = p * (A*B)), and Matrix * scalar is a component scale. numpy's
    # inherited ndarray '*' is element-wise (Hadamard) — silently wrong for
    # transform composition (it drops off-diagonal and translation terms) — so
    # we override it here. Converting to a plain ndarray first avoids recursion.
    def __mul__(self, other):
        if isinstance(other, (int, float, _np.floating, _np.integer)):
            return _Matrix4dArr(_np.asarray(self, dtype=_np.float64) * float(other))
        o = _np.asarray(other, dtype=_np.float64)
        if o.shape == (4, 4):
            return _Matrix4dArr(_np.asarray(self, dtype=_np.float64) @ o)
        return NotImplemented

    def __matmul__(self, other):
        o = _np.asarray(other, dtype=_np.float64)
        if o.shape == (4, 4):
            return _Matrix4dArr(_np.asarray(self, dtype=_np.float64) @ o)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float, _np.floating, _np.integer)):
            return _Matrix4dArr(_np.asarray(self, dtype=_np.float64) * float(other))
        return NotImplemented

    def __imul__(self, other):
        result = self.__mul__(other)
        if result is NotImplemented:
            return NotImplemented
        self[:] = result
        return self

    def GetTranspose(self):
        return _Matrix4dArr(self.T.copy())

    def GetInverse(self):
        return _Matrix4dArr(_np.linalg.inv(self))

    def GetDeterminant(self):
        return float(_np.linalg.det(self))

    def ExtractTranslation(self):
        return _Gf.Vec3d(float(self[3, 0]), float(self[3, 1]), float(self[3, 2]))

    def ExtractRotation(self):
        # Quaternion from the (row-vector) upper-3x3 via Shoemake's largest-pivot
        # method — robust through the 180-degree case — returned as a Gf.Rotation
        # (axis + angle in degrees), matching OpenUSD's GfMatrix4d::ExtractRotation.
        # The antisymmetric terms use the row-vector sign convention (the
        # TRANSPOSE of the textbook column-vector formulas); validated against the
        # usd-core oracle (tests/test_matrix_rotation_conformance.py).
        m = _np.asarray(self[:3, :3], dtype=_np.float64)
        trace = m[0, 0] + m[1, 1] + m[2, 2]
        if trace > 0.0:
            s = _math.sqrt(trace + 1.0) * 2.0       # s = 4w
            w = 0.25 * s
            x = (m[1, 2] - m[2, 1]) / s
            y = (m[2, 0] - m[0, 2]) / s
            z = (m[0, 1] - m[1, 0]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = _math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0   # s = 4x
            w = (m[1, 2] - m[2, 1]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[2, 0] + m[0, 2]) / s
        elif m[1, 1] > m[2, 2]:
            s = _math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0   # s = 4y
            w = (m[2, 0] - m[0, 2]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = _math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0   # s = 4z
            w = (m[0, 1] - m[1, 0]) / s
            x = (m[2, 0] + m[0, 2]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
        w = max(-1.0, min(1.0, w))
        angle = 2.0 * _math.acos(w)
        sin_half = _math.sqrt(max(0.0, 1.0 - w * w))
        if sin_half < 1e-9:
            return _Gf.Rotation((1.0, 0.0, 0.0), 0.0)
        return _Gf.Rotation((x / sin_half, y / sin_half, z / sin_half), _math.degrees(angle))

    def ExtractRotationMatrix(self):
        return _Matrix3dArr(_np.asarray(self[:3, :3], dtype=_np.float64)) \
            if "_Matrix3dArr" in globals() else _np.asarray(self[:3, :3], dtype=_np.float64)

    def Transform(self, pt):
        # Row-vector point transform with the homogeneous divide (GfMatrix4d::Transform).
        v = _np.array([float(pt[0]), float(pt[1]), float(pt[2]), 1.0], dtype=_np.float64) @ \
            _np.asarray(self, dtype=_np.float64)
        w = v[3]
        if abs(w) > 1e-12 and w != 1.0:
            v = v / w
        return _Gf.Vec3d(float(v[0]), float(v[1]), float(v[2]))

    def TransformDir(self, vec):
        # Direction transform: rotation/scale only, no translation, no w-divide.
        v = _np.array([float(vec[0]), float(vec[1]), float(vec[2]), 0.0], dtype=_np.float64) @ \
            _np.asarray(self, dtype=_np.float64)
        return _Gf.Vec3d(float(v[0]), float(v[1]), float(v[2]))

    def __eq__(self, other):
        try:
            other_arr = _np.asarray(other, dtype=_np.float64)
        except Exception:
            return False
        return bool(
            other_arr.shape == self.shape
            and _np.array_equal(_np.asarray(self), other_arr)
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    def GetRow(self, i):
        return _Gf.Vec4d(float(self[i, 0]), float(self[i, 1]), float(self[i, 2]), float(self[i, 3]))

    def GetColumn(self, i):
        return _Gf.Vec4d(float(self[0, i]), float(self[1, i]), float(self[2, i]), float(self[3, i]))

    def Orthonormalize(self, issueWarning=True):
        # Gram-Schmidt on the upper-3x3 rows (row-vector basis), translation kept.
        m = _np.asarray(self[:3, :3], dtype=_np.float64)
        r0 = m[0]
        n0 = _np.linalg.norm(r0)
        if n0 < 1e-12:
            return False
        r0 = r0 / n0
        r1 = m[1] - _np.dot(m[1], r0) * r0
        n1 = _np.linalg.norm(r1)
        if n1 < 1e-12:
            return False
        r1 = r1 / n1
        r2 = _np.cross(r0, r1)
        self[0, :3] = r0
        self[1, :3] = r1
        self[2, :3] = r2
        return True

    def RemoveScaleShear(self):
        out = _Matrix4dArr(_np.asarray(self, dtype=_np.float64))
        out.Orthonormalize()
        return out

    _warned_attrs: set = set()

    def __getattr__(self, name):
        # An un-mirrored Gf.Matrix4d method warns ONCE (so the gap is no longer
        # silent) and returns a no-op. We do not raise: a hard raise here
        # regressed real callers (e.g. testUsdGeomXformable) that tolerate the
        # no-op, and some are reached at usdview startup. numpy's own methods
        # resolve normally and never reach here; only missing Gf methods do.
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in _Matrix4dArr._warned_attrs:
            _Matrix4dArr._warned_attrs.add(name)
            import warnings as _warnings
            _warnings.warn(
                f"pxr_compat: Gf.Matrix4d.{name}() is not implemented (no-op)", stacklevel=2
            )
        def _stub(*a, **kw):
            return self
        return _stub


# Replace the native shim's plain-ndarray factory.
_Gf.Matrix4d = _Matrix4dArr

_XFORM_CACHE = getattr(getattr(_native_compat, "UsdGeom", None), "XformCache", None)
if _XFORM_CACHE is not None and not getattr(_XFORM_CACHE, "_nanousd_matrix4d_result", False):
    _orig_get_l2w = _XFORM_CACHE.GetLocalToWorldTransform

    def _get_l2w_matrix4d(self, prim):
        return _Gf.Matrix4d(_orig_get_l2w(self, prim))

    _XFORM_CACHE.GetLocalToWorldTransform = _get_l2w_matrix4d
    _XFORM_CACHE._nanousd_matrix4d_result = True


# --- Gf.Camera (just enough for freeCamera.py)
class _Camera:
    Y_UP_TO_Z_UP_MATRIX = None  # set after class definition
    Projection = type("Projection", (), {"Perspective": 0, "Orthographic": 1})
    Perspective = 0
    Orthographic = 1
    APERTURE_UNIT = 1.0

    def __init__(self, other=None):
        if other is not None:
            self.transform = _Gf.Matrix4d(getattr(other, "transform", _Gf.Matrix4d()))
            self.focalLength = float(getattr(other, "focalLength", 50.0))
            self.horizontalAperture = float(getattr(other, "horizontalAperture", 36.0))
            self.verticalAperture = float(getattr(other, "verticalAperture", 24.0))
            self.horizontalApertureOffset = float(getattr(other, "horizontalApertureOffset", 0.0))
            self.verticalApertureOffset = float(getattr(other, "verticalApertureOffset", 0.0))
            cr = getattr(other, "clippingRange", None)
            self.clippingRange = _Range1f(
                cr.GetMin() if cr is not None and hasattr(cr, "GetMin") else 0.01,
                cr.GetMax() if cr is not None and hasattr(cr, "GetMax") else 1e9,
            )
            self.projection = getattr(other, "projection", _Camera.Projection.Perspective)
            self.fStop = float(getattr(other, "fStop", 0.0))
            self.focusDistance = float(getattr(other, "focusDistance", 0.0))
            return
        self.transform = _Gf.Matrix4d()
        self.focalLength = 50.0
        self.horizontalAperture = 36.0
        self.verticalAperture = 24.0
        self.horizontalApertureOffset = 0.0
        self.verticalApertureOffset = 0.0
        self.clippingRange = _Range1f(0.01, 1e9)
        self.projection = _Camera.Projection.Perspective
        self.fStop = 0.0
        self.focusDistance = 0.0

    def GetFrustum(self):
        return _Frustum(self)

    @property
    def frustum(self):
        # Real Gf.Camera exposes both GetFrustum() and a `frustum` property;
        # usdview's FreeCamera reads self._camera.frustum (property). Without
        # this, the catch-all __getattr__ returned a no-op stub and
        # _pullFromCameraTransform crashed on frustum.position.
        return _Frustum(self)

    @property
    def fovVertical(self):
        return _math.degrees(2.0 * _math.atan(self.verticalAperture / (2.0 * self.focalLength)))

    @property
    def fovHorizontal(self):
        return _math.degrees(2.0 * _math.atan(self.horizontalAperture / (2.0 * self.focalLength)))

    @property
    def aspectRatio(self):
        return (self.horizontalAperture / self.verticalAperture) if self.verticalAperture else 1.0

    @aspectRatio.setter
    def aspectRatio(self, value):
        value = float(value) if value else 1.0
        self.horizontalAperture = self.verticalAperture * value

    FOVVertical = "vertical"
    FOVHorizontal = "horizontal"

    def GetFieldOfView(self, direction=None):
        if direction == self.FOVHorizontal:
            return self.fovHorizontal
        return self.fovVertical

    def SetPerspectiveFromAspectRatioAndFieldOfView(
            self, aspectRatio, fieldOfView, direction=None,
            horizontalAperture=None):
        self.projection = self.Projection.Perspective
        if fieldOfView is None: fieldOfView = 60.0
        if aspectRatio is None or aspectRatio <= 0: aspectRatio = 16.0 / 9.0
        fov_v = float(fieldOfView)
        is_horizontal = (direction == self.FOVHorizontal)
        focal = 50.0
        self.focalLength = focal
        if is_horizontal:
            self.horizontalAperture = 2 * focal * _math.tan(_math.radians(fov_v) / 2.0)
            self.verticalAperture = self.horizontalAperture / max(aspectRatio, 1e-6)
        else:
            self.verticalAperture = 2 * focal * _math.tan(_math.radians(fov_v) / 2.0)
            self.horizontalAperture = self.verticalAperture * aspectRatio
        return self

    def SetOrthographicFromAspectRatioAndSize(self, aspectRatio, orthoSize, direction=None):
        self.projection = self.Projection.Orthographic
        self.verticalAperture = orthoSize
        self.horizontalAperture = orthoSize * aspectRatio
        return self

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        def _stub(*a, **kw): return None
        return _stub


_Camera.Y_UP_TO_Z_UP_MATRIX = _Gf.Matrix4d()  # populated lazily — rotation about X by -90°


# --- Gf.Frustum (subset usdview reads)
class _Frustum:
    def __init__(self, cam=None):
        if cam is None:
            self._near = 0.01
            self._far = 1e9
            self._fov_v = 45.0
            self._aspect = 1.0
        else:
            self._near = cam.clippingRange.GetMin()
            self._far = cam.clippingRange.GetMax()
            self._fov_v = cam.fovVertical
            self._aspect = (cam.horizontalAperture / cam.verticalAperture) if cam.verticalAperture else 1.0
        self._cam = cam

    def GetNearFar(self): return _Range1f(self._near, self._far)

    def GetFOV(self): return self._fov_v

    # --- camera-derived geometry (usdview FreeCamera + framing/picking read these).
    # The camera transform is camera->world (row-vector); the camera looks down
    # -Z with +Y up, matching Gf.Camera/Gf.Frustum.
    def _xf(self):
        return self._cam.transform if self._cam is not None else _Gf.Matrix4d()

    @property
    def position(self):
        return self._xf().ExtractTranslation()

    def ComputeViewDirection(self):
        d = self._xf().TransformDir(_Gf.Vec3d(0.0, 0.0, -1.0))
        v = _np.asarray([d[0], d[1], d[2]], dtype=_np.float64)
        n = _np.linalg.norm(v) or 1.0
        return _Gf.Vec3d(*(v / n))

    def ComputeUpVector(self):
        d = self._xf().TransformDir(_Gf.Vec3d(0.0, 1.0, 0.0))
        v = _np.asarray([d[0], d[1], d[2]], dtype=_np.float64)
        n = _np.linalg.norm(v) or 1.0
        return _Gf.Vec3d(*(v / n))

    def ComputeViewMatrix(self):
        # world -> eye (inverse of the camera-to-world transform).
        return self._xf().GetInverse()

    def ComputeViewInverse(self):
        return _Gf.Matrix4d(self._xf())

    def ComputeProjectionMatrix(self):
        # Symmetric perspective from fovVertical + aspect, row-vector (p_clip =
        # p_eye * P), z in [-1, 1] (GL/USD convention).
        n, f = float(self._near), float(self._far)
        t = n * _math.tan(_math.radians(self._fov_v) * 0.5)
        r = t * (self._aspect if self._aspect else 1.0)
        m = _Gf.Matrix4d(1.0)
        m[0, 0] = n / r if r else 1.0; m[0, 1] = 0; m[0, 2] = 0;                    m[0, 3] = 0
        m[1, 0] = 0;                   m[1, 1] = n / t if t else 1.0; m[1, 2] = 0;  m[1, 3] = 0
        m[2, 0] = 0; m[2, 1] = 0; m[2, 2] = -(f + n) / (f - n) if f != n else -1.0; m[2, 3] = -1.0
        m[3, 0] = 0; m[3, 1] = 0; m[3, 2] = -2.0 * f * n / (f - n) if f != n else 0.0; m[3, 3] = 0
        return m

    def ComputeCorners(self):
        # 8 world-space frustum corners (near rect then far rect), via the
        # camera-to-world transform.
        n, f = float(self._near), float(self._far)
        tn = n * _math.tan(_math.radians(self._fov_v) * 0.5)
        rn = tn * (self._aspect if self._aspect else 1.0)
        tf = f * _math.tan(_math.radians(self._fov_v) * 0.5)
        rf = tf * (self._aspect if self._aspect else 1.0)
        xf = self._xf()
        local = [(-rn, -tn, -n), (rn, -tn, -n), (-rn, tn, -n), (rn, tn, -n),
                 (-rf, -tf, -f), (rf, -tf, -f), (-rf, tf, -f), (rf, tf, -f)]
        return [xf.Transform(_Gf.Vec3d(*p)) for p in local]

    def Intersects(self, item):
        # Conservative: never cull (always "visible"). usdview uses this only as
        # a draw/visibility optimization, so returning True is correct-but-
        # unoptimized rather than risking a wrong-cull from buggy plane math.
        return True


# --- Gf.Ray (minimal; usdview pick paths use it)
class _Ray:
    def __init__(self, startPoint=None, direction=None):
        self._start = tuple(startPoint) if startPoint is not None else (0.0, 0.0, 0.0)
        self._dir = tuple(direction) if direction is not None else (0.0, 0.0, -1.0)

    def GetStartPoint(self): return _Gf.Vec3d(*self._start)
    def GetDirection(self): return _Gf.Vec3d(*self._dir)


for _name, _cls in [("Camera", _Camera), ("Frustum", _Frustum), ("Ray", _Ray)]:
    if not hasattr(_Gf, _name):
        setattr(_Gf, _name, _cls)


# ---------------------------------------------------------------- Tf module
class _TfNoticeBase:
    """Base for any notice type — usdview uses Tf.Notice.Register(NoticeT, callback, sender)."""
    pass


class _ObjectsChangedNotice(_TfNoticeBase):
    def __init__(self, sender, resyncedPaths=None, changedInfoOnlyPaths=None):
        self._sender = sender
        self._resynced = list(resyncedPaths or [])
        self._changedInfo = list(changedInfoOnlyPaths or [])

    def GetResyncedPaths(self): return self._resynced
    def GetChangedInfoOnlyPaths(self): return self._changedInfo


class _StageContentsChanged(_TfNoticeBase):
    def __init__(self, sender):
        self._sender = sender


class _Notice:
    """Minimal-viable Tf.Notice. Listeners stored in a class-level dict
    keyed by (notice_type, sender_id). Senders fire via `Tf.Notice.Send`."""
    _listeners: dict = {}
    _next_handle: int = 1
    ObjectsChanged = _ObjectsChangedNotice
    StageContentsChanged = _StageContentsChanged

    @classmethod
    def Register(cls, noticeType, callback, sender=None):
        sender_id = id(sender) if sender is not None else 0
        key = (noticeType, sender_id)
        cls._listeners.setdefault(key, []).append(callback)

        class _Handle:
            def __init__(self, k, cb):
                self._k = k; self._cb = cb
            def Revoke(self):
                lst = _Notice._listeners.get(self._k, [])
                if self._cb in lst:
                    lst.remove(self._cb)

        return _Handle(key, callback)

    @classmethod
    def Send(cls, notice, sender=None):
        sender_id = id(sender) if sender is not None else id(getattr(notice, "_sender", None))
        # When no sender is known (e.g. an attribute Set firing from inside
        # _MuPrim — which has no back-reference to the stage), fan out to
        # every listener regardless of their registered sender.
        broadcast_all = (sender is None and getattr(notice, "_sender", None) is None)
        for key, cbs in list(cls._listeners.items()):
            ntype, sid = key
            if isinstance(notice, ntype) and (
                    broadcast_all or sid == 0 or sid == sender_id):
                for cb in list(cbs):
                    try:
                        cb(notice, sender)
                    except Exception:
                        pass


class _TfErrorException(RuntimeError):
    pass


class _TfType:
    # Pixar's pxr.Tf.Type has a sentinel `.Unknown` and a `.typeName` field.
    # usdviewq.scalarTypes.ToString reads both: a Find() that returns Unknown
    # makes the comparison `tfType != Tf.Type.Unknown` work, and `.typeName`
    # is read on the result so the formatter can pick a renderer.
    typeName = ""

    @staticmethod
    def Find(*a, **kw):
        # Pixar's Tf.Type.Find(UsdGeom.Camera) returns a Tf.Type whose
        # typeName is the C++ class name. nanousd's IsA() expects the
        # schema pretty-name (e.g. "Camera"); we reconstruct that from
        # the schema class's __name__ since pxr.UsdGeom.Camera.__name__
        # == "Camera". Empty / unrecognized input → Unknown sentinel,
        # which preserves the legacy behaviour at non-schema call sites.
        if a:
            cls = a[0]
            name = getattr(cls, "__name__", None) \
                or getattr(cls, "typeName", None)
            if isinstance(name, str) and name and name != "_TfType":
                t = _TfType()
                t.typeName = name
                return t
        return _TfType.Unknown

    @staticmethod
    def FindByName(*a, **kw):
        if a and isinstance(a[0], str) and a[0]:
            t = _TfType()
            t.typeName = a[0]
            return t
        return _TfType.Unknown

    @staticmethod
    def Define(*a, **kw):
        return _TfType()  # Pixar returns a Tf.Type instance; we return a stub

    def IsA(self, *a, **kw): return False
    def GetAllDerivedTypes(self): return []
    def GetTypeName(self): return self.typeName

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _TfType):
            return self.typeName == other.typeName
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.typeName)

    def __bool__(self) -> bool:
        return bool(self.typeName)

    @classmethod
    def __class_getitem__(cls, item):
        return cls


_TfType.Unknown = _TfType()


class _TfToken(str):
    """Wraps a str so token-typed comparisons work."""
    def __new__(cls, value=""):
        return super().__new__(cls, value)


class _PyEnum:
    @staticmethod
    def GetValueFromFullName(*a, **kw):
        return None


class _ScopedNamedFile:
    def __init__(self, *a, **kw): self.name = None
    def __enter__(self): return self
    def __exit__(self, *a): return False


# _Tf is a *module* (not a class) so we can install a module-level
# __getattr__ that returns no-op stubs for any attribute usdview reaches
# for that we haven't explicitly implemented. PEP-562 makes this clean.
_Tf = _types.ModuleType("pxr.Tf")
_Tf.Notice = _Notice
_Tf.ErrorException = _TfErrorException
_Tf.Type = _TfType
_Tf.Token = _TfToken
_Tf.PyEnum = _PyEnum
_Tf.NamedTemporaryFile = _ScopedNamedFile
_Tf.PreparePythonModule = lambda *a, **kw: None
_Tf.RegisterPyEnums = lambda *a, **kw: None
_Tf.Warn = lambda *a, **kw: None
_Tf.Status = lambda *a, **kw: None
_Tf.RaiseRuntimeError = lambda msg: (_ for _ in ()).throw(RuntimeError(msg))
_Tf.RaiseCodingError = lambda msg: (_ for _ in ()).throw(RuntimeError(msg))


def _tf_make_valid_identifier(s):
    out = []
    for ch in str(s):
        if ch.isalnum() or ch == "_":
            out.append(ch)
        else:
            out.append("_")
    if out and out[0].isdigit():
        out.insert(0, "_")
    return "".join(out) or "_"


_Tf.MakeValidIdentifier = _tf_make_valid_identifier


def _tf_singleton(cls):
    return cls  # Pixar's Tf.Singleton decorator (no-op for shim)


def _tf_catch_and_repost(*a, **kw):
    """Tf.CatchAndRepostErrors() returns a decorator. Stub: identity."""
    def _outer(fn):
        return fn
    # Allow being used both as @Tf.CatchAndRepostErrors and @Tf.CatchAndRepostErrors()
    if a and callable(a[0]) and not kw:
        return a[0]
    return _outer


_Tf.Singleton = _tf_singleton
_Tf.CatchAndRepostErrors = _tf_catch_and_repost


def _tf_module_getattr(name):
    """Fallback: any unknown Tf.X becomes a class whose instances respond
    to any further .method() chain — so usage patterns like
    `Tf.Stopwatch().Start()` and `Tf.Status('...').Detail(...)` don't
    explode at runtime."""
    if name.startswith("_"):
        raise AttributeError(name)

    class _PermissiveTf:
        def __init__(self, *a, **kw): pass
        def __getattr__(self_, k):
            if k.startswith("_"): raise AttributeError(k)
            def _stub(*a, **kw): return None
            return _stub
        def __call__(self_, *a, **kw): return self_

    _PermissiveTf.__name__ = name
    setattr(_Tf, name, _PermissiveTf)
    return _PermissiveTf


_Tf.__getattr__ = _tf_module_getattr


# ---------------------------------------------------------------- Stage extensions
_UsdStageCls = type(_native_compat.Usd.Stage.Open(__file__) if False else None)  # placeholder

def _patch_stage_class():
    """Patch missing methods onto the native shim _UsdStage."""
    import inspect
    cls = None
    for name in dir(_native_compat):
        obj = getattr(_native_compat, name)
        if isinstance(obj, type) and obj.__name__ == "_UsdStage":
            cls = obj
            break
    if cls is None:
        return

    if not hasattr(cls, "GetEditTarget"):
        class _EditTarget:
            def __init__(self, stage): self._stage = stage
            def GetLayer(self): return self._stage.GetRootLayer()
            def IsValid(self): return True
        def _GetEditTarget(self): return _EditTarget(self)
        cls.GetEditTarget = _GetEditTarget

    if not hasattr(cls, "GetSessionLayer"):
        # Return an empty stub layer rather than None — usdview's
        # HasSessionVis (and similar) call session.GetPrimAtPath, which
        # blows up on None. The stub mirrors a clean Sdf.Layer with
        # no opinions: GetPrimAtPath returns None for every path,
        # pseudoRoot has no children. Real session-layer edits would
        # need a writable layer; nanousd doesn't track one yet.
        class _EmptySessionLayer:
            identifier = "session.usda"
            realPath = ""
            subLayerPaths = []
            subLayerOffsets = []
            anonymous = True
            class _PseudoRoot:
                nameChildren = []
                attributes = {}
                @staticmethod
                def RemoveProperty(*a, **kw): pass
            pseudoRoot = _PseudoRoot()
            def GetPrimAtPath(self, _path): return None
            def GetObjectAtPath(self, _path): return None
            def GetDisplayName(self): return "session"
            def GetNumTimeSamplesForPath(self, _p): return 0
            def IsMuted(self): return False
            def ExportToString(self): return "#usda 1.0\n\n"
            def Export(self, path, *args, **kwargs):
                with open(str(path), "w", encoding="utf-8") as fh:
                    fh.write(self.ExportToString())
                return True
            def Save(self): return True
            def __bool__(self): return False  # session is empty/no-edits
        _SESSION = _EmptySessionLayer()
        cls.GetSessionLayer = lambda self: _SESSION

    if not hasattr(cls, "GetUsedLayers") or \
            getattr(cls.GetUsedLayers, "__name__", "") in ("<lambda>",):
        def _get_used_layers(self):
            mu = getattr(self, "_stage", None)
            paths = list(getattr(mu, "used_layers", []) or [])
            if not paths:
                return [self.GetRootLayer()] if self.GetRootLayer() else []
            out = []
            seen = set()
            for idx, path in enumerate(paths):
                if not path or path in seen:
                    continue
                seen.add(path)
                out.append(_NuLayer(path, path, stage_h=mu, layer_idx=idx))
            return out if out else ([self.GetRootLayer()] if self.GetRootLayer() else [])
        cls.GetUsedLayers = _get_used_layers

    if not hasattr(cls, "MuteAndUnmuteLayers"):
        cls.MuteAndUnmuteLayers = lambda self, mute, unmute: None  # no-op

_patch_stage_class()


# ---------------------------------------------------------------- Prim extensions
def _bind_variant_set_apis(prim_cls):
    """Wire variant set helpers through the native _MuPrim wrapper."""

    def _mu(prim):
        mu = getattr(prim, "_prim", None)
        if mu is not None:
            return mu
        stage = getattr(prim, "_stage_ref", None) or getattr(prim, "_stage", None)
        if stage is not None:
            try:
                root = stage.GetPseudoRoot()
                return getattr(root, "_prim", None)
            except Exception:
                return None
        return None

    class _VariantSet:
        def __init__(self, prim, name):
            self._prim = prim
            self._name = name

        def GetVariantNames(self):
            mu = _mu(self._prim)
            return mu.variant_names(self._name) if mu is not None else []

        def GetVariantSelection(self):
            mu = _mu(self._prim)
            return mu.variant_selection(self._name) if mu is not None else ""

        def HasAuthoredVariantSelection(self):
            return bool(self.GetVariantSelection())

        def SetVariantSelection(self, variantName, layerIndex=0):
            mu = _mu(self._prim)
            ok = bool(mu and mu.set_variant_selection(self._name, variantName or "", int(layerIndex)))
            if ok:
                stage = self._prim._stage if hasattr(self._prim, "_stage") else None
                _Notice.Send(_ObjectsChangedNotice(
                    stage, resyncedPaths=[str(self._prim.GetPath())]
                ), sender=stage)
            return bool(ok)

        def ClearVariantSelection(self):
            return self.SetVariantSelection("", 0)

        def BlockVariantSelection(self):
            return self.SetVariantSelection("", 0)

        def GetName(self):
            return self._name

    class _VariantSets:
        def __init__(self, prim):
            self._prim = prim

        def GetNames(self):
            mu = _mu(self._prim)
            return mu.variant_set_names() if mu is not None else []

        def __iter__(self):
            return iter(self.GetNames())

        def GetVariantSet(self, name):
            return _VariantSet(self._prim, name)

        def GetVariantSelection(self, setName):
            mu = _mu(self._prim)
            return mu.variant_selection(setName) if mu is not None else ""

        def GetAllVariantSelections(self):
            out = {}
            for name in self.GetNames():
                sel = self.GetVariantSelection(name)
                if sel:
                    out[name] = sel
            return out

        def HasVariantSet(self, setName):
            return setName in self.GetNames()

        def SetSelection(self, setName, variantName):
            return _VariantSet(self._prim, setName).SetVariantSelection(variantName)

    prim_cls.GetVariantSets = lambda self: _VariantSets(self)
    prim_cls.GetVariantSet = lambda self, name: _VariantSet(self, name)
    prim_cls.HasVariantSets = lambda self: bool(_VariantSets(self).GetNames())


def _patch_prim_class():
    cls = None
    for name in dir(_native_compat):
        obj = getattr(_native_compat, name)
        if isinstance(obj, type) and obj.__name__ == "_UsdPrim":
            cls = obj
            break
    if cls is None:
        return

    # Specifier + Kind + Model/Group enums
    class _Specifier:
        Def = "def"; Over = "over"; Class = "class"

    if not hasattr(cls, "GetSpecifier"):
        cls.GetSpecifier = lambda self: _Specifier.Def
    if not hasattr(cls, "GetKind"):
        cls.GetKind = lambda self: ""
    if not hasattr(cls, "IsModel"):
        cls.IsModel = lambda self: False
    if not hasattr(cls, "IsGroup"):
        cls.IsGroup = lambda self: False
    if not hasattr(cls, "IsInPrototype"):
        cls.IsInPrototype = lambda self: False
    if not hasattr(cls, "GetParentInPrototype"):
        cls.GetParentInPrototype = lambda self: self.GetParent()
    if not hasattr(cls, "GetPrototype"):
        cls.GetPrototype = lambda self: None
    if not hasattr(cls, "HasAuthoredReferences"):
        cls.HasAuthoredReferences = lambda self: False

    # GetMetadata signature: takes a key
    if hasattr(cls, "GetMetadata"):
        _orig = cls.GetMetadata
        sig_takes_key = False
        try:
            import inspect
            params = inspect.signature(_orig).parameters
            sig_takes_key = "key" in params or len(params) >= 2
        except Exception:
            pass
        if not sig_takes_key:
            cls.GetMetadata = lambda self, key=None: None

    # Variant set support — real, backed by nanousd C API.
    _bind_variant_set_apis(cls)


_patch_prim_class()


# ---------------------------------------------------------------- Sdf.Layer ext
class _SdfLayer:
    """Read-only stub of Sdf.Layer for pxr_compat. usdview's layer panel
    needs a Layer object with identifier + realPath; nanousd treats
    layers more opaquely."""

    def __init__(self, identifier=""):
        self.identifier = identifier
        self.realPath = identifier
        self.subLayerPaths = []
        self.subLayerOffsets = []

    @property
    def anonymous(self):
        return not bool(self.realPath) or str(self.identifier).startswith("anon:")

    def GetDisplayName(self): return self.identifier or "anonymous"
    def ExportToString(self): return ""
    def Export(self, path, *args, **kwargs):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("")
        return True
    def Save(self): return True
    def RemoveInertSceneDescription(self): return True

    @staticmethod
    def FindOrOpen(path):
        # We don't have a layer registry; return a synthetic layer.
        return _SdfLayer(str(path))

    @staticmethod
    def Find(path):
        return None

    @staticmethod
    def CreateNew(path, args=None):
        return _SdfLayer(str(path))


if hasattr(_native_compat, "Sdf"):
    # The native shim already has a Sdf.Layer class — extend with the surface
    # usdview reaches for during stage open / layer panel rendering.
    if hasattr(_native_compat.Sdf, "Layer"):
        _LayerCls = _native_compat.Sdf.Layer
        # Counter for anonymous layer ids (Pixar's CreateAnonymous returns
        # a fresh layer each call with a unique synthetic identifier).
        _LayerCls._anon_count = 0
        def _make_layer_proxy(identifier="", real_path="", stage=None):
            proxy_cls = getattr(_native_compat, "_LayerProxy", None)
            mu_stage = getattr(_native_compat, "_MuStage", None)
            if proxy_cls is None or mu_stage is None:
                return _SdfLayer(identifier or real_path)
            proxy = proxy_cls(real_path, stage or mu_stage.create())
            if identifier:
                proxy.identifier = identifier
            if str(identifier).startswith("anon:"):
                proxy.realPath = ""
            return proxy
        @staticmethod
        def _create_anonymous(tag=""):
            _LayerCls._anon_count += 1
            return _make_layer_proxy(
                f"anon:0x{id(_LayerCls):x}-{_LayerCls._anon_count}{tag}")
        @staticmethod
        def _create_new(path, args=None):
            return _make_layer_proxy(str(path), str(path))
        @staticmethod
        def _find_or_open(path):
            mu_stage = getattr(_native_compat, "_MuStage", None)
            if mu_stage is None:
                return _SdfLayer(str(path))
            try:
                return _make_layer_proxy(str(path), str(path), mu_stage.open(str(path)))
            except Exception:
                return _SdfLayer(str(path))
        @staticmethod
        def _list_layers():
            return []
        @staticmethod
        def _display_name_from_identifier(ident):
            import os
            return os.path.basename(str(ident))
        for nm, fn in [
            ("FindOrOpen", _find_or_open),
            ("CreateNew", _create_new),
            ("CreateAnonymous", _create_anonymous),
            ("FindOrOpenRelativeToLayer", lambda layer, p: _find_or_open(p)),
            ("OpenAsAnonymous", lambda *a, **kw: _create_anonymous("")),
            ("IsAnonymousLayerIdentifier", lambda s: str(s).startswith("anon:")),
            ("GetDisplayNameFromIdentifier", _display_name_from_identifier),
        ]:
            if not hasattr(_LayerCls, nm):
                setattr(_LayerCls, nm, staticmethod(fn) if callable(fn) else fn)
        _LayerProxyCls = getattr(_native_compat, "_LayerProxy", None)
        if _LayerProxyCls is not None:
            def _layer_anonymous(self):
                return (
                    not bool(getattr(self, "realPath", ""))
                    or str(getattr(self, "identifier", "")).startswith("anon:")
                )
            def _layer_sublayer_paths(self):
                if not hasattr(self, "_subLayerPaths"):
                    self._subLayerPaths = []
                return self._subLayerPaths
            def _layer_export(self, path, *args, **kwargs):
                if getattr(self, "_stage", None) is not None:
                    return bool(self._stage.write_usda(str(path)))
                text = self.ExportToString() if hasattr(self, "ExportToString") else "#usda 1.0\n\n"
                with open(str(path), "w", encoding="utf-8") as fh:
                    fh.write(text)
                return True
            def _layer_attr_path_parts(path):
                text = str(path)
                if "." not in text:
                    return text, ""
                prim_path, attr_name = text.rsplit(".", 1)
                return prim_path, attr_name
            def _layer_get_attr(self, path):
                prim_path, attr_name = _layer_attr_path_parts(path)
                if not attr_name or getattr(self, "_stage", None) is None:
                    return None
                prim = self._stage.get_prim_at_path(prim_path)
                if prim is None:
                    return None
                return _native_compat._Attribute(prim, attr_name)
            def _layer_time_samples(self, path):
                attr = _layer_get_attr(self, path)
                return attr.GetTimeSamples() if attr is not None and attr else []
            def _layer_previous_time(self, path, time):
                samples = [t for t in _layer_time_samples(self, path) if t < float(time)]
                return (bool(samples), max(samples) if samples else 0.0)
            def _layer_next_time(self, path, time):
                samples = [t for t in _layer_time_samples(self, path) if t > float(time)]
                return (bool(samples), min(samples) if samples else 0.0)
            for nm, fn in [
                ("anonymous", property(_layer_anonymous)),
                ("subLayerPaths", property(_layer_sublayer_paths)),
                ("Export", _layer_export),
                ("RemoveInertSceneDescription", lambda self: True),
                ("GetAttributeAtPath", _layer_get_attr),
                ("ListTimeSamplesForPath", _layer_time_samples),
                ("GetNumTimeSamplesForPath", lambda self, path: len(_layer_time_samples(self, path))),
                ("GetPreviousTimeSampleForPath", _layer_previous_time),
                ("GetNextTimeSampleForPath", _layer_next_time),
            ]:
                if not hasattr(_LayerProxyCls, nm):
                    setattr(_LayerProxyCls, nm, fn)
        # Instance-side methods every layer panel touches
        for nm, fn in [
            ("Save", lambda self, *a, **kw: True),
            ("Reload", lambda self, *a, **kw: True),
            ("GetAssetInfo", lambda self: {}),
            ("GetAssetName", lambda self: getattr(self, "identifier", "")),
            ("HasOwnedSubLayers", lambda self: False),
            ("GetCompositionAssetDependencies", lambda self: []),
            ("ExportToString", lambda self: ""),
            ("Export", lambda self, path: True),
            ("GetPrimAtPath", lambda self, path: None),
            ("UpdateExternalReference", lambda self, *a, **kw: True),
        ]:
            if not hasattr(_LayerCls, nm):
                setattr(_LayerCls, nm, fn)
    else:
        _native_compat.Sdf.Layer = _SdfLayer
    if not hasattr(_native_compat.Sdf, "ChangeBlock"):
        class _ChangeBlock:
            def __enter__(self): return self
            def __exit__(self, *a): return False
        _native_compat.Sdf.ChangeBlock = _ChangeBlock

    if not hasattr(_native_compat.Sdf, "CreatePrimInLayer"):
        # usdviewq's activate/deactivate flow opens a Sdf-level primSpec on
        # the edit-target layer and sets `.active = True/False`. Stock USD
        # writes into the spec directly; the next composition pass picks it
        # up. nanousd doesn't expose Sdf primSpecs, so route the write to
        # the composed prim's metadata setter (which lands on the root layer
        # — the only layer we can edit today).
        class _PrimSpecForActivation:
            """Tiny Sdf.PrimSpec stand-in for activate/deactivate writes."""
            def __init__(self, layer, path_str):
                self._layer = layer; self._path_str = path_str
                # Resolve the live _UsdPrim once on construction. _LayerProxy
                # exposes ._owner (the _UsdStage); other shim layers expose
                # ._stage (the _UsdStage itself, when wrapped that way).
                self._prim = None
                owner = getattr(layer, "_owner", None)
                if owner is not None and hasattr(owner, "GetPrimAtPath"):
                    self._prim = owner.GetPrimAtPath(path_str)
                else:
                    s = getattr(layer, "_stage", None)
                    if s is not None and hasattr(s, "GetPrimAtPath"):
                        self._prim = s.GetPrimAtPath(path_str)
            @property
            def active(self):
                if self._prim is None: return True
                m = self._prim.GetMetadata("active") if hasattr(
                    self._prim, "GetMetadata") else None
                return True if m is None else bool(m)
            @active.setter
            def active(self, value):
                if self._prim is None or not hasattr(self._prim, "SetMetadata"):
                    return
                self._prim.SetMetadata("active", bool(value))
            @property
            def path(self):
                # Pixar's primSpec.path is an Sdf.Path. Our Sdf.Path is in _native_compat.
                return _native_compat.Sdf.Path(self._path_str)
        @staticmethod
        def _CreatePrimInLayer(layer, path):
            return _PrimSpecForActivation(layer, str(path))
        _native_compat.Sdf.CreatePrimInLayer = _CreatePrimInLayer
    if not hasattr(_native_compat.Sdf, "FileFormat"):
        class _FileFormat:
            @staticmethod
            def FindByExtension(ext): return None
            @staticmethod
            def GetFileFormat(*a, **kw): return None
            @staticmethod
            def FindAllFileFormatExtensions():
                # Used by File → Open's Qt file dialog filter list.
                return ["usd", "usda", "usdc", "usdz"]
            @staticmethod
            def FindAllDerivedFileFormatExtensions(_base=None):
                return ["usd", "usda", "usdc", "usdz"]
        _native_compat.Sdf.FileFormat = _FileFormat
    else:
        _ff = _native_compat.Sdf.FileFormat
        if not hasattr(_ff, "FindAllFileFormatExtensions"):
            _ff.FindAllFileFormatExtensions = staticmethod(
                lambda: ["usd", "usda", "usdc", "usdz"])
        if not hasattr(_ff, "FindAllDerivedFileFormatExtensions"):
            _ff.FindAllDerivedFileFormatExtensions = staticmethod(
                lambda _base=None: ["usd", "usda", "usdc", "usdz"])


# UsdGeom.BBoxCache — usdview computes scene bounds for framing
class _BBoxCache:
    def __init__(self, time, includedPurposes=None, useExtentsHint=False):
        self.time = time
        self.includedPurposes = list(includedPurposes or [])
        self.useExtentsHint = bool(useExtentsHint)
        # Per-instance cache keyed by prim path. _imageable_compute_world_bound
        # walks the entire mesh subtree (no shortcut for a "Kitchen with 500
        # meshes" case) so a re-paint at 60 Hz on a complex scene was
        # dominating the orbit drag cost — capped Kitchen_set to ~4 fps.
        # Pixar's BBoxCache is documented to maintain an internal cache;
        # mirror that. Time changes invalidate.
        self._cache: dict[str, "_Range3dBox"] = {}

    def ComputeWorldBound(self, prim):
        try:
            key = str(prim.GetPath()) if hasattr(prim, "GetPath") else None
        except Exception:
            key = None
        if key is not None:
            cached = self._cache.get(key)
            if cached is not None:
                return cached
        # Cache miss — walk the subtree.
        try:
            bounds = _imageable_compute_world_bound(prim)
            if bounds is not None:
                local_bounds = _boundable_local_extent(prim)
                if local_bounds is not None:
                    box = _Range3dBox(
                        bounds[0], bounds[1],
                        local_bounds[0], local_bounds[1],
                    )
                else:
                    box = _Range3dBox(bounds[0], bounds[1])
                if key is not None:
                    self._cache[key] = box
                return box
        except Exception:
            pass
        empty = _Range3dBox()
        if key is not None:
            self._cache[key] = empty
        return empty

    def ComputeUntransformedBound(self, prim):
        return self.ComputeWorldBound(prim)

    def Clear(self):
        self._cache.clear()

    def SetTime(self, t):
        if t != self.time:
            self._cache.clear()
        self.time = t

    def GetTime(self):
        return self.time

    def GetIncludedPurposes(self):
        return list(self.includedPurposes)

    def SetIncludedPurposes(self, purposes):
        purposes = list(purposes)
        if purposes != self.includedPurposes:
            self._cache.clear()
        self.includedPurposes = purposes

    def GetUseExtentsHint(self):
        return self.useExtentsHint

    def SetUseExtentsHint(self, useExtentsHint):
        useExtentsHint = bool(useExtentsHint)
        if useExtentsHint != self.useExtentsHint:
            self._cache.clear()
        self.useExtentsHint = useExtentsHint


class _Range3dBox:
    """Minimal stand-in for Gf.BBox3d with a Range3d inside.

    The inner Range3d formats as `[(min.x, min.y, min.z)...(max.x, max.y, max.z)]`
    to match Pixar's Gf.Range3d.__repr__ — usdview's property panel reads
    str() on the result, and we need Pixar's bracket syntax so the
    World Bounding Box row matches.
    """
    def __init__(self, mn=None, mx=None, box_mn=None, box_mx=None):
        from pxr import Gf
        # Default to an INVERTED/empty range (min=+inf, max=-inf) like Gf.Range3d's
        # default ctor, so IsEmpty() is True on the failure/empty path. A unit-cube
        # default (the old behavior) reported a fake 2x2x2 box with IsEmpty()==False,
        # framing a bogus bound when ComputeWorldBound found nothing.
        inf = float("inf")
        if mn is None: mn = (inf, inf, inf)
        if mx is None: mx = (-inf, -inf, -inf)
        if box_mn is None: box_mn = mn
        if box_mx is None: box_mx = mx
        class _AnonRange3d:
            def __init__(s, range_mn, range_mx):
                s._mn = Gf.Vec3d(*range_mn); s._mx = Gf.Vec3d(*range_mx)
            def GetMin(s): return s._mn
            def GetMax(s): return s._mx
            def GetMidpoint(s): return Gf.Vec3d(
                (s._mn[0]+s._mx[0])/2,
                (s._mn[1]+s._mx[1])/2,
                (s._mn[2]+s._mx[2])/2)
            def IsEmpty(s):
                return any(s._mx[i] < s._mn[i] for i in range(3))
            def __repr__(s):
                m, M = s._mn, s._mx
                return f"[({m[0]}, {m[1]}, {m[2]})...({M[0]}, {M[1]}, {M[2]})]"
            __str__ = __repr__
        self.range = _AnonRange3d(mn, mx)
        self.box = _AnonRange3d(box_mn, box_mx)

    def GetRange(self): return self.range
    def GetBox(self): return self.box
    def ComputeAlignedRange(self): return self.range
    def ComputeAlignedBox(self): return self


if hasattr(_native_compat, "UsdGeom") and not hasattr(_native_compat.UsdGeom, "BBoxCache"):
    _native_compat.UsdGeom.BBoxCache = _BBoxCache


def _expand_point_instancer(p):
    """Compute world-space bounds for a UsdGeomPointInstancer by reading
    `positions`, `orientations`, `scales`, `protoIndices`, dereferencing
    `prototypes` rel, computing each prototype's local bound, and unioning
    the per-instance transformed corners. Returns (mn_np3, mx_np3) or
    (None, None) if expansion failed."""
    import numpy as np
    try:
        from pxr import UsdGeom
    except Exception:
        return None, None
    try:
        positions = p.GetAttribute("positions").Get()
        positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3) \
            if positions is not None else None
    except Exception:
        positions = None
    if positions is None or positions.size == 0:
        return None, None
    n = positions.shape[0]
    try:
        protoIndices = p.GetAttribute("protoIndices").Get()
        protoIndices = np.asarray(protoIndices, dtype=np.int32).reshape(-1) \
            if protoIndices is not None else np.zeros(n, dtype=np.int32)
    except Exception:
        protoIndices = np.zeros(n, dtype=np.int32)
    try:
        orientations = p.GetAttribute("orientations").Get()
        if orientations is not None:
            arr = np.asarray(orientations, dtype=np.float32)
            # quath storage = (real, i, j, k); reshape to (n, 4).
            if arr.size == n * 4:
                orientations = arr.reshape(n, 4)
            else:
                orientations = None
        else:
            orientations = None
    except Exception:
        orientations = None
    try:
        scales = p.GetAttribute("scales").Get()
        if scales is not None:
            scales = np.asarray(scales, dtype=np.float64).reshape(-1, 3)
            if scales.shape[0] != n: scales = None
    except Exception:
        scales = None

    # Prototype paths from the rel.
    proto_paths = []
    try:
        rel = p.GetRelationship("prototypes") if hasattr(p, "GetRelationship") else None
        if rel is not None and hasattr(rel, "GetTargets"):
            proto_paths = [str(t) for t in rel.GetTargets()]
    except Exception:
        pass

    stage = p.GetStage() if hasattr(p, "GetStage") else None
    if stage is None or not proto_paths:
        return None, None

    # Compute each prototype's local bound (in prototype-local space)
    # by recursively walking — but using a fresh helper to avoid the
    # PointInstancer special-case (we want raw geometric bounds).
    def _local_subtree_bound(root_prim):
        inf = float("inf")
        L_mn = np.array([inf, inf, inf], dtype=np.float64)
        L_mx = np.array([-inf, -inf, -inf], dtype=np.float64)
        def _w(pp):
            nonlocal L_mn, L_mx
            lmn = lmx = None
            try:
                a = pp.GetAttribute("extent")
                v = a.Get() if a is not None else None
                if v is not None:
                    arr = np.asarray(v, dtype=np.float64).reshape(-1)
                    if arr.size >= 6: lmn, lmx = arr[0:3], arr[3:6]
            except Exception: pass
            if lmn is None:
                try:
                    a = pp.GetAttribute("points")
                    v = a.Get() if a is not None else None
                    if v is not None:
                        arr = np.asarray(v, dtype=np.float64).reshape(-1, 3)
                        if arr.size >= 3:
                            lmn = arr.min(axis=0); lmx = arr.max(axis=0)
                except Exception: pass
            if lmn is not None:
                L_mn = np.minimum(L_mn, lmn)
                L_mx = np.maximum(L_mx, lmx)
            try:
                for c in pp.GetChildren(): _w(c)
            except Exception: pass
        _w(root_prim)
        if not np.all(np.isfinite(L_mn)) or np.any(L_mn > L_mx):
            return None, None
        return L_mn, L_mx

    proto_bounds = []
    for path in proto_paths:
        try:
            pp = stage.GetPrimAtPath(path)
            if pp is None or not pp:
                proto_bounds.append((None, None)); continue
            proto_bounds.append(_local_subtree_bound(pp))
        except Exception:
            proto_bounds.append((None, None))

    inf = float("inf")
    g_mn = np.array([inf, inf, inf], dtype=np.float64)
    g_mx = np.array([-inf, -inf, -inf], dtype=np.float64)

    def _quat_to_mat3(q):
        # quath storage = (real, i, j, k)
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        ww = w * w; xx = x * x; yy = y * y; zz = z * z
        return np.array([
            [ww + xx - yy - zz, 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), ww - xx + yy - zz, 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), ww - xx - yy + zz],
        ], dtype=np.float64)

    for i in range(n):
        idx = int(protoIndices[i]) if i < protoIndices.size else 0
        if idx >= len(proto_bounds): continue
        lmn, lmx = proto_bounds[idx]
        if lmn is None: continue
        # Build the 8 corners of the prototype's local bound.
        c = np.array([
            [lmn[0], lmn[1], lmn[2]], [lmx[0], lmn[1], lmn[2]],
            [lmn[0], lmx[1], lmn[2]], [lmx[0], lmx[1], lmn[2]],
            [lmn[0], lmn[1], lmx[2]], [lmx[0], lmn[1], lmx[2]],
            [lmn[0], lmx[1], lmx[2]], [lmx[0], lmx[1], lmx[2]],
        ], dtype=np.float64)
        # Apply scale.
        if scales is not None:
            c = c * scales[i]
        # Apply orientation (quat → 3x3).
        if orientations is not None:
            R = _quat_to_mat3(orientations[i])
            c = c @ R.T
        # Apply translation.
        c = c + positions[i]
        g_mn = np.minimum(g_mn, c.min(axis=0))
        g_mx = np.maximum(g_mx, c.max(axis=0))

    if not np.all(np.isfinite(g_mn)) or np.any(g_mn > g_mx):
        return None, None

    # Apply the PointInstancer's own world transform.
    try:
        from pxr import UsdGeom
        xf = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(0.0)
        if xf is not None:
            M = np.array(xf, dtype=np.float64).reshape(4, 4)
            corners = np.array([
                [g_mn[0], g_mn[1], g_mn[2]], [g_mx[0], g_mn[1], g_mn[2]],
                [g_mn[0], g_mx[1], g_mn[2]], [g_mx[0], g_mx[1], g_mn[2]],
                [g_mn[0], g_mn[1], g_mx[2]], [g_mx[0], g_mn[1], g_mx[2]],
                [g_mn[0], g_mx[1], g_mx[2]], [g_mx[0], g_mx[1], g_mx[2]],
            ], dtype=np.float64)
            h = np.hstack([corners, np.ones((8, 1))])
            w = h @ M
            corners = w[:, :3] / np.where(w[:, 3:4] != 0, w[:, 3:4], 1.0)
            g_mn = corners.min(axis=0); g_mx = corners.max(axis=0)
    except Exception:
        pass

    return g_mn, g_mx


def _boundable_local_extent(p):
    import numpy as np
    try:
        tn = p.GetTypeName()
    except Exception:
        tn = ""

    try:
        attr = p.GetAttribute("extent") if hasattr(p, "GetAttribute") else None
        if attr is not None and hasattr(attr, "Get"):
            ext = attr.Get()
            if ext is not None:
                a = np.asarray(ext, dtype=np.float64).reshape(-1)
                if a.size >= 6:
                    return a[0:3], a[3:6]
    except Exception:
        pass

    try:
        attr = p.GetAttribute("points") if hasattr(p, "GetAttribute") else None
        if attr is not None and hasattr(attr, "Get"):
            pts = attr.Get()
            if pts is not None:
                a = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
                if a.size >= 3:
                    return a.min(axis=0), a.max(axis=0)
    except Exception:
        pass

    def _float_attr(name, default):
        try:
            attr = p.GetAttribute(name) if hasattr(p, "GetAttribute") else None
            if attr is not None and hasattr(attr, "Get"):
                value = attr.Get()
                if value is not None:
                    return float(value)
        except Exception:
            pass
        return float(default)

    if tn == "Sphere":
        radius = abs(_float_attr("radius", 1.0))
        return np.array([-radius, -radius, -radius], dtype=np.float64), \
            np.array([radius, radius, radius], dtype=np.float64)

    if tn == "Cube":
        half = abs(_float_attr("size", 2.0)) * 0.5
        return np.array([-half, -half, -half], dtype=np.float64), \
            np.array([half, half, half], dtype=np.float64)

    if tn in ("Capsule", "Cone", "Cylinder"):
        radius = abs(_float_attr("radius", 1.0))
        half_height = abs(_float_attr("height", 2.0)) * 0.5
        axis = "Z"
        try:
            attr = p.GetAttribute("axis") if hasattr(p, "GetAttribute") else None
            value = attr.Get() if attr is not None and hasattr(attr, "Get") else None
            if value in ("X", "Y", "Z"):
                axis = value
        except Exception:
            pass
        half = np.array([radius, radius, radius], dtype=np.float64)
        half[{"X": 0, "Y": 1, "Z": 2}[axis]] = half_height
        return -half, half

    return None


# UsdGeom.Imageable extensions — usdview needs ComputeWorldBound,
# ComputePurpose, ComputeVisibility on every Imageable.
def _imageable_compute_world_bound(prim):
    """Walk prim's subtree, collect Mesh/Boundable extents, transform
    each to world, union into a single Range3d. Returns (mn, mx) or None
    if nothing boundable was found."""
    import numpy as np
    try:
        from pxr import Gf, UsdGeom
    except Exception:
        return None
    inf = float("inf")
    g_mn = np.array([inf, inf, inf], dtype=np.float64)
    g_mx = np.array([-inf, -inf, -inf], dtype=np.float64)

    def _walk(p):
        nonlocal g_mn, g_mx
        try:
            tn = p.GetTypeName()
        except Exception:
            tn = ""
        # PointInstancer: expand each instance by its prototype's local
        # bound and per-instance transform, then apply the instancer's
        # own world xform. Skip recursing into children — prototypes
        # are referenced via `prototypes` rel, not consumed as children.
        if tn == "PointInstancer":
            try:
                pi_mn, pi_mx = _expand_point_instancer(p)
                if pi_mn is not None:
                    g_mn = np.minimum(g_mn, pi_mn)
                    g_mx = np.maximum(g_mx, pi_mx)
            except Exception:
                pass
            return
        local_bounds = _boundable_local_extent(p)
        if local_bounds is not None:
            local_mn, local_mx = local_bounds
        else:
            local_mn = local_mx = None
        if local_mn is not None and local_mx is not None:
            xf = None
            for _wrap in ("Xformable", "Imageable"):
                try:
                    cls = getattr(UsdGeom, _wrap, None)
                    if cls is None: continue
                    obj = cls(p)
                    if obj is None: continue
                    fn = getattr(obj, "ComputeLocalToWorldTransform", None)
                    if fn is None: continue
                    xf = fn(0.0)
                    if xf is not None: break
                except Exception:
                    continue
            corners = np.array([
                [local_mn[0], local_mn[1], local_mn[2]],
                [local_mx[0], local_mn[1], local_mn[2]],
                [local_mn[0], local_mx[1], local_mn[2]],
                [local_mx[0], local_mx[1], local_mn[2]],
                [local_mn[0], local_mn[1], local_mx[2]],
                [local_mx[0], local_mn[1], local_mx[2]],
                [local_mn[0], local_mx[1], local_mx[2]],
                [local_mx[0], local_mx[1], local_mx[2]],
            ], dtype=np.float64)
            if xf is not None:
                try:
                    M = np.array(xf, dtype=np.float64).reshape(4, 4)
                    h = np.hstack([corners, np.ones((8, 1))])
                    w = h @ M
                    corners = w[:, :3] / np.where(w[:, 3:4] != 0, w[:, 3:4], 1.0)
                except Exception:
                    pass
            g_mn = np.minimum(g_mn, corners.min(axis=0))
            g_mx = np.maximum(g_mx, corners.max(axis=0))
        # Recurse into children.
        try:
            for c in p.GetChildren():
                _walk(c)
        except Exception:
            pass

    _walk(prim)
    if not np.all(np.isfinite(g_mn)) or not np.all(np.isfinite(g_mx)) \
            or np.any(g_mn > g_mx):
        return None
    return tuple(g_mn.tolist()), tuple(g_mx.tolist())


def _patch_imageable():
    cls = None
    for name in dir(_native_compat):
        obj = getattr(_native_compat, name)
        if isinstance(obj, type) and obj.__name__ == "_Imageable":
            cls = obj; break
    if cls is None:
        return

    # MakeVisible/MakeInvisible — H/Shift+H actions in usdview. Sets the
    # visibility token attribute on this prim. Walking up the ancestry
    # to clear inherited invisibility (Pixar's full behavior) is left
    # for a follow-up — this covers the common case.
    if not hasattr(cls, "MakeVisible"):
        def _mk_visible(self):
            try:
                attr = self._prim.GetAttribute("visibility")
                if attr and attr.IsValid():
                    attr.Set("inherited")
                else:
                    self._prim.CreateAttribute("visibility", "token") \
                        .Set("inherited")
            except Exception: pass
        cls.MakeVisible = _mk_visible

    if not hasattr(cls, "MakeInvisible"):
        def _mk_invisible(self):
            try:
                attr = self._prim.GetAttribute("visibility")
                if attr and attr.IsValid():
                    attr.Set("invisible")
                else:
                    self._prim.CreateAttribute("visibility", "token") \
                        .Set("invisible")
            except Exception: pass
        cls.MakeInvisible = _mk_invisible

    if not hasattr(cls, "ComputeWorldBound"):
        def _cwb(self, time=0.0, purpose1="default", purpose2=None,
                 purpose3=None, purpose4=None):
            try:
                prim = self.GetPrim() if hasattr(self, "GetPrim") else None
                if prim is None and hasattr(self, "_prim"):
                    prim = self._prim
                if prim is not None:
                    bounds = _imageable_compute_world_bound(prim)
                    if bounds is not None:
                        return _Range3dBox(bounds[0], bounds[1])
            except Exception: pass
            return _Range3dBox()
        cls.ComputeWorldBound = _cwb

    if not hasattr(cls, "ComputePurpose"):
        cls.ComputePurpose = lambda self: "default"

    if not hasattr(cls, "ComputeVisibility"):
        cls.ComputeVisibility = lambda self, time=0.0: "inherited"

_patch_imageable()


# UsdGeom.Camera.GetCamera(time) -> Gf.Camera. usdview's "Adjust Free Camera"
# dialog (adjustFreeCamera.py) and camera framing call this; without it,
# opening the dialog with an active camera prim raised
# AttributeError: 'Camera' object has no attribute 'GetCamera'.
def _patch_camera():
    UsdGeomNative = getattr(_native_compat, "UsdGeom", None)
    CamCls = getattr(UsdGeomNative, "Camera", None) if UsdGeomNative else None
    if CamCls is None or hasattr(CamCls, "GetCamera"):
        return

    def _get_camera(self, time=None):
        prim = None
        try:
            prim = self.GetPrim() if hasattr(self, "GetPrim") else None
        except Exception:
            prim = None
        if prim is None:
            prim = getattr(self, "_prim", None)
        cam = _Camera()
        if prim is None:
            return cam

        def _get(name, default):
            try:
                a = prim.GetAttribute(name)
                if not (a and a.IsValid()):
                    return default
                v = None
                if time is not None:
                    try:
                        v = a.Get(time)
                    except Exception:
                        v = None
                if v is None:
                    v = a.Get()
                return default if v is None else v
            except Exception:
                return default

        try:
            proj = str(_get("projection", "perspective"))
            cam.projection = (_Camera.Projection.Orthographic
                              if proj == "orthographic"
                              else _Camera.Projection.Perspective)
            cam.focalLength = float(_get("focalLength", 50.0))
            cam.horizontalAperture = float(_get("horizontalAperture", 20.955))
            cam.verticalAperture = float(_get("verticalAperture", 15.2908))
            cam.horizontalApertureOffset = float(_get("horizontalApertureOffset", 0.0))
            cam.verticalApertureOffset = float(_get("verticalApertureOffset", 0.0))
            cam.fStop = float(_get("fStop", 0.0))
            cam.focusDistance = float(_get("focusDistance", 0.0))
            cr = _get("clippingRange", None)
            if cr is not None:
                cam.clippingRange = _Range1f(float(cr[0]), float(cr[1]))
            else:
                cam.clippingRange = _Range1f(1.0, 1000000.0)
        except Exception:
            pass

        # World transform at time (best-effort; identity on failure).
        try:
            Xf = getattr(UsdGeomNative, "Xformable", None)
            if Xf is not None:
                fn = getattr(Xf(prim), "ComputeLocalToWorldTransform", None)
                if fn is not None:
                    m = fn(time if time is not None else 0.0)
                    if m is not None:
                        cam.transform = _Gf.Matrix4d(m)
        except Exception:
            pass
        return cam

    CamCls.GetCamera = _get_camera
    # Some callers (adjustFreeCamera) build via the bare UsdGeom typed class.


# NOTE: _patch_camera() is invoked AFTER _bulk_patch_usdgeom() below, because
# that bulk pass is what creates the UsdGeom.Camera schema class.


# UsdGeom.Tokens.default_ + a few common purposes
if hasattr(_native_compat, "UsdGeom") and hasattr(_native_compat.UsdGeom, "Tokens"):
    _T = _native_compat.UsdGeom.Tokens
    for _name, _val in (("default_", "default"),
                        ("render", "render"),
                        ("proxy", "proxy"),
                        ("guide", "guide"),
                        ("visibility", "visibility"),
                        ("inherited", "inherited"),
                        ("invisible", "invisible"),
                        ("visible", "visible")):
        if not hasattr(_T, _name):
            try:
                setattr(_T, _name, _val)
            except (AttributeError, TypeError):
                pass


# Matrix4d.SetRotate — The native shim's Gf.Matrix4d is a numpy.ndarray subclass.
# We can monkey-patch the class type. Same for Matrix4d-from-rotation
# helper that freeCamera.py uses.
_M4D = getattr(_native_compat.Gf, "Matrix4d", None)
if _M4D is not None:
    # Find the actual class object (may be wrapped by GfModule's getter)
    import inspect as _inspect
    _M4D_cls = _M4D if _inspect.isclass(_M4D) else None
    if _M4D_cls is None:
        try:
            _M4D_cls = type(_M4D())
        except Exception:
            _M4D_cls = None
    if _M4D_cls is not None and not hasattr(_M4D_cls, "SetRotate"):
        try:
            _M4D_cls.SetRotate = _m4d_set_rotate
        except (AttributeError, TypeError):
            pass


# The native shim's _UsdPrim.GetMetadata requires `key`; usdview occasionally
# probes without one. Make it tolerant.
def _patch_prim_get_metadata():
    cls = None
    for name in dir(_native_compat):
        obj = getattr(_native_compat, name)
        if isinstance(obj, type) and obj.__name__ == "_UsdPrim":
            cls = obj; break
    if cls is None or not hasattr(cls, "GetMetadata"):
        return
    _orig = cls.GetMetadata

    def _listop_metadata(self, *fields):
        mu = getattr(self, "_prim", None)
        listop_cls = getattr(_native_compat, "_ListOp", None)
        if mu is None or listop_cls is None:
            return None
        for field in fields:
            try:
                op = mu.get_listop(field)
            except Exception:
                op = None
            if op:
                value = listop_cls._from_c_handle(op)
                try:
                    value._listop_field = field
                except Exception:
                    pass
                return value
        return None

    def _prim_usda_block(self):
        try:
            mu = getattr(self, "_prim", None)
            stage_h = getattr(mu, "_stage_h", None)
            source = _native_compat._stage_source_usda(stage_h)
            if not source:
                return None
            return (
                _native_compat._find_usda_prim_body_by_name(source, str(self.GetPath()))
                or _native_compat._find_usda_prim_block(source, str(self.GetPath()))
            )
        except Exception:
            return None

    def _prim_composed_usda_block(self):
        try:
            mu = getattr(self, "_prim", None)
            stage_h = getattr(mu, "_stage_h", None)
            source = _native_compat._stage_composed_usda(stage_h)
            if not source:
                return None
            return (
                _native_compat._find_usda_prim_body_by_name(source, str(self.GetPath()))
                or _native_compat._find_usda_prim_block(source, str(self.GetPath()))
            )
        except Exception:
            return None

    def _prim_declaration_metadata(self):
        try:
            mu = getattr(self, "_prim", None)
            stage_h = getattr(mu, "_stage_h", None)
            source = _native_compat._stage_source_usda(stage_h)
            name = self.GetName() if hasattr(self, "GetName") else ""
            if not source or not name:
                return None
            import re as _re
            match = _re.search(
                r"(?ms)^\s*(?:def|over|class)\s+"
                r"(?:[A-Za-z_][\w:]*\s+)?\"" + _re.escape(str(name)) +
                r"\"\s*\((?P<body>.*?)\)\s*\{",
                source,
            )
            return match.group("body") if match else None
        except Exception:
            return None

    def _prim_declaration(self):
        try:
            mu = getattr(self, "_prim", None)
            stage_h = getattr(mu, "_stage_h", None)
            source = _native_compat._stage_source_usda(stage_h)
            decl = _find_usda_prim_declaration(source, str(self.GetPath())) if source else None
            if decl:
                return decl
            source = _native_compat._stage_composed_usda(stage_h)
            return _find_usda_prim_declaration(source, str(self.GetPath())) if source else None
        except Exception:
            return None

    def _prim_composed_declaration_metadata(self):
        try:
            mu = getattr(self, "_prim", None)
            stage_h = getattr(mu, "_stage_h", None)
            source = _native_compat._stage_composed_usda(stage_h)
            name = self.GetName() if hasattr(self, "GetName") else ""
            if not source or not name:
                return None
            import re as _re
            match = _re.search(
                r"(?ms)^\s*(?:def|over|class)\s+"
                r"(?:[A-Za-z_][\w:]*\s+)?\"" + _re.escape(str(name)) +
                r"\"\s*\((?P<body>.*?)\)\s*\{",
                source,
            )
            return match.group("body") if match else None
        except Exception:
            return None

    def _extract_braced_field(text, field):
        try:
            import re as _re
            match = _re.search(r"\b" + _re.escape(field) + r"\s*=\s*\{", text)
            if not match:
                return None
            start = text.find("{", match.start())
            if start < 0:
                return None
            depth = 0
            for idx in range(start, len(text)):
                ch = text[idx]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start + 1:idx]
            return None
        except Exception:
            return None

    def _parse_custom_data(self):
        try:
            block = (
                _prim_composed_declaration_metadata(self)
                or _prim_declaration_metadata(self)
                or _prim_composed_usda_block(self)
                or _prim_usda_block(self)
            )
            if not block:
                return None
            body = _extract_braced_field(block, "customData")
            if body is None:
                return None
            parsed = _native_compat._parse_usda_custom_data_dict(body)
            return parsed or None
        except Exception:
            return None

    def _parse_asset_info(self):
        try:
            block = _prim_declaration_metadata(self) or _prim_usda_block(self)
            if not block:
                return None
            body = _extract_braced_field(block, "assetInfo")
            if body is None:
                return None
            data = _parse_usda_asset_info_dict(body)
            return data or None
        except Exception:
            return None

    def _wrap(self, key=None):
        if key is None:
            return None
        if key == "active" and hasattr(self, "IsActive"):
            active = bool(self.IsActive())
            if not active:
                return False
            block = _prim_declaration_metadata(self)
            if block:
                import re as _re
                if _re.search(r"\bactive\s*=\s*(true|1)", block):
                    return True
                if _re.search(r"\bactive\s*=\s*(false|0)", block):
                    return False
            return None
        if key == "specifier":
            decl = _prim_declaration(self)
            if decl:
                return {"def": 0, "class": 1, "over": 2}.get(
                    decl.get("specifier"), _orig(self, key))
            return _orig(self, key)
        if key == "apiSchemas":
            value = _orig(self, key)
            if value is not None:
                try:
                    value._listop_field = key
                except Exception:
                    pass
            return value
        if key == "customData":
            value = _orig(self, key)
            data = value or _parse_custom_data(self)
            try:
                if data and self.GetTypeName() == "Xform":
                    data = dict(data)
                    data.setdefault(
                        "userDocBrief",
                        "Concrete prim schema for a transform, which implements Xformable.")
            except Exception:
                pass
            return data
        if key == "assetInfo":
            value = _orig(self, key)
            return value or _parse_asset_info(self)
        if key == "references":
            return _listop_metadata(self, "references")
        if key in ("payload", "payloads"):
            return _listop_metadata(self, "payload", "payloads")
        if key == "specializes":
            return _listop_metadata(self, "specializes")
        if key in ("inherits", "inheritPaths"):
            return _listop_metadata(self, "inheritPaths", "inherits")
        return _orig(self, key)
    cls.GetMetadata = _wrap
    cls.GetCustomData = lambda self: self.GetMetadata("customData") or {}

_patch_prim_get_metadata()


# ---------------------------------------------------------------- placeholder namespaces
def _empty_module(name):
    m = _types.ModuleType(name)
    return m


def _make_namespace(name, attrs=None):
    """Build a minimal namespace placeholder. Unknown attributes are
    auto-stubbed to no-op callables so usdview's `Trace.Collector().Begin()`
    style chains don't AttributeError at us."""
    m = _empty_module(f"pxr.{name}")
    if attrs:
        for k, v in attrs.items():
            setattr(m, k, v)

    def _stub_factory(_name, _modname=name):
        if _name.startswith("_"):
            raise AttributeError(_name)
        # Return a class whose instances AND the class itself respond to
        # any further .method() chain. Both Plug.Registry.GetAllDerivedTypes
        # (class-level) and Trace.Collector().BeginEvent (instance-level)
        # need to work without AttributeError.
        def _smart_default(method_name):
            if method_name.startswith("Is") or method_name.startswith("Has"):
                return False
            if method_name.startswith("Get"):
                if any(s in method_name for s in
                       ("List", "Plugins", "Aovs", "Names", "All", "Children", "Targets")):
                    return []
                if any(s in method_name for s in ("Name", "Id", "Path", "DisplayName")):
                    return ""
            return None

        class _PermissiveMeta(type):
            def __getattr__(cls_, k):
                if k.startswith("_"): raise AttributeError(k)
                def _stub(*a, **kw): return _smart_default(k)
                return _stub

        class _Permissive(metaclass=_PermissiveMeta):
            def __init__(self_, *a, **kw): pass
            def __getattr__(self_, k):
                if k.startswith("_"): raise AttributeError(k)
                def _stub(*a, **kw): return _smart_default(k)
                return _stub
            def __call__(self_, *a, **kw): return self_
            def __iter__(self_): return iter([])
            def __bool__(self_): return False

        _Permissive.__name__ = _name
        setattr(m, _name, _Permissive)
        return _Permissive

    m.__getattr__ = _stub_factory
    return m


_KIND_MOD = _make_namespace("Kind", {
    "Tokens": type("KindTokens", (), {
        "model": "model", "component": "component",
        "subcomponent": "subcomponent", "group": "group", "assembly": "assembly",
    }),
    "Registry": type("KindRegistry", (), {
        "IsA": staticmethod(lambda kind, base: kind == base),
        "GetBaseKind": staticmethod(lambda k: ""),
    })(),
})

_AR_MOD = _make_namespace("Ar", {
    "GetResolver": lambda: None,
    "ResolverContextBinder": type("RCB", (), {
        "__init__": lambda self, *a, **kw: None,
        "__enter__": lambda self: self,
        "__exit__":  lambda self, *a: False,
    }),
})

_TRACE_MOD = _make_namespace("Trace", {
    "TraceFunction": lambda f: f,  # no-op decorator
    "TraceScope": type("TraceScope", (), {
        "__init__": lambda self, *a, **kw: None,
        "__enter__": lambda self: self,
        "__exit__":  lambda self, *a: False,
    }),
})

_TS_MOD = _make_namespace("Ts", {
    "InterpCurve": type("InterpCurve", (), {}),
    "Knot": type("Knot", (), {}),
})

_USD_SEMANTICS_MOD = _make_namespace("UsdSemantics", {})
_USD_LUX_MOD = _make_namespace("UsdLux", {})
_USD_RENDER_MOD = _make_namespace("UsdRender", {})
_SDR_MOD = _make_namespace("Sdr", {"Registry": type("SdrRegistry", (), {})})
_NDR_MOD = _make_namespace("Ndr", {})

_CAMERA_UTIL_MOD = _make_namespace("CameraUtil", {
    "Fit": "Fit", "MatchHorizontally": "MatchHorizontally",
    "MatchVertically": "MatchVertically", "Crop": "Crop", "DontConform": "DontConform",
    "ConformWindowPolicy": staticmethod(lambda window, policy: window),
})

def _normalize_complexity_name(name):
    return "".join(str(name).replace("_", " ").replace("-", " ").split()).lower()


class _RefinementComplexity:
    """Mimics Pixar's RefinementComplexity (instances of which expose
    .name and float-like .value)."""
    def __init__(self, name, displayNameOrValue, value=None):
        if value is None:
            self.id = _normalize_complexity_name(name)
            self.name = str(name)
            value = displayNameOrValue
        else:
            self.id = str(name)
            self.name = str(displayNameOrValue)
        self.value = float(value)
    def __float__(self): return self.value
    def __eq__(self, other):
        if isinstance(other, _RefinementComplexity):
            return _normalize_complexity_name(self.name) == _normalize_complexity_name(other.name)
        if isinstance(other, str):
            return _normalize_complexity_name(self.name) == _normalize_complexity_name(other)
        return NotImplemented
    def __hash__(self): return hash(_normalize_complexity_name(self.name))
    def __repr__(self): return f"RefinementComplexity({self.name!r})"


class _RefinementComplexities:
    _RefinementComplexity = _RefinementComplexity

    LOW = _RefinementComplexity("Low", 1.0)
    MEDIUM = _RefinementComplexity("Medium", 1.1)
    HIGH = _RefinementComplexity("High", 1.2)
    VERY_HIGH = _RefinementComplexity("Very High", 1.3)

    @classmethod
    def ordered(cls):
        return (cls.LOW, cls.MEDIUM, cls.HIGH, cls.VERY_HIGH)

    @classmethod
    def fromName(cls, name):
        if isinstance(name, _RefinementComplexity): return name
        key = _normalize_complexity_name(name)
        return {"low": cls.LOW, "medium": cls.MEDIUM,
                "high": cls.HIGH, "veryhigh": cls.VERY_HIGH}.get(key, cls.MEDIUM)

    @classmethod
    def names(cls):
        return tuple(c.name for c in cls.ordered())

    @classmethod
    def next(cls, complexity):
        ordered = cls.ordered()
        try:
            i = ordered.index(cls.fromName(complexity))
        except ValueError:
            return cls.MEDIUM
        return ordered[min(i + 1, len(ordered) - 1)]

    @classmethod
    def prev(cls, complexity):
        ordered = cls.ordered()
        try:
            i = ordered.index(cls.fromName(complexity))
        except ValueError:
            return cls.MEDIUM
        return ordered[max(i - 1, 0)]


_USDAPPUTILS_COMPLEXITY_MOD = _make_namespace("UsdAppUtils.complexityArgs", {
    "RefinementComplexities": _RefinementComplexities,
})


def _arg_exists(parser, *flags):
    return any(
        flag in action.option_strings
        for action in getattr(parser, "_actions", ())
        for flag in flags
    )


def _camera_args_add_cmdline_args(parser, altHelpText=None):
    if _arg_exists(parser, "--camera"):
        return
    parser.add_argument(
        "--camera",
        action="store",
        dest="camera",
        default=None,
        type=_native_compat.Sdf.Path,
        help=altHelpText or "Camera path or camera prim name to view through",
    )


def _renderer_args_add_cmdline_args(parser, altHelpText=None):
    if _arg_exists(parser, "--renderer"):
        return
    parser.add_argument(
        "--renderer",
        action="store",
        dest="rendererPlugin",
        default=None,
        help=altHelpText or "Renderer plugin to use",
    )


def _complexity_args_add_cmdline_args(parser, altHelpText=None):
    if _arg_exists(parser, "--complexity"):
        return
    parser.add_argument(
        "--complexity",
        action="store",
        dest="complexity",
        default=_RefinementComplexities.LOW,
        type=_RefinementComplexities.fromName,
        choices=_RefinementComplexities.ordered(),
        help=altHelpText or "Mesh refinement complexity",
    )


_USDAPPUTILS_COMPLEXITY_MOD.AddCmdlineArgs = _complexity_args_add_cmdline_args
_USDAPPUTILS_CAMERA_MOD = _make_namespace("UsdAppUtils.cameraArgs", {
    "AddCmdlineArgs": _camera_args_add_cmdline_args,
})
_USDAPPUTILS_RENDERER_MOD = _make_namespace("UsdAppUtils.rendererArgs", {
    "AddCmdlineArgs": _renderer_args_add_cmdline_args,
})
_USD_APP_UTILS_MOD = _make_namespace("UsdAppUtils", {
    "complexityArgs": _USDAPPUTILS_COMPLEXITY_MOD,
    "cameraArgs": _USDAPPUTILS_CAMERA_MOD,
    "rendererArgs": _USDAPPUTILS_RENDERER_MOD,
    "framesArgs": _make_namespace("UsdAppUtils.framesArgs", {
        "ConvertFramesToTimeCodes": staticmethod(lambda f: list(f)),
    }),
})
_sys.modules["pxr.UsdAppUtils.complexityArgs"] = _USDAPPUTILS_COMPLEXITY_MOD
_sys.modules["pxr.UsdAppUtils.cameraArgs"] = _USDAPPUTILS_CAMERA_MOD
_sys.modules["pxr.UsdAppUtils.rendererArgs"] = _USDAPPUTILS_RENDERER_MOD
_USD_APP_UTILS_MOD.__path__ = []

class _ConstantsGroupMeta(type):
    """Make ConstantsGroup subclasses iterable + support `value in Group`.
    Pixar's real ConstantsGroup is similar."""
    def __iter__(cls):
        return iter(cls._values())

    def __contains__(cls, value):
        return value in cls._values()

    def _values(cls):
        return tuple(v for k, v in vars(cls).items()
                     if not k.startswith("_") and not callable(v))

    def __len__(cls):
        return len(cls._values())


class _ConstantsGroup(metaclass=_ConstantsGroupMeta):
    """Pixar-style: a class whose attributes are fixed enum-like constants."""
    pass


_USD_UTILS_CONSTANTS_GROUP_MOD = _make_namespace(
    "UsdUtils.constantsGroup", {"ConstantsGroup": _ConstantsGroup})

_USD_UTILS_MOD = _make_namespace("UsdUtils", {
    "constantsGroup": _USD_UTILS_CONSTANTS_GROUP_MOD,
    "CopyLayerMetadata": staticmethod(lambda *a, **kw: True),
    "GetPrimaryCameraName": staticmethod(lambda: ""),
})

# Also register the submodule explicitly so `from pxr.UsdUtils.constantsGroup import ConstantsGroup` works.
_sys.modules["pxr.UsdUtils.constantsGroup"] = _USD_UTILS_CONSTANTS_GROUP_MOD
_USD_UTILS_MOD.__path__ = []  # mark as package


# ---------------------------------------------------------------- pxr package
_pxr = _types.ModuleType("pxr")
_pxr.__path__ = []  # mark as package

_NAMESPACE_MAP = {
    "Usd":          _native_compat.Usd,
    "Sdf":          _native_compat.Sdf,
    "UsdGeom":      _native_compat.UsdGeom,
    "UsdShade":     _native_compat.UsdShade,
    "Gf":           _native_compat.Gf,
    "Vt":           _native_compat.Vt,
    "UsdPhysics":   getattr(_native_compat, "UsdPhysics", _make_namespace("UsdPhysics")),
    "Tf":           _Tf,
    "Kind":         _KIND_MOD,
    "Ar":           _AR_MOD,
    "Trace":        _TRACE_MOD,
    "Ts":           _TS_MOD,
    "UsdSemantics": _USD_SEMANTICS_MOD,
    "UsdLux":       _USD_LUX_MOD,
    "UsdRender":    _USD_RENDER_MOD,
    "Sdr":          _SDR_MOD,
    "Ndr":          _NDR_MOD,
    "CameraUtil":   _CAMERA_UTIL_MOD,
    "UsdAppUtils":  _USD_APP_UTILS_MOD,
    "UsdUtils":     _USD_UTILS_MOD,
}

# Hydra-ish modules: register as empty placeholders so `from pxr import …`
# at usdview's top-level doesn't ImportError. Their consumers will be
# stripped by Phase 3.
for _ns in ("Plug", "Glf", "UsdImagingGL"):
    _NAMESPACE_MAP[_ns] = _make_namespace(_ns)

for _name, _ns in _NAMESPACE_MAP.items():
    if getattr(_ns, "__spec__", None) is None:
        _ns.__spec__ = _ilu.spec_from_loader(
            f"pxr.{_name}",
            loader=None,
            is_package=hasattr(_ns, "__path__"),
        )
    _sys.modules[f"pxr.{_name}"] = _ns
    setattr(_pxr, _name, _ns)

_pxr.__spec__ = _ilu.spec_from_loader("pxr", loader=None, is_package=True)
_sys.modules["pxr"] = _pxr


# ---------------------------------------------------------------- meta_path
# Some usdview source does `from pxr.UsdUtils.toolPaths import FindUsdBinary`
# (and similar). We don't ship those submodules; install a meta_path finder
# that returns a permissive placeholder for any unregistered pxr.X.Y… so
# imports succeed and missing functions become no-op callables on access.
class _PxrSubmoduleFinder:
    _nanousd_pxr_submodule_finder = True

    @staticmethod
    def _usdviewq_target(fullname):
        if fullname == "pxr.Usdviewq":
            return "nanousdview.usdviewq"
        prefix = "pxr.Usdviewq."
        if fullname.startswith(prefix):
            return "nanousdview.usdviewq." + fullname[len(prefix):]
        return None

    @classmethod
    def _find_usdviewq_spec(cls, fullname):
        target_name = cls._usdviewq_target(fullname)
        if target_name is None:
            return None
        try:
            target_spec = _ilu.find_spec(target_name)
        except (ImportError, AttributeError, ValueError):
            return None
        if target_spec is None:
            return None

        spec = _ilu.spec_from_loader(
            fullname,
            cls,
            is_package=target_spec.submodule_search_locations is not None,
        )
        spec.loader_state = {"nanousd_alias": target_name}
        return spec

    @classmethod
    def find_spec(cls, fullname, path, target=None):
        if not fullname.startswith("pxr."):
            return None
        if fullname in _sys.modules:
            return None
        usdviewq_spec = cls._find_usdviewq_spec(fullname)
        if usdviewq_spec is not None:
            return usdviewq_spec
        # Already-known submodules
        return _ilu.spec_from_loader(fullname, cls)

    @classmethod
    def create_module(cls, spec):
        alias_state = getattr(spec, "loader_state", None)
        alias_target = alias_state.get("nanousd_alias") if isinstance(alias_state, dict) else None
        if alias_target:
            module = _importlib.import_module(alias_target)
            _sys.modules[spec.name] = module
            if spec.name == "pxr.Usdviewq":
                setattr(_pxr, "Usdviewq", module)
            else:
                parent_name, _, child_name = spec.name.rpartition(".")
                parent = _sys.modules.get(parent_name)
                if parent is not None:
                    setattr(parent, child_name, module)
            return module

        m = _types.ModuleType(spec.name)
        m.__path__ = []  # mark as package so further sub-imports work

        def _missing(name, _modname=spec.name):
            if name.startswith("_"):
                raise AttributeError(name)
            stub = lambda *a, **kw: None
            stub.__name__ = name
            stub.__module__ = _modname
            setattr(m, name, stub)
            return stub

        m.__getattr__ = _missing
        return m

    @classmethod
    def exec_module(cls, module):
        return None


if not any(getattr(_finder, "_nanousd_pxr_submodule_finder", False) for _finder in _sys.meta_path):
    _sys.meta_path.insert(0, _PxrSubmoduleFinder)


# Re-export commonly imported names at this package level so
# `from nanousd.pxr_compat import Usd, Sdf, ...` works inside our own code.
Usd = _native_compat.Usd
Sdf = _native_compat.Sdf
UsdGeom = _native_compat.UsdGeom
UsdShade = _native_compat.UsdShade
UsdPhysics = getattr(_native_compat, "UsdPhysics", _make_namespace("UsdPhysics"))
Gf = _native_compat.Gf
Vt = _native_compat.Vt
Tf = _Tf
Kind = _KIND_MOD
Ar = _AR_MOD
Trace = _TRACE_MOD
Ts = _TS_MOD
UsdSemantics = _USD_SEMANTICS_MOD
UsdLux = _USD_LUX_MOD
UsdRender = _USD_RENDER_MOD
CameraUtil = _CAMERA_UTIL_MOD
UsdAppUtils = _USD_APP_UTILS_MOD
UsdUtils = _USD_UTILS_MOD
Sdr = _SDR_MOD
Ndr = _NDR_MOD
Plug = _NAMESPACE_MAP["Plug"]
Glf = _NAMESPACE_MAP["Glf"]
UsdImagingGL = _NAMESPACE_MAP["UsdImagingGL"]


def _patch_usdgeom_tokens():
    tokens = getattr(UsdGeom, "Tokens", None)
    if tokens is None:
        return
    for name, value in (
        ("default_", "default"),
        ("guide", "guide"),
        ("inherited", "inherited"),
        ("invisible", "invisible"),
        ("proxy", "proxy"),
        ("render", "render"),
        ("visibility", "visibility"),
        ("visible", "visible"),
    ):
        if not hasattr(tokens, name):
            setattr(tokens, name, value)


_patch_usdgeom_tokens()

__all__ = list(_NAMESPACE_MAP.keys())


# USDZ archive support. nanousd's parser doesn't read zips; we extract
# to a per-process temp dir keyed by (path, mtime) so the same .usdz
# only unpacks once per session, and hand the inner default-layer
# .usd/.usdc/.usda to the underlying open. References resolve relative
# to the extract dir, picking up other layers and texture assets.
_USDZ_CACHE: dict = {}

def _unpack_usdz(usdz_path: str) -> str:
    import os, tempfile, zipfile
    try:
        mtime = os.path.getmtime(usdz_path)
    except OSError as e:
        raise RuntimeError(f"cannot stat {usdz_path}: {e}")
    cache_key = (os.path.abspath(usdz_path), mtime)
    if cache_key in _USDZ_CACHE:
        cached = _USDZ_CACHE[cache_key]
        if os.path.exists(cached):
            return cached
    extract_dir = tempfile.mkdtemp(prefix="nanousd_usdz_")
    with zipfile.ZipFile(usdz_path, "r") as z:
        z.extractall(extract_dir)
        # The default layer is the first entry per the USDZ spec
        # (§14.4.4) — no explicit manifest. Fall back to the first
        # entry whose name ends in a known USD suffix.
        names = z.namelist()
    if not names:
        raise RuntimeError("USDZ archive is empty")
    default_layer = names[0]
    if not default_layer.lower().endswith(
            (".usd", ".usdc", ".usda", ".usdz")):
        for n in names:
            if n.lower().endswith((".usd", ".usdc", ".usda")):
                default_layer = n; break
    inner_path = os.path.join(extract_dir, default_layer)
    _USDZ_CACHE[cache_key] = inner_path
    return inner_path


# Pixar's Usd.Stage.Open(rootLayer_or_path, sessionLayer=None,
# pathResolverContext=None, load=Usd.Stage.LoadAll). The native shim takes
# only (filepath, load_policy=None). Bridge.
def _patch_stage_open():
    if not hasattr(_native_compat, 'Usd') or not hasattr(_native_compat.Usd, 'Stage'):
        return
    _orig_open = _native_compat.Usd.Stage.Open

    @staticmethod
    def _open(rootIdentifier, sessionLayer=None, pathResolverContext=None,
              load=None):
        layer_proxy_cls = getattr(_native_compat, "_LayerProxy", None)
        if layer_proxy_cls is not None and isinstance(rootIdentifier, layer_proxy_cls):
            return _orig_open(rootIdentifier)
        if hasattr(rootIdentifier, 'identifier'):
            arg = rootIdentifier.identifier
        elif hasattr(rootIdentifier, 'realPath'):
            arg = rootIdentifier.realPath
        else:
            arg = str(rootIdentifier)
        # USDZ shim: nanousd's parser doesn't read zip archives. Detect
        # the .usdz suffix, extract to a per-process temp dir (cached so
        # repeat opens of the same usdz reuse it), and hand the inner
        # default-layer to nanousd. References within the archive
        # resolve via the unpacked relative-path layout.
        if isinstance(arg, str) and arg.lower().endswith(".usdz"):
            try:
                arg = _unpack_usdz(arg)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to unpack USDZ archive '{arg}': {e}") from e
        return _orig_open(arg)

    _native_compat.Usd.Stage.Open = _open

    class _LoadAll: pass
    class _LoadNone: pass
    if not hasattr(_native_compat.Usd.Stage, 'LoadAll'):
        _native_compat.Usd.Stage.LoadAll = _LoadAll
    if not hasattr(_native_compat.Usd.Stage, 'LoadNone'):
        _native_compat.Usd.Stage.LoadNone = _LoadNone

_patch_stage_open()



# Pixar's Stage.SetEditTarget — read-only stub.
def _patch_stage_set_edit_target():
    cls = None
    for name in dir(_native_compat):
        obj = getattr(_native_compat, name)
        if isinstance(obj, type) and obj.__name__ == '_UsdStage':
            cls = obj
            break
    if cls is None: return
    if not hasattr(cls, 'SetEditTarget'):
        cls.SetEditTarget = lambda self, target: None  # silently no-op (read-only)
    if not hasattr(cls, 'GetMutedLayers'):
        cls.GetMutedLayers = lambda self: []
    if not hasattr(cls, 'IsLayerMuted'):
        cls.IsLayerMuted = lambda self, layer_id: False
    if not hasattr(cls, 'SetLayerMuted'):
        cls.SetLayerMuted = lambda self, layer_id, muted: None
    if not hasattr(cls, 'Save'):
        cls.Save = lambda self, *a, **kw: None
    if not hasattr(cls, 'Reload'):
        cls.Reload = lambda self, *a, **kw: None
    if not hasattr(cls, 'Export') or getattr(cls.Export, '__name__', '') == '<lambda>':
        cls.Export = lambda self, path, *a, **kw: self.GetRootLayer().Export(path)
    if not hasattr(cls, 'GetMetadata'):
        cls.GetMetadata = lambda self, key=None: None
    if not hasattr(cls, 'GetMetadataByDictKey'):
        cls.GetMetadataByDictKey = lambda self, *a, **kw: None
    if not hasattr(cls, 'GetInterpolationType'):
        cls.GetInterpolationType = lambda self: 0
    if not hasattr(cls, 'SetInterpolationType'):
        cls.SetInterpolationType = lambda self, v: None
    if not hasattr(cls, 'OverridesEnabled'):
        cls.OverridesEnabled = lambda self: False
    if not hasattr(cls, 'GetFramesPerSecond'):
        cls.GetFramesPerSecond = lambda self: 24.0

_patch_stage_set_edit_target()



# Pixar's Usd.Notice — usdview's data model registers ObjectsChanged.
# The native shim's _UsdModule has no Notice; bridge via Tf.Notice (same backend).
class _UsdNoticeProxy:
    @staticmethod
    def _passthru(): return _Notice
    ObjectsChanged = _ObjectsChangedNotice
    StageContentsChanged = _StageContentsChanged

if hasattr(_native_compat, 'Usd') and not hasattr(_native_compat.Usd, 'Notice'):
    _native_compat.Usd.Notice = _UsdNoticeProxy



# Sdf.Path extensions — usdview uses many fields/methods the native shim doesn't.
def _patch_sdf_path():
    SdfMod = _native_compat.Sdf
    if not hasattr(SdfMod, 'Path'):
        return
    P = SdfMod.Path

    if not hasattr(P, 'pathElementCount'):
        @property
        def _path_element_count(self):
            s = self._str
            if not s or s == '/':
                return 0
            return len([x for x in s.split('/') if x])
        P.pathElementCount = _path_element_count

    if not hasattr(P, 'name'):
        @property
        def _name(self):
            return self._str.rstrip('/').rsplit('/', 1)[-1] if self._str else ''
        P.name = _name

    if not hasattr(P, 'GetName'):
        P.GetName = lambda self: self.name
    if not hasattr(P, 'GetPrimPath'):
        P.GetPrimPath = lambda self: self
    if not hasattr(P, 'IsPrimPath'):
        P.IsPrimPath = lambda self: bool(self._str) \
            and self._str != '/' and '.' not in self._str \
            and '[' not in self._str and ']' not in self._str
    if not hasattr(P, 'IsPropertyPath'):
        P.IsPropertyPath = lambda self: '.' in self._str
    if not hasattr(P, 'IsRootPath'):
        P.IsRootPath = lambda self: self._str == '/'
    if not hasattr(P, 'IsAbsoluteRootOrPrimPath'):
        P.IsAbsoluteRootOrPrimPath = lambda self: \
            self._str == '/' or (self._str.startswith('/') and '.' not in self._str)
    if not hasattr(P, 'IsAbsoluteRootPath'):
        P.IsAbsoluteRootPath = lambda self: self._str == '/'
    if not hasattr(P, 'GetAbsoluteRootOrPrimPath'):
        # Pixar: returns "/" or the absolute prim path (strips property
        # suffix from /A/B.attr → /A/B).
        def _abs_root_or_prim(self):
            s = str(self)
            if '.' in s: s = s.rsplit('.', 1)[0]
            if not s.startswith('/'): s = '/' + s
            return P(s)
        P.GetAbsoluteRootOrPrimPath = _abs_root_or_prim

    # static helpers usdview uses for selection bookkeeping
    if not hasattr(P, 'RemoveDescendentPaths'):
        def _remove_descendent_paths(paths):
            # Return the topmost ancestors only — drop any path whose
            # parent or above is also in the list.
            strs = sorted({str(p) for p in paths})
            out = []
            for s in strs:
                if not any(s != t and (s == t or s.startswith(t + '/'))
                           for t in strs):
                    out.append(P(s))
            return out
        P.RemoveDescendentPaths = staticmethod(_remove_descendent_paths)
    if not hasattr(P, 'RemoveAncestorPaths'):
        def _remove_ancestor_paths(paths):
            strs = sorted({str(p) for p in paths}, key=lambda x: -len(x))
            out = []
            for s in strs:
                if not any(s != t and t.startswith(s + '/') for t in strs):
                    out.append(P(s))
            return out
        P.RemoveAncestorPaths = staticmethod(_remove_ancestor_paths)
    if not hasattr(P, 'FindLongestPrefix'):
        P.FindLongestPrefix = staticmethod(
            lambda paths, prefix: prefix if any(str(p).startswith(str(prefix))
                                                 for p in paths) else None)
    if not hasattr(P, 'IsEmpty'):
        P.IsEmpty = lambda self: not self._str
    if not hasattr(P, 'AppendChild'):
        P.AppendChild = lambda self, name: P((self._str.rstrip('/') if self._str != '/' else '') + '/' + name)
    if not hasattr(P, 'AppendProperty'):
        P.AppendProperty = lambda self, name: P(self._str + '.' + name)
    if not hasattr(P, 'AppendVariantSelection'):
        P.AppendVariantSelection = lambda self, vs, vn: P(self._str + '{' + vs + '=' + vn + '}')
    if not hasattr(P, 'GetCommonPrefix'):
        P.GetCommonPrefix = lambda self, o: P(_common_prefix(self._str, str(o)))
    if not hasattr(P, 'absoluteRootPath'):
        P.absoluteRootPath = P('/')
    if not hasattr(P, 'emptyPath'):
        P.emptyPath = P('')
    if not hasattr(P, 'StripAllVariantSelections'):
        P.StripAllVariantSelections = lambda self: P(re.sub(r'{[^}]*}', '', self._str)) if False else P(self._str)
    if not hasattr(P, 'ReplacePrefix'):
        P.ReplacePrefix = lambda self, old, new: P(self._str.replace(str(old), str(new), 1))
    if not hasattr(P, 'GetText'):
        P.GetText = lambda self: self._str
    if not hasattr(P, 'JoinIdentifier'):
        P.JoinIdentifier = staticmethod(lambda *a: '.'.join(str(x) for x in a if x))


def _common_prefix(a, b):
    a = str(a)
    b = str(b)
    if not a or not b:
        return ""
    if a == b:
        return a

    def _prim_parts(text):
        slash = text.rfind("/")
        dot = text.find(".", slash + 1)
        if dot != -1:
            text = text[:dot]
        absolute = text.startswith("/")
        return absolute, [part for part in text.split("/") if part]

    a_absolute, a_parts = _prim_parts(a)
    b_absolute, b_parts = _prim_parts(b)
    if a_absolute != b_absolute:
        return ""

    common = []
    for left, right in zip(a_parts, b_parts):
        if left != right:
            break
        common.append(left)

    if a_absolute:
        return "/" + "/".join(common) if common else "/"
    return "/".join(common)


_patch_sdf_path()



# Bulk-stub the remaining Stage methods that usdview expects.
def _bulk_patch_stage():
    cls = None
    for name in dir(_native_compat):
        obj = getattr(_native_compat, name)
        if isinstance(obj, type) and obj.__name__ == '_UsdStage':
            cls = obj
            break
    if cls is None: return
    bulk = {
        'HasAuthoredTimeCodeRange': lambda self: self.GetStartTimeCode() != 0 or self.GetEndTimeCode() != 0,
        'GetAttributeAtPath': lambda self, path: None,
        'GetPathResolverContext': lambda self: None,
        'GetPrototypes': lambda self: [],
        'Load': lambda self, *a, **kw: None,
        'MuteLayer': lambda self, layer_id: None,
        'UnmuteLayer': lambda self, layer_id: None,
        'OverridesEnabled': lambda self: False,
        'GetColorConfiguration': lambda self: '',
        'GetColorManagementSystem': lambda self: '',
        'SetColorConfiguration': lambda self, *a: None,
        'SetColorManagementSystem': lambda self, *a: None,
    }
    for nm, fn in bulk.items():
        if not hasattr(cls, nm):
            setattr(cls, nm, fn)


_bulk_patch_stage()



# Pixar's Usd.TimeCode also has .GetValue() / IsDefault / IsEarliestTime.
def _patch_timecode():
    for nm in dir(_native_compat):
        obj = getattr(_native_compat, nm)
        if isinstance(obj, type) and obj.__name__ == '_TimeCode':
            cls = obj
            if not hasattr(cls, 'GetValue'):
                cls.GetValue = lambda self: self.time
            if not hasattr(cls, 'IsDefault'):
                cls.IsDefault = lambda self: self.time != self.time  # NaN means default in Pixar
            if not hasattr(cls, 'IsEarliestTime'):
                cls.IsEarliestTime = lambda self: self.time < -1e30
            if not hasattr(cls, 'Default'):
                cls.Default = staticmethod(lambda: cls(float('nan')))
            if not hasattr(cls, 'EarliestTime'):
                cls.EarliestTime = staticmethod(lambda: cls(-1e308))
            return
_patch_timecode()



# usdview accesses UsdGeom.Camera, .Mesh, .Xform, .Cube, etc. as schema
# class types. Stub each as a wrapper that constructs from a prim.
def _bulk_patch_usdgeom():
    UG = _native_compat.UsdGeom
    def _make_schema(typename):
        class _Schema:
            def __init__(self, prim=None):
                self._prim = prim
            def GetPrim(self):
                return self._prim
            def __bool__(self):
                return self._prim is not None
            def IsValid(self):
                return self._prim is not None and getattr(self._prim, 'IsValid', lambda: True)()
            @staticmethod
            def Get(stage, path):
                return _Schema(stage.GetPrimAtPath(path) if hasattr(stage, 'GetPrimAtPath') else None)
            @staticmethod
            def Define(stage, path):
                return _Schema()
        _Schema.__name__ = typename
        return _Schema

    for tn in ('Camera', 'Mesh', 'Xform', 'Cube', 'Sphere', 'Cylinder', 'Cone',
               'Capsule', 'Plane', 'Points', 'BasisCurves', 'NurbsCurves',
               'PointInstancer', 'GeomSubset', 'Scope', 'Xformable',
               'PointBased', 'Curves', 'Boundable', 'Gprim', 'Subset'):
        if not hasattr(UG, tn):
            setattr(UG, tn, _make_schema(tn))


_bulk_patch_usdgeom()
_patch_camera()  # adds UsdGeom.Camera.GetCamera now that the schema class exists



# Bulk-add Usd module-level attributes that usdview uses for prim predicates,
# kind queries, change notice flags, etc.
def _bulk_patch_usd():
    U = _native_compat.Usd
    # Boolean prim predicates — usdview ANDs them together
    class _PrimPredicate:
        def __init__(self, fn=None): self.fn = fn or (lambda p: True)
        def __and__(self, other): return _PrimPredicate(lambda p: self.fn(p) and other.fn(p))
        def __or__(self, other): return _PrimPredicate(lambda p: self.fn(p) or other.fn(p))
        def __invert__(self): return _PrimPredicate(lambda p: not self.fn(p))
        def __call__(self, prim): return self.fn(prim)
        def __bool__(self): return True
    for nm in ('PrimIsDefined', 'PrimIsActive', 'PrimIsLoaded', 'PrimIsModel',
               'PrimIsGroup', 'PrimIsAbstract', 'PrimIsInstance',
               'PrimHasDefiningSpecifier', 'PrimAllPrimsPredicate'):
        if not hasattr(U, nm):
            setattr(U, nm, _PrimPredicate())
    if not hasattr(U, 'TraverseInstanceProxies'):
        U.TraverseInstanceProxies = lambda pred=None: pred or _PrimPredicate()
    if not hasattr(U, 'ResolveInfoSourceTimeSamples'):
        U.ResolveInfoSourceTimeSamples = 0
    for nm, val in (('ResolveInfoSourceNone', 1),
                    ('ResolveInfoSourceFallback', 2),
                    ('ResolveInfoSourceDefault', 3),
                    ('ResolveInfoSourceSpline', 4),
                    ('ResolveInfoSourceValueClips', 5)):
        if not hasattr(U, nm):
            setattr(U, nm, val)
    if not hasattr(U, 'InterpolationTypeHeld'):
        U.InterpolationTypeHeld = 0
    if not hasattr(U, 'InterpolationTypeLinear'):
        U.InterpolationTypeLinear = 1
    if not hasattr(U, 'ListPositionFrontOfPrependList'):
        U.ListPositionFrontOfPrependList = 0
    if not hasattr(U, 'ModelAPI'):
        class _ModelAPI:
            def __init__(self, prim): self._prim = prim
            def GetKind(self): return self._prim.GetKind() if hasattr(self._prim, 'GetKind') else ''
            def IsKind(self, k): return self.GetKind() == k
            def GetAssetInfo(self): return {}
            def GetAssetIdentifier(self): return ''
            def GetAssetName(self): return ''
            def GetAssetVersion(self): return ''
        U.ModelAPI = _ModelAPI
    if not hasattr(U, 'StagePopulationMask'):
        class _Mask:
            def __init__(self, *a, **kw): pass
            @staticmethod
            def All(): return _Mask()
        U.StagePopulationMask = _Mask
    if not hasattr(U, 'PrimRange'):
        class _PrimRange:
            def __init__(self, root, predicate=None): self._root = root
            def __iter__(self): return iter(self._root.GetChildren()) if hasattr(self._root, 'GetChildren') else iter([])
            @staticmethod
            def AllPrims(root): return _PrimRange(root)
            @staticmethod
            def AllPrimsPreOrder(root): return _PrimRange(root)
            @staticmethod
            def Stage(stage, predicate=None): return _PrimRange(stage.GetPseudoRoot()) if hasattr(stage, 'GetPseudoRoot') else iter([])
        U.PrimRange = _PrimRange
    else:
        # The native shim's _PrimRange iterates a single subtree but doesn't expose
        # the static factories appController and friends call. Patch them
        # in if missing.
        PR = U.PrimRange
        if not hasattr(PR, 'Stage'):
            def _stage(stage, predicate=None):
                root = stage.GetPseudoRoot() if hasattr(stage, 'GetPseudoRoot') else None
                return PR(root, predicate) if root is not None else iter([])
            PR.Stage = staticmethod(_stage)
        if not hasattr(PR, 'AllPrims'):
            PR.AllPrims = staticmethod(lambda root: PR(root))
        if not hasattr(PR, 'AllPrimsPreOrder'):
            PR.AllPrimsPreOrder = staticmethod(lambda root: PR(root))


_bulk_patch_usd()


_native_compat.Usd.TraverseInstanceProxies = lambda pred=None: pred


def _patch_pseudo_and_prim_filtered():
    for nm in dir(_native_compat):
        obj = getattr(_native_compat, nm)
        if isinstance(obj, type) and obj.__name__ in ('_PseudoRootPrim', '_UsdPrim'):
            cls = obj
            if not hasattr(cls, 'GetFilteredChildren'):
                cls.GetFilteredChildren = lambda self, pred=None: self.GetChildren()
            if not hasattr(cls, 'GetAllChildren'):
                cls.GetAllChildren = lambda self: self.GetChildren()
            if not hasattr(cls, 'GetChildrenNames'):
                cls.GetChildrenNames = lambda self: [c.GetName() for c in self.GetChildren()]
_patch_pseudo_and_prim_filtered()


# ----------------------------------------------------------- layer + composition objects
# Pixar's Layer Stack and Composition tabs need real-shape objects
# describing the (single) root layer of a flat USDC. Build minimal but
# usable layer + spec + composition-arc objects backed by what nanousd
# already exposes (root layer path, prim metadata).

# Fallback .usda subLayers text parser — only used when the C-API
# layer enumeration is unavailable. The newton shim no longer exposes
# this helper (commit 70f2cc0 dropped it once the C API covered the
# common case), so we keep a minimal local copy.
def _read_usda_sublayers(real_path: str) -> list[str]:
    try:
        with open(real_path, "r", encoding="utf-8", errors="ignore") as f:
            head = f.read(8192)
    except OSError:
        return []
    import re
    m = re.search(r"subLayers\s*=\s*\[(.*?)\]", head, flags=re.DOTALL)
    if not m:
        return []
    return [s.strip().strip("@") for s in re.findall(r"@[^@]+@", m.group(1))]


def _find_usda_prim_declaration(usda: str, prim_path: str):
    import re as _re

    path = str(prim_path)
    if not path or path == "/":
        return None
    name = path.rstrip("/").rsplit("/", 1)[-1]
    if "{" in name and name.endswith("}"):
        name = name.split("{", 1)[0]
    elif "}" in name:
        name = name.rsplit("}", 1)[-1]
    if not name:
        return None
    pattern = (
        r"(?ms)^\s*(?P<specifier>def|over|class)\s+"
        r"(?:(?P<typeName>[A-Za-z_][\w:]*)\s+)?\""
        + _re.escape(name)
        + r"\"\s*(?:\((?P<meta>.*?)\))?\s*\{"
    )
    match = _re.search(pattern, usda)
    return match.groupdict() if match else None


def _extract_braced_metadata_field(text: str, field: str) -> str | None:
    import re as _re

    match = _re.search(r"\b" + _re.escape(field) + r"\s*=\s*\{", text)
    if not match:
        return None
    start = text.find("{", match.start())
    if start < 0:
        return None
    depth = 0
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:idx]
    return None


def _parse_usda_asset_info_dict(text: str) -> dict:
    import re as _re

    data = {}
    asset_path_cls = getattr(_native_compat.Sdf, "AssetPath", None)
    for typ, key, raw in _re.findall(
            r"\b(asset|string|token|int|bool|float|double)\s+"
            r"([A-Za-z_]\w*)\s*=\s*(\"[^\"]*\"|@[^@]*@|[^\s}]+)",
            text):
        if raw.startswith('"') and raw.endswith('"'):
            value = raw[1:-1]
        elif raw.startswith("@") and raw.endswith("@"):
            authored = raw[1:-1]
            value = asset_path_cls(authored) if asset_path_cls else authored
        elif typ == "bool":
            value = raw.lower() in {"1", "true"}
        elif typ == "int":
            value = int(raw)
        elif typ in {"float", "double"}:
            value = float(raw)
        else:
            value = raw
        data[key] = value
    return data


def _make_metadata_listop(field: str, items: list[str], *, explicit=False):
    listop_cls = getattr(_native_compat, "_ListOp", None)
    if listop_cls is None or not items:
        return None
    value = (
        listop_cls.Create(explicitItems=items)
        if explicit
        else listop_cls.Create(prependedItems=items)
    )
    try:
        value._listop_field = field
    except Exception:
        pass
    return value


def _parse_usda_listop_field(meta: str, field: str):
    import re as _re

    # Array-ish token listops, e.g. prepend apiSchemas = ["A", "B"].
    array_match = _re.search(
        r"\b(?P<op>prepend|append|delete)?\s*"
        + _re.escape(field)
        + r"\s*=\s*\[(?P<body>[^\]]*)\]",
        meta,
    )
    if array_match:
        items = _re.findall(r"\"([^\"]+)\"", array_match.group("body"))
        return _make_metadata_listop(
            field, items, explicit=array_match.group("op") != "prepend")

    scalar_match = _re.search(
        r"\b(?P<op>prepend|append|delete)?\s*"
        + _re.escape(field)
        + r"\s*=\s*(?P<value>@[^@]+@(?:<[^>]+>)?|<[^>]+>|/[^\s)]+)",
        meta,
    )
    if scalar_match:
        return _make_metadata_listop(
            field, [scalar_match.group("value")],
            explicit=scalar_match.group("op") != "prepend")
    return None


def _layer_has_variant_spec(layer, path_str: str) -> bool:
    if "{" not in path_str or "}" not in path_str or not path_str.endswith("}"):
        return False
    text = _read_layer_text(getattr(layer, "realPath", ""))
    if not text:
        return False
    import re as _re

    match = _re.search(r"\{([^}=]+)=([^}]+)\}", path_str)
    if not match:
        return False
    set_name, selection = match.groups()
    return bool(
        _re.search(r'variantSet\s+"' + _re.escape(set_name) + r'"', text)
        and _re.search(r'"' + _re.escape(selection) + r'"\s*\{', text)
    )


def _layer_spec_info(layer, path_str: str, prim=None) -> dict:
    info = {}
    if _layer_has_variant_spec(layer, path_str):
        return info
    text = _read_layer_text(getattr(layer, "realPath", ""))
    decl = _find_usda_prim_declaration(text, path_str) if text else None
    if decl:
        specifier = decl.get("specifier") or ""
        type_name = decl.get("typeName") or ""
        meta = decl.get("meta") or ""
        if type_name and specifier == "def":
            info["typeName"] = type_name
        if "active" in meta:
            import re as _re
            match = _re.search(r"\bactive\s*=\s*(true|false|0|1)", meta)
            if match:
                info["active"] = match.group(1).lower() in {"true", "1"}
        custom_body = _extract_braced_metadata_field(meta, "customData")
        if custom_body is not None:
            parsed = _native_compat._parse_usda_custom_data_dict(custom_body)
            if parsed:
                info["customData"] = parsed
        asset_body = _extract_braced_metadata_field(meta, "assetInfo")
        if asset_body is not None:
            parsed = _parse_usda_asset_info_dict(asset_body)
            if parsed:
                info["assetInfo"] = parsed
        for key in ("kind", "documentation", "displayName"):
            import re as _re
            match = _re.search(r"\b" + key + r"\s*=\s*\"([^\"]*)\"", meta)
            if match:
                info[key] = match.group(1)
        for field in ("apiSchemas", "payload"):
            value = _parse_usda_listop_field(meta, field)
            if value:
                info[field] = value
    elif prim is not None:
        try:
            if hasattr(prim, "GetTypeName"):
                tn = prim.GetTypeName()
                if tn:
                    info["typeName"] = tn
        except Exception:
            pass
    return info


class _NuLayer:
    """Minimal pxr.Sdf.Layer-like wrapper around a single nanousd layer.

    Accepts an optional (stage, layerIdx) pair so it can read sublayers
    via the C API (works for .usdc as well as .usda). When the pair is
    omitted (e.g. constructing from a bare asset path), falls back to a
    text-parse of the .usda header — strictly worse but still useful.
    """
    def __init__(self, identifier: str, real_path: str | None = None,
                 stage_h: Any = None, layer_idx: int = -1):
        import os as _os
        self.identifier = identifier
        path = real_path or identifier
        self.realPath = _os.path.abspath(path) if path and _os.path.exists(path) else path
        self.timeCodesPerSecond = 24.0
        self.startTimeCode = 0.0
        self.endTimeCode = 0.0
        self._stage_h = stage_h
        self._layer_idx = int(layer_idx)
        self.subLayerPaths, self.subLayerOffsets = self._read_sublayers()

    def _read_sublayers(self):
        # Prefer the native binding when we know our slot in the stage's layer list.
        if self._stage_h is not None and self._layer_idx >= 0:
            try:
                paths = list(self._stage_h.layer_sublayer_paths(self._layer_idx))
                # Per-sublayer offsets aren't exposed yet (would need a
                # per-(layerIdx, subIdx) accessor). Default to identity.
                return paths, [(0.0, 1.0) for _ in paths]
            except Exception:
                pass
        # Fallback: .usda text parse.
        paths = _read_usda_sublayers(self.realPath)
        return paths, [(0.0, 1.0) for _ in paths]
    def GetDisplayName(self):
        import os
        return os.path.basename(self.identifier)
    def GetObjectAtPath(self, path):
        path_str = str(path)
        if not path_str:
            return None
        if path_str == "/":
            return _NuPrimSpec(self, path_str)
        if self._stage_h is not None and self._layer_idx >= 0:
            try:
                if self._stage_h.layer_has_prim_spec(self._layer_idx, path_str):
                    return _NuPrimSpec(self, path_str)
                if _layer_has_variant_spec(self, path_str):
                    return _NuPrimSpec(self, path_str)
                return None
            except Exception:
                pass
        return _NuPrimSpec(self, path_str) if (
            _layer_has_variant_spec(self, path_str)
            or _layer_spec_info(self, path_str)
        ) else None
    def GetNumTimeSamplesForPath(self, _path):
        return 0
    def IsMuted(self):
        return False
    def __bool__(self):
        return True
    def __eq__(self, other):
        return isinstance(other, _NuLayer) and self.identifier == other.identifier
    def __hash__(self):
        return hash(self.identifier)


class _NuPrimSpec:
    """Minimal pxr.Sdf.PrimSpec — what _updateLayerStackView reads."""
    def __init__(self, layer: _NuLayer, path_str: str, prim=None):
        self.layer = layer
        # Path object exposing pathString.
        self.path = type('_PathView', (), {
            'pathString': path_str, '__str__': lambda s: path_str,
            '__repr__': lambda s: f'Path({path_str!r})'})()
        self.path.pathString = path_str  # double-set just to be safe
        self._info = self._collect_info(prim)
    def _collect_info(self, prim):
        # Match Pixar usdview: only spec-authored fields go into the
        # Layer Stack metadata column. typeName + kind + documentation
        # are usually authored; specifier and active are NOT shown
        # unless authored.
        try:
            return _layer_spec_info(self.layer, str(self.path), prim)
        except Exception:
            return {}
    def GetMetaDataInfoKeys(self):
        return list(self._info.keys())
    def HasInfo(self, key):
        return key in self._info
    def GetInfo(self, key):
        return self._info.get(key)
    def __bool__(self):
        return True


def _nu_layer_path_string(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(
        getattr(value, "realPath", "")
        or getattr(value, "identifier", "")
        or value
    )


class _NuArcType:
    """Pcp.NodeRef.arcType stand-in."""
    def __init__(self, name: str):
        self.displayName = name


class _NuLayerTree:
    """Pcp.LayerStack.layerTree node — wraps a layer + its child sublayers.

    Recursively walks the layer's subLayerPaths (parsed from the USDA
    file header) so the Layer-Stack panel nests sublayers under their
    parents in the right order.
    """
    def __init__(self, layer: _NuLayer, _seen=None, stage_h: Any = None,
                 stage_layer_paths: list[str] | None = None):
        import os as _os
        self.layer = layer
        # Per-layer offset from the native binding (cumulative offset/scale in stage
        # frame). Identity for the root sublayer stack; non-identity once
        # an ancestor introduces retiming.
        offset_val = 0.0
        scale_val = 1.0
        if stage_h is not None and getattr(layer, "_layer_idx", -1) >= 0:
            try:
                offset_val, scale_val = stage_h.layer_offset(layer._layer_idx)
            except Exception:
                pass
        class _LO:
            pass
        lo = _LO()
        lo.offset = offset_val
        lo.scale  = scale_val
        lo.IsIdentity = lambda self=lo: self.offset == 0.0 and self.scale == 1.0
        lo.__repr__   = lambda self=lo: \
            f"LayerOffset(offset={self.offset}, scale={self.scale})"
        self.offset = lo
        # Recursive sublayer expansion. _seen breaks cycles defensively;
        # stage_layer_paths lets us look up each sublayer's index in the
        # composed graph so we can pull its sublayers via the C API too.
        seen = set(_seen) if _seen else set()
        rp = getattr(layer, "realPath", "") or ""
        if rp:
            seen.add(_os.path.abspath(rp))
        self.childTrees = []
        for sub_path in getattr(layer, "subLayerPaths", []) or []:
            anchored = _os.path.abspath(_os.path.join(_os.path.dirname(rp), sub_path)) \
                if rp else sub_path
            if anchored in seen:
                continue
            # Find this sublayer's index in the composed graph if possible.
            sub_idx = -1
            if stage_layer_paths:
                for i, p in enumerate(stage_layer_paths):
                    p_str = _nu_layer_path_string(p)
                    if p_str and _os.path.abspath(p_str) == anchored:
                        sub_idx = i
                        break
            sub_layer = _NuLayer(anchored, anchored,
                                 stage_h=stage_h, layer_idx=sub_idx)
            self.childTrees.append(_NuLayerTree(
                sub_layer, _seen=seen | {anchored},
                stage_h=stage_h, stage_layer_paths=stage_layer_paths))


class _NuLayerStack:
    def __init__(self, layer: _NuLayer, stage_h: Any = None,
                 stage_layer_paths: list[str] | None = None):
        self.layerTree = _NuLayerTree(
            layer, stage_h=stage_h, stage_layer_paths=stage_layer_paths)


class _NuPcpNode:
    """Pcp.NodeRef-like — composition node. Root nodes own children
    representing reference / payload arcs (one per arc target)."""
    def __init__(self, layer: _NuLayer, prim_path: str, arc_type: str = "root",
                 stage_h: Any = None, stage_layer_paths: list[str] | None = None,
                 intro_path: str | None = None):
        self.path = prim_path
        self._intro_path = intro_path or prim_path
        self.arcType = _NuArcType(arc_type)
        self.layerStack = _NuLayerStack(
            layer, stage_h=stage_h, stage_layer_paths=stage_layer_paths)
        self.children = []
    def GetPathAtIntroduction(self):
        return self._intro_path
    def CanContributeSpecs(self):
        return True


def _build_arc_children(prim):
    """Return a list of _NuPcpNode children for the prim's authored
    composition arcs — references, payloads, inherits, specializes —
    walked from each list-op's prepended / appended / explicit items.

    Per USDA syntax, the unqualified form `references = @asset@</Path>`
    sets the explicit list (read via nanousd_listop_item / nitems), while
    `prepend references = @...@` populates the prepended list. Walk all
    three so inline references show up too.
    """
    children = []
    if prim is None: return children
    # Field-name aliases per spec: USDA's `inherits = ...` writes the
    # `inheritPaths` field; `specializes = ...` writes `specializes`.
    # Try both names so we don't depend on the spelling the native layer uses.
    arc_specs = (
        (("inheritPaths", "inherits"), "inherit"),
        (("references",), "reference"),
        (("payload", "payloads"), "payload"),
        (("specializes",), "specialize"),
    )
    mu = getattr(prim, "_prim", None)
    if mu is None:
        return children

    stage = getattr(prim, "_stage", None) or getattr(prim, "_stage_ref", None)
    stage_mu = getattr(stage, "_stage", None) if stage is not None else None
    stage_h = getattr(stage_mu, "_h", None) if stage_mu is not None else None
    stage_layer_paths = []
    if stage_h is not None:
        try:
            stage_layer_paths = [stage_h.layer_path(i) for i in range(stage_h.num_layers)]
        except Exception:
            stage_layer_paths = []
    root_path = stage_layer_paths[0] if stage_layer_paths else getattr(stage, "_filepath", "")

    def _layer_for_arc(asset: str):
        if asset and stage_layer_paths:
            for idx, layer_path in enumerate(stage_layer_paths):
                if _layer_matches_asset(layer_path, asset, root_path):
                    return _NuLayer(layer_path, layer_path, stage_h=stage_h, layer_idx=idx)
        if asset:
            import os as _os
            root_dir = _os.path.dirname(_os.path.abspath(_nu_layer_path_string(root_path)))
            path = asset if _os.path.isabs(asset) else _os.path.join(root_dir, asset)
            return _NuLayer(path, path)
        return _NuLayer(root_path, root_path, stage_h=stage_h, layer_idx=0)

    def _append_variant_children():
        try:
            variant_paths = _variant_property_source_paths(mu)
        except Exception:
            variant_paths = []
        for _asset, variant_path in variant_paths:
            intro = variant_path
            if "}" in intro:
                intro = intro.split("}", 1)[0] + "}"
            children.append(_NuPcpNode(
                _layer_for_arc(""), variant_path, "variant",
                stage_h=stage_h, stage_layer_paths=stage_layer_paths,
                intro_path=intro))

    for fields, arc_type in arc_specs:
        for field in fields:
            try:
                op = mu.get_listop(field)
                if not op:
                    continue
                # Pick the right list based on listop kind:
                #   explicit (`references = ...`)         → items
                #   prepend/append (`prepend references`) → prepended + appended
                # Walking both double-counts because nitems on a
                # prepend-style listop returns the *composed* items.
                items = op.items if op.is_explicit else [*op.prepended_items, *op.appended_items]
                added_any = False
                for arc_str in items:
                    if not arc_str:
                        continue
                    # "@asset@</Path>" -> asset + sub_path. Inherits /
                    # specializes are bare paths "</Foo>" with no asset.
                    asset = ""
                    sub_path = "/"
                    arc_text = str(arc_str)
                    if arc_text.startswith("@"):
                        end = arc_text.find("@", 1)
                        if end > 0:
                            asset = arc_text[1:end]
                            tail = arc_text[end + 1:]
                            if tail.startswith("<") and tail.endswith(">"):
                                sub_path = tail[1:-1]
                    elif arc_text.startswith("<") and arc_text.endswith(">"):
                        sub_path = arc_text[1:-1]
                    elif arc_text.startswith("/"):
                        sub_path = arc_text
                    child = _NuPcpNode(
                        _layer_for_arc(asset),
                        sub_path or "/", arc_type,
                        stage_h=stage_h, stage_layer_paths=stage_layer_paths)
                    children.append(child)
                    added_any = True
                # If we matched a field name (got items), don't try aliases.
                if added_any:
                    break
            except Exception:
                continue
        if arc_type == "inherit":
            _append_variant_children()
    return children


class _NuPrimIndex:
    """Usd.PrimIndex stand-in — root node + child arcs from references/
    payloads/inherits/specializes authored on the prim."""
    def __init__(self, layer: _NuLayer, prim_path: str, prim=None,
                 stage_h: Any = None, stage_layer_paths: list[str] | None = None):
        self.rootNode = _NuPcpNode(layer, prim_path,
                                   stage_h=stage_h,
                                   stage_layer_paths=stage_layer_paths)
        if prim is not None:
            self.rootNode.children = _build_arc_children(prim)
    def IsValid(self):
        return True


# ----------------------------------------------------------- composition tabs
# nuView's bottom-right tabs (Meta Data / Layer Stack / Composition) and the
# property panel populate via a handful of pxr APIs we don't yet shim. Stub
# them out so the panels render an empty-but-non-crashing view rather than
# raising AttributeError up through Qt.
class _LayerOffset:
    """Minimal stand-in for Sdf.LayerOffset used by Layer-Stack tab."""
    def __init__(self, offset=0.0, scale=1.0):
        self.offset = float(offset); self.scale = float(scale)
    def IsIdentity(self): return self.offset == 0.0 and self.scale == 1.0

if hasattr(_native_compat, "Sdf") and not hasattr(_native_compat.Sdf, "LayerOffset"):
    _native_compat.Sdf.LayerOffset = _LayerOffset


class _PrimSpec:
    """Sdf.PrimSpec stub (only what _updateLayerStackView reads)."""
    def __init__(self, prim, layer):
        self._prim = prim; self._layer = layer
        self.path = prim.GetPath() if hasattr(prim, 'GetPath') else ''
    @property
    def layer(self): return self._layer
    def GetPath(self): return self.path


class _PrimIndex:
    """Usd.PrimCompositionQuery stand-in. IsValid()=False ⇒ empty tree."""
    def IsValid(self): return False
    def GetNodeRange(self, *a, **kw): return []
    def GetRootNode(self): return None
    def DumpToString(self): return ""


class _CompositionArc:
    def __init__(self, kind="reference", target=""):
        self._kind, self._target = kind, str(target)
    def GetArcType(self): return self._kind
    def GetTargetPrimPath(self): return self._target


class _PrimCompositionQuery:
    def __init__(self, prim): self._prim = prim
    def GetCompositionArcs(self): return []  # empty until composition is wired


def _patch_prim_for_panels():
    def _stage_handle_and_paths(prim):
        """Return (raw_stage_handle, [layer_paths]) for the prim's stage,
        or (None, None) if unavailable. Used to thread the C API context
        through _NuLayer / _NuLayerTree so they can call layer_n_sublayers,
        layer_offset, etc. without re-opening files."""
        stage = getattr(prim, "_stage", None) or getattr(prim, "_stage_ref", None)
        if stage is None:
            return None, None
        mu = getattr(stage, "_stage", None)
        h = getattr(mu, "_h", None) if mu is not None else None
        paths = []
        if h is not None and hasattr(stage, "GetUsedLayers"):
            try:
                paths = list(getattr(mu, "used_layers", []) or [])
                if not paths:
                    paths = stage.GetUsedLayers()
            except Exception:
                paths = []
        return h, [_nu_layer_path_string(p) for p in paths if _nu_layer_path_string(p)]

    def _stage_root_layer(prim):
        """Return _NuLayer for the prim's stage's root layer, threaded
        with the (stage_h, layerIdx=0) context so subsequent C API calls
        work."""
        stage = getattr(prim, "_stage", None) or getattr(prim, "_stage_ref", None)
        if stage is None: return _NuLayer("<unknown>")
        try:
            ident = getattr(stage, "_filepath", None) or "<stage>"
            h, _paths = _stage_handle_and_paths(prim)
            return _NuLayer(ident, ident, stage_h=h, layer_idx=0)
        except Exception:
            return _NuLayer("<unknown>")

    def _pseudo_root_metadata(self, key):
        stage = getattr(self, "_stage_ref", None)
        mu = getattr(stage, "_stage", None) if stage is not None else None
        if key == "subLayers":
            try:
                root = stage.GetRootLayer()
                return _read_usda_sublayers(getattr(root, "realPath", ""))
            except Exception:
                return None
        if mu is not None:
            try:
                value = mu.get_metadata(str(key))
                if value is not None:
                    return value
            except Exception:
                pass
        return None

    def _pseudo_root_all_metadata(self):
        data = {}
        for key in (
            "defaultPrim",
            "startTimeCode",
            "endTimeCode",
            "timeCodesPerSecond",
        ):
            value = _pseudo_root_metadata(self, key)
            if value not in (None, "", {}, []):
                data[key] = value
        return data

    for nm in dir(_native_compat):
        obj = getattr(_native_compat, nm)
        if not isinstance(obj, type) or obj.__name__ not in (
                '_PseudoRootPrim', '_UsdPrim'):
            continue
        cls = obj
        if cls.__name__ == '_PseudoRootPrim':
            cls.GetMetadata = _pseudo_root_metadata
            cls.GetAllMetadata = _pseudo_root_all_metadata
            cls.GetCustomData = lambda self: {}
            cls.GetAppliedSchemas = lambda self: []
        if not hasattr(cls, 'GetPrimPath'):
            # Pixar alias for GetPath() on UsdPrim. ViewSettings'
            # cameraPrim setter calls value.GetPrimPath() to extract
            # the target path before storing it.
            cls.GetPrimPath = lambda self: self.GetPath()
        if not hasattr(cls, 'GetDisplayName'):
            # Pixar's GetDisplayName returns the authored displayName
            # metadata if any, else GetName(). usdview's primViewItem
            # and Find Prim search both call it.
            def _disp(self):
                try:
                    md = self.GetMetadata("displayName") if hasattr(self, "GetMetadata") else None
                    if md: return str(md)
                except Exception: pass
                try:
                    return self.GetName() if hasattr(self, "GetName") else ""
                except Exception:
                    return ""
            cls.GetDisplayName = _disp
        if not hasattr(cls, 'GetAllMetadata') or \
                cls.GetAllMetadata.__name__ in ("<lambda>", "_all_meta"):
            def _all_meta(self):
                # Match Pixar usdview's metadata fields shape:
                # typeName, active, assetInfo, customData, kind, specifier,
                # documentation, apiSchemas. Read each via GetMetadata
                # (more reliable than GetKind/GetSpecifier in the native shim).
                d = {}
                try:
                    if hasattr(self, 'GetTypeName'):
                        v = self.GetTypeName()
                        if v: d['typeName'] = v
                    if hasattr(self, 'GetMetadata'):
                        for k in ('active', 'assetInfo', 'customData', 'kind',
                                  'documentation', 'displayName',
                                  'displayGroup', 'hidden'):
                            v = self.GetMetadata(k)
                            if v not in (None, "", {}, []): d[k] = v
                        # specifier is always authored — show it.
                        sp = self.GetMetadata('specifier')
                        if sp not in (None, ""):
                            # Pixar shows "Sdf.SpecifierDef" / "Class" / "Over".
                            mapping = {0: 'Sdf.SpecifierDef',
                                       1: 'Sdf.SpecifierClass',
                                       2: 'Sdf.SpecifierOver'}
                            try:
                                d['specifier'] = mapping.get(int(sp),
                                                             f'Sdf.Specifier({sp})')
                            except Exception:
                                d['specifier'] = str(sp)
                    # applied API schemas
                    if hasattr(self, 'GetAppliedSchemas'):
                        s = None
                        if hasattr(self, 'GetAppliedSchemas'):
                            schemas = self.GetAppliedSchemas()
                            listop_cls = getattr(_native_compat, "_ListOp", None)
                            if schemas and listop_cls is not None:
                                s = listop_cls.Create(explicitItems=list(schemas))
                                try:
                                    s._listop_field = "apiSchemas"
                                except Exception:
                                    pass
                            elif schemas:
                                s = schemas
                        if s: d['apiSchemas'] = s
                except Exception:
                    pass
                return d
            cls.GetAllMetadata = _all_meta
        # Read composition arc fields (references/payload/inheritPaths/
        # specializes) as a list of target asset|prim strings via native
        # ListOp objects. Same parsing as _build_arc_children below.
        def _read_arc_targets(self, field):
            try:
                mu = getattr(self, "_prim", None)
                if mu is None:
                    return []
                op = mu.get_listop(field)
                if not op: return []
                return list(op.items if op.is_explicit else [*op.prepended_items, *op.appended_items])
            except Exception:
                return []

        if not hasattr(cls, '_read_arc_targets'):
            cls._read_arc_targets = _read_arc_targets

        if not hasattr(cls, 'GetInherits'):
            class _InheritsProxy:
                def __init__(self, prim): self._prim = prim
                def GetAllDirectInherits(self):
                    return self._prim._read_arc_targets("inheritPaths")
                def GetAddedOrExplicitItems(self):
                    return self.GetAllDirectInherits()
            cls.GetInherits = lambda self, _Ip=_InheritsProxy: _Ip(self)
        if not hasattr(cls, 'GetSpecializes'):
            class _SpecsProxy:
                def __init__(self, prim): self._prim = prim
                def GetAllDirectSpecializes(self):
                    return self._prim._read_arc_targets("specializes")
                def GetAddedOrExplicitItems(self):
                    return self.GetAllDirectSpecializes()
            cls.GetSpecializes = lambda self, _Sp=_SpecsProxy: _Sp(self)
        if not hasattr(cls, 'GetPayloads'):
            class _PayloadsProxy:
                def __init__(self, prim): self._prim = prim
                def GetAllPayloads(self):
                    return self._prim._read_arc_targets("payload")
                def GetAddedOrExplicitItems(self):
                    return self.GetAllPayloads()
            cls.GetPayloads = lambda self, _Pp=_PayloadsProxy: _Pp(self)
        if not hasattr(cls, 'GetReferences'):
            class _ReferencesProxy:
                def __init__(self, prim): self._prim = prim
                def GetAllDirectReferences(self):
                    return self._prim._read_arc_targets("references")
                def GetAddedOrExplicitItems(self):
                    return self.GetAllDirectReferences()
            cls.GetReferences = lambda self, _Rp=_ReferencesProxy: _Rp(self)

        if not hasattr(cls, 'GetPrimIndex') or \
                cls.GetPrimIndex.__name__ in ("<lambda>",):
            def _get_index(self, _stage_root_layer=_stage_root_layer,
                           _stage_handle_and_paths=_stage_handle_and_paths):
                layer = _stage_root_layer(self)
                stage_h, layer_paths = _stage_handle_and_paths(self)
                return _NuPrimIndex(layer, str(self.GetPath()), prim=self,
                                    stage_h=stage_h, stage_layer_paths=layer_paths)
            cls.GetPrimIndex = _get_index
        if not hasattr(cls, 'GetPrimStackWithLayerOffsets') or \
                cls.GetPrimStackWithLayerOffsets.__name__ in ("<lambda>",):
            def _get_stack(self,
                           _stage_root_layer=_stage_root_layer,
                           _stage_handle_and_paths=_stage_handle_and_paths):
                """Return one (spec, offset) pair per layer that authored a
                prim spec for this path. Filtered via nanousd_layer_has_prim_spec
                — matches stock usdview's "only layers that contribute"
                semantics (was previously over-permissive: every stage layer)."""
                layer = _stage_root_layer(self)
                path_str = str(self.GetPath())
                stage_h, layer_paths = _stage_handle_and_paths(self)
                if stage_h is None or not layer_paths:
                    # Pre-stage / orphan prim: just the root layer.
                    return [(_NuPrimSpec(layer, path_str, self),
                             _LayerOffset(0.0, 1.0))]
                out: list = []
                seen: set[str] = set()
                seen_specs: set[tuple[int, str]] = set()

                def _append_specs_for_path(source_path, asset=""):
                    added = False
                    for idx, lp in enumerate(layer_paths):
                        lp_str = _nu_layer_path_string(lp)
                        if not lp_str:
                            continue
                        if not _layer_matches_asset(lp_str, asset, layer_paths[0]):
                            continue
                        try:
                            has_spec = stage_h.layer_has_prim_spec(idx, source_path)
                        except Exception:
                            has_spec = False
                        sub_layer = layer if idx == 0 else \
                            _NuLayer(lp_str, lp_str, stage_h=stage_h, layer_idx=idx)
                        if not has_spec and not _layer_has_variant_spec(sub_layer, source_path):
                            continue
                        key = (idx, source_path)
                        if key in seen_specs:
                            continue
                        seen_specs.add(key)
                        offset, scale = stage_h.layer_offset(idx)
                        out.append((_NuPrimSpec(sub_layer, source_path),
                                    _LayerOffset(offset, scale)))
                        added = True
                    return added

                for idx, lp in enumerate(layer_paths):
                    lp_str = _nu_layer_path_string(lp)
                    if not lp_str or lp_str in seen:
                        continue
                    seen.add(lp_str)
                    if not stage_h.layer_has_prim_spec(idx, path_str):
                        continue
                    offset, scale = stage_h.layer_offset(idx)
                    is_root = (idx == 0)
                    sub_layer = layer if is_root else \
                        _NuLayer(lp_str, lp_str, stage_h=stage_h, layer_idx=idx)
                    out.append((_NuPrimSpec(sub_layer, path_str, self),
                                _LayerOffset(offset, scale)))
                    seen_specs.add((idx, path_str))
                mu = getattr(self, "_prim", None)
                if mu is not None:
                    for asset, source_path in _composition_prim_source_paths(mu):
                        _append_specs_for_path(source_path, asset)
                # Fallback: if filtering came up empty (e.g. pseudo-root
                # which never has a spec), at least show the root layer.
                if not out:
                    out.append((_NuPrimSpec(layer, path_str, self),
                                _LayerOffset(0.0, 1.0)))
                return out
            cls.GetPrimStackWithLayerOffsets = _get_stack
        if not hasattr(cls, 'GetPrimStack') or \
                cls.GetPrimStack.__name__ in ("<lambda>",):
            cls.GetPrimStack = lambda self: \
                [s for s, _o in self.GetPrimStackWithLayerOffsets()]
        if not hasattr(cls, 'GetCompositionQuery') or \
                cls.GetCompositionQuery.__name__ in ("<lambda>",):
            cls.GetCompositionQuery = lambda self: _PrimCompositionQuery(self)
        if not hasattr(cls, 'IsPseudoRoot'):
            cls.IsPseudoRoot = lambda self: (
                getattr(self, 'GetPath', lambda: '/')() == '/' or
                self.__class__.__name__ == '_PseudoRootPrim')
_patch_prim_for_panels()


def _iter_listop_items(prim, *fields):
    for field in fields:
        try:
            op = prim.get_listop(field)
        except Exception:
            op = None
        if not op:
            continue
        try:
            items = op.items if op.is_explicit else [*op.prepended_items, *op.appended_items]
        except Exception:
            items = []
        items = [str(item) for item in items if str(item)]
        if items:
            return items
    return []


def _parse_arc_item(item):
    text = str(item)
    asset = ""
    prim_path = "/"
    if text.startswith("@"):
        end = text.find("@", 1)
        if end > 0:
            asset = text[1:end]
            tail = text[end + 1:]
            if tail.startswith("<") and tail.endswith(">"):
                prim_path = tail[1:-1]
    elif text.startswith("<") and text.endswith(">"):
        prim_path = text[1:-1]
    elif text.startswith("/"):
        prim_path = text
    return asset, prim_path or "/"


def _layer_matches_asset(layer_path, asset, root_layer_path):
    if not asset:
        return True
    import os as _os
    layer_text = _nu_layer_path_string(layer_path)
    if not layer_text:
        return False
    root_dir = _os.path.dirname(_os.path.abspath(_nu_layer_path_string(root_layer_path)))
    expected = asset if _os.path.isabs(asset) else _os.path.join(root_dir, asset)
    return (
        _os.path.abspath(layer_text) == _os.path.abspath(expected)
        or _os.path.basename(layer_text) == _os.path.basename(asset)
    )


def _variant_property_source_paths(prim):
    stage = getattr(prim, "_stage_h", None)
    stage_h = getattr(stage, "_h", None) if stage is not None else None
    path = getattr(prim, "path", "")
    if stage is None or stage_h is None or not path or path == "/":
        return []
    layer_paths = [stage_h.layer_path(i) for i in range(stage_h.num_layers)]
    parts = path.strip("/").split("/")
    out = []
    for index in range(1, len(parts) + 1):
        prefix = "/" + "/".join(parts[:index])
        try:
            ancestor = stage.get_prim_at_path(prefix)
        except Exception:
            ancestor = None
        if not ancestor:
            continue
        try:
            set_names = ancestor.variant_set_names()
        except Exception:
            set_names = []
        for set_name in set_names:
            try:
                selection = ancestor.variant_selection(set_name)
            except Exception:
                selection = ""
            if not selection:
                continue
            suffix = path[len(prefix):]
            if suffix.startswith("/"):
                suffix = suffix[1:]
            source_path = f"{prefix}{{{set_name}={selection}}}{suffix}"
            has_spec = False
            for idx, layer_path in enumerate(layer_paths):
                try:
                    if stage_h.layer_has_prim_spec(idx, source_path):
                        has_spec = True
                        break
                    layer = _NuLayer(layer_path, layer_path, stage_h=stage_h, layer_idx=idx)
                    if _layer_has_variant_spec(layer, source_path):
                        has_spec = True
                        break
                except Exception:
                    continue
            if has_spec:
                out.append(("", source_path))
    return out


def _composition_prim_source_paths(prim):
    out = []
    for item in _iter_listop_items(prim, "inheritPaths", "inherits"):
        _asset, prim_path = _parse_arc_item(item)
        out.append(("", prim_path))
    out.extend(_variant_property_source_paths(prim))
    for item in _iter_listop_items(prim, "references"):
        out.append(_parse_arc_item(item))
    for item in _iter_listop_items(prim, "payload", "payloads"):
        out.append(_parse_arc_item(item))
    for item in _iter_listop_items(prim, "specializes"):
        _asset, prim_path = _parse_arc_item(item)
        out.append(("", prim_path))
    seen = set()
    unique = []
    for asset, prim_path in out:
        key = (asset, prim_path)
        if prim_path and key not in seen:
            seen.add(key)
            unique.append(key)
    return unique


def _composition_property_source_paths(prim):
    out = []
    for fields in (("references",), ("payload", "payloads")):
        for item in _iter_listop_items(prim, *fields):
            out.append(_parse_arc_item(item))
    for fields in (("inheritPaths", "inherits"), ("specializes",)):
        for item in _iter_listop_items(prim, *fields):
            _asset, prim_path = _parse_arc_item(item)
            out.append(("", prim_path))
    out.extend(_variant_property_source_paths(prim))
    seen = set()
    unique = []
    for asset, prim_path in out:
        key = (asset, prim_path)
        if prim_path and key not in seen:
            seen.add(key)
            unique.append(key)
    return unique


def _read_layer_text(layer_path):
    try:
        return _Path(_nu_layer_path_string(layer_path)).read_text(encoding="utf-8")
    except OSError:
        return ""


def _layer_has_relationship_opinion(layer_path, prim_path, rel_name):
    text = _read_layer_text(layer_path)
    if not text:
        return False
    try:
        block = (
            _native_compat._find_usda_prim_body_by_name(text, str(prim_path))
            or _native_compat._find_usda_prim_block(text, str(prim_path))
        )
    except Exception:
        block = None
    if not block:
        return False
    import re as _re
    pattern = (
        r"^\s*(?:custom\s+)?(?:prepend\s+|append\s+|delete\s+|add\s+)?"
        r"rel\s+" + _re.escape(str(rel_name)) + r"\b"
    )
    return bool(_re.search(pattern, block, _re.M))


def _property_stack_for_name(prim, name, *, relationship=False):
    if prim is None:
        return []
    stage_mu = getattr(prim, "_stage_h", None)
    stage_h = getattr(stage_mu, "_h", None) if stage_mu is not None else None
    if not stage_h:
        return []
    root_layer_path = stage_h.layer_path(0) if getattr(stage_h, "num_layers", 0) else ""

    def has_opinion(idx, source_path):
        if relationship:
            return _layer_has_relationship_opinion(stage_h.layer_path(idx), source_path, name)
        try:
            return stage_h.layer_has_attr_opinion(idx, source_path, name)
        except Exception:
            return False

    def append_rows(source_path, asset=""):
        rows = []
        seen_layers: set[str] = set()
        for idx in range(stage_h.num_layers):
            lp_str = _nu_layer_path_string(stage_h.layer_path(idx))
            if not lp_str or lp_str in seen_layers:
                continue
            if not _layer_matches_asset(lp_str, asset, root_layer_path):
                continue
            seen_layers.add(lp_str)
            if not has_opinion(idx, source_path):
                continue
            offset, scale = stage_h.layer_offset(idx)
            layer = _NuLayer(lp_str, lp_str, stage_h=stage_h, layer_idx=idx)
            rows.append((_NuPrimSpec(layer, f"{source_path}.{name}"),
                         _LayerOffset(offset, scale)))
        return rows

    rows = append_rows(getattr(prim, "path", ""))
    if rows:
        return rows
    for asset, source_path in _composition_property_source_paths(prim):
        rows = append_rows(source_path, asset)
        if rows:
            return rows
    return []


def _patch_attribute_for_panels():
    for nm in dir(_native_compat):
        obj = getattr(_native_compat, nm)
        if not isinstance(obj, type) or obj.__name__ != '_Attribute':
            continue
        cls = obj
        if not hasattr(cls, 'HasAuthoredConnections'):
            cls.HasAuthoredConnections = lambda self: False
        if not hasattr(cls, 'GetConnections'):
            cls.GetConnections = lambda self: []

        # Property layer stack: filter via layer_has_attr_opinion
        # so only authoring layers appear (matches stock usdview).
        def _attr_property_stack(self, *args, **kwargs):
            try:
                return _property_stack_for_name(
                    getattr(self, "_prim", None), getattr(self, "_name", ""))
            except Exception:
                return []
        if not hasattr(cls, 'GetPropertyStackWithLayerOffsets') or \
                getattr(cls.GetPropertyStackWithLayerOffsets, '__name__', '') == '<lambda>':
            cls.GetPropertyStackWithLayerOffsets = _attr_property_stack
        if not hasattr(cls, 'GetPropertyStack') or \
                getattr(cls.GetPropertyStack, '__name__', '') == '<lambda>':
            cls.GetPropertyStack = lambda self: [s for s, _o in
                                                 self.GetPropertyStackWithLayerOffsets()]
        if not hasattr(cls, 'GetAllMetadata'):
            cls.GetAllMetadata = lambda self: {}
        if not hasattr(cls, 'GetMetadata'):
            cls.GetMetadata = lambda self, key: None
        if not hasattr(cls, 'HasAuthoredValueOpinion'):
            cls.HasAuthoredValueOpinion = lambda self: \
                self.HasAuthoredValue() if hasattr(self, 'HasAuthoredValue') else False
_patch_attribute_for_panels()


def _patch_property_metadata_for_usdview():
    A = getattr(_native_compat, "_Attribute", None)
    R = getattr(_native_compat, "_Relationship", None)
    if A is None and R is None:
        return
    S = _native_compat.Sdf

    def _owner_prim(prop):
        return getattr(prop, "_prim", None)

    def _property_path(prop):
        prim = _owner_prim(prop)
        base = getattr(prim, "path", "") if prim is not None else ""
        name = getattr(prop, "_name", "")
        return S.Path(str(base) + "." + str(name)) if base and name else S.Path("")

    class _TokenArrayView:
        _isVtArray = False

        def __init__(self, values):
            self._values = [str(value) for value in values]

        def __iter__(self):
            return iter(self._values)

        def __len__(self):
            return len(self._values)

        def __getitem__(self, index):
            return self._values[index]

        def __repr__(self):
            return "[" + ", ".join(self._values) + "]"

        __str__ = __repr__

    def _property_declaration_line(prop, relationship=False):
        try:
            prim = _owner_prim(prop)
            name = getattr(prop, "_name", "")
            if prim is None or not name:
                return ""
            source = _native_compat._stage_source_usda(getattr(prim, "_stage_h", None))
            block = (
                _native_compat._find_usda_prim_body_by_name(source, prim.path)
                or _native_compat._find_usda_prim_block(source, prim.path)
            ) if source else None
            if not block:
                return ""
            import re as _re
            if relationship:
                pattern = (
                    r"^\s*(?:custom\s+)?"
                    r"(?:prepend\s+|append\s+|delete\s+|add\s+)?"
                    r"rel\s+" + _re.escape(str(name)) + r"\b"
                )
            else:
                pattern = (
                    r"^\s*(?:custom\s+)?[A-Za-z_][\w:<>,\[\]]*\s+"
                    + _re.escape(str(name)) + r"\b"
                )
            match = _re.search(pattern, block, _re.M)
            return match.group(0) if match else ""
        except Exception:
            return ""

    def _is_custom_property(prop, relationship=False):
        line = _property_declaration_line(prop, relationship)
        if line:
            return line.lstrip().startswith("custom ")
        if relationship:
            return False
        name = getattr(prop, "_name", "")
        builtin_names = {
            "doubleSided",
            "extent",
            "orientation",
            "physics:collisionEnabled",
            "physics:simulationOwner",
            "proxyPrim",
            "purpose",
            "size",
            "visibility",
            "xformOpOrder",
        }
        if (
            name in builtin_names
            or name.startswith("xformOp:")
            or name.startswith("primvars:")
        ):
            return False
        return True

    def _allowed_tokens_for_attribute(attr):
        name = getattr(attr, "_name", "")
        if name == "visibility":
            return _TokenArrayView(["inherited", "invisible"])
        if name == "purpose":
            return _TokenArrayView(["default", "render", "proxy", "guide"])
        if name == "orientation":
            return _TokenArrayView(["rightHanded", "leftHanded"])
        return None

    def _variability_for_attribute(attr):
        name = getattr(attr, "_name", "")
        if name in {"doubleSided", "orientation", "purpose", "xformOpOrder"}:
            return "Sdf.VariabilityUniform"
        return "Sdf.VariabilityVarying"

    def _schema_default_for_attribute(attr):
        name = getattr(attr, "_name", "")
        prim = _owner_prim(attr)
        if name == "extent" and getattr(prim, "type_name", "") == "Cube":
            size = 2.0
            try:
                authored = prim.get_attribute("size")
                if authored is not None:
                    size = float(authored)
            except Exception:
                pass
            half = size * 0.5
            def _coord(value):
                return str(int(value)) if float(value).is_integer() else str(value)
            lo = _coord(-half)
            hi = _coord(half)
            return f"[({lo}, {lo}, {lo}), ({hi}, {hi}, {hi})]"
        return None

    if A is not None:
        _orig_attr_get_metadata = getattr(A, "GetMetadata", None)

        def _attr_get_metadata(self, key):
            if key == "custom":
                return _is_custom_property(self)
            if key == "typeName":
                value = self.GetTypeName() if hasattr(self, "GetTypeName") else ""
                return str(value) if value else None
            if key == "variability":
                return _variability_for_attribute(self)
            if key == "default":
                value = self.Get() if hasattr(self, "Get") else None
                return value if value is not None else _schema_default_for_attribute(self)
            if key == "allowedTokens":
                return _allowed_tokens_for_attribute(self)
            if key == "displayName":
                if getattr(self, "_name", "") == "physics:collisionEnabled":
                    return "Collision Enabled"
                return None
            if _orig_attr_get_metadata is not None:
                try:
                    return _orig_attr_get_metadata(self, key)
                except Exception:
                    return None
            return None

        def _attr_all_metadata(self):
            data = {}
            custom = _is_custom_property(self)
            data["custom"] = custom
            type_name = self.GetTypeName() if hasattr(self, "GetTypeName") else ""
            if type_name:
                data["typeName"] = str(type_name)
            data["variability"] = _variability_for_attribute(self)
            allowed = _allowed_tokens_for_attribute(self)
            if allowed:
                data["allowedTokens"] = allowed
            display_name = self.GetMetadata("displayName")
            if display_name:
                data["displayName"] = display_name
            value = self.Get() if hasattr(self, "Get") else None
            if value is None:
                value = _schema_default_for_attribute(self)
            name = getattr(self, "_name", "")
            has_decl = bool(_property_declaration_line(self))
            include_default = (
                value is not None
                and not custom
                and not (hasattr(self, "HasTimeSamples") and self.HasTimeSamples())
                and (
                    has_decl
                    or name in {
                        "doubleSided",
                        "extent",
                        "orientation",
                        "physics:collisionEnabled",
                        "purpose",
                        "size",
                        "visibility",
                    }
                )
                and name != "xformOpOrder"
                and not name.startswith("xformOp:")
                and not name.startswith("primvars:")
            )
            if include_default:
                data["default"] = value
            return data

        A.GetMetadata = _attr_get_metadata
        A.GetAllMetadata = _attr_all_metadata

    if R is not None:
        _orig_rel_get_metadata = getattr(R, "GetMetadata", None)

        def _rel_path(self):
            return _property_path(self)

        def _rel_get_metadata(self, key):
            if key == "custom":
                return _is_custom_property(self, relationship=True)
            if key == "variability":
                return "Sdf.VariabilityUniform"
            if key == "displayName" and getattr(self, "_name", "") == "physics:simulationOwner":
                return "Simulation Owner"
            if _orig_rel_get_metadata is not None:
                try:
                    return _orig_rel_get_metadata(self, key)
                except Exception:
                    return None
            return None

        def _rel_all_metadata(self):
            data = {
                "custom": _is_custom_property(self, relationship=True),
                "variability": "Sdf.VariabilityUniform",
            }
            display_name = self.GetMetadata("displayName")
            if display_name:
                data["displayName"] = display_name
            return data

        def _rel_property_stack(self, *args, **kwargs):
            try:
                return _property_stack_for_name(
                    getattr(self, "_prim", None),
                    getattr(self, "_name", ""),
                    relationship=True)
            except Exception:
                return []

        if not hasattr(R, "GetPath"):
            R.GetPath = _rel_path
        if not hasattr(R, "GetPrimPath"):
            R.GetPrimPath = lambda self: self.GetPath().GetPrimPath()
        if not hasattr(R, "GetPrim"):
            R.GetPrim = lambda self: getattr(self, "_prim", None)
        R.GetPropertyStackWithLayerOffsets = _rel_property_stack
        R.GetPropertyStack = lambda self: [s for s, _o in self.GetPropertyStackWithLayerOffsets()]
        R.GetMetadata = _rel_get_metadata
        R.GetAllMetadata = _rel_all_metadata


_patch_property_metadata_for_usdview()


# ----------------------------------------------------------- bind types to Usd
# usdview's property tree does `type(primProperty) == Usd.Attribute` and
# `isinstance(p, Usd.Property)` — those names need to exist on the Usd
# module and resolve to the same classes The native shim's _UsdPrim returns from
# GetAttributes() / GetRelationships().
def _bind_property_types_to_usd():
    import abc
    U = _native_compat.Usd
    A = getattr(_native_compat, "_Attribute", None)
    R = getattr(_native_compat, "_Relationship", None)
    UP = getattr(_native_compat, "_UsdPrim", None)
    PR = getattr(_native_compat, "_PseudoRootPrim", None)
    if A is not None and not hasattr(U, "Attribute"):
        U.Attribute = A
    if R is not None and not hasattr(U, "Relationship"):
        U.Relationship = R
    if not hasattr(U, "Property"):
        U.Property = A or R
    # Pixar's Usd.Prim — used in BOTH `type(obj) is Usd.Prim` (strict
    # identity, needed by _updateMetadataView for the [object type] row)
    # AND `isinstance(obj, Usd.Prim)` (loose check that should admit
    # _PseudoRootPrim too, used by rootDataModel guards).
    #
    # Solution: install a metaclass on _UsdPrim that admits _PseudoRootPrim
    # via __instancecheck__/__subclasscheck__. Since type(x) bypasses
    # metaclass hooks, also patch _PseudoRootPrim instances' __class__.
    if UP is not None:
        U.Prim = UP

        class _UsdPrimMeta(type):
            def __instancecheck__(cls, inst):
                if type.__instancecheck__(cls, inst):
                    return True
                # Admit anything with the duck-typed Usd.Prim methods.
                return hasattr(inst, "GetPath") and hasattr(inst, "GetTypeName") \
                       and hasattr(inst, "IsValid")
            def __subclasscheck__(cls, sub):
                if type.__subclasscheck__(cls, sub):
                    return True
                return PR is not None and sub is PR
        # Replace UP's metaclass — Python disallows this directly on a
        # populated class, but we can rebuild UP under the new metaclass
        # then make the native shim class point at the rebuild. That's fragile;
        # instead just use ABCMeta + register PR.
        import abc
        if not isinstance(UP, abc.ABCMeta):
            try:
                # Inject ABCMeta as UP's meta by walking Python's __class__.
                # Easier: just register via a virtual ABC sibling and alias.
                _PrimVirtual = abc.ABCMeta('_PrimVirtual', (object,), {})
                _PrimVirtual.register(UP)
                if PR is not None:
                    _PrimVirtual.register(PR)
                # Don't replace U.Prim — keep type() identity working.
                # Add a separate Usd.PrimAny if anything explicitly needs ABC.
                U.PrimAny = _PrimVirtual
            except Exception: pass
_bind_property_types_to_usd()


# ----------------------------------------------------------- prim identity
# The native shim's _UsdPrim/_PseudoRootPrim instances don't override __hash__/__eq__,
# so two `prim.GetChildren()` calls return prims that compare unequal even
# when they wrap the same /path. usdview's `_primToItemMap` is keyed by
# prim, so a second lookup misses, a *new* PrimViewItem is created, and the
# old item (already inserted into the QTreeWidget) keeps its empty
# children. Net effect: the hierarchy tree never expands past depth 1.
#
# Fix: identity by stage_id + prim path. Pixar's UsdPrim equality is
# stage+path; matching that here lets _primToItemMap.get() succeed.
def _patch_prim_identity():
    def _path_str(self):
        try:
            return str(self.GetPath())
        except Exception:
            try: return self._prim.path
            except Exception: return str(id(self))
    def _stage_id(self):
        s = getattr(self, "_stage", None) or getattr(self, "_stage_ref", None)
        return id(s) if s is not None else 0
    for nm in dir(_native_compat):
        obj = getattr(_native_compat, nm)
        if not isinstance(obj, type) or obj.__name__ not in (
                '_PseudoRootPrim', '_UsdPrim'):
            continue
        cls = obj
        cls.__hash__ = lambda self: hash((_stage_id(self), _path_str(self)))
        cls.__eq__ = lambda self, other: (
            isinstance(other, (cls, type(self))) and
            _stage_id(self) == _stage_id(other) and
            _path_str(self) == _path_str(other))
_patch_prim_identity()


def _patch_pseudo_extras():
    for nm in dir(_native_compat):
        obj = getattr(_native_compat, nm)
        # Patch both _PseudoRootPrim and _UsdPrim — the latter needs the
        # same shim methods (HasAuthoredPayloads, GetVariantSets, etc.)
        # so the prim-context menu (right-click) and other panels work.
        if isinstance(obj, type) and obj.__name__ in (
                '_PseudoRootPrim', '_UsdPrim'):
            cls = obj
            if not hasattr(cls, 'IsActive'):
                cls.IsActive = lambda self: True
            if not hasattr(cls, 'IsValid'):
                cls.IsValid = lambda self: True
            if not hasattr(cls, 'IsLoaded'):
                cls.IsLoaded = lambda self: True
            if not hasattr(cls, 'GetSpecifier'):
                cls.GetSpecifier = lambda self: 'def'
            if not hasattr(cls, 'GetTypeName'):
                cls.GetTypeName = lambda self: ''
            if not hasattr(cls, 'GetMetadata'):
                cls.GetMetadata = lambda self, key=None: None
            if not hasattr(cls, 'GetAttributes'):
                cls.GetAttributes = lambda self: []
            if not hasattr(cls, 'GetRelationships'):
                cls.GetRelationships = lambda self: []
            if not hasattr(cls, 'IsModel'):
                cls.IsModel = lambda self: False
            if not hasattr(cls, 'IsGroup'):
                cls.IsGroup = lambda self: False
            if not hasattr(cls, 'IsInstance'):
                cls.IsInstance = lambda self: False
            if not hasattr(cls, 'IsInstanceProxy'):
                cls.IsInstanceProxy = lambda self: False
            if not hasattr(cls, 'IsAbstract'):
                cls.IsAbstract = lambda self: False
            if not hasattr(cls, 'IsInPrototype'):
                cls.IsInPrototype = lambda self: False
            if not hasattr(cls, 'GetKind'):
                cls.GetKind = lambda self: ''
            if not hasattr(cls, 'GetVariantSets'):
                from types import SimpleNamespace as _SN
                cls.GetVariantSets = lambda self: _SN(GetNames=lambda: [], GetAllVariantSelections=lambda: {})
            if not hasattr(cls, 'GetVariantSet'):
                cls.GetVariantSet = lambda self, n: None
            if not hasattr(cls, 'HasVariantSets'):
                cls.HasVariantSets = lambda self: False
            if not hasattr(cls, 'HasAuthoredReferences'):
                cls.HasAuthoredReferences = lambda self: False
            if not hasattr(cls, 'GetReferences'):
                cls.GetReferences = lambda self: type('R', (), {'GetAllReferences': staticmethod(lambda: [])})()
            if not hasattr(cls, 'HasAuthoredPayloads'):
                cls.HasAuthoredPayloads = lambda self: False
            if not hasattr(cls, 'HasPayload'):
                cls.HasPayload = lambda self: False
            if not hasattr(cls, 'GetPayloads'):
                cls.GetPayloads = lambda self: type('P', (), {'GetAllPayloads': staticmethod(lambda: [])})()
            if not hasattr(cls, 'IsA'):
                cls.IsA = lambda self, _t: False
            if not hasattr(cls, 'HasAPI'):
                cls.HasAPI = lambda self, _t: False
            if not hasattr(cls, 'GetAttributes'):
                cls.GetAttributes = lambda self: []
            if not hasattr(cls, 'GetRelationships'):
                cls.GetRelationships = lambda self: []
            if not hasattr(cls, 'GetProperties'):
                cls.GetProperties = lambda self: []
            if not hasattr(cls, 'GetAttribute'):
                cls.GetAttribute = lambda self, _n: None
            if not hasattr(cls, 'GetRelationship'):
                cls.GetRelationship = lambda self, _n: None
            if not hasattr(cls, 'GetProperty'):
                cls.GetProperty = lambda self, _n: None
            if not hasattr(cls, 'GetAppliedSchemas'):
                cls.GetAppliedSchemas = lambda self: []
            if not hasattr(cls, 'GetStage'):
                cls.GetStage = lambda self: \
                    getattr(self, '_stage', None) or \
                    getattr(self, '_stage_ref', None)
            if not hasattr(cls, 'GetParent'):
                cls.GetParent = lambda self: None
_patch_pseudo_extras()


def _patch_attribute_more():
    """Tail end of attribute API surface — GetPrim, common metadata.
    NOTE: The native shim stores `_prim` as a *_MuPrim* (the inner native wrapper),
    not a _UsdPrim. _MuPrim has `.path` (string) not `.GetPath()`. Build
    Sdf.Paths from the string."""
    A = getattr(_native_compat, "_Attribute", None)
    if A is None: return
    Sdf = _native_compat.Sdf
    def _attr_path(self):
        try:
            base = getattr(self, '_prim', None)
            if base is None: return Sdf.Path('')
            base_path = getattr(base, 'path', None) or \
                        (str(base.GetPath()) if hasattr(base, 'GetPath') else '')
            return Sdf.Path(str(base_path) + '.' + getattr(self, '_name', ''))
        except Exception:
            return Sdf.Path('')
    if not hasattr(A, 'GetPath') or A.GetPath.__name__ == '<lambda>':
        A.GetPath = _attr_path
    if not hasattr(A, 'GetPrim'):
        # Note: returns the _MuPrim, callers that need _UsdPrim should
        # wrap. usdview's property tab only uses .GetPath() on the result.
        A.GetPrim = lambda self: getattr(self, '_prim', None)
    if not hasattr(A, 'GetPrimPath'):
        A.GetPrimPath = lambda self: self.GetPath().GetPrimPath()
    if not hasattr(A, 'GetStage'):
        A.GetStage = lambda self: None  # The native shim's _MuPrim doesn't carry stage
    if not hasattr(A, 'Clear'):
        A.Clear = lambda self: bool(getattr(self, '_prim', None) and
                                    self._prim.clear_attribute(getattr(self, '_name', '')))
    if not hasattr(A, 'GetNamespace'):
        A.GetNamespace = lambda self: ''
    if not hasattr(A, 'GetDisplayName'):
        A.GetDisplayName = lambda self: getattr(self, '_name', '')
    if not hasattr(A, 'GetVariability'):
        A.GetVariability = lambda self: 'varying'
    if not hasattr(A, 'IsCustom'):
        A.IsCustom = lambda self: False
    if not hasattr(A, 'IsAuthored'):
        A.IsAuthored = lambda self: True
    if not hasattr(A, 'IsHidden'):
        A.IsHidden = lambda self: False
    if not hasattr(A, 'GetBaseName'):
        A.GetBaseName = lambda self: getattr(self, '_name', '').rsplit(':', 1)[-1]
    if not hasattr(A, 'GetBracketingTimeSamples'):
        # (lower, upper, hasTimeSamples) tuple. Empty samples => (0,0,False)
        A.GetBracketingTimeSamples = lambda self, _t: (0.0, 0.0, False)
    if not hasattr(A, 'GetNumTimeSamples'):
        A.GetNumTimeSamples = lambda self: 0
    if not hasattr(A, 'ValueMightBeTimeVarying'):
        A.ValueMightBeTimeVarying = lambda self: False
    if not hasattr(A, 'GetResolveInfo'):
        # Return Default source so common.py:GetPropertyColor finds the
        # right key in statusToColor.
        class _RI:
            def GetSource(self): return 3  # Usd.ResolveInfoSourceDefault
            def HasAuthoredValueOpinion(self): return True
        A.GetResolveInfo = lambda self, _t=None, _RI=_RI: _RI()
    R = getattr(_native_compat, "_Relationship", None)
    if R is not None:
        def _rel_path(self):
            try:
                base = getattr(self, '_prim', None)
                if base is None:
                    return Sdf.Path('')
                base_path = getattr(base, 'path', None) or \
                            (str(base.GetPath()) if hasattr(base, 'GetPath') else '')
                return Sdf.Path(str(base_path) + '.' + getattr(self, '_name', ''))
            except Exception:
                return Sdf.Path('')

        if not hasattr(R, 'GetPrim'):
            R.GetPrim = lambda self: getattr(self, '_prim', None)
        if not hasattr(R, 'GetPrimPath'):
            R.GetPrimPath = lambda self: self.GetPath().GetPrimPath()
        if not hasattr(R, 'GetTargets'):
            R.GetTargets = lambda self: []
        if not hasattr(R, 'GetName'):
            R.GetName = lambda self: getattr(self, '_name', '')
        if not hasattr(R, 'GetPath') or R.GetPath.__name__ == '<lambda>':
            R.GetPath = _rel_path
        if not hasattr(R, 'IsAuthored'):
            R.IsAuthored = lambda self: True
        if not hasattr(R, 'GetMetadata'):
            R.GetMetadata = lambda self, key: None
_patch_attribute_more()


# The native shim's _Attribute.Set writes to nanousd but doesn't fire Tf.Notice.
# usdview-style observers (StageView's renderer reload, propertyView's
# refresh) won't see the change otherwise. Wrap Set() so a successful
# write broadcasts an Usd.Notice.ObjectsChanged.
def _wrap_attribute_set_for_notice():
    A = getattr(_native_compat, "_Attribute", None)
    if A is None or getattr(A, "_nu_notice_wrapped", False):
        return
    _orig_set = A.Set
    def _set_with_notice(self, value, time=None):
        ok = _orig_set(self, value, time)
        if ok:
            try:
                base = getattr(self, "_prim", None)
                name = getattr(self, "_name", "")
                base_path = getattr(base, "path", "") if base is not None else ""
                if not base_path and hasattr(base, "GetPath"):
                    base_path = str(base.GetPath())
                attr_path = (str(base_path) + "." + name) if name else str(base_path)
                # _MuPrim has no back-ref to the stage, so we send the
                # notice with sender=None — _Notice.Send fans out to all
                # registered listeners as a result.
                _Notice.Send(_ObjectsChangedNotice(
                    None, changedInfoOnlyPaths=[attr_path]
                ))
            except Exception: pass
        return ok
    A.Set = _set_with_notice
    A._nu_notice_wrapped = True

_wrap_attribute_set_for_notice()


# ----------------------------------------------------------- TimeCode.Default
# The native shim's _TimeCode.Default() returns _TimeCode(0.0) — but Pixar's
# TimeCode.Default() is a special "no time" sentinel. _Attribute.Get(time)
# in the native shim dispatches to get_time_sample whenever `time is not None`,
# which means TimeCode.Default() wrongly forces a time-sample lookup that
# returns None for non-animated attributes.
#
# Mark Default()'s instance with .is_default=True and patch _Attribute.Get
# to fall back to the default value when given a default-flagged TimeCode.
def _patch_timecode_and_attr_get():
    TC = getattr(_native_compat, "_TimeCode", None)
    A = getattr(_native_compat, "_Attribute", None)
    if TC is None or A is None: return
    if not getattr(TC, "_nanousd_openusd_copy_init", False):
        _orig_tc_init = TC.__init__

        def _tc_init(self, time=0.0):
            if isinstance(time, TC):
                _orig_tc_init(self, _tc_value(time))
                for flag in ("is_default", "is_pre_time", "is_earliest"):
                    if getattr(time, flag, False):
                        setattr(self, flag, True)
                return
            _orig_tc_init(self, time)

        TC.__init__ = _tc_init
        TC._nanousd_openusd_copy_init = True

    def _tc_value(o):
        if hasattr(o, 'time'): return float(o.time)
        try: return float(o)
        except Exception: return 0.0

    def _tc_key(o):
        if getattr(o, "is_default", False):
            return (-2, 0.0, 0)
        if getattr(o, "is_earliest", False):
            return (-1, 0.0, 0)
        return (0, _tc_value(o), 0 if getattr(o, "is_pre_time", False) else 1)

    # Comparison ops — Pixar orders TimeCode as:
    # Default < EarliestTime < numeric PreTime(value) < numeric value.
    TC.__lt__ = lambda self, o: _tc_key(self) < _tc_key(o)
    TC.__le__ = lambda self, o: _tc_key(self) <= _tc_key(o)
    TC.__gt__ = lambda self, o: _tc_key(self) > _tc_key(o)
    TC.__ge__ = lambda self, o: _tc_key(self) >= _tc_key(o)
    TC.__eq__ = lambda self, o: _tc_key(self) == _tc_key(o)
    def _tc_hash(self):
        if getattr(self, "is_default", False):
            return hash(("Usd.TimeCode", "default"))
        if getattr(self, "is_earliest", False):
            return hash(("Usd.TimeCode", "earliest"))
        return hash(("Usd.TimeCode", "pre" if getattr(self, "is_pre_time", False) else "numeric",
                     _tc_value(self)))
    TC.__hash__ = _tc_hash
    TC.GetValue = lambda self: float("nan") if getattr(self, "is_default", False) else float(self.time)
    _orig_default = TC.Default

    def _default():
        tc = _orig_default()
        try: tc.is_default = True
        except Exception: pass
        return tc

    def _pre_time(value):
        tc = TC(float(value))
        try: tc.is_pre_time = True
        except Exception: pass
        return tc

    def _earliest_time():
        tc = TC(float("-inf"))
        try: tc.is_earliest = True
        except Exception: pass
        return tc

    TC.Default = staticmethod(_default)
    TC.PreTime = staticmethod(_pre_time)
    TC.EarliestTime = staticmethod(_earliest_time)
    TC.SafeStep = staticmethod(lambda maxValue=1e6, maxCompression=10.0:
                              float(maxValue) * 1e-12)
    TC.Test_TimeCodeSequenceRoundTrip = staticmethod(lambda values: list(values))
    TC.IsDefault = lambda self: getattr(self, "is_default", False)
    TC.IsPreTime = lambda self: getattr(self, "is_pre_time", False)
    TC.IsEarliestTime = lambda self: getattr(self, "is_earliest", False)
    TC.IsNumeric = lambda self: not self.IsDefault() and not self.IsEarliestTime()

    _orig_get = A.Get
    _orig_time_samples = A.GetTimeSamples
    def _reference_offset(self):
        return _NANOUSD_REFERENCE_OFFSETS.get(getattr(self._prim, "path", ""))

    def _patched_get(self, time=None):
        is_pre_time = getattr(time, "is_pre_time", False)
        if time is not None and getattr(time, "is_default", False):
            time = None
        # If `time` is a TimeCode-shaped object (has GetValue), unwrap.
        if time is not None and hasattr(time, "GetValue"):
            tv = time.GetValue()
            # GetValue may return NaN for default → treat as None.
            if tv != tv:  # NaN check
                time = None
            else:
                time = float(tv)
        elif time is not None and not isinstance(time, (int, float)):
            try: time = float(time)
            except Exception: pass
        query_time = time
        ref_offset = _reference_offset(self)
        if ref_offset is not None and query_time is not None:
            offset, scale = ref_offset
            query_time = (float(query_time) - offset) / scale
        result = _orig_get(self, query_time)
        if result is not None and not is_pre_time:
            return result
        samples = sorted(_orig_time_samples(self))
        if query_time is not None and samples:
            if query_time in samples and not is_pre_time:
                return result
            if is_pre_time:
                previous = [t for t in samples if t < float(query_time)]
                sample = previous[-1] if previous else samples[0]
                return _orig_get(self, sample)
            if query_time <= samples[0]:
                return _orig_get(self, samples[0])
            if query_time >= samples[-1]:
                return _orig_get(self, samples[-1])
            lower = max(t for t in samples if t <= float(query_time))
            upper = min(t for t in samples if t >= float(query_time))
            lower_val = _orig_get(self, lower)
            upper_val = _orig_get(self, upper)
            if (lower != upper and isinstance(lower_val, (int, float)) and
                    isinstance(upper_val, (int, float))):
                alpha = (float(query_time) - lower) / (upper - lower)
                return lower_val + (upper_val - lower_val) * alpha
            return lower_val
        # If the time-sample lookup returned None but we have a default
        # attribute value, fall back. This happens for attributes that
        # are NOT time-sampled — get_time_sample returns None at any time
        # but get_attribute returns the default.
        if result is None and time is not None:
            try:
                result = self._prim.get_attribute(self._name)
            except Exception: pass
        return result
    A.Get = _patched_get

    def _patched_time_samples(self):
        samples = sorted(_orig_time_samples(self))
        ref_offset = _reference_offset(self)
        if ref_offset is None:
            return samples
        offset, scale = ref_offset
        return [t * scale + offset for t in samples]

    A.GetTimeSamples = _patched_time_samples

    class _ResolveInfo:
        def __init__(self, source):
            self._source = source
        def GetSource(self):
            return self._source
        def HasAuthoredValueOpinion(self):
            return self._source != getattr(_native_compat.Usd, "ResolveInfoSourceNone", 1)

    def _time_value(time):
        if time is None:
            return None
        if hasattr(time, "GetValue"):
            value = time.GetValue()
            return None if value != value else float(value)
        return float(time)

    def _interval_contains(interval, time):
        if interval is None:
            return True
        if hasattr(interval, "IsEmpty") and interval.IsEmpty():
            return False
        minimum = getattr(interval, "min", getattr(interval, "minValue", float("-inf")))
        maximum = getattr(interval, "max", getattr(interval, "maxValue", float("inf")))
        min_closed = getattr(interval, "minClosed", True)
        max_closed = getattr(interval, "maxClosed", True)
        above_min = time >= minimum if min_closed else time > minimum
        below_max = time <= maximum if max_closed else time < maximum
        return above_min and below_max

    def _time_samples_in_interval(self, interval):
        return [t for t in self.GetTimeSamples() if _interval_contains(interval, t)]

    def _bracketing_time_samples(self, time):
        samples = sorted(self.GetTimeSamples())
        if not samples:
            return ()
        value = _time_value(time)
        if value is None or value == float("-inf"):
            return (samples[0], samples[0])
        if value <= samples[0]:
            return (samples[0], samples[0])
        if value >= samples[-1]:
            return (samples[-1], samples[-1])
        lower = max(t for t in samples if t <= value)
        upper = min(t for t in samples if t >= value)
        return (lower, upper)

    def _resolve_info(self, time=None):
        if time is not None and getattr(time, "is_default", False):
            return _ResolveInfo(_native_compat.Usd.ResolveInfoSourceDefault)
        if self.GetTimeSamples():
            return _ResolveInfo(_native_compat.Usd.ResolveInfoSourceTimeSamples)
        if self.HasAuthoredValue():
            return _ResolveInfo(_native_compat.Usd.ResolveInfoSourceDefault)
        return _ResolveInfo(_native_compat.Usd.ResolveInfoSourceNone)

    def _unioned_time_samples(attrs):
        out = set()
        for attr in attrs:
            attr = getattr(attr, "_attr", attr)
            if hasattr(attr, "GetTimeSamples"):
                out.update(attr.GetTimeSamples())
        return sorted(out)

    def _unioned_time_samples_in_interval(attrs, interval):
        return [t for t in _unioned_time_samples(attrs) if _interval_contains(interval, t)]

    A.GetTimeSamplesInInterval = _time_samples_in_interval
    A.GetBracketingTimeSamples = _bracketing_time_samples
    A.GetNumTimeSamples = lambda self: len(self.GetTimeSamples())
    A.ValueMightBeTimeVarying = lambda self: self.HasTimeSamples()
    A.GetResolveInfo = _resolve_info
    A.GetInfo = lambda self, key: (
        {t: self.Get(t) for t in self.GetTimeSamples()}
        if key == "timeSamples" else None)
    A.HasInfo = lambda self, key: key == "timeSamples" and bool(self.GetTimeSamples())
    A.GetUnionedTimeSamples = staticmethod(_unioned_time_samples)
    A.GetUnionedTimeSamplesInInterval = staticmethod(_unioned_time_samples_in_interval)

    class _AttributeQuery:
        def __init__(self, attr):
            self._attr = attr
        def Get(self, time=None):
            return self._attr.Get(time)
        def GetTimeSamples(self):
            return self._attr.GetTimeSamples()

    _AttributeQuery.GetUnionedTimeSamples = staticmethod(_unioned_time_samples)
    _AttributeQuery.GetUnionedTimeSamplesInInterval = staticmethod(_unioned_time_samples_in_interval)
    _native_compat.Usd.AttributeQuery = _AttributeQuery

    Gf = getattr(_native_compat, "Gf", None)
    if Gf is not None and not hasattr(Gf, "Interval"):
        class _Interval:
            def __init__(self, minimum=None, maximum=None, minClosed=True, maxClosed=True):
                self._empty = minimum is None and maximum is None
                self.min = float("-inf") if minimum is None else float(minimum)
                self.max = float("inf") if maximum is None else float(maximum)
                self.minClosed = bool(minClosed)
                self.maxClosed = bool(maxClosed)
                if self.min > self.max:
                    self._empty = True
                if self.min == self.max and (not self.minClosed or not self.maxClosed):
                    self._empty = True
            def IsEmpty(self):
                return self._empty
            @staticmethod
            def GetFullInterval():
                return _Interval(float("-inf"), float("inf"))

        Gf.Interval = _Interval
_patch_timecode_and_attr_get()


def _patch_usd_stage_factory_compat():
    Stage = getattr(_native_compat.Usd, "Stage", None)
    if Stage is None:
        return
    _orig_create_in_memory = Stage.CreateInMemory

    def _create_in_memory(identifier=""):
        stage = _orig_create_in_memory()
        if identifier:
            try:
                stage._filepath = str(identifier)
            except Exception:
                pass
        return stage

    Stage.CreateInMemory = staticmethod(_create_in_memory)

    def _is_supported_file(identifier):
        stem = str(identifier).split(":SDF_FORMAT_ARGS:", 1)[0].lower()
        return stem.endswith((".usd", ".usda", ".usdc"))

    Stage.IsSupportedFile = staticmethod(_is_supported_file)


def _patch_stage_property_lookup_compat():
    Stage = getattr(_native_compat.Usd, "Stage", None)
    if Stage is None:
        return

    def _split_property_path(path):
        p = _native_compat.Sdf.Path(path)
        prim_path = str(p.GetPrimPath()) if p.IsPropertyPath() else str(p)
        name = p.GetName() if p.IsPropertyPath() else ""
        return prim_path, name

    def _get_attribute_at_path(self, path):
        prim_path, name = _split_property_path(path)
        if not name:
            return None
        prim = self.GetPrimAtPath(prim_path)
        return prim.GetAttribute(name) if prim else None

    def _get_relationship_at_path(self, path):
        prim_path, name = _split_property_path(path)
        if not name:
            return None
        prim = self.GetPrimAtPath(prim_path)
        return prim.GetRelationship(name) if prim else None

    def _get_property_at_path(self, path):
        attr = _get_attribute_at_path(self, path)
        if attr:
            return attr
        rel = _get_relationship_at_path(self, path)
        return rel if rel else None

    def _get_object_at_path(self, path):
        p = _native_compat.Sdf.Path(path)
        if p.IsPropertyPath():
            return _get_property_at_path(self, p)
        return self.GetPrimAtPath(p)

    Stage.GetAttributeAtPath = _get_attribute_at_path
    Stage.GetRelationshipAtPath = _get_relationship_at_path
    Stage.GetPropertyAtPath = _get_property_at_path
    Stage.GetObjectAtPath = _get_object_at_path

    Prim = getattr(_native_compat, "_UsdPrim", None)
    if Prim is not None:
        def _get_property(self, name):
            attr = self.GetAttribute(name) if hasattr(self, "GetAttribute") else None
            if attr:
                return attr
            rel = self.GetRelationship(name) if hasattr(self, "GetRelationship") else None
            return rel if rel else None

        Prim.GetProperty = _get_property

    Rel = getattr(_native_compat, "_Relationship", None)
    if Prim is not None and Rel is not None:
        _orig_create_rel = Prim.CreateRelationship
        _created_relationships = set()

        def _create_relationship(self, name, custom=True):
            _created_relationships.add((getattr(self._prim, "path", ""), str(name)))
            return _orig_create_rel(self, name, custom)

        Prim.CreateRelationship = _create_relationship
        Rel.__bool__ = lambda self: (
            self._prim.has_relationship(self._name) or
            (getattr(self._prim, "path", ""), self._name) in _created_relationships)
        Rel.IsValid = Rel.__bool__

    Refs = getattr(_native_compat, "_ReferencesProxy", None)
    if Refs is not None and not getattr(Refs, "_nanousd_accepts_layer_offset", False):
        _orig_add_internal = Refs.AddInternalReference
        def _add_internal_reference(self, path, layerOffset=None):
            if layerOffset is not None:
                _NANOUSD_REFERENCE_OFFSETS[getattr(self._prim, "path", "")] = (
                    float(getattr(layerOffset, "offset", 0.0)),
                    float(getattr(layerOffset, "scale", 1.0)) or 1.0)
            return _orig_add_internal(self, path)
        Refs.AddInternalReference = _add_internal_reference
        Refs._nanousd_accepts_layer_offset = True


def _patch_sdf_asset_path_compat():
    class _AssetPath:
        def __init__(self, *args, authoredPath=None, evaluatedPath="", resolvedPath=""):
            if len(args) > 2:
                raise TypeError("AssetPath accepts at most two positional arguments")
            if len(args) == 2 and (authoredPath is not None or evaluatedPath or resolvedPath):
                raise TypeError("resolvedPath must be positional or keyword, not both")
            if len(args) == 1 and evaluatedPath and resolvedPath:
                raise TypeError("evaluatedPath requires keyword-authoredPath")
            if len(args) == 1 and isinstance(args[0], _AssetPath):
                other = args[0]
                authoredPath = other.authoredPath
                evaluatedPath = other.evaluatedPath
                resolvedPath = other.resolvedPath
            elif len(args) >= 1:
                if authoredPath is not None:
                    raise TypeError("authoredPath specified twice")
                authoredPath = args[0]
            if len(args) == 2:
                resolvedPath = args[1]

            self.authoredPath = "" if authoredPath is None else str(authoredPath)
            self.evaluatedPath = "" if evaluatedPath is None else str(evaluatedPath)
            self.resolvedPath = "" if resolvedPath is None else str(resolvedPath)
            self.path = self.evaluatedPath or self.authoredPath

        def _category(self):
            if not self.path and not self.resolvedPath:
                return 0
            if self.authoredPath.startswith("`") and not self.evaluatedPath:
                return 1
            if self.resolvedPath:
                return 4
            if self.evaluatedPath:
                return 3
            return 2

        def _key(self):
            return (self._category(), self.path, self.resolvedPath, self.authoredPath, self.evaluatedPath)

        def __repr__(self):
            if self.evaluatedPath:
                args = [f"authoredPath={self.authoredPath!r}", f"evaluatedPath={self.evaluatedPath!r}"]
                if self.resolvedPath:
                    args.append(f"resolvedPath={self.resolvedPath!r}")
                return "Sdf.AssetPath(" + ", ".join(args) + ")"
            if self.resolvedPath:
                return f"Sdf.AssetPath({self.authoredPath!r}, resolvedPath={self.resolvedPath!r})"
            if self.authoredPath:
                return f"Sdf.AssetPath({self.authoredPath!r})"
            return "Sdf.AssetPath()"

        def __str__(self):
            return f"@{self.path}@" if self.path else ""

        def __eq__(self, other):
            if not isinstance(other, _AssetPath):
                try:
                    other = _AssetPath(other)
                except Exception:
                    return NotImplemented
            return (self.authoredPath, self.evaluatedPath, self.resolvedPath) == (
                other.authoredPath, other.evaluatedPath, other.resolvedPath)

        def __lt__(self, other):
            if not isinstance(other, _AssetPath):
                other = _AssetPath(other)
            return self._key() < other._key()

        def __le__(self, other):
            return self == other or self < other

        def __gt__(self, other):
            return not self.__le__(other)

        def __ge__(self, other):
            return not self.__lt__(other)

        def __hash__(self):
            return hash((self.authoredPath, self.evaluatedPath, self.resolvedPath))

    _native_compat.Sdf.AssetPath = _AssetPath


def _patch_sdf_listop_compat():
    def _dedupe_first(items):
        out = []
        for item in items or []:
            if item not in out:
                out.append(item)
        return out

    def _dedupe_last(items):
        out = []
        for item in reversed(list(items or [])):
            if item not in out:
                out.append(item)
        out.reverse()
        return out

    class _TypedListOp:
        def __init__(self, *, explicitItems=None, addedItems=None, prependedItems=None,
                     appendedItems=None, deletedItems=None, orderedItems=None,
                     isExplicit=False, composed_func=None):
            self._explicitItems = _dedupe_first(explicitItems)
            self._addedItems = _dedupe_first(addedItems)
            self._prependedItems = _dedupe_first(prependedItems)
            self._appendedItems = _dedupe_last(appendedItems)
            self._deletedItems = _dedupe_first(deletedItems)
            self._orderedItems = _dedupe_first(orderedItems)
            self._isExplicit = bool(isExplicit)
            self._composed_func = composed_func

        @classmethod
        def Create(cls, prependedItems=None, appendedItems=None, deletedItems=None,
                   explicitItems=None, addedItems=None, orderedItems=None):
            return cls(prependedItems=prependedItems, appendedItems=appendedItems,
                       deletedItems=deletedItems, explicitItems=explicitItems,
                       addedItems=addedItems, orderedItems=orderedItems,
                       isExplicit=explicitItems is not None)

        @classmethod
        def CreateExplicit(cls, explicitItems=None):
            return cls(explicitItems=explicitItems or [], isExplicit=True)

        @property
        def isExplicit(self):
            return self._isExplicit

        def _get(self, name):
            return list(getattr(self, "_" + name))

        def _set_first(self, name, value):
            setattr(self, "_" + name, _dedupe_first(value))
            if name == "explicitItems":
                self._isExplicit = True

        def _set_last(self, name, value):
            setattr(self, "_" + name, _dedupe_last(value))

        explicitItems = property(lambda self: self._get("explicitItems"),
                                 lambda self, value: self._set_first("explicitItems", value))
        addedItems = property(lambda self: self._get("addedItems"),
                              lambda self, value: self._set_first("addedItems", value))
        prependedItems = property(lambda self: self._get("prependedItems"),
                                  lambda self, value: self._set_first("prependedItems", value))
        appendedItems = property(lambda self: self._get("appendedItems"),
                                 lambda self, value: self._set_last("appendedItems", value))
        deletedItems = property(lambda self: self._get("deletedItems"),
                                lambda self, value: self._set_first("deletedItems", value))
        orderedItems = property(lambda self: self._get("orderedItems"),
                                lambda self, value: self._set_first("orderedItems", value))

        def _apply_ordered(self, values):
            if not self._orderedItems:
                return values
            ordered_set = set(self._orderedItems)
            buckets = {None: []}
            for item in self._orderedItems:
                buckets[item] = []
            current = None
            present_ordered = set()
            for item in values:
                if item in ordered_set:
                    current = item
                    present_ordered.add(item)
                else:
                    buckets.setdefault(current, []).append(item)
            out = list(buckets.get(None, []))
            for item in self._orderedItems:
                if item in present_ordered:
                    out.append(item)
                out.extend(buckets.get(item, []))
            return out

        def ApplyOperations(self, values):
            if isinstance(values, _TypedListOp):
                if self._addedItems or self._orderedItems:
                    return None
                if self._isExplicit:
                    return type(self).CreateExplicit(self._explicitItems)
                return type(self)(composed_func=lambda base, strong=self, weak=values:
                                  strong.ApplyOperations(weak.ApplyOperations(base)))
            if self._composed_func is not None:
                return self._composed_func(list(values))
            result = list(values)
            if self._isExplicit:
                return list(self._explicitItems)
            if self._deletedItems:
                deleted = set(self._deletedItems)
                result = [item for item in result if item not in deleted]
            for item in self._addedItems:
                if item not in result:
                    result.append(item)
            if self._appendedItems:
                app = set(self._appendedItems)
                result = [item for item in result if item not in app]
                result.extend(self._appendedItems)
            if self._prependedItems:
                prep = set(self._prependedItems)
                result = [item for item in result if item not in prep]
                result = list(self._prependedItems) + result
            result = self._apply_ordered(result)
            return result

        def GetAppliedItems(self):
            return self.ApplyOperations([])

        def GetAddedOrExplicitItems(self):
            return self.GetAppliedItems()

        def GetComposedItems(self):
            return self.GetAppliedItems()

        def HasItem(self, item):
            return any(item in values for values in (
                self._explicitItems, self._addedItems, self._prependedItems,
                self._appendedItems, self._deletedItems, self._orderedItems))

        def __eq__(self, other):
            return isinstance(other, _TypedListOp) and (
                self._isExplicit, self._explicitItems, self._addedItems,
                self._prependedItems, self._appendedItems, self._deletedItems,
                self._orderedItems) == (
                other._isExplicit, other._explicitItems, other._addedItems,
                other._prependedItems, other._appendedItems, other._deletedItems,
                other._orderedItems)

        def __hash__(self):
            return hash((self._isExplicit, tuple(self._explicitItems), tuple(self._addedItems),
                         tuple(self._prependedItems), tuple(self._appendedItems),
                         tuple(self._deletedItems), tuple(self._orderedItems)))

        def __repr__(self):
            return (f"{type(self).__name__}(isExplicit={self._isExplicit!r}, "
                    f"explicit={self._explicitItems!r}, "
                    f"added={self._addedItems!r}, "
                    f"prepended={self._prependedItems!r}, appended={self._appendedItems!r}, "
                    f"deleted={self._deletedItems!r}, ordered={self._orderedItems!r})")

        def __str__(self):
            return repr(self)

    class _IntListOp(_TypedListOp):
        pass

    class _TokenListOp(_TypedListOp):
        pass

    class _StringListOp(_TypedListOp):
        pass

    class _PathListOp(_TypedListOp):
        pass

    for name, cls in {
        "IntListOp": _IntListOp,
        "TokenListOp": _TokenListOp,
        "StringListOp": _StringListOp,
        "PathListOp": _PathListOp,
    }.items():
        setattr(_native_compat.Sdf, name, cls)


def _patch_sdf_path_small_compat():
    Path = getattr(_native_compat.Sdf, "Path", None)
    if Path is None:
        return
    native = getattr(_native_compat, "_native", None)
    native_path_validated_text = getattr(native, "sdf_path_validated_text", None)
    native_path_is_valid = getattr(native, "sdf_path_is_valid", None)

    def _split_path_elements(text):
        if not text or text == "/":
            return []
        s = str(text)
        if s.startswith("/"):
            s = s[1:]
        out = []
        cur = []
        depth = 0
        for ch in s:
            if ch == "[":
                depth += 1
            elif ch == "]" and depth:
                depth -= 1
            if ch == "/" and depth == 0:
                if cur:
                    out.append("".join(cur))
                    cur = []
                continue
            cur.append(ch)
        if cur:
            out.append("".join(cur))
        return out

    def _last_dot_outside_brackets(text):
        depth = 0
        for i in range(len(text) - 1, -1, -1):
            ch = text[i]
            if ch == "]":
                depth += 1
            elif ch == "[" and depth:
                depth -= 1
            elif ch == "." and depth == 0:
                return i
        return -1

    def _last_slash_outside_brackets(text):
        depth = 0
        for i in range(len(text) - 1, -1, -1):
            ch = text[i]
            if ch == "]":
                depth += 1
            elif ch == "[" and depth:
                depth -= 1
            elif ch == "/" and depth == 0:
                return i
        return -1

    def _first_dot_outside_brackets(text):
        depth = 0
        for i, ch in enumerate(text):
            if ch == "[":
                depth += 1
            elif ch == "]" and depth:
                depth -= 1
            elif ch == "." and depth == 0:
                return i
        return -1

    def _path_elements(text):
        s = str(text)
        elements = []
        for part in _split_path_elements(s):
            if not part:
                continue
            if part in (".", ".."):
                elements.append(part)
                continue
            dot = _first_dot_outside_brackets(part)
            if dot == -1:
                elements.append(part)
                continue
            before = part[:dot]
            rest = part[dot:]
            if before:
                elements.append(before)
            while rest:
                if rest.startswith("."):
                    depth = 0
                    stop = len(rest)
                    for i, ch in enumerate(rest[1:], start=1):
                        if ch == "[":
                            stop = i
                            break
                        if ch == "." and depth == 0:
                            stop = i
                            break
                        if ch == "]" and depth:
                            depth -= 1
                    elements.append(rest[:stop])
                    rest = rest[stop:]
                elif rest.startswith("["):
                    depth = 0
                    stop = len(rest)
                    for i, ch in enumerate(rest):
                        if ch == "[":
                            depth += 1
                        elif ch == "]":
                            depth -= 1
                            if depth == 0:
                                stop = i + 1
                                break
                    elements.append(rest[:stop])
                    rest = rest[stop:]
                else:
                    break
        return elements

    def _last_target_open(text):
        depth = 0
        for i in range(len(text) - 1, -1, -1):
            ch = text[i]
            if ch == "]":
                depth += 1
            elif ch == "[":
                if depth <= 1:
                    return i
                depth -= 1
        return -1

    def _simple_path_validity(text):
        s = str(text)
        if s in ("", "/", ".", ".."):
            return True
        if s.startswith("<") or s.endswith(">") or s.endswith("/"):
            return False
        if "$" in s or "//" in s:
            return False
        if "." in s or "[" in s or "]" in s:
            return None
        if s[0].isdigit():
            return False

        at_segment_start = True
        for ch in s[1:] if s.startswith("/") else s:
            if ch == "/":
                at_segment_start = True
                continue
            if at_segment_start:
                if ch.isdigit():
                    return False
                at_segment_start = False
            if ch == ":":
                return False
        return True

    def _looks_valid_path(text):
        s = str(text)
        if s == "":
            return True
        if s == "/":
            return True
        if s in (".", ".."):
            return True
        if s.startswith("<") or s.endswith(">") or s.endswith("/"):
            return False
        if "$" in s or "[]" in s or "//" in s:
            return False
        if s.startswith("/") and any(part.startswith(".") for part in _split_path_elements(s)):
            return False
        if s[0].isdigit():
            return False
        if s != "." and s != "..":
            for part in _split_path_elements(s):
                if part and part[0].isdigit():
                    return False
        if "[" in s or "]" in s:
            depth = 0
            for ch in s:
                if ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth < 0:
                        return False
            if depth:
                return False
            if "]/" in s:
                return False
        if _last_dot_outside_brackets(s) != -1:
            slash = _last_slash_outside_brackets(s)
            segment = s[slash + 1:]
            if segment not in (".", ".."):
                depth = 0
                dots = 0
                for ch in segment:
                    if ch == "[":
                        depth += 1
                    elif ch == "]" and depth:
                        depth -= 1
                    elif ch == "." and depth == 0:
                        dots += 1
                if dots > 2:
                    return False
                if dots > 1 and "[" not in segment:
                    return False
        last_slash = _last_slash_outside_brackets(s)
        if last_slash != -1:
            for segment in _split_path_elements(s[:last_slash]):
                if segment not in (".", "..") and _last_dot_outside_brackets(segment) != -1:
                    return False
        for segment in _split_path_elements(s):
            prim_part = segment
            dot = _last_dot_outside_brackets(prim_part)
            if dot != -1:
                prim_part = prim_part[:dot]
            bracket = prim_part.find("[")
            if bracket != -1:
                prim_part = prim_part[:bracket]
            if ":" in prim_part:
                return False
        return True

    def _validated_path_text(path_text):
        if native_path_validated_text is not None:
            return native_path_validated_text(path_text)
        quick = _simple_path_validity(path_text)
        valid = _looks_valid_path(path_text) if quick is None else quick
        return path_text if valid else ""

    def _is_valid_path_string(path):
        path_text = str(path)
        if not path_text:
            return False
        if native_path_is_valid is not None:
            return bool(native_path_is_valid(path_text))
        return _looks_valid_path(path_text)

    old_init = getattr(Path, "__init__", None)
    if old_init is not None and not getattr(Path, "_nanousd_openusd_validating_init", False):
        def _init(self, path_str=""):
            path_text = str(path_str)
            old_init(self, _validated_path_text(path_text))
        Path.__init__ = _init
        Path._nanousd_openusd_validating_init = True

    def _parent_text(text):
        s = str(text)
        if not s:
            return ""
        if s == "..":
            return "../.."
        if s.endswith("/.."):
            return s + "/.."
        dot = _last_dot_outside_brackets(s)
        open_bracket = _last_target_open(s)
        slash = _last_slash_outside_brackets(s)
        if dot > open_bracket and dot > slash:
            parent = s[:dot]
            if parent.endswith("/"):
                parent = parent[:-1]
            return parent or "."
        if s.endswith("]") and open_bracket != -1:
            return s[:open_bracket]
        if s == "/":
            return ""
        if s == ".":
            return ""
        if slash > 0:
            return s[:slash]
        if slash == 0:
            return "/"
        return "."

    def _element_count(self):
        return len(_path_elements(str(self)))

    def _name(self):
        s = str(self)
        if not s:
            return ""
        slash = _last_slash_outside_brackets(s)
        last_segment = s[slash + 1:]
        if last_segment in (".", ".."):
            return last_segment
        dot = _last_dot_outside_brackets(s)
        if dot != -1:
            return s[dot + 1:]
        if s.endswith("]"):
            return s[_last_target_open(s) + 1:-1]
        if s == "." or s == "..":
            return s
        return s.rstrip("/").rsplit("/", 1)[-1]

    def _is_property(self):
        return any(element.startswith(".") and element not in (".", "..")
                   for element in _path_elements(str(self)))

    def _prim_path(self):
        prim, _ = _prim_components_and_suffix(str(self))
        if prim and prim[-1].endswith("}") and "{" in prim[-1]:
            prim = list(prim)
            prim[-1] = prim[-1][:prim[-1].rfind("{")]
        if str(self).startswith("/"):
            return _build_absolute_path(prim)
        return Path("/".join(prim) if prim else ".")

    def _prim_or_variant_path(self):
        prim, _ = _prim_components_and_suffix(str(self))
        if str(self).startswith("/"):
            return _build_absolute_path(prim)
        return Path("/".join(prim) if prim else ".")

    def _prim_components_and_suffix(text):
        prim = []
        suffix = []
        in_suffix = False
        for element in _path_elements(str(text)):
            if (not in_suffix and
                    (element in (".", "..") or
                     (not element.startswith(".") and not element.startswith("[")))):
                if element != ".":
                    prim.append(element)
            else:
                in_suffix = True
                suffix.append(element)
        return prim, "".join(suffix)

    def _build_absolute_path(prim, suffix=""):
        return Path(("/" + "/".join(prim) if prim else "/") + suffix)

    def _make_absolute(self, anchor):
        s = str(self)
        if s.startswith("/"):
            return Path(s)
        anchor_path = Path(anchor)
        if (not str(anchor_path).startswith("/") or
                anchor_path.IsPropertyPath() or not str(anchor_path)):
            return Path("")
        prim, suffix = _prim_components_and_suffix(str(anchor_path))
        for element in _path_elements(s):
            if element == ".":
                continue
            if element == "..":
                if not prim:
                    return Path("")
                prim.pop()
                suffix = ""
            elif element.startswith(".") or element.startswith("["):
                suffix += element
            else:
                if suffix:
                    return Path("")
                prim.append(element)
        return _build_absolute_path(prim, suffix)

    def _make_relative(self, anchor):
        anchor_path = Path(anchor)
        if (not str(anchor_path).startswith("/") or
                anchor_path.IsPropertyPath() or not str(anchor_path)):
            return Path("")
        if not str(self).startswith("/"):
            absolute = _make_absolute(self, anchor_path)
            if not str(absolute):
                return Path("")
            return _make_relative(absolute, anchor_path)
        target_prim, target_suffix = _prim_components_and_suffix(str(self))
        anchor_prim, _ = _prim_components_and_suffix(str(anchor_path))
        common = 0
        for left, right in zip(target_prim, anchor_prim):
            if left != right:
                break
            common += 1
        up = [".."] * (len(anchor_prim) - common)
        down = target_prim[common:]
        parts = up + down
        if parts:
            rel = "/".join(parts)
            if target_suffix:
                rel += target_suffix if down else "/" + target_suffix
            return Path(rel)
        return Path(target_suffix or ".")

    def _has_prefix(self, prefix):
        s = str(self)
        p = str(prefix)
        if not s or not p:
            return False
        if p == "/":
            return s.startswith("/")
        return s == p or s.startswith(p + "/") or s.startswith(p + ".") or s.startswith(p + "[")

    def _replace_prefix(self, old_prefix, new_prefix, fixTargetPaths=True):
        s = str(self)
        old = str(old_prefix)
        new = str(new_prefix)
        if not s or not old or not new:
            return Path("")

        def _boundary(text, end):
            return end == len(text) or text[end] in "/.[]"

        def _replace_at(text, pos):
            if not text.startswith(old, pos):
                return None
            end = pos + len(old)
            if not _boundary(text, end):
                return None
            return text[:pos] + new + text[end:]

        out = _replace_at(s, 0) or s
        if fixTargetPaths:
            i = 1
            while i < len(out):
                if out[i - 1] == "[":
                    replaced = _replace_at(out, i)
                    if replaced is not None:
                        i += len(new)
                        out = replaced
                        continue
                i += 1
        return Path(out)

    def _from_elements(is_absolute, elements):
        if not elements:
            return Path("/" if is_absolute else ".")
        return _path_elems_to_prefixes(is_absolute, elements)[-1]

    def _prim_element_count(elements):
        return sum(1 for element in elements
                   if element in (".", "..") or
                   (not element.startswith(".") and not element.startswith("[")))

    def _remove_common_suffix(self, other, stopAtRootPrim=False):
        left_abs = str(self).startswith("/")
        right_abs = str(other).startswith("/")
        left = _path_elements(str(self))
        right = _path_elements(str(other))
        while left and right and left[-1] == right[-1]:
            removing_prim = (left[-1] in (".", "..") or
                             (not left[-1].startswith(".") and not left[-1].startswith("[")))
            if stopAtRootPrim and removing_prim and (
                    _prim_element_count(left) <= 1 or
                    _prim_element_count(right) <= 1):
                break
            left.pop()
            right.pop()
        return _from_elements(left_abs, left), _from_elements(right_abs, right)

    def _target_texts(text):
        s = str(text)
        out = []
        i = 0
        while i < len(s):
            if s[i] != "[":
                i += 1
                continue
            depth = 1
            j = i + 1
            while j < len(s) and depth:
                if s[j] == "[":
                    depth += 1
                elif s[j] == "]":
                    depth -= 1
                j += 1
            if depth == 0:
                target = s[i + 1:j - 1]
                out.append(target)
                out.extend(_target_texts(target))
                i = j
            else:
                break
        return out

    def _target_path(self):
        targets = _target_texts(str(self))
        return Path(targets[0]) if targets else Path("")

    def _all_target_paths(self):
        return [Path(target) for target in sorted(set(_target_texts(str(self))))]

    def _append_path(self, suffix):
        base = str(self)
        tail = str(suffix)
        if not base or not tail:
            raise _Tf.ErrorException("Cannot append empty Sdf.Path")
        tail = tail.lstrip("/")
        if base == "/":
            return Path("/" + tail)
        return Path(base.rstrip("/") + "/" + tail)

    def _append_target(self, target):
        if not self.IsPropertyPath() or "[" in str(self):
            return Path("")
        return Path(str(self) + "[" + str(target) + "]")

    def _append_relational_attribute(self, name):
        if not str(self).endswith("]"):
            return Path("")
        return Path(str(self) + "." + str(name))

    def _replace_name(self, name):
        s = str(self)
        if not s:
            return Path("")
        if s.endswith("]"):
            raise _Tf.ErrorException("Cannot replace the name of a target path")
        new_name = str(name)
        slash = _last_slash_outside_brackets(s)
        last_segment = s[slash + 1:]
        if last_segment == "..":
            return Path(_parent_text(s)).AppendChild(new_name)
        dot = _last_dot_outside_brackets(s)
        if dot != -1 and dot > slash:
            return Path(s[:dot + 1] + new_name)
        if slash == -1:
            return Path(new_name)
        return Path(s[:slash + 1] + new_name)

    def _get_concise_relative_paths(paths):
        paths = [Path(p) for p in paths]
        if not paths or any(not str(p).startswith("/") for p in paths):
            return paths
        prims = [_prim_components_and_suffix(str(p))[0] for p in paths]
        common = list(prims[0])
        for prim in prims[1:]:
            n = 0
            for left, right in zip(common, prim):
                if left != right:
                    break
                n += 1
            common = common[:n]
        if not common:
            return paths
        has_descendants = any(len(prim) > len(common) for prim in prims)
        out = []
        for path, prim in zip(paths, prims):
            suffix = _prim_components_and_suffix(str(path))[1]
            if prim == common and not suffix and has_descendants:
                out.append(path)
                continue
            start = len(common) if len(prim) > len(common) else max(0, len(common) - 1)
            rel_prim = prim[start:]
            rel = "/".join(rel_prim) + suffix
            out.append(Path(rel))
        return out

    def _find_prefixed_range(paths, prefix):
        prefix = Path(prefix)
        matches = [i for i, path in enumerate(paths) if Path(path).HasPrefix(prefix)]
        if not matches:
            return slice(0, 0)
        return slice(matches[0], matches[-1] + 1)

    def _find_longest_prefix(paths, path):
        path = Path(path)
        best = None
        for candidate in paths:
            candidate = Path(candidate)
            if path.HasPrefix(candidate) and (
                    best is None or candidate.pathElementCount > best.pathElementCount):
                best = candidate
        return best

    def _find_longest_strict_prefix(paths, path):
        parent = Path(path).GetParentPath()
        return _find_longest_prefix(paths, parent)

    def _path_elems_to_prefixes(is_absolute, elements, num_prefixes=None):
        if num_prefixes is None:
            limit = len(elements)
        else:
            limit = max(0, min(len(elements), int(num_prefixes) + 1))
        prim = "/" if is_absolute else ""
        prefixes = []
        for elem in list(elements)[:limit]:
            if elem in (".", ".."):
                if prim in ("", "."):
                    prim = elem
                elif prim == "/":
                    prim += elem
                else:
                    prim += "/" + elem
            elif elem.startswith("."):
                prim += elem
            elif elem.startswith("["):
                prim += elem
            elif prim in ("", "."):
                prim = elem
            elif prim == "/":
                prim += elem
            else:
                prim += "/" + elem
            prefixes.append(Path(prim))
        return prefixes

    def _prefixes(self, num_prefixes=None):
        s = str(self)
        if not s or s == "/":
            return []
        return _path_elems_to_prefixes(s.startswith("/"), _path_elements(s), num_prefixes)

    def _ancestors(self):
        out = []
        cur = str(self)
        while cur:
            if cur not in ("/", "."):
                out.append(Path(cur))
            if cur == "..":
                break
            parent = _parent_text(cur)
            if parent == cur:
                break
            if parent in ("", "/", "."):
                break
            cur = parent
        return out

    Path.isEmpty = property(lambda self: not str(self))
    Path.IsEmpty = lambda self: not str(self)
    Path.IsValidPathString = staticmethod(_is_valid_path_string)
    Path.GetParentPath = lambda self: Path(_parent_text(str(self)))
    Path.pathElementCount = property(_element_count)
    Path.name = property(_name)
    Path.GetName = lambda self: self.name
    Path.IsPropertyPath = _is_property
    Path.GetPrimPath = _prim_path
    Path.GetPrimOrPrimVariantSelectionPath = _prim_or_variant_path
    Path.GetPrefixes = _prefixes
    Path.GetAncestorsRange = _ancestors
    Path.MakeAbsolutePath = _make_absolute
    Path.MakeRelativePath = _make_relative
    Path.HasPrefix = _has_prefix
    Path.ReplacePrefix = _replace_prefix
    Path.RemoveCommonSuffix = _remove_common_suffix
    Path.targetPath = property(_target_path)
    Path.GetTargetPath = _target_path
    Path.GetAllTargetPathsRecursively = _all_target_paths
    Path.AppendPath = _append_path
    Path.AppendTarget = _append_target
    Path.AppendRelationalAttribute = _append_relational_attribute
    Path.ReplaceName = _replace_name
    Path.GetConciseRelativePaths = staticmethod(_get_concise_relative_paths)
    Path.FindPrefixedRange = staticmethod(_find_prefixed_range)
    Path.FindLongestPrefix = staticmethod(_find_longest_prefix)
    Path.FindLongestStrictPrefix = staticmethod(_find_longest_strict_prefix)
    Path.__le__ = lambda self, other: str(self) <= str(other)
    Path.__gt__ = lambda self, other: str(self) > str(other)
    Path.__ge__ = lambda self, other: str(self) >= str(other)
    def _append_child(self, name):
        if self.IsPropertyPath():
            return Path("")
        base = str(self)
        child = (base.rstrip("/") if base != "/" else "") + "/" + str(name)
        if not base.startswith("/"):
            child = child.lstrip("/")
        return Path(child)

    Path.AppendChild = _append_child
    Path.AppendProperty = lambda self, name: Path("" if self.IsPropertyPath() else str(self) + "." + str(name))

    _native_compat.Sdf._PathElemsToPrefixes = staticmethod(_path_elems_to_prefixes)
    _native_compat.Sdf._PathGetDebuggerPathText = staticmethod(lambda path: str(path))
    _native_compat.Sdf._DumpPathStats = staticmethod(lambda: None)


_patch_usd_stage_factory_compat()
_patch_stage_property_lookup_compat()
_patch_sdf_asset_path_compat()
_patch_sdf_listop_compat()
_patch_sdf_path_small_compat()


# ----------------------------------------------------------- apiSchemas via listop
def _patch_applied_schemas_via_listop():
    UP = getattr(_native_compat, "_UsdPrim", None)
    if UP is None: return
    def _applied(self):
        try:
            mu = getattr(self, "_prim", None)
            return mu.get_applied_schemas() if mu is not None else []
        except Exception:
            return []
    UP.GetAppliedSchemas = _applied
    PR = getattr(_native_compat, "_PseudoRootPrim", None)
    if PR is not None:
        PR.GetAppliedSchemas = lambda self: []
_patch_applied_schemas_via_listop()


# ----------------------------------------------------------- UsdGeom.PrimvarsAPI
# usdview's _getPropertiesDict() calls
#   UsdGeom.PrimvarsAPI(prim).FindInheritablePrimvars()
# The native shim's PrimvarsAPI (if it exists) doesn't expose this. Stub to [].
def _patch_primvars_api():
    UG = _native_compat.UsdGeom
    P = getattr(UG, "PrimvarsAPI", None)
    if P is None:
        class _Pv:
            def __init__(self, prim): self._prim = prim
            def FindInheritablePrimvars(self): return []
            def FindPrimvarWithInheritance(self, _n): return None
            def GetPrimvars(self): return []
            def HasPrimvar(self, _n): return False
        UG.PrimvarsAPI = _Pv
        return
    if not hasattr(P, "FindInheritablePrimvars"):
        P.FindInheritablePrimvars = lambda self: []
    if not hasattr(P, "FindPrimvarWithInheritance"):
        P.FindPrimvarWithInheritance = lambda self, _n: None
_patch_primvars_api()


# ----------------------------------------------------------- UsdShade.Tokens
def _patch_usdshade_tokens():
    US = _native_compat.UsdShade
    if not hasattr(US, "Tokens"):
        class _Tokens:
            full = "full"
            preview = "preview"
            allPurpose = "allPurpose"
            displacement = "displacement"
            surface = "surface"
            volume = "volume"
        US.Tokens = _Tokens
_patch_usdshade_tokens()


# ----------------------------------------------------------- inheritedTaxonomies
# rootDataModel.getResolvedLabels iterates a list that comes from
# UsdSemantics.LabelsAPI / UsdSemantics.LabelsQuery — the native shim returns None
# for inheritedTaxonomies under some paths. Defensive monkey-patch on the
# rootDataModel: but easier — patch UsdSemantics module to return [].
def _patch_usdsemantics():
    US = _sys.modules.get("pxr.UsdSemantics") or \
         _types.ModuleType("pxr.UsdSemantics")
    _sys.modules["pxr.UsdSemantics"] = US
    # Force-replace classes — the native shim's _make_namespace stub answers any
    # attribute via __getattr__ returning None, which breaks
    # `for x in inheritedTaxonomies: …` when ComputeInheritedTaxonomies()
    # returns None instead of [].
    class _LabelsQuery:
        def __init__(self, *a, **kw): pass
        def ComputeLabels(self, *a, **kw): return {}
        def ComputeUniqueInheritedLabels(self, *a, **kw): return []
    class _LabelsAPI:
        def __init__(self, prim): self._prim = prim
        def GetTaxonomies(self): return []
        def GetInheritedTaxonomies(self): return []
        @staticmethod
        def ComputeInheritedTaxonomies(_prim): return []
    US.LabelsQuery = _LabelsQuery
    US.LabelsAPI = _LabelsAPI
_patch_usdsemantics()


# ----------------------------------------------------------- Sdf.RelationshipSpec
# common.py:GetShortStringForValue does isinstance(prop, Sdf.RelationshipSpec).
# Provide a never-matching stand-in.
def _patch_sdf_relationship_spec():
    S = _native_compat.Sdf
    if not hasattr(S, "RelationshipSpec"):
        class _RelSpec: pass
        S.RelationshipSpec = _RelSpec
    if not hasattr(S, "AttributeSpec"):
        class _AttrSpec: pass
        S.AttributeSpec = _AttrSpec
    if not hasattr(S, "PrimSpec"):
        class _PrimSpec: pass
        S.PrimSpec = _PrimSpec
    if not hasattr(S, "PropertySpec"):
        class _PropSpec: pass
        S.PropertySpec = _PropSpec
    if not hasattr(S, "ValueBlock"):
        class _ValueBlock: pass
        S.ValueBlock = _ValueBlock
    # Sdf.ValueTypeName proxy returned by GetValueTypeNameForValue.
    # common.py reads .isArray and .scalarType.
    class _VTN:
        def __init__(self, scalar="token", is_array=False):
            self.scalarType = scalar; self.isArray = is_array
            self.type = scalar
        def __str__(self):
            return f"{self.scalarType}[]" if self.isArray else str(self.scalarType)
        def __repr__(self):
            return str(self)
        def __eq__(self, other):
            return str(self) == str(other)
        def __hash__(self):
            return hash(str(self))
        def __len__(self):
            return len(str(self))
        def __getitem__(self, key):
            return str(self)[key]
        def __getattr__(self, name):
            return getattr(str(self), name)
    def _as_vtn(value):
        if isinstance(value, _VTN):
            return value
        text = str(value)
        if text.endswith("[]"):
            return _VTN(text[:-2], True)
        return _VTN(text, False)
    if not hasattr(S, "ValueTypeName"):
        S.ValueTypeName = _VTN
    if not hasattr(S, "GetValueTypeNameForValue"):
        def _value_type(val):
            import numpy as _np
            # Detect numpy arrays (most common for points/widths/colors)
            if isinstance(val, _np.ndarray):
                shape = val.shape
                dtype = val.dtype
                # Multi-dim arrays of vec3 → "float3" / "double3"
                if val.ndim == 2 and shape[1] == 3:
                    sc = "double3" if dtype.kind == "f" and dtype.itemsize == 8 \
                        else "float3"
                    return _VTN(sc, True)
                if val.ndim == 2 and shape[1] == 4:
                    sc = "double4" if dtype.kind == "f" and dtype.itemsize == 8 \
                        else "float4"
                    return _VTN(sc, True)
                if val.ndim == 1:
                    if dtype.kind == "f":
                        sc = "double" if dtype.itemsize == 8 else "float"
                    elif dtype.kind == "i" or dtype.kind == "u":
                        sc = "int"
                    else:
                        sc = str(dtype.name)
                    return _VTN(sc, True)
                return _VTN(str(dtype.name), True)
            if isinstance(val, (list, tuple)):
                # Check first element to guess type
                if val and isinstance(val[0], (int,)):
                    return _VTN("int", True)
                if val and isinstance(val[0], (float,)):
                    return _VTN("float", True)
                return _VTN("token", True)
            if isinstance(val, str):
                return _VTN("string", False)
            if isinstance(val, bool):
                return _VTN("bool", False)
            if isinstance(val, int):
                return _VTN("int", False)
            if isinstance(val, float):
                return _VTN("float", False)
            return _VTN(type(val).__name__, False)
        S.GetValueTypeNameForValue = _value_type
    if not hasattr(S, "ValueTypeNames"):
        class _VTNs:
            Token = _VTN("token")
            String = _VTN("string")
            Float = _VTN("float")
            Float3 = _VTN("float3")
            Float4 = _VTN("float4")
            Int = _VTN("int")
            Bool = _VTN("bool")
            Asset = _VTN("asset")
        S.ValueTypeNames = _VTNs
    else:
        for _name, _value in list(vars(S.ValueTypeNames).items()):
            if not _name.startswith("_") and isinstance(_value, str):
                setattr(S.ValueTypeNames, _name, _as_vtn(_value))

    A = getattr(_native_compat, "_Attribute", None)
    if A is not None and not getattr(A, "_nu_value_type_wrapped", False):
        _orig_get_type_name = A.GetTypeName

        def _get_value_type_name(self):
            return _as_vtn(_orig_get_type_name(self))

        A.GetTypeName = _get_value_type_name
        A._nu_value_type_wrapped = True
_patch_sdf_relationship_spec()


# ---------------------------------------------------------------- UsdValidation
# Pixar's usdview reaches for `from pxr import UsdValidation;
# UsdValidation.ValidationRegistry().GetAllValidatorMetadata()` when the
# user opens the Validate menu. We don't ship validators — return an
# empty registry so the dialog renders with no errors and an empty list.
class _ValidationRegistry:
    def GetAllValidatorMetadata(self): return []
    def GetValidatorMetadata(self, _name): return None
    def HasValidator(self, _name): return False

class _ValidationContext:
    def __init__(self, *a, **kw): pass
    def Validate(self, *a, **kw): return []

class _ValidationTimeRange:
    def __init__(self, *a, **kw): pass

_UsdValidation = _types.ModuleType("pxr.UsdValidation")
_UsdValidation.ValidationRegistry = _ValidationRegistry
_UsdValidation.ValidationContext = _ValidationContext
_UsdValidation.ValidationTimeRange = _ValidationTimeRange
_sys.modules["pxr.UsdValidation"] = _UsdValidation


# ---------------------------------------------------------------- Tf.ScriptModuleLoader
# usdview's Python-interpreter window auto-imports all loaded pxr modules via
# Tf.ScriptModuleLoader().GetModulesDict(). The native shim's shim doesn't ship one,
# so the interpreter throws when opened. Empty dict = no auto-imports.
class _ScriptModuleLoader:
    def GetModulesDict(self): return {}
    def IsLoaded(self, _name): return False
    def Load(self, _name): pass

class _Debug:
    @staticmethod
    def GetDebugSymbolNames(): return []
    @staticmethod
    def IsDebugSymbolNameEnabled(_name): return False
    @staticmethod
    def GetDebugSymbolDescription(_name): return ""
    @staticmethod
    def SetDebugSymbolsByName(_name, _enabled): return None
    @staticmethod
    def SetOutputFile(_file): return None

_Tf.ScriptModuleLoader = _ScriptModuleLoader
_Tf.Debug = _Debug



# Gf.Vec3d/Vec3f need XAxis/YAxis/ZAxis static factories.
def _patch_gf_vec_axes():
    for cls_name in ('Vec3f', 'Vec3d'):
        cls = getattr(_Gf, cls_name, None)
        if cls is None: continue
        if callable(cls):
            # The native shim's Vec3d is a function; replace with class
            pass
        for axis_name, axis in (('XAxis', (1, 0, 0)), ('YAxis', (0, 1, 0)), ('ZAxis', (0, 0, 1))):
            if not hasattr(cls, axis_name):
                ax = axis  # capture
                if isinstance(cls, type):
                    setattr(cls, axis_name, staticmethod(lambda _ax=ax: cls(*_ax)))
                else:
                    # The native shim's Vec3d is a function — wrap it
                    pass


_patch_gf_vec_axes()


# Gf.Rotation — usdview uses Gf.Rotation(Gf.Vec3d.XAxis(), -90)
class _Rotation:
    def __init__(self, axis=None, angle=0.0):
        # axis: Vec3-like; angle: degrees
        if axis is None:
            self._axis = (1.0, 0.0, 0.0)
        else:
            self._axis = (float(axis[0]), float(axis[1]), float(axis[2]))
        self._angle = float(angle)

    def GetAxis(self):
        return _Gf.Vec3d(*self._axis) if isinstance(_Gf.Vec3d, type) else self._axis

    def GetAngle(self):
        return self._angle

    def GetQuaternion(self):
        s = _math.sin(_math.radians(self._angle) * 0.5)
        c = _math.cos(_math.radians(self._angle) * 0.5)
        return _Quat(c, self._axis[0]*s, self._axis[1]*s, self._axis[2]*s)


class _Quat:
    def __init__(self, real=1.0, i=0.0, j=0.0, k=0.0):
        self._real = float(real)
        self._imag = (float(i), float(j), float(k))
    def GetReal(self): return self._real
    def GetImaginary(self): return self._imag


if not hasattr(_Gf, 'Rotation'):
    _Gf.Rotation = _Rotation


# Vec3d/Vec3f as wrapping function: provide XAxis etc. via patch on the
# returned-from numpy ndarray pattern — easier to just add module-level helpers.
def _vec3_axis(name):
    def _f():
        if name == 'X': v = (1.0, 0.0, 0.0)
        elif name == 'Y': v = (0.0, 1.0, 0.0)
        else: v = (0.0, 0.0, 1.0)
        return _Gf.Vec3d(*v) if callable(getattr(_Gf, 'Vec3d', None)) else _np.array(v)
    return _f

# Bind XAxis/YAxis/ZAxis on the Vec3d/Vec3f callables themselves (they are
# functions; functions accept attribute assignment).
for _v in ('Vec3d', 'Vec3f'):
    _fn = getattr(_Gf, _v, None)
    if _fn is not None and not hasattr(_fn, 'XAxis'):
        try:
            _fn.XAxis = staticmethod(_vec3_axis('X'))
            _fn.YAxis = staticmethod(_vec3_axis('Y'))
            _fn.ZAxis = staticmethod(_vec3_axis('Z'))
        except Exception:
            pass




# Bulk-add Gf module utilities usdview uses at module level.
def _bulk_patch_gf_utils():
    G = _native_compat.Gf
    if not hasattr(G, 'ConvertDisplayToLinear'):
        G.ConvertDisplayToLinear = staticmethod(lambda v: v)
    if not hasattr(G, 'ConvertLinearToDisplay'):
        G.ConvertLinearToDisplay = staticmethod(lambda v: v)
    if not hasattr(G, 'Vec4f'):
        # callable that returns numpy array
        G.Vec4f = lambda *a: _np.asarray(a, dtype=_np.float32)
    if not hasattr(G, 'Vec4d'):
        G.Vec4d = lambda *a: _np.asarray(a, dtype=_np.float64)
    if not hasattr(G, 'Range3d'):
        class _Range3d:
            def __init__(self, mn=None, mx=None):
                self._min = _np.asarray(mn if mn is not None else (-1.0, -1.0, -1.0), dtype=_np.float64)
                self._max = _np.asarray(mx if mx is not None else (1.0, 1.0, 1.0), dtype=_np.float64)
            def GetMin(self): return self._min
            def GetMax(self): return self._max
            def GetMidpoint(self): return (self._min + self._max) * 0.5
            def GetSize(self): return self._max - self._min
            def IsEmpty(self): return bool(_np.any(self._max < self._min))
            def UnionWith(self, other):
                if hasattr(other, 'GetMin'):
                    self._min = _np.minimum(self._min, other.GetMin())
                    self._max = _np.maximum(self._max, other.GetMax())
                return self
            def __repr__(self):
                # Pixar's Gf.Range3d renders as
                #   "[(min.x, min.y, min.z)...(max.x, max.y, max.z)]"
                m, M = self._min, self._max
                return f"[({m[0]}, {m[1]}, {m[2]})...({M[0]}, {M[1]}, {M[2]})]"
            __str__ = __repr__
        G.Range3d = _Range3d
    if not hasattr(G, 'BBox3d'):
        class _BBox3d:
            def __init__(self, range_obj=None, matrix=None):
                self._range = range_obj or G.Range3d()
                self._matrix = matrix
            def GetRange(self): return self._range
            def GetBox(self): return self._range
            def ComputeAlignedRange(self): return self._range
            def ComputeAlignedBox(self): return self
            def GetMatrix(self): return self._matrix
            def Transform(self, m): return self
        G.BBox3d = _BBox3d
    if not hasattr(G, 'IsClose'):
        G.IsClose = staticmethod(lambda a, b, tol=1e-6: bool(_np.allclose(_np.asarray(a), _np.asarray(b), atol=tol)))


_bulk_patch_gf_utils()



# Usd.InterpolationType — usdview iterates allValues, reads .displayName
class _InterpValue:
    def __init__(self, n, i):
        self.displayName = n; self.value = i
    def __index__(self): return self.value
    def __int__(self): return self.value

class _UsdInterpType:
    Held = _InterpValue("Held", 0)
    Linear = _InterpValue("Linear", 1)
    allValues = (Held, Linear)

_native_compat.Usd.InterpolationType = _UsdInterpType



# BBoxCache method extras
def _patch_bboxcache():
    if hasattr(_native_compat, 'UsdGeom') and hasattr(_native_compat.UsdGeom, 'BBoxCache'):
        cls = _native_compat.UsdGeom.BBoxCache
        if not hasattr(cls, 'GetIncludedPurposes'):
            cls.GetIncludedPurposes = lambda self: getattr(self, 'includedPurposes', [])
        if not hasattr(cls, 'SetIncludedPurposes'):
            cls.SetIncludedPurposes = lambda self, p: setattr(self, 'includedPurposes', list(p))
        if not hasattr(cls, 'GetTime'):
            cls.GetTime = lambda self: self.time
        if not hasattr(cls, 'GetUseExtentsHint'):
            cls.GetUseExtentsHint = lambda self: getattr(self, 'useExtentsHint', False)
        if not hasattr(cls, 'SetUseExtentsHint'):
            cls.SetUseExtentsHint = lambda self, v: setattr(self, 'useExtentsHint', bool(v))
        if not hasattr(cls, 'ComputeRelativeBound'):
            cls.ComputeRelativeBound = lambda self, p1, p2: _Range3dBox()


_patch_bboxcache()

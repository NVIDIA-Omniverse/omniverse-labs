# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Copied rigid frames and the stock export's meter/kilogram/second contract.

Authoring lengths use Blender units, masses kilograms, times seconds and angles
radians. Density is kg/BU**3, inertia kg*BU**2, force kg*BU/s**2 and torque
kg*BU**2/s**2. USD output uses meters, kilograms and degrees. Bridge rotations
are xyzw; this module and Blender use wxyz. No RNA or mathutils value escapes
the evaluated snapshot boundary. Callers must snapshot the authoring state,
not a transient runtime presentation pose.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import threading


class FrameError(ValueError):
    """An input cannot be represented by the admitted rigid-frame contract."""


def _scalar(value):
    result = float(value)
    if not math.isfinite(result):
        raise FrameError("dimensional value must be finite")
    return result


def _vector(value, count, label):
    result = tuple(float(x) for x in value)
    if len(result) != count or not all(math.isfinite(x) for x in result):
        raise FrameError(f"{label} requires {count} finite values")
    return result


def normalize_quaternion(value):
    q = _vector(value, 4, "wxyz quaternion")
    norm = math.hypot(*q)
    if norm < 1e-12:
        raise FrameError("zero quaternion has no rotation")
    return tuple(x / norm for x in q)


def wxyz_to_xyzw(value):
    w, x, y, z = normalize_quaternion(value)
    return x, y, z, w


def xyzw_to_wxyz(value):
    x, y, z, w = _vector(value, 4, "xyzw quaternion")
    return normalize_quaternion((w, x, y, z))


def _multiply(a, b):
    w, x, y, z = a
    v, i, j, k = b
    return (w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
            w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v)


def _rotate(q, v):
    return _multiply(_multiply(q, (0.0, *v)), (q[0], -q[1], -q[2], -q[3]))[1:]


@dataclass(frozen=True)
class RigidFrame:
    """A scale-free column-vector transform; positions have caller-defined units."""
    position: tuple = (0.0, 0.0, 0.0)
    rotation_wxyz: tuple = (1.0, 0.0, 0.0, 0.0)

    def __post_init__(self):
        object.__setattr__(self, "position", _vector(self.position, 3, "position"))
        object.__setattr__(self, "rotation_wxyz", normalize_quaternion(self.rotation_wxyz))

    def compose(self, child):
        offset = _rotate(self.rotation_wxyz, child.position)
        return RigidFrame(tuple(a+b for a, b in zip(self.position, offset)),
                          _multiply(self.rotation_wxyz, child.rotation_wxyz))

    def inverse(self):
        w, x, y, z = self.rotation_wxyz
        q = w, -x, -y, -z
        return RigidFrame(_rotate(q, tuple(-v for v in self.position)), q)

    def column_matrix(self):
        w, x, y, z = self.rotation_wxyz
        a, b, c = self.position
        return ((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w), a),
                (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w), b),
                (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y), c),
                (0.0, 0.0, 0.0, 1.0))

    def gf_matrix(self):
        """Return copied row-vector Gf matrix values (explicit transpose)."""
        return tuple(zip(*self.column_matrix()))

    def bridge_pose(self):
        return self.position, wxyz_to_xyzw(self.rotation_wxyz)


def local_frame(body_world, anchor_world):
    return body_world.inverse().compose(anchor_world)


def owner_frame_from_body(body_world, offset):
    """Undo a body offset already converted to the body's distance units."""
    return body_world.compose(offset.inverse())


@dataclass(frozen=True)
class PhysicsUnits:
    meters_per_blender_unit: float = 1.0

    def __post_init__(self):
        value = float(self.meters_per_blender_unit)
        if not math.isfinite(value) or value <= 0:
            raise FrameError("scene unit scale must be positive and finite")
        object.__setattr__(self, "meters_per_blender_unit", value)

    @classmethod
    def from_scene(cls, scene):
        return cls(scene.unit_settings.scale_length)

    def length(self, value):
        return _scalar(value) * self.meters_per_blender_unit

    def length_from_meters(self, value):
        return _scalar(value) / self.meters_per_blender_unit

    def vector(self, value):
        return tuple(self.length(v) for v in _vector(value, 3, "length vector"))

    def frame(self, value):
        return RigidFrame(self.vector(value.position), value.rotation_wxyz)

    def frame_from_meters(self, value):
        return RigidFrame(tuple(self.length_from_meters(v) for v in value.position), value.rotation_wxyz)

    def density(self, value):
        return _scalar(value) / self.meters_per_blender_unit ** 3

    def inertia(self, value):
        return _scalar(value) * self.meters_per_blender_unit ** 2

    def force(self, value):
        return self.length(value)

    def torque(self, value):
        return self.inertia(value)

    @staticmethod
    def angle(value):
        return math.degrees(_scalar(value))

    @staticmethod
    def angular_velocity(value):
        """USD degrees/second; bridge velocity callers keep radians/second."""
        return tuple(math.degrees(v) for v in _vector(value, 3, "angular velocity"))

    def drive_gain(self, value, *, angular=False, mode="force"):
        """Stiffness/damping: kg/s² or kg/s; angular kg*BU²/(rad*s² or rad*s).

        USD angular force gains are per degree, so multiply by radians/degree.
        Acceleration interpretation is intentionally gated until importer and
        worker behavior is validated; its dimensions must not be guessed.
        Reference: https://openusd.org/release/api/class_usd_physics_drive_a_p_i.html
        """
        if mode != "force":
            raise FrameError("acceleration drive conversion requires runtime validation")
        return self.inertia(value) * math.pi / 180.0 if angular else _scalar(value)

    def drive_effort(self, value, *, angular=False):
        if value == math.inf:
            return math.inf  # USD's explicit unlimited effort sentinel.
        return self.torque(value) if angular else self.force(value)


@dataclass(frozen=True)
class EvaluatedTransform:
    frame: RigidFrame
    uniform_scale: float
    # Per-axis owner-local scale; only differs from uniform_scale when a
    # caller explicitly admitted nonuniform scale (collision-only owners).
    axis_scale: tuple = None

    def __post_init__(self):
        if self.axis_scale is None:
            object.__setattr__(self, "axis_scale", (self.uniform_scale,) * 3)


def matrix_to_transform(matrix, *, tolerance=1e-5, allow_nonuniform=False):
    """Remove admitted positive uniform scale, preserving it for geometry.

    Shear, singularity, reflection and nonuniform scale fail closed. Paired
    negative scale axes are indistinguishable from rotation in a matrix;
    evaluated_owner_transform additionally checks Blender's signed channels.
    """
    rows = tuple(_vector(row, 4, "matrix row") for row in matrix)
    if len(rows) != 4 or any(abs(rows[3][i] - (1.0 if i == 3 else 0.0)) > tolerance for i in range(4)):
        raise FrameError("expected affine 4x4 column-vector matrix")
    columns = tuple(tuple(rows[r][c] for r in range(3)) for c in range(3))
    scales = tuple(math.hypot(*column) for column in columns)
    if min(scales) < 1e-10:
        raise FrameError("singular transform")
    axes = tuple(tuple(v/s for v in column) for column, s in zip(columns, scales))
    if any(abs(sum(a*b for a, b in zip(axes[i], axes[j]))) > tolerance for i, j in ((0, 1), (0, 2), (1, 2))):
        raise FrameError("sheared transform is unsupported")
    a, b, c = axes
    determinant = (a[0]*(b[1]*c[2]-b[2]*c[1]) - a[1]*(b[0]*c[2]-b[2]*c[0]) + a[2]*(b[0]*c[1]-b[1]*c[0]))
    if determinant < 0:
        raise FrameError("negative scale/reflection is unsupported")
    if not allow_nonuniform and max(scales)-min(scales) > tolerance * max(scales):
        raise FrameError("nonuniform scale is unsupported")
    r = tuple(tuple(axes[c][i] for c in range(3)) for i in range(3))
    trace = sum(r[i][i] for i in range(3))
    if trace > 0:
        s = math.sqrt(trace+1.0)*2
        q = (s/4, (r[2][1]-r[1][2])/s, (r[0][2]-r[2][0])/s, (r[1][0]-r[0][1])/s)
    else:
        i = max(range(3), key=lambda k: r[k][k])
        j, k = (i+1) % 3, (i+2) % 3
        s = math.sqrt(1+r[i][i]-r[j][j]-r[k][k])*2
        xyz = [0.0]*3
        xyz[i], xyz[j], xyz[k] = s/4, (r[j][i]+r[i][j])/s, (r[k][i]+r[i][k])/s
        q = ((r[k][j]-r[j][k])/s, *xyz)
    return EvaluatedTransform(RigidFrame(tuple(row[3] for row in rows[:3]), q), sum(scales)/3, scales)


def matrix_to_frame(matrix):
    return matrix_to_transform(matrix).frame


def evaluated_owner_transform(obj, depsgraph=None, *, bone_name=None, offset=None, units=None, allow_nonuniform=None):
    """Snapshot an evaluated Object/bone on the main thread, returning meters.

    With no depsgraph the caller must supply an already evaluated object (also
    useful for offline numerical fixtures). FULL and NONE bone inherit-scale
    modes are admitted under positive uniform transforms; others fail closed.
    Body offsets use owner-local Blender units and inherit its uniform scale.
    """
    if threading.current_thread() is not threading.main_thread():
        raise FrameError("Blender frame evaluation requires the main thread")
    owner = obj.evaluated_get(depsgraph) if depsgraph is not None else obj
    current = owner
    while current is not None:
        if any(v <= 0 for v in getattr(current, "scale", (1, 1, 1))):
            raise FrameError("negative or zero owner/parent scale is unsupported")
        current = getattr(current, "parent", None)
    matrix = owner.matrix_world.copy()
    if bone_name is not None:
        if getattr(owner, "type", None) != "ARMATURE":
            raise FrameError("bone owner is not an armature")
        bone = owner.pose.bones.get(bone_name)
        if bone is None:
            raise FrameError(f"missing evaluated bone: {bone_name}")
        current = bone
        while current is not None:
            if current.bone.inherit_scale not in {"FULL", "NONE"}:
                raise FrameError(f"unsupported bone inherit-scale mode: {current.bone.inherit_scale}")
            if any(v <= 0 for v in current.scale):
                raise FrameError("negative or zero bone scale is unsupported")
            current = current.parent
        matrix = matrix @ bone.matrix.copy()
    if allow_nonuniform is None:
        # Objects bake nonuniform scale into colliders; bone hierarchies stay strict.
        allow_nonuniform = bone_name is None
    transform = matrix_to_transform(matrix, allow_nonuniform=allow_nonuniform)
    frame = transform.frame
    if offset is not None:
        frame = frame.compose(RigidFrame(tuple(v*s for v, s in zip(offset.position, transform.axis_scale)), offset.rotation_wxyz))
    return EvaluatedTransform((units or PhysicsUnits()).frame(frame), transform.uniform_scale, transform.axis_scale)


def evaluated_owner_frame(obj, depsgraph=None, *, bone_name=None, offset=None, units=None):
    return evaluated_owner_transform(obj, depsgraph, bone_name=bone_name, offset=offset, units=units).frame


def normalize_stock_export_units(stage, units):
    """Normalize a caller-owned *private stock export*, never a source USD stage.

    Blender's METERS exporter writes metadata 1 but leaves Blender-unit numeric
    geometry unchanged, even for scene scale .01. Scale its top-level Xforms
    before adding generated physics siblings. A layer marker makes this
    operation idempotent and rejects reuse with a different scene scale. The
    caller saves its private generation after normalization; no file is saved
    or source layer opened by this function. Physics attributes authored later
    still use PhysicsUnits (meters), not the raw stock geometry units.
    """
    from pxr import Gf, UsdGeom
    if not math.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0):
        raise FrameError("expected stock METERS export with metersPerUnit=1")
    layer = stage.GetRootLayer()
    if stage.GetEditTarget().GetLayer() != layer or layer.subLayerPaths:
        raise FrameError("unit normalization requires a private flat stock export root layer")
    marker = "ovPhysicsStockMetersPerBlenderUnit"
    metadata = dict(layer.customLayerData)
    scale = units.meters_per_blender_unit
    if marker in metadata:
        if not math.isclose(float(metadata[marker]), scale, rel_tol=1e-12):
            raise FrameError("stock export was already normalized with a different scene scale")
        return
    roots = list(stage.GetPseudoRoot().GetChildren())
    if any(not prim.IsA(UsdGeom.Xformable) for prim in roots):
        raise FrameError("stock export root must be transformable for unit normalization")
    # Blender exports an identity /World container. Scale the stock Object
    # children, not this shared container: generated meter-valued physics is
    # added later, and the fixed physics importer inherits an ancestor scale
    # even for a body carrying USD resetXformStack.
    scale_roots = []
    for prim in roots:
        if str(prim.GetPath()) == "/World" and UsdGeom.Xformable(prim).GetLocalTransformation() == Gf.Matrix4d(1):
            scale_roots.extend(child for child in prim.GetChildren() if child.IsA(UsdGeom.Xformable))
        else:
            scale_roots.append(prim)
    # Prepending the scale scales Object translation as well as its geometry.
    for prim in scale_roots:
        xform = UsdGeom.Xformable(prim)
        op = xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble, "ovSceneUnits")
        op.Set(Gf.Vec3d(scale))
        order = xform.GetOrderedXformOps()
        xform.SetXformOpOrder([op] + [old for old in order if old != op], xform.GetResetXformStack())
    metadata[marker] = scale
    layer.customLayerData = metadata

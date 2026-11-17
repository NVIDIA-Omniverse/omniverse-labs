# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PVD camera math and navigation, independent of widgets and the native reader.

Behavior follows Camera.cpp, CameraController.cpp and PVDRenderer's camera
actions. Angles are degrees here. Legacy movement retains its actual 10-times
speed factor; the controller's effective_speed reports world units per second.
"""
from dataclasses import dataclass, field
from enum import IntEnum
import math

import numpy as np


_AXIS_THRESHOLD = np.float32(.8)


def vector(value):
    result = np.asarray(value, dtype="f8")
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError("Camera coordinates must be three finite numbers")
    return result.copy()


def unit(value):
    value = vector(value)
    length = float(np.linalg.norm(value))
    if length < 1e-12:
        raise ValueError("Camera direction must have nonzero length")
    return value / length


def rotate(value, axis, degrees):
    axis = unit(axis)
    angle = math.radians(degrees)
    return (value * math.cos(angle) + np.cross(axis, value) * math.sin(angle)
            + axis * np.dot(axis, value) * (1 - math.cos(angle)))


def normalize_key(key):
    """Accept both native GLFW keys and Inspector's ImGui named key space."""
    named={512:258,513:263,514:262,515:265,516:264,517:266,518:267,
           519:268,520:269,521:260,522:261,523:259,524:32,525:257,526:256,
           527:341,528:340,529:342,530:343,531:345,532:344,533:346,534:347,
           597:44,598:45,599:46,600:47,601:59,602:61,625:333,626:334,627:335}
    if key in named:return named[key]
    if 536<=key<546:return 48+key-536
    if 546<=key<572:return 65+key-546
    if 572<=key<596:return 290+key-572
    return key


class MovementMode(IntEnum):
    FIRST_PERSON = 0
    THIRD_PERSON = 1


def near_plane_from_slider(position):
    """Original PvdSettings near-plane transform: slider 1..100, seven decades."""
    position = float(position)
    if not math.isfinite(position):
        raise ValueError("Near-plane slider position must be finite")
    position = max(1., min(100., position))
    return max(.0001, min(1000., 10. ** ((position - 1.) * (7. / 99.) - 4.)))


def near_plane_slider_position(near):
    """Locate an exact near value without quantizing or changing that value."""
    near = float(near)
    if not math.isfinite(near) or near <= 0:
        raise ValueError("Near plane must be a positive finite number")
    bounded = max(.0001, min(1000., near))
    # Qt passes this transformed value to QSlider::setValue(int), truncating
    # toward zero. Synchronization must not write the quantized value back.
    return int(max(1., min(100., 1. + (math.log10(bounded) + 4.) * (99. / 7.))))


@dataclass
class Camera:
    position: np.ndarray = field(default_factory=lambda: np.array([0., 0., 30.]))
    world_up: np.ndarray = field(default_factory=lambda: np.array([0., 1., 0.]))
    yaw: float = 0.
    pitch: float = 0.
    distance: float = 30.
    fov: float = 60.
    near: float = 1.
    tolerance_length: float = 1.
    left_handed: bool = False

    @property
    def world_basis(self):
        up = unit(self.world_up)
        forward = np.array([0., 0., -1.])
        # Qt compares its QVector3D dot product to 0.8f. A double comparison
        # can choose the opposite basis/roll for the same recorded F32 vector.
        if abs(np.float32(up[2])) > _AXIS_THRESHOLD:
            forward = np.array([0., 1., 0.])
        right = unit(np.cross(forward, up))
        return right, up, unit(np.cross(up, right))

    @property
    def basis(self):
        world_right, world_up, world_forward = self.world_basis
        direction = unit(rotate(rotate(world_forward, world_right, self.pitch), world_up, self.yaw))
        side = np.cross(direction, world_up)
        right = unit(side) if np.linalg.norm(side) > 1e-10 else world_right
        return right, unit(np.cross(right, direction)), direction

    @property
    def target(self):
        return self.position + self.basis[2] * self.distance

    def mirror(self, value):
        value = vector(value)
        if self.left_handed:
            value[2 if abs(np.float32(self.world_up[2])) < _AXIS_THRESHOLD else 0] *= -1
        return value

    def set_distance(self, value):
        if not math.isfinite(value): raise ValueError("Camera distance must be finite")
        # Camera::setOrbitDistance's actual relative guard, including its floor.
        self.distance = min(max(float(value), .003), 300000 * self.tolerance_length**2)

    def set_rotation(self, yaw, pitch):
        if not math.isfinite(yaw) or not math.isfinite(pitch):
            raise ValueError("Camera angles must be finite")
        self.yaw, self.pitch = yaw % 360., max(-88., min(88., pitch))

    def set_fov(self, value):
        if not math.isfinite(value): raise ValueError("Field of view must be finite")
        self.fov = max(1., min(179., float(value)))

    def set_world_up(self, value):
        target = self.target
        self.world_up = unit(value)
        self.position = target - self.basis[2] * self.distance

    def center_at(self, target):
        """Legacy coordinate dialog: move the orbit pivot, preserve orientation."""
        self.position = self.mirror(target) - self.basis[2] * self.distance

    def look_at(self, origin, target, up=None, *, world=True):
        origin, target = vector(origin), vector(target)
        if up is not None:
            self.world_up = unit(self.mirror(up) if world else up)
        if world:
            origin, target = self.mirror(origin), self.mirror(target)
        direction = target - origin
        self.position = origin
        length = float(np.linalg.norm(direction))
        self.set_distance(length)
        if length <= 1e-12: return
        right, world_up, forward = self.world_basis
        flat = direction - world_up * np.dot(direction, world_up)
        if np.linalg.norm(flat) > 1e-12:
            flat = unit(flat)
            self.yaw = math.degrees(math.atan2(np.dot(np.cross(forward, flat), world_up), np.dot(flat, forward)))
        unyawed = rotate(direction, world_up, -self.yaw)
        self.pitch = math.degrees(math.atan2(np.dot(np.cross(forward, unit(unyawed)), right),
                                            np.dot(unit(unyawed), forward)))

    def turn(self, yaw, pitch, *, orbit=False, pivot=None):
        target = self.target
        basis = np.asarray(self.basis) if orbit and pivot is not None else None
        self.set_rotation(self.yaw + yaw, self.pitch + pitch)
        if basis is not None:
            # Rotate the entire eye offset, not just its forward component.
            self.position = pivot + np.asarray(self.basis).T @ basis @ (self.position - pivot)
        elif orbit: self.position = target - self.basis[2] * self.distance

    def move(self, movement, *, world_up=False):
        right, up, forward = self.basis
        if world_up:
            right, up, forward = self.world_basis
            right, forward = rotate(right, up, self.yaw), rotate(forward, up, self.yaw)
        movement = vector(movement)
        self.position += right * movement[0] + up * movement[1] + forward * movement[2]

    def zoom(self, factor, *, pivot=None):
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError("Camera zoom factor must be positive")
        if pivot is not None:
            offset = self.position - pivot
            radius = float(np.linalg.norm(offset))
            if radius <= 1e-12: return
            # Already outside a limit? Allow return without snapping to it.
            radius_next = np.clip(radius * factor, min(radius, .003),
                                  max(radius, 300000 * self.tolerance_length**2))
            ratio = radius_next / radius
            self.position = pivot + offset * ratio
            self.set_distance(self.distance * ratio)
            return
        target = self.target
        self.set_distance(self.distance * factor)
        self.position = target - self.basis[2] * self.distance

    def reset(self):
        self.set_rotation(0., 0.)
        self.set_distance(30. * self.tolerance_length)
        self.center_at([0., 0., 0.])
        self.turn(0., -20., orbit=True)

    def view_matrix(self):
        right, up, forward = self.basis
        result = np.eye(4, dtype="f8")
        result[:3, :3] = [right, up, -forward]
        result[:3, 3] = -result[:3, :3] @ self.position
        if self.left_handed:
            reflection = np.eye(4)
            axis = 2 if abs(np.float32(self.world_up[2])) < _AXIS_THRESHOLD else 0
            reflection[axis, axis] = -1
            result = result @ reflection
        return result.astype("f4")

    def projection(self, width, height):
        if min(width, height) <= 0: raise ValueError("Camera viewport must have positive dimensions")
        aspect = width / height
        factor = 1. / math.tan(math.radians(self.fov) * .5)
        # Original PVD keeps the shorter dimension's FOV and uses an infinite
        # far plane. This matrix preserves its actual depth and portrait math.
        if aspect < 1: factor *= aspect
        result = np.zeros((4, 4), dtype="f4")
        result[0, 0], result[1, 1] = factor / aspect, factor
        result[2, 3], result[3, 2] = -self.near, -1.
        return result

    def pan(self, dx, dy, height):
        right, up, _ = self.basis
        scale = self.distance * 2 * math.tan(math.radians(self.fov) / 2) / max(height, 1)
        self.position += (-right * dx + up * dy) * scale

    def frame_bounds(self, bounds, *, center_only=False):
        bounds = np.asarray(bounds, dtype="f8")
        if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or (bounds[1] < bounds[0]).any(): return False
        center = (bounds[0] + bounds[1]) * .5
        eye = self.mirror(self.position)
        if center_only:
            self.look_at(eye, center)
        else:
            distance = float(np.max(bounds[1] - bounds[0])) / 1.8 * (1 / math.tan(math.radians(self.fov) / 2) + 1)
            direction = eye - center
            direction = unit(direction) if np.linalg.norm(direction) > 1e-12 else -self.mirror(self.basis[2])
            self.look_at(center + direction * max(distance, .003), center)
        return True


class CameraController:
    MOVE_KEYS = {87: (0, 0, 1), 83: (0, 0, -1), 65: (-1, 0, 0),
                 68: (1, 0, 0), 69: (0, 1, 0), 81: (0, -1, 0)}
    LOOK_KEYS = {265: (0, 1), 73: (0, 1), 264: (0, -1), 75: (0, -1),
                 262: (-1, 0), 76: (-1, 0), 263: (1, 0), 74: (1, 0)}

    def __init__(self, camera):
        self.camera = camera
        self.mode = MovementMode.FIRST_PERSON
        self.speed_scale = 1.
        self.scene_length = 0.
        self.held = set()
        self.modifiers = 0
        self.tracking_id = None
        self.tracking_offset = None
        self.captured = False
        self.available_cameras = []

    def state(self):
        return {"position":self.camera.position.tolist(),"target":self.camera.target.tolist(),
                "up":self.camera.world_up.tolist(),"yaw":self.camera.yaw,"pitch":self.camera.pitch,
                "distance":self.camera.distance,"fov":self.camera.fov,"near":self.camera.near,
                "left_handed":self.camera.left_handed,"mode":self.mode.name,
                "speed_scale":self.speed_scale,"effective_speed":self.effective_speed,
                "tracking":self.tracking_id,"captured":self.captured,
                "captured_cameras":[item[0] for item in self.available_cameras],"moving":self.moving}

    @property
    def effective_speed(self): return 10 * self.speed_scale
    @property
    def moving(self): return bool(self.held)

    def stop(self):
        self.held.clear()
        self.modifiers = 0

    def manual_change(self):
        self.tracking_id = self.tracking_offset = None

    def key(self, key, modifiers, pressed, *, active=True, editing=False):
        self.modifiers = modifiers
        if not pressed:
            self.held.discard(key)
            return key in self.MOVE_KEYS or key in self.LOOK_KEYS
        if not active or editing or modifiers & 6:
            self.stop()
            return False
        if key in self.MOVE_KEYS or key in self.LOOK_KEYS:
            self.held.add(key)
            self.manual_change()
            return True
        if key in (43, 61, 45, 333, 334):
            self.camera.set_fov(self.camera.fov + (1 if key in (45, 333) else -1))
            self.manual_change()
            return True
        return False

    def update(self, dt):
        if not math.isfinite(dt) or dt <= 0: return False
        movement = sum((np.array(self.MOVE_KEYS[key], dtype="f8") for key in self.held if key in self.MOVE_KEYS), np.zeros(3))
        look = sum((np.array(self.LOOK_KEYS[key], dtype="f8") for key in self.held if key in self.LOOK_KEYS), np.zeros(2))
        look = np.clip(look, -1, 1)
        fast = 2. if self.modifiers & 1 else 1.
        changed = False
        if np.linalg.norm(movement) > 0:
            movement = unit(movement) * self.effective_speed * fast * dt
            self.camera.move(movement, world_up=self.mode == MovementMode.THIRD_PERSON)
            if self.mode == MovementMode.FIRST_PERSON:
                self.camera.set_distance(self.camera.distance - movement[2])
            changed = True
        if np.linalg.norm(look) > 0:
            self.camera.turn(look[0] * 90 * fast * dt, look[1] * 90 * fast * dt)
            changed = True
        return changed

    def drag(self, dx, dy, width, height, button, *, pivot=None):
        if min(width, height) <= 0: return False
        self.manual_change()
        if button == 2:
            self.camera.pan(dx, dy, height)
        else:
            self.camera.turn(-dx / width * 200, -dy / height * 200, orbit=button == 0, pivot=pivot)
        return True

    def wheel(self, amount, modifiers=0, *, pivot=None):
        if not math.isfinite(amount) or not amount: return False
        sign = 1 if amount > 0 else -1
        if modifiers & 1:
            position = max(0, min(1000, int(math.log10(max(self.speed_scale, .001)) * 250 + 500 + .5)))
            self.speed_scale = 10 ** ((max(0, min(1000, position + sign * 3)) - 500) / 250)
            return True
        factor = 1 / 1.1 if sign > 0 else 1.1
        radius = self.camera.distance if pivot is None else float(np.linalg.norm(self.camera.position - pivot))
        if radius <= 1e-12: return False
        minimum = .02 * self.scene_length
        if abs(radius * factor - radius) < minimum:
            effective = radius - sign * minimum
            if effective > 0: factor = effective / radius
        self.camera.zoom(factor, pivot=pivot)
        self.manual_change()
        return True

    def calibrate(self, snapshot, scene_id=None):
        if snapshot is None: return
        mask = snapshot.kinds != 4
        if scene_id is not None: mask &= (snapshot.scene_ids == scene_id) | (snapshot.scene_ids <= 0)
        bounds = snapshot.bounds[mask]
        bounds = bounds[np.isfinite(bounds).all(axis=(1, 2))]
        sizes = np.max(bounds[:, 1] - bounds[:, 0], axis=1) if len(bounds) else []
        if len(sizes):
            self.scene_length = float(np.partition(sizes, len(sizes)//2)[len(sizes)//2])
            if self.scene_length > 0:
                self.speed_scale = self.scene_length
                self.camera.near = max(.0001, min(1., self.scene_length * .001))

    def apply_frame(self, view, adapter, *, reset=False):
        # Like Qt settings, current units affect subsequent reset/zoom guards;
        # adopting a frame must not move the camera or recalibrate near/up.
        self.camera.tolerance_length = view.tolerance_length
        if reset:
            self.stop()
            self.manual_change()
            self.captured = False
            self.camera.set_world_up([0,1,0])
            scene = next(s for s in view.scenes() if s.id == view.scene_id)
            instance = adapter.instances.get(scene.instance_id)
            if instance is not None:
                gravity = captured_gravity(view, instance)
                if gravity is not None:
                    # PVDRenderer::setCameraWorldUp converts the captured
                    # vector into the current camera's handed coordinate basis.
                    self.camera.set_world_up(self.camera.mirror(-gravity))
            self.camera.reset()
            self.calibrate(view.geometry_snapshot(), scene.instance_id)
        self.available_cameras = captured_cameras(view, adapter)
        if self.captured and self.available_cameras:
            _, origin, target, up = self.available_cameras[0]
            self.camera.look_at(origin, target, up)
        if self.tracking_id is not None:
            center = tracking_center(view, self.tracking_id, adapter)
            if center is not None:
                if self.tracking_offset is None:
                    self.tracking_offset = self.camera.mirror(self.camera.position) - center
                self.camera.look_at(center + self.tracking_offset, center)


def captured_gravity(view, instance):
    """Owned scalar Vec3 only; preserve Qt's squared near-zero threshold."""
    prop = next((p for p in view.properties(instance.class_id) if p.name.casefold() == "gravity"), None)
    if (prop is None or prop.type != "PxVec3" or prop.array
            or prop.field_count not in (3, -1)
            or not view.has_read(instance.id, prop.id, 0, 12)
            or view.property_size(instance.id, prop.id) != 12):
        return None
    gravity = np.frombuffer(view.read(instance.id, prop.id, 0, 12), dtype="<f4")
    if not np.isfinite(gravity).all():
        return None
    with np.errstate(over="ignore"):
        if float(np.dot(gravity, gravity)) < 1e-6:
            return None
    return gravity


def property_bytes(view, instance, name):
    prop = next((p for p in view.properties(instance.class_id) if p.name.casefold() == name.casefold()), None)
    if prop is None or not view.has_read(instance.id, prop.id, 0, 64): return b""
    return view.read(instance.id, prop.id, 0, 64)


def captured_cameras(view, adapter=None):
    names = {c.id: c.name for c in (adapter.classes.values() if adapter else view.classes())}
    result = []
    if "Camera" not in names.values(): return result
    instances = (adapter.instances.values() if adapter else
                 (i for start in range(0, view.info().instance_count, 10000) for i in view.instances(start, 10000)))
    for instance in instances:
        if not instance.alive or names.get(instance.class_id) != "Camera": continue
        values = [property_bytes(view, instance, name) for name in ("Origin", "Target", "Up")]
        if all(len(value) == 12 for value in values):
            origin, target, up = (np.frombuffer(value, dtype="<f4").copy() for value in values)
            if all(np.isfinite(value).all() for value in (origin, target, up)) and np.linalg.norm(up) > 0:
                result.append((instance.id, origin, target, up))
    return result


def tracking_center(view, identity, adapter=None):
    instance = adapter.instances.get(identity) if adapter else next((i for start in range(0, view.info().instance_count, 10000)
                     for i in view.instances(start, 10000) if i.id == identity and i.alive), None)
    if instance is None or not instance.alive: return None
    cls = adapter.class_name(identity) if adapter else next(c.name for c in view.classes() if c.id == instance.class_id)
    if cls != "PxRigidDynamic": return None
    sleeping = property_bytes(view, instance, "IsSleeping")
    if any(sleeping): return None
    pose = property_bytes(view, instance, "GlobalPose")
    mass = property_bytes(view, instance, "CMassLocalPose")
    if len(pose) != 28 or len(mass) != 28: return None
    pose, mass = np.frombuffer(pose, dtype="<f4"), np.frombuffer(mass, dtype="<f4")
    q, local = pose[:3], mass[4:]
    rotated = local + 2 * (pose[3] * np.cross(q, local) + np.cross(q, np.cross(q, local)))
    result = pose[4:] + rotated
    return result if np.isfinite(result).all() else None

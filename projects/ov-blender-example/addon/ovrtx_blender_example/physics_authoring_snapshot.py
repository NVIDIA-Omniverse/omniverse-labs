# SPDX-License-Identifier: Apache-2.0
"""Copied physics indicators: main-thread capture, pure geometry, owned cache.

Geometry is body-local: every stroke and label names an anchor (a body identity,
or None for world) and is expressed in that anchor's rigid frame, in Blender
units. The anchor's current world frame is drawn as a separate transform, so a
body that only moves keeps equal strokes. Links that span two anchors are Link
records, not strokes. Presentation poses are rigid meter frames tagged with
their physics generation. A caller owns IndicatorCache and refreshes it outside
draw callbacks; draw code only reads its immutable result. No Blender values,
operators, GPU resources or runtime clients enter snapshots.
"""
from dataclasses import dataclass
import math
import threading

from .physics_frames import RigidFrame, PhysicsUnits, evaluated_owner_transform
from .physics_authoring import anchor_pending, read_frame, handle_frame, inactive_animation
from .physics_endpoints import AXIS_KEYS, resolve_joint, validate_axes
from .physics_colliders import collider_records


class SnapshotError(ValueError):
    pass


@dataclass(frozen=True)
class Scope:
    scene: int
    view_layer: str
    viewport: int
    generation: int
    visibility: tuple = ()


@dataclass(frozen=True)
class PresentationPoses:
    generation: int
    # Pairs of resolved body identity and RigidFrame in meters.
    frames: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "frames", tuple((str(key), RigidFrame(frame.position, frame.rotation_wxyz)) for key, frame in self.frames))


@dataclass(frozen=True)
class Axis:
    key: str
    state: str
    lower: float
    upper: float
    driven: bool
    target: float


@dataclass(frozen=True)
class Anchor:
    """One body's current rigid world frame (BU) and how to read it again."""
    identity: str
    frame: RigidFrame
    # True: the runtime owns the pose (presented packet); False: evaluated Blender owner only.
    runtime_owned: bool = False
    owner: str = ""
    owner_uid: int = 0
    bone: str | None = None
    offset: RigidFrame = RigidFrame()  # Body offset in owner-local BU, as authored.


@dataclass(frozen=True)
class Joint:
    identity: str
    label: str
    type: str
    axis: str
    # Body origins and joint frames, each relative to its side's anchor (world when None).
    body_a: RigidFrame
    body_b: RigidFrame
    frame_a: RigidFrame
    frame_b: RigidFrame
    axes: tuple = ()
    pending: RigidFrame | None = None  # World frame of the joint handle.
    errors: tuple = ()
    enabled: bool = True
    anchor_a: str | None = None
    anchor_b: str | None = None


@dataclass(frozen=True)
class Collider:
    identity: str
    label: str
    shape: str
    frame: RigidFrame  # Relative to the anchor (world when None).
    radius: float = 0.0
    dimensions: tuple = (0.0, 0.0, 0.0)
    height: float = 0.0
    wire: tuple = ()  # Copied convex hull edge pairs in collider-local BU.
    enabled: bool = True
    error: str = ""
    anchor: str | None = None


@dataclass(frozen=True)
class Source:
    scope: Scope
    joints: tuple = ()
    colliders: tuple = ()
    size: float = 0.25
    diagnostics: tuple = ()
    # Current Anchor records; poses only, never geometry inputs.
    anchors: tuple = ()


@dataclass(frozen=True)
class Stroke:
    role: str
    identity: str
    vertices: tuple  # Consecutive pairs in local_frame, suitable for GPU LINES.
    color: tuple
    anchor: str | None = None
    local_frame: RigidFrame = RigidFrame()  # Relative to the anchor (world when None).


@dataclass(frozen=True)
class Link:
    """One segment between two anchored points: (anchor, local position) each."""
    role: str
    identity: str
    start: tuple
    end: tuple
    color: tuple


@dataclass(frozen=True)
class Label:
    role: str
    identity: str
    position: tuple  # Relative to the anchor (world when None).
    text: str
    color: tuple
    anchor: str | None = None


@dataclass(frozen=True)
class Snapshot:
    scope: Scope
    strokes: tuple
    labels: tuple
    diagnostics: tuple
    links: tuple = ()


AXIS_COLORS = ((1.0, .2, .2, 1.0), (.25, 1.0, .25, 1.0), (.25, .5, 1.0, 1.0))
GOOD = (.15, .9, .8, 1.0)
PENDING = (1.0, .6, .05, 1.0)
ERROR = (1.0, .15, .15, 1.0)
TARGET = (.8, .35, 1.0, 1.0)
LIMIT = (1.0, .9, .3, 1.0)
DISABLED = (.5, .5, .5, .8)


def _axis(index, value):
    return tuple(value if i == index else 0.0 for i in range(3))


def _circle(index, radius, lower=0.0, upper=math.tau, count=32):
    result = []
    for i in range(count+1):
        angle = lower + (upper-lower)*i/count
        point = [0.0, 0.0, 0.0]
        point[(index+1)%3], point[(index+2)%3] = radius*math.cos(angle), radius*math.sin(angle)
        result.append(tuple(point))
    return tuple(result)


def _stable_scale(scale):
    # Scales extracted from a rotating float32 matrix jitter in their last digits.
    # Local geometry uses six significant digits, so pure motion keeps it equal.
    return tuple(float("%.6g" % v) for v in scale)


def _pairs(points, close=False):
    pairs = [(points[i], points[i+1]) for i in range(len(points)-1)]
    if close:
        pairs.append((points[-1], points[0]))
    return tuple(pairs)


def geometry(source):
    """Pure numeric geometry. Shape and color both distinguish each state.

    Strokes, labels and link endpoints stay in their anchor's local frame;
    source.anchors (the poses) never enter the result.
    """
    if not math.isfinite(source.size) or source.size <= 0:
        raise SnapshotError("Indicator size must be finite and positive")
    strokes, labels, links = [], [], []
    size = source.size
    def finite(points):
        points = tuple(tuple(float(v) for v in p) for p in points)
        if any(len(p) != 3 or not all(math.isfinite(v) for v in p) for p in points):
            raise SnapshotError("Indicator geometry contains nonfinite coordinates")
        return points
    def line(role, identity, pairs, color, frame, anchor):
        vertices = finite(p for pair in pairs for p in pair)
        if vertices:
            strokes.append(Stroke(role, identity, vertices, color, anchor, frame))
    def link(role, identity, start, end, color):
        first, second = finite((start[1], end[1]))
        links.append(Link(role, identity, (start[0], first), (end[0], second), color))
    def label(role, identity, frame, text, color, anchor):
        labels.append(Label(role, identity, frame.position, text, color, anchor))
    def cross(role, identity, frame, color, extent, anchor):
        line(role, identity, tuple((_axis(i, -extent), _axis(i, extent)) for i in range(3)), color, frame, anchor)
    def diamond(role, identity, frame, color, anchor):
        line(role, identity, _pairs(((size, 0, 0), (0, size, 0), (-size, 0, 0), (0, -size, 0)), True), color, frame, anchor)
    for joint in source.joints:
        identity = joint.identity
        color = GOOD if joint.enabled else DISABLED
        anchor_a, anchor_b = joint.anchor_a, joint.anchor_b
        link("endpoint-links", identity, (anchor_a, joint.body_a.position), (anchor_a, joint.frame_a.position), color)
        link("endpoint-links", identity, (anchor_b, joint.body_b.position), (anchor_b, joint.frame_b.position), color)
        link("frame-separation", identity, (anchor_a, joint.frame_a.position), (anchor_b, joint.frame_b.position), color)
        for side, frame, anchor in (("a", joint.frame_a, anchor_a), ("b", joint.frame_b, anchor_b)):
            for index in range(3):
                line("frame-"+side+"-axis", identity, (((0, 0, 0), _axis(index, size)),), AXIS_COLORS[index], frame, anchor)
            if side == "a":
                line("frame-a-marker", identity, _pairs(((-size/3, -size/3, 0), (size/3, -size/3, 0),
                                                        (size/3, size/3, 0), (-size/3, size/3, 0)), True), color, frame, anchor)
            else:
                line("frame-b-marker", identity, _pairs(((0, size/2, 0), (-size/2, -size/2, 0), (size/2, -size/2, 0)), True), color, frame, anchor)
        axis_index = "XYZ".find(joint.axis)
        axis_index = max(0, axis_index)
        if joint.type == "REVOLUTE":
            line("hinge-glyph", identity, _pairs(_circle(axis_index, size)), color, joint.frame_a, anchor_a)
        elif joint.type == "PRISMATIC":
            line("slider-glyph", identity, ((_axis(axis_index, -size), _axis(axis_index, size)),), color, joint.frame_a, anchor_a)
            for sign in (-1, 1):
                tip = _axis(axis_index, sign*size)
                wing = list(_axis(axis_index, sign*size*.5))
                wing[(axis_index+1)%3] = size*.4
                line("slider-glyph", identity, ((tip, tuple(wing)),), color, joint.frame_a, anchor_a)
        else:
            diamond("joint-glyph", identity, joint.frame_a, color, anchor_a)
        label("joint-name", identity, joint.frame_a, joint.label + " / " + joint.type + (" (disabled)" if not joint.enabled else ""), color, anchor_a)
        for axis in joint.axes:
            index = "XYZ".index(axis.key[-1])
            angular = axis.key.startswith("rot")
            relevant = joint.type == "D6" or (joint.type == "REVOLUTE" and axis.key == "rot"+joint.axis) or (joint.type == "PRISMATIC" and axis.key == "trans"+joint.axis)
            if not relevant:
                continue
            if axis.state == "LIMITED":
                if angular:
                    points = _circle(index, size*1.5, axis.lower, axis.upper)
                    pairs = _pairs(points) + (((0, 0, 0), points[0]), ((0, 0, 0), points[-1]))
                else:
                    pairs = ((_axis(index, axis.lower), _axis(index, axis.upper)),)
                    ticks = []
                    for value in (axis.lower, axis.upper):
                        a, b = list(_axis(index, value)), list(_axis(index, value))
                        a[(index+1)%3], b[(index+1)%3] = -size/3, size/3
                        ticks.append((tuple(a), tuple(b)))
                    pairs += tuple(ticks)
                line("limit-"+axis.key, identity, pairs, LIMIT, joint.frame_a, anchor_a)
            elif axis.state == "LOCKED":
                cross("locked-"+axis.key, identity, joint.frame_a.compose(RigidFrame(_axis(index, size))), LIMIT, size/5, anchor_a)
            else:
                line("free-"+axis.key, identity, ((_axis(index, -size*2), _axis(index, size*2)),), GOOD, joint.frame_a, anchor_a)
            if axis.driven:
                point = _circle(index, size*1.7, axis.target, axis.target, 1)[0] if angular else _axis(index, axis.target)
                target = joint.frame_a.compose(RigidFrame(point))
                cross("authored-target-"+axis.key, identity, target, TARGET, size/5, anchor_a)
                label("authored-target", identity, target, "Authored target " + axis.key + " (not measured)", TARGET, anchor_a)
        if joint.pending is not None:
            # The pending handle is an unapplied authoring edit: drawn in world space.
            diamond("pending-anchor", identity, joint.pending, PENDING, None)
            link("pending-link", identity, (anchor_a, joint.frame_a.position), (None, joint.pending.position), PENDING)
            label("pending-anchor", identity, joint.pending, "Pending anchor: apply or cancel", PENDING, None)
        for error in joint.errors:
            cross("broken-link", identity, joint.frame_a, ERROR, size, anchor_a)
            label("diagnostic", identity, joint.frame_a, error, ERROR, anchor_a)
    for collider in source.colliders:
        color = GOOD if collider.enabled else DISABLED
        anchor = collider.anchor
        if collider.error:
            cross("collider-error", collider.identity, collider.frame, ERROR, size, anchor)
            label("diagnostic", collider.identity, collider.frame, collider.error, ERROR, anchor)
            continue
        if collider.shape == "ANALYTIC_SPHERE":
            pairs = tuple(pair for i in range(3) for pair in _pairs(_circle(i, collider.radius)))
        elif collider.shape == "CONVEX_HULL":
            pairs = collider.wire
        elif collider.shape == "BOX":
            x, y, z = (v/2 for v in collider.dimensions)
            corners = tuple((a*x, b*y, c*z) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1))
            pairs = tuple((corners[i], corners[j]) for i in range(8) for j in range(i+1, 8) if (i ^ j) in (1, 2, 4))
        elif collider.shape == "CAPSULE":
            # USD capsules use Z and height measures the cylindrical section.
            half = collider.height / 2
            pairs = tuple(pair for sign in (-1, 1) for pair in _pairs(tuple(
                (x, y, z + sign*half) for x, y, z in _circle(2, collider.radius))))
            for axis in (0, 1):
                for sign in (-1, 1):
                    points = []
                    for i in range(17):
                        angle = math.pi*i/16
                        point = [0.0, 0.0, sign*(half + collider.radius*math.sin(angle))]
                        point[axis] = collider.radius*math.cos(angle)
                        points.append(tuple(point))
                    pairs += _pairs(points)
                for sign in (-1, 1):
                    a, b = [0.0, 0.0, -half], [0.0, 0.0, half]
                    a[axis] = b[axis] = sign*collider.radius
                    pairs += ((tuple(a), tuple(b)),)
        else:
            cross("collider-error", collider.identity, collider.frame, ERROR, size, anchor)
            label("diagnostic", collider.identity, collider.frame, "Unsupported collider indicator: " + collider.shape, ERROR, anchor)
            continue
        line("collider-wire", collider.identity, pairs, color, collider.frame, anchor)
        if not collider.enabled:
            cross("disabled-collider", collider.identity, collider.frame, DISABLED, size/3, anchor)
            label("collider-state", collider.identity, collider.frame, "Collider disabled", DISABLED, anchor)
    return Snapshot(source.scope, tuple(strokes), tuple(labels), source.diagnostics, tuple(links))


def _same_geometry(a, b):
    # Anchors carry poses only; equal remaining inputs give an equal Snapshot.
    return (a.scope == b.scope and a.size == b.size and a.joints == b.joints
            and a.colliders == b.colliders and a.diagnostics == b.diagnostics)


class HullWireCache:
    """Local-space convex hull wires, computed once per owner/source/scale/geometry generation."""
    def __init__(self):
        self._wires = {}
        self._generations = {}

    def note_geometry_update(self, session_uid):
        # A depsgraph update reported new geometry for this hull source.
        session_uid = int(session_uid)
        self._generations[session_uid] = self._generations.get(session_uid, 0) + 1

    def wire(self, owner, source, depsgraph, scale):
        """Return (wire, error text); a failed hull is cached as its error too."""
        pair = (int(owner.session_uid), int(source.session_uid))
        key = (tuple(float(v) for v in scale), self._generations.get(pair[1], 0))
        entry = self._wires.get(pair)
        if entry is None or entry[0] != key:
            try:
                entry = (key, _hull_wire(source, depsgraph, key[0]), "")
            except (ValueError, RuntimeError, AttributeError) as exc:
                entry = (key, (), str(exc))
            self._wires[pair] = entry
        return entry[1], entry[2]

    def clear(self):
        self._wires.clear()
        self._generations.clear()


class IndicatorCache:
    """Owned by the viewport service, never a module-global simulation owner."""
    def __init__(self):
        self._entries = {}
        self._generations = {}
        self.hulls = HullWireCache()

    def refresh(self, source):
        scope = source.scope
        prior = self._generations.get(scope.scene)
        if prior is not None and scope.generation < prior:
            raise SnapshotError("Cannot publish indicators from a stale generation")
        if prior is not None and prior != scope.generation:
            self.invalidate(scene=scope.scene)
        self._generations[scope.scene] = scope.generation
        # One current visibility snapshot per scene/view-layer/viewport.
        for key in tuple(self._entries):
            if (key.scene, key.view_layer, key.viewport) == (scope.scene, scope.view_layer, scope.viewport) and key != scope:
                del self._entries[key]
        prior = self._entries.get(scope)
        if prior is not None and _same_geometry(prior[0], source):
            # A pose-only change: keep the snapshot, record the newer anchors.
            self._entries[scope] = (source, prior[1])
            return prior[1]
        self._entries.pop(scope, None)
        result = geometry(source)
        self._entries[scope] = (source, result)
        return result

    def get(self, scope):
        entry = self._entries.get(scope)
        return entry[1] if entry is not None else None

    def invalidate(self, *, scene=None):
        self.hulls.clear()
        if scene is None:
            self._entries.clear()
            self._generations.clear()
        else:
            self._entries = {k: v for k, v in self._entries.items() if k.scene != scene}
            self._generations.pop(scene, None)


def _visible(obj, view_layer, viewport, selected_only):
    if not obj.visible_get(view_layer=view_layer, viewport=viewport):
        return False
    if viewport is not None and getattr(viewport, "local_view", None) is not None and not obj.local_view_get(viewport):
        return False
    return not selected_only or obj.select_get(view_layer=view_layer)


def _hull_wire(obj, depsgraph, scale):
    import bmesh
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    bm = bmesh.new()
    try:
        for vertex in mesh.vertices:
            bm.verts.new(tuple(float(v)*k for v, k in zip(vertex.co, scale)))
        if len(bm.verts) < 4:
            raise SnapshotError("Hull source requires at least four vertices")
        result = bmesh.ops.convex_hull(bm, input=list(bm.verts), use_existing_faces=False)
        hull = set(result["geom"])
        return tuple(tuple(tuple(float(v) for v in vertex.co) for vertex in edge.verts)
                     for edge in bm.edges if edge in hull)
    finally:
        bm.free()
        evaluated.to_mesh_clear()


def anchor_world_frame(anchor, scene, depsgraph, units=None):
    """Re-read one anchor's evaluated Blender owner frame (BU): matrices only, no capture."""
    obj = scene.objects.get(anchor.owner)
    if obj is None or int(obj.session_uid) != anchor.owner_uid:
        raise SnapshotError("Indicator anchor owner is gone: " + anchor.identity)
    units = units or PhysicsUnits.from_scene(scene)
    return units.frame_from_meters(evaluated_owner_transform(
        obj, depsgraph, bone_name=anchor.bone, offset=anchor.offset, units=units).frame)


def capture(scene, view_layer, depsgraph, *, generation, viewport=None, viewport_id=0,
            selected_only=False, overlays=True, show_colliders=True, size=.25, presentation=None,
            hulls=None):
    """Copy one viewport's Blender data on the main thread, without mutations.

    Refresh after depsgraph/file/undo/selection/visibility events, never in draw.
    Missing presentation bodies are drawn with explicit error markers at their
    authoring location; consumers should pass complete admitted publications.
    Joints and colliders are local to their body anchors; Source.anchors holds
    each anchor's current world frame. A HullWireCache in `hulls` keeps hull
    wires across pose-only refreshes.
    """
    if threading.current_thread() is not threading.main_thread():
        raise SnapshotError("Physics indicator capture requires the main thread")
    if presentation is not None and presentation.generation != generation:
        raise SnapshotError("Presentation poses belong to a stale physics generation")
    poses = dict(presentation.frames) if presentation is not None else {}
    units = PhysicsUnits.from_scene(scene)
    if not viewport_id and viewport is not None:
        viewport_id = viewport.as_pointer()
    overlays = overlays and bool(getattr(getattr(viewport, "overlay", None), "show_overlays", True))
    visible = tuple(obj for obj in scene.objects if overlays and _visible(obj, view_layer, viewport, selected_only))
    scope = Scope(int(scene.session_uid), str(view_layer.name), int(viewport_id), int(generation),
                  (bool(overlays), bool(selected_only), bool(show_colliders), tuple(int(o.session_uid) for o in visible)))
    joints, colliders, diagnostics, anchors = [], [], [], {}
    def anchor(identity, frame, runtime_owned, owner, bone_name, offset):
        if identity not in anchors:
            anchors[identity] = Anchor(identity, frame, bool(runtime_owned), str(owner.name),
                                       int(owner.session_uid), bone_name, offset)
        return identity
    for obj in visible:
        settings = obj.ov.physics.joint
        if settings.persistent_id and settings.show_indicator:
            errors = list(inactive_animation(obj))
            try:
                fallback = handle_frame(obj)
            except ValueError as error:
                fallback = RigidFrame(tuple(obj.matrix_world.translation))
                errors.append(str(error))
            # Unresolved sides stay at the handle, in world space.
            body_frames = [fallback, fallback]
            joint_frames = [fallback, fallback]
            joint_anchors = [None, None]
            try:
                validate_axes(settings)
                endpoints = resolve_joint(settings, scene, depsgraph)
                for i, endpoint in enumerate(endpoints):
                    if presentation is not None and endpoint.role == "DYNAMIC" and endpoint.body_identity not in poses:
                        errors.append("Runtime body pose missing: " + endpoint.body_identity)
                    body = units.frame_from_meters(poses.get(endpoint.body_identity, endpoint.authoring_world_frame))
                    local = read_frame(settings.frame_a if i == 0 else settings.frame_b)
                    if endpoint.kind == "WORLD":
                        body_frames[i], joint_frames[i], joint_anchors[i] = body, body.compose(local), None
                    else:
                        target = endpoint.presentation_target
                        owner = (settings.endpoint_a if i == 0 else settings.endpoint_b).object
                        identity = anchor(endpoint.body_identity, body, endpoint.role == "DYNAMIC", owner,
                                          target.name if endpoint.kind == "BONE" else None,
                                          read_frame(target.ov.physics.rigid_body.offset))
                        body_frames[i], joint_frames[i], joint_anchors[i] = RigidFrame(), local, identity
                if not settings.frames_initialized:
                    errors.append("Joint frames are not initialized")
                if settings.type not in {"FIXED", "REVOLUTE", "PRISMATIC", "D6"}:
                    errors.append(settings.type + " simulation is unavailable")
            except ValueError as error:
                errors.append(str(error))
            axes = []
            for axis in settings.axes:
                values = (float(axis.lower), float(axis.upper), float(axis.drive.target_position))
                if axis.name not in AXIS_KEYS or not all(math.isfinite(value) for value in values):
                    errors.append("Invalid joint axis indicator data")
                    continue
                if axis.state == "LIMITED" and values[0] > values[1]:
                    errors.append("Invalid reversed joint limits: " + axis.name)
                    continue
                axes.append(Axis(axis.name, axis.state, *values[:2], bool(axis.drive.enabled), values[2]))
            joints.append(Joint(str(settings.persistent_id), str(obj.name), str(settings.type), str(settings.axis),
                                *body_frames, *joint_frames, tuple(axes), fallback if anchor_pending(obj) else None,
                                tuple(errors), bool(settings.enabled), *joint_anchors))
        if not show_colliders:
            continue
        owners = [(obj, None)]
        if obj.type == "ARMATURE" and obj.pose:
            owners += [(bone, bone.name) for bone in obj.pose.bones if not bone.bone.hide and (not selected_only or bone.bone.select)]
        for target, bone_name in owners:
            physics = target.ov.physics
            try:
                records = collider_records(target)
                if not any(record.schema_opinion == "APPLY" for record in records):
                    continue
                body_offset = read_frame(physics.rigid_body.offset)
                transform = evaluated_owner_transform(obj, depsgraph, bone_name=bone_name, offset=body_offset, units=units)
                identity = (obj.ov.physics.persistent_id or "object:" + str(obj.session_uid)) + ("_" + (target.bone.ov.persistent_id or "bone:" + bone_name) if bone_name else "")
                runtime_owned = (physics.rigid_body.schema_opinion == "APPLY"
                                 and physics.rigid_body.role == "DYNAMIC" and physics.rigid_body.enabled)
                frame = units.frame_from_meters(poses.get(identity, transform.frame) if runtime_owned else transform.frame)
                anchor(identity, frame, runtime_owned, obj, bone_name, body_offset)
                axis_scale = _stable_scale(transform.axis_scale)
                for index, record in enumerate(records):
                    if record.schema_opinion != "APPLY":
                        continue
                    collider_id = identity + ":" + str(getattr(record, "persistent_id", index))
                    local = read_frame(record.offset) if hasattr(record, "offset") else RigidFrame()
                    local = RigidFrame(tuple(v*k for v, k in zip(local.position, axis_scale)), local.rotation_wxyz)
                    radius = units.length_from_meters(record.radius_m)
                    dimensions = tuple(float(v)*k for v, k in zip(getattr(record, "dimensions", (0, 0, 0)), axis_scale))
                    height = units.length_from_meters(getattr(record, "height_m", 0))
                    wire, error = (), ""
                    if record.shape == "CONVEX_HULL":
                        try:
                            source = (record.mesh_source if "mesh_source" in record.bl_rna.properties and record.is_property_set("mesh_source") else None) or obj
                            source_scale = _stable_scale(evaluated_owner_transform(source, depsgraph).axis_scale)
                            if hulls is None:
                                wire = _hull_wire(source, depsgraph, source_scale)
                            else:
                                wire, error = hulls.wire(obj, source, depsgraph, source_scale)
                        except (ValueError, RuntimeError, AttributeError) as exc:
                            error = str(exc)
                    elif record.shape == "BOX":
                        if any(not math.isfinite(v) or v <= 0 for v in dimensions):
                            error = "Box dimensions must be finite and positive"
                    elif record.shape == "CAPSULE":
                        if not math.isfinite(radius) or radius <= 0:
                            error = "Capsule radius must be finite and positive"
                        elif not math.isfinite(height) or height < 0:
                            error = "Capsule cylinder height must be finite and nonnegative"
                    elif record.shape != "ANALYTIC_SPHERE":
                        error = record.shape + " collider is unavailable"
                    if record.shape == "ANALYTIC_SPHERE" and (not math.isfinite(radius) or radius <= 0):
                        error = "Sphere radius must be finite and positive"
                    if presentation is not None and runtime_owned and identity not in poses:
                        error = "Runtime body pose missing: " + identity
                    colliders.append(Collider(collider_id, str(target.name), str(record.shape), local, radius,
                        dimensions, height, wire, bool(getattr(record, "enabled", True)), error, identity))
            except (ValueError, RuntimeError) as error:
                diagnostics.append(str(target.name) + ": " + str(error))
    return Source(scope, tuple(joints), tuple(colliders), float(size), tuple(diagnostics), tuple(anchors.values()))

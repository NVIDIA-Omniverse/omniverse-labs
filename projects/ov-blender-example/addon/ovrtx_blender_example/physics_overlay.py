# SPDX-License-Identifier: Apache-2.0
"""Viewport-only resource owner; capture and GPU batch creation happen in a timer.

Draw callbacks consume prepared immutable snapshots and never contact the runtime.
This service owns no simulation state. Evaluated transforms include the existing
runtime's main-thread presentation mirror, without moving joint handles.
Depsgraph updates (every pose mirror) only request one coalesced refresh; prepared
lines stay drawable until it replaces them. Load, undo and redo clear everything.

Strokes are body-local. One LINES batch per distinct stroke (role, identity,
local vertices, color) is cached across refreshes, so motion creates no GPU
buffers: each draw multiplies the anchor's world matrix from the view's
transform table with the stroke's local frame. Depsgraph and frame-change
handlers refresh that table from evaluated matrices, without a capture. Links
between two anchors reuse one unit-segment batch with a per-draw matrix.

In OVRTX LDR redraws the display pass (viewport_display_overlay) draws the lines
right after the presented image, with the runtime-owned bodies at the poses of
that frame's presentation packet; the standalone POST_VIEW pass skips those
views and remains the fallback for every other view and shading mode. The
labels follow the lines into that pass (pixel space, same anchor matrices), and
the standalone POST_PIXEL pass skips the views whose labels it drew.
"""
from dataclasses import dataclass
import math
from pathlib import Path
import sys
import bpy
from bpy.app.handlers import persistent
from .physics_authoring_snapshot import IndicatorCache, anchor_world_frame, capture
from .physics_frames import PhysicsUnits, RigidFrame, normalize_quaternion
from . import viewport_display_overlay


@dataclass(frozen=True)
class Prepared:
    snapshot: object
    strokes: tuple  # (Stroke, cached batch, local frame Matrix) per drawn stroke.
    links: tuple
    labels: tuple
    transforms: object  # This view's TransformTable, refreshed in place.
    signature: tuple


def _key(context):
    return (context.window.as_pointer(), context.area.as_pointer(), context.region.as_pointer())


def _signature(context):
    space = context.space_data
    settings = context.scene.ov
    return (int(context.scene.session_uid), context.view_layer.name,
            space.as_pointer(), space.local_view is not None,
            space.overlay.show_overlays, settings.indicators_enabled,
            settings.indicators_selected_only, settings.indicators_colliders,
            settings.indicators_size)


def _generation(scene):
    # A read-only lookup, never demand/create/step the private runtime.
    scene_generation_sessions = sys.modules.get(__package__ + ".scene_generation_sessions")
    if scene_generation_sessions is None:
        return 0
    owner = scene_generation_sessions._owners.get(int(scene.session_uid))
    current = None if owner is None else owner.current_generation
    return 0 if current is None else int(current.number)


def _evaluating():
    # Kinematic prefetch steps transient frames; their matrices are not the shown pose.
    kinematics = sys.modules.get(__package__ + ".physics_live_kinematics")
    return kinematics is not None and kinematics.evaluating()


def _matrix(frame):
    from mathutils import Matrix
    return Matrix(frame.column_matrix())


def _stroke_key(stroke):
    # The batch inputs; anchor and local frame are draw-time transforms.
    return (stroke.role, stroke.identity, stroke.vertices, stroke.color)


def presented_frame(target, pose):
    """A presented BodyPose as its anchor's rigid world frame (BU).

    The pose mirror writes the owner as body * body_offset^-1 and the anchor is
    owner * body_offset again (capture(presentation=...) reads it the same way),
    so the anchor is the body pose itself, converted from meters.
    """
    q = pose.orient  # x, y, z, w
    body = RigidFrame(tuple(float(v) for v in pose.translate), (float(q[3]), float(q[0]), float(q[1]), float(q[2])))
    return PhysicsUnits(target.meters_per_blender_unit).frame_from_meters(body)


def presented_matrix(meters_per_blender_unit, pose):
    """presented_frame's world Matrix without intermediate frames (per-draw hot path).

    Same arithmetic as RigidFrame(...).column_matrix() of presented_frame, so the
    result is identical; runs once per body for every newly presented packet.
    """
    from mathutils import Matrix
    q = pose.orient  # x, y, z, w
    w, x, y, z = normalize_quaternion((q[3], q[0], q[1], q[2]))
    scale = float(meters_per_blender_unit)
    a, b, c = (float(v) / scale for v in pose.translate)
    return Matrix(((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w), a),
                   (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w), b),
                   (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y), c),
                   (0.0, 0.0, 0.0, 1.0)))


def _presentation_generation(sessions, scene):
    """The current generation object (read-only), or None if it cannot be identified."""
    owners = getattr(sessions, "_owners", None)
    if owners is None:
        return None
    owner = owners.get(int(scene.session_uid))
    return None if owner is None else owner.current_generation


def link_matrix(start, end):
    """Rows of the rigid+X-scale matrix mapping (0,0,0)->(1,0,0) onto start->end."""
    d = tuple(float(b)-float(a) for a, b in zip(start, end))
    length = math.sqrt(sum(v*v for v in d))
    x = tuple(v/length for v in d) if length > 1e-12 else (1.0, 0.0, 0.0)
    # Any unit Y orthogonal to X completes a right-handed rotation; the line ignores it.
    helper = (0.0, 0.0, 1.0) if abs(x[2]) < .9 else (1.0, 0.0, 0.0)
    y = (helper[1]*x[2]-helper[2]*x[1], helper[2]*x[0]-helper[0]*x[2], helper[0]*x[1]-helper[1]*x[0])
    norm = math.sqrt(sum(v*v for v in y))
    y = tuple(v/norm for v in y)
    z = (x[1]*y[2]-x[2]*y[1], x[2]*y[0]-x[0]*y[2], x[0]*y[1]-x[1]*y[0])
    return tuple((x[i]*length, y[i], z[i], float(start[i])) for i in range(3)) + ((0.0, 0.0, 0.0, 1.0),)


class TransformTable:
    """One view's anchor identity -> rigid world Matrix (BU); location and rotation only."""
    def __init__(self, scene, view_layer, anchors):
        self.scene = scene
        self.view_layer = view_layer
        self.reset(anchors)

    def reset(self, anchors):
        self.anchors = tuple(anchors)
        self.matrices = {anchor.identity: _matrix(anchor.frame) for anchor in self.anchors}

    def update(self, scene, depsgraph):
        """Re-read every anchor from evaluated matrices; returns True if any changed."""
        units = PhysicsUnits.from_scene(scene)
        changed = False
        for anchor in self.anchors:
            try:
                matrix = _matrix(anchor_world_frame(anchor, scene, depsgraph, units))
            except (ReferenceError, RuntimeError, ValueError, AttributeError):
                continue  # The next coalesced refresh recaptures a removed or invalid owner.
            if matrix != self.matrices.get(anchor.identity):
                self.matrices[anchor.identity] = matrix
                changed = True
        return changed


class OverlayService:
    def __init__(self):
        self.cache = IndicatorCache()
        self.prepared = {}
        self.batches = {}
        self.unit_link = None
        self.handles = []
        self.shader = None
        self.font = None
        self.font_path = str(Path(bpy.utils.system_resource("DATAFILES", path="fonts")) / "Inter.woff2")
        self.last_error = ""
        self.running = False
        self.timer = self.refresh
        self.dirty = False
        self.pending = False
        self.pending_timer = self._refresh_pending
        self.packet = None  # (presentation packet, anchor -> Matrix) for the last packet drawn.
        # (generation, ((body path, anchor identity, meters per BU), ...)) of its
        # enabled dynamic presentation targets; a generation's targets never change.
        self.targets = None

    def clear(self):
        self.prepared.clear()
        self.batches.clear()
        self.packet = None
        self.targets = None
        self.cache.invalidate()

    def packet_transforms(self, scene, packet):
        """Anchor identity -> world Matrix of the runtime-owned bodies a packet presents.

        packet is the engine's (frame, identity, pending, applied, initial); the
        poses are the presented frame's tick poses, mapped through the owning
        generation's presentation targets. Converted once per packet.
        """
        if packet is None:
            return None
        cached = self.packet
        if cached is not None and cached[0] is packet:
            return cached[1]
        tick = getattr(packet[0], "runtime_tick_result", None)
        poses = {pose.prim_path: pose for pose in (getattr(tick, "physics_pose_set", None) or ())}
        sessions = sys.modules.get(__package__ + ".scene_generation_sessions")
        matrices = {}
        if poses and sessions is not None:
            for path, identity, scale in self._dynamic_targets(sessions, scene):
                pose = poses.get(path)
                if pose is not None:
                    matrices[identity] = presented_matrix(scale, pose)
        self.packet = (packet, matrices)
        return matrices

    def _dynamic_targets(self, sessions, scene):
        # Building presentation targets walks every opinion record of the
        # generation, so they are read once per generation, not per packet.
        generation = _presentation_generation(sessions, scene)
        cached = self.targets
        if generation is not None and cached is not None and cached[0] is generation:
            return cached[1]
        targets = tuple((target.body_path, target.body_identity, target.meters_per_blender_unit)
                        for target in sessions.current_presentation_context(scene)[2]
                        if target.enabled and target.role == "DYNAMIC")
        self.targets = None if generation is None else (generation, targets)
        return targets

    def request_refresh(self):
        # Runs inside depsgraph handlers: never capture here. Mark dirty and register
        # at most one zero-interval refresh; later updates before it runs register nothing.
        self.dirty = True
        if not self.running or self.pending:
            return
        self.pending = True
        # Persistent, so a file load cannot drop it and leave `pending` stuck.
        bpy.app.timers.register(self.pending_timer, first_interval=0.0, persistent=True)

    def note_geometry_updates(self, depsgraph):
        # Hull wires are rebuilt only for sources whose geometry the depsgraph updated.
        for update in tuple(getattr(depsgraph, "updates", ())):
            if not update.is_updated_geometry:
                continue
            data = update.id
            if getattr(data, "id_type", None) == "OBJECT":
                self.cache.hulls.note_geometry_update(getattr(data, "original", data).session_uid)

    def update_transforms(self, scene, depsgraph):
        """Matrix reads only: no capture and no batches."""
        if not self.running or not self.prepared or depsgraph is None or _evaluating():
            return
        layer = getattr(getattr(depsgraph, "view_layer", None), "name", None)
        uid = int(scene.session_uid)
        for entry in tuple(self.prepared.values()):
            table = entry.transforms
            if table.scene == uid and table.view_layer == layer:
                table.update(scene, depsgraph)

    def _refresh_pending(self):
        try:
            # The periodic timer may already have consumed this request.
            if self.dirty:
                self.refresh()
        finally:
            self.pending = False
        return None

    def start(self):
        if self.running or bpy.app.background:
            return
        import blf
        self.font = blf.load(self.font_path)
        if self.font < 0:
            raise RuntimeError("Physics indicators could not load their private font")
        try:
            for callback, stage in ((self.draw_lines, "POST_VIEW"), (self.draw_labels, "POST_PIXEL")):
                self.handles.append(bpy.types.SpaceView3D.draw_handler_add(callback, (), "WINDOW", stage))
            self.running = True
            bpy.app.timers.register(self.timer, first_interval=0.1, persistent=True)
        except Exception:
            self.stop()
            raise

    def stop(self):
        self.running = False
        for timer in (self.timer, self.pending_timer):
            if bpy.app.timers.is_registered(timer):
                bpy.app.timers.unregister(timer)
        self.pending = False
        for handle in self.handles:
            bpy.types.SpaceView3D.draw_handler_remove(handle, "WINDOW")
        self.handles.clear()
        self.clear()
        self.unit_link = None
        self.shader = None
        if self.font is not None:
            import blf
            blf.unload(self.font_path)
            self.font = None

    def _strokes(self, snapshot, batch_for_shader):
        # Only strokes whose local inputs are new get a batch.
        result = []
        for stroke in snapshot.strokes:
            key = _stroke_key(stroke)
            batch = self.batches.get(key)
            if batch is None:
                batch = self.batches[key] = batch_for_shader(self.shader, "LINES", {"pos": stroke.vertices})
            result.append((stroke, batch, _matrix(stroke.local_frame)))
        return tuple(result)

    def refresh(self):
        if not self.running:
            return None
        # Registration and file-load timers need not have a GPU context.
        # Wait for a viewport draw to create the shader before preparing batches.
        if self.shader is None:
            return 0.1
        self.dirty = False
        from gpu_extras.batch import batch_for_shader
        if self.unit_link is None:
            self.unit_link = batch_for_shader(self.shader, "LINES", {"pos": ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))})
        live = set()
        for window in tuple(bpy.context.window_manager.windows):
            for area in tuple(window.screen.areas):
                if area.type != "VIEW_3D":
                    continue
                for region in tuple(area.regions):
                    if region.type != "WINDOW":
                        continue
                    key = (window.as_pointer(), area.as_pointer(), region.as_pointer())
                    live.add(key)
                    try:
                        with bpy.context.temp_override(window=window, area=area, region=region):
                            context = bpy.context
                            settings = context.scene.ov
                            if not settings.indicators_enabled or not context.space_data.overlay.show_overlays:
                                self.prepared.pop(key, None)
                                continue
                            source = capture(context.scene, context.view_layer,
                                             context.evaluated_depsgraph_get(),
                                             generation=_generation(context.scene),
                                             viewport=context.space_data, viewport_id=region.as_pointer(),
                                             selected_only=settings.indicators_selected_only,
                                             show_colliders=settings.indicators_colliders,
                                             size=settings.indicators_size, hulls=self.cache.hulls)
                            snapshot = self.cache.refresh(source)
                            signature = _signature(context)
                            prior = self.prepared.get(key)
                            if prior is None or prior.snapshot is not snapshot or prior.signature != signature:
                                table = TransformTable(int(context.scene.session_uid), context.view_layer.name, source.anchors)
                                self.prepared[key] = Prepared(snapshot, self._strokes(snapshot, batch_for_shader),
                                                              snapshot.links, snapshot.labels, table, signature)
                                area.tag_redraw()
                            elif prior.transforms.anchors != source.anchors:
                                # Pose only: keep every batch, take the captured matrices.
                                prior.transforms.reset(source.anchors)
                                area.tag_redraw()
                    except (ReferenceError, RuntimeError, ValueError, AttributeError) as exc:
                        self.prepared.pop(key, None)
                        self.last_error = str(exc)
        for key in tuple(self.prepared):
            if key not in live:
                del self.prepared[key]
                # Retain no copied scene geometry for a closed viewport.
                self.cache.invalidate()
        # Drop batches no prepared view draws any more.
        used = {_stroke_key(stroke) for entry in self.prepared.values() for stroke, _batch, _local in entry.strokes}
        for key in tuple(self.batches):
            if key not in used:
                del self.batches[key]
        return 0.1

    def current(self, context=None):
        context = context or bpy.context
        if not self.running or context.area is None or context.area.type != "VIEW_3D" or context.region is None:
            return None
        entry = self.prepared.get(_key(context))
        # Suppress immediately when scope/settings change before the timer runs.
        return entry if entry is not None and entry.signature == _signature(context) else None

    @staticmethod
    def _table(entry, transforms):
        """Anchor identity -> world Matrix for one draw; lines and labels share it."""
        table = entry.transforms.matrices
        return {**table, **transforms} if transforms else table

    def draw(self, context, transforms=None):
        """Draw one view's lines on the current view/projection stack; returns True if drawn.

        transforms optionally overrides anchor identity -> world Matrix (e.g. the
        presented packet's poses); other anchors use the view's transform table.
        """
        if self.running and self.shader is None:
            import gpu
            self.shader = gpu.shader.from_builtin("POLYLINE_UNIFORM_COLOR")
        entry = self.current(context)
        if entry is None:
            return False
        import gpu
        from mathutils import Matrix, Vector
        table = self._table(entry, transforms)
        shader = self.shader
        state = gpu.state
        prior = (state.blend_get(), state.depth_test_get(), state.depth_mask_get())
        try:
            state.blend_set("ALPHA")
            state.depth_test_set("NONE" if context.space_data.shading.type == "RENDERED" else "LESS_EQUAL")
            state.depth_mask_set(False)
            shader.bind()
            shader.uniform_float("viewportSize", state.viewport_get()[2:])
            shader.uniform_float("lineWidth", 2.0)
            for stroke, batch, local in entry.strokes:
                if stroke.anchor is None:
                    model = local
                else:
                    world = table.get(stroke.anchor)
                    if world is None:
                        continue
                    model = world @ local
                with gpu.matrix.push_pop():
                    gpu.matrix.multiply_matrix(model)
                    shader.uniform_float("color", stroke.color)
                    batch.draw(shader)
            if self.unit_link is not None:
                for link in entry.links:
                    points = []
                    for anchor, point in (link.start, link.end):
                        world = None if anchor is None else table.get(anchor)
                        points.append(Vector(point) if anchor is None else None if world is None else world @ Vector(point))
                    if None in points:
                        continue
                    with gpu.matrix.push_pop():
                        gpu.matrix.multiply_matrix(Matrix(link_matrix(*points)))
                        shader.uniform_float("color", link.color)
                        self.unit_link.draw(shader)
        finally:
            gpu.shader.unbind()
            state.blend_set(prior[0])
            state.depth_test_set(prior[1])
            state.depth_mask_set(prior[2])
        return True

    def draw_lines(self):
        # Standalone POST_VIEW pass: registered before the display pass, so it
        # decides first; views the display pass draws in this redraw are skipped.
        context = bpy.context
        if viewport_display_overlay.claims_view(context):
            return
        self.draw_standalone(context)

    def draw_standalone(self, context):
        # The HDR path drew the image in view_draw; its presented packet's poses still apply.
        packet = viewport_display_overlay.presented_packet(context)
        return self.draw(context, self.packet_transforms(context.scene, packet))

    def draw_labels(self):
        # Standalone POST_PIXEL pass: views whose labels the display pass drew in
        # this redraw are skipped; the others use the standalone lines' transforms.
        context = bpy.context
        if viewport_display_overlay.labels_drawn(context):
            return
        packet = viewport_display_overlay.presented_packet(context)
        self.labels(context, self.packet_transforms(context.scene, packet))

    def labels(self, context, transforms=None):
        """Draw one view's labels in pixel space (the current matrix stack); True if drawn.

        transforms is the line pass's anchor override, so each label is placed
        with the matrix its anchor's lines were drawn with in this redraw.
        """
        entry = self.current(context)
        if entry is None:
            return False
        import blf
        import gpu
        from mathutils import Vector
        from bpy_extras.view3d_utils import location_3d_to_region_2d
        table = self._table(entry, transforms)
        prior = (gpu.state.blend_get(), gpu.state.depth_test_get(), gpu.state.depth_mask_get())
        try:
            gpu.state.blend_set("ALPHA")
            gpu.state.depth_test_set("NONE")
            blf.size(self.font, 18)
            for label in entry.labels:
                world = None if label.anchor is None else table.get(label.anchor)
                if label.anchor is not None and world is None:
                    continue
                position = Vector(label.position) if world is None else world @ Vector(label.position)
                point = location_3d_to_region_2d(context.region, context.region_data, position)
                if point is not None:
                    blf.position(self.font, point.x + 5, point.y + 5, 0)
                    blf.color(self.font, *label.color)
                    blf.draw(self.font, label.text)
        finally:
            gpu.state.blend_set(prior[0])
            gpu.state.depth_test_set(prior[1])
            gpu.state.depth_mask_set(prior[2])
        return True


def draw_indicators(context, transforms=None):
    """Stable drawing entry: one view's indicator lines with the active matrix stack.

    Display passes call this with the region's view/projection matrices current;
    transforms may override anchor world matrices. Returns True if lines were drawn.
    """
    service = _service
    return service is not None and service.draw(context, transforms)


def draw_display_pass(context, packet=None):
    """The OVRTX display pass's line pass; runtime-owned anchors use packet's poses."""
    service = _service
    return service is not None and service.draw(context, service.packet_transforms(context.scene, packet))


def draw_display_labels(context, packet=None):
    """The OVRTX display pass's label pass, in pixel space, with the line pass's packet poses."""
    service = _service
    return service is not None and service.labels(context, service.packet_transforms(context.scene, packet))


def draw_standalone_pass(context):
    """The standalone pass for one view, without the display-pass claim check."""
    service = _service
    return service is not None and service.draw_standalone(context)


# Reload must release callbacks holding bound methods of the previous class.
_previous = globals().get("_service")
if _previous is not None:
    _previous.stop()
for _callback in globals().get("_installed", ()):
    for _handlers in (bpy.app.handlers.load_pre, bpy.app.handlers.undo_pre, bpy.app.handlers.redo_pre,
                      bpy.app.handlers.depsgraph_update_post, bpy.app.handlers.frame_change_post):
        if _callback in _handlers:
            _handlers.remove(_callback)
_service = None
_installed = []


@persistent
def _clear(*_args):
    # RNA identities change on load, undo and redo: drop every prepared batch now.
    if _service is not None:
        _service.clear()


@persistent
def _invalidate(scene=None, depsgraph=None, *_args):
    # Fires on every pose mirror; the prepared lines stay drawable until refreshed,
    # and this view's anchor matrices follow the update right away.
    if _service is not None:
        if depsgraph is not None:
            _service.note_geometry_updates(depsgraph)
            if scene is not None:
                _service.update_transforms(scene, depsgraph)
        _service.request_refresh()


@persistent
def _frame_changed(scene=None, depsgraph=None, *_args):
    if _service is not None and scene is not None:
        _service.update_transforms(scene, depsgraph)


def register():
    global _service
    if bpy.app.background or _service is not None:
        return
    _service = OverlayService()
    try:
        _service.start()
        for handlers in (bpy.app.handlers.load_pre, bpy.app.handlers.undo_pre, bpy.app.handlers.redo_pre):
            handlers.append(_clear)
        bpy.app.handlers.depsgraph_update_post.append(_invalidate)
        bpy.app.handlers.frame_change_post.append(_frame_changed)
        _installed.extend((_clear, _invalidate, _frame_changed))
    except Exception:
        _service.stop()
        _service = None
        raise


def unregister():
    global _service
    if _service is not None:
        _service.stop()
        _service = None
    for callback in _installed:
        for handlers in (bpy.app.handlers.load_pre, bpy.app.handlers.undo_pre, bpy.app.handlers.redo_pre,
                         bpy.app.handlers.depsgraph_update_post, bpy.app.handlers.frame_change_post):
            if callback in handlers:
                handlers.remove(callback)
    _installed.clear()

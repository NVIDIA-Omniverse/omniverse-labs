# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-view display-encoded presentation on Blender's overlay attachment.

The base render attachment always receives Blender's scene display transform.
POST_VIEW targets its separate sRGB overlay attachment. Existing overlays are
read as encoded values and replayed above the opaque LDR image, preserving their
premultiplied alpha. All GPU access stays on Blender's draw thread.

Physics indicator lines get their own pass between the image and the replay,
with the view's matrices and the presented packet's poses, before the frame is
committed as presented. The standalone indicator pass skips every view this
pass claims in a redraw; a claimed view whose display draw is rejected gets the
standalone pass from here instead, so each view gets exactly one line pass.
The labels are drawn next, in pixel space with the same packet poses; the
standalone label pass (POST_PIXEL, after this pass) skips the views marked here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import weakref
import time
import threading
from . import performance_profile


def view_key(context: Any) -> tuple[int, int, int]:
    return tuple(int(getattr(context, field).as_pointer()) for field in ("window", "area", "region"))


def _indicator_line_pass(context: Any, packet: Any) -> bool:
    from . import physics_overlay
    return physics_overlay.draw_display_pass(context, packet)


def _indicator_label_pass(context: Any, packet: Any) -> bool:
    from . import physics_overlay
    return physics_overlay.draw_display_labels(context, packet)


def _indicator_standalone_pass(context: Any) -> bool:
    from . import physics_overlay
    return physics_overlay.draw_standalone_pass(context)


# Every drawer, so the indicator overlay can ask which views they claim.
_OVERLAYS: "weakref.WeakSet[DisplayOverlay]" = weakref.WeakSet()


def claims_view(context: Any) -> bool:
    """True if a display pass draws this view's indicator lines in this redraw."""
    claimed = False
    for overlay in tuple(_OVERLAYS):
        claimed = overlay.claim(context) or claimed
    return claimed


def labels_drawn(context: Any) -> bool:
    """True once if a display pass drew this view's indicator labels in this redraw."""
    try:
        key = view_key(context)
    except (AttributeError, ReferenceError):
        return False
    drawn = False
    for overlay in tuple(_OVERLAYS):
        if key in overlay.labelled:
            overlay.labelled.discard(key)
            drawn = True
    return drawn


def presented_packet(context: Any) -> Any:
    """The presented packet of a view whose image view_draw drew itself (HDR path)."""
    for overlay in tuple(_OVERLAYS):
        packet = overlay.direct_packet(context)
        if packet is not None:
            return packet
    return None


@dataclass
class DisplayWork:
    owner: Any
    epoch: object
    texture: Any
    result: Any
    position: tuple[float, float]
    size: tuple[float, float]
    context_signature: tuple[Any, ...]
    completion: dict[str, Any] | None = None
    presentation_frame: Any = None
    presentation_packet: Any = None


class DisplayOverlay:
    """Weak engine ownership, one latest draw per actual window/area/region."""

    def __init__(self) -> None:
        self.entries: dict[tuple[int, int, int], DisplayWork] = {}
        self.buffers: dict[tuple[int, int, int], tuple[tuple[int, int], Any]] = {}
        self.handler: Any = None
        self.shader: Any = None
        self.bpy: Any = None
        self.retired_owners: list[Any] = []
        # Queued draws discarded off the main thread, dropped by the main timer.
        self.discarded_work: list[tuple[Any, tuple[DisplayWork, ...]]] = []
        self.retire_lock = threading.Lock()
        self.blocked = lambda: False
        # Indicator lines: views claimed from the standalone pass this redraw,
        # views whose labels this pass drew this redraw, views drawn directly by
        # view_draw (owner, packet), errors reported once.
        self.claimed: set[tuple[int, int, int]] = set()
        self.labelled: set[tuple[int, int, int]] = set()
        self.direct: dict[tuple[int, int, int], tuple[Any, Any]] = {}
        self.line_errors: set[str] = set()
        self.lines_drawn = False
        self.line_pass = _indicator_line_pass
        self.label_pass = _indicator_label_pass
        self.standalone_pass = _indicator_standalone_pass
        _OVERLAYS.add(self)

    @staticmethod
    def context_signature(context: Any) -> tuple[Any, ...]:
        return (int(context.space_data.as_pointer()), context.region.width,
                context.region.height, int(context.scene.as_pointer()),
                int(context.view_layer.as_pointer()))

    def queue(self, context: Any, owner: Any, texture: Any, result: Any,
              position: tuple[float, float], size: tuple[float, float]) -> DisplayWork:
        key = view_key(context)
        for previous, work in tuple(self.entries.items()):
            if work.owner() is None or (work.owner() is owner and previous != key):
                self.entries.pop(previous, None)
                self.buffers.pop(previous, None)
        work = DisplayWork(weakref.ref(owner), owner._viewport_display_epoch,
                           texture, result, position, size, self.context_signature(context))
        self.entries[key] = work
        self.direct.pop(key, None)
        return work

    def note_direct(self, context: Any, owner: Any, packet: Any) -> None:
        """view_draw drew this view's image itself; packet is the presented one, or None."""
        try:
            self.direct[view_key(context)] = (weakref.ref(owner), packet)
        except (AttributeError, ReferenceError, TypeError):
            pass

    def direct_packet(self, context: Any) -> Any:
        try:
            item = self.direct.get(view_key(context))
            if item is None or item[1] is None or context.space_data.shading.type != "RENDERED":
                return None
            owner = item[0]()
            guard = getattr(owner, "_presentation_packet_is_current", None)
            if owner is None or (callable(guard) and not guard(item[1])):
                return None
        except (AttributeError, ReferenceError):
            return None
        return item[1]

    def discard_owner(self, owner: Any) -> None:
        if threading.current_thread() is not threading.main_thread():
            # RNA destruction can occur on a job thread. Retain GPU references
            # until the main event loop can release them and remove handlers.
            with self.retire_lock:
                self.retired_owners.append(weakref.ref(owner))
            if self.bpy is not None:
                self.bpy.app.timers.register(self.flush_retired, first_interval=0.0)
            return
        self._discard_refs([weakref.ref(owner)])

    def flush_retired(self) -> None:
        with self.retire_lock:
            retired, self.retired_owners = self.retired_owners, []
        self._discard_refs(retired)

    def discard_work(self, owner: Any) -> None:
        """Drop the owner's queued draws; keep the handler, shader and buffers.

        A new mirrored world must never show an older image, but the drawer
        itself stays ready for the next frame. Full teardown is discard_owner.
        """
        works = tuple(work for work in tuple(self.entries.values()) if work.owner() is owner)
        if threading.current_thread() is not threading.main_thread():
            # Refused by _admit from now on; dropped on the main event loop.
            with self.retire_lock:
                self.discarded_work.append((weakref.ref(owner), works))
            if self.bpy is not None:
                self.bpy.app.timers.register(self.flush_discarded, first_interval=0.0)
            return
        self._drop_work([(weakref.ref(owner), works)])

    def flush_discarded(self) -> None:
        with self.retire_lock:
            discarded, self.discarded_work = self.discarded_work, []
        self._drop_work(discarded)

    def _drop_work(self, discarded: list[tuple[Any, tuple[DisplayWork, ...]]]) -> None:
        for key, work in tuple(self.entries.items()):
            if any(work is old for _ref, works in discarded for old in works):
                self.entries.pop(key, None)
                self.claimed.discard(key)
                self.labelled.discard(key)
        owners = [ref() for ref, _works in discarded]
        for key, (ref, _packet) in tuple(self.direct.items()):
            if ref() is None or any(ref() is owner for owner in owners if owner is not None):
                self.direct.pop(key, None)

    def _discard_refs(self, retired: list[Any]) -> None:
        for key, work in tuple(self.entries.items()):
            if work.owner() is None or any(work.owner() is ref() for ref in retired):
                self.entries.pop(key, None)
                self.buffers.pop(key, None)
        for key, (ref, _packet) in tuple(self.direct.items()):
            if ref() is None or any(ref() is retired_ref() for retired_ref in retired):
                self.direct.pop(key, None)
        if not self.entries and self.bpy is not None:
            self.unregister(self.bpy)

    def unregister(self, bpy: Any) -> None:
        if self.handler is not None:
            bpy.types.SpaceView3D.draw_handler_remove(self.handler, "WINDOW")
            self.handler = None
        self.entries.clear()
        self.buffers.clear()
        self.claimed.clear()
        self.labelled.clear()
        self.direct.clear()
        self.shader = None

    def register(self, bpy: Any, blocked: Any) -> None:
        self.bpy = bpy
        self.blocked = blocked
        if self.handler is None:
            def draw() -> None:
                if not blocked():
                    self.present(bpy.context)
            self.handler = bpy.types.SpaceView3D.draw_handler_add(draw, (), "WINDOW", "POST_VIEW")

    def _admit(self, context: Any) -> tuple[Any, ...] | None:
        """(key, work, owner, packet, valid) for this view's queued draw, or None."""
        if self.blocked():
            return None
        if getattr(context, "area", None) is None or getattr(context, "region", None) is None:
            return None
        key = view_key(context)
        work = self.entries.get(key)
        if work is None:
            return None
        owner = work.owner()
        with self.retire_lock:
            retired = any(ref() is owner for ref in self.retired_owners)
            discarded = any(work is old for _ref, works in self.discarded_work for old in works)
        if discarded:
            # Its main-thread drop is pending; draw nothing from it until then.
            return None
        try:
            valid = (not retired and owner is not None and owner._viewport_display_epoch is work.epoch
                     and context.area.type == "VIEW_3D"
                     and context.space_data.shading.type == "RENDERED"
                     and self.context_signature(context) == work.context_signature)
        except ReferenceError:
            valid = False
        packet = work.presentation_packet
        if work.completion is not None:
            packet = work.completion.get('prepared_presentation')
        packet_guard = getattr(owner, '_presentation_packet_is_current', None)
        if valid and callable(packet_guard):
            valid = packet_guard(packet)
        return key, work, owner, packet, bool(valid)

    def claim(self, context: Any) -> bool:
        """Decide up front whether present() draws this view's indicator lines.

        Called by the standalone indicator pass, whose handler runs before
        present() in the same redraw; each call replaces this view's marker and
        drops a label marker the previous redraw left unconsumed.
        """
        try:
            key = view_key(context)
            admitted = self._admit(context)
        except (AttributeError, ReferenceError):
            return False
        self.labelled.discard(key)
        claimed = self.handler is not None and admitted is not None and admitted[4]
        if claimed:
            self.claimed.add(key)
        else:
            self.claimed.discard(key)
        return claimed

    def present(self, context: Any) -> None:
        try:
            key = view_key(context)
        except (AttributeError, ReferenceError):
            return
        claimed = key in self.claimed
        self.claimed.discard(key)
        self.labelled.discard(key)
        admitted = self._admit(context)
        if admitted is None or not admitted[4]:
            if admitted is not None:
                self.entries.pop(key, None)
                self.buffers.pop(key, None)
                if not self.entries and self.bpy is not None:
                    self.unregister(self.bpy)
            if claimed:
                self._standalone_lines(context, None if admitted is None else admitted[2])
            return
        key, work, owner, packet, _valid = admitted
        self.lines_drawn = False
        try:
            started = time.perf_counter_ns()
            self.draw_pixels(context, work, key)
        except Exception as exc:
            # Never advertise a frame whose actual display draw failed.
            work.completion = None
            owner._report_viewport_error(exc)
            if claimed and not self.lines_drawn:
                self._standalone_lines(context, owner)
            return
        # A job-thread retirement can arrive after draw admission. Let an
        # already issued GPU draw finish, but never commit a retired publication.
        # The queued main-loop cleanup retains responsibility for its resources.
        with self.retire_lock:
            retired = any(ref() is owner for ref in self.retired_owners)
            retired = retired or any(work is old for _ref, works in self.discarded_work for old in works)
        try:
            stale = owner._viewport_display_epoch is not work.epoch
        except ReferenceError:
            stale = True
        if retired or stale or self.blocked() or self.entries.get(key) is not work:
            work.completion = None
            return
        completion = work.completion
        work.completion = None
        if completion is not None:
            completion["draw_started_ns"] = started
            completion["span_boundaries"]["viewport_draw_started_monotonic_ns"] = started
            owner._finish_viewport_presentation(**completion)
        elif work.presentation_frame is not None:
            owner._finish_cached_frame_presentation(work.presentation_frame, packet)

    @performance_profile.timed("main.viewport.display_overlay")
    def draw_pixels(self, context: Any, work: DisplayWork, key: tuple[int, int, int]) -> None:
        import gpu
        from gpu_extras.batch import batch_for_shader
        from gpu_extras.presets import draw_texture_2d
        from mathutils import Matrix

        width, height = max(1, context.region.width), max(1, context.region.height)
        framebuffer = gpu.state.active_framebuffer_get()
        blend, depth = gpu.state.blend_get(), gpu.state.depth_test_get()
        overlay = None
        try:
            gpu.state.blend_set("NONE")
            gpu.state.depth_test_set("NONE")
            if context.space_data.overlay.show_overlays:
                resource = self.buffers.get(key)
                if resource is None or resource[0] != (width, height):
                    resource = ((width, height), gpu.types.Buffer("FLOAT", width * height * 4))
                    self.buffers[key] = resource
                # SRGBA8 readback returns encoded normalized channel values.
                # FLOAT directly feeds the public constructor, avoiding Python
                # arrays/conversion copies. RGBA8 retains the attachment's 8-bit
                # precision; IMAGE performs the one target sRGB decode on replay.
                framebuffer.read_color(0, 0, width, height, 4, 0, "FLOAT", data=resource[1])
                overlay = gpu.types.GPUTexture((width, height), format="RGBA8", data=resource[1])
            framebuffer.clear(color=(0.0, 0.0, 0.0, 1.0))
            if self.shader is None:
                interface = gpu.types.GPUStageInterfaceInfo("ovrtx_display_uv")
                interface.smooth("VEC2", "uv")
                info = gpu.types.GPUShaderCreateInfo()
                info.push_constant("MAT4", "mvp")
                info.vertex_in(0, "VEC2", "pos")
                info.vertex_in(1, "VEC2", "texCoord")
                info.vertex_out(interface)
                info.fragment_out(0, "VEC4", "fragColor")
                info.sampler(0, "FLOAT_2D", "image")
                info.vertex_source("void main(){uv=texCoord;gl_Position=mvp*vec4(pos,0.0,1.0);}")
                # Native LDR is display encoded and its presentation is opaque.
                # The SRGBA8 target encodes this linear output exactly once.
                info.fragment_source("void main(){vec3 c=texture(image,uv).rgb;vec3 linear=mix(c/12.92,pow((c+0.055)/1.055,vec3(2.4)),greaterThan(c,vec3(0.04045)));fragColor=vec4(linear,1.0);}")
                self.shader = gpu.shader.create_from_info(info)
            x, y = work.position
            w, h = work.size
            pixels = Matrix(((2/width, 0, 0, -1), (0, 2/height, 0, -1), (0, 0, -1, 0), (0, 0, 0, 1)))
            with gpu.matrix.push_pop(), gpu.matrix.push_pop_projection():
                gpu.matrix.load_matrix(Matrix.Identity(4))
                gpu.matrix.load_projection_matrix(pixels)
                shader = self.shader
                shader.bind()
                shader.uniform_float("mvp", gpu.matrix.get_projection_matrix() @ gpu.matrix.get_model_view_matrix())
                shader.uniform_sampler("image", work.texture)
                batch = batch_for_shader(shader, "TRI_FAN", {"pos": ((x,y),(x+w,y),(x+w,y+h),(x,y+h)), "texCoord": ((0,0),(1,0),(1,1),(0,1))})
                batch.draw(shader)
            # Back on the region's view/projection: the indicator lines of this
            # frame's poses, above the image and below Blender's own overlays.
            self._draw_lines(context, work)
            with gpu.matrix.push_pop(), gpu.matrix.push_pop_projection():
                gpu.matrix.load_matrix(Matrix.Identity(4))
                gpu.matrix.load_projection_matrix(pixels)
                # Their labels in pixel space, from the same packet poses.
                self._draw_labels(context, work, key)
                if overlay is not None:
                    gpu.state.blend_set("ALPHA_PREMULT")
                    gpu.state.depth_test_set("NONE")
                    draw_texture_2d(overlay, (0,0), width, height)
        finally:
            gpu.state.blend_set(blend)
            gpu.state.depth_test_set(depth)

    @staticmethod
    def _work_packet(work: DisplayWork) -> Any:
        if work.completion is not None:
            return work.completion.get('prepared_presentation')
        return work.presentation_packet

    def _draw_lines(self, context: Any, work: DisplayWork) -> None:
        """The indicator line pass; a failure is reported once and never blocks the frame."""
        self.lines_drawn = True
        try:
            self.line_pass(context, self._work_packet(work))
        except Exception as exc:
            self._line_error(work.owner(), exc)

    def _draw_labels(self, context: Any, work: DisplayWork, key: tuple[int, int, int]) -> None:
        """The indicator label pass; marks the view so the standalone label pass skips it."""
        try:
            self.label_pass(context, self._work_packet(work))
        except Exception as exc:
            # Unmarked: the standalone label pass still draws this view's labels.
            self._line_error(work.owner(), exc, "labels")
            return
        self.labelled.add(key)

    def _standalone_lines(self, context: Any, owner: Any) -> None:
        """A claimed view whose display draw was rejected still gets one line pass."""
        try:
            self.standalone_pass(context)
        except Exception as exc:
            self._line_error(owner, exc)

    def _line_error(self, owner: Any, exc: Exception, what: str = "lines") -> None:
        message = f"Physics indicator {what} were not drawn: {exc}"
        if message in self.line_errors:
            return
        self.line_errors.add(message)
        report = getattr(owner, "_report_viewport_error", None)
        if callable(report):
            report(RuntimeError(message))

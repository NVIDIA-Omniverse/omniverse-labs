# SPDX-License-Identifier: Apache-2.0
"""The indicator line pass inside the OVRTX display pass, and its standalone fallback.

Runs ``physics_overlay`` against the fake ``bpy``/``gpu``/``mathutils`` of the
body-local batch tests, with a real ``DisplayOverlay`` deciding which pass draws.
"""
import sys
from types import SimpleNamespace

import pytest

from ovrtx_blender_example.physics_frames import RigidFrame
from ovrtx_blender_example.physics_presentation_targets import PhysicsPresentationTarget
from ovrtx_blender_example.viewport_display_overlay import DisplayOverlay
from test_physics_overlay_local_batches import Vector, close, entry, frame_matrix, overlay, view  # noqa: F401

SESSIONS = "ovrtx_blender_example.scene_generation_sessions"


def target(identity, path, role="DYNAMIC", meters_per_blender_unit=.5):
    return PhysicsPresentationTarget(path, identity, 11, identity, "", "", role, True, meters_per_blender_unit,
                                     1.0, RigidFrame(), RigidFrame())


def packet_for(poses):
    """The engine's (frame, identity, pending, applied, initial) with the frame's tick poses."""
    pose_set = tuple(SimpleNamespace(prim_path=path, translate=translate, orient=orient)
                     for path, (translate, orient) in poses.items())
    frame = SimpleNamespace(runtime_tick_result=SimpleNamespace(physics_pose_set=pose_set))
    return (frame, ("identity",), {}, {}, False)


@pytest.fixture
def sessions(monkeypatch):
    # The cube is runtime-owned; the arm's body is not dynamic, so Blender's transform applies.
    state = SimpleNamespace(calls=0, targets=(target("cube", "/World/cube"),
                                              target("arm", "/World/arm", role="KINEMATIC")))

    def current_presentation_context(scene):
        state.calls += 1
        return 3, 1, state.targets

    monkeypatch.setitem(sys.modules, SESSIONS, SimpleNamespace(current_presentation_context=current_presentation_context))
    return state


# A packet pose (meters, quaternion x, y, z, w) far from the Blender transforms in make_source().
PACKET = {"/World/cube": ((2.0, -1.0, 4.0), (0.0, 0.0, 0.6, 0.8)),
          "/World/arm": ((50.0, 50.0, 50.0), (0.0, 0.0, 0.0, 1.0))}


def stroke_draw(overlay, draws, identity):
    prepared = entry(overlay)
    for drawn, (stroke, _batch, _local) in zip(draws, prepared.strokes):
        if stroke.identity == identity:
            return stroke, drawn[1]
    raise AssertionError(identity)


def lines(overlay, call):
    overlay.gpu.draws.clear()
    with view(overlay):
        call(overlay.bpy.context)
    return list(overlay.gpu.draws)


def expected_world(anchor_frame, stroke, vertex):
    return anchor_frame.compose(stroke.local_frame).compose(RigidFrame(vertex)).position


def test_line_pass_uses_presented_packet_poses(overlay, sessions):
    overlay.service.refresh()
    blender = {a.identity: a.frame for a in overlay.state.source.anchors}
    packet = packet_for(PACKET)
    draws = lines(overlay, lambda context: overlay.module.draw_display_pass(context, packet))
    # The runtime-owned cube is drawn at the packet pose (meters -> BU, xyzw -> wxyz).
    presented = RigidFrame((4.0, -2.0, 8.0), (0.8, 0.0, 0.0, 0.6))
    stroke, model = stroke_draw(overlay, draws, "cube:box")
    for vertex in stroke.vertices[:4]:
        assert close(model @ Vector(vertex), expected_world(presented, stroke, vertex), 1e-9)
        assert not close(model @ Vector(vertex), expected_world(blender["cube"], stroke, vertex), 1e-3)
    # The arm is not a dynamic target: its Blender transform applies, not the packet's.
    stroke, model = stroke_draw(overlay, draws, "arm:ball")
    assert close(model @ Vector(stroke.vertices[0]), expected_world(blender["arm"], stroke, stroke.vertices[0]), 1e-9)
    # Without a packet (no physics frame) the transform table is used.
    draws = lines(overlay, lambda context: overlay.module.draw_display_pass(context, None))
    stroke, model = stroke_draw(overlay, draws, "cube:box")
    assert close(model @ Vector(stroke.vertices[0]), expected_world(blender["cube"], stroke, stroke.vertices[0]), 1e-9)


def test_packet_pose_conversion_cached_per_packet(overlay, sessions):
    overlay.service.refresh()
    packet = packet_for(PACKET)
    with view(overlay):
        context = overlay.bpy.context
        first = overlay.service.packet_transforms(context.scene, packet)
        for _ in range(5):
            assert overlay.module.draw_display_pass(context, packet)
        assert overlay.service.packet_transforms(context.scene, packet) is first
        assert sessions.calls == 1
        assert set(first) == {"cube"}
        assert first["cube"] == frame_matrix(RigidFrame((4.0, -2.0, 8.0), (0.8, 0.0, 0.0, 0.6)))
        # A new packet (the next presented frame) is converted once more.
        moved = packet_for({"/World/cube": ((0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0))})
        assert overlay.module.draw_display_pass(context, moved)
        assert overlay.module.draw_display_pass(context, moved)
        assert sessions.calls == 2
        assert overlay.service.packet_transforms(context.scene, moved)["cube"] == frame_matrix(RigidFrame((0.0, 0.0, 2.0)))
        # Load/undo drops the cached conversion with the batches.
        overlay.service.clear()
        assert overlay.service.packet is None


class Owner:
    def __init__(self, packet):
        self._viewport_display_epoch = object()
        self.packet = packet
        self.events = []

    def _presentation_packet_is_current(self, packet):
        return packet is self.packet

    def _finish_viewport_presentation(self, **values):
        self.events.append("presented")

    def _report_viewport_error(self, error):
        self.events.append(str(error))


@pytest.fixture
def display(overlay, monkeypatch):
    """A registered DisplayOverlay whose draw_pixels only runs its line pass."""
    manager = DisplayOverlay()
    manager.handler = 17
    monkeypatch.setattr(manager, "context_signature", lambda context: ("view",))
    monkeypatch.setattr(manager, "line_pass", overlay.module.draw_display_pass)
    monkeypatch.setattr(manager, "standalone_pass", overlay.module.draw_standalone_pass)
    monkeypatch.setattr(manager, "draw_pixels", lambda context, work, key: manager._draw_lines(context, work))
    yield manager
    manager.entries.clear()
    manager.direct.clear()


def test_standalone_pass_skips_view_drawn_by_display_pass(overlay, sessions, display):
    overlay.service.refresh()
    strokes = len(entry(overlay).strokes) + len(entry(overlay).links)
    packet = packet_for(PACKET)
    owner = Owner(packet)
    with view(overlay):
        work = display.queue(overlay.bpy.context, owner, object(), object(), (0, 0), (640, 480))
    work.completion = {"prepared_presentation": packet, "span_boundaries": {}}
    # One redraw: the standalone handler (registered first) skips the claimed view ...
    assert lines(overlay, lambda context: overlay.service.draw_lines()) == []
    # ... and the display pass draws it once, with the packet's poses, before the commit.
    draws = lines(overlay, display.present)
    assert len(draws) == strokes and owner.events == ["presented"]
    stroke, model = stroke_draw(overlay, draws, "cube:box")
    presented = RigidFrame((4.0, -2.0, 8.0), (0.8, 0.0, 0.0, 0.6))
    assert close(model @ Vector(stroke.vertices[0]), expected_world(presented, stroke, stroke.vertices[0]), 1e-9)
    # A view the display pass does not draw (no queued work) keeps the standalone pass.
    display.entries.clear()
    assert len(lines(overlay, lambda context: overlay.service.draw_lines())) == strokes
    assert len(lines(overlay, display.present)) == 0


def test_hdr_standalone_pass_uses_presented_packet(overlay, sessions, display):
    overlay.service.refresh()
    packet = packet_for(PACKET)
    owner = Owner(packet)
    with view(overlay):
        # view_draw drew the HDR image itself and committed this packet.
        display.note_direct(overlay.bpy.context, owner, packet)
    draws = lines(overlay, lambda context: overlay.service.draw_lines())
    stroke, model = stroke_draw(overlay, draws, "cube:box")
    presented = RigidFrame((4.0, -2.0, 8.0), (0.8, 0.0, 0.0, 0.6))
    assert close(model @ Vector(stroke.vertices[0]), expected_world(presented, stroke, stroke.vertices[0]), 1e-9)
    # A packet that is no longer current falls back to the Blender transforms.
    owner.packet = object()
    blender = {a.identity: a.frame for a in overlay.state.source.anchors}
    draws = lines(overlay, lambda context: overlay.service.draw_lines())
    stroke, model = stroke_draw(overlay, draws, "cube:box")
    assert close(model @ Vector(stroke.vertices[0]), expected_world(blender["cube"], stroke, stroke.vertices[0]), 1e-9)


@pytest.fixture
def text(overlay, monkeypatch):
    """Fake blf and view3d_utils: each label's 3D position and text, in draw order."""
    record = SimpleNamespace(projected=[], drawn=[])

    def location_3d_to_region_2d(region, region_data, position):
        record.projected.append(tuple(position))
        return SimpleNamespace(x=position[0], y=position[1])

    blf = SimpleNamespace(load=lambda path: 1, unload=lambda path: None, size=lambda font, size: None,
                          position=lambda font, x, y, z: None, color=lambda font, *rgba: None,
                          draw=lambda font, value: record.drawn.append((value, record.projected[-1])))
    view3d_utils = SimpleNamespace(location_3d_to_region_2d=location_3d_to_region_2d)
    monkeypatch.setitem(sys.modules, "blf", blf)
    monkeypatch.setitem(sys.modules, "bpy_extras", SimpleNamespace(view3d_utils=view3d_utils))
    monkeypatch.setitem(sys.modules, "bpy_extras.view3d_utils", view3d_utils)
    overlay.bpy.context.region_data = object()
    return record


def labels(overlay, text, call):
    text.drawn.clear()
    with view(overlay):
        call(overlay.bpy.context)
    return list(text.drawn)


def test_labels_use_same_transforms_as_lines(overlay, sessions, text):
    overlay.service.refresh()
    prepared = entry(overlay)
    assert prepared.labels and all(label.anchor == "cube" for label in prepared.labels)
    packet = packet_for(PACKET)
    draws = lines(overlay, lambda context: overlay.module.draw_display_pass(context, packet))
    drawn = labels(overlay, text, lambda context: overlay.module.draw_display_labels(context, packet))
    assert [value for value, _position in drawn] == [label.text for label in prepared.labels]
    # Each label sits at its anchor-local position under the matrix its anchor's
    # lines were drawn with in this pass: world = (anchor world @ local) @ local^-1.
    stroke, model = stroke_draw(overlay, draws, "cube:box")
    anchor_world = model @ frame_matrix(stroke.local_frame.inverse())
    presented = RigidFrame((4.0, -2.0, 8.0), (0.8, 0.0, 0.0, 0.6))
    for label, (_value, position) in zip(prepared.labels, drawn):
        assert close(position, anchor_world @ Vector(label.position), 1e-9)
        assert close(position, presented.compose(RigidFrame(label.position)).position, 1e-9)
    # One packet conversion serves both passes.
    assert sessions.calls == 1
    # The standalone passes share their transforms the same way (HDR path: the
    # committed packet; otherwise the view's transform table).
    blender = {a.identity: a.frame for a in overlay.state.source.anchors}
    drawn = labels(overlay, text, lambda context: overlay.service.draw_labels())
    for label, (_value, position) in zip(prepared.labels, drawn):
        assert close(position, blender["cube"].compose(RigidFrame(label.position)).position, 1e-9)


def test_standalone_label_pass_skips_view_drawn_by_display_pass(overlay, sessions, text, display, monkeypatch):
    overlay.service.refresh()
    count = len(entry(overlay).labels)
    packet = packet_for(PACKET)
    owner = Owner(packet)

    def draw_pixels(context, work, key):
        display._draw_lines(context, work)
        display._draw_labels(context, work, key)

    monkeypatch.setattr(display, "label_pass", overlay.module.draw_display_labels)
    monkeypatch.setattr(display, "draw_pixels", draw_pixels)

    def redraw():
        # Handler order of one redraw: standalone POST_VIEW, display POST_VIEW, standalone POST_PIXEL.
        with view(overlay):
            context = overlay.bpy.context
            overlay.service.draw_lines()
            display.present(context)
            overlay.service.draw_labels()

    def queue():
        with view(overlay):
            work = display.queue(overlay.bpy.context, owner, object(), object(), (0, 0), (640, 480))
        work.completion = {"prepared_presentation": packet, "span_boundaries": {}}

    queue()
    text.drawn.clear()
    redraw()
    # Drawn once, by the display pass, at the packet pose.
    presented = RigidFrame((4.0, -2.0, 8.0), (0.8, 0.0, 0.0, 0.6))
    assert len(text.drawn) == count and owner.events == ["presented"]
    assert close(text.drawn[0][1], presented.compose(RigidFrame(entry(overlay).labels[0].position)).position, 1e-9)
    # The next redraw has no queued image: the standalone label pass draws them.
    text.drawn.clear()
    redraw()
    assert len(text.drawn) == count
    # A claimed view whose display draw is rejected (stale packet) keeps its labels.
    queue()
    owner.packet = object()
    text.drawn.clear()
    redraw()
    assert len(text.drawn) == count
    # A marker a redraw left unconsumed never suppresses the next redraw's labels.
    owner.packet = packet
    queue()
    with view(overlay):
        overlay.service.draw_lines()
        display.present(overlay.bpy.context)
    display.entries.clear()
    text.drawn.clear()
    redraw()
    assert len(text.drawn) == count


def test_packet_targets_read_once_per_generation(overlay, sessions, monkeypatch):
    # Presentation targets are rebuilt from every opinion record of a generation,
    # so a new packet of the same generation reuses them; a new generation rereads.
    owner = SimpleNamespace(current_generation=object())
    monkeypatch.setattr(sys.modules[SESSIONS], "_owners", {7: owner}, raising=False)
    overlay.service.refresh()
    with view(overlay):
        context = overlay.bpy.context
        for step in range(4):
            moved = packet_for({"/World/cube": ((0.0, 0.0, float(step)), (0.0, 0.0, 0.0, 1.0))})
            matrices = overlay.service.packet_transforms(context.scene, moved)
            assert matrices["cube"] == frame_matrix(RigidFrame((0.0, 0.0, 2.0*step)))
        assert sessions.calls == 1
        owner.current_generation = object()
        overlay.service.packet_transforms(context.scene, packet_for(PACKET))
        assert sessions.calls == 2
        # The fast conversion equals the frame-based one exactly.
        pose = packet_for(PACKET)[0].runtime_tick_result.physics_pose_set[0]
        expected = frame_matrix(overlay.module.presented_frame(target("cube", "/World/cube"), pose))
        assert overlay.module.presented_matrix(.5, pose) == expected
        # Load/undo drops the cached targets with the batches.
        overlay.service.clear()
        assert overlay.service.targets is None

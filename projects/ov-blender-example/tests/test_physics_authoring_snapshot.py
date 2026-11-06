# SPDX-License-Identifier: Apache-2.0
from dataclasses import FrozenInstanceError, replace
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from ovrtx_blender_example import physics_authoring_snapshot as snapshot_module
from ovrtx_blender_example.physics_endpoints import ResolvedEndpoint
from ovrtx_blender_example.physics_frames import RigidFrame, evaluated_owner_frame
from ovrtx_blender_example.physics_authoring_snapshot import (
    Axis, Collider, IndicatorCache, Joint, Scope, Source, SnapshotError,
    capture, geometry, ERROR, PENDING,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "physics_indicator_world_geometry.json"
LINK_ROLES = ("endpoint-links", "frame-separation", "pending-link")


def point(frame, value):
    return frame.compose(RigidFrame(value)).position


def anchored(anchors, anchor, frame):
    return frame if anchor is None else anchors[anchor].compose(frame)


def world(stroke, anchors=None):
    frame = anchored(anchors or {}, stroke.anchor, stroke.local_frame)
    return tuple(point(frame, v) for v in stroke.vertices)


def link_points(link, anchors=None):
    return tuple(point(anchored(anchors or {}, anchor, RigidFrame()), p) for anchor, p in (link.start, link.end))


def joint(**changes):
    result = Joint("joint", "Hinge", "REVOLUTE", "X", RigidFrame(), RigidFrame((5, 0, 0)),
                   RigidFrame((1, 2, 3), (1, 0, 0, 1)), RigidFrame((4, 2, 3)),
                   (Axis("rotX", "LIMITED", 0, math.pi/2, True, math.pi/4),))
    return replace(result, **changes)


def test_transformed_hinge_frames_limits_and_labeled_authored_target():
    result = geometry(Source(Scope(1, "L", 2, 3), (joint(),), size=2))
    arc = next(s for s in result.strokes if s.role == "limit-rotX")
    assert world(arc)[0] == pytest.approx((-2, 2, 3))
    assert world(arc)[-1] == pytest.approx((1, 2, 6))
    links = [link_points(l) for l in result.links if l.role == "endpoint-links"]
    assert links == [((0, 0, 0), (1, 2, 3)), ((5, 0, 0), (4, 2, 3))]
    assert not any(s.role in LINK_ROLES for s in result.strokes)
    assert any(s.role == "frame-a-marker" for s in result.strokes)
    assert any(s.role == "frame-b-marker" for s in result.strokes)
    assert "not measured" in next(t.text for t in result.labels if t.role == "authored-target")


def test_slider_nondefault_axis_and_d6_state_shapes():
    j = joint(type="D6", frame_a=RigidFrame(), axes=(
        Axis("transZ", "LIMITED", -2, 4, True, 3),
        Axis("rotY", "LOCKED", 0, 0, False, 0),
        Axis("transX", "FREE", 0, 0, False, 0)))
    result = geometry(Source(Scope(1, "L", 1, 1), (j,)))
    track = next(s for s in result.strokes if s.role == "limit-transZ")
    assert track.vertices[:2] == ((0, 0, -2), (0, 0, 4))
    assert {"locked-rotY", "free-transX"} <= {s.role for s in result.strokes}
    target = next(t for t in result.labels if t.role == "authored-target")
    assert target.position == (0, 0, 3)
    slider = geometry(Source(result.scope, (replace(j, type="PRISMATIC", axis="Z"),)))
    assert any(s.role == "slider-glyph" for s in slider.strokes)
    assert not any(s.role == "locked-rotY" for s in slider.strokes)


def test_pending_and_broken_have_distinct_shapes_colors():
    j = joint(pending=RigidFrame((9, 8, 7)), errors=("Missing endpoint",))
    result = geometry(Source(Scope(1, "L", 1, 1), (j,)))
    pending = next(s for s in result.strokes if s.role == "pending-anchor")
    broken = next(s for s in result.strokes if s.role == "broken-link")
    assert pending.color == PENDING and broken.color == ERROR
    assert len(pending.vertices) == 8 and len(broken.vertices) == 6
    assert next(t for t in result.labels if t.role == "pending-anchor").position == (9, 8, 7)
    assert link_points(next(l for l in result.links if l.role == "pending-link")) == ((1, 2, 3), (9, 8, 7))
    assert j.frame_a.position == (1, 2, 3)


def test_collider_wire_geometry_and_disabled_shape():
    scope = Scope(1, "L", 1, 1)
    colliders = (Collider("sphere", "Sphere", "ANALYTIC_SPHERE", RigidFrame((1, 2, 3)), radius=2),
                 Collider("box", "Box", "BOX", RigidFrame(), dimensions=(2, 4, 6), enabled=False))
    result = geometry(Source(scope, colliders=colliders))
    sphere = next(s for s in result.strokes if s.identity == "sphere")
    assert max(p[0] for p in world(sphere)) == pytest.approx(3)
    assert min(p[2] for p in world(sphere)) == pytest.approx(1)
    box = next(s for s in result.strokes if s.identity == "box" and s.role == "collider-wire")
    assert len(box.vertices) == 24
    assert any(s.role == "disabled-collider" for s in result.strokes)


def test_cache_reuses_copied_content_and_invalidates_visibility_generation_errors():
    cache = IndicatorCache()
    source = Source(Scope(1, "L", 2, 1), (joint(),))
    first = cache.refresh(source)
    assert cache.refresh(source) is first
    with pytest.raises(FrozenInstanceError):
        first.strokes = ()
    hidden = replace(source, scope=replace(source.scope, visibility=(False,)), joints=())
    assert not cache.refresh(hidden).strokes
    assert cache.get(source.scope) is None
    newer = replace(source, scope=replace(source.scope, generation=2))
    cache.refresh(newer)
    assert cache.get(hidden.scope) is None
    with pytest.raises(SnapshotError, match="stale"):
        cache.refresh(source)
    with pytest.raises(SnapshotError, match="size"):
        cache.refresh(replace(newer, size=0))
    assert cache.get(newer.scope) is None
    cache.refresh(newer)
    cache.invalidate()
    assert cache.get(newer.scope) is None


@pytest.mark.parametrize("height", (0., 6.))
def test_capsule_wire_has_z_cylinder_and_hemisphere_bounds(height):
    # Rotating Z onto X proves caps, rings and sides share the collider frame.
    frame = RigidFrame((10, 20, 30), (math.sqrt(.5), 0, math.sqrt(.5), 0))
    collider = Collider("capsule", "Capsule", "CAPSULE", frame, radius=2, height=height)
    result = geometry(Source(Scope(1, "L", 1, 1), colliders=(collider,)))
    wire = world(next(s for s in result.strokes if s.role == "collider-wire"))
    assert not result.labels
    for axis, expected in enumerate(((10-height/2-2, 10+height/2+2), (18, 22), (28, 32))):
        assert (min(p[axis] for p in wire), max(p[axis] for p in wire)) == pytest.approx(expected)


# -- body-local strokes (n-02-02) ------------------------------------------------

def _frame(data):
    return RigidFrame(tuple(data["position"]), tuple(data["rotation_wxyz"]))


def _anchored_inputs(data):
    """The pre-change world-frame inputs, re-expressed relative to body anchors."""
    anchors, joints, colliders = {}, [], []
    for j in data["joints"]:
        sides = []
        for key in "ab":
            body, frame = _frame(j["body_" + key]), _frame(j["frame_" + key])
            if j.get("world_" + key):
                sides.append((None, body, frame))
            else:
                name = j["identity"] + "/body-" + key
                anchors[name] = body
                sides.append((name, RigidFrame(), body.inverse().compose(frame)))
        joints.append(Joint(j["identity"], j["label"], j["type"], j["axis"], sides[0][1], sides[1][1],
                            sides[0][2], sides[1][2],
                            tuple(Axis(a["key"], a["state"], a["lower"], a["upper"], a["driven"], a["target"]) for a in j["axes"]),
                            _frame(j["pending"]) if j["pending"] else None, tuple(j["errors"]), j["enabled"],
                            sides[0][0], sides[1][0]))
    for index, c in enumerate(data["colliders"]):
        name = c["identity"] + "/body"
        anchors[name] = RigidFrame((.5*index, -1+index, 2), (.8, .2*index, .4, -.1))
        colliders.append(Collider(c["identity"], c["label"], c["shape"], anchors[name].inverse().compose(_frame(c["frame"])),
                                  c.get("radius", 0.0), tuple(c.get("dimensions", (0.0, 0.0, 0.0))), c.get("height", 0.0),
                                  tuple(tuple(tuple(p) for p in pair) for pair in c.get("wire", ())),
                                  c.get("enabled", True), c.get("error", ""), name))
    return anchors, tuple(joints), tuple(colliders)


def _close(actual, expected, tolerance=1e-9):
    return len(actual) == len(expected) and all(
        len(a) == len(e) and all(abs(x - y) <= tolerance for x, y in zip(a, e)) for a, e in zip(actual, expected))


def test_world_space_reconstruction_matches_previous_geometry():
    data = json.loads(FIXTURE.read_text())
    anchors, joints, colliders = _anchored_inputs(data)
    assert {j["type"] for j in data["joints"]} >= {"FIXED", "REVOLUTE", "PRISMATIC", "D6"}
    assert {c["shape"] for c in data["colliders"]} >= {"ANALYTIC_SPHERE", "CONVEX_HULL", "BOX", "CAPSULE"}
    for case in data["cases"]:
        result = geometry(Source(Scope(1, "L", 2, 3), joints, colliders, case["size"]))
        old = [s for s in case["strokes"] if s["role"] not in LINK_ROLES]
        assert [(s.role, s.identity, list(s.color)) for s in result.strokes] == [(s["role"], s["identity"], s["color"]) for s in old]
        for stroke, expected in zip(result.strokes, old):
            assert _close(world(stroke, anchors), expected["vertices"]), (stroke.role, stroke.identity)
        # Each pre-change link stroke is the concatenation of its Link segments.
        grouped = {}
        for link in result.links:
            grouped.setdefault((link.role, link.identity), []).extend(link_points(link, anchors))
        old_links = {(s["role"], s["identity"]): s["vertices"] for s in case["strokes"] if s["role"] in LINK_ROLES}
        assert grouped.keys() == old_links.keys()
        for key, points in grouped.items():
            assert _close(points, old_links[key]), key
        assert [(l.role, l.identity, l.text, list(l.color)) for l in result.labels] == [
            (l["role"], l["identity"], l["text"], l["color"]) for l in case["labels"]]
        for label, expected in zip(result.labels, case["labels"]):
            assert _close((point(anchored(anchors, label.anchor, RigidFrame()), label.position),), (expected["position"],))


class FakeMatrix(tuple):
    def copy(self):
        return self


def _matrix(position, rotation=(1, 0, 0, 0)):
    return FakeMatrix(RigidFrame(position, rotation).column_matrix())


class FakeObjects(list):
    def get(self, name):
        return next((o for o in self if o.name == name), None)


def _record(**values):
    values.setdefault("schema_opinion", "APPLY")
    values.setdefault("offset", SimpleNamespace(translation=(0, 0, 0), rotation=(1, 0, 0, 0)))
    values.setdefault("radius_m", 0.0)
    values.setdefault("height_m", 0.0)
    values.setdefault("bl_rna", SimpleNamespace(properties={}))
    return SimpleNamespace(**values)


def _object(name, uid, matrix, *, colliders=(), joint=None):
    body = SimpleNamespace(schema_opinion="APPLY", role="DYNAMIC", enabled=True,
                           offset=SimpleNamespace(translation=(0, 0, .5), rotation=(1, 0, 0, 0)))
    joint = joint or SimpleNamespace(persistent_id="", show_indicator=False)
    obj = SimpleNamespace(name=name, session_uid=uid, type="MESH", matrix_world=matrix, scale=(1, 1, 1),
                          parent=None, ov=SimpleNamespace(physics=SimpleNamespace(
                              persistent_id=name.lower(), rigid_body=body, colliders=list(colliders), joint=joint)))
    obj.visible_get = lambda **_kwargs: True
    obj.select_get = lambda **_kwargs: True
    obj.evaluated_get = lambda _depsgraph: obj
    return obj


@pytest.fixture
def scene(monkeypatch):
    """A body with a box and a hull collider, and a hinge from that body to WORLD."""
    hull_calls = []
    body = _object("Body", 11, _matrix((1, 2, 3), (.9, .1, .2, .3)), colliders=(
        _record(persistent_id="box", shape="BOX", dimensions=(1, 2, 3)),
        _record(persistent_id="hull", shape="CONVEX_HULL")))
    settings = SimpleNamespace(
        persistent_id="hinge", show_indicator=True, type="REVOLUTE", axis="X", axes=(), enabled=True,
        frames_initialized=True, endpoint_a=SimpleNamespace(object=body), endpoint_b=SimpleNamespace(object=None),
        frame_a=SimpleNamespace(translation=(.25, 0, 0), rotation=(1, 0, 0, 0)),
        frame_b=SimpleNamespace(translation=(4, 4, 4), rotation=(1, 0, 0, 0)))
    hinge = _object("Hinge", 12, _matrix((4, 4, 4)), joint=settings)
    hinge.ov.physics.persistent_id = ""

    def resolve_joint(joint, scene, depsgraph):
        offset = RigidFrame((0, 0, .5))
        frame = evaluated_owner_frame(body, depsgraph, offset=offset)
        return (ResolvedEndpoint("OBJECT", "body", "/Bodies/b_body", frame, body, "DYNAMIC"),
                ResolvedEndpoint("WORLD", None, None, RigidFrame(), None, None))

    def hull_wire(source, depsgraph, scale):
        hull_calls.append(source.name)
        return (((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)),)

    monkeypatch.setattr(snapshot_module, "resolve_joint", resolve_joint)
    monkeypatch.setattr(snapshot_module, "validate_axes", lambda settings: None)
    monkeypatch.setattr(snapshot_module, "inactive_animation", lambda obj: ())
    monkeypatch.setattr(snapshot_module, "anchor_pending", lambda obj: False)
    monkeypatch.setattr(snapshot_module, "_hull_wire", hull_wire)
    scene = SimpleNamespace(session_uid=1, unit_settings=SimpleNamespace(scale_length=1.0),
                            objects=FakeObjects((body, hinge)))
    return SimpleNamespace(scene=scene, body=body, hinge=hinge, hull_calls=hull_calls,
                           view_layer=SimpleNamespace(name="ViewLayer"))


def _capture(fake, cache=None):
    return capture(fake.scene, fake.view_layer, object(), generation=1, viewport_id=5,
                   hulls=None if cache is None else cache.hulls)


def test_geometry_strokes_are_local_to_their_anchor(scene):
    cache = IndicatorCache()
    first_source = _capture(scene, cache)
    first = cache.refresh(first_source)
    scene.body.matrix_world = _matrix((-3, 0, 7), (.3, -.5, .1, .8))
    moved_source = _capture(scene, cache)
    moved = geometry(moved_source)
    # Only the transform table moves: strokes, links and labels compare equal.
    assert moved.strokes == first.strokes and moved.links == first.links and moved.labels == first.labels
    assert cache.refresh(moved_source) is first
    assert {a.identity for a in moved_source.anchors} == {"body"}
    anchor = moved_source.anchors[0]
    assert anchor != first_source.anchors[0] and anchor.runtime_owned and anchor.owner == "Body" and anchor.owner_uid == 11
    assert all(s.anchor == "body" for s in moved.strokes if s.role in ("collider-wire", "hinge-glyph", "frame-a-marker"))
    assert all(s.anchor is None for s in moved.strokes if s.role == "frame-b-marker")
    # The anchor's frame is the body frame (owner + offset); local geometry rides on it.
    body = RigidFrame((-3, 0, 7), (.3, -.5, .1, .8)).compose(RigidFrame((0, 0, .5)))
    assert _close((anchor.frame.position,), (body.position,))
    box = next(s for s in moved.strokes if s.identity == "body:box" and s.role == "collider-wire")
    assert _close(world(box, {"body": anchor.frame}), tuple(point(body, v) for v in box.vertices))
    # Re-reading the anchor without a capture gives the same frame.
    assert snapshot_module.anchor_world_frame(anchor, scene.scene, object()) == anchor.frame


def test_hull_wire_cached_across_pose_only_refresh(scene):
    cache = IndicatorCache()
    for index in range(10):
        scene.body.matrix_world = _matrix((index, 0, 0), (1, 0, .1*index, 0))
        source = _capture(scene, cache)
        cache.refresh(source)
    assert scene.hull_calls == ["Body"]
    hull = next(c for c in source.colliders if c.shape == "CONVEX_HULL")
    assert hull.wire == (((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)),) and not hull.error
    # Without a cache, every capture computes the hull again.
    _capture(scene)
    assert scene.hull_calls == ["Body", "Body"]


def test_hull_wire_rebuilt_after_geometry_update(scene):
    cache = IndicatorCache()
    _capture(scene, cache)
    cache.hulls.note_geometry_update(scene.hinge.session_uid)
    _capture(scene, cache)
    assert scene.hull_calls == ["Body"]
    cache.hulls.note_geometry_update(scene.body.session_uid)
    _capture(scene, cache)
    assert scene.hull_calls == ["Body", "Body"]
    _capture(scene, cache)
    assert scene.hull_calls == ["Body", "Body"]


def test_hull_cache_cleared_by_invalidate(scene):
    cache = IndicatorCache()
    _capture(scene, cache)
    cache.invalidate()
    _capture(scene, cache)
    assert scene.hull_calls == ["Body", "Body"]
    cache.invalidate(scene=1)
    _capture(scene, cache)
    assert scene.hull_calls == ["Body", "Body", "Body"]


def test_failed_hull_is_cached_as_its_error(scene, monkeypatch):
    calls = []

    def failing(source, depsgraph, scale):
        calls.append(source.name)
        raise SnapshotError("Hull source requires at least four vertices")

    monkeypatch.setattr(snapshot_module, "_hull_wire", failing)
    cache = IndicatorCache()
    for _ in range(3):
        source = _capture(scene, cache)
    assert calls == ["Body"]
    assert next(c for c in source.colliders if c.shape == "CONVEX_HULL").error == "Hull source requires at least four vertices"

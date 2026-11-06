# SPDX-License-Identifier: Apache-2.0
"""Body-local indicator batches: cached across refreshes, drawn with per-anchor transforms.

``physics_overlay`` runs against fake ``bpy``, ``gpu`` and ``mathutils`` modules.
The snapshot geometry is real; only the Blender capture is replaced by Sources
built here, so pose-only and shape changes can be stepped exactly.
"""
from contextlib import contextmanager
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

# Import the bpy-free snapshot dependencies before any fake bpy is installed.
from ovrtx_blender_example.physics_authoring_snapshot import (
    Anchor, Collider, Joint, Scope, Source,
)
from ovrtx_blender_example.physics_frames import RigidFrame

ROOT = Path(__file__).resolve().parents[1]
MODULE = "ovrtx_blender_example.physics_overlay"


class Matrix:
    """Pure-Python stand-in for mathutils.Matrix (4x4, row-major)."""

    def __init__(self, rows):
        self.rows = tuple(tuple(float(v) for v in row) for row in rows)

    def __matmul__(self, other):
        if isinstance(other, Matrix):
            return Matrix(tuple(tuple(sum(self.rows[i][k]*other.rows[k][j] for k in range(4)) for j in range(4))
                                for i in range(4)))
        v = tuple(other) + (1.0,)
        return Vector(sum(self.rows[i][k]*v[k] for k in range(4)) for i in range(3))

    def __eq__(self, other):
        return isinstance(other, Matrix) and self.rows == other.rows

    __hash__ = None


class Vector(tuple):
    def __new__(cls, values):
        return super().__new__(cls, (float(v) for v in values))


class Pointer(SimpleNamespace):
    def as_pointer(self):
        return id(self)


class FakeTimers:
    def __init__(self):
        self.registered = []

    def register(self, fn, first_interval=0.0, persistent=False):
        self.registered.append(fn)

    def is_registered(self, fn):
        return fn in self.registered

    def unregister(self, fn):
        self.registered.remove(fn)


class FakeObjects(list):
    def get(self, name):
        return next((o for o in self if o.name == name), None)


class FakeGpu:
    """Records the model matrix in effect for every batch draw."""

    def __init__(self):
        self.stack = [Matrix(RigidFrame().column_matrix())]
        self.draws = []
        self.binds = 0
        self.uniforms = []
        gpu = self
        self.state = SimpleNamespace(
            blend_get=lambda: "NONE", depth_test_get=lambda: "NONE", depth_mask_get=lambda: True,
            blend_set=lambda value: None, depth_test_set=lambda value: None, depth_mask_set=lambda value: None,
            viewport_get=lambda: (0, 0, 640, 480))

        @contextmanager
        def push_pop():
            gpu.stack.append(gpu.stack[-1])
            try:
                yield
            finally:
                gpu.stack.pop()

        def multiply_matrix(matrix):
            assert isinstance(matrix, Matrix)
            gpu.stack[-1] = gpu.stack[-1] @ matrix

        self.matrix = SimpleNamespace(push_pop=push_pop, multiply_matrix=multiply_matrix)
        self.shader = SimpleNamespace(from_builtin=lambda name: Shader(gpu), unbind=lambda: None)


class Shader:
    def __init__(self, gpu):
        self.gpu = gpu
        self.color = None

    def bind(self):
        self.gpu.binds += 1

    def uniform_float(self, name, value):
        if name == "color":
            self.color = value
        self.gpu.uniforms.append(name)


class Batch:
    def __init__(self, gpu, content):
        self.gpu, self.content = gpu, content

    def draw(self, shader):
        self.gpu.draws.append((self, self.gpu.stack[-1], shader.color))


def body(identity, uid, position, rotation=(1, 0, 0, 0)):
    return Anchor(identity, RigidFrame(position, rotation), identity == "cube", identity.title(), uid)


def make_source(cube=((1, 2, 3), (.9, .1, .2, .3)), arm=((-4, 0, 1), (.5, .5, .5, .5)), box=(1, 2, 3)):
    """A box on 'cube', a sphere on 'arm', and a hinge from cube to arm."""
    anchors = (body("cube", 11, *cube), body("arm", 12, *arm))
    colliders = (Collider("cube:box", "Cube", "BOX", RigidFrame((0, 0, .5)), dimensions=box, anchor="cube"),
                 Collider("arm:ball", "Arm", "ANALYTIC_SPHERE", RigidFrame((1, 0, 0), (0, 0, 1, 0)), radius=.5,
                          anchor="arm"))
    hinge = Joint("hinge", "Hinge", "REVOLUTE", "X", RigidFrame(), RigidFrame(),
                  RigidFrame((.25, 0, 0)), RigidFrame((-.5, 0, 0), (0, 1, 0, 0)), anchor_a="cube", anchor_b="arm")
    return Source(Scope(7, "ViewLayer", 0, 0), (hinge,), colliders, 0.25, (), anchors)


def frame_matrix(frame):
    return Matrix(frame.column_matrix())


@pytest.fixture
def overlay(monkeypatch):
    region = Pointer(type="WINDOW")
    space = Pointer(local_view=None, overlay=SimpleNamespace(show_overlays=True), shading=SimpleNamespace(type="RENDERED"))
    area = Pointer(type="VIEW_3D", regions=(region,), space=space)
    area.tag_redraw = lambda: None
    window = Pointer(screen=SimpleNamespace(areas=(area,)))
    settings = SimpleNamespace(indicators_enabled=True, indicators_selected_only=False,
                               indicators_colliders=True, indicators_size=.25)
    objects = FakeObjects()
    scene = Pointer(session_uid=7, ov=settings, objects=objects, unit_settings=SimpleNamespace(scale_length=1.0))
    view_layer = SimpleNamespace(name="ViewLayer")
    handlers = SimpleNamespace(load_pre=[], undo_pre=[], redo_pre=[], depsgraph_update_post=[],
                               frame_change_post=[], persistent=lambda fn: fn)
    context = SimpleNamespace(window_manager=SimpleNamespace(windows=(window,)), area=None, region=None)

    @contextmanager
    def temp_override(**values):
        prior = dict(vars(context))
        values.setdefault("scene", scene)
        values.setdefault("view_layer", view_layer)
        values.setdefault("space_data", values["area"].space)
        vars(context).update(values)
        context.evaluated_depsgraph_get = lambda: object()
        try:
            yield
        finally:
            vars(context).clear()
            vars(context).update(prior)

    context.temp_override = temp_override
    gpu = FakeGpu()
    bpy = SimpleNamespace(
        app=SimpleNamespace(background=False, timers=FakeTimers(), handlers=handlers),
        utils=SimpleNamespace(system_resource=lambda *_args, **_kwargs: str(ROOT)),
        types=SimpleNamespace(SpaceView3D=SimpleNamespace(
            draw_handler_add=lambda *args: object(), draw_handler_remove=lambda *args: None)),
        context=context)
    batches = []

    def batch_for_shader(shader, kind, content):
        assert kind == "LINES"
        batch = Batch(gpu, content)
        batches.append(batch)
        return batch

    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setitem(sys.modules, "bpy.app", bpy.app)
    monkeypatch.setitem(sys.modules, "bpy.app.handlers", handlers)
    monkeypatch.setitem(sys.modules, "blf", SimpleNamespace(load=lambda path: 1, unload=lambda path: None))
    monkeypatch.setitem(sys.modules, "gpu", gpu)
    monkeypatch.setitem(sys.modules, "gpu_extras", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "gpu_extras.batch", SimpleNamespace(batch_for_shader=batch_for_shader))
    monkeypatch.setitem(sys.modules, "mathutils", SimpleNamespace(Matrix=Matrix, Vector=Vector))
    kinematics = SimpleNamespace(active=False)
    kinematics.evaluating = lambda: kinematics.active
    monkeypatch.setitem(sys.modules, "ovrtx_blender_example.physics_live_kinematics", kinematics)
    spec = importlib.util.spec_from_file_location(MODULE, ROOT / "addon/ovrtx_blender_example/physics_overlay.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, MODULE, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_generation", lambda scene: 0)
    state = SimpleNamespace(source=make_source(), captures=0)

    def capture(*_args, **kwargs):
        state.captures += 1
        assert kwargs["hulls"] is module._service.cache.hulls
        return state.source

    monkeypatch.setattr(module, "capture", capture)
    module.register()
    service = module._service
    with temp_override(window=window, area=area, region=region):
        # The first viewport draw creates the shader; nothing is prepared yet.
        assert module.draw_indicators(context) is False
    assert service.shader is not None
    yield SimpleNamespace(module=module, service=service, bpy=bpy, gpu=gpu, batches=batches, state=state,
                          window=window, area=area, region=region, scene=scene, handlers=handlers,
                          kinematics=kinematics)
    module.unregister()


def view(overlay):
    return overlay.bpy.context.temp_override(window=overlay.window, area=overlay.area, region=overlay.region)


def entry(overlay):
    with view(overlay):
        return overlay.service.current()


def draw(overlay, transforms=None):
    overlay.gpu.draws.clear()
    with view(overlay):
        assert overlay.module.draw_indicators(overlay.bpy.context, transforms)
    return list(overlay.gpu.draws)


def anchors(source):
    return {a.identity: a.frame for a in source.anchors}


def close(a, b, tolerance=1e-9):
    return all(abs(x - y) <= tolerance for x, y in zip(a, b))


def same(matrix, frame, tolerance=1e-12):
    return all(close(a, b, tolerance) for a, b in zip(matrix.rows, frame.column_matrix()))


def test_pose_only_refresh_creates_no_batches(overlay):
    overlay.service.refresh()
    first = entry(overlay)
    created = len(overlay.batches)
    # One batch per stroke and the unit-segment batch for links.
    assert created == len(first.strokes) + 1 and first.links
    for step in range(10):
        angle = .3 * (step + 1)
        overlay.state.source = make_source(cube=((step, 1, 2), (math.cos(angle), 0, math.sin(angle), 0)),
                                           arm=((-step, 0, 1), (1, step * .1, 0, 0)))
        overlay.service.refresh()
    assert len(overlay.batches) == created
    current = entry(overlay)
    assert [batch for _stroke, batch, _local in current.strokes] == [batch for _stroke, batch, _local in first.strokes]
    # The table holds the latest pose; the draw uses it.
    assert current.transforms.matrices["cube"] == frame_matrix(anchors(overlay.state.source)["cube"])


def test_shape_change_rebuilds_only_that_stroke(overlay):
    overlay.service.refresh()
    before = entry(overlay).strokes
    created = len(overlay.batches)
    overlay.state.source = make_source(box=(1, 2, 4))
    overlay.service.refresh()
    assert len(overlay.batches) == created + 1
    after = entry(overlay).strokes
    assert len(after) == len(before)
    changed = [(new[0].identity, new[0].role) for old, new in zip(before, after) if new[1] is not old[1]]
    assert changed == [("cube:box", "collider-wire")]
    replaced = next(old[1] for old, new in zip(before, after) if new[1] is not old[1])
    assert next(new[1] for old, new in zip(before, after) if new[1] is not old[1]) is overlay.batches[-1]
    # The replaced stroke's batch is no longer cached.
    assert replaced not in overlay.service.batches.values()
    assert len(overlay.service.batches) == len({id(batch) for _stroke, batch, _local in after})


def test_draw_uses_anchor_world_times_local_frame(overlay):
    overlay.service.refresh()
    prepared = entry(overlay)
    frames = anchors(overlay.state.source)
    draws = draw(overlay)
    assert overlay.gpu.binds == 1
    strokes = [d for d in draws if d[0] is not overlay.service.unit_link]
    assert len(strokes) == len(prepared.strokes)
    for (stroke, batch, _local), (drawn, model, color) in zip(prepared.strokes, strokes):
        assert drawn is batch and color == stroke.color
        frame = stroke.local_frame if stroke.anchor is None else frames[stroke.anchor].compose(stroke.local_frame)
        # Each local vertex lands where the anchor and local frame put it.
        for vertex in stroke.vertices[:4]:
            assert close(model @ Vector(vertex), frame.compose(RigidFrame(vertex)).position, 1e-9)
    links = [d for d in draws if d[0] is overlay.service.unit_link]
    assert len(links) == len(prepared.links) == 3
    for link, (_batch, model, color) in zip(prepared.links, links):
        expected = [frames[anchor].compose(RigidFrame(p)).position for anchor, p in (link.start, link.end)]
        assert close(model @ Vector((0, 0, 0)), expected[0]) and close(model @ Vector((1, 0, 0)), expected[1])
        assert color == link.color
    # Overrides (for example presented packet poses) replace the table for their anchors only.
    moved = RigidFrame((9, 9, 9))
    draws = draw(overlay, {"cube": frame_matrix(moved)})
    box = next(d for d, (stroke, _b, _l) in zip(draws, prepared.strokes) if stroke.identity == "cube:box")
    vertex = next(s for s, _b, _l in prepared.strokes if s.identity == "cube:box").vertices[0]
    assert close(box[1] @ Vector(vertex), moved.compose(RigidFrame((0, 0, .5))).compose(RigidFrame(vertex)).position)


@pytest.mark.parametrize("start, end", [
    ((1, 2, 3), (4, -2, 8)),
    ((1, 2, 3), (1, 2, 3)),          # zero length
    ((1, 2, 3), (-5, 2, 3)),         # antiparallel to the unit segment
    ((0, 0, 0), (0, 0, -2)),         # along -Z
    ((0, 0, 0), (0, 0, 3)),          # along +Z
])
def test_link_unit_segment_matrix_maps_endpoints(overlay, start, end):
    rows = overlay.module.link_matrix(start, end)
    matrix = Matrix(rows)
    assert close(matrix @ Vector((0, 0, 0)), start) and close(matrix @ Vector((1, 0, 0)), end)
    assert rows[3] == (0.0, 0.0, 0.0, 1.0)
    # Location, rotation and an X scale only: Y and Z stay orthonormal, X is orthogonal to them.
    columns = [tuple(rows[r][c] for r in range(3)) for c in range(3)]
    dot = lambda a, b: sum(x*y for x, y in zip(a, b))
    assert abs(dot(columns[1], columns[1]) - 1) < 1e-12 and abs(dot(columns[2], columns[2]) - 1) < 1e-12
    assert abs(dot(columns[0], columns[1])) < 1e-12 and abs(dot(columns[0], columns[2])) < 1e-12
    assert abs(dot(columns[1], columns[2])) < 1e-12
    length = math.dist(start, end)
    assert abs(math.sqrt(dot(columns[0], columns[0])) - length) < 1e-12
    if length:
        x, y, z = columns
        cross = (y[1]*z[2]-y[2]*z[1], y[2]*z[0]-y[0]*z[2], y[0]*z[1]-y[1]*z[0])
        assert close(cross, tuple(v/length for v in x), 1e-12)  # right-handed


def evaluated_object(name, uid, frame):
    obj = SimpleNamespace(name=name, session_uid=uid, scale=(1, 1, 1), parent=None)
    obj.matrix_world = SimpleNamespace(copy=lambda: frame.column_matrix())
    obj.evaluated_get = lambda _depsgraph: obj
    return obj


def test_transform_table_updates_on_frame_change_without_capture(overlay):
    overlay.service.refresh()
    prepared = entry(overlay)
    captures, created = overlay.state.captures, len(overlay.batches)
    assert overlay.handlers.frame_change_post == [overlay.module._frame_changed]
    owner = RigidFrame((5, 6, 7), (.6, 0, .8, 0))
    cube = evaluated_object("Cube", 11, owner)
    overlay.scene.objects[:] = [cube, evaluated_object("Arm", 12, RigidFrame((-4, 0, 1), (.5, .5, .5, .5)))]
    depsgraph = SimpleNamespace(view_layer=SimpleNamespace(name="ViewLayer"), updates=())
    # Skipped while kinematic prefetch evaluates transient frames.
    overlay.kinematics.active = True
    for handler in overlay.handlers.frame_change_post:
        handler(overlay.scene, depsgraph)
    assert prepared.transforms.matrices["cube"] == frame_matrix(anchors(overlay.state.source)["cube"])
    overlay.kinematics.active = False
    for handler in overlay.handlers.frame_change_post:
        handler(overlay.scene, depsgraph)
    assert entry(overlay) is prepared
    assert same(prepared.transforms.matrices["cube"], owner)
    assert overlay.state.captures == captures and len(overlay.batches) == created
    draws = draw(overlay)
    box = next(d for d, (stroke, _b, _l) in zip(draws, prepared.strokes) if stroke.identity == "cube:box")
    vertex = next(s for s, _b, _l in prepared.strokes if s.identity == "cube:box").vertices[0]
    assert close(box[1] @ Vector(vertex), owner.compose(RigidFrame((0, 0, .5))).compose(RigidFrame(vertex)).position)
    # depsgraph_update_post moves the table synchronously too; the capture stays deferred.
    owner = RigidFrame((-1, -2, -3))
    cube.matrix_world = SimpleNamespace(copy=lambda: owner.column_matrix())
    for handler in overlay.handlers.depsgraph_update_post:
        handler(overlay.scene, depsgraph)
    assert same(prepared.transforms.matrices["cube"], owner)
    assert overlay.state.captures == captures and overlay.service.dirty
    # Another view layer's depsgraph leaves this view's table alone.
    cube.matrix_world = SimpleNamespace(copy=lambda: RigidFrame((3, 3, 3)).column_matrix())
    overlay.module._frame_changed(overlay.scene, SimpleNamespace(view_layer=SimpleNamespace(name="Other"), updates=()))
    assert same(prepared.transforms.matrices["cube"], owner)


def test_depsgraph_geometry_update_bumps_hull_generation(overlay):
    hulls = overlay.service.cache.hulls
    geometry = lambda uid, flag: SimpleNamespace(is_updated_geometry=flag, id=SimpleNamespace(
        id_type="OBJECT", session_uid=uid, original=SimpleNamespace(session_uid=uid)))
    depsgraph = SimpleNamespace(view_layer=SimpleNamespace(name="ViewLayer"),
                                updates=(geometry(21, True), geometry(22, False),
                                         SimpleNamespace(is_updated_geometry=True, id=SimpleNamespace(id_type="MESH"))))
    overlay.module._invalidate(overlay.scene, depsgraph)
    assert hulls._generations == {21: 1}


def test_load_clears_cached_batches(overlay):
    overlay.service.refresh()
    assert overlay.service.batches
    unit = overlay.service.unit_link
    for handler in overlay.handlers.load_pre:
        handler(overlay.scene, None)
    assert not overlay.service.batches and not overlay.service.prepared
    created = len(overlay.batches)
    overlay.service.refresh()
    # Every stroke batch is rebuilt after a load; the unit segment is kept per service.
    assert len(overlay.batches) == created + len(entry(overlay).strokes)
    assert overlay.service.unit_link is unit

# SPDX-License-Identifier: Apache-2.0
"""Depsgraph updates keep prepared indicator lines; load/undo/redo still clear them.

``physics_overlay`` is loaded against a fake ``bpy`` so the handler wiring, the
coalesced refresh timer and the in-place batch swap run without Blender.
"""
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

# Import the bpy-free snapshot dependencies before any fake bpy is installed.
from ovrtx_blender_example.physics_authoring_snapshot import Stroke

ROOT = Path(__file__).resolve().parents[1]
MODULE = "ovrtx_blender_example.physics_overlay"


class Pointer(SimpleNamespace):
    def as_pointer(self):
        return id(self)


class FakeTimers:
    """``bpy.app.timers`` stand-in: records registrations, fires on demand."""

    def __init__(self):
        self.registered = []
        self.register_calls = []

    def register(self, fn, first_interval=0.0, persistent=False):
        self.register_calls.append((fn, first_interval, persistent))
        self.registered.append(fn)

    def is_registered(self, fn):
        return fn in self.registered

    def unregister(self, fn):
        self.registered.remove(fn)

    def run(self, fn):
        # A callback returning None unregisters itself, as in Blender.
        if fn() is None:
            self.registered.remove(fn)


class FakeCache:
    def __init__(self):
        self.invalidations = 0
        self.hulls = None

    def refresh(self, source):
        return source.snapshot

    def invalidate(self, *, scene=None):
        self.invalidations += 1


def viewport():
    settings = SimpleNamespace(indicators_enabled=True, indicators_selected_only=False,
                               indicators_colliders=True, indicators_size=1.0)
    scene = Pointer(session_uid=7, ov=settings)
    space = Pointer(local_view=None, overlay=SimpleNamespace(show_overlays=True))
    region = Pointer(type="WINDOW")
    area = Pointer(type="VIEW_3D", regions=(region,), redraws=0, space=space)
    area.tag_redraw = lambda: setattr(area, "redraws", area.redraws + 1)
    window = Pointer(screen=SimpleNamespace(areas=(area,)))
    return window, area, region, scene


@pytest.fixture
def overlay(monkeypatch):
    window, area, region, scene = viewport()
    timers = FakeTimers()
    handlers = SimpleNamespace(load_pre=[], undo_pre=[], redo_pre=[], depsgraph_update_post=[],
                               frame_change_post=[], persistent=lambda fn: fn)
    context = SimpleNamespace(window_manager=SimpleNamespace(windows=(window,)))

    @contextmanager
    def temp_override(**values):
        prior = dict(vars(context))
        values.setdefault("scene", scene)
        values.setdefault("view_layer", SimpleNamespace(name="ViewLayer"))
        values.setdefault("space_data", values["area"].space)
        vars(context).update(values)
        context.evaluated_depsgraph_get = lambda: object()
        try:
            yield
        finally:
            vars(context).clear()
            vars(context).update(prior)

    context.temp_override = temp_override
    bpy = SimpleNamespace(
        app=SimpleNamespace(background=False, timers=timers, handlers=handlers),
        utils=SimpleNamespace(system_resource=lambda *_args, **_kwargs: str(ROOT)),
        types=SimpleNamespace(SpaceView3D=SimpleNamespace(
            draw_handler_add=lambda *args: object(), draw_handler_remove=lambda *args: None)),
        context=context)
    batches = []

    def batch_for_shader(shader, kind, content):
        batch = SimpleNamespace(content=content)
        batches.append(batch)
        return batch

    monkeypatch.setitem(sys.modules, "bpy", bpy)
    monkeypatch.setitem(sys.modules, "bpy.app", bpy.app)
    monkeypatch.setitem(sys.modules, "bpy.app.handlers", handlers)
    monkeypatch.setitem(sys.modules, "blf", SimpleNamespace(load=lambda path: 1, unload=lambda path: None))
    monkeypatch.setitem(sys.modules, "gpu_extras", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "gpu_extras.batch", SimpleNamespace(batch_for_shader=batch_for_shader))
    monkeypatch.setitem(sys.modules, "mathutils", SimpleNamespace(Matrix=lambda rows: tuple(map(tuple, rows))))
    spec = importlib.util.spec_from_file_location(MODULE, ROOT / "addon/ovrtx_blender_example/physics_overlay.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, MODULE, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_generation", lambda scene: 0)
    module.register()
    service = module._service
    service.cache = FakeCache()
    service.shader = object()
    yield SimpleNamespace(module=module, service=service, timers=timers, handlers=handlers, bpy=bpy,
                          window=window, area=area, region=region, scene=scene, batches=batches)
    module.unregister()


def snapshot(*strokes):
    return SimpleNamespace(strokes=tuple(Stroke("collider-wire", str(i), vertices, (1, 0, 0, 1))
                                         for i, vertices in enumerate(strokes)), links=(), labels=())


def use_snapshot(monkeypatch, overlay, value, observe=None):
    def capture(*_args, **_kwargs):
        if observe is not None:
            observe()
        return SimpleNamespace(snapshot=value, anchors=())
    monkeypatch.setattr(overlay.module, "capture", capture)


def current(overlay):
    with overlay.bpy.context.temp_override(window=overlay.window, area=overlay.area, region=overlay.region):
        return overlay.service.current()


def prepare(monkeypatch, overlay):
    use_snapshot(monkeypatch, overlay, snapshot(((0, 0, 0), (1, 0, 0))))
    assert overlay.service.refresh() == 0.1
    entry = current(overlay)
    assert entry is not None and entry.strokes
    return entry


def periodic(overlay):
    return [call for call in overlay.timers.register_calls if call[0] is overlay.service.timer]


def test_depsgraph_update_keeps_prepared_lines(monkeypatch, overlay):
    entry = prepare(monkeypatch, overlay)
    prepared = dict(overlay.service.prepared)
    assert overlay.handlers.depsgraph_update_post == [overlay.module._invalidate]
    invalidations = overlay.service.cache.invalidations
    for handler in overlay.handlers.depsgraph_update_post:
        handler(overlay.scene, object())
    assert overlay.service.prepared == prepared
    assert overlay.service.cache.invalidations == invalidations
    assert current(overlay) is entry
    assert overlay.service.dirty


def test_depsgraph_updates_coalesce_into_one_refresh_timer(monkeypatch, overlay):
    prepare(monkeypatch, overlay)
    captures = []
    use_snapshot(monkeypatch, overlay, snapshot(((0, 0, 0), (2, 0, 0))), lambda: captures.append(1))
    before = len(overlay.timers.register_calls)
    for _ in range(5):
        overlay.module._invalidate(overlay.scene, object())
    calls = overlay.timers.register_calls[before:]
    # The handler never captures; it registers exactly one zero-interval refresh.
    assert captures == []
    assert [(fn, interval) for fn, interval, _persistent in calls] == [(overlay.service.pending_timer, 0.0)]
    overlay.timers.run(overlay.service.pending_timer)
    assert captures == [1]
    assert not overlay.timers.is_registered(overlay.service.pending_timer)
    # Once it has run, the next update registers a new one.
    overlay.module._invalidate(overlay.scene, object())
    assert len(overlay.timers.register_calls) == before + 2
    # The 0.1 s periodic timer remains the fallback.
    assert len(periodic(overlay)) == 1 and overlay.timers.is_registered(overlay.service.timer)


def test_load_undo_redo_still_clear(monkeypatch, overlay):
    for name in ("load_pre", "undo_pre", "redo_pre"):
        prepare(monkeypatch, overlay)
        invalidations = overlay.service.cache.invalidations
        handlers = getattr(overlay.handlers, name)
        assert handlers == [overlay.module._clear]
        for handler in handlers:
            handler(overlay.scene, None)
        assert overlay.service.prepared == {}, name
        assert overlay.service.cache.invalidations == invalidations + 1, name
        assert current(overlay) is None


def test_refresh_replaces_lines_in_place(monkeypatch, overlay):
    old = prepare(monkeypatch, overlay)
    seen = []
    use_snapshot(monkeypatch, overlay, snapshot(((0, 0, 0), (0, 3, 0)), ((0, 0, 0), (0, 0, 3))),
                 lambda: seen.append(current(overlay)))
    builder = sys.modules["gpu_extras.batch"]
    original = builder.batch_for_shader

    def observed(*args):
        # Batches are built before the swap; the old entry must still draw meanwhile.
        seen.append(current(overlay))
        return original(*args)

    monkeypatch.setattr(builder, "batch_for_shader", observed)
    overlay.module._invalidate(overlay.scene, object())
    overlay.timers.run(overlay.service.pending_timer)
    new = current(overlay)
    assert len(seen) == 3 and all(entry is old for entry in seen)
    assert new is not None and new is not old
    assert [batch.content["pos"] for _stroke, batch, _local in new.strokes] == [((0, 0, 0), (0, 3, 0)), ((0, 0, 0), (0, 0, 3))]
    assert overlay.area.redraws >= 2
    assert not overlay.service.dirty

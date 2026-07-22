# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime scheduler on the render thread and the pose-mirror handoff (task02-07).

The scheduler tick runs on the render thread; its result crosses to the
main thread through ``engine._handoff_runtime_tick_result`` — a data-only
render-thread half that registers a coalesced one-shot ``bpy.app.timers``
absorb callback (the documented thread-safe crossing). Every Blender data
read and write stays on the main thread: the scene-object scan in
``operator_state.prepare_runtime_pose_mirror`` (audit finding: it reads
``scene.objects`` and id-properties), the pose apply, and the physics
playback-lock transitions (``clear`` on the initial-condition frame,
``lock_object`` during mirroring).
"""

from __future__ import annotations

import importlib.util
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

import ovrtx_blender_example  # noqa: E402,F401
from ovrtx_blender_example import ovrtx_session_controller as controller_module  # noqa: E402
from ovrtx_blender_example.ovrtx_runtime_client import RenderResult  # noqa: E402
from ovrtx_blender_example.ovrtx_value_updates import OvrtxValueUpdateResult  # noqa: E402
from ovrtx_blender_example.runtime_scheduler import (  # noqa: E402
    EditSubmissionResult,
    RuntimeTickResult,
    RuntimeTickStatus,
)
from ovrtx_blender_example.interactive_edit_planner import EditStatus  # noqa: E402
from ovrtx_blender_example.shared_stage_composition import BodyPose  # noqa: E402


WAIT_S = 5.0


# ---------------------------------------------------------------------------
# Fakes: bpy (timers, scene, windows), mathutils, scene objects
# ---------------------------------------------------------------------------


class _FakeRenderEngine:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.stats: list[tuple[str, str]] = []
        self.redraw_count = 0
        self.reports: list[tuple[set[str], str]] = []

    def update_stats(self, engine: str, message: str) -> None:
        self.stats.append((engine, message))

    def tag_redraw(self) -> None:
        self.redraw_count += 1

    def report(self, levels: set[str], message: str) -> None:
        self.reports.append((set(levels), message))


class _FakeTimers:
    """``bpy.app.timers`` stand-in: records registrations, fires on demand."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.pending: list[object] = []
        self.register_calls = 0

    def register(self, fn, first_interval: float = 0.0) -> None:
        with self._lock:
            self.register_calls += 1
            self.pending.append(fn)

    def run_pending(self) -> int:
        """Fire pending timer callbacks as Blender's main loop would."""

        with self._lock:
            callbacks = list(self.pending)
            self.pending.clear()
        recurring: list[object] = []
        for fn in callbacks:
            interval = fn()
            if interval is not None:
                assert interval >= 0.0
                recurring.append(fn)
        with self._lock:
            self.pending.extend(recurring)
        return len(callbacks)


class _FakeQuaternion:
    def __init__(self, wxyz) -> None:
        self.wxyz = tuple(float(value) for value in wxyz)


class _FakeVector(tuple):
    def __new__(cls, values):
        return super().__new__(cls, tuple(float(value) for value in values))


class _FakeMatrix:
    """Minimal mathutils.Matrix stand-in for LocRotScale pose application."""

    def __init__(self, rows=None) -> None:
        self.rows = rows
        self.loc: tuple[float, ...] = (0.0, 0.0, 0.0)
        self.quat: _FakeQuaternion | None = None
        self.scale: tuple[float, ...] = (1.0, 1.0, 1.0)

    @classmethod
    def LocRotScale(cls, loc, quat, scale):
        matrix = cls()
        matrix.loc = tuple(float(value) for value in loc)
        matrix.quat = quat
        matrix.scale = tuple(float(value) for value in scale)
        return matrix

    def decompose(self):
        return (self.loc, self.quat, self.scale)

    def copy(self):
        return self

    def __matmul__(self, other):
        return self


_FAKE_MATHUTILS = SimpleNamespace(
    Matrix=_FakeMatrix,
    Quaternion=_FakeQuaternion,
    Vector=_FakeVector,
)


class _RecordingSceneObject:
    """Scene object recording the thread ident of every id-property read."""

    def __init__(self, name: str, prim_path: str, access_log: list) -> None:
        self.name = name
        self._props = {"ovrtx.usd_prim_path": prim_path}
        self._access_log = access_log
        self.matrix_world = _FakeMatrix()
        self.lock_location = [False, False, False]
        self.lock_rotation = [False, False, False]
        self.lock_scale = [False, False, False]

    def get(self, key: str, default=None):
        self._access_log.append((key, threading.get_ident()))
        return self._props.get(key, default)


def _load_engine_with_fake_bpy(monkeypatch: pytest.MonkeyPatch):
    module_name = "ovrtx_blender_example._engine_pose_mirror_handoff_test"
    module_path = ROOT / "addon" / "ovrtx_blender_example" / "engine.py"
    access_log: list = []
    view3d_area = SimpleNamespace(type="VIEW_3D", redraws=[])
    view3d_area.tag_redraw = lambda: view3d_area.redraws.append(1)
    scene = SimpleNamespace(
        name="Scene",
        objects=[_RecordingSceneObject("Cube", "/World/Cube", access_log)],
    )
    window = SimpleNamespace(screen=SimpleNamespace(areas=(view3d_area,)))
    fake_bpy = SimpleNamespace(
        types=SimpleNamespace(RenderEngine=_FakeRenderEngine),
        app=SimpleNamespace(timers=_FakeTimers()),
        context=SimpleNamespace(
            scene=scene,
            view_layer=None,
            window_manager=SimpleNamespace(windows=(window,)),
        ),
        data=SimpleNamespace(scenes={"Scene": scene}),
    )
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    monkeypatch.setitem(sys.modules, "mathutils", _FAKE_MATHUTILS)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    module._test_scene = scene
    module._test_access_log = access_log
    module._test_view3d_area = view3d_area
    return module


def _tick_result(module, *, tx: float = 4.0, status=None, update=None):
    return module.RuntimeTickResult(
        status=status or RuntimeTickStatus.STEPPED,
        enabled=True,
        should_request_redraw=False,
        generation=5,
        physics_pose_set=(
            BodyPose("/World/Cube", (float(tx), 5.0, 6.0), (0.0, 0.0, 0.0, 1.0)),
        ),
        update=update or {},
    )


def _request(module, **overrides):
    fields = {
        "input_usd_path": "/tmp/scene.usda",
        "sensor_paths": ("/Render/Product",),
        "selected_sensor_paths": ("/Render/Product",),
        "width": 2,
        "height": 2,
        "min_samples": 1,
        "max_samples": 4,
        "camera_prim_path": "/World/Camera",
        "camera_matrix": (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (1.0, 0.0, 0.0, 1.0),
        ),
        "worker_command": "worker",
        "native_client_module": "client",
    }
    fields.update(overrides)
    return module.RenderRequest(**fields)


def _initial_condition_request(module):
    return _request(
        module,
        timeline_controls_enabled=True,
        timeline_playing=False,
        timeline_frame=1,
        timeline_start=1,
        timeline_end=50,
    )


def _handoff_from_worker(engine, result, request) -> int:
    """Run the render-thread half on a worker thread; return its ident."""

    idents: list[int] = []

    def _target() -> None:
        idents.append(threading.get_ident())
        engine._handoff_runtime_tick_result(result, request)

    worker = threading.Thread(target=_target, name="fake-render-thread", daemon=True)
    worker.start()
    worker.join(WAIT_S)
    assert not worker.is_alive()
    return idents[0]


# ---------------------------------------------------------------------------
# Handoff unit tests (worker thread -> fake timers -> main thread)
# ---------------------------------------------------------------------------


def test_handoff_coalesces_and_absorbs_newest_result_on_main_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_engine_with_fake_bpy(monkeypatch)
    timers = module.bpy.app.timers
    engine = module.OvrtxExampleRenderEngine()
    request = _request(module)

    worker_ident = _handoff_from_worker(engine, _tick_result(module, tx=1.0), request)
    _handoff_from_worker(engine, _tick_result(module, tx=9.0), request)
    # Coalesced: one absorb timer for the burst; latest handoff wins.
    assert timers.register_calls == 1

    assert timers.run_pending() == 1
    # The absorb prepared the mirror (main-thread scene scan) and
    # registered the existing pose-mirror apply timer.
    pending = engine._pending_pose_mirror
    assert pending["poses_by_path"]["/World/Cube"]["translate"] == (9.0, 5.0, 6.0)
    assert engine._pose_mirror["status"] == "scheduled"
    assert timers.register_calls == 2

    assert timers.run_pending() == 1
    obj = module._test_scene.objects[0]
    assert engine._pose_mirror["status"] == "applied"
    assert engine._pose_mirror["mirrored_paths"] == ["/World/Cube"]
    assert obj.matrix_world.loc == (9.0, 5.0, 6.0)
    # Mirroring locked the object under the runtime's authority (main
    # thread) and tagged the viewport redraw.
    lock = engine._physics_playback_lock.diagnostics()
    assert lock["active"] is True
    assert lock["locked_object_paths"] == ["/World/Cube"]
    assert lock["owning_physics_generation"] == 5
    assert module._test_view3d_area.redraws
    # Audit (task02-07): every Blender id-property read happened on this
    # (main) thread, never on the worker that handed the result off.
    read_idents = {ident for _key, ident in module._test_access_log}
    assert threading.get_ident() in read_idents
    assert worker_ident not in read_idents


def test_initial_condition_handoff_clears_playback_lock_on_main_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_engine_with_fake_bpy(monkeypatch)
    timers = module.bpy.app.timers
    engine = module.OvrtxExampleRenderEngine()
    obj = module._test_scene.objects[0]
    engine._physics_playback_lock.lock_object("/World/Cube", obj, generation=4)
    assert engine._physics_playback_lock.is_active() is True

    # The thread passes the at-initial-condition fact derived from the
    # snapshot-carried timeline fields; the clear itself runs in the timer.
    _handoff_from_worker(
        engine, _tick_result(module, tx=2.0), _initial_condition_request(module)
    )
    assert engine._physics_playback_lock.is_active() is True

    timers.run_pending()  # absorb: clear + prepare (lock_runtime_owned=False)
    diagnostics = engine._physics_playback_lock.diagnostics()
    assert diagnostics["frame1_cleared"] is True
    assert diagnostics["active"] is False

    timers.run_pending()  # apply: mirrors without re-locking
    assert engine._pose_mirror["status"] == "applied"
    assert obj.matrix_world.loc == (2.0, 5.0, 6.0)
    assert engine._physics_playback_lock.is_active() is False


def test_failed_tick_handoff_records_update_result_without_transitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_engine_with_fake_bpy(monkeypatch)
    timers = module.bpy.app.timers
    engine = module.OvrtxExampleRenderEngine()
    obj = module._test_scene.objects[0]
    engine._physics_playback_lock.lock_object("/World/Cube", obj, generation=4)
    recorded: list[dict] = []
    engine._interactive_edit_workflow = SimpleNamespace(
        record_update_result=lambda update: recorded.append(dict(update))
    )
    update = {"update_result": {"failed": True, "skipped_reason": "boom"}}

    _handoff_from_worker(
        engine,
        _tick_result(module, status=RuntimeTickStatus.FAILED, update=update),
        _initial_condition_request(module),
    )
    timers.run_pending()

    assert recorded == [{"failed": True, "skipped_reason": "boom"}]
    # FAILED ticks perform no lock transitions and schedule no mirror.
    assert engine._physics_playback_lock.is_active() is True
    assert engine._pending_pose_mirror == {}


def test_idle_tick_results_do_not_register_absorb_timers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_engine_with_fake_bpy(monkeypatch)
    timers = module.bpy.app.timers
    engine = module.OvrtxExampleRenderEngine()
    idle = module.RuntimeTickResult(
        status=RuntimeTickStatus.NOOP,
        enabled=True,
    )

    _handoff_from_worker(engine, idle, _request(module))

    assert timers.register_calls == 0
    assert engine._pending_tick_absorb is None


def test_handoff_registration_failure_releases_latch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_engine_with_fake_bpy(monkeypatch)
    timers = module.bpy.app.timers
    engine = module.OvrtxExampleRenderEngine()
    original_register = timers.register

    def _broken_register(fn, first_interval: float = 0.0) -> None:
        raise RuntimeError("timers unavailable")

    timers.register = _broken_register
    _handoff_from_worker(engine, _tick_result(module), _request(module))
    assert engine._tick_absorb_timer_pending is False

    # The next tick retries instead of wedging the handoff forever.
    timers.register = original_register
    _handoff_from_worker(engine, _tick_result(module), _request(module))
    assert timers.register_calls == 1
    assert timers.run_pending() == 1
    assert engine._pending_pose_mirror["poses_by_path"]["/World/Cube"]


def test_session_end_drops_pending_tick_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_engine_with_fake_bpy(monkeypatch)
    timers = module.bpy.app.timers
    engine = module.OvrtxExampleRenderEngine()
    engine._write_viewport_session_outputs = lambda **_kwargs: None

    _handoff_from_worker(engine, _tick_result(module), _request(module))
    assert engine._pending_tick_absorb is not None

    engine._end_viewport_session(module.ViewportSessionEndReason.ENGINE_DESTROYED)
    # The stopped session's pose set is dropped; the still-pending absorb
    # timer no-ops.
    timers.run_pending()
    assert engine._pending_pose_mirror == {}
    assert engine._pose_mirror == {}
    assert engine._runtime_tick_result is None


# ---------------------------------------------------------------------------
# End to end: tick on the render thread -> pose mirror via main-thread timer
# ---------------------------------------------------------------------------


class _FakeClient:
    """Fake srtx client compatible with OvrtxSessionController."""

    def __init__(self) -> None:
        self.render_calls = 0
        self.startup_diagnostics = {"render_worker": {"status": "running"}}
        self.last_render_timings: dict = {}
        self.last_value_update_timings: dict = {}

    def start_session(self, _spec: object, simulation_id: str | None = None) -> str:
        return simulation_id or "sim"

    def render_result(self, _simulation_id: str, **kwargs: object) -> RenderResult:
        self.render_calls += 1
        return RenderResult(
            width=2,
            height=2,
            rgba8=b"\x00" * 16,
            completed_samples=int(kwargs["additional_samples"]),
            session_completed_samples=self.render_calls,
            simulation_time_ns=0,
        )

    def update_transforms(self, _simulation_id: str, values) -> OvrtxValueUpdateResult:
        return OvrtxValueUpdateResult(len(tuple(values)), pending_simulation_time_ns=1)

    def delete_simulation(self, _simulation_id: str) -> str:
        return "stopped"

    def shutdown(self) -> None:
        return None


class _PoseProducingScheduler:
    """Fake physics: every tick reports the same body pose set."""

    has_pending_view_updates = False
    has_pending_sim_updates = False

    def __init__(self) -> None:
        self.tick_idents: list[int] = []
        self.shutdown_idents: list[int] = []

    def set_edit_wake_hook(self, hook) -> None:
        pass

    def submit_edit(self, intent) -> EditSubmissionResult:
        return EditSubmissionResult(status=EditStatus.QUEUED, reason="queued")

    def tick_viewport(self, request, *, ovrtx_updates, project_complete_pose):
        self.tick_idents.append(threading.get_ident())
        return RuntimeTickResult(
            status=RuntimeTickStatus.STEPPED,
            enabled=True,
            should_request_redraw=False,
            generation=7,
            physics_pose_set=(
                BodyPose("/World/Cube", (4.0, 5.0, 6.0), (0.0, 0.0, 0.0, 1.0)),
            ),
        )

    def shutdown(self) -> None:
        self.shutdown_idents.append(threading.get_ident())

    def diagnostics(self) -> dict:
        return {"enabled": True}


def test_tick_on_render_thread_mirrors_pose_via_main_thread_timer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acceptance seam: tick on thread → timer-applied mirror → redraw."""

    module = _load_engine_with_fake_bpy(monkeypatch)
    timers = module.bpy.app.timers
    client = _FakeClient()
    monkeypatch.setattr(
        controller_module,
        "_runtime_client_from_request",
        lambda _request: client,
    )
    engine = module.OvrtxExampleRenderEngine()
    scheduler = _PoseProducingScheduler()
    engine._runtime_scheduler = scheduler
    request = _request(module)
    try:
        engine._begin_async_viewport_session(request)
        engine._camera_mailbox.write(
            module.viewport_handoff.snapshot_from_render_request(request)
        )
        deadline = time.monotonic() + WAIT_S
        newest = None
        index = 0
        while time.monotonic() < deadline:
            candidate = engine._frame_slot.wait_for_newer(index, timeout=WAIT_S)
            if candidate is None:
                continue
            newest = candidate
            index = candidate.publication_index
            if (
                candidate.status == module.viewport_handoff.FRAME_STATUS_FRAME
                and candidate.completed_samples >= request.max_samples
            ):
                break
        assert newest is not None
        assert newest.completed_samples == request.max_samples
        # Every scheduler tick ran on the render thread.
        assert scheduler.tick_idents
        assert threading.get_ident() not in scheduler.tick_idents
        render_thread_ident = scheduler.tick_idents[0]
        # The handoff registered the absorb timer; firing it prepares the
        # mirror (main-thread scene scan) and chains the apply timer.
        assert timers.run_pending() >= 1
        assert timers.run_pending() >= 1
        obj = module._test_scene.objects[0]
        assert engine._pose_mirror["status"] == "applied"
        assert obj.matrix_world.loc == (4.0, 5.0, 6.0)
        assert engine._physics_playback_lock.diagnostics()["locked_object_paths"] == [
            "/World/Cube"
        ]
        assert module._test_view3d_area.redraws
        # No Blender id-property read ever ran on the render thread.
        read_idents = {ident for _key, ident in module._test_access_log}
        assert render_thread_ident not in read_idents
    finally:
        engine._end_viewport_session(module.ViewportSessionEndReason.ENGINE_DESTROYED)
    # Shutdown ordering: the loop shut the scheduler down on the render
    # thread (before the join returned); the engine's main-thread fallback
    # call is the idempotent second entry.
    assert scheduler.shutdown_idents
    assert scheduler.shutdown_idents[0] == render_thread_ident
    # The stopped loop's tick handoff was dropped with the session.
    assert engine._pending_tick_absorb is None

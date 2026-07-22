# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Thread-aware viewport diagnostics and profile (task02-09).

Schema version 3 removes the redundant refinement milestones while retaining
the version 2 split between main-thread presentation cost and render-thread
work: per-draw records carry a
``thread`` attribution and cross-thread span phases
(``snapshot_to_render_start_ms`` / ``publish_to_redraw_ms`` /
``publish_to_draw_ms``); the render thread appends its own
per-loop-iteration records (correlated by publication index + snapshot
key at artifact-write time on the main thread); the artifact embeds the
thread/loop/signaler/absorb diagnostics and the latest-view (ADR 0013)
evidence fields.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

import ovrtx_blender_example  # noqa: E402,F401
from ovrtx_blender_example import ovrtx_session_controller as controller_module  # noqa: E402
from ovrtx_blender_example import viewport_artifact_recorder, viewport_handoff, viewport_profile  # noqa: E402
from ovrtx_blender_example.ovrtx_runtime_client import RenderClientError, RenderResult  # noqa: E402
from ovrtx_blender_example.ovrtx_value_updates import OvrtxValueUpdateResult  # noqa: E402
from ovrtx_blender_example.runtime_scheduler import RuntimeScheduler  # noqa: E402
from ovrtx_blender_example.viewport_handoff import (  # noqa: E402
    CameraRequestMailbox,
    FrameState,
    LatestFrameSlot,
    ViewSnapshot,
)
from ovrtx_blender_example.viewport_render_thread import (  # noqa: E402
    ITERATION_RECORD_LIMIT,
    LatestViewRenderLoop,
)


WAIT_S = 5.0


def _wait_until(predicate, timeout: float = WAIT_S) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return bool(predicate())


# ---------------------------------------------------------------------------
# viewport_profile: phases, thread attribution, summary
# ---------------------------------------------------------------------------


def _timings(**values: float) -> dict[str, float]:
    timings = {phase: 0.0 for phase in viewport_profile.TIMING_PHASES}
    timings.update(values)
    return timings


def _draw_record(**overrides) -> dict:
    record = {
        "rendered": True,
        "camera_changed": False,
        "snapshot_changed": False,
        "timeline_reset": False,
        "requested_additional_samples": 1,
        "completed_samples": 1,
        "max_samples": 4,
        "timings_ms": _timings(),
    }
    record.update(overrides)
    return record


def test_timing_phases_add_cross_thread_spans_without_renames() -> None:
    # Existing vocabulary preserved (schema consumers key on these names).
    for phase in (
        "request_ms",
        "ensure_session_ms",
        "camera_update_ms",
        "render_ms",
        "result_convert_ms",
        "composition_update_ms",
        "texture_upload_ms",
        "artifact_write_ms",
        "viewport_texture_draw_ms",
        "viewport_callback_ms",
    ):
        assert phase in viewport_profile.TIMING_PHASES
    # New cross-thread spans are additive.
    for phase in (
        "snapshot_to_render_start_ms",
        "publish_to_redraw_ms",
        "publish_to_draw_ms",
    ):
        assert phase in viewport_profile.TIMING_PHASES
        assert phase in viewport_profile.CROSS_THREAD_PHASES
        # Latency spans overlap real work: they must not join the
        # render-interval work decomposition.
        assert phase not in viewport_profile.RENDER_INTERVAL_WORK_PHASES
    # Every phase has a thread attribution; the split matches the spec
    # (request/texture/draw = main; session/value-apply/composition/render/
    # readback = render).
    assert set(viewport_profile.PHASE_THREADS) == set(viewport_profile.TIMING_PHASES)
    assert set(viewport_profile.MAIN_THREAD_PHASES) == {
        "request_ms",
        "texture_upload_ms",
        "artifact_write_ms",
        "viewport_texture_draw_ms",
        "viewport_callback_ms",
    }
    assert set(viewport_profile.RENDER_THREAD_PHASES) == {
        "ensure_session_ms",
        "camera_update_ms",
        "render_ms",
        "result_convert_ms",
        "composition_update_ms",
    }


def test_record_defaults_thread_attribution_to_main() -> None:
    profile = viewport_profile.new()
    defaulted = _draw_record()
    viewport_profile.record(profile, defaulted)
    tagged = _draw_record(thread="render")
    viewport_profile.record(profile, tagged)

    assert profile["recent_draws"][0]["thread"] == "main"
    assert profile["recent_draws"][1]["thread"] == "render"


def test_summary_thread_attribution_separates_main_and_render_cost() -> None:
    profile = viewport_profile.new()
    viewport_profile.record(
        profile,
        _draw_record(
            timings_ms=_timings(
                viewport_callback_ms=2.0,
                render_ms=30.0,
                composition_update_ms=10.0,
                publish_to_draw_ms=5.0,
            )
        ),
    )
    viewport_profile.record(
        profile,
        _draw_record(
            timings_ms=_timings(viewport_callback_ms=4.0, render_ms=50.0)
        ),
    )

    attribution = viewport_profile.summary(profile)["thread_attribution"]
    assert attribution["phase_threads"]["render_ms"] == "render"
    assert attribution["phase_threads"]["viewport_callback_ms"] == "main"
    assert attribution["cross_thread_phases"] == list(
        viewport_profile.CROSS_THREAD_PHASES
    )
    # Main-thread callback cost is the callback span (2, 4) — the RPC-side
    # render/composition cost (40, 50) is attributed to the render thread
    # and never inflates it (spec success criterion 2, answered directly).
    main_stats = attribution["recent_main_thread_callback_stats_ms"]
    assert main_stats["count"] == 2
    assert main_stats["max_ms"] == 4.0
    render_stats = attribution["recent_render_thread_attributed_stats_ms"]
    assert render_stats["max_ms"] == 50.0
    assert render_stats["min_ms"] == 40.0
    # The cross-thread span joins neither side's cost.
    assert main_stats["max_ms"] + render_stats["min_ms"] == 44.0


def test_render_thread_summary_aggregates_and_correlates_records() -> None:
    key_token = viewport_profile.snapshot_key_token(("sensor",), )
    loop_records = [
        {
            "thread": "render",
            "status": "published",
            "publication_index": 1,
            "snapshot_key": key_token,
            "timings_ms": {"render_ms": 10.0, "composition_update_ms": 2.0},
        },
        {
            "thread": "render",
            "status": "published",
            "publication_index": 2,
            "snapshot_key": key_token,
            "timings_ms": {"render_ms": 30.0, "snapshot_to_render_start_ms": 1.5},
        },
        {
            "thread": "render",
            "status": "failed",
            "publication_index": 3,
            "snapshot_key": key_token,
            "timings_ms": {},
            "detail": "RenderClientError: boom",
        },
    ]
    recent_draws = [
        {"publication_index": 1, "snapshot_key": key_token},  # correlated + key match
        {"publication_index": 2, "snapshot_key": "other"},  # correlated, key mismatch
        {"publication_index": 99, "snapshot_key": key_token},  # uncorrelated
    ]

    summary = viewport_profile.render_thread_summary(loop_records, recent_draws)

    assert summary["thread"] == "render"
    assert summary["record_count"] == 3
    assert summary["status_counts"] == {"published": 2, "failed": 1}
    assert summary["published_record_count"] == 3
    assert summary["correlated_draw_count"] == 2
    assert summary["snapshot_key_matched_draw_count"] == 1
    stats = summary["phase_stats"]
    assert stats["render_ms"]["count"] == 2
    assert stats["render_ms"]["mean_ms"] == 20.0
    assert stats["snapshot_to_render_start_ms"]["max_ms"] == 1.5
    assert summary["records"][2]["detail"] == "RenderClientError: boom"


# ---------------------------------------------------------------------------
# Recorder aggregation (post-ledger artifact) with fake two-thread records
# ---------------------------------------------------------------------------


def _artifact_request() -> SimpleNamespace:
    return SimpleNamespace(
        width=2,
        height=2,
        min_samples=1,
        max_samples=4,
        camera_prim_path="/World/Camera",
        timeline_controls_enabled=False,
        timeline_playing=False,
        timeline_frame=1,
        timeline_start=1,
        timeline_end=1,
        simulation_reset_token=0,
        render_product_path="/Render/Product",
    )


def test_recorder_artifact_is_post_ledger_with_thread_model_and_latest_view() -> None:
    recorder = viewport_artifact_recorder.Recorder(
        profile_factory=viewport_profile.new,
        record_profile=viewport_profile.record,
        profile_summary=viewport_profile.summary,
        enabled=lambda: True,
        render_records_summary=viewport_profile.render_thread_summary,
    )
    key_token = viewport_profile.snapshot_key_token(("view", 1))
    recorder.record(
        _draw_record(
            publication_index=1,
            snapshot_key=key_token,
            timings_ms=_timings(viewport_callback_ms=2.0),
        )
    )
    loop_records = [
        {
            "thread": "render",
            "status": "published",
            "publication_index": 1,
            "snapshot_key": key_token,
            "timings_ms": {"render_ms": 12.0},
        }
    ]
    artifact = recorder.artifact(
        viewport_artifact_recorder.State(
            simulation_id="sim-1",
            request=_artifact_request(),
            result=None,
            snapshot_index=1,
            render_count=1,
            draw_count=1,
            snapshot_count=1,
            camera_update_count=0,
            camera_controls_mode="blender_view",
            render_thread={"name": "ovrtx-render-x", "status": "running"},
            render_loop={
                "running": True,
                "snapshots_superseded": 2,
                "final_view_refined": True,
                "mailbox": {"superseded_snapshots": 3},
            },
            redraw_signaling={"signals": 5, "timer_fires": 4},
            tick_absorb={"handoffs": 7, "absorbs_applied": 6},
            render_loop_records=loop_records,
        )
    )

    assert artifact["schema_version"] == 3
    assert "milestones" not in artifact
    assert artifact["thread_model"]["render_thread"]["name"] == "ovrtx-render-x"
    assert artifact["thread_model"]["render_loop"]["running"] is True
    assert artifact["thread_model"]["redraw_signaling"]["signals"] == 5
    assert artifact["thread_model"]["tick_absorb"]["handoffs"] == 7
    # Latest-view evidence (ADR 0013): mailbox-dropped views + abandoned
    # refinements, refinement abandonment alone, and the final-view state.
    assert artifact["latest_view"] == {
        "superseded_snapshot_count": 5,
        "abandoned_refinement_count": 2,
        "final_view_refined": True,
    }
    # Two record streams, one artifact: the loop record correlates to the
    # per-draw record via publication index + snapshot key.
    render_profile = artifact["render_thread_profile"]
    assert render_profile["record_count"] == 1
    assert render_profile["correlated_draw_count"] == 1
    assert render_profile["snapshot_key_matched_draw_count"] == 1
    assert render_profile["phase_stats"]["render_ms"]["max_ms"] == 12.0
    # The per-draw stream carries its own thread attribution.
    assert artifact["profile"]["recent_draws"][0]["thread"] == "main"
    assert artifact["profile"]["thread_attribution"]["main_thread_phases"]
    # Everything added in v2 serializes.
    json.dumps(artifact)


def test_recorder_without_aggregator_still_embeds_render_records() -> None:
    recorder = viewport_artifact_recorder.Recorder(
        profile_factory=lambda: {},
        record_profile=lambda _profile, _record: None,
        profile_summary=lambda _profile, _latency_ms: {"enabled": True},
        enabled=lambda: True,
    )
    artifact = recorder.artifact(
        viewport_artifact_recorder.State(
            simulation_id=None,
            request=_artifact_request(),
            result=None,
            snapshot_index=0,
            render_count=0,
            draw_count=0,
            snapshot_count=0,
            camera_update_count=0,
            camera_controls_mode="usd_camera",
            render_loop_records=[{"status": "published", "publication_index": 1}],
        )
    )

    assert artifact["schema_version"] == 3
    assert artifact["render_thread_profile"]["record_count"] == 1
    assert artifact["latest_view"]["final_view_refined"] is False


# ---------------------------------------------------------------------------
# Mailbox supersession evidence
# ---------------------------------------------------------------------------


def _snapshot(tx: float = 2.0, **overrides) -> ViewSnapshot:
    matrix = (
        (1.0, 0.0, 0.0, float(tx)),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    fields = {
        "camera_matrix": matrix,
        "camera_prim_path": "/World/Camera",
        "min_samples": 1,
        "max_samples": 4,
        "selected_sensor_paths": ("/Render/Product",),
        "width": 1,
        "height": 1,
    }
    fields.update(overrides)
    return ViewSnapshot(**fields)


def test_mailbox_counts_only_distinct_view_supersessions() -> None:
    mailbox = CameraRequestMailbox()
    mailbox.write(_snapshot(tx=1.0))
    # Same-key rewrite: a routine draw-path refresh, not a supersession.
    mailbox.write(_snapshot(tx=1.0))
    assert mailbox.diagnostics()["superseded_snapshots"] == 0
    assert mailbox.diagnostics()["overwrites"] == 1
    # A distinct pending view replaced before adoption is superseded.
    mailbox.write(_snapshot(tx=9.0))
    assert mailbox.diagnostics()["superseded_snapshots"] == 1
    # Writing into an empty slot supersedes nothing.
    assert mailbox.take(0) is not None
    mailbox.write(_snapshot(tx=3.0))
    assert mailbox.diagnostics()["superseded_snapshots"] == 1


# ---------------------------------------------------------------------------
# Render loop: per-iteration records on the render thread
# ---------------------------------------------------------------------------


class _Client:
    """Fake srtx client compatible with OvrtxSessionController."""

    def __init__(self) -> None:
        self.fail_render = False
        self.render_calls = 0
        self.startup_diagnostics = {"render_worker": {"status": "ready"}}
        self.last_render_timings: dict = {}
        self.last_value_update_timings: dict = {}

    def start_session(self, spec: object, simulation_id: str | None = None) -> str:
        return simulation_id or "sim"

    def render_result(self, simulation_id: str, **kwargs: object) -> RenderResult:
        self.render_calls += 1
        if self.fail_render:
            raise RenderClientError("render failed")
        return RenderResult(
            width=1,
            height=1,
            rgba8=b"\x00\x00\x00\xff",
            completed_samples=int(kwargs["additional_samples"]),
            session_completed_samples=self.render_calls,
            simulation_time_ns=0,
        )

    def update_transforms(self, simulation_id: str, values) -> OvrtxValueUpdateResult:
        return OvrtxValueUpdateResult(len(tuple(values)), pending_simulation_time_ns=1)

    def delete_simulation(self, simulation_id: str) -> str:
        return "stopped"

    def shutdown(self) -> None:
        return None


class _LoopHarness:
    """Real controller + real scheduler (physics disabled) + fake client."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        self.client = _Client()
        monkeypatch.setattr(
            controller_module,
            "_runtime_client_from_request",
            lambda request: self.client,
        )
        self.controller = controller_module.OvrtxSessionController()
        self.base_request = SimpleNamespace  # placeholder, replaced below
        from ovrtx_blender_example.render_requests import RenderRequest

        self.base_request = RenderRequest(
            input_usd_path=str(tmp_path / "scene.usda"),
            sensor_paths=("/Render/Product",),
            selected_sensor_paths=("/Render/Product",),
            width=1,
            height=1,
            min_samples=1,
            max_samples=4,
            camera_prim_path="/World/Camera",
            camera_matrix=_snapshot().camera_matrix,
            worker_command="worker",
            native_client_module="client",
        )
        self.controller.ensure(self.base_request)
        self.scheduler = RuntimeScheduler(
            config_factory=lambda path: SimpleNamespace(enabled=False)
        )
        self.mailbox = CameraRequestMailbox()
        self.slot = LatestFrameSlot()
        self.loop = LatestViewRenderLoop(
            mailbox=self.mailbox,
            frame_slot=self.slot,
            controller=self.controller,
            scheduler=self.scheduler,
            request_for_snapshot=lambda snapshot: viewport_handoff.request_from_snapshot(
                self.base_request, snapshot
            ),
        )


@contextmanager
def _running(loop: LatestViewRenderLoop):
    thread = threading.Thread(target=loop.run, name="latest-view-test", daemon=True)
    thread.start()
    try:
        yield thread
    finally:
        loop.request_stop()
        thread.join(WAIT_S)
        assert not thread.is_alive()


def test_loop_appends_iteration_records_with_render_thread_attribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _LoopHarness(monkeypatch, tmp_path)
    snapshot = _snapshot(tx=2.0)

    with _running(harness.loop):
        harness.mailbox.write(snapshot)
        assert _wait_until(
            lambda: (harness.slot.peek_latest() or SimpleNamespace(completed_samples=0)).completed_samples
            >= 4
        )

    records = harness.loop.iteration_records()
    assert records, "expected per-iteration records"
    assert all(record["thread"] == "render" for record in records)
    published = [record for record in records if record["status"] == "published"]
    assert published
    # Correlation identity: publication indexes are the slot's stamps and
    # the snapshot key token matches the adopted snapshot.
    indexes = [record["publication_index"] for record in published]
    assert indexes == sorted(indexes)
    assert indexes[-1] == harness.slot.latest_index()
    expected_token = viewport_profile.snapshot_key_token(snapshot.key)
    assert all(record["snapshot_key"] == expected_token for record in published)
    # Render-thread phases and the cross-thread snapshot→render-start span
    # are measured per iteration.
    for record in published:
        assert record["timings_ms"]["render_ms"] >= 0.0
        assert record["timings_ms"]["composition_update_ms"] >= 0.0
        assert record["timings_ms"]["snapshot_to_render_start_ms"] >= 0.0
        assert record["span_boundaries"]["snapshot_written_monotonic_ns"] > 0
    diagnostics = harness.loop.diagnostics()
    assert diagnostics["iteration_record_count"] == len(records)
    assert diagnostics["final_view_refined"] is True
    assert diagnostics["mailbox"]["superseded_snapshots"] == 0
    assert len(records) <= ITERATION_RECORD_LIMIT


def test_long_session_retains_only_recent_iteration_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _LoopHarness(monkeypatch, tmp_path)
    snapshot = _snapshot(max_samples=ITERATION_RECORD_LIMIT + 5)

    with _running(harness.loop):
        harness.mailbox.write(snapshot)
        assert _wait_until(
            lambda: (
                harness.slot.peek_latest() or SimpleNamespace(completed_samples=0)
            ).completed_samples
            >= snapshot.max_samples
        )

    records = harness.loop.iteration_records()
    diagnostics = harness.loop.diagnostics()
    assert diagnostics["iteration_record_count"] > len(records)
    assert len(records) == ITERATION_RECORD_LIMIT
    assert records[-1]["status"] == "published"
    assert records[-1]["completed_samples"] == snapshot.max_samples
    assert "refinement" not in harness.controller.diagnostics()


def test_loop_records_failed_iterations_and_unrefined_final_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _LoopHarness(monkeypatch, tmp_path)
    harness.client.fail_render = True

    with _running(harness.loop):
        harness.mailbox.write(_snapshot(tx=2.0))
        assert _wait_until(
            lambda: any(
                record["status"] == "failed"
                for record in harness.loop.iteration_records()
            )
        )

    failed = [
        record
        for record in harness.loop.iteration_records()
        if record["status"] == "failed"
    ]
    assert failed
    assert failed[-1]["publication_index"] == harness.slot.latest_index()
    assert "render failed" in failed[-1]["detail"]
    assert harness.loop.diagnostics()["final_view_refined"] is False


# ---------------------------------------------------------------------------
# Engine: cross-thread spans per draw, artifact embedding, absorb counters
# ---------------------------------------------------------------------------


class _FakeRenderEngine:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.stats: list[tuple[str, str]] = []
        self.redraw_requested = False
        self.reports: list[tuple[set[str], str]] = []

    def update_stats(self, engine: str, message: str) -> None:
        self.stats.append((engine, message))

    def tag_redraw(self) -> None:
        self.redraw_requested = True

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
        with self._lock:
            callbacks = list(self.pending)
            self.pending.clear()
        for fn in callbacks:
            assert fn() is None
        return len(callbacks)


def _load_engine_with_fake_bpy(monkeypatch: pytest.MonkeyPatch):
    module_name = "ovrtx_blender_example._engine_thread_aware_diagnostics_test"
    module_path = ROOT / "addon" / "ovrtx_blender_example" / "engine.py"
    fake_bpy = SimpleNamespace(
        types=SimpleNamespace(RenderEngine=_FakeRenderEngine),
        app=SimpleNamespace(timers=_FakeTimers()),
        context=SimpleNamespace(window_manager=SimpleNamespace(windows=())),
    )
    monkeypatch.setitem(sys.modules, "bpy", fake_bpy)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _viewport_context() -> SimpleNamespace:
    return SimpleNamespace(
        region_data=SimpleNamespace(view_perspective="PERSP"),
        scene=SimpleNamespace(),
        space_data=None,
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
            (1.0, 0.0, 0.0, 1.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        "worker_command": "worker",
        "native_client_module": "client",
    }
    fields.update(overrides)
    return module.RenderRequest(**fields)


def _result(completed_samples: int = 4) -> RenderResult:
    return RenderResult(
        width=2,
        height=2,
        rgba8=b"\x00" * 16,
        completed_samples=completed_samples,
        session_completed_samples=completed_samples,
        simulation_time_ns=0,
    )


class _StubController:
    def diagnostics(self):
        return {
            "simulation_id": "sim",
            "session_reuse": {},
            "lifecycle_events": (),
            "startup": {"render_worker": {"status": "running"}},
            "render_timings": {},
            "value_update_timings": {},
            "active": True,
        }

    def shutdown(self):
        return None


def _engine_with_stubbed_session(module, monkeypatch: pytest.MonkeyPatch, request):
    engine = module.OvrtxExampleRenderEngine()

    class _Adapter:
        def view_update(self, context: object, depsgraph: object) -> object:
            return request

        def view_draw(self, context: object, depsgraph: object) -> object:
            return request

    monkeypatch.setattr(module, "_render_callback_adapter", lambda engine_id="": _Adapter())
    engine._begin_async_viewport_session = lambda *_args, **_kwargs: None
    engine._ovrtx_session_controller = _StubController()
    engine._render_thread = SimpleNamespace(
        status=lambda: module.viewport_render_thread.STATUS_RUNNING,
        diagnostics=lambda: {"name": "ovrtx-render-stub", "status": "running"},
    )
    engine._write_viewport_artifact = lambda *_args, **_kwargs: None
    engine._upload_viewport_texture = lambda _result: "texture"
    engine._draw_viewport_texture = lambda *_args: None
    return engine


def test_view_draw_records_cross_thread_spans_and_correlation_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_engine_with_fake_bpy(monkeypatch)
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_VIEWPORT_PROFILE", "1")
    request = _request(module)
    engine = _engine_with_stubbed_session(module, monkeypatch, request)

    snapshot = module.viewport_handoff.snapshot_from_render_request(request)
    # Thread-side marks around the publication, same perf_counter clock.
    snapshot_written_ns = time.perf_counter_ns()
    render_started_ns = time.perf_counter_ns()
    render_completed_ns = time.perf_counter_ns()
    engine._frame_slot.publish(
        module.viewport_handoff.FrameState(
            render_result=_result(),
            snapshot_key=snapshot.key,
            completed_samples=4,
            timing_marks={
                "snapshot_written_monotonic_ns": snapshot_written_ns,
                "render_call_started_monotonic_ns": render_started_ns,
                "render_call_completed_monotonic_ns": render_completed_ns,
            },
        )
    )

    engine.view_draw(_viewport_context(), SimpleNamespace(scene=SimpleNamespace()))

    record = engine._viewport_artifact_recorder._profile["recent_draws"][-1]
    assert record["thread"] == "main"
    assert record["publication_index"] == 1
    assert record["snapshot_key"] == module.viewport_profile.snapshot_key_token(
        snapshot.key
    )
    timings = record["timings_ms"]
    # Cross-thread spans: mailbox write → render start, and publication →
    # presenting redraw/draw (publication precedes the draw callback).
    assert timings["snapshot_to_render_start_ms"] >= 0.0
    assert timings["publish_to_redraw_ms"] > 0.0
    assert timings["publish_to_draw_ms"] >= timings["publish_to_redraw_ms"]
    boundaries = record["span_boundaries"]
    assert boundaries["frame_published_monotonic_ns"] > 0
    assert boundaries["snapshot_written_monotonic_ns"] == snapshot_written_ns

    # Re-presenting the same publication measures no publication latency.
    engine.view_draw(_viewport_context(), SimpleNamespace(scene=SimpleNamespace()))
    reused = engine._viewport_artifact_recorder._profile["recent_draws"][-1]
    assert reused["publication_index"] == 1
    assert reused["timings_ms"]["publish_to_draw_ms"] == 0.0


def test_artifact_embeds_live_thread_model_and_cleanup_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_engine_with_fake_bpy(monkeypatch)
    request = _request(module)
    engine = _engine_with_stubbed_session(module, monkeypatch, request)
    engine._viewport_request = request
    engine._render_loop = SimpleNamespace(
        request_stop=lambda: None,
        diagnostics=lambda: {
            "running": True,
            "snapshots_superseded": 1,
            "final_view_refined": False,
            "mailbox": {"superseded_snapshots": 2},
        },
        iteration_records=lambda: [
            {
                "thread": "render",
                "status": "published",
                "publication_index": 4,
                "snapshot_key": "key",
                "timings_ms": {"render_ms": 3.0},
            }
        ],
    )
    engine._redraw_signaler.start()
    engine._redraw_signaler.signal()  # counts one signal (registration ok)
    engine._viewport_cleanup_diagnostics = {
        "status": "teardown_deadline_exceeded",
        "thread_status": "failed",
        "thread_name": "ovrtx-render-x",
        "joined": False,
        "leaked_thread": True,
        "join_timeout_seconds": 0.1,
        "failure": "leaked_thread: ...",
        "teardown_errors": ["detach_viewport: unconfirmed"],
    }

    artifact = engine._viewport_artifact(running=True)

    thread_model = artifact["thread_model"]
    assert thread_model["render_thread"]["name"] == "ovrtx-render-stub"
    assert thread_model["render_loop"]["running"] is True
    assert thread_model["redraw_signaling"]["signals"] == 1
    assert thread_model["redraw_signaling"]["timer_registrations"] == 1
    assert thread_model["tick_absorb"]["handoffs"] == 0
    assert artifact["latest_view"] == {
        "superseded_snapshot_count": 3,
        "abandoned_refinement_count": 1,
        "final_view_refined": False,
    }
    assert artifact["render_thread_profile"]["record_count"] == 1
    # The task02-08 teardown outcome keys surface through
    # session_lifecycle.cleanup unchanged.
    cleanup = artifact["session_lifecycle"]["cleanup"]
    assert cleanup["status"] == "teardown_deadline_exceeded"
    for key in (
        "thread_status",
        "thread_name",
        "joined",
        "leaked_thread",
        "join_timeout_seconds",
        "failure",
        "teardown_errors",
    ):
        assert key in cleanup
    json.dumps(artifact)


def test_tick_absorb_counters_witness_the_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_engine_with_fake_bpy(monkeypatch)
    timers = module.bpy.app.timers
    engine = module.OvrtxExampleRenderEngine()
    request = _request(module)
    update = {"update_result": {"value_apply_ms": 1.0}}
    failed_result = module.RuntimeTickResult(
        status=module.RuntimeTickStatus.FAILED,
        enabled=True,
        update=update,
    )

    # Idle ticks skip the handoff entirely.
    idle = module.RuntimeTickResult(
        status=module.RuntimeTickStatus.STEPPED, enabled=True
    )
    engine._handoff_runtime_tick_result(idle, request)
    assert engine._tick_absorb_diagnostics()["idle_skipped"] == 1
    assert timers.register_calls == 0

    # First handoff registers the absorb timer; a second before the fire
    # coalesces into the pending one.
    engine._handoff_runtime_tick_result(failed_result, request)
    engine._handoff_runtime_tick_result(failed_result, request)
    diagnostics = engine._tick_absorb_diagnostics()
    assert diagnostics["handoffs"] == 2
    assert diagnostics["coalesced"] == 1
    assert diagnostics["timer_registrations"] == 1
    assert diagnostics["timer_pending"] is True
    assert timers.register_calls == 1

    assert timers.run_pending() == 1
    diagnostics = engine._tick_absorb_diagnostics()
    assert diagnostics["absorbs_applied"] == 1
    assert diagnostics["timer_pending"] is False
    assert diagnostics["pending_handoff"] is False

    # A stale-loop handoff (leaked thread resuming after teardown) drops.
    engine._render_loop = SimpleNamespace(request_stop=lambda: None)
    engine._handoff_runtime_tick_result(
        failed_result, request, source_loop=object()
    )
    assert engine._tick_absorb_diagnostics()["stale_loop_dropped"] == 1
    assert engine._tick_absorb_diagnostics()["handoffs"] == 2
    engine._render_loop = None


def test_session_end_artifact_preserves_final_thread_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end: real thread + loop; the session-end artifact carries the
    stopped thread's identity, the loop's records, and latest-view evidence."""

    module = _load_engine_with_fake_bpy(monkeypatch)
    client = _Client()
    monkeypatch.setattr(
        controller_module,
        "_runtime_client_from_request",
        lambda _request: client,
    )
    artifact_path = tmp_path / "viewport-preview.json"
    profile_path = tmp_path / "viewport-profile.json"
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_VIEWPORT_ARTIFACT", str(artifact_path))
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_VIEWPORT_PROFILE", str(profile_path))
    engine = module.OvrtxExampleRenderEngine()
    engine._runtime_scheduler = RuntimeScheduler(
        config_factory=lambda _path: SimpleNamespace(enabled=False)
    )
    request = _request(module)
    try:
        engine._begin_async_viewport_session(request)
        engine._camera_mailbox.write(
            module.viewport_handoff.snapshot_from_render_request(request)
        )
        deadline = time.monotonic() + WAIT_S
        newest = engine._frame_slot.wait_for_newer(0, timeout=WAIT_S)
        assert newest is not None
        while time.monotonic() < deadline and newest.completed_samples < request.max_samples:
            candidate = engine._frame_slot.wait_for_newer(
                newest.publication_index, timeout=WAIT_S
            )
            if candidate is not None:
                newest = candidate
        assert newest.completed_samples == request.max_samples
    finally:
        engine._end_viewport_session(module.ViewportSessionEndReason.ENGINE_DESTROYED)

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["schema_version"] == 3
    thread_model = artifact["thread_model"]
    assert thread_model["render_thread"]["name"].startswith("ovrtx-render-")
    assert thread_model["render_thread"]["status"] == "stopped"
    assert thread_model["render_loop"]["running"] is False
    assert thread_model["render_loop"]["publications"] >= 1
    render_profile = artifact["render_thread_profile"]
    assert render_profile["record_count"] >= 1
    statuses = {record["status"] for record in render_profile["records"]}
    assert "published" in statuses
    assert all(record["thread"] == "render" for record in render_profile["records"])
    assert artifact["latest_view"]["final_view_refined"] is True
    assert artifact["latest_view"]["abandoned_refinement_count"] == 0
    # The profile wrapper carries the same schema version.
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    assert profile["schema_version"] == 3
    assert profile["viewport_artifact"]["schema_version"] == 3

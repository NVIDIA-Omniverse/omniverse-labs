# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import scene_generation_sessions  # noqa: E402
from ovrtx_blender_example import interactive_operator_state  # noqa: E402
from ovrtx_blender_example import engine  # noqa: E402
from ovrtx_blender_example.blender_callback_adapters import BlenderEditCallbackAdapter  # noqa: E402
from ovrtx_blender_example.render_requests import RenderRequest  # noqa: E402
from ovrtx_blender_example.runtime_scheduler import (  # noqa: E402
    RuntimeScheduler,
    RuntimeTickResult,
    RuntimeTickStatus,
)
from ovrtx_blender_example.ovphysx_stage import (  # noqa: E402
    OvphysxStageResult,
    OvphysxStageStatus,
)
from ovrtx_blender_example.scene_generation import (  # noqa: E402
    BlenderId,
    BlenderPrimPath,
    SceneGenerationOwner,
)
from ovrtx_blender_example.ovrtx_value_updates import (  # noqa: E402
    OvrtxAttributeValue,
    OvrtxTransformValue,
    OvrtxValueUpdateResult,
)
from ovrtx_blender_example.world_dome_conversion import DEFAULT_DOME_OWNER_PATH  # noqa: E402
from ovrtx_blender_example.interactive_edit_planner import (  # noqa: E402
    DataAuthority,
    EditShape,
    EditStatus,
    InteractiveEdit,
    InteractiveEditPlanner,
    edit_location,
)
from ovrtx_blender_example.interactive_edit_workflow import (  # noqa: E402
    EditWorkflowResult,
    WorkflowAction,
)


class _Timers:
    def __init__(self) -> None:
        self.callbacks = []

    def register(self, callback, *, first_interval: float = 0.0) -> None:
        assert first_interval == 0.0
        self.callbacks.append(callback)


def _blender(scene: object) -> SimpleNamespace:
    return SimpleNamespace(
        context=SimpleNamespace(scene=scene),
        app=SimpleNamespace(timers=_Timers()),
    )


def test_mark_scene_dirty_requires_affected_blender_ids(monkeypatch) -> None:
    monkeypatch.setattr(scene_generation_sessions, "_dirty", {})
    monkeypatch.setattr(scene_generation_sessions, "_rejected_unscoped_dirty_requests", {})
    monkeypatch.setattr(scene_generation_sessions, "_accepted_world_dirty_requests", {})

    assert not scene_generation_sessions.mark_scene_dirty(
        SimpleNamespace(session_uid=0)
    )
    scene = SimpleNamespace(session_uid=77)
    assert not scene_generation_sessions.mark_scene_dirty(scene)
    generation = SimpleNamespace(number=0)
    monkeypatch.setattr(
        scene_generation_sessions,
        "_owners",
        {
            77: SimpleNamespace(
                current_generation=generation,
                pending_generation=None,
                reuse=lambda: generation,
            )
        },
    )
    assert scene_generation_sessions.generation_for_scene(scene) is generation
    assert scene_generation_sessions.mark_scene_dirty(
        scene,
        {BlenderId("WORLD", 30)},
    )
    assert scene_generation_sessions.diagnostics()["dirty_scene_uids"] == [77]
    assert scene_generation_sessions.diagnostics()[
        "rejected_unscoped_dirty_requests"
    ] == {"77": 1}
    assert scene_generation_sessions.diagnostics()[
        "accepted_world_dirty_requests"
    ] == {"77": 1}


def test_ignored_scene_updates_are_diagnostic(monkeypatch) -> None:
    monkeypatch.setattr(scene_generation_sessions, "_ignored_scene_updates", {})
    scene_generation_sessions.record_ignored_scene_update(
        SimpleNamespace(session_uid=77)
    )
    assert scene_generation_sessions.diagnostics()["ignored_scene_updates"] == {
        "77": 1
    }


def test_viewport_generation_defers_dirty_reconciliation_to_one_timer(monkeypatch) -> None:
    updated = []
    scene = SimpleNamespace(session_uid=77, update_tag=lambda: updated.append(True))
    generation = SimpleNamespace(number=0)
    candidate = SimpleNamespace(number=1)
    owner = SimpleNamespace(current_generation=generation, pending_generation=None)
    owner.reuse = lambda: owner.pending_generation or owner.current_generation
    callbacks = []
    reconciled = []
    monkeypatch.setattr(scene_generation_sessions, "_owners", {77: owner})
    monkeypatch.setattr(scene_generation_sessions, "_dirty", {})
    monkeypatch.setattr(scene_generation_sessions, "_blocked_reconciliations", {})
    monkeypatch.setattr(scene_generation_sessions, "_reconciliation_timers", set())
    monkeypatch.setattr(
        scene_generation_sessions,
        "bpy",
        SimpleNamespace(
            app=SimpleNamespace(
                timers=SimpleNamespace(
                    register=lambda callback, **_kwargs: callbacks.append(callback)
                )
            )
        ),
    )
    def reconcile(received: object) -> object:
        if owner.pending_generation is not None:
            return owner.pending_generation
        reconciled.append(received)
        scene_generation_sessions._dirty.pop(77)
        owner.pending_generation = candidate
        return candidate

    monkeypatch.setattr(scene_generation_sessions, "generation_for_scene", reconcile)

    assert scene_generation_sessions.mark_scene_dirty(
        scene,
        {BlenderId("WORLD", 30)},
        defer_world_reconciliation=True,
    )
    assert scene_generation_sessions.generation_for_viewport(scene) is generation
    assert scene_generation_sessions.generation_for_viewport(scene) is generation
    assert len(callbacks) == 1
    assert reconciled == []

    assert callbacks[0]() is None
    assert reconciled == [scene]
    assert updated == [True]
    assert scene_generation_sessions.generation_for_viewport(scene) is candidate


def test_blocked_world_reconciliation_waits_for_assignment_change(monkeypatch) -> None:
    old_id = BlenderId("WORLD", 30)
    scene = SimpleNamespace(session_uid=77, world=None, objects=())
    generation = SimpleNamespace(
        number=0,
        world_session_uid=30,
        blender_prim_paths={},
    )
    owner = SimpleNamespace(current_generation=generation, reuse=lambda: generation)
    callbacks = []
    monkeypatch.setattr(scene_generation_sessions, "_owners", {77: owner})
    monkeypatch.setattr(scene_generation_sessions, "_dirty", {77: {old_id}})
    monkeypatch.setattr(
        scene_generation_sessions,
        "_blocked_reconciliations",
        {77: ("failed", {old_id})},
    )
    monkeypatch.setattr(scene_generation_sessions, "_reconciliation_timers", set())
    monkeypatch.setattr(
        scene_generation_sessions,
        "bpy",
        SimpleNamespace(
            app=SimpleNamespace(
                timers=SimpleNamespace(
                    register=lambda callback, **_kwargs: callbacks.append(callback)
                )
            )
        ),
    )

    assert scene_generation_sessions.mark_scene_dirty(
        scene, {old_id}, defer_world_reconciliation=True
    )
    assert scene_generation_sessions.generation_for_viewport(scene) is generation
    assert callbacks == []

    scene.world = SimpleNamespace(
        session_uid=31,
        library=None,
        override_library=None,
    )
    assert scene_generation_sessions.mark_scene_dirty(
        scene,
        {old_id, BlenderId("WORLD", 31)},
        defer_world_reconciliation=True,
    )
    assert scene_generation_sessions.generation_for_viewport(scene) is generation
    assert len(callbacks) == 1


def test_blocked_world_reconciliation_retries_for_new_topology_scope(monkeypatch) -> None:
    old_id = BlenderId("WORLD", 30)
    object_id = BlenderId("OBJECT", 99)
    scene = SimpleNamespace(session_uid=77, world=None, objects=())
    generation = SimpleNamespace(
        number=0,
        world_session_uid=30,
        blender_prim_paths={},
    )
    owner = SimpleNamespace(current_generation=generation, reuse=lambda: generation)
    callbacks = []
    monkeypatch.setattr(scene_generation_sessions, "_owners", {77: owner})
    monkeypatch.setattr(scene_generation_sessions, "_dirty", {77: {old_id}})
    monkeypatch.setattr(
        scene_generation_sessions,
        "_blocked_reconciliations",
        {77: ("failed", {old_id})},
    )
    monkeypatch.setattr(scene_generation_sessions, "_reconciliation_timers", set())
    monkeypatch.setattr(
        scene_generation_sessions,
        "bpy",
        SimpleNamespace(
            app=SimpleNamespace(
                timers=SimpleNamespace(
                    register=lambda callback, **_kwargs: callbacks.append(callback)
                )
            )
        ),
    )

    assert scene_generation_sessions.mark_scene_dirty(
        scene, {object_id}, defer_world_reconciliation=True
    )
    assert len(callbacks) == 1


def test_viewport_schedules_world_dirty_admitted_without_live_viewport(monkeypatch) -> None:
    world_id = BlenderId("WORLD", 30)
    scene = SimpleNamespace(session_uid=77)
    generation = SimpleNamespace(number=0)
    owner = SimpleNamespace(current_generation=generation, reuse=lambda: generation)
    scheduled = []
    monkeypatch.setattr(scene_generation_sessions, "_owners", {77: owner})
    monkeypatch.setattr(scene_generation_sessions, "_dirty", {77: {world_id}})
    monkeypatch.setattr(
        scene_generation_sessions,
        "_schedule_world_reconciliation",
        lambda received_scene, uid, affected: scheduled.append(
            (received_scene, uid, affected)
        ),
    )

    assert scene_generation_sessions.generation_for_viewport(scene) is generation
    assert scheduled == [(scene, 77, {world_id})]


def test_viewport_reschedules_world_dirty_after_prior_candidate_clears(monkeypatch) -> None:
    world_id = BlenderId("WORLD", 30)
    scene = SimpleNamespace(session_uid=77)
    current = SimpleNamespace(number=0)
    pending = SimpleNamespace(number=1)
    owner = SimpleNamespace(current_generation=current, pending_generation=pending)
    owner.reuse = lambda: owner.pending_generation or owner.current_generation
    callbacks = []
    monkeypatch.setattr(scene_generation_sessions, "_owners", {77: owner})
    monkeypatch.setattr(scene_generation_sessions, "_dirty", {77: {world_id}})
    monkeypatch.setattr(scene_generation_sessions, "_blocked_reconciliations", {})
    monkeypatch.setattr(scene_generation_sessions, "_reconciliation_timers", set())
    monkeypatch.setattr(
        scene_generation_sessions,
        "bpy",
        SimpleNamespace(
            app=SimpleNamespace(
                timers=SimpleNamespace(
                    register=lambda callback, **_kwargs: callbacks.append(callback)
                )
            )
        ),
    )
    monkeypatch.setattr(
        scene_generation_sessions,
        "generation_for_scene",
        lambda _scene: owner.reuse(),
    )

    assert scene_generation_sessions.generation_for_viewport(scene) is pending
    assert callbacks.pop(0)() is None
    owner.pending_generation = None
    assert scene_generation_sessions.generation_for_viewport(scene) is current
    assert len(callbacks) == 1


def test_world_assignment_identity_changes_ignore_scene_callback_noise(monkeypatch) -> None:
    old_world = SimpleNamespace(
        session_uid=30,
        library=None,
        override_library=None,
    )
    new_world = SimpleNamespace(
        session_uid=31,
        library=None,
        override_library=None,
    )
    old_id = BlenderId("WORLD", 30)
    new_id = BlenderId("WORLD", 31)
    old_generation = SimpleNamespace(
        blender_prim_paths={},
        world_session_uid=30,
    )
    owner = SimpleNamespace(
        current_generation=old_generation,
        pending_generation=None,
    )
    scene = SimpleNamespace(
        session_uid=12,
        world=old_world,
        objects=(),
    )
    monkeypatch.setattr(scene_generation_sessions, "_owners", {12: owner})

    assert scene_generation_sessions.topology_identity_changes(scene) == set()

    scene.world = None
    assert scene_generation_sessions.topology_identity_changes(scene) == {old_id}

    scene.world = new_world
    assert scene_generation_sessions.topology_identity_changes(scene) == {
        old_id,
        new_id,
    }

    owner.pending_generation = SimpleNamespace(
        blender_prim_paths={},
        world_session_uid=31,
    )
    assert scene_generation_sessions.topology_identity_changes(scene) == set()


def test_world_assignment_tracks_linked_world_identity(monkeypatch) -> None:
    linked_world = SimpleNamespace(
        session_uid=31,
        library=object(),
        override_library=None,
    )
    scene = SimpleNamespace(session_uid=77, world=linked_world, objects=())
    generation = SimpleNamespace(
        world_session_uid=0,
        blender_prim_paths={},
    )
    owner = SimpleNamespace(current_generation=generation, pending_generation=None)
    monkeypatch.setattr(scene_generation_sessions, "_owners", {77: owner})

    assert scene_generation_sessions.topology_identity_changes(scene) == {
        BlenderId("WORLD", 31)
    }


def test_topology_identity_changes_compare_only_export_reachable_identities(
    monkeypatch,
) -> None:
    used_material = SimpleNamespace(
        session_uid=30,
        library=None,
        override_library=None,
    )
    unused_material = SimpleNamespace(
        session_uid=31,
        library=None,
        override_library=None,
    )
    class Mesh:
        materials = (used_material, unused_material)

        @property
        def polygons(self):
            raise AssertionError("identity comparison must not scale with polygon count")

    mesh = Mesh()
    obj = SimpleNamespace(
        session_uid=20,
        type="MESH",
        data=mesh,
        library=None,
        override_library=None,
    )
    generation = SimpleNamespace(
        blender_prim_paths={
            BlenderId("OBJECT", 20): object(),
            BlenderId("MATERIAL", 30): object(),
        },
        world_session_uid=0,
    )
    owner = SimpleNamespace(current_generation=generation, pending_generation=None)
    scene = SimpleNamespace(session_uid=12, world=None, objects=(obj,))
    monkeypatch.setattr(scene_generation_sessions, "_owners", {12: owner})

    assert scene_generation_sessions.topology_identity_changes(scene) == set()

    mesh.materials = (unused_material,)
    assert scene_generation_sessions.topology_identity_changes(scene) == {
        BlenderId("MATERIAL", 30)
    }


def test_initial_generation_is_scheduled_without_render_or_camera(monkeypatch) -> None:
    scene = SimpleNamespace(session_uid=31, camera=None)
    blender = _blender(scene)
    generation = SimpleNamespace(number=0, digest="eager")
    calls = []
    cursors = []
    blender.context.window = SimpleNamespace(cursor_set=cursors.append)

    def generate(received: object) -> object:
        calls.append(received)
        assert cursors == ["WAIT"]
        return generation

    monkeypatch.setattr(
        scene_generation_sessions,
        "generation_for_scene",
        generate,
    )
    monkeypatch.setattr(scene_generation_sessions, "_initial_generation_revision", 0)
    monkeypatch.setattr(
        scene_generation_sessions,
        "_initial_generation_diagnostics",
        {"status": "unavailable", "scene_uid": 0, "error": "", "last_error": ""},
    )

    assert scene_generation_sessions.schedule_initial_generation(blender)
    assert scene_generation_sessions.diagnostics()["initial_generation"]["status"] == "scheduled"

    blender.app.timers.callbacks.pop()()

    assert calls == [scene]
    assert cursors == ["WAIT", "DEFAULT"]
    assert scene_generation_sessions.diagnostics()["initial_generation"] == {
        "status": "ready",
        "scene_uid": 31,
        "error": "",
        "last_error": "",
        "number": 0,
        "digest": "eager",
    }


def test_initial_generation_deduplicates_and_invalidates_stale_file_work(monkeypatch) -> None:
    first = SimpleNamespace(session_uid=41)
    second = SimpleNamespace(session_uid=42)
    blender = _blender(first)
    calls = []
    monkeypatch.setattr(scene_generation_sessions, "bpy", blender)
    monkeypatch.setattr(
        scene_generation_sessions,
        "generation_for_scene",
        lambda scene: calls.append(scene.session_uid)
        or SimpleNamespace(number=0, digest=str(scene.session_uid)),
    )
    monkeypatch.setattr(scene_generation_sessions, "_initial_generation_revision", 0)
    monkeypatch.setattr(
        scene_generation_sessions,
        "_initial_generation_diagnostics",
        {"status": "unavailable", "scene_uid": 0, "error": "", "last_error": ""},
    )
    monkeypatch.setattr(engine, "stop_viewport_render_threads_for_file_load", lambda: None)
    monkeypatch.setattr(
        sys.modules["ovrtx_blender_example"],
        "start_runtime_services_async",
        lambda: True,
    )
    monkeypatch.setattr(scene_generation_sessions, "_load_pre_completed", False)
    monkeypatch.setattr(scene_generation_sessions, "_owners", {})
    monkeypatch.setattr(scene_generation_sessions, "_runtimes", {})

    assert scene_generation_sessions.schedule_initial_generation(blender)
    assert not scene_generation_sessions.schedule_initial_generation(blender)
    stale = blender.app.timers.callbacks.pop()

    scene_generation_sessions.load_pre()
    blender.context.scene = second
    scene_generation_sessions.load_post()
    current = blender.app.timers.callbacks.pop()

    stale()
    current()

    assert calls == [42]


def test_eager_generation_failure_is_diagnostic_and_ordinary_demand_retries(
    monkeypatch,
    tmp_path: Path,
) -> None:
    scene = SimpleNamespace(session_uid=51)
    blender = _blender(scene)
    generation = SimpleNamespace(number=0, digest="recovered")
    cursors = []
    blender.context.window = SimpleNamespace(cursor_set=cursors.append)

    class Owner:
        def __init__(self, _path: Path, _cache: Path) -> None:
            self.current_generation = None
            self.pending_generation = None
            self.calls = 0

        def replace(self, _scene: object) -> object:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("export failed")
            self.current_generation = generation
            return generation

    monkeypatch.setenv("OV_BLENDER_EXAMPLE_SCENE_GENERATION_DIR", str(tmp_path))
    monkeypatch.setattr(scene_generation_sessions, "SceneGenerationOwner", Owner)
    monkeypatch.setattr(scene_generation_sessions, "_owners", {})
    monkeypatch.setattr(scene_generation_sessions, "_dirty", {})
    monkeypatch.setattr(scene_generation_sessions, "_constructing", set())
    monkeypatch.setattr(scene_generation_sessions, "_initial_generation_revision", 0)
    monkeypatch.setattr(
        scene_generation_sessions,
        "_initial_generation_diagnostics",
        {"status": "unavailable", "scene_uid": 0, "error": "", "last_error": ""},
    )

    scene_generation_sessions.schedule_initial_generation(blender)
    blender.app.timers.callbacks.pop()()

    assert cursors == ["WAIT", "DEFAULT"]
    assert scene_generation_sessions.diagnostics()["initial_generation"] == {
        "status": "failed",
        "scene_uid": 51,
        "error": "RuntimeError: export failed",
        "last_error": "RuntimeError: export failed",
    }
    assert not scene_generation_sessions.is_authoring(scene)
    assert scene_generation_sessions.generation_for_scene(scene) is generation
    recovered = scene_generation_sessions.diagnostics()["initial_generation"]
    assert recovered["status"] == "ready"
    assert recovered["last_error"] == "RuntimeError: export failed"


def test_load_pre_stops_viewport_threads_before_closing(monkeypatch) -> None:
    events: list[str] = []
    monkeypatch.setattr(scene_generation_sessions, "_load_pre_completed", False)
    monkeypatch.setattr(
        engine,
        "stop_viewport_render_threads_for_file_load",
        lambda: events.append("stop"),
    )
    monkeypatch.setattr(scene_generation_sessions, "close", lambda: events.append("close"))

    scene_generation_sessions.load_pre()

    assert events == ["stop", "close"]


def test_file_load_stop_failure_does_not_skip_other_engines_or_close(monkeypatch) -> None:
    events: list[str] = []

    class _Engine:
        def __init__(self, name: str, *, fail: bool = False) -> None:
            self.name = name
            self.fail = fail

        def _end_viewport_session(self, _reason: object) -> bool:
            events.append(self.name)
            if self.fail:
                raise RuntimeError("dead Blender wrapper")
            return True

    failing = _Engine("failing", fail=True)
    healthy = _Engine("healthy")
    failed_runtime = SimpleNamespace(reuse_blocked=False)
    monkeypatch.setattr(engine, "_ACTIVE_VIEWPORT_ENGINES", {failing, healthy})
    monkeypatch.setattr(
        engine,
        "_ENGINE_RUNTIMES",
        {id(failing): {"authored": True, "generation_runtime": failed_runtime}},
    )
    monkeypatch.setattr(scene_generation_sessions, "_load_pre_completed", False)
    monkeypatch.setattr(scene_generation_sessions, "close", lambda: events.append("close"))

    scene_generation_sessions.load_pre()

    assert set(events[:-1]) == {"failing", "healthy"}
    assert events[-1] == "close"
    assert failed_runtime.reuse_blocked is True


def test_unregister_stop_ends_every_viewport_session(monkeypatch) -> None:
    ended: list[engine.ViewportSessionEndReason] = []

    class _Engine:
        def _end_viewport_session(self, reason: engine.ViewportSessionEndReason) -> bool:
            ended.append(reason)
            return True

    active = _Engine()
    monkeypatch.setattr(engine, "_ACTIVE_VIEWPORT_ENGINES", {active})

    engine.stop_viewport_sessions_for_unregister()

    assert ended == [engine.ViewportSessionEndReason.ENGINE_DESTROYED]


def test_file_load_dead_wrapper_tears_down_sidecar(monkeypatch) -> None:
    class _DeadEngine:
        def __getattribute__(self, _name: str):
            raise ReferenceError("StructRNA has been removed")

    class _Loop:
        def request_stop(self) -> None:
            pass

    class _Thread:
        def stop(self) -> dict[str, object]:
            return {"joined": True, "leaked_thread": False}

    dead = _DeadEngine()
    runtime = {"render_loop": _Loop(), "render_thread": _Thread()}
    monkeypatch.setattr(engine, "_ACTIVE_VIEWPORT_ENGINES", {dead})
    monkeypatch.setattr(engine, "_ENGINE_RUNTIMES", {id(dead): runtime})

    engine.stop_viewport_render_threads_for_file_load()

    assert id(dead) not in engine._ENGINE_RUNTIMES
    assert runtime == {
        "render_loop": None,
        "render_thread": None,
        "teardown": None,
        "teardown_state": None,
        "stop_confirmed": True,
    }


def test_affected_blender_ids_include_light_data() -> None:
    light = SimpleNamespace(
        session_uid=22,
        library=None,
        override_library=None,
        bl_rna=SimpleNamespace(identifier="Light"),
    )

    assert scene_generation_sessions.affected_blender_ids(
        SimpleNamespace(updates=(SimpleNamespace(id=light),))
    ) == {BlenderId("LIGHT", 22)}


def test_affected_blender_ids_include_world_data() -> None:
    world = SimpleNamespace(
        session_uid=30,
        library=None,
        override_library=None,
        bl_rna=SimpleNamespace(identifier="World"),
    )

    assert scene_generation_sessions.affected_blender_ids(
        SimpleNamespace(updates=(SimpleNamespace(id=world),))
    ) == {BlenderId("WORLD", 30)}


def test_affected_blender_ids_use_concrete_light_type_and_skip_camera_object() -> None:
    point_light = SimpleNamespace(
        session_uid=22,
        library=None,
        override_library=None,
        bl_rna=SimpleNamespace(identifier="PointLight"),
    )
    camera = SimpleNamespace(
        session_uid=23,
        type="CAMERA",
        library=None,
        override_library=None,
        bl_rna=SimpleNamespace(identifier="Object"),
    )

    assert scene_generation_sessions.affected_blender_ids(
        SimpleNamespace(
            updates=(SimpleNamespace(id=point_light), SimpleNamespace(id=camera))
        )
    ) == {BlenderId("LIGHT", 22)}


def test_render_generation_is_reused_independently_of_callback_source(
    monkeypatch,
    tmp_path: Path,
) -> None:
    generations = [SimpleNamespace(number=0)]
    reconciled = [SimpleNamespace(number=1)]
    owners = []

    class Owner:
        def __init__(self, path: Path, _cache: Path) -> None:
            self.path = path
            self.current_generation = None
            self.replace_calls = 0
            self.reconcile_calls = []
            owners.append(self)

        def replace(self, _scene: object) -> object | None:
            self.replace_calls += 1
            if generations:
                self.current_generation = generations.pop(0)
                return self.current_generation
            return None

        def reuse(self) -> object:
            return self.current_generation

        def reconcile(self, _scene: object, affected: set[BlenderId]) -> object | None:
            self.reconcile_calls.append(set(affected))
            self.current_generation = reconciled.pop(0)
            return self.current_generation

    monkeypatch.setattr(scene_generation_sessions, "SceneGenerationOwner", Owner)
    monkeypatch.setattr(scene_generation_sessions, "_owners", {})
    monkeypatch.setattr(scene_generation_sessions, "_dirty", {})
    scene = SimpleNamespace(session_uid=12)

    first = scene_generation_sessions.generation_for_scene(
        scene,
        work_root=tmp_path,
    )
    drawn = scene_generation_sessions.generation_for_scene(
        scene,
        work_root=tmp_path,
    )
    affected = {BlenderId("MESH", 91)}
    scene_generation_sessions.mark_scene_dirty(scene, affected)
    changed = scene_generation_sessions.generation_for_scene(
        scene,
        work_root=tmp_path,
    )
    final = scene_generation_sessions.generation_for_scene(
        scene,
        work_root=tmp_path,
    )

    assert (first.number, drawn.number, changed.number, final.number) == (0, 0, 1, 1)
    assert owners[0].replace_calls == 1
    assert owners[0].reconcile_calls == [affected]
    assert owners[0].current_generation is changed
    assert len(owners) == 1
    assert owners[0].path.parent == (tmp_path / "scene-12").resolve()
    assert owners[0].path.name.startswith("scene-generations-")


def test_initial_generation_construction_rejects_reentrant_lookup(
    monkeypatch,
    tmp_path: Path,
) -> None:
    errors = []

    class Owner:
        def __init__(self, _path: Path, _cache: Path) -> None:
            self.current_generation = None

        def replace(self, scene: object) -> object:
            with pytest.raises(RuntimeError) as error:
                scene_generation_sessions.generation_for_scene(scene)
            errors.append(str(error.value))
            self.current_generation = SimpleNamespace(number=0)
            return self.current_generation

    monkeypatch.setattr(scene_generation_sessions, "SceneGenerationOwner", Owner)
    monkeypatch.setattr(scene_generation_sessions, "_owners", {})
    monkeypatch.setattr(scene_generation_sessions, "_dirty", {})
    monkeypatch.setattr(scene_generation_sessions, "_constructing", set())

    generation = scene_generation_sessions.generation_for_scene(
        SimpleNamespace(session_uid=13),
        work_root=tmp_path,
    )

    assert generation.number == 0
    assert errors == ["scene generation construction is already in progress"]
    assert scene_generation_sessions.diagnostics()["constructing_scene_uids"] == []


def test_initial_generation_preserves_dirty_notifications_received_during_export(
    monkeypatch,
    tmp_path: Path,
) -> None:
    affected = {BlenderId("MESH", 91)}
    first = SimpleNamespace(number=0)
    second = SimpleNamespace(number=1)
    suppressed = []

    class Owner:
        def __init__(self, _path: Path, _cache: Path) -> None:
            self.current_generation = None
            self.reconcile_calls = []

        def replace(self, scene: object) -> object:
            suppressed.append(
                interactive_operator_state.interactive_edit_bridge_suppressed()
            )
            scene_generation_sessions.mark_scene_dirty(scene, affected)
            self.current_generation = first
            return first

        def reconcile(self, _scene: object, received: set[BlenderId]) -> object:
            self.reconcile_calls.append(set(received))
            self.current_generation = second
            return second

        def reuse(self) -> object:
            return self.current_generation

    monkeypatch.setattr(scene_generation_sessions, "SceneGenerationOwner", Owner)
    monkeypatch.setattr(scene_generation_sessions, "_owners", {})
    monkeypatch.setattr(scene_generation_sessions, "_dirty", {})
    monkeypatch.setattr(scene_generation_sessions, "_constructing", set())
    scene = SimpleNamespace(session_uid=12)

    assert (
        scene_generation_sessions.generation_for_scene(scene, work_root=tmp_path)
        is first
    )
    assert scene_generation_sessions._dirty[12] == affected
    assert suppressed == [True]

    assert (
        scene_generation_sessions.generation_for_scene(scene, work_root=tmp_path)
        is second
    )
    assert scene_generation_sessions._owners[12].reconcile_calls == [affected]


def test_initial_generation_failure_clears_authoring_and_allows_retry(
    monkeypatch,
    tmp_path: Path,
) -> None:
    generation = SimpleNamespace(number=0)

    class Owner:
        def __init__(self, _path: Path, _cache: Path) -> None:
            self.current_generation = None
            self.replace_calls = 0

        def replace(self, scene: object) -> object:
            assert scene_generation_sessions.is_authoring(scene)
            self.replace_calls += 1
            if self.replace_calls == 1:
                raise RuntimeError("export failed")
            self.current_generation = generation
            return generation

    monkeypatch.setattr(scene_generation_sessions, "SceneGenerationOwner", Owner)
    monkeypatch.setattr(scene_generation_sessions, "_owners", {})
    monkeypatch.setattr(scene_generation_sessions, "_dirty", {})
    monkeypatch.setattr(scene_generation_sessions, "_constructing", set())
    scene = SimpleNamespace(session_uid=14)

    with pytest.raises(RuntimeError, match="export failed"):
        scene_generation_sessions.generation_for_scene(scene, work_root=tmp_path)

    assert not scene_generation_sessions.is_authoring(scene)
    assert scene_generation_sessions.generation_for_scene(scene, work_root=tmp_path) is generation


def test_reconciliation_authoring_preserves_concurrent_dirty_notification(
    monkeypatch,
) -> None:
    first = BlenderId("OBJECT", 91)
    concurrent = BlenderId("MESH", 92)
    generation = SimpleNamespace(number=0)
    scene = SimpleNamespace(session_uid=15)
    intervals = []

    class Owner:
        current_generation = generation
        pending_generation = None

        def reconcile(self, current_scene: object, affected: object) -> None:
            assert affected == {first}
            intervals.append(
                (
                    scene_generation_sessions.is_authoring(current_scene),
                    interactive_operator_state.interactive_edit_bridge_suppressed(),
                )
            )
            scene_generation_sessions.mark_scene_dirty(current_scene, {concurrent})
            return None

        def reuse(self) -> object:
            return self.current_generation

    monkeypatch.setattr(scene_generation_sessions, "_owners", {15: Owner()})
    monkeypatch.setattr(scene_generation_sessions, "_dirty", {15: {first}})
    monkeypatch.setattr(scene_generation_sessions, "_reconciling", set())
    monkeypatch.setattr(scene_generation_sessions, "_blocked_reconciliations", {})

    assert scene_generation_sessions.generation_for_scene(scene) is generation
    assert intervals == [(True, True)]
    assert scene_generation_sessions._dirty[15] == {concurrent}
    assert not scene_generation_sessions.is_authoring(scene)


def test_scene_owner_retains_typed_edit_without_viewport_runtime(monkeypatch) -> None:
    retained = []
    generation = SimpleNamespace(number=0)
    owner = SimpleNamespace(
        current_generation=generation,
        retain_transform_values=lambda values: retained.extend(values),
        retain_attribute_values=lambda _values: None,
        retain_initial_conditions=lambda _values: None,
        close=lambda: None,
    )
    scene = SimpleNamespace(session_uid=12)
    monkeypatch.setattr(scene_generation_sessions, "_owners", {12: owner})

    scene_generation_sessions.retain_interactive_edit(
        scene,
        InteractiveEdit(
            shape=EditShape.VALUE,
            data_authority=DataAuthority.VIEW,
            **edit_location(
                usd_prim_path="/World/Cube",
                usd_attribute="xformOp:transform",
                blender_property_path="matrix_world",
            ),
            value=((1.0, 0.0, 0.0, 2.0),) * 4,
        ),
    )

    assert retained[0].prim_path == "/World/Cube"


def test_edit_during_activation_queues_on_authoring_scheduler(monkeypatch) -> None:
    retained = []
    owner = SimpleNamespace(
        current_generation=SimpleNamespace(number=0),
        retain_transform_values=lambda values: retained.extend(values),
        retain_attribute_values=lambda _values: None,
    )
    scheduler = RuntimeScheduler(
        ovrtx_transform_sink=owner.retain_transform_values,
        ovrtx_attribute_sink=owner.retain_attribute_values,
    )
    scene = SimpleNamespace(session_uid=12)
    monkeypatch.setattr(scene_generation_sessions, "_owners", {12: owner})
    monkeypatch.setattr(
        scene_generation_sessions,
        "_runtimes",
        {12: SimpleNamespace(scheduler=scheduler, viewport_ids=set())},
    )
    monkeypatch.setattr(scene_generation_sessions, "_wake_preparation", lambda _uid: None)

    result = scene_generation_sessions.retain_interactive_edit(
        scene,
        InteractiveEdit(
            shape=EditShape.VALUE,
            data_authority=DataAuthority.VIEW,
            **edit_location(
                usd_prim_path="/World/Hat",
                usd_attribute="xformOp:transform",
                blender_property_path="matrix_world",
            ),
            value=((1.0, 0.0, 0.0, 2.0),) * 4,
        ),
    )

    assert result is not None and result.status is EditStatus.QUEUED
    assert scheduler.has_pending_view_updates is True
    assert [value.prim_path for value in retained] == ["/World/Hat"]


def test_activation_drains_only_latest_queued_desired_value(monkeypatch) -> None:
    applied: list[OvrtxTransformValue] = []
    retained: list[OvrtxTransformValue] = []

    class Port:
        def update_transforms(self, values: object) -> OvrtxValueUpdateResult:
            applied.extend(values)
            return OvrtxValueUpdateResult(len(values), 0)

        def update_attribute_values(self, values: object) -> OvrtxValueUpdateResult:
            return OvrtxValueUpdateResult(len(values), 0 if values else None)

    class Controller:
        def apply_runtime_updates(self, operation: object) -> object:
            return operation(Port(), False)

    class RenderAdapter:
        def __init__(self, _controller: object) -> None:
            self.controller = Controller()
            self.last_error = ""
            self.last_ensure_result = SimpleNamespace(session_started=True)

        def update_request(self, _request: object) -> None:
            pass

        def activate(self, _generation: object) -> bool:
            return True

    class PhysicsAdapter:
        controller = None
        active_generation = None

        def reset(self) -> bool:
            return True

    owner = SimpleNamespace(
        retained_values_for=lambda _generation: (tuple(retained[-1:]), (), ()),
        retain_transform_values=lambda values: retained.extend(values),
        retain_attribute_values=lambda _values: None,
        retain_initial_conditions=lambda _values: None,
    )
    monkeypatch.setattr(scene_generation_sessions, "OvrtxGenerationAdapter", RenderAdapter)
    monkeypatch.setattr(scene_generation_sessions, "OvphysxGenerationAdapter", PhysicsAdapter)
    monkeypatch.setattr(scene_generation_sessions, "generation_requires_physics", lambda _value: False)
    runtime = scene_generation_sessions.AuthoringGenerationRuntime(object(), owner)
    for translation in (1.0, 2.0):
        edit = InteractiveEdit(
            shape=EditShape.VALUE,
            data_authority=DataAuthority.VIEW,
            **edit_location(
                usd_prim_path="/World/Hat",
                usd_attribute="xformOp:transform",
                blender_property_path="matrix_world",
            ),
            value=((1.0, 0.0, 0.0, translation),) * 4,
        )
        runtime.scheduler.submit_edit(InteractiveEditPlanner().plan(edit).to_intent())

    runtime.activate(
        SimpleNamespace(number=1),
        RenderRequest(input_usd_path="/tmp/scene.usda"),
    )

    assert len(applied) == 1
    assert applied[0].matrix[0][3] == 2.0
    assert runtime.last_activation_update.values_written is True
    assert runtime.scheduler.has_pending_view_updates is False


def test_edit_committed_during_activation_applies_before_ready(monkeypatch) -> None:
    activation_started = threading.Event()
    continue_activation = threading.Event()
    applied: list[OvrtxTransformValue] = []

    class Port:
        def update_transforms(self, values: object) -> OvrtxValueUpdateResult:
            applied.extend(values)
            return OvrtxValueUpdateResult(len(values), 0)

        def update_attribute_values(self, values: object) -> OvrtxValueUpdateResult:
            return OvrtxValueUpdateResult(len(values), 0 if values else None)

    class Controller:
        def apply_runtime_updates(self, operation: object) -> object:
            return operation(Port(), False)

    class RenderAdapter:
        def __init__(self, _controller: object) -> None:
            self.controller = Controller()
            self.last_error = ""
            self.last_ensure_result = SimpleNamespace(session_started=True)

        def update_request(self, _request: object) -> None:
            pass

        def activate(self, _generation: object) -> bool:
            activation_started.set()
            assert continue_activation.wait(1.0)
            return True

    class PhysicsAdapter:
        controller = None
        active_generation = None

        def reset(self) -> bool:
            return True

    owner = SimpleNamespace(
        retained_values_for=lambda _generation: ((), (), ()),
        retain_transform_values=lambda _values: None,
        retain_attribute_values=lambda _values: None,
        retain_initial_conditions=lambda _values: None,
    )
    monkeypatch.setattr(scene_generation_sessions, "OvrtxGenerationAdapter", RenderAdapter)
    monkeypatch.setattr(scene_generation_sessions, "OvphysxGenerationAdapter", PhysicsAdapter)
    monkeypatch.setattr(scene_generation_sessions, "generation_requires_physics", lambda _value: False)
    runtime = scene_generation_sessions.AuthoringGenerationRuntime(object(), owner)
    generation = SimpleNamespace(number=1)
    activation = threading.Thread(
        target=runtime.activate,
        args=(generation, RenderRequest(input_usd_path="/tmp/scene.usda")),
    )
    activation.start()
    assert activation_started.wait(1.0)

    result = runtime.submit_edit_group(
        (
            InteractiveEdit(
                shape=EditShape.VALUE,
                data_authority=DataAuthority.VIEW,
                **edit_location(
                    usd_prim_path="/World/Hat",
                    usd_attribute="xformOp:transform",
                    blender_property_path="matrix_world",
                ),
                value=((1.0, 0.0, 0.0, 3.0),) * 4,
            ),
        )
    )
    continue_activation.set()
    activation.join(1.0)

    assert result[0].status is EditStatus.QUEUED
    assert activation.is_alive() is False
    assert runtime.preparation_status == "ready"
    assert [value.prim_path for value in applied] == ["/World/Hat"]
    assert applied[0].matrix[0][3] == 3.0
    assert runtime.scheduler.has_pending_view_updates is False


def test_scene_owner_retains_camera_edit_without_viewport_runtime(monkeypatch) -> None:
    retained = []
    owner = SimpleNamespace(
        current_generation=SimpleNamespace(number=0),
        retain_transform_values=lambda values: retained.extend(values),
        retain_attribute_values=lambda _values: None,
        retain_initial_conditions=lambda _values: None,
    )
    scene = SimpleNamespace(session_uid=12)
    monkeypatch.setattr(scene_generation_sessions, "_owners", {12: owner})
    monkeypatch.setattr(scene_generation_sessions, "_runtimes", {})
    monkeypatch.setattr(scene_generation_sessions, "_wake_preparation", lambda _uid: None)
    matrix = ((1.0, 0.0, 0.0, 2.0),) * 4

    scene_generation_sessions.retain_interactive_edit(
        scene,
        InteractiveEdit(
            shape=EditShape.VALUE,
            data_authority=DataAuthority.VIEW,
            **edit_location(
                usd_prim_path="/World/Camera",
                usd_attribute="omni:xform",
                blender_property_path="viewport_camera_matrix",
                provenance={"source": "viewport_camera"},
            ),
            value=matrix,
        ),
    )

    assert retained == [
        OvrtxTransformValue("/World/Camera", [list(row) for row in matrix])
    ]


def test_scene_owner_rebinds_retained_camera_across_generation_handoff(
    monkeypatch,
    tmp_path: Path,
) -> None:
    camera_id = BlenderId("OBJECT", 10)
    predecessor = SimpleNamespace(
        blender_prim_paths={
            camera_id: BlenderPrimPath(
                "Camera", "CAMERA", "/World/Camera", "/World/Camera/Camera"
            )
        }
    )
    replacement = SimpleNamespace(
        blender_prim_paths={
            camera_id: BlenderPrimPath(
                "Camera",
                "CAMERA",
                "/World/Generation_1/Camera",
                "/World/Generation_1/Camera/Camera",
            )
        }
    )
    owner = SceneGenerationOwner(tmp_path / "generations")
    owner._current = predecessor  # noqa: SLF001
    scene = SimpleNamespace(session_uid=12)
    monkeypatch.setattr(scene_generation_sessions, "_owners", {12: owner})
    monkeypatch.setattr(scene_generation_sessions, "_runtimes", {})
    monkeypatch.setattr(scene_generation_sessions, "_wake_preparation", lambda _uid: None)
    matrix = ((1.0, 0.0, 0.0, 2.0),) * 4

    scene_generation_sessions.retain_interactive_edit(
        scene,
        InteractiveEdit(
            shape=EditShape.VALUE,
            data_authority=DataAuthority.VIEW,
            **edit_location(
                usd_prim_path="/World/Camera",
                usd_attribute="omni:xform",
                blender_property_path="viewport_camera_matrix",
                provenance={"source": "viewport_camera"},
            ),
            value=matrix,
        ),
    )

    transforms, _attributes, _initial_conditions = owner.retained_values_for(
        replacement
    )

    assert transforms == (
        OvrtxTransformValue(
            "/World/Generation_1/Camera", [list(row) for row in matrix]
        ),
    )


def test_physics_rejected_transform_is_not_replayed_by_immediate_final_render(
    monkeypatch,
    tmp_path: Path,
) -> None:
    object_id = BlenderId("OBJECT", 10)
    generation = SimpleNamespace(
        number=0,
        usd_path="/tmp/current.usda",
        blender_prim_paths={
            object_id: BlenderPrimPath(
                "Cube", "MESH", "/World/Cube", "/World/Cube/Cube"
            )
        },
    )
    owner = SceneGenerationOwner(tmp_path / "generations")
    owner._current = generation  # noqa: SLF001
    accepted_matrix = ((1.0, 0.0, 0.0, 0.0),) * 4
    rejected_matrix = ((2.0, 0.0, 0.0, 0.0),) * 4
    owner.retain_transform_values(
        (OvrtxTransformValue("/World/Cube", accepted_matrix),)
    )
    scene = SimpleNamespace(session_uid=12)
    edit = InteractiveEdit(
        shape=EditShape.VALUE,
        data_authority=DataAuthority.VIEW,
        **edit_location(
            usd_prim_path="/World/Cube",
            usd_attribute="xformOp:transform",
            blender_property_path="matrix_world",
        ),
        value=rejected_matrix,
    )

    class ActiveEngine:
        def build_interactive_edits_from_depsgraph(
            self, _depsgraph: object, **_kwargs: object
        ) -> list[InteractiveEdit]:
            return [edit]

        def submit_interactive_edit(self, _edit: InteractiveEdit) -> EditWorkflowResult:
            return EditWorkflowResult(
                action=WorkflowAction.UNSUPPORTED,
                status=EditStatus.UNSUPPORTED,
                reason="physics_playback_locked",
            )

    callback = BlenderEditCallbackAdapter(
        active_engines=lambda: (ActiveEngine(),),
        selection_resolver=lambda _context: {
            "changed": False,
            "group_rejected": False,
        },
        edit_observer=scene_generation_sessions.retain_interactive_edit,
    )
    result = callback.submit_depsgraph_interactive_edits(
        SimpleNamespace(updates=()),
        scene=scene,
    )
    activations = []

    class FinalAdapter:
        last_error = ""

        def __init__(self, _controller: object) -> None:
            pass

        def update_request(self, _request: object) -> None:
            pass

        def activate(self, _generation: object, **values: object) -> bool:
            activations.append(values)
            return True

        def deactivate(self) -> str:
            return "stopped"

    monkeypatch.setattr(scene_generation_sessions, "_owners", {12: owner})
    monkeypatch.setattr(scene_generation_sessions, "_runtimes", {})
    monkeypatch.setattr(scene_generation_sessions, "OvrtxGenerationAdapter", FinalAdapter)
    scene_generation_sessions.activate_for_final_render(
        scene,
        RenderRequest(input_usd_path=generation.usd_path),
        controller=object(),
    )

    assert result[0].reason == "physics_playback_locked"
    assert activations[0]["transform_values"][0].matrix == accepted_matrix


def test_scene_owner_reports_topology_replacement_without_viewport(monkeypatch) -> None:
    scene = SimpleNamespace(session_uid=12)
    monkeypatch.setattr(
        scene_generation_sessions,
        "_owners",
        {12: SimpleNamespace(current_generation=SimpleNamespace(number=0))},
    )

    result = scene_generation_sessions.retain_interactive_edit(
        scene,
        InteractiveEdit(
            shape=EditShape.TOPOLOGY,
            data_authority=DataAuthority.VIEW,
            **edit_location(
                usd_prim_path="/World/Materials/Paint",
                blender_property_path="material_topology",
            ),
        ),
    )

    assert result is not None
    assert result.accepted
    assert result.plan is not None
    assert result.plan.impact.scene_generation_replacement_requested


def test_current_generation_edit_context_scans_owned_generation(monkeypatch) -> None:
    scans = []

    class Resolver:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def scan(self, request: RenderRequest) -> None:
            scans.append(request.input_usd_path)

    scene = SimpleNamespace(session_uid=12, objects=("light",))
    monkeypatch.setattr(
        scene_generation_sessions,
        "_owners",
        {
            12: SimpleNamespace(
                current_generation=SimpleNamespace(
                    usd_path="/tmp/current.usda",
                    blender_prim_paths={
                        BlenderId("OBJECT", 91): BlenderPrimPath(
                            "Cube", "MESH", "/World/Cube", "/World/Cube/Cube"
                        )
                    },
                )
            )
        },
    )
    monkeypatch.setattr(scene_generation_sessions, "UsdPrimResolver", Resolver)

    resolver, lights = scene_generation_sessions.current_generation_edit_context(scene)

    assert isinstance(resolver, Resolver)
    assert scans == ["/tmp/current.usda"]
    assert lights == ("light",)
    topology_resolver = resolver.kwargs.pop("mesh_topology_change_resolver")
    assert callable(topology_resolver)
    assert resolver.kwargs == {
        "object_paths_by_session_uid": {91: "/World/Cube"},
        "light_paths_by_object_session_uid": {},
    }


def _selected_edit(
    source_uid: int,
    prim_path: str,
    *,
    blender_property_path: str = "matrix_world",
    usd_attribute: str = "xformOp:transform",
) -> InteractiveEdit:
    return InteractiveEdit(
        shape=EditShape.VALUE,
        data_authority=DataAuthority.VIEW,
        **edit_location(
            usd_prim_path=prim_path,
            usd_attribute=usd_attribute,
            blender_property_path=blender_property_path,
            provenance={
                "blender_id_kind": "OBJECT",
                "blender_session_uid": source_uid,
                "selection_resolution": {
                    "source_session_uid": source_uid,
                }
            },
        ),
        value=((1.0, 0.0, 0.0, 0.0),) * 4,
    )


def _selection(*source_uids: int) -> dict[str, object]:
    return {
        "selected_object_count": len(source_uids),
        "sources": [
            {"source_session_uid": source_uid, "status": "unresolved"}
            for source_uid in source_uids
        ],
    }


def _mapped_generation(*source_uids: int) -> SimpleNamespace:
    return SimpleNamespace(
        blender_prim_paths={
            BlenderId("OBJECT", source_uid): BlenderPrimPath(
                f"Object{source_uid}",
                "MESH",
                f"/World/Object{source_uid}",
                f"/World/Object{source_uid}/Schema",
            )
            for source_uid in source_uids
        }
    )


def test_current_scene_group_admission_uses_stable_generation_mapping(monkeypatch) -> None:
    scene = SimpleNamespace(session_uid=12)
    generation = _mapped_generation(101)
    monkeypatch.setattr(
        scene_generation_sessions,
        "_owners",
        {12: SimpleNamespace(current_generation=generation)},
    )
    edit = _selected_edit(101, "/World/Object101")

    assert scene_generation_sessions.resolve_current_scene_edit_group(
        scene, (edit,), _selection(101)
    ) == (edit,)
    assert scene_generation_sessions.resolve_current_scene_edit_group(
        scene, (edit,), _selection(101, 202)
    ) == ()


def test_single_selected_source_allows_one_authoritatively_mapped_data_edit(
    monkeypatch,
) -> None:
    scene = SimpleNamespace(session_uid=12)
    generation = _mapped_generation(101)
    generation.blender_prim_paths[BlenderId("MATERIAL", 303)] = BlenderPrimPath(
        "Paint",
        "MATERIAL",
        "/World/Materials/Paint",
        "/World/Materials/Paint",
    )
    monkeypatch.setattr(
        scene_generation_sessions,
        "_owners",
        {12: SimpleNamespace(current_generation=generation)},
    )
    edit = InteractiveEdit(
        shape=EditShape.VALUE,
        data_authority=DataAuthority.VIEW,
        **edit_location(
            usd_prim_path="/World/Materials/Paint",
            usd_attribute="inputs:roughness",
            blender_property_path="material.roughness",
            provenance={
                "material_path": "/World/Materials/Paint",
                "blender_id_kind": "MATERIAL",
                "blender_session_uid": 303,
            },
        ),
        value=0.5,
    )

    assert scene_generation_sessions.resolve_current_scene_edit_group(
        scene, (edit,), _selection(101)
    ) == (edit,)


def test_shared_mesh_data_topology_uses_the_edit_prim_path() -> None:
    first = BlenderPrimPath(
        "First", "MESH", "/World/First", "/World/First/Mesh", data_session_uid=303
    )
    second = BlenderPrimPath(
        "Second", "MESH", "/World/Second", "/World/Second/Mesh", data_session_uid=303
    )
    generation = SimpleNamespace(
        blender_prim_paths={
            BlenderId("OBJECT", 101): first,
            BlenderId("OBJECT", 102): second,
        }
    )
    edit = InteractiveEdit(
        shape=EditShape.TOPOLOGY,
        data_authority=DataAuthority.VIEW,
        **edit_location(
            usd_prim_path=second.schema_path,
            blender_property_path="vertices",
            provenance={
                "blender_id_kind": "MESH",
                "blender_session_uid": 303,
            },
        ),
    )

    assert scene_generation_sessions._edit_source_mapping(generation, edit) is second


def test_direct_light_data_edit_maps_without_selection(monkeypatch) -> None:
    scene = SimpleNamespace(session_uid=12)
    generation = _mapped_generation(101)
    generation.blender_prim_paths[BlenderId("OBJECT", 101)] = BlenderPrimPath(
        "Key",
        "LIGHT",
        "/World/Key",
        "/World/Key/KeyData",
        data_session_uid=303,
    )
    monkeypatch.setattr(
        scene_generation_sessions,
        "_owners",
        {12: SimpleNamespace(current_generation=generation)},
    )
    edit = InteractiveEdit(
        shape=EditShape.VALUE,
        data_authority=DataAuthority.VIEW,
        **edit_location(
            usd_prim_path="/World/Key/KeyData",
            usd_attribute="inputs:intensity",
            blender_property_path="energy",
            provenance={
                "blender_id_kind": "POINTLIGHT",
                "blender_session_uid": 303,
            },
        ),
        value=100.0,
    )

    assert scene_generation_sessions.resolve_current_scene_edit_group(
        scene, (edit,), _selection()
    ) == (edit,)


def test_incomplete_callbacks_do_not_combine_selection_driven_edits(monkeypatch) -> None:
    scene = SimpleNamespace(session_uid=12)
    generation = _mapped_generation(101, 202)
    monkeypatch.setattr(
        scene_generation_sessions,
        "_owners",
        {12: SimpleNamespace(current_generation=generation)},
    )

    assert scene_generation_sessions.resolve_current_scene_edit_group(
        scene, (_selected_edit(101, "/World/Object101"),), _selection(101, 202)
    ) == ()
    assert scene_generation_sessions.resolve_current_scene_edit_group(
        scene, (_selected_edit(202, "/World/Object202"),), _selection(101, 202)
    ) == ()


def test_current_scene_group_rejects_another_sources_unique_target(monkeypatch) -> None:
    scene = SimpleNamespace(session_uid=12)
    generation = _mapped_generation(101, 202)
    monkeypatch.setattr(
        scene_generation_sessions,
        "_owners",
        {12: SimpleNamespace(current_generation=generation)},
    )
    edit = _selected_edit(101, "/World/Object202")

    assert scene_generation_sessions.resolve_current_scene_edit_group(
        scene, (edit,), _selection(101)
    ) == ()


def test_current_scene_group_retains_latest_sim_value_before_activation(monkeypatch) -> None:
    generation = _mapped_generation(101)
    owner = SimpleNamespace(
        current_generation=generation,
        pending_generation=None,
        retain_transform_values=lambda _values: None,
        retain_attribute_values=lambda _values: None,
        retain_initial_conditions=lambda _values: None,
    )
    scene = SimpleNamespace(session_uid=12)
    monkeypatch.setattr(scene_generation_sessions, "_owners", {12: owner})
    monkeypatch.setattr(scene_generation_sessions, "_runtimes", {})
    monkeypatch.setattr(scene_generation_sessions, "_wake_preparation", lambda _uid: None)

    for translation in (1.0, 2.0):
        edit = replace(
            _selected_edit(101, "/World/Object101"),
            data_authority=DataAuthority.SIM,
            value={
                "translate": (translation, 0.0, 0.0),
                "orient": (0.0, 0.0, 0.0, 1.0),
            },
        )
        results = scene_generation_sessions.submit_current_scene_edit_group(
            scene, (edit,), _selection(101)
        )
        assert len(results) == 1 and results[0].accepted is True

    runtime = scene_generation_sessions.active_runtime_for_scene(scene)
    assert runtime is not None

    class Controller:
        applied = []

        def apply_initial_condition_values(
            self,
            poses: tuple[object, ...],
            *,
            reset: bool,
        ) -> OvphysxStageResult:
            assert reset is False
            self.applied.extend(poses)
            return OvphysxStageResult(
                OvphysxStageStatus.OK,
                "updated",
                poses,
                tuple(pose.prim_path for pose in poses),
                0,
                0,
                1,
            )

    controller = Controller()
    result = runtime.scheduler.apply_pending_sim_values(controller)

    assert result.values_written is True
    assert [pose.translate for pose in controller.applied] == [(2.0, 0.0, 0.0)]


def test_runtime_playback_rejection_queues_none_of_the_group(monkeypatch) -> None:
    generation = _mapped_generation(101, 202)
    owner = SimpleNamespace(
        current_generation=generation,
        pending_generation=None,
        retain_transform_values=lambda _values: None,
        retain_attribute_values=lambda _values: None,
        retain_initial_conditions=lambda _values: None,
    )
    runtime = scene_generation_sessions.AuthoringGenerationRuntime(object(), owner)
    locked = EditWorkflowResult(
        action=WorkflowAction.UNSUPPORTED,
        status=EditStatus.UNSUPPORTED,
        reason="physics_playback_locked",
    )
    runtime.playback_lock = SimpleNamespace(
        reject_edit=lambda edit: locked if edit.usd_prim_path.endswith("202") else None
    )
    revision = runtime.scheduler.presentation_revision

    results = runtime.submit_edit_group(
        (
            _selected_edit(101, "/World/Object101"),
            _selected_edit(202, "/World/Object202"),
        )
    )

    assert [result.reason for result in results] == [
        "edit_group_rejected",
        "physics_playback_locked",
    ]
    assert runtime.scheduler.has_pending_view_updates is False
    assert runtime.scheduler.presentation_revision == revision


def test_current_scene_group_admission_accepts_same_multi_edit_operation(monkeypatch) -> None:
    scene = SimpleNamespace(session_uid=12)
    generation = _mapped_generation(101, 202)
    monkeypatch.setattr(
        scene_generation_sessions,
        "_owners",
        {12: SimpleNamespace(current_generation=generation)},
    )
    edits = tuple(
        _selected_edit(
            source_uid,
            f"/World/Object{source_uid}/Schema",
            blender_property_path=property_path,
            usd_attribute=f"inputs:{property_path}",
        )
        for source_uid in (101, 202)
        for property_path in ("energy", "color")
    )

    assert scene_generation_sessions.resolve_current_scene_edit_group(
        scene, edits, _selection(101, 202)
    ) == edits


def test_current_scene_group_admission_rejects_different_operations(monkeypatch) -> None:
    scene = SimpleNamespace(session_uid=12)
    generation = _mapped_generation(101, 202)
    monkeypatch.setattr(
        scene_generation_sessions,
        "_owners",
        {12: SimpleNamespace(current_generation=generation)},
    )
    edits = (
        _selected_edit(101, "/World/Object101"),
        _selected_edit(
            202,
            "/World/Object202/Schema",
            blender_property_path="energy",
            usd_attribute="inputs:intensity",
        ),
    )

    assert scene_generation_sessions.resolve_current_scene_edit_group(
        scene, edits, _selection(101, 202)
    ) == ()


def test_scene_runtime_creation_is_atomic(monkeypatch) -> None:
    created: list[object] = []
    returned: list[object] = []

    class Runtime:
        reuse_blocked = False

        def __init__(self, *_args: object) -> None:
            self.scheduler = object()
            created.append(self)

    monkeypatch.setattr(scene_generation_sessions, "_runtimes", {})
    monkeypatch.setattr(scene_generation_sessions, "AuthoringGenerationRuntime", Runtime)
    owner = object()

    def obtain() -> None:
        returned.append(
            scene_generation_sessions._runtime_for_owner(
                12,
                owner,
            )
        )

    threads = [threading.Thread(target=obtain) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(created) == 1
    assert returned == [created[0]] * 8


def test_viewport_activation_waits_for_contended_preparation(monkeypatch) -> None:
    entered = threading.Event()
    release = threading.Event()
    activation_lock = threading.Lock()
    completed: list[str] = []
    scheduler = object()

    class Controller:
        def _allow_serialized_threads(self) -> None:
            pass

        def adopt_owning_thread(self) -> None:
            pass

    class Runtime:
        reuse_blocked = False
        viewport_ids: set[str] = set()
        ovrtx = SimpleNamespace(controller=Controller())

        def __init__(self) -> None:
            self.scheduler = scheduler

        def attach(self, viewport_id: str, _wake_hook: object) -> None:
            self.viewport_ids.add(viewport_id)

        def activate_blocking(self, *_args: object, **_kwargs: object) -> None:
            with activation_lock:
                if not entered.is_set():
                    entered.set()
                    assert release.wait(2.0)

    generation = SimpleNamespace(number=1, usd_path="/tmp/current.usda")
    owner = SimpleNamespace(current_generation=generation, pending_generation=None)
    runtime = Runtime()
    scene = SimpleNamespace(session_uid=12)
    request = RenderRequest(input_usd_path=generation.usd_path)
    monkeypatch.setattr(scene_generation_sessions, "_owners", {12: owner})
    monkeypatch.setattr(scene_generation_sessions, "_runtimes", {12: runtime})

    def activate(viewport_id: str) -> None:
        scene_generation_sessions.activate_for_viewport(
            scene,
            request,
            viewport_id=viewport_id,
            expected_runtime=runtime,
        )
        completed.append(viewport_id)

    first = threading.Thread(target=activate, args=("first",))
    second = threading.Thread(target=activate, args=("second",))
    first.start()
    assert entered.wait(1.0)
    second.start()
    second.join(0.05)
    assert second.is_alive()
    assert completed == []
    release.set()
    first.join()
    second.join()

    assert sorted(completed) == ["first", "second"]


def test_runtime_identity_rejected_before_viewport_attachment(monkeypatch) -> None:
    attached: list[str] = []
    runtime = SimpleNamespace(
        reuse_blocked=False,
        scheduler=object(),
        attach=lambda viewport_id, _wake_hook: attached.append(viewport_id),
    )
    generation = SimpleNamespace(number=1, usd_path="/tmp/current.usda")
    scene = SimpleNamespace(session_uid=12)
    monkeypatch.setattr(
        scene_generation_sessions,
        "_owners",
        {12: SimpleNamespace(current_generation=generation, pending_generation=None)},
    )
    monkeypatch.setattr(scene_generation_sessions, "_runtimes", {12: runtime})

    with pytest.raises(RuntimeError, match="replace the authoring runtime"):
        scene_generation_sessions.activate_for_viewport(
            scene,
            RenderRequest(input_usd_path=generation.usd_path),
            viewport_id="viewport",
            expected_runtime=object(),
        )

    assert attached == []


def test_authored_panes_obtain_one_runtime_binding_before_activation(monkeypatch) -> None:
    controllers: list[object] = []

    class Controller:
        def __init__(self) -> None:
            controllers.append(self)

    generation = SimpleNamespace(number=1, usd_path="/tmp/current.usda")
    owner = SimpleNamespace(
        current_generation=generation,
        pending_generation=None,
        retain_transform_values=lambda _values: None,
        retain_attribute_values=lambda _values: None,
        retain_initial_conditions=lambda _values: None,
    )
    scene = SimpleNamespace(session_uid=12)
    monkeypatch.setattr(scene_generation_sessions, "_owners", {12: owner})
    monkeypatch.setattr(scene_generation_sessions, "_runtimes", {})
    monkeypatch.setattr(
        "ovrtx_blender_example.ovrtx_session_controller.OvrtxSessionController",
        Controller,
    )

    first = scene_generation_sessions.runtime_for_viewport(
        scene, viewport_id="first"
    )
    second = scene_generation_sessions.runtime_for_viewport(
        scene, viewport_id="second"
    )

    assert second is first
    assert second.scheduler is first.scheduler
    assert second.ovrtx.controller is first.ovrtx.controller
    assert controllers == [first.ovrtx.controller]


def test_viewport_runtime_activates_render_and_physics_for_owned_generation(
    monkeypatch,
) -> None:
    events: list[str] = []
    wakes: list[str] = []

    class RenderAdapter:
        def __init__(self, controller: object) -> None:
            self.controller = controller
            self.active_generation = None
            self.last_error = ""
            self.last_ensure_result = SimpleNamespace(session_started=True)

        def update_request(self, request: RenderRequest) -> None:
            events.append(f"request:{request.input_usd_path}")

        def activate(self, generation: object, **_kwargs: object) -> bool:
            self.active_generation = generation.number
            events.append(f"render:{generation.number}")
            return True

        def ensure_request(self) -> bool:
            events.append("render:reuse")
            self.last_ensure_result = SimpleNamespace(session_started=False)
            return True

        def deactivate(self) -> str:
            events.append("render:stop")
            self.active_generation = None
            return "stopped"

    class PhysicsAdapter:
        active_generation = None
        controller = object()
        last_error = ""

        def activate(self, generation: object, **_kwargs: object) -> bool:
            self.active_generation = generation.number
            events.append(f"physics:{generation.number}")
            return True

        def deactivate(self) -> str:
            events.append("physics:stop")
            return "stopped"

        def reset(self) -> bool:
            return True

    class Scheduler:
        def __init__(self, **_kwargs: object) -> None:
            self.wake_hook = None
            self.applied_content_notifications = 0

        def set_edit_wake_hook(self, hook: object) -> None:
            self.wake_hook = hook

        @property
        def has_pending_view_updates(self) -> bool:
            return False

        def pending_view_targets(self) -> tuple[frozenset[str], frozenset[tuple[str, str]]]:
            return frozenset(), frozenset()

        def note_applied_content(self) -> None:
            self.applied_content_notifications += 1

        def shutdown(self) -> None:
            events.append("scheduler:stop")

    generation = SimpleNamespace(
        number=4,
        usd_path="/tmp/generation.usdc",
    )
    owner = SimpleNamespace(
        current_generation=generation,
        pending_generation=None,
        retained_values_for=lambda _generation: ((), (), ()),
        retain_transform_values=lambda _values: None,
        retain_attribute_values=lambda _values: None,
        retain_initial_conditions=lambda _values: None,
        close=lambda: None,
    )
    scene = SimpleNamespace(session_uid=12)
    monkeypatch.setattr(scene_generation_sessions, "_owners", {12: owner})
    monkeypatch.setattr(scene_generation_sessions, "_runtimes", {})
    monkeypatch.setattr(scene_generation_sessions, "OvrtxGenerationAdapter", RenderAdapter)
    monkeypatch.setattr(scene_generation_sessions, "OvphysxGenerationAdapter", PhysicsAdapter)
    monkeypatch.setattr(scene_generation_sessions, "RuntimeScheduler", Scheduler)
    monkeypatch.setattr(scene_generation_sessions, "generation_requires_physics", lambda _value: True)
    request = RenderRequest(input_usd_path=generation.usd_path)

    runtime = scene_generation_sessions.activate_for_viewport(
        scene,
        request,
        viewport_id="viewport",
        wake_hook=lambda: wakes.append("viewport"),
    )
    scene_generation_sessions.activate_for_viewport(
        scene,
        request,
        viewport_id="second-viewport",
        wake_hook=lambda: wakes.append("second-viewport"),
    )

    assert scene_generation_sessions.active_runtime_for_scene(scene) is runtime
    assert events[:3] == [
        "request:/tmp/generation.usdc",
        "physics:4",
        "render:4",
    ]
    assert "render:reuse" in events
    assert runtime.scheduler.applied_content_notifications == 1
    runtime.scheduler.wake_hook()
    assert wakes == ["viewport", "second-viewport"]
    assert scene_generation_sessions.detach_viewport("viewport") is True
    assert runtime.viewport_ids == {"second-viewport"}
    runtime.scheduler.wake_hook()
    assert wakes == ["viewport", "second-viewport", "second-viewport"]
    assert "render:stop" not in events
    scene_generation_sessions.detach_viewport("second-viewport")
    physics_controller = runtime.ovphysx.controller
    assert scene_generation_sessions.pause_preparation() is True
    try:
        assert scene_generation_sessions.deactivate_all_ovrtx() is True
        scene_generation_sessions.activate_for_viewport(
            scene,
            request,
            viewport_id="restarted-viewport",
        )
    finally:
        scene_generation_sessions.resume_preparation()
    assert runtime.ovphysx.controller is physics_controller
    assert events.count("physics:4") == 1
    assert events.count("render:4") == 2
    scene_generation_sessions.close()
    assert events[-3:] == ["scheduler:stop", "render:stop", "physics:stop"]


def test_hidden_value_wake_replays_latest_state_off_presentation(monkeypatch) -> None:
    replayed = threading.Event()
    generation = SimpleNamespace(number=4)

    class Controller:
        def adopt_owning_thread(self) -> None:
            pass

    class Runtime:
        viewport_ids = set()
        ovrtx = SimpleNamespace(controller=Controller())
        ovphysx = SimpleNamespace(active_generation=None)
        preparation_status = "unavailable"
        preparation_error = ""
        _preparation_lock = threading.RLock()

        def replay_retained_values(self, value: object) -> None:
            assert value is generation
            replayed.set()

    monkeypatch.setattr(
        scene_generation_sessions,
        "_owners",
        {12: SimpleNamespace(current_generation=generation, pending_generation=None)},
    )
    monkeypatch.setattr(scene_generation_sessions, "_runtimes", {12: Runtime()})
    monkeypatch.setattr(scene_generation_sessions, "_dirty", {})
    monkeypatch.setattr(scene_generation_sessions, "_preparation_pending", {})
    monkeypatch.setattr(scene_generation_sessions, "_preparation_revisions", {})
    monkeypatch.setattr(scene_generation_sessions, "generation_requires_physics", lambda _value: False)
    scene_generation_sessions._stop_preparation_worker()
    scene_generation_sessions._wake_preparation(12)

    assert replayed.wait(timeout=1.0)
    scene_generation_sessions._stop_preparation_worker()


def test_hidden_retryable_sim_busy_retries_without_another_wake(monkeypatch) -> None:
    replayed = threading.Event()
    generation = SimpleNamespace(number=4)

    class Controller:
        def adopt_owning_thread(self) -> None:
            pass

    class Scheduler:
        has_pending_sim_updates = True

    class Runtime:
        viewport_ids = set()
        ovrtx = SimpleNamespace(controller=Controller())
        ovphysx = SimpleNamespace(active_generation=None)
        scheduler = Scheduler()
        preparation_status = "unavailable"
        preparation_error = ""
        _preparation_lock = threading.RLock()
        attempts = 0

        def replay_retained_values(self, value: object) -> None:
            assert value is generation
            self.attempts += 1
            if self.attempts == 1:
                self.last_activation_update = RuntimeTickResult(
                    status=RuntimeTickStatus.BUSY,
                    enabled=True,
                )
                raise RuntimeError("physics runtime busy")
            self.scheduler.has_pending_sim_updates = False
            replayed.set()

    runtime = Runtime()
    monkeypatch.setattr(
        scene_generation_sessions,
        "_owners",
        {12: SimpleNamespace(current_generation=generation, pending_generation=None)},
    )
    monkeypatch.setattr(scene_generation_sessions, "_runtimes", {12: runtime})
    monkeypatch.setattr(scene_generation_sessions, "_dirty", {})
    monkeypatch.setattr(scene_generation_sessions, "_preparation_pending", {})
    monkeypatch.setattr(scene_generation_sessions, "_preparation_revisions", {})
    monkeypatch.setattr(
        scene_generation_sessions, "generation_requires_physics", lambda _value: False
    )
    scene_generation_sessions._stop_preparation_worker()
    scene_generation_sessions._wake_preparation(12)

    assert replayed.wait(timeout=2.0)
    assert runtime.attempts == 2
    assert runtime.preparation_status == "ready"
    scene_generation_sessions._stop_preparation_worker()


def test_hidden_terminal_failure_with_pending_sim_does_not_retry(monkeypatch) -> None:
    failed = threading.Event()
    retried = threading.Event()
    generation = SimpleNamespace(number=4)

    class Controller:
        def adopt_owning_thread(self) -> None:
            pass

    class Runtime:
        viewport_ids = set()
        ovrtx = SimpleNamespace(controller=Controller())
        ovphysx = SimpleNamespace(active_generation=None)
        scheduler = SimpleNamespace(has_pending_sim_updates=True)
        preparation_status = "unavailable"
        preparation_error = ""
        _preparation_lock = threading.RLock()
        attempts = 0

        def replay_retained_values(self, value: object) -> None:
            assert value is generation
            self.attempts += 1
            if self.attempts > 1:
                retried.set()
            failed.set()
            raise RuntimeError("terminal activation failure")

    runtime = Runtime()
    monkeypatch.setattr(
        scene_generation_sessions,
        "_owners",
        {12: SimpleNamespace(current_generation=generation, pending_generation=None)},
    )
    monkeypatch.setattr(scene_generation_sessions, "_runtimes", {12: runtime})
    monkeypatch.setattr(scene_generation_sessions, "_dirty", {})
    monkeypatch.setattr(scene_generation_sessions, "_preparation_pending", {})
    monkeypatch.setattr(scene_generation_sessions, "_preparation_revisions", {})
    monkeypatch.setattr(
        scene_generation_sessions, "generation_requires_physics", lambda _value: False
    )
    scene_generation_sessions._stop_preparation_worker()
    scene_generation_sessions._wake_preparation(12)

    assert failed.wait(timeout=1.0)
    assert retried.wait(timeout=0.7) is False
    assert runtime.attempts == 1
    assert runtime.preparation_status == "failed"
    assert "terminal activation failure" in runtime.preparation_error
    scene_generation_sessions._stop_preparation_worker()


def test_fail_closed_runtime_does_not_block_unrelated_cleanup(monkeypatch) -> None:
    events: list[str] = []

    class Runtime:
        def __init__(self, name: str) -> None:
            self.name = name
            self.reuse_blocked = False
            self.lifecycle_status = "open"

        def close(self) -> bool:
            events.append(f"close:{self.name}")
            self.lifecycle_status = "closed"
            return True

    blocked = Runtime("blocked")
    healthy = Runtime("healthy")
    owners = {
        1: SimpleNamespace(close=lambda: events.append("owner:blocked")),
        2: SimpleNamespace(close=lambda: events.append("owner:healthy")),
    }
    monkeypatch.setattr(scene_generation_sessions, "_runtimes", {1: blocked, 2: healthy})
    monkeypatch.setattr(scene_generation_sessions, "_owners", owners)
    monkeypatch.setattr(scene_generation_sessions, "_preparation_pending", {})

    scene_generation_sessions.fail_closed_runtime_reuse(blocked)
    scene_generation_sessions.close()

    assert events == ["close:healthy", "owner:healthy"]
    assert scene_generation_sessions._runtimes == {1: blocked}
    assert scene_generation_sessions._owners == {1: owners[1]}


def test_no_pane_close_stops_runtime_once_on_owner_thread_before_owner(
    monkeypatch,
) -> None:
    caller = threading.get_ident()
    events: list[tuple[str, int]] = []

    class Runtime:
        reuse_blocked = False
        playback_lock = SimpleNamespace(clear=lambda **_kwargs: None)
        lifecycle_status = "open"

        def close(self) -> bool:
            events.append(("runtime", threading.get_ident()))
            self.lifecycle_status = "closed"
            return True

    owner = SimpleNamespace(
        close=lambda: events.append(("owner", threading.get_ident()))
    )
    scene_generation_sessions._stop_preparation_worker()
    monkeypatch.setattr(scene_generation_sessions, "_runtimes", {12: Runtime()})
    monkeypatch.setattr(scene_generation_sessions, "_owners", {12: owner})
    monkeypatch.setattr(scene_generation_sessions, "_preparation_pending", {})
    monkeypatch.setattr(scene_generation_sessions, "_preparation_revisions", {})

    scene_generation_sessions.close()
    scene_generation_sessions.close()

    assert [name for name, _ident in events] == ["runtime", "owner"]
    assert events[0][1] != caller
    assert events[1][1] == caller


def test_physics_playback_reconciles_barrier_before_waking_preparation(monkeypatch) -> None:
    events = []
    scene = SimpleNamespace(session_uid=12)
    generation = SimpleNamespace(number=3)
    owner = SimpleNamespace(current_generation=generation)

    class Runtime:
        def __init__(self, _controller: object, received: object) -> None:
            events.append(("runtime", received))

    monkeypatch.setattr(
        scene_generation_sessions,
        "generation_for_scene",
        lambda received: events.append(("barrier", received)) or generation,
    )
    monkeypatch.setattr(scene_generation_sessions, "_owners", {12: owner})
    monkeypatch.setattr(scene_generation_sessions, "_runtimes", {})
    monkeypatch.setattr(scene_generation_sessions, "AuthoringGenerationRuntime", Runtime)
    monkeypatch.setattr(
        scene_generation_sessions,
        "_wake_preparation",
        lambda uid: events.append(("wake", uid)),
    )

    scene_generation_sessions.demand_physics_playback(scene)

    assert events[0] == ("barrier", scene)
    assert events[-1] == ("wake", 12)

def test_pending_generation_is_published_only_after_runtime_activation(monkeypatch) -> None:
    predecessor = SimpleNamespace(number=1, usd_path="/tmp/previous.usda")
    candidate = SimpleNamespace(number=2, usd_path="/tmp/candidate.usda")
    events = []

    class Owner:
        current_generation = predecessor
        pending_generation = candidate

        def accept(self, generation: object) -> None:
            events.append(("accept", generation))
            self.current_generation = generation
            self.pending_generation = None

        def reject(self, generation: object) -> None:
            events.append(("reject", generation))

    class Runtime:
        viewport_ids: set[str] = set()
        ovrtx = SimpleNamespace(controller=object())

        def attach(self, viewport_id: str, _wake_hook: object = None) -> None:
            self.viewport_ids.add(viewport_id)

        def activate_blocking(
            self, generation: object, _request: object, *, predecessor: object
        ) -> None:
            events.append(("activate", generation, predecessor))

    owner = Owner()
    runtime = Runtime()
    scene = SimpleNamespace(session_uid=12)
    request = RenderRequest(input_usd_path=candidate.usd_path)
    monkeypatch.setattr(scene_generation_sessions, "_owners", {12: owner})
    monkeypatch.setattr(scene_generation_sessions, "_runtimes", {12: runtime})
    monkeypatch.setattr(
        scene_generation_sessions,
        "_dirty",
        {12: {BlenderId("WORLD", 30)}},
    )
    monkeypatch.setattr(
        scene_generation_sessions,
        "_blocked_reconciliations",
        {12: ("RuntimeError: prior failure", {BlenderId("OBJECT", 91)})},
    )

    assert scene_generation_sessions.activate_for_viewport(
        scene,
        request,
        viewport_id="viewport",
        on_generation_settled=lambda: events.append(("wake",)),
    ) is runtime
    assert events == [
        ("activate", candidate, predecessor),
        ("accept", candidate),
        ("wake",),
    ]
    assert owner.current_generation is candidate
    assert 12 not in scene_generation_sessions._blocked_reconciliations


def test_runtime_restores_predecessor_after_candidate_handoff_failure(monkeypatch) -> None:
    events: list[tuple[str, int]] = []

    class RenderAdapter:
        request = None
        last_error = "candidate render failed"

        def __init__(self, _controller: object) -> None:
            pass

        def update_request(self, request: RenderRequest) -> None:
            self.request = request

        def activate(self, generation: object, **_kwargs: object) -> bool:
            events.append(("render", generation.number))
            return generation.number != 2

    class PhysicsAdapter:
        controller = object()
        active_generation = None
        last_error = ""

        def activate(self, generation: object, **_kwargs: object) -> bool:
            self.active_generation = generation
            events.append(("physics", generation.number))
            return True

        def reset(self) -> bool:
            return True

    class Scheduler:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def set_edit_wake_hook(self, _hook: object) -> None:
            pass

        @property
        def has_pending_view_updates(self) -> bool:
            return False

        def pending_view_targets(self) -> tuple[frozenset[str], frozenset[tuple[str, str]]]:
            return frozenset(), frozenset()

    predecessor = SimpleNamespace(number=1, usd_path="/tmp/previous.usda")
    candidate = SimpleNamespace(number=2, usd_path="/tmp/candidate.usda")
    monkeypatch.setattr(scene_generation_sessions, "OvrtxGenerationAdapter", RenderAdapter)
    monkeypatch.setattr(scene_generation_sessions, "OvphysxGenerationAdapter", PhysicsAdapter)
    monkeypatch.setattr(scene_generation_sessions, "RuntimeScheduler", Scheduler)
    monkeypatch.setattr(
        scene_generation_sessions,
        "generation_requires_physics",
        lambda _generation: True,
    )
    runtime = scene_generation_sessions.AuthoringGenerationRuntime(
        object(),
        SimpleNamespace(
            retained_values_for=lambda _generation: ((), (), ()),
            retain_transform_values=lambda _values: None,
            retain_attribute_values=lambda _values: None,
            retain_initial_conditions=lambda _values: None,
        ),
    )
    runtime.activate(
        predecessor,
        RenderRequest(input_usd_path=predecessor.usd_path),
    )

    with pytest.raises(RuntimeError, match="predecessor_restore=succeeded"):
        runtime.activate(
            candidate,
            RenderRequest(input_usd_path=candidate.usd_path),
            predecessor=predecessor,
        )

    assert events == [
        ("physics", 1),
        ("render", 1),
        ("physics", 2),
        ("render", 2),
        ("physics", 1),
        ("render", 1),
    ]
    assert runtime._generation_number == 1


def test_failed_candidate_activation_rejects_candidate_and_keeps_predecessor(monkeypatch) -> None:
    predecessor = SimpleNamespace(number=1, usd_path="/tmp/previous.usda")
    candidate = SimpleNamespace(number=2, usd_path="/tmp/candidate.usda")
    rejected = []

    class Owner:
        current_generation = predecessor
        pending_generation = candidate

        def accept(self, _generation: object) -> None:
            raise AssertionError("failed candidate was accepted")

        def reject(self, generation: object) -> None:
            rejected.append(generation)
            self.pending_generation = None

    class Runtime:
        viewport_ids: set[str] = set()
        ovrtx = SimpleNamespace(controller=object())

        def attach(self, viewport_id: str, _wake_hook: object = None) -> None:
            self.viewport_ids.add(viewport_id)

        def activate_blocking(
            self, _generation: object, _request: object, *, predecessor: object
        ) -> None:
            assert predecessor is not None
            raise RuntimeError("activation failed; predecessor_restore=succeeded")

    owner = Owner()
    scene = SimpleNamespace(session_uid=12)
    monkeypatch.setattr(scene_generation_sessions, "_owners", {12: owner})
    monkeypatch.setattr(scene_generation_sessions, "_runtimes", {12: Runtime()})
    affected = {BlenderId("OBJECT", 91), BlenderId("WORLD", 30)}
    wake = []
    monkeypatch.setattr(scene_generation_sessions, "_dirty", {})
    monkeypatch.setattr(
        scene_generation_sessions,
        "_blocked_reconciliations",
        {12: ("RuntimeError: prior failure", set(affected))},
    )

    with pytest.raises(RuntimeError, match="predecessor_restore=succeeded"):
        scene_generation_sessions.activate_for_viewport(
            scene,
            RenderRequest(input_usd_path=candidate.usd_path),
            viewport_id="viewport",
            on_generation_settled=lambda: wake.append(True),
        )
    assert rejected == [candidate]
    assert owner.current_generation is predecessor
    assert scene_generation_sessions._dirty[12] == affected
    assert wake == [True]


def test_first_candidate_activation_failure_requeues_reconciled_ids(monkeypatch) -> None:
    predecessor = SimpleNamespace(number=1, usd_path="/tmp/previous.usda")
    candidate = SimpleNamespace(number=2, usd_path="/tmp/candidate.usda")
    affected = {BlenderId("OBJECT", 91)}
    concurrent = BlenderId("MESH", 92)

    class Owner:
        current_generation = predecessor
        pending_generation = None

        def reconcile(self, _scene: object, identities: object) -> object:
            assert identities == affected
            self.pending_generation = candidate
            return candidate

        def reject(self, generation: object) -> None:
            assert generation is candidate
            self.pending_generation = None

        def reuse(self) -> object:
            return self.pending_generation or self.current_generation

    class Runtime:
        viewport_ids: set[str] = set()
        ovrtx = SimpleNamespace(controller=object())

        def attach(self, viewport_id: str, _wake_hook: object = None) -> None:
            self.viewport_ids.add(viewport_id)

        def activate_blocking(
            self, _generation: object, _request: object, *, predecessor: object
        ) -> None:
            assert predecessor is not None
            scene_generation_sessions.mark_scene_dirty(scene, {concurrent})
            raise RuntimeError("activation failed; predecessor_restore=succeeded")

    owner = Owner()
    scene = SimpleNamespace(session_uid=12, objects=())
    monkeypatch.setattr(scene_generation_sessions, "_owners", {12: owner})
    monkeypatch.setattr(scene_generation_sessions, "_runtimes", {12: Runtime()})
    monkeypatch.setattr(scene_generation_sessions, "_dirty", {12: set(affected)})
    monkeypatch.setattr(scene_generation_sessions, "_pending_affected", {})
    monkeypatch.setattr(scene_generation_sessions, "_blocked_reconciliations", {})

    assert scene_generation_sessions.generation_for_scene(scene) is candidate
    with pytest.raises(RuntimeError, match="predecessor_restore=succeeded"):
        scene_generation_sessions.activate_for_viewport(
            scene,
            RenderRequest(input_usd_path=candidate.usd_path),
            viewport_id="viewport",
        )

    assert scene_generation_sessions._dirty[12] == affected | {concurrent}
    assert scene_generation_sessions.diagnostics()["blocked_reconciliations"]["12"][
        "affected_ids"
    ] == [
        {"kind": "MESH", "session_uid": 92},
        {"kind": "OBJECT", "session_uid": 91},
    ]


def test_final_runtime_accepts_candidate_only_after_retained_value_replay(monkeypatch) -> None:
    predecessor = SimpleNamespace(number=1, usd_path="/tmp/previous.usda")
    candidate = SimpleNamespace(number=2, usd_path="/tmp/candidate.usda")
    retained = (("transform",), ("attribute",), ())
    events = []

    class Owner:
        current_generation = predecessor
        pending_generation = candidate

        def retained_values_for(self, generation: object) -> object:
            assert generation is candidate
            return retained

        def accept(self, generation: object) -> None:
            events.append(("accept", generation))
            self.current_generation = generation
            self.pending_generation = None

        def reject(self, _generation: object) -> None:
            raise AssertionError("successful candidate was rejected")

    class Adapter:
        last_error = ""

        def __init__(self, _controller: object) -> None:
            pass

        def update_request(self, _request: object) -> None:
            pass

        def activate(self, generation: object, **kwargs: object) -> bool:
            events.append(("activate", generation, kwargs))
            return True

        def deactivate(self) -> str:
            return "stopped"

    owner = Owner()
    monkeypatch.setattr(scene_generation_sessions, "_owners", {12: owner})
    monkeypatch.setattr(scene_generation_sessions, "_runtimes", {})
    monkeypatch.setattr(
        scene_generation_sessions,
        "_pending_affected",
        {12: {BlenderId("OBJECT", 91)}},
    )
    monkeypatch.setattr(
        scene_generation_sessions,
        "_blocked_reconciliations",
        {12: ("RuntimeError: prior failure", {BlenderId("OBJECT", 91)})},
    )
    monkeypatch.setattr(scene_generation_sessions, "OvrtxGenerationAdapter", Adapter)

    adapter = scene_generation_sessions.activate_for_final_render(
        SimpleNamespace(session_uid=12),
        RenderRequest(input_usd_path=candidate.usd_path),
        controller=object(),
    )

    assert isinstance(adapter, Adapter)
    assert events == [
        (
            "activate",
            candidate,
            {
                "transform_values": retained[0],
                "attribute_values": retained[1],
            },
        ),
        ("accept", candidate),
    ]
    assert 12 not in scene_generation_sessions._pending_affected
    assert 12 not in scene_generation_sessions._blocked_reconciliations


def test_final_render_borrows_prepared_session_and_restores_view_request(monkeypatch) -> None:
    generation = SimpleNamespace(number=1, usd_path="/tmp/current.usda")
    viewport_request = RenderRequest(
        input_usd_path=generation.usd_path,
        width=640,
        height=480,
    )
    final_request = RenderRequest(
        input_usd_path=generation.usd_path,
        width=1920,
        height=1080,
    )
    activations = []

    class Runtime:
        _request = viewport_request
        ovrtx = SimpleNamespace(
            controller=object(),
            last_ensure_result=SimpleNamespace(composition=object()),
        )

        def activate(self, value: object, request: RenderRequest, **_kwargs: object) -> None:
            activations.append((value, request))

    runtime = Runtime()
    monkeypatch.setattr(
        scene_generation_sessions,
        "_owners",
        {12: SimpleNamespace(current_generation=generation, pending_generation=None)},
    )
    monkeypatch.setattr(scene_generation_sessions, "_runtimes", {12: runtime})

    lease = scene_generation_sessions.activate_for_final_render(
        SimpleNamespace(session_uid=12),
        final_request,
        controller=object(),
    )

    assert lease.controller is runtime.ovrtx.controller
    assert lease.deactivate() == "stopped"
    assert activations == [
        (generation, final_request),
        (generation, viewport_request),
    ]


def test_failed_final_render_restore_blocks_runtime_reuse() -> None:
    generation = SimpleNamespace(number=1, usd_path="/tmp/current.usda")

    class Runtime:
        reuse_blocked = False

        def activate(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("restore failed")

    runtime = Runtime()
    lease = scene_generation_sessions.PreparedFinalRenderLease(
        runtime,
        generation,
        RenderRequest(input_usd_path=generation.usd_path),
    )

    assert lease.deactivate() == "failed"
    assert runtime.reuse_blocked is True


@pytest.mark.parametrize("status", ("stopped", "not_found", "failed"))
def test_final_render_without_prior_request_stops_or_blocks_runtime(status: str) -> None:
    runtime = SimpleNamespace(
        reuse_blocked=False,
        ovrtx=SimpleNamespace(deactivate=lambda: status),
    )
    lease = scene_generation_sessions.PreparedFinalRenderLease(
        runtime,
        SimpleNamespace(number=1),
        None,
    )

    assert lease.deactivate() == ("failed" if status == "failed" else "stopped")
    assert runtime.reuse_blocked is (status == "failed")


def test_final_runtime_replays_scene_owned_world_dome_value(
    monkeypatch,
    tmp_path: Path,
) -> None:
    world_id = BlenderId("WORLD", 30)
    generation = SimpleNamespace(
        number=1,
        usd_path="/tmp/current.usda",
        blender_prim_paths={
            world_id: BlenderPrimPath(
                "World",
                "WORLD",
                DEFAULT_DOME_OWNER_PATH,
                DEFAULT_DOME_OWNER_PATH,
            )
        },
    )
    owner = SceneGenerationOwner(tmp_path / "generations")
    owner._current = generation  # noqa: SLF001
    world_value = OvrtxAttributeValue(
        DEFAULT_DOME_OWNER_PATH,
        "inputs:intensity",
        1133.8,
        "Float",
    )
    owner.retain_attribute_values((world_value,))
    activations = []

    class Adapter:
        last_error = ""

        def __init__(self, _controller: object) -> None:
            pass

        def update_request(self, _request: object) -> None:
            pass

        def activate(self, _generation: object, **kwargs: object) -> bool:
            activations.append(kwargs)
            return True

        def deactivate(self) -> str:
            return "stopped"

    monkeypatch.setattr(scene_generation_sessions, "_owners", {12: owner})
    monkeypatch.setattr(scene_generation_sessions, "_runtimes", {})
    monkeypatch.setattr(scene_generation_sessions, "OvrtxGenerationAdapter", Adapter)

    scene_generation_sessions.activate_for_final_render(
        SimpleNamespace(session_uid=12),
        RenderRequest(input_usd_path=generation.usd_path),
        controller=object(),
    )

    assert activations[0]["attribute_values"] == (world_value,)


def test_final_runtime_rejects_candidate_when_retained_value_rebinding_fails(
    monkeypatch,
) -> None:
    predecessor = SimpleNamespace(number=1, usd_path="/tmp/previous.usda")
    candidate = SimpleNamespace(number=2, usd_path="/tmp/candidate.usda")
    rejected = []

    class Owner:
        current_generation = predecessor
        pending_generation = candidate

        def retained_values_for(self, _generation: object) -> object:
            raise RuntimeError("retained material target is unavailable")

        def reject(self, generation: object) -> None:
            rejected.append(generation)

    class Adapter:
        def __init__(self, _controller: object) -> None:
            pass

        def update_request(self, _request: object) -> None:
            pass

        def deactivate(self) -> str:
            return "stopped"

    owner = Owner()
    monkeypatch.setattr(scene_generation_sessions, "_owners", {12: owner})
    monkeypatch.setattr(scene_generation_sessions, "_runtimes", {})
    monkeypatch.setattr(
        scene_generation_sessions,
        "_pending_affected",
        {12: {BlenderId("OBJECT", 91)}},
    )
    monkeypatch.setattr(
        scene_generation_sessions,
        "_blocked_reconciliations",
        {12: ("RuntimeError: prior failure", {BlenderId("OBJECT", 91)})},
    )
    monkeypatch.setattr(scene_generation_sessions, "OvrtxGenerationAdapter", Adapter)

    with pytest.raises(RuntimeError, match="retained material target is unavailable"):
        scene_generation_sessions.activate_for_final_render(
            SimpleNamespace(session_uid=12),
            RenderRequest(input_usd_path=candidate.usd_path),
            controller=object(),
        )

    assert rejected == [candidate]
    assert owner.current_generation is predecessor
    assert 12 not in scene_generation_sessions._pending_affected
    assert 12 not in scene_generation_sessions._blocked_reconciliations


def test_dirty_edit_arriving_while_candidate_is_pending_waits_for_next_reconcile(monkeypatch) -> None:
    predecessor = SimpleNamespace(number=1)
    candidate = SimpleNamespace(number=2)

    class Owner:
        current_generation = predecessor
        pending_generation = candidate

        def reuse(self) -> object:
            return self.pending_generation or self.current_generation

        def reconcile(self, _scene: object, _affected: object) -> object:
            raise AssertionError("pending candidate was bypassed")

    affected = {BlenderId("MESH", 91)}
    scene = SimpleNamespace(session_uid=12)
    monkeypatch.setattr(scene_generation_sessions, "_owners", {12: Owner()})
    monkeypatch.setattr(scene_generation_sessions, "_dirty", {12: set(affected)})

    assert scene_generation_sessions.generation_for_scene(scene) is candidate
    assert scene_generation_sessions._dirty[12] == affected


def test_failed_reconcile_retains_affected_and_concurrent_dirty_ids(monkeypatch) -> None:
    first = BlenderId("OBJECT", 91)
    concurrent = BlenderId("MESH", 92)
    scene = SimpleNamespace(session_uid=12, objects=())

    class Owner:
        current_generation = SimpleNamespace(number=0, blender_prim_paths={})
        pending_generation = None

        def reconcile(self, current_scene: object, affected: object) -> object:
            assert affected == {first}
            scene_generation_sessions.mark_scene_dirty(current_scene, {concurrent})
            raise RuntimeError("candidate failed")

        def reuse(self) -> object:
            return self.current_generation

    monkeypatch.setattr(scene_generation_sessions, "_owners", {12: Owner()})
    monkeypatch.setattr(scene_generation_sessions, "_dirty", {12: {first}})
    monkeypatch.setattr(scene_generation_sessions, "_blocked_reconciliations", {})

    with pytest.raises(RuntimeError, match="candidate failed"):
        scene_generation_sessions.generation_for_scene(scene)

    assert scene_generation_sessions._dirty[12] == {first, concurrent}
    assert scene_generation_sessions.diagnostics()["blocked_reconciliations"] == {
        "12": {
            "error": "RuntimeError: candidate failed",
            "affected_ids": [
                {"kind": "MESH", "session_uid": 92},
                {"kind": "OBJECT", "session_uid": 91},
            ],
        }
    }


def test_undo_unaccepted_addition_clears_blocked_reconciliation(monkeypatch) -> None:
    addition = BlenderId("OBJECT", 91)
    generation = SimpleNamespace(number=0, blender_prim_paths={})

    class Owner:
        current_generation = generation
        pending_generation = None

        def reconcile(self, _scene: object, _affected: object) -> object:
            raise AssertionError("undone addition was reconciled")

        def reuse(self) -> object:
            return generation

    scene = SimpleNamespace(session_uid=12, objects=())
    monkeypatch.setattr(scene_generation_sessions, "_owners", {12: Owner()})
    monkeypatch.setattr(scene_generation_sessions, "_dirty", {12: {addition}})
    monkeypatch.setattr(
        scene_generation_sessions,
        "_blocked_reconciliations",
        {12: ("RuntimeError: candidate failed", {addition})},
    )

    assert scene_generation_sessions.generation_for_scene(scene) is generation
    assert 12 not in scene_generation_sessions._dirty
    assert 12 not in scene_generation_sessions._blocked_reconciliations


def test_corrected_edit_retains_mapped_object_identity_after_block(monkeypatch) -> None:
    identity = BlenderId("OBJECT", 91)
    current_object = SimpleNamespace(
        session_uid=91,
        type="MESH",
        data=SimpleNamespace(materials=()),
        library=None,
        override_library=None,
    )
    predecessor = SimpleNamespace(
        number=0,
        blender_prim_paths={identity: SimpleNamespace()},
    )
    candidate = SimpleNamespace(number=1)

    class Owner:
        current_generation = predecessor
        pending_generation = None

        def reconcile(self, _scene: object, affected: object) -> object:
            assert affected == {identity}
            return candidate

        def reuse(self) -> object:
            return predecessor

    scene = SimpleNamespace(session_uid=12, objects=(current_object,))
    monkeypatch.setattr(scene_generation_sessions, "_owners", {12: Owner()})
    monkeypatch.setattr(scene_generation_sessions, "_dirty", {12: {identity}})
    monkeypatch.setattr(
        scene_generation_sessions,
        "_blocked_reconciliations",
        {12: ("RuntimeError: candidate failed", {identity})},
    )

    assert scene_generation_sessions.generation_for_scene(scene) is candidate


def test_diagnostics_resolve_the_current_generation(monkeypatch) -> None:
    generation = SimpleNamespace(
        number=1,
        digest="current",
        predecessor_number=0,
        usd_path="/tmp/current.usdc",
        diagnostics={},
    )
    monkeypatch.setattr(
        scene_generation_sessions,
        "_owners",
        {12: SimpleNamespace(current_generation=generation, pending_generation=None)},
    )

    diagnostics = scene_generation_sessions.diagnostics_for_scene(
        SimpleNamespace(session_uid=12),
        input_usd_path=generation.usd_path,
    )

    assert diagnostics["status"] == "current"
    assert diagnostics["usd_path"] == generation.usd_path


def test_deactivate_all_ovrtx_preserves_scheduler_and_ovphysx(monkeypatch) -> None:
    uid = 79
    scheduler = object()
    ovphysx = object()
    controller = SimpleNamespace(adopt_owning_thread=lambda: None)
    ovrtx = SimpleNamespace(controller=controller, deactivate=lambda: "stopped")
    runtime = SimpleNamespace(
        ovrtx=ovrtx,
        ovphysx=ovphysx,
        scheduler=scheduler,
        _generation_number=3,
        reuse_blocked=False,
    )
    monkeypatch.setattr(scene_generation_sessions, "_runtimes", {uid: runtime})

    assert scene_generation_sessions.deactivate_all_ovrtx() is True
    assert runtime.scheduler is scheduler
    assert runtime.ovphysx is ovphysx
    assert runtime._generation_number == 3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Physics RNA observation stays on Blender's main thread (OMPE-115792 n-02-03).

``capture`` and ``scene_authoring_observation`` refuse other threads unless
``render_init_capture`` is active on that thread. ``_authored`` and
``RenderOnlyDiscovery`` carry no guard of their own; these tests prove every
production caller reaches a guard first, so a worker thread fails closed before
it walks Objects. The static inventory makes any new caller fail until its
thread has been reviewed.
"""
from __future__ import annotations

import ast
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

ADDON = Path(__file__).resolve().parents[1] / 'addon'
sys.path.insert(0, str(ADDON))
from ovrtx_blender_example import engine  # noqa: E402
from ovrtx_blender_example import physics_edit_snapshot as snapshots  # noqa: E402
from ovrtx_blender_example import scene_generation_sessions as sessions  # noqa: E402

PACKAGE = ADDON / 'ovrtx_blender_example'
UID = 77


def off_main(function):
    """Run ``function`` on a plain worker thread; return (result, exception)."""
    outcome = {}

    def work():
        try:
            outcome['value'] = function()
        except BaseException as exc:  # noqa: BLE001 - asserted by the caller
            outcome['error'] = exc
    thread = threading.Thread(target=work, name='n-02-03-worker')
    thread.start()
    thread.join(5)
    assert not thread.is_alive(), 'worker did not finish'
    return outcome.get('value'), outcome.get('error')


class ForbiddenObjects:
    def __iter__(self):
        pytest.fail(f'{threading.current_thread().name} walked scene.objects')

    def __len__(self):
        pytest.fail(f'{threading.current_thread().name} sized scene.objects')


def make_scene(objects=()):
    return SimpleNamespace(
        session_uid=UID, ov=SimpleNamespace(bl_rna=SimpleNamespace(properties=())),
        unit_settings=SimpleNamespace(scale_length=1.0), objects=objects)


@pytest.fixture
def observed(monkeypatch):
    """A physics-free accepted generation: the state the viewport reuse path trusts."""
    for name in ('_owners', '_dirty', '_physics_snapshots', '_physics_view_observations',
                 '_physics_initial_bases', '_physics_restart_pending', '_physics_idle_discovery',
                 '_runtimes'):
        monkeypatch.setattr(sessions, name, {})
    monkeypatch.setattr(sessions, '_physics_authoring_dirty', set())
    monkeypatch.setattr(sessions, '_baked_playback_paused', set())
    monkeypatch.setattr(sessions, '_bake_capture_owner', None)
    monkeypatch.setattr(sessions, 'bpy', None)
    scene = make_scene()
    generation = SimpleNamespace(number=1)
    sessions._owners[UID] = SimpleNamespace(
        current_generation=generation, pending_generation=None, reuse=lambda: generation)
    sessions._remember_physics_snapshot(UID, snapshots.capture(scene))
    assert sessions._physics_view_observations[UID][1] is False
    return scene, generation


@pytest.fixture
def unguarded_spies(monkeypatch):
    """Record the thread of every ``_authored`` / ``RenderOnlyDiscovery`` use."""
    calls = []

    def authored(group):
        calls.append(('_authored', threading.current_thread() is threading.main_thread()))
        return False

    class Discovery:
        def __init__(self, scene):
            calls.append(('RenderOnlyDiscovery', threading.current_thread() is threading.main_thread()))

        def changed(self, budget=128):
            calls.append(('RenderOnlyDiscovery.changed',
                          threading.current_thread() is threading.main_thread()))
            return False
    monkeypatch.setattr(snapshots, '_authored', authored)
    monkeypatch.setattr(snapshots, 'RenderOnlyDiscovery', Discovery)
    return calls


def update(kind, *, geometry=False, transform=False, shading=True, obj_type='MESH'):
    source = SimpleNamespace(bl_rna=SimpleNamespace(identifier=kind), type=obj_type,
                             ov=SimpleNamespace(physics=object()), keys=lambda: ())
    return SimpleNamespace(id=source, is_updated_geometry=geometry,
                           is_updated_transform=transform, is_updated_shading=shading)


CAMERA_ONLY = SimpleNamespace(updates=(SimpleNamespace(id=SimpleNamespace(type='CAMERA')),))
SHADER_ONLY = SimpleNamespace(updates=(update('Material'),))


@pytest.mark.parametrize('observe', [snapshots.capture, snapshots.scene_authoring_observation],
                         ids=['capture', 'scene_authoring_observation'])
def test_direct_observations_refuse_worker_threads(observe):
    scene = make_scene(ForbiddenObjects())
    _, error = off_main(lambda: observe(scene))
    assert isinstance(error, RuntimeError) and "main thread" in str(error)
    assert observe(make_scene()).leaves  # The main thread is admitted.


def test_render_init_capture_admits_only_the_thread_that_entered_it():
    scene = make_scene()

    def inside_and_after():
        with snapshots.render_init_capture():
            inside = snapshots.capture(scene)
        try:
            snapshots.scene_authoring_observation(scene)
        except RuntimeError:
            return inside
        pytest.fail('render-init admission outlived its context')
    value, error = off_main(inside_and_after)
    assert error is None and value.leaves
    # The flag is thread-local: the main thread entering it admits no worker.
    with snapshots.render_init_capture():
        _, error = off_main(lambda: snapshots.capture(make_scene(ForbiddenObjects())))
    assert isinstance(error, RuntimeError)


ENTRIES = {
    # view_update / view_draw -> _translate_scene_generation -> provider
    'generation_for_viewport': lambda scene: sessions.generation_for_viewport(scene),
    # depsgraph_update_post camera-only skip and its predicate
    'observe_physics_depsgraph_camera': lambda scene: sessions.observe_physics_depsgraph(scene, CAMERA_ONLY),
    '_render_only_physics_current': lambda scene: sessions._render_only_physics_current(scene),
    # depsgraph_update_post shader-only fast path (calls _authored per update)
    'observe_depsgraph_physics_edits_shader': (
        lambda scene: sessions.observe_depsgraph_physics_edits(scene, SHADER_ONLY)),
    'observe_physics_edits': lambda scene: sessions.observe_physics_edits(scene),
    # bpy.app.timers 0.2 s poll (creates RenderOnlyDiscovery)
    '_poll_physics_edits': lambda scene: sessions._poll_physics_edits(),
    # suppress_interactive_edit_bridge() exit
    'absorb_physics_presentation': lambda scene: sessions.absorb_physics_presentation(),
}


@pytest.mark.parametrize('entry', ENTRIES, ids=list(ENTRIES))
def test_viewport_reuse_and_handler_entries_fail_closed_off_main_thread(
        observed, unguarded_spies, monkeypatch, entry):
    scene, _generation = observed
    scene.objects = ForbiddenObjects()
    monkeypatch.setattr(sessions, 'bpy', SimpleNamespace(data=SimpleNamespace(scenes=[scene])))
    monkeypatch.setattr(sessions, 'repair_duplicate_identities',
                        lambda *a, **k: pytest.fail('poll repaired identities off the main thread'))
    _, error = off_main(lambda: ENTRIES[entry](scene))
    assert isinstance(error, RuntimeError) and 'main thread' in str(error), error
    assert unguarded_spies == [], 'unguarded RNA helpers ran before the main-thread guard'
    assert UID not in sessions._physics_idle_discovery


def test_unguarded_helpers_are_live_on_the_main_thread(observed, unguarded_spies, monkeypatch):
    """Non-vacuity: the worker test above patches the paths that really run."""
    scene, generation = observed
    monkeypatch.setattr(sessions, 'bpy', SimpleNamespace(data=SimpleNamespace(scenes=[scene])))
    assert sessions.observe_depsgraph_physics_edits(scene, SHADER_ONLY) == snapshots.PhysicsChanges()
    assert sessions._poll_physics_edits() == 0.2
    assert sessions.observe_physics_depsgraph(scene, CAMERA_ONLY) == snapshots.PhysicsChanges()
    monkeypatch.setattr(sessions, 'bpy', None)
    assert sessions.generation_for_viewport(scene) is generation
    assert unguarded_spies == [('_authored', True), ('RenderOnlyDiscovery', True),
                               ('RenderOnlyDiscovery.changed', True)]


@pytest.mark.parametrize('thread', ['main', 'worker'])
def test_note_physics_authoring_changed_only_latches_scene_uids(monkeypatch, thread):
    reads = []

    class Scene:
        def __init__(self, uid):
            object.__setattr__(self, '_uid', uid)

        def __getattribute__(self, name):
            if name.startswith('_') and name != '__class__':
                return object.__getattribute__(self, name)
            reads.append(name)
            if name == 'session_uid':
                return object.__getattribute__(self, '_uid')
            pytest.fail(f'note_physics_authoring_changed read Scene.{name}')
    monkeypatch.setattr(sessions, '_physics_authoring_dirty', set())
    monkeypatch.setattr(sessions, 'bpy', SimpleNamespace(data=SimpleNamespace(scenes=[Scene(5), Scene(6)])))
    monkeypatch.setattr(snapshots, 'capture', lambda *a: pytest.fail('latch captured'))
    monkeypatch.setattr(snapshots, '_authored', lambda *a: pytest.fail('latch inspected authoring'))
    monkeypatch.setattr(sessions, 'observe_physics_edits', lambda *a, **k: pytest.fail('latch observed'))
    note = lambda: sessions.note_physics_authoring_changed(object())  # noqa: E731
    if thread == 'main':
        note()
    else:
        assert off_main(note)[1] is None
    assert reads == ['session_uid', 'session_uid']
    assert sessions._physics_authoring_dirty == {5, 6}


def test_final_render_job_marshals_observation_to_main_thread(monkeypatch):
    """render() runs on the job thread; its preparation runs from a main-loop timer."""
    queued = []
    monkeypatch.setattr(engine, 'bpy', SimpleNamespace(app=SimpleNamespace(
        timers=SimpleNamespace(register=lambda fn, first_interval: queued.append(fn)))))
    threads = []

    def prepare():
        threads.append(threading.current_thread() is threading.main_thread())
        return snapshots.capture(make_scene())
    worker_done = {}

    def job():
        worker_done['value'] = engine._run_on_main_thread(prepare, timeout=5)
    thread = threading.Thread(target=job)
    thread.start()
    while thread.is_alive() and not queued:
        thread.join(0.01)
    assert len(queued) == 1
    queued.pop()()  # Blender's main event loop services the timer.
    thread.join(5)
    assert not thread.is_alive() and threads == [True]
    assert worker_done['value'].leaves


def test_render_init_handler_on_job_thread_prepares_nothing(monkeypatch):
    monkeypatch.setattr(engine, '_FINAL_RENDER_ACTIVE', False)
    monkeypatch.setattr(engine, '_render_callback_adapter',
                        lambda *a: pytest.fail('job-thread render_init translated the scene'))
    monkeypatch.setattr(snapshots, 'capture', lambda *a: pytest.fail('job-thread render_init captured'))
    scene = SimpleNamespace(render=SimpleNamespace(engine=engine.ENGINE_ID))
    assert off_main(lambda: engine._final_render_init_handler(scene))[1] is None
    assert engine._FINAL_RENDER_REQUEST is None


class ForbiddenDepsgraph:
    @property
    def updates(self):
        pytest.fail(f'{threading.current_thread().name} read depsgraph.updates')


def test_frame_change_handler_off_main_latches_without_reading_rna(observed, monkeypatch, unguarded_spies):
    scene, _ = observed
    scene.objects = ForbiddenObjects()
    monkeypatch.setattr(snapshots, 'object_participates',
                        lambda *a: pytest.fail('frame handler probed an Object off the main thread'))
    monkeypatch.setattr(snapshots, 'animation_targets_physics',
                        lambda *a: pytest.fail('frame handler read AnimData off the main thread'))
    monkeypatch.setattr(sessions, '_render_only_physics_current',
                        lambda *a: pytest.fail('frame handler observed off the main thread'))
    _, error = off_main(lambda: sessions.physics_frame_changed(scene, ForbiddenDepsgraph()))
    assert error is None
    assert UID in sessions._physics_authoring_dirty
    assert unguarded_spies == []


def _animated(participates):
    obj = SimpleNamespace(bl_rna=SimpleNamespace(identifier='Object'), animation_data=object(),
                          participates=participates)
    obj.original = obj
    return SimpleNamespace(id=obj)


def test_frame_change_handler_latches_only_animated_participants_on_main(observed, monkeypatch):
    scene, _ = observed
    probed = []
    monkeypatch.setattr(snapshots, 'object_participates',
                        lambda obj, cache=None: probed.append(obj) or obj.participates)
    monkeypatch.setattr(snapshots, 'animation_targets_physics',
                        lambda obj: getattr(obj, 'animation_data', None) is not None)
    still = SimpleNamespace(bl_rna=SimpleNamespace(identifier='Object'), animation_data=None)
    graph = SimpleNamespace(updates=(SimpleNamespace(id=still), _animated(False)))
    sessions.physics_frame_changed(scene, graph)
    assert len(probed) == 1 and UID not in sessions._physics_authoring_dirty
    graph = SimpleNamespace(updates=(_animated(False), _animated(True)))
    sessions.physics_frame_changed(scene, graph)
    assert UID in sessions._physics_authoring_dirty
    # Latched (not render-only current): the next frame reads no updates at all.
    sessions.physics_frame_changed(scene, ForbiddenDepsgraph())


# --- Static caller inventory -------------------------------------------------

SNAPSHOT_TARGETS = {'capture', 'scene_authoring_observation', '_authored',
                    'RenderOnlyDiscovery', 'render_init_capture', 'object_participates',
                    'animation_targets_physics'}
SESSION_TARGETS = {'note_physics_authoring_changed', 'observe_physics_edits',
                   'observe_physics_depsgraph', 'observe_depsgraph_physics_edits',
                   '_render_only_physics_current', 'absorb_physics_presentation'}

#: (module, enclosing function, callee). Thread of each caller is recorded in
#: docs/graphs/ompe-115792-physics-scan-overhead/n-02-03-main-thread.md (root repo).
EXPECTED_CALLERS = {
    # physics_edit_snapshot: guarded entry points and their internal use.
    ('physics_edit_snapshot', '_authored', '_authored'),
    ('physics_edit_snapshot', 'capture', 'scene_authoring_observation'),
    ('physics_edit_snapshot', 'capture', '_authored'),
    ('physics_edit_snapshot', '_owner_authored', '_authored'),
    ('physics_edit_snapshot', 'RenderOnlyDiscovery.changed', 'object_participates'),
    # Full observation.
    ('scene_generation_sessions', '_observe_physics_edits', 'capture'),
    ('scene_generation_sessions', '_generation_for_scene', 'capture'),
    ('scene_generation_sessions', '_generation_for_scene', 'observe_physics_edits'),
    ('scene_generation_sessions', 'generation_for_viewport', 'scene_authoring_observation'),
    ('scene_generation_sessions', 'generation_for_viewport', 'observe_physics_edits'),
    ('scene_generation_sessions', 'observe_depsgraph_physics_edits', 'scene_authoring_observation'),
    ('scene_generation_sessions', 'observe_depsgraph_physics_edits', '_authored'),
    ('scene_generation_sessions', 'observe_depsgraph_physics_edits', 'observe_physics_edits'),
    ('scene_generation_sessions', '_render_only_physics_current', 'scene_authoring_observation'),
    ('scene_generation_sessions', 'observe_physics_depsgraph', '_render_only_physics_current'),
    ('scene_generation_sessions', 'observe_physics_depsgraph', 'observe_depsgraph_physics_edits'),
    ('scene_generation_sessions', '_poll_physics_edits', '_render_only_physics_current'),
    ('scene_generation_sessions', '_poll_physics_edits', 'RenderOnlyDiscovery'),
    ('scene_generation_sessions', '_poll_physics_edits', 'observe_physics_edits'),
    ('scene_generation_sessions', 'absorb_physics_presentation', 'observe_physics_edits'),
    # frame_change_post (n-02-02): main thread probes only animated Objects of a
    # physics-free scene after the guarded _render_only_physics_current; off main
    # (animation render job thread) it latches without reading RNA.
    ('scene_generation_sessions', 'physics_frame_changed', '_render_only_physics_current'),
    ('scene_generation_sessions', 'physics_frame_changed', 'animation_targets_physics'),
    ('scene_generation_sessions', 'physics_frame_changed', 'object_participates'),
    ('scene_generation_sessions', 'physics_frame_changed', 'note_physics_authoring_changed'),
    ('interactive_operator_state', 'suppress_interactive_edit_bridge', 'absorb_physics_presentation'),
    ('engine', '_live_interactive_edit_depsgraph_handler', 'observe_physics_depsgraph'),
    # Bake capture (each caller first asserts _main_thread()).
    ('physics_bake_capture', '_source_stamp', 'capture'),
    ('physics_bake_capture', '_frozen_configuration', 'capture'),
    # Latch only.
    ('authoring_properties', '_physics_property_updated', 'note_physics_authoring_changed'),
    ('physics_authoring', 'mark_dirty', 'note_physics_authoring_changed'),
    ('physics_endpoints', 'initialize_joint', 'note_physics_authoring_changed'),
    ('engine', '_physics_authoring_history_restored', 'note_physics_authoring_changed'),
    # depsgraph_update_post while kinematic sampling skips observation (n-02-02, R3).
    ('engine', '_live_interactive_edit_depsgraph_handler', 'note_physics_authoring_changed'),
}


def _imports(node, wanted):
    names = set()
    for child in ast.walk(node):
        if isinstance(child, ast.ImportFrom) and child.level == 1 and child.module in wanted:
            names |= {a.asname or a.name for a in child.names if a.name in wanted[child.module]}
    return names


class _CallInventory(ast.NodeVisitor):
    WANTED = {'physics_edit_snapshot': SNAPSHOT_TARGETS, 'scene_generation_sessions': SESSION_TARGETS}
    MODULE_ALIASES = {'snapshots': SNAPSHOT_TARGETS, 'physics_edit_snapshot': SNAPSHOT_TARGETS,
                      'sessions': SESSION_TARGETS, 'scene_generation_sessions': SESSION_TARGETS}

    def __init__(self, module, own):
        self.module, self.scope, self.found = module, [], set()
        self.names = [set(own)]

    def _enter(self, node):
        self.scope.append(node.name)
        self.names.append(self.names[-1] | _imports(node, self.WANTED))
        self.generic_visit(node)
        self.names.pop()
        self.scope.pop()

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _enter

    def visit_Call(self, node):
        func, where = node.func, '.'.join(self.scope) or '<module>'
        if isinstance(func, ast.Name) and func.id in self.names[-1]:
            self.found.add((self.module, where, func.id))
        elif (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
              and func.attr in self.MODULE_ALIASES.get(func.value.id, ())):
            self.found.add((self.module, where, func.attr))
        self.generic_visit(node)


def _callers():
    found = set()
    for path in sorted(PACKAGE.glob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        own = {'physics_edit_snapshot': SNAPSHOT_TARGETS,
               'scene_generation_sessions': SESSION_TARGETS}.get(path.stem, set())
        inventory = _CallInventory(path.stem, own | _imports_module_level(tree))
        inventory.visit(tree)
        found |= inventory.found
    return found


def _imports_module_level(tree):
    names = set()
    for child in tree.body:
        if isinstance(child, ast.ImportFrom) and child.level == 1 and child.module in _CallInventory.WANTED:
            names |= {a.asname or a.name for a in child.names
                      if a.name in _CallInventory.WANTED[child.module]}
    return names


def test_observation_caller_inventory_matches_reviewed_thread_table():
    assert _callers() == EXPECTED_CALLERS


def test_render_init_capture_has_no_production_entry():
    """The sanctioned off-thread admission exists, but no add-on code enters it."""
    assert not {c for c in _callers() if c[2] == 'render_init_capture'}


def test_real_blender_observation_refuses_worker_threads():
    import subprocess
    from blender_test_support import blender_executable
    blender = blender_executable()
    if blender is None:
        pytest.skip('set BLENDER_COMMAND to run the real-RNA main-thread check')
    fixture = Path(__file__).with_name('blender_physics_observation_main_thread.py')
    result = subprocess.run(
        [str(blender), '--background', '--factory-startup', '--python-exit-code', '17',
         '--python', str(fixture)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=180)
    assert result.returncode == 0, result.stdout
    assert 'PHYSICS_OBSERVATION_MAIN_THREAD_OK' in result.stdout, result.stdout

from pathlib import Path
from types import SimpleNamespace
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'addon'))
from ovrtx_blender_example import scene_generation_sessions as sessions
from ovrtx_blender_example import physics_edit_snapshot as snapshots


@pytest.fixture
def observed(monkeypatch):
    for name in ('_owners','_dirty','_physics_snapshots','_physics_view_observations',
                 '_physics_initial_bases','_physics_restart_pending'):
        monkeypatch.setattr(sessions,name,{})
    monkeypatch.setattr(sessions,'_physics_authoring_dirty',set())
    monkeypatch.setattr(sessions,'_baked_playback_paused',set())
    monkeypatch.setattr(sessions,'_bake_capture_owner',None)
    monkeypatch.setattr(sessions,'bpy',None)
    scene=SimpleNamespace(session_uid=77,ov=SimpleNamespace(bl_rna=SimpleNamespace(properties=())),
                          unit_settings=SimpleNamespace(scale_length=1.0),objects=())
    generation=SimpleNamespace(number=1)
    owner=SimpleNamespace(current_generation=generation,pending_generation=None,reuse=lambda:generation)
    sessions._owners[77]=owner
    sessions._remember_physics_snapshot(77,snapshots.scene_authoring_observation(scene))
    return scene,generation


def test_no_physics_view_reuse_never_walks_objects_or_observes_full_scene(observed,monkeypatch):
    scene,generation=observed
    class ForbiddenObjects:
        def __iter__(self):pytest.fail('Pure view reuse walked objects')
    scene.objects=ForbiddenObjects()
    monkeypatch.setattr(sessions,'observe_physics_edits',lambda *a,**k:pytest.fail('Pure view reuse captured scene'))
    for _ in range(3):assert sessions.generation_for_viewport(scene) is generation


def test_latched_explicit_default_authoring_observed_once_before_reuse(observed,monkeypatch):
    scene,generation=observed;calls=[]
    original=sessions.observe_physics_edits
    monkeypatch.setattr(sessions,'observe_physics_edits',lambda *a,**k:(calls.append(True),original(*a,**k))[1])
    sessions.note_physics_authoring_changed(scene)
    assert sessions.generation_for_viewport(scene) is generation
    assert sessions.generation_for_viewport(scene) is generation
    assert calls==[True] and 77 not in sessions._physics_authoring_dirty


def test_unit_scale_and_unlatched_scene_reset_observed_on_view_demand(observed):
    scene,generation=observed
    scene.unit_settings.scale_length=2
    assert sessions.generation_for_viewport(scene) is generation
    assert sessions._physics_snapshots[77].leaves[0][-1]==2
    assert 77 in sessions._physics_restart_pending
    scene.unit_settings.scale_length=1
    assert sessions.generation_for_viewport(scene) is generation
    assert sessions._physics_snapshots[77].leaves[0][-1]==1


@pytest.mark.parametrize('kind',['OBJECT','MATERIAL'])
def test_authored_participants_keep_conservative_capture_for_unset_without_callback(observed,monkeypatch,kind):
    scene,generation=observed;calls=[]
    previous=snapshots.PhysicsSnapshot(snapshots.scene_authoring_observation(scene).leaves+
                                      ((kind,91,'','physics.mass_kg',True,1.),))
    sessions._remember_physics_snapshot(77,previous)
    def capture(received):calls.append(received);return snapshots.scene_authoring_observation(received)
    monkeypatch.setattr(snapshots,'capture',capture)
    assert sessions.generation_for_viewport(scene) is generation
    assert calls==[scene] and 77 in sessions._physics_restart_pending
    assert not sessions._physics_view_observations[77][1]
    assert sessions.generation_for_viewport(scene) is generation
    assert calls==[scene]


def test_shared_property_notification_invalidates_all_scenes_without_object_scan(monkeypatch):
    class Scene:
        session_uid=77
        @property
        def objects(self):pytest.fail('Notification traversed scene objects')
    monkeypatch.setattr(sessions,'_physics_authoring_dirty',set())
    monkeypatch.setattr(sessions,'bpy',SimpleNamespace(data=SimpleNamespace(scenes=[Scene(),SimpleNamespace(session_uid=88)])))
    sessions.note_physics_authoring_changed(SimpleNamespace(session_uid=3))
    assert sessions._physics_authoring_dirty=={77,88}


def test_failed_capture_retains_dirty_notification_for_retry(observed,monkeypatch):
    scene,_=observed;sessions.note_physics_authoring_changed(scene)
    monkeypatch.setattr(snapshots,'capture',lambda *a:(_ for _ in ()).throw(ValueError('invalid authoring')))
    with pytest.raises(ValueError):sessions.generation_for_viewport(scene)
    assert 77 in sessions._physics_authoring_dirty


def test_shader_drag_reuses_physics_observation_without_walking_scene(observed, monkeypatch):
    scene,_ = observed
    class Objects:
        def __iter__(self): pytest.fail('Shader drag walked scene physics')
    scene.objects=Objects()
    graph=SimpleNamespace(updates=[SimpleNamespace(
        id=SimpleNamespace(bl_rna=SimpleNamespace(identifier=kind),type='MESH'),
        is_updated_geometry=False,is_updated_transform=False,is_updated_shading=True)
        for kind in ('Object','Material','Mesh','ShaderNodeTree')])
    assert sessions.observe_depsgraph_physics_edits(scene,graph)==snapshots.PhysicsChanges()


@pytest.mark.parametrize('change',['geometry','transform','unknown','authored','dirty','participant','unit_scale'])
def test_shader_physics_fast_path_fails_closed_for_other_changes(observed,monkeypatch,change):
    scene,_=observed
    source=SimpleNamespace(bl_rna=SimpleNamespace(identifier='Material'))
    update=SimpleNamespace(id=source,is_updated_geometry=False,is_updated_transform=False,is_updated_shading=True)
    if change=='geometry': update.is_updated_geometry=True
    if change=='transform': update.is_updated_transform=True
    if change=='unknown': source.bl_rna.identifier='Scene'
    if change=='authored': source.ov=SimpleNamespace(physics=object());monkeypatch.setattr(snapshots,'_authored',lambda g:g is not None)
    if change=='dirty': sessions._physics_authoring_dirty.add(77)
    if change=='participant': sessions._physics_view_observations[77]=(sessions._physics_view_observations[77][0],True)
    if change=='unit_scale': scene.unit_settings.scale_length=2
    calls=[]
    monkeypatch.setattr(sessions,'observe_physics_edits',lambda received:calls.append(received))
    sessions.observe_depsgraph_physics_edits(scene,SimpleNamespace(updates=[update]))
    assert calls==[scene]


def test_undo_redo_hooks_latch_only_and_register_idempotently(monkeypatch):
    from ovrtx_blender_example import engine
    calls=[];hooks=SimpleNamespace(undo_post=[],redo_post=[],frame_change_post=[])
    monkeypatch.setattr(engine,'bpy',SimpleNamespace(app=SimpleNamespace(handlers=hooks)))
    monkeypatch.setattr(engine.operator_state,'register_interactive_edit_bridge',lambda *a:False)
    monkeypatch.setattr(engine.operator_state,'unregister_interactive_edit_bridge',lambda *a:False)
    monkeypatch.setattr(sessions,'note_physics_authoring_changed',lambda *a:calls.append(True))
    assert engine.register_interactive_edit_bridge()
    assert not engine.register_interactive_edit_bridge()
    assert len(hooks.undo_post)==len(hooks.redo_post)==1
    hooks.undo_post[0](None);hooks.redo_post[0](None)
    assert calls==[True,True]
    assert engine.unregister_interactive_edit_bridge()
    assert not hooks.undo_post and not hooks.redo_post


# --- n-02-02 repairs (OMPE-115792) -------------------------------------------

def test_empty_depsgraph_batch_is_not_camera_only(observed, monkeypatch):
    """GAP-R5: an empty update list proves nothing; observe like the shader fast path."""
    scene, _ = observed
    calls = []
    monkeypatch.setattr(sessions, 'observe_physics_edits', lambda received: calls.append(received))
    assert sessions.observe_physics_depsgraph(scene, SimpleNamespace(updates=())) is None
    assert calls == [scene]
    calls.clear()
    camera = SimpleNamespace(updates=(SimpleNamespace(id=SimpleNamespace(type='CAMERA')),))
    assert sessions.observe_physics_depsgraph(scene, camera) == snapshots.PhysicsChanges()
    assert calls == []


def test_suppressed_observation_keeps_authoring_latch(observed, monkeypatch):
    """GAP-NOTIFY: suppression absorbs transforms only; a pending authoring diff stays latched."""
    from ovrtx_blender_example import interactive_operator_state
    scene, generation = observed
    base = snapshots.scene_authoring_observation(scene).leaves
    authored = snapshots.PhysicsSnapshot(base + (('OBJECT', 91, '', 'physics.colliders.__count__', False, 1),))
    monkeypatch.setattr(snapshots, 'capture', lambda received: authored)
    sessions.note_physics_authoring_changed(scene)  # physics_authoring.mark_dirty
    monkeypatch.setattr(interactive_operator_state, 'interactive_edit_bridge_suppressed', lambda: True)
    assert sessions.observe_physics_edits(scene) == snapshots.PhysicsChanges()  # reconcile timer
    assert 77 in sessions._physics_authoring_dirty
    assert not sessions.viewport_joint_admission_clear(scene)
    monkeypatch.setattr(interactive_operator_state, 'interactive_edit_bridge_suppressed', lambda: False)
    assert sessions.generation_for_viewport(scene) is generation  # the redraw observes
    assert sessions._physics_snapshots[77] == authored
    assert 77 in sessions._physics_restart_pending and 77 not in sessions._physics_authoring_dirty


def test_suppressed_transform_only_observation_clears_latch(observed, monkeypatch):
    from ovrtx_blender_example import interactive_operator_state
    scene, _ = observed
    sessions.note_physics_authoring_changed(scene)
    monkeypatch.setattr(interactive_operator_state, 'interactive_edit_bridge_suppressed', lambda: True)
    sessions.observe_physics_edits(scene)
    assert 77 not in sessions._physics_authoring_dirty


def _history_blender(scene):
    timers = []
    return SimpleNamespace(context=SimpleNamespace(scene=scene, window=None),
                           app=SimpleNamespace(timers=SimpleNamespace(
                               register=lambda fn, first_interval=0.0: timers.append(fn)))), timers


@pytest.mark.parametrize('status', ['open', 'closed'])
def test_history_schedule_is_noop_while_generation_accepted(observed, monkeypatch, status):
    """GAP-R7: undo/redo on an accepted (viewport-built) generation schedules no repair walk or capture."""
    scene, _ = observed
    monkeypatch.setattr(sessions, 'close_status', lambda: {'status': status})
    monkeypatch.setattr(sessions, '_initial_generation_diagnostics',
                        {'status': 'unavailable', 'scene_uid': 0, 'error': '', 'last_error': ''})
    blender, timers = _history_blender(scene)
    assert sessions.schedule_initial_generation(blender) is False
    assert timers == []


@pytest.mark.parametrize('status', ['closing', 'failed'])
def test_history_schedule_still_rebuilds_while_owner_is_being_closed(observed, monkeypatch, status):
    scene, _ = observed
    monkeypatch.setattr(sessions, 'close_status', lambda: {'status': status})
    monkeypatch.setattr(sessions, '_initial_generation_diagnostics',
                        {'status': 'unavailable', 'scene_uid': 0, 'error': '', 'last_error': ''})
    blender, timers = _history_blender(scene)
    assert sessions.schedule_initial_generation(blender) is True
    assert len(timers) == 1


def test_history_schedule_builds_when_no_generation_is_accepted(observed, monkeypatch):
    scene, _ = observed
    sessions._owners.clear()
    monkeypatch.setattr(sessions, '_initial_generation_diagnostics',
                        {'status': 'unavailable', 'scene_uid': 0, 'error': '', 'last_error': ''})
    blender, timers = _history_blender(scene)
    assert sessions.schedule_initial_generation(blender) is True
    assert len(timers) == 1


def _deferred(name, **keywords):
    return SimpleNamespace(function=SimpleNamespace(__name__=name), keywords=dict(keywords))


def test_prepare_authoring_updates_is_idempotent_and_keeps_existing_updates(monkeypatch):
    """Review of _prepare_authoring_updates: eval'd string annotations, re-register, add-on reload."""
    from ovrtx_blender_example import authoring_properties as props
    own_update = object()
    monkeypatch.setattr(props, 'FakeFloat', lambda **k: _deferred('FloatProperty', **k), raising=False)
    monkeypatch.setattr(props, 'FakeCollection', lambda **k: _deferred('CollectionProperty', **k), raising=False)
    monkeypatch.setattr(props, 'OWN_UPDATE', own_update, raising=False)

    def make_class():
        # Postponed (string) annotations, as `from __future__ import annotations` stores them.
        return type('Group', (), {'__annotations__': {
            'mass': 'FakeFloat(default=0.0)',
            'colliders': 'FakeCollection()',
            'matte': 'FakeFloat(update=OWN_UPDATE)',
            'plain': 'int',
        }})
    cls = make_class()
    props._prepare_authoring_updates(cls)
    mass, colliders, matte = (cls.__annotations__[k] for k in ('mass', 'colliders', 'matte'))
    assert mass.keywords['update'] is props._physics_property_updated
    assert 'update' not in colliders.keywords  # collections have explicit owner notifications
    assert matte.keywords['update'] is own_update  # an existing update callback is never replaced
    assert cls.__annotations__['plain'] == 'int'  # non-property annotations are left as declared
    # Re-register (unregister + register keeps the class): no re-eval, nothing replaced.
    before = dict(cls.__annotations__)
    props._prepare_authoring_updates(cls)
    assert cls.__annotations__ == before and cls.__annotations__['mass'] is mass
    assert mass.keywords == {'default': 0.0, 'update': props._physics_property_updated}
    # Add-on reload re-executes the class body: a fresh class gets the same policy.
    fresh = make_class()
    props._prepare_authoring_updates(fresh)
    assert fresh.__annotations__['mass'] is not mass
    assert fresh.__annotations__['mass'].keywords['update'] is props._physics_property_updated
    assert fresh.__annotations__['matte'].keywords['update'] is own_update

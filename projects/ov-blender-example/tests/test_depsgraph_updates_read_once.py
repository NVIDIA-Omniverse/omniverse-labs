"""The live depsgraph handler reads ``depsgraph.updates`` once (OMPE-115792 n-02-05).

Blender rebuilds ``Depsgraph.updates`` on every access, at a cost that grows
with the scene. The handler snapshots it once and passes the tuple, next to
the real depsgraph, to every callee that iterates it. These tests run the real
handler and the real callees against a depsgraph whose ``updates`` property
counts reads.
"""

from functools import wraps
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))

import ovrtx_blender_example.engine as engine_module  # noqa: E402
from ovrtx_blender_example import blender_callback_adapters as adapters  # noqa: E402
from ovrtx_blender_example import blender_interactive_edit_builders as builders  # noqa: E402
from ovrtx_blender_example import depsgraph_updates  # noqa: E402
from ovrtx_blender_example import physics_edit_snapshot as snapshots  # noqa: E402
from ovrtx_blender_example import physics_live_kinematics  # noqa: E402
from ovrtx_blender_example import scene_generation_sessions as sessions  # noqa: E402
from ovrtx_blender_example.physics_edit_snapshot import PhysicsChanges  # noqa: E402

UID = 9301


class _CountingDepsgraph:
    """``updates`` is rebuilt per access, like ``bpy.types.Depsgraph.updates``."""

    def __init__(self, updates, view_layer=None) -> None:
        self._updates = list(updates)
        self.view_layer = view_layer
        self.mode = "VIEWPORT"
        self.scene = None
        self.reads = 0

    @property
    def updates(self):
        self.reads += 1
        return iter(list(self._updates))


def _id(identifier, name, uid, kind=""):
    value = SimpleNamespace(
        name=name, name_full=name, type=kind, session_uid=uid,
        bl_rna=SimpleNamespace(identifier=identifier), library=None,
        override_library=None, keys=lambda: (),
        ov=SimpleNamespace(physics=None),
    )
    value.original = value
    return value


def _update(value, *, transform=False, geometry=False, shading=False):
    return SimpleNamespace(id=value, is_updated_transform=transform,
                           is_updated_geometry=geometry, is_updated_shading=shading)


def _camera_update():
    return _update(_id("Object", "Camera", 5, "CAMERA"), transform=True)


def _mixed_updates():
    return [
        _camera_update(),
        _update(_id("Object", "Cube", 6, "MESH"), shading=True),
        _update(_id("Material", "Paint", 7)),
        _update(_id("Scene", "Scene", 8)),
        _update(_id("Action", "Walk", 9)),
    ]


@pytest.fixture
def live_scene(monkeypatch):
    """Real handler and callees; only the generation/runtime state is faked."""

    scene = SimpleNamespace(session_uid=UID, objects=(), world=None, view_layers=())
    context = SimpleNamespace(scene=scene, view_layer=None, selected_objects=[])
    monkeypatch.setattr(engine_module, "bpy", SimpleNamespace(context=context))
    monkeypatch.setattr(physics_live_kinematics, "evaluating", lambda: False)
    monkeypatch.setattr(physics_live_kinematics, "note_animation_edit", lambda: None)
    # A proven physics-free observation: the camera-only fast path is live.
    observation = object()
    monkeypatch.setitem(sessions._physics_view_observations, UID, (observation, False))
    monkeypatch.setattr(snapshots, "scene_authoring_observation", lambda _scene: observation)
    monkeypatch.setattr(snapshots, "_authored", lambda _value: False)
    generation = SimpleNamespace(blender_prim_paths={}, scene_object_count=0)
    owner = SimpleNamespace(current_generation=generation, pending_generation=None,
                            _direct_state=SimpleNamespace(
                                visibility=SimpleNamespace(states={}, handles={}, view_layer=None),
                                data_users={}))
    monkeypatch.setitem(sessions._owners, UID, owner)
    monkeypatch.setitem(sessions._camera_capture_owners, UID, object())
    monkeypatch.setattr(sessions, "current_generation_shader_context",
                        lambda _scene: SimpleNamespace(generation=generation, materials=(), mesh_owners={}))
    monkeypatch.setattr(sessions, "current_generation_edit_context", lambda _scene: (None, ()))
    monkeypatch.setattr(sessions, "submit_current_scene_edit_group", lambda *_a: ())
    monkeypatch.setattr(sessions, "mark_scene_dirty", lambda *_a, **_k: None)
    monkeypatch.setattr(sessions, "observe_physics_edits", lambda *_a, **_k: PhysicsChanges())
    errors = []
    monkeypatch.setattr(engine_module, "_record_interactive_edit_bridge_diagnostics",
                        lambda **kwargs: errors.append(kwargs.get("last_error")))
    return scene, errors


@pytest.mark.parametrize("updates", [[_camera_update()], _mixed_updates()], ids=["camera-only", "mixed"])
def test_handler_reads_depsgraph_updates_at_most_once(live_scene, updates):
    scene, errors = live_scene
    for _ in range(3):
        depsgraph = _CountingDepsgraph(updates)
        engine_module._live_interactive_edit_depsgraph_handler(scene, depsgraph)
        assert [error for error in errors if error] == []
        assert depsgraph.reads <= 1, f"depsgraph.updates read {depsgraph.reads} times in one invocation"


def test_every_iterating_callee_sees_the_same_updates_in_order(live_scene, monkeypatch):
    scene, errors = live_scene
    updates = _mixed_updates()
    seen = {}

    def spy(module, name):
        original = getattr(module, name)

        @wraps(original)
        def wrapper(*args, **kwargs):
            seen[name] = kwargs.get("updates")
            return original(*args, **kwargs)
        monkeypatch.setattr(module, name, wrapper)

    for name in ("observe_physics_depsgraph", "observe_depsgraph_physics_edits", "affected_blender_ids",
                 "affected_blender_update_provenance", "shader_only_generation_update",
                 "topology_identity_changes_for_depsgraph"):
        spy(sessions, name)
    for name in ("_capture_mesh_geometry_ids", "_capture_visibility_ids"):
        spy(engine_module, name)
    for name in ("depsgraph_id_items", "_material_owners_for_depsgraph"):
        spy(adapters, name)
    spy(builders, "build_interactive_edits_from_depsgraph")
    monkeypatch.setattr(engine_module, "build_interactive_edits_from_depsgraph",
                        builders.build_interactive_edits_from_depsgraph)

    depsgraph = _CountingDepsgraph(updates)
    engine_module._live_interactive_edit_depsgraph_handler(scene, depsgraph)

    assert [error for error in errors if error] == []
    assert depsgraph.reads == 1
    # shader_only_generation_update proves the mixed batch is not shader-only,
    # so topology_identity_changes_for_depsgraph also runs.
    assert set(seen) == {
        "observe_physics_depsgraph", "observe_depsgraph_physics_edits", "affected_blender_ids",
        "affected_blender_update_provenance", "shader_only_generation_update",
        "topology_identity_changes_for_depsgraph", "_capture_mesh_geometry_ids", "_capture_visibility_ids",
        "depsgraph_id_items", "_material_owners_for_depsgraph", "build_interactive_edits_from_depsgraph",
    }
    for name, received in seen.items():
        assert received == tuple(updates), name
        assert all(a is b for a, b in zip(received, updates)), name


def test_live_engine_resolvers_receive_the_snapshot_or_read_themselves(live_scene, monkeypatch):
    """Engines with an ``updates`` keyword share the snapshot; older ones still work."""

    scene, errors = live_scene
    calls = []

    class _Engine:
        def __init__(self, name, *, takes_updates):
            self.name = name
            if takes_updates:
                self.build_interactive_edits_from_depsgraph = self._with_updates
            else:
                self.build_interactive_edits_from_depsgraph = self._legacy

        def _with_updates(self, depsgraph, *, context=None, selection_resolution=None,
                          value_edit_conversion_policies=None, edit_translator_factory=None, updates=None):
            calls.append((self.name, updates))
            return []

        def _legacy(self, depsgraph, *, context=None, selection_resolution=None,
                    value_edit_conversion_policies=None, edit_translator_factory=None):
            calls.append((self.name, tuple(depsgraph.updates)))
            return []

        def submit_interactive_edit(self, edit):
            return edit

    engines = [_Engine("current", takes_updates=True), _Engine("legacy", takes_updates=False)]
    monkeypatch.setattr(engine_module, "_EXACT_STAGE_CONFIGURATION", {"input_usd_path": "x.usda"})
    monkeypatch.setattr(engine_module, "_edit_submission_engines", lambda _engines: tuple(engines))
    updates = [_camera_update()]
    depsgraph = _CountingDepsgraph(updates)

    engine_module._live_interactive_edit_depsgraph_handler(scene, depsgraph)

    assert [error for error in errors if error] == []
    assert calls == [("current", tuple(updates)), ("legacy", tuple(updates))]
    # One read by the handler, one by the legacy resolver that cannot take the snapshot.
    assert depsgraph.reads == 2


def test_callees_default_to_reading_the_depsgraph():
    updates = _mixed_updates()
    depsgraph = _CountingDepsgraph(updates)
    assert adapters.depsgraph_id_items(depsgraph) == tuple(update.id for update in updates)
    assert sessions.affected_blender_ids(depsgraph) == sessions.affected_blender_ids(
        depsgraph, updates=tuple(updates))
    assert depsgraph.reads == 2


def test_empty_snapshot_still_fails_closed_to_full_observation(monkeypatch):
    scene = SimpleNamespace(session_uid=UID)
    observation = object()
    monkeypatch.setitem(sessions._physics_view_observations, UID, (observation, False))
    monkeypatch.setattr(snapshots, "scene_authoring_observation", lambda _scene: observation)
    full = []
    monkeypatch.setattr(sessions, "observe_physics_edits", lambda received: full.append(received) or "full")
    depsgraph = _CountingDepsgraph([_camera_update()])

    assert sessions.observe_depsgraph_physics_edits(scene, depsgraph, updates=()) == "full"
    assert full == [scene]
    assert depsgraph.reads == 0


def test_accepts_updates_follows_signatures():
    assert depsgraph_updates.accepts_updates(lambda depsgraph, *, updates=None: None)
    assert depsgraph_updates.accepts_updates(lambda *_a, **_k: None)
    assert not depsgraph_updates.accepts_updates(lambda *_a: None)
    assert not depsgraph_updates.accepts_updates(lambda depsgraph, *, context=None, scene=None: None)
    assert depsgraph_updates.keyword(lambda *_a: None, ()) == {}
    assert depsgraph_updates.keyword(lambda *, updates=None: None, None) == {}
    assert depsgraph_updates.keyword(lambda *, updates=None: None, ()) == {"updates": ()}


def test_shader_only_proof_rejects_non_shading_updates_before_sizing_the_scene(monkeypatch):
    """``current_generation_shader_context`` sizes scene.objects (O(scene) in Blender)."""

    built = []
    monkeypatch.setattr(sessions, "current_generation_shader_context",
                        lambda scene: built.append(scene) or None)
    scene = SimpleNamespace(session_uid=UID)
    camera = (_camera_update(),)
    assert sessions.shader_only_generation_update(scene, None, updates=camera) is False
    assert sessions.shader_only_generation_update(scene, None, updates=(*_mixed_updates(),)) is False
    assert built == []
    # A shading-only batch still builds the context and is decided by it.
    shading = (_update(_id("Object", "Cube", 6, "MESH"), shading=True),)
    assert sessions.shader_only_generation_update(scene, None, updates=shading) is False
    assert built == [scene]

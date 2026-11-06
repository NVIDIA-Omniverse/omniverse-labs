# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unrelated RNA inventories are forbidden on the verified reuse paths."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from ovrtx_blender_example import blender_callback_adapters as callbacks
from ovrtx_blender_example import blender_interactive_edit_builders as builders
from ovrtx_blender_example import blender_signal_translation as translation
from ovrtx_blender_example import scene_generation_sessions as sessions
from ovrtx_blender_example.blender_signals import BlenderRenderSignalSource


class NoTraversal:
    def __iter__(self):
        raise AssertionError("unrelated scene RNA was traversed")


def test_known_camera_identity_check_does_not_scan_scene(monkeypatch):
    from ovrtx_blender_example.scene_generation import BlenderId
    scene = SimpleNamespace(session_uid=123, objects=NoTraversal())
    generation = SimpleNamespace(blender_prim_paths={BlenderId('OBJECT', 456): object()})
    monkeypatch.setattr(sessions, '_owners', {123: SimpleNamespace(current_generation=generation)})
    camera = SimpleNamespace(type='CAMERA', session_uid=456)
    graph = SimpleNamespace(updates=[SimpleNamespace(id=camera), update('Camera')])
    assert sessions.topology_identity_changes_for_depsgraph(scene, graph) == set()
    scans = []
    monkeypatch.setattr(sessions, 'topology_identity_changes', lambda s: scans.append(s) or set())
    camera.session_uid = 457
    sessions.topology_identity_changes_for_depsgraph(scene, graph)
    assert scans == [scene]
    scans.clear()
    camera.session_uid = 456
    graph.updates.append(update('Scene'))
    sessions.topology_identity_changes_for_depsgraph(scene, graph)
    assert scans == [scene]


def test_camera_depsgraph_reuses_physics_free_observation_but_dirty_edits_do_not(monkeypatch):
    from ovrtx_blender_example.physics_edit_snapshot import scene_authoring_observation
    scene = SimpleNamespace(session_uid=123, objects=NoTraversal())
    observation = scene_authoring_observation(scene)
    monkeypatch.setattr(sessions, '_physics_view_observations', {123: (observation, False)})
    monkeypatch.setattr(sessions, '_physics_authoring_dirty', set())
    scans = []
    monkeypatch.setattr(sessions, 'observe_physics_edits', lambda s: scans.append(s))
    graph = SimpleNamespace(updates=[SimpleNamespace(id=SimpleNamespace(type='CAMERA'))])
    assert not sessions.observe_physics_depsgraph(scene, graph).restart
    assert not scans
    sessions._physics_authoring_dirty.add(123)
    sessions.observe_physics_depsgraph(scene, graph)
    assert scans == [scene]
    scans.clear()
    sessions._physics_authoring_dirty.clear()
    graph.updates.append(SimpleNamespace(id=SimpleNamespace(type='MESH')))
    sessions.observe_physics_depsgraph(scene, graph)
    assert scans == [scene]


def _discovery_objects(count):
    return [SimpleNamespace(ov=SimpleNamespace(physics=SimpleNamespace(authored=False)),
                            keys=lambda: ()) for _ in range(count)]


def test_idle_discovery_is_bounded_and_eventually_finds_raw_authoring(monkeypatch):
    """With the time budget spent, each tick still probes MIN_BATCH objects."""
    from ovrtx_blender_example import physics_edit_snapshot as snapshots
    checks = []
    monkeypatch.setattr(snapshots, '_authored', lambda p: checks.append(p) or p.authored)
    objects = _discovery_objects(300)
    discovery = snapshots.RenderOnlyDiscovery(SimpleNamespace(objects=objects))
    objects[-1].ov.physics.authored = True
    expired = lambda: 1.0  # noqa: E731 - the deadline is already reached
    assert not discovery.changed(time_budget_s=0, clock=expired)
    assert len(checks) == 128
    assert not discovery.changed(time_budget_s=0, clock=expired)
    assert len(checks) == 256
    assert discovery.changed(time_budget_s=0, clock=expired)
    assert len(checks) == 300


def test_idle_discovery_batch_is_sized_by_time_budget(monkeypatch):
    from ovrtx_blender_example import physics_edit_snapshot as snapshots
    checks = []
    now = [0.0]

    def clock():
        return now[0]

    def probe(group):
        now[0] += 1e-5  # 10 us per probed Object
        checks.append(group)
        return group.authored
    monkeypatch.setattr(snapshots, '_authored', probe)
    objects = _discovery_objects(5000)
    discovery = snapshots.RenderOnlyDiscovery(SimpleNamespace(objects=objects))
    assert not discovery.changed(time_budget_s=0.003, clock=clock)
    # A 3 ms budget at 10 us per Object probes ~300 Objects, not just 128.
    assert 290 <= len(checks) <= 310
    objects[4999].ov.physics.authored = True
    ticks = 1
    while not discovery.changed(time_budget_s=0.003, clock=clock):
        ticks += 1
        assert ticks < 40
    # Worst case is one pass: ceil(5000 / ~300) ticks, versus ceil(5000 / 128) = 40.
    assert ticks <= -(-5000 // 290) + 1


def test_idle_discovery_refreshes_roster_each_pass(monkeypatch):
    """An Object linked after discovery started is probed in the next pass (R3)."""
    from ovrtx_blender_example import physics_edit_snapshot as snapshots
    monkeypatch.setattr(snapshots, '_authored', lambda p: p.authored)
    objects = _discovery_objects(3)
    scene = SimpleNamespace(objects=list(objects))
    discovery = snapshots.RenderOnlyDiscovery(scene)
    late = _discovery_objects(1)[0]
    late.ov.physics.authored = True
    scene.objects.append(late)  # not in the creation-time roster
    assert not discovery.changed()  # one pass over the creation-time roster
    assert discovery.changed()  # the roster refreshed at the pass boundary
    assert discovery.passes == 1


def test_idle_discovery_probes_at_most_one_pass_per_tick(monkeypatch):
    from ovrtx_blender_example import physics_edit_snapshot as snapshots
    calls = []
    monkeypatch.setattr(snapshots, '_authored', lambda p: calls.append(p) or False)
    objects = _discovery_objects(5)
    discovery = snapshots.RenderOnlyDiscovery(SimpleNamespace(objects=objects))
    assert not discovery.changed()
    assert calls == [o.ov.physics for o in objects]
    calls.clear()
    discovery.index = 3  # a cursor mid-roster still covers every Object once
    assert not discovery.changed()
    assert sorted(map(id, calls)) == sorted(id(o.ov.physics) for o in objects)
    empty = snapshots.RenderOnlyDiscovery(SimpleNamespace(objects=[]))
    assert not empty.changed()


def test_object_participation_reads_shared_mesh_materials_once():
    from ovrtx_blender_example import physics_edit_snapshot as snapshots
    reads = []

    class Materials:
        def __iter__(self):
            reads.append(True)
            return iter(())
    mesh = SimpleNamespace(session_uid=5, materials=Materials())
    objects = [SimpleNamespace(ov=SimpleNamespace(physics=None), keys=lambda: (), pose=None, data=mesh)
               for _ in range(4)]
    for obj in objects:
        obj.ov.physics = SimpleNamespace(keys=lambda: [])
    cache = {}
    assert not any(snapshots.object_participates(obj, cache) for obj in objects)
    assert reads == [True]


def update(kind):
    return SimpleNamespace(id=SimpleNamespace(bl_rna=SimpleNamespace(identifier=kind)))


@pytest.mark.parametrize("kind", ["Object", "Mesh", "Material", "Light", "Camera", "World", "Scene", "Action"])
def test_direct_id_updates_do_not_enumerate_material_slots(kind):
    scene = SimpleNamespace(objects=NoTraversal())
    assert callbacks._material_owners_for_depsgraph(scene, SimpleNamespace(updates=[update(kind)])) == ()


@pytest.mark.parametrize("kind", ["ShaderNodeTree", "Image", "Unknown", ""])
def test_indirect_and_unknown_updates_keep_material_owner_lookup(kind):
    material = object()
    scene = SimpleNamespace(objects=[SimpleNamespace(material_slots=[SimpleNamespace(material=material)])])
    assert callbacks._material_owners_for_depsgraph(scene, SimpleNamespace(updates=[update(kind)])) == (material,)


def test_empty_edit_callback_does_not_build_light_inventory():
    assert builders.build_interactive_edits_from_depsgraph(
        SimpleNamespace(updates=[]), light_objects=NoTraversal()
    ) == []


def test_joint_free_admission_requires_clean_accepted_observation(monkeypatch):
    scene = SimpleNamespace(session_uid=731)
    monkeypatch.setattr(sessions, "_owners", {731: SimpleNamespace(current_generation=object())})
    monkeypatch.setattr(sessions, "_physics_view_observations", {731: (object(), False)})
    monkeypatch.setattr(sessions, "_physics_authoring_dirty", set())
    monkeypatch.setattr(sessions, "_dirty", {})
    assert sessions.viewport_joint_admission_clear(scene)
    sessions._physics_authoring_dirty.add(731)
    assert not sessions.viewport_joint_admission_clear(scene)
    sessions._physics_authoring_dirty.clear()
    sessions._dirty[731] = set()
    assert not sessions.viewport_joint_admission_clear(scene)
    sessions._dirty.clear()
    sessions._physics_view_observations[731] = (object(), True)
    assert not sessions.viewport_joint_admission_clear(scene)
    sessions._physics_view_observations.clear()
    assert not sessions.viewport_joint_admission_clear(scene)
    sessions._physics_view_observations[731] = (object(), False)
    sessions._owners[731].current_generation = None
    assert not sessions.viewport_joint_admission_clear(scene)


def test_admission_reuse_never_bypasses_final_render(monkeypatch):
    from ovrtx_blender_example import physics_authoring
    from ovrtx_blender_example.scene_generation import BlenderId, BlenderPrimPath
    scene = SimpleNamespace(camera=SimpleNamespace(session_uid=1))
    generation = SimpleNamespace(
        blender_prim_paths={BlenderId("OBJECT", 1): BlenderPrimPath("Camera", "CAMERA", "/Camera", "/Camera/Camera")},
        materialize_usd=lambda: "/generation.usda",
    )
    checks = []
    monkeypatch.setattr(sessions, "viewport_joint_admission_clear", lambda _scene: True)
    monkeypatch.setattr(physics_authoring, "require_scene_ready", lambda _scene, **kwargs: checks.append(kwargs))
    adapter = callbacks.BlenderRenderCallbackAdapter(
        generation_for_scene=lambda _scene: generation,
        translator=SimpleNamespace(translate=lambda signal: signal),
    )
    adapter.view_draw_from_scene(scene, None)
    assert checks == []
    adapter.final_render_from_scene(scene)
    assert checks == [{"allow_pending_anchors": False}]


def test_newly_enabled_invalid_joint_invalidates_before_viewport_submission(monkeypatch):
    from ovrtx_blender_example import physics_authoring
    joint = SimpleNamespace(enabled=True)
    obj = SimpleNamespace(name="Invalid joint", ov=SimpleNamespace(physics=SimpleNamespace(joint=joint)))
    scene = SimpleNamespace(session_uid=733, objects=[obj])
    monkeypatch.setattr(sessions, "_owners", {733: SimpleNamespace(current_generation=object())})
    monkeypatch.setattr(sessions, "_physics_view_observations", {733: (object(), False)})
    monkeypatch.setattr(sessions, "_physics_authoring_dirty", set())
    monkeypatch.setattr(sessions, "_dirty", {})
    monkeypatch.setattr(sessions, "bpy", SimpleNamespace(data=SimpleNamespace(scenes=[scene])))
    sessions.note_physics_authoring_changed(obj)
    monkeypatch.setattr(physics_authoring, "inactive_animation", lambda _obj: ("inactive animated field",))
    adapter = callbacks.BlenderRenderCallbackAdapter(
        generation_for_scene=lambda _scene: pytest.fail("invalid joint reached generation submission"),
    )
    with pytest.raises(translation.BlenderSignalTranslationError, match="inactive animated field"):
        adapter.view_draw_from_scene(scene, None)


def test_edit_context_keeps_live_collection_without_copying_it(monkeypatch):
    generation = SimpleNamespace(usd_path="/generation.usda", blender_prim_paths={})
    resolver = SimpleNamespace(begin_edit_callback=lambda: None)
    objects = NoTraversal()
    scene = SimpleNamespace(session_uid=732, objects=objects)
    runtime = SimpleNamespace(ovrtx=None, _current_edit_context=(generation, generation.usd_path, resolver))
    monkeypatch.setattr(sessions, "_owners", {732: SimpleNamespace(current_generation=generation)})
    monkeypatch.setattr(sessions, "_runtimes", {732: runtime})
    assert sessions.current_generation_edit_context(scene) == (resolver, objects)


def test_generation_layer_reuse_avoids_global_inventory_but_new_paths_refresh(monkeypatch):
    data = SimpleNamespace(materials=[], objects=[])
    translator = translation.RenderRequestTranslator(blender_module_provider=lambda: SimpleNamespace(data=data))
    calls = []
    material_layer, light_layer = object(), object()
    monkeypatch.setattr(translation.materialx_openpbr_conversion, "scene_layer_from_materials",
        lambda materials, path, **kwargs: calls.append(("material", path)) or SimpleNamespace(status=None, value=material_layer))
    monkeypatch.setattr(translation.light_scene_layer, "scene_layer_from_lights",
        lambda lights, path: calls.append(("light", path)) or light_layer)
    kwargs = dict(reuse=True, generation_owned=True)
    material_kwargs = dict(kwargs, source=BlenderRenderSignalSource.VIEW_DRAW)
    assert translator._material_scene_layer_from_scene("/generation.usda", **material_kwargs) is material_layer
    assert translator._light_scene_layer_from_scene("/generation.usda", **kwargs) is light_layer
    data.materials = data.objects = NoTraversal()
    assert translator._material_scene_layer_from_scene("/generation.usda", **material_kwargs) is material_layer
    assert translator._light_scene_layer_from_scene("/generation.usda", **kwargs) is light_layer
    assert len(calls) == 2
    # Exact-stage requests retain inventory verification; no generation proof.
    with pytest.raises(AssertionError):
        translator._light_scene_layer_from_scene("/generation.usda", reuse=True)
    data.materials = data.objects = []
    translator._material_scene_layer_from_scene("/new-generation.usda", **material_kwargs)
    translator._light_scene_layer_from_scene("/new-generation.usda", **kwargs)
    assert len(calls) == 4
    translator._light_scene_layer_from_scene("/new-generation.usda", reuse=False, generation_owned=True)
    assert len(calls) == 5


def test_animation_filter_reads_paths_once_until_the_action_is_reported_edited():
    from ovrtx_blender_example import physics_edit_snapshot as snapshots
    reads = []

    class Curve:
        def __init__(self, path):
            self._path = path

        @property
        def data_path(self):
            reads.append(self._path)
            return self._path
    action = SimpleNamespace(session_uid=901, fcurves=[Curve('location')])
    owner = SimpleNamespace(session_uid=902, animation_data=SimpleNamespace(drivers=[], action=action))
    assert not snapshots.animation_targets_physics(owner)
    assert not snapshots.animation_targets_physics(owner)
    assert reads == ['location']
    action.fcurves.append(Curve('ov.physics.rigid_body.mass_kg'))
    assert not snapshots.animation_targets_physics(owner)  # unreported edit: idle discovery's job
    snapshots.forget_animation_paths([901])  # depsgraph_update_post reported the Action
    assert snapshots.animation_targets_physics(owner)
    driven = SimpleNamespace(session_uid=903, animation_data=SimpleNamespace(drivers=[], action=None))
    assert not snapshots.animation_targets_physics(driven)
    driven.animation_data.drivers.append(Curve('pose.bones["B"].ov.physics.rigid_body.mass_kg'))
    assert snapshots.animation_targets_physics(driven)  # driver count changed: re-read
    assert not snapshots.animation_targets_physics(SimpleNamespace(animation_data=None))

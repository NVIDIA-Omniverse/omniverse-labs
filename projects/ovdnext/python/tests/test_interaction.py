# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from capture_semantics import test_scene_roles, multi_geom_roles, unique_instance_id, dynamic_geometry_owner
from pathlib import Path

import pytest
from _pvd_native import Session

from ovd_ui.interaction import InteractionState, TransparencyMode

ROOT = Path(__file__).resolve().parents[2]


def state():
    result = InteractionState()
    result.begin_capture("capture-A")
    result.update_frame([1, 2, 3, 4, 5, 10, 11], [1, 2, 3, 4, 5, 10, 11],
                        children={1: [2], 2: [1], 10: [1, 3], 11: [3, 4, 5]},
                        owners={1: [10], 2: [10], 3: [10, 11], 4: [11], 5: [11]}, scene_ids=[10, 11])
    return result


def test_modifier_selection_range_anchor_primary_and_empty_viewport():
    interaction = state()
    assert interaction.pick(2).selected == (2,)
    assert interaction.pick(4, toggle=True).selected == (2, 4)
    assert interaction.pick(2, toggle=True).selected == (4,)
    assert interaction.pick(5, extend=True, order=[1, 2, 3, 4, 5]).selected == (2, 3, 4, 5)
    assert interaction.primary == 5
    assert interaction.pick(1, extend=True, order=[1, 2, 3, 4, 5]).selected == (2, 1)
    assert interaction.primary == 1
    assert interaction.pick(None, toggle=True).selected == (2, 1)
    assert interaction.pick(None).selected == ()
    assert interaction.pick(999).selected == ()


def test_immutable_owner_reuse_keeps_validation_and_rejects_equal_boolean_ids():
    interaction = state()
    owners = {1: frozenset((10, 0, -1)), 2: frozenset((11,))}
    interaction.update_frame(interaction.known, interaction.alive, owners=owners)
    accepted = interaction._owners[1]
    assert accepted == frozenset((10,))
    interaction.update_frame(interaction.known, interaction.alive, owners=owners)
    assert interaction._owners[1] is accepted
    # Equality alone cannot prove that a new immutable input is valid.
    interaction.update_frame(interaction.known, interaction.alive, owners={1: frozenset((1,))})
    with pytest.raises(TypeError, match='native integer ID'):
        interaction.update_frame(interaction.known, interaction.alive, owners={1: frozenset((True,))})


def test_mutable_owner_inputs_and_replaced_keys_are_always_refreshed():
    interaction = state()
    membership = [10]
    owners = {1: membership, 2: frozenset((11,))}
    interaction.update_frame(interaction.known, interaction.alive, owners=owners)
    interaction.set_visible([10], False)
    assert not interaction.is_visible(1)
    membership[:] = [11]
    interaction.update_frame(interaction.known, interaction.alive, owners=owners)
    assert interaction.is_visible(1)
    owners[1] = frozenset((10,))
    del owners[2]
    interaction.update_frame(interaction.known, interaction.alive, owners=owners)
    assert not interaction.is_visible(1) and 2 not in interaction._owners
    assert set(interaction._owner_inputs)=={1}
    interaction.begin_capture('replacement')
    assert not interaction._owners and not interaction._owner_inputs


def test_mutating_an_owner_list_to_an_invalid_id_cannot_reuse_validation():
    interaction = state()
    owners = {1: [10]}
    interaction.update_frame(interaction.known, interaction.alive, owners=owners)
    owners[1][:] = [True]
    with pytest.raises(TypeError, match='native integer ID'):
        interaction.update_frame(interaction.known, interaction.alive, owners=owners)


def test_history_back_forward_does_not_append_and_new_branch_discards_forward():
    interaction = state()
    interaction.select([1])
    interaction.select([2, 3])
    interaction.select([4])
    assert interaction.back().selected == (2, 3)
    assert interaction.back().selected == (1,)
    assert interaction.forward().selected == (2, 3)
    interaction.select([5])
    assert not interaction.can_forward
    assert interaction.back().selected == (2, 3)
    assert interaction.forward().selected == (5,)
    revision = interaction.revision
    interaction.select([5])
    assert interaction.revision == revision


def test_selection_and_tags_survive_death_resurrection_and_resampling():
    interaction = state()
    interaction.update_frame(interaction.known, interaction.alive, segment_epoch=0)
    interaction.select([1, 3])
    interaction.hide_selected()
    interaction.tag_selected()
    interaction.set_expanded("class:rigid/object:1/shapes")
    interaction.update_frame(interaction.known, [2, 4, 5, 10, 11], segment_epoch=0)
    assert interaction.selected == () and interaction.sticky == (1, 3)
    assert 1 in interaction.hidden and 1 in interaction.transparent
    interaction.update_frame([1, 2, 3, 4, 5, 6, 10, 11], [1, 2, 3, 4, 5, 6, 10, 11], segment_epoch=1)
    assert interaction.selected == (1, 3)
    assert 6 not in interaction.selected and 6 not in interaction.hidden
    assert "class:rigid/object:1/shapes" in interaction.expanded
    interaction.begin_capture("capture-A")  # Explicit reopen is a fresh owning Session.
    assert not interaction.selected and not interaction.hidden and not interaction.transparent
    assert not interaction.expanded and not interaction.can_back


def test_transient_query_identity_does_not_transfer_to_next_segment():
    interaction = state()
    interaction.update_frame([1, -2], [1, -2], segment_epoch=0)
    interaction.select([-2])
    interaction.hide_selected()
    interaction.tag_selected()
    interaction.update_frame([1, -2], [1, -2], segment_epoch=1)
    assert interaction.sticky == ()
    assert -2 not in interaction.hidden and -2 not in interaction.transparent
    assert not interaction.can_back


def test_new_user_selection_replaces_sticky_dead_object_intent():
    interaction = state()
    interaction.select([1, 2])
    interaction.update_frame(interaction.known, [2, 3, 4, 5, 10, 11])
    assert interaction.selected == (2,)
    interaction.pick(3, additive=True)
    assert interaction.sticky == (2, 3)
    interaction.update_frame(interaction.known, interaction.known)
    assert interaction.selected == (2, 3)


def test_visibility_cascade_cycles_and_shared_scene_ownership():
    interaction = state()
    interaction.set_visible([1], False)
    assert not interaction.is_visible(1) and not interaction.is_visible(2)
    assert interaction.is_visible(3)
    interaction.show_all()
    interaction.set_visible([10], False)
    assert interaction.hidden == {10}  # Scene hides through ownership, not direct actor tags.
    assert not interaction.is_visible(1)
    assert interaction.is_visible(3)  # Shared actor remains owned by visible scene 11.
    interaction.set_visible([11], False)
    assert not interaction.is_visible(3)
    interaction.set_visible([10], True)
    assert interaction.is_visible(3)
    interaction.update_frame(interaction.known, interaction.alive - {10})
    assert not interaction.is_visible(3)  # A dead owner cannot override a hidden live owner.


def test_isolation_preserves_selected_ownership_and_supports_scene_members():
    interaction = state()
    interaction.select([1])
    interaction.isolate()
    assert interaction.is_visible(1) and interaction.is_visible(2)
    assert not interaction.is_visible(3) and not interaction.is_visible(4)
    interaction.select([10])
    interaction.isolate()
    assert interaction.is_visible(1) and interaction.is_visible(3)
    assert not interaction.is_visible(4)
    interaction.hide_all()
    assert not any(interaction.is_visible(value) for value in interaction.alive)
    interaction.show_all()
    assert all(interaction.is_visible(value) for value in interaction.alive)


def test_transparency_modes_override_tags_and_use_continuous_amount():
    interaction = state()
    interaction.select([1])
    interaction.tag_selected()
    assert {1, 2} <= interaction.transparent
    for mode, expected in ((TransparencyMode.NONE, (1., 1.)),
                           (TransparencyMode.ALL, (.25, .25)),
                           (TransparencyMode.SELECTED, (.25, 1.)),
                           (TransparencyMode.UNSELECTED, (1., .25)),
                           (TransparencyMode.TAGGED, (.25, .25))):
        interaction.set_transparency(mode, .75)
        assert (interaction.alpha_for(1), interaction.alpha_for(2)) == expected
    interaction.set_transparency(TransparencyMode.TAGGED, 1.)
    assert interaction.alpha_for(1) == 0
    interaction.set_transparency(TransparencyMode.TAGGED, 0.)
    assert interaction.alpha_for(1) == .999
    interaction.clear_transparency()
    assert interaction.alpha_for(1) == 1.
    for bad in (-.1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            interaction.set_transparency(TransparencyMode.ALL, bad)


def test_history_bound_and_owned_snapshots():
    interaction = InteractionState(history_limit=3)
    interaction.update_frame(range(1, 8), range(1, 8))
    snapshot = interaction.select([1])
    for object_id in range(2, 8):
        interaction.select([object_id])
    assert interaction.back().selected == (6,)
    assert interaction.back().selected == (5,)
    assert not interaction.can_back
    assert snapshot.selected == (1,) and snapshot.hidden == frozenset()
    with pytest.raises(TypeError):
        interaction.select(["7"])


def test_real_native_lifetimes_restore_selection_and_visibility_without_new_history(tmp_path):
    session = Session(str(ROOT / "test/data/test_scene.ovd"), str(tmp_path))
    scene = session.scenes()[0]
    interaction = InteractionState()
    session.seek(scene.id, 1)
    instances = session.instances()
    names = {item.id: item.name for item in session.classes()}
    actor = next(item for item in instances if names[item.class_id] == "PxRigidDynamic")
    interaction.update_frame([item.id for item in instances], [item.id for item in instances if item.alive])
    interaction.select([actor.id])
    interaction.hide_selected()
    interaction.tag_selected()
    session.seek(scene.id, 3)
    instances = session.instances()
    interaction.update_frame([item.id for item in instances], [item.id for item in instances if item.alive])
    assert interaction.selected == () and interaction.sticky == (actor.id,)
    session.seek(scene.id, 1)
    instances = session.instances()
    interaction.update_frame([item.id for item in instances], [item.id for item in instances if item.alive])
    assert interaction.selected == (actor.id,)
    assert actor.id in interaction.hidden and actor.id in interaction.transparent
    assert interaction.back().selected == ()  # Frame changes did not append history.


def test_integer_validation_preserves_numpy_and_rejects_coercible_non_ids():
    import numpy as np
    interaction=InteractionState()
    interaction.update_frame([1,np.int32(2),np.uint64(3)], [1,2,3])
    assert interaction.select([np.int64(3),2,2,1]).selected==(3,2,1)
    for invalid in (True,False,1.0,np.float32(2),'1',None):
        with pytest.raises(TypeError):interaction.select([invalid])


def native_interaction(tmp_path, filename, frame=1):
    """Use the owned publication and canonical graph supplied to interaction."""
    from ovd_ui.adapters import PvdDataAdapter

    session = Session(str(ROOT / "test/data" / filename), str(tmp_path))
    view = session.presentation(session.scenes()[0].id, frame)
    assert view.ready and view.request_applied
    adapter = PvdDataAdapter(view)
    children, owners = adapter.ownership_graph()
    result = InteractionState()
    result.begin_capture(view.capture_key)
    result.update_frame(adapter.instances,
        [item.id for item in adapter.instances.values() if item.alive],
        segment_epoch=view.segment_epoch, children=children, owners=owners,
        scene_ids=[scene.instance_id for scene in view.scenes()])
    return session, view, adapter, result


def test_real_scene_barrier_and_child_flags_remain_independent(tmp_path):
    _, _, adapter, interaction = native_interaction(tmp_path, "test_scene.ovd")
    ids=test_scene_roles(adapter.session)
    actor, shape, geometry, scene = ids.actor, ids.shape, ids.geometry, ids.scene
    assert [adapter.class_name(i) for i in (actor, shape, geometry, scene)] == [
        "PxRigidDynamic", "PxShape", "PxSphereGeometry", "PxScene"]
    interaction.select([actor])
    history = tuple(interaction._history)

    interaction.set_visible([scene], False)
    assert interaction.hidden == {scene}
    assert not any(interaction.is_visible(i) for i in (actor, shape, geometry))
    interaction.set_visible([actor], True)
    assert interaction.hidden == {scene}  # Child action never changes the parent indicator.
    assert not interaction.is_visible(actor)
    interaction.set_visible([actor], False)
    interaction.set_visible([scene], True)
    assert interaction.hidden == {actor, shape, geometry}
    assert not interaction.is_visible(actor)  # Showing a scene preserves explicit child tags.
    interaction.set_visible([actor], True)
    assert all(interaction.is_visible(i) for i in (actor, shape, geometry))
    assert interaction.selected == (actor,)
    assert tuple(interaction._history) == history


def test_real_scene_transparency_cascades_and_child_override_keeps_parent_tag(tmp_path):
    _, _, adapter, interaction = native_interaction(tmp_path, "test_scene.ovd")
    ids=test_scene_roles(adapter.session)
    interaction.set_transparent([ids.scene], True)
    assert interaction.transparent == {ids.scene, ids.actor, ids.shape, ids.geometry}
    assert interaction.transparency_mode == TransparencyMode.NONE  # A tag is not a mode switch.
    assert interaction.alpha_for(ids.actor) == 1.
    interaction.set_transparency(TransparencyMode.TAGGED, .75)
    assert interaction.alpha_for(ids.actor) == .25
    interaction.set_transparent([ids.actor], False)
    assert interaction.transparent == {ids.scene}
    assert interaction.alpha_for(ids.actor) == 1.
    interaction.set_transparent([ids.shape], True)
    assert interaction.transparent == {ids.scene, ids.shape, ids.geometry}
    assert interaction.alpha_for(ids.shape) == .25
    interaction.clear_transparency()
    assert interaction.transparent == set()


def test_real_shape_and_geometry_tags_do_not_follow_material_aliases(tmp_path):
    _, _, adapter, interaction = native_interaction(tmp_path, "test_scene.ovd")
    ids=test_scene_roles(adapter.session)
    nodes, _ = adapter.hierarchy("scene")
    alias = next(node for node in nodes.values()
                 if node.kind == "reference" and node.instance_id == ids.material)
    assert alias.target_key == f"object:{ids.material}" and f"object:{ids.shape}" in alias.parent_key

    interaction.set_visible([ids.shape], False)
    assert interaction.hidden == {ids.shape, ids.geometry}
    assert interaction.is_visible(ids.actor)  # Actor flag remains on; its specific shape is hidden.
    assert interaction.is_visible(ids.material)
    interaction.set_transparent([ids.actor], True)
    assert interaction.transparent == {ids.actor, ids.shape, ids.geometry}
    assert ids.material not in interaction.transparent
    interaction.show_all()
    interaction.clear_transparency()
    interaction.set_visible([ids.material], False)
    interaction.set_transparent([ids.material], True)
    assert interaction.hidden == interaction.transparent == {ids.material}
    assert all(interaction.is_visible(i) for i in (ids.actor, ids.shape, ids.geometry))


def test_real_collapsed_articulation_cascade_has_no_parent_or_sibling_effect(tmp_path):
    _, _, adapter, interaction = native_interaction(tmp_path, "multi_geom.ovd")
    ids=multi_geom_roles(adapter.session)
    # Canonical ownership is available without constructing/expanding TreeView rows.
    assert adapter.class_name(ids.middle2) == adapter.class_name(ids.leaf) == "PxArticulationLink"
    assert adapter.class_name(ids.middle2_joint) == "PxArticulationJointReducedCoordinate"
    assert adapter.class_name(ids.middle2_shape) == "PxShape"
    descendants = ids.families[ids.middle2] | ids.families[ids.leaf]
    interaction.set_visible([ids.middle2], False)
    assert interaction.hidden == descendants
    assert all(interaction.is_visible(i) for i in (ids.middle1, ids.root, ids.articulation, ids.box))
    interaction.set_visible([ids.leaf], True)
    assert interaction.hidden == ids.families[ids.middle2]
    assert interaction.is_visible(ids.leaf)  # Parent tag is local, not a permanent inherited override.
    interaction.set_transparent([ids.middle2], True)
    assert interaction.transparent == descendants
    interaction.set_transparent([ids.leaf], False)
    assert interaction.transparent == ids.families[ids.middle2]
    assert not interaction.selected and not interaction.can_back


def test_real_reused_scene_handle_never_transfers_visibility_between_creation_ids(tmp_path):
    _, view, adapter, interaction = native_interaction(tmp_path, "resample_two_segment.ovd")
    first, second = view.scenes()
    assert first.alive and second.alive
    assert first.stream_id == second.stream_id and first.instance_id != second.instance_id
    geometry = view.geometry_snapshot()
    actors_by_scene = {
        scene.instance_id: {int(actor) for actor, owner in zip(geometry.object_ids, geometry.scene_ids)
                            if int(owner) == scene.instance_id}
        for scene in (first, second)}
    assert all(actors_by_scene.values())
    assert actors_by_scene[first.instance_id].isdisjoint(actors_by_scene[second.instance_id])
    assert all(adapter.class_name(actor) in ("PxRigidDynamic", "PxRigidStatic")
               for actors in actors_by_scene.values() for actor in actors)

    interaction.set_visible([first.instance_id], False)
    assert interaction.hidden == {first.instance_id}
    assert not any(interaction.is_visible(actor) for actor in actors_by_scene[first.instance_id])
    assert all(interaction.is_visible(actor) for actor in actors_by_scene[second.instance_id])
    interaction.set_visible([second.instance_id], False)
    interaction.set_visible([first.instance_id], True)
    assert interaction.hidden == {second.instance_id}
    assert all(interaction.is_visible(actor) for actor in actors_by_scene[first.instance_id])
    assert not any(interaction.is_visible(actor) for actor in actors_by_scene[second.instance_id])


def test_real_row_scope_handoff_keeps_local_alias_and_native_flags_separate(tmp_path):
    from ovd_ui.models import ObjectTreeModel
    from ovd_ui.row_actions import RowActionState

    _, _, adapter, interaction = native_interaction(tmp_path, "test_scene.ovd")
    ids=test_scene_roles(adapter.session)
    model = ObjectTreeModel()
    model.populate(adapter)
    rows = RowActionState()
    rows.sync(model.capture_key, model.segment_epoch)
    cache_size = len(model.cache)

    scene_scope = model.action_scope([f"object:{ids.scene}"])
    assert scene_scope.native_roots == (ids.scene,) and not scene_scope.local_rows
    assert rows.apply(scene_scope, False)
    interaction.set_visible(scene_scope.native_roots, False)
    assert interaction.hidden == {ids.scene}
    assert rows.is_hidden(model.row_identity(f"object:{ids.scene}"), interaction.hidden)
    interaction.show_all()

    actor_scope = model.action_scope([f"object:{ids.actor}"])
    assert actor_scope.native_roots == (ids.actor,)
    alias = next(row for row in actor_scope.local_rows
                 if row.kind == "reference" and row.instance_id == ids.material)
    assert rows.apply(actor_scope, False)
    interaction.set_visible(actor_scope.native_roots, False)
    assert rows.is_hidden(alias, interaction.hidden)
    assert interaction.hidden == {ids.actor, ids.shape, ids.geometry} and interaction.is_visible(ids.material)

    alias_scope = model.action_scope([alias])
    assert alias_scope.native_roots == ()
    assert rows.apply(alias_scope, True)
    interaction.set_visible(alias_scope.native_roots, True)
    assert not rows.is_hidden(alias, interaction.hidden)
    assert interaction.hidden == {ids.actor, ids.shape, ids.geometry}
    assert len(model.cache) == cache_size  # Resolving/applying never expands a TreeView branch.
    assert not interaction.selected and not interaction.can_back

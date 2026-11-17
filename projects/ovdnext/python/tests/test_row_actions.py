# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tree selection and local tags retain native identity without expanding widgets."""
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from _pvd_native import Session

from capture_semantics import test_scene_roles
from ovd_ui.adapters import PvdDataAdapter
from ovd_ui.hierarchy import HierarchyNode
from ovd_ui.models import ObjectTreeModel
from ovd_ui.row_actions import RowActionState

ROOT = Path(__file__).resolve().parents[2]


def capture(tmp_path, filename="test_scene.ovd", frame=1, page_size=256):
    session = Session(str(ROOT / "test/data" / filename), str(tmp_path))
    session.seek(session.scenes()[0].id, frame)
    adapter = PvdDataAdapter(session)
    model = ObjectTreeModel(page_size=page_size)
    model.populate(adapter)
    if filename == 'test_scene.ovd':
        session.presentation(frame=frame)
        model.fixture_roles = test_scene_roles(session)
    return session, adapter, model


def local_state(model):
    state = RowActionState()
    state.sync(model.capture_key, model.segment_epoch)
    return state


def fail_materialization(*_):
    pytest.fail("Row action attempted to expand or materialize a UI branch")


def test_actual_group_selection_retains_only_explicit_rows_and_stable_columns(tmp_path):
    _, _, model = capture(tmp_path)
    ids = model.fixture_roles
    key = f"group:object:{ids.scene}:pxrigiddynamic"
    before = len(model.cache)
    rows = model.row_identities([key, key])
    assert len(rows) == 1 and rows[0].kind == "group" and rows[0].instance_id is None
    assert rows[0].owner_id == rows[0].source_instance_id == ids.scene
    assert len(model.cache) == before
    state = local_state(model)
    assert state.select(rows)
    assert state.selected == rows
    restored = model.restore_rows(state.selected)
    assert [item.key for item in restored] == [key]
    assert dict.get(model.cache, f"object:{ids.actor}") is None  # No actor was selected/materialized.
    assert model.get_item_value_model_count() == 4
    assert [value.as_string for value in restored[0].values] == ["PxRigidDynamic", "", "", ""]


def test_native_group_scope_is_frozen_filter_independent_and_never_expands(tmp_path, monkeypatch):
    _, _, model = capture(tmp_path)
    ids = model.fixture_roles
    key = f"group:object:{ids.scene}:pxrigiddynamic"
    model.set_filter("does not match any row")
    before = (len(model.cache), len(model._specs))
    monkeypatch.setattr(model, "_item", fail_materialization)
    monkeypatch.setattr(model, "_array_children", fail_materialization)
    scope = model.action_scope([key])
    assert scope.native_roots == (ids.actor,)
    assert key in scope.local_keys and f"group:object:{ids.shape}:materials" in scope.local_keys
    assert ids.material not in scope.native_roots
    assert (len(model.cache), len(model._specs)) == before
    with pytest.raises(FrozenInstanceError):
        scope.native_roots = ()
    with pytest.raises(FrozenInstanceError):
        scope.selected_rows[0].owner_id = 999
    assert scope.local_keys == tuple(row.key for row in scope.local_rows)


def test_material_alias_flags_do_not_transfer_to_canonical_target_or_parent(tmp_path):
    _, adapter, model = capture(tmp_path)
    ids = model.fixture_roles
    key = f"reference:group:object:{ids.shape}:materials:{ids.material}"
    row = model.row_identity(key)
    prop = adapter.property(ids.shape, "materials")
    assert row.instance_id == ids.material and row.owner_id == row.source_instance_id == ids.shape
    assert row.property_id == prop.id
    scope = model.action_scope([key])
    assert scope.native_roots == () and scope.local_keys == (key,)
    state = local_state(model)
    assert state.apply(scope, False)
    assert state.is_hidden(row)
    assert not state.is_hidden(model.row_identity(f"object:{ids.material}"))
    assert not state.is_hidden(model.row_identity(f"group:object:{ids.shape}:materials"))
    assert not state.is_transparent(row)
    assert state.apply(model.action_scope([key], action="transparency"), True)
    assert state.is_transparent(row)
    # A native target's own tag never turns another alias's local dot on.
    assert not state.is_hidden(model.row_identity(f"reference:group:object:{ids.physics}:shapes:{ids.shape}"), {ids.shape})
    assert state.is_hidden(model.row_identity(f"object:{ids.shape}"), {ids.shape})


def test_scene_barrier_and_explicit_actor_keep_distinct_scopes(tmp_path):
    _, _, model = capture(tmp_path)
    ids = model.fixture_roles
    scene = model.action_scope([f"object:{ids.scene}"])
    assert scene.native_roots == (ids.scene,) and not scene.local_keys
    explicit = model.action_scope([f"object:{ids.scene}", f"object:{ids.actor}"])
    assert explicit.native_roots == (ids.scene, ids.actor)
    assert f"group:object:{ids.actor}:shapes" in explicit.local_keys
    transparent = model.action_scope([f"object:{ids.scene}"], action="transparency")
    assert transparent.native_roots == (ids.scene,)
    assert f"group:object:{ids.scene}:pxrigiddynamic" in transparent.local_keys
    assert f"group:object:{ids.shape}:materials" in transparent.local_keys


def test_local_downcascade_never_changes_ancestor_indicator(tmp_path):
    _, _, model = capture(tmp_path)
    ids = model.fixture_roles
    key = f"group:object:{ids.scene}:pxrigiddynamic"
    child = f"group:object:{ids.actor}:shapes"
    state = local_state(model)
    assert state.apply(model.action_scope([key]), False)
    assert state.is_hidden(model.row_identity(key))
    assert state.is_hidden(model.row_identity(child))
    assert state.apply(model.action_scope([child]), True)
    assert not state.is_hidden(model.row_identity(child))
    assert state.is_hidden(model.row_identity(key))
    revision = state.revision
    state.sync(model.capture_key, model.segment_epoch)
    state.apply(model.action_scope([child]), True)
    assert state.revision == revision
    state.show_all()
    assert not state.is_hidden(model.row_identity(key))


def test_real_dead_row_intent_restores_without_selecting_new_objects(tmp_path):
    session, adapter, model = capture(tmp_path)
    ids = model.fixture_roles
    rows = model.row_identities([f"object:{ids.actor}", f"group:object:{ids.actor}:shapes"])
    state = local_state(model)
    state.select(rows)
    state.apply(model.action_scope([f"group:object:{ids.actor}:shapes"]), False)
    session.seek(session.scenes()[0].id, 3)
    adapter.reload()
    model.populate(adapter)
    state.sync(model.capture_key, model.segment_epoch)
    assert model.restore_rows(state.selected) == ()
    assert state.selected == rows
    session.seek(session.scenes()[0].id, 1)
    adapter.reload()
    model.populate(adapter)
    assert [item.key for item in model.restore_rows(state.selected)] == [row.key for row in rows]
    assert state.is_hidden(model.row_identity(f"group:object:{ids.actor}:shapes"))


def test_actual_query_rows_are_local_segment_scoped_and_restore_after_rebuild(tmp_path):
    _, adapter, model = capture(tmp_path, "cloth_physx34.pxd2")
    item = model.canonical_item(-6)
    row = model.row_identity(item.key)
    assert row.kind == "array_element" and row.query_id == -6 and row.instance_id is None
    assert row.source_instance_id == 6 and row.property_id is not None and row.element_index == 0
    assert row.segment_epoch == model.segment_epoch
    scope = model.action_scope([row.key])
    assert scope.native_roots == () and scope.local_rows == (row,)
    state = local_state(model)
    state.select([row])
    state.apply(scope, False)
    # Force a fresh pure projection: selected record path must restore lazily.
    replacement = ObjectTreeModel()
    replacement.populate(adapter)
    assert row.key not in replacement._specs
    assert [item.key for item in replacement.restore_rows(state.selected)] == [row.key]
    state.sync(model.capture_key, model.segment_epoch + 1)
    assert not state.selected and not state.accepts(scope)
    revision = state.revision
    assert not state.apply(scope, True) and state.revision == revision


def test_capture_replacement_atomically_rejects_scope_and_selection(tmp_path):
    _, _, model = capture(tmp_path)
    ids = model.fixture_roles
    rows = model.row_identities([f"group:object:{ids.scene}:pxrigiddynamic", f"object:{ids.actor}"])
    scope = model.action_scope(rows)
    state = local_state(model)
    state.select(rows)
    state.apply(scope, False)
    _, _, other = capture(tmp_path / "other")
    state.sync(other.capture_key, other.segment_epoch)
    revision = state.revision
    assert not state.selected and not state.apply(scope, True)
    assert not state.select(rows) and state.revision == revision
    assert not other.restore_rows(rows)
    assert not other.row_identities(rows)
    assert not state.is_hidden(other.row_identity(rows[0].key))


def test_native_resampling_retains_positive_rows_and_flags_only_for_the_same_source(tmp_path):
    session, adapter, model = capture(tmp_path, "resample_two_segment.ovd", frame=1)
    object_key = next(key for key, node in model._specs.items()
                      if node.class_name == "PxPhysics" and node.kind == "object")
    group_key = next(key for key, node in model._specs.items()
                     if node.kind == "group" and node.source_instance_id is not None)
    state = local_state(model)
    rows = model.row_identities([object_key, group_key])
    state.select(rows)
    state.apply(model.action_scope([group_key]), False)
    session.seek(session.scenes()[0].id, 45)
    adapter.reload()
    model.populate(adapter)
    state.sync(model.capture_key, model.segment_epoch)
    # This file loads both recording segments up front; seeking does not
    # fabricate a new native publication epoch.
    assert len({node.instance_id for node in model._specs.values()
                if node.class_name == "PxScene" and node.kind == "object"}) == 2
    assert state.selected == rows
    assert state.is_hidden(rows[1])
    # Native objects in either recording segment retain their own ID state.
    assert all(not state.is_hidden(row) for row in model.row_identities(model._specs)
               if row.kind == "object")


def test_class_group_scopes_use_real_members_without_changing_native_row_identity(tmp_path):
    from capture_semantics import multi_geom_roles
    session, _, model = capture(tmp_path, "multi_geom.ovd")
    sphere = multi_geom_roles(session).sphere
    actor = model.row_identity(f"object:{sphere}")
    model.set_grouping("class")
    assert model.row_identity(f"object:{sphere}").token == actor.token
    key = model._specs[actor.key].parent_key
    scope = model.action_scope([key])
    expected = {model._specs[child].instance_id for child in model._specs[key].children}
    assert set(scope.native_roots) == expected
    assert [item.key for item in model.restore_rows([actor])] == [actor.key]


def test_hide_all_and_clear_tags_cover_local_rows_without_touching_native_sets(tmp_path):
    _, _, model = capture(tmp_path)
    ids = model.fixture_roles
    state = local_state(model)
    rows = model.row_identities(model._specs)
    assert state.hide_all(rows)
    assert all(state.is_hidden(row) for row in rows if not row.native)
    assert not any(state.is_hidden(row) for row in rows if row.native)
    state.apply(model.action_scope([f"object:{ids.physics}"], action="transparency"), True)
    assert any(state.is_transparent(row) for row in rows)
    state.clear_transparency()
    state.show_all()
    assert not any(state.is_hidden(row) or state.is_transparent(row) for row in rows)


def test_paged_actions_are_range_bounded_and_do_not_materialize_20000_records(monkeypatch):
    root = HierarchyNode("objects", "Objects", children=[f"object:{i}" for i in range(1, 5001)])
    query = HierarchyNode("query", "Raycasts", "PvdRaycast", kind="array",
        source_instance_id=6000, property_id=22, element_count=20000, element_size=64)
    specs = {root.key: root, query.key: query}
    for i in range(1, 5001):
        specs[f"object:{i}"] = HierarchyNode(f"object:{i}", str(i), "PxRigidDynamic", i,
                                           "object", root.key)
    adapter = SimpleNamespace(session=SimpleNamespace(capture_key=1, segment_epoch=7),
                              hierarchy=lambda _: (specs, [root.key, query.key]))
    model = ObjectTreeModel(page_size=64)
    model.populate(adapter)
    page = model._specs[model._specs[root.key].children[0]]
    before = (len(model.cache), len(model._specs))
    monkeypatch.setattr(model, "_item", fail_materialization)
    monkeypatch.setattr(model, "_array_children", fail_materialization)
    scope = model.action_scope([page.key])
    assert scope.native_roots == tuple(range(1, page.element_count + 1))
    assert len(scope.native_roots) < 5000
    assert scope.selected_rows[0].segment_epoch == 7
    query_scope = model.action_scope([query.key])
    assert query_scope.native_roots == () and query_scope.local_keys == (query.key,)
    assert query_scope.selected_rows[0].element_count == 20000
    assert (len(model.cache), len(model._specs)) == before


def test_selected_array_page_restores_only_its_bounded_path_and_invalidates_on_segment():
    query = HierarchyNode("query", "Raycasts", "PvdRaycast", kind="array",
        source_instance_id=6000, property_id=22, element_count=20000, element_size=64)
    adapter = SimpleNamespace(session=SimpleNamespace(capture_key=1, segment_epoch=7),
                              hierarchy=lambda _: ({query.key: query}, [query.key]))
    model = ObjectTreeModel(page_size=64)
    model.populate(adapter)
    page = model.get_item_children(model.roots[0])[0]
    row = model.row_identity(page.key)
    assert row.kind == "array_page" and row.element_index == 0 and row.element_count == 4096
    replacement = ObjectTreeModel(page_size=64)
    replacement.populate(adapter)
    restored = replacement.restore_rows([row])
    assert len(restored) == 1 and restored[0].key == row.key
    assert len(replacement.cache) == 2  # Root + explicit page, no record widgets.
    assert len(replacement._specs) <= 65
    adapter.session.segment_epoch = 8
    replacement.populate(adapter)
    assert replacement.restore_rows([row]) == ()


def test_invalid_action_never_silently_becomes_visibility(tmp_path):
    _, _, model = capture(tmp_path)
    ids = model.fixture_roles
    with pytest.raises(ValueError, match="Unknown row action"):
        model.action_scope([f"object:{ids.actor}"], action="anything")
    state = local_state(model)
    with pytest.raises(ValueError, match="Unknown row action"):
        state.apply(replace(model.action_scope([f"object:{ids.actor}"]), action="anything"), True)

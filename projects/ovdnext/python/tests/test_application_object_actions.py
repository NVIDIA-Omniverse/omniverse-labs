# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real native publications exercise the application boundary for tree actions.

Reuse the presentation suite's widget/renderer sinks; no desktop or GPU is
started. Scopes and row identities come from actual OVD/PXD2 ownership data.
"""
from types import SimpleNamespace

import pytest

from capture_semantics import test_scene_roles, role_key
from ovd_ui.interaction import TransparencyMode
from test_application_presentation import app, open_capture


GROUP = "group:object:{scene}:pxrigiddynamic"
CHILD = "group:object:{actor}:shapes"
ALIAS = "reference:group:object:{shape}:materials:{material}"


def settle(application):
    # Restoring a previously unavailable selection can request its properties
    # in a second publication. Preserve that real application behavior.
    for _ in range(3):
        if not application.presentation_dirty:
            return
        assert application._refresh()
    assert not application.presentation_dirty


def choose(application, *keys):
    if any('{' in key for key in keys):
        ids = test_scene_roles(application.session)
        keys = tuple(role_key(key, ids) for key in keys)
    application._tree_selection([application.object_model._item(key) for key in keys])
    settle(application)


def row_keys(application):
    return tuple(row.key for row in application.row_actions.selected)


def selected_keys(application):
    return tuple(item.key for item in application.tree.selection)


def selection_and_publication(application):
    """State that appearance controls must preserve, including history position."""
    return (tuple(application.selection.get_snapshot().paths()),
            tuple(row.token for row in application.row_actions.selected),
            selected_keys(application), application.interaction.sticky,
            tuple(application.interaction._history), application.interaction._history_index,
            application.presentation.generation, application.frame, application.playback.frame,
            application.presentation_dirty, application.pending_presentation,
            application.property_frame.rebuilds)


def fail_presentation(*_, **__):
    pytest.fail("Appearance-only action requested a native presentation or camera change")


class PresentationForbiddenSession:
    """Keep the real session identity/metadata but detect any fresh frame read."""
    def __init__(self, source):
        self.source = source

    def __getattr__(self, name):
        if name in ("presentation", "seek", "geometry_snapshot"):
            return fail_presentation
        return getattr(self.source, name)


def test_class_bucket_selection_is_one_row_without_native_descendant_selection(app, tmp_path):
    assert app._display(*open_capture(tmp_path, "test_scene.ovd"))
    ids = test_scene_roles(app.session)
    choose(app, role_key(GROUP, ids))
    assert row_keys(app) == selected_keys(app) == (role_key(GROUP, ids),)
    assert app.row_actions.selected[0].instance_id is None
    assert app.selection.get_snapshot().paths() == []
    assert app.interaction.sticky == () and app.renderer.selected == set()
    assert app.presentation.inspected_ids == []
    # The same row resolves a native root only when an action is requested.
    assert app.object_model.action_scope([role_key(GROUP, ids)]).native_roots == (ids.actor,)


@pytest.mark.parametrize("keys", [(GROUP,), (GROUP, "object:{actor}"), (ALIAS,), (GROUP, ALIAS)])
def test_container_mixed_and_reference_rows_survive_frame_replay(app, tmp_path, keys):
    assert app._display(*open_capture(tmp_path, "test_scene.ovd"))
    ids = test_scene_roles(app.session)
    keys = tuple(role_key(key, ids) for key in keys)
    choose(app, *keys)
    identities = tuple(row.token for row in app.row_actions.selected)
    history = tuple(app.interaction._history)
    assert selected_keys(app) == keys
    for frame in (2, 3, 1):
        assert app.seek(frame, snap=False)
        settle(app)
        assert app.frame == app.presentation.frame == frame
        assert tuple(row.token for row in app.row_actions.selected) == identities
        assert tuple(app.interaction._history) == history
        if frame in (1, 2):
            assert selected_keys(app) == keys
    assert selected_keys(app) == keys


@pytest.mark.parametrize("action", ["hide", "show", "tag", "untag"])
def test_frozen_scope_changes_appearance_without_selection_history_or_native_request(
        app, tmp_path, monkeypatch, action):
    assert app._display(*open_capture(tmp_path, "test_scene.ovd"))
    ids = test_scene_roles(app.session)
    domain = "transparency" if action in ("tag", "untag") else "visibility"
    scope = app.object_model.action_scope([role_key(GROUP, ids)], action=domain)
    if action == "show":
        assert app._object_action("hide", scope)
    if action == "untag":
        assert app._object_action("tag", scope)
    # The menu's target remains its original bucket after a different object
    # is selected. It must not reinterpret the current selection at invocation.
    choose(app, f"object:{ids.material}")
    before = selection_and_publication(app)
    geometry = app.renderer.snapshot
    events = []
    subscription = app.selection.subscribe(events.append)
    monkeypatch.setattr(app, "_request_presentation", fail_presentation)
    monkeypatch.setattr(app, "_refresh", fail_presentation)
    monkeypatch.setattr(app, "frame_selected", fail_presentation)
    monkeypatch.setattr(app, "session", PresentationForbiddenSession(app.session))
    assert app._object_action(action, scope)
    assert selection_and_publication(app) == before
    assert app.renderer.snapshot is geometry and not events
    assert subscription is not None
    row = app.object_model.row_identity(role_key(GROUP, ids))
    if domain == "visibility":
        assert app.row_actions.is_hidden(row) == (action == "hide")
        assert (ids.actor in app.interaction.hidden) == (action == "hide")
        assert ids.material not in app.interaction.hidden
    else:
        assert app.row_actions.is_transparent(row) == (action == "tag")
        assert (ids.actor in app.interaction.transparent) == (action == "tag")
        assert ids.material not in app.interaction.transparent
        assert app.interaction.transparency_mode == TransparencyMode.NONE


def test_hide_show_selected_and_child_flags_do_not_change_parent_or_selection(app, tmp_path):
    assert app._display(*open_capture(tmp_path, "test_scene.ovd"))
    ids = test_scene_roles(app.session)
    choose(app, role_key(GROUP, ids))
    before = selection_and_publication(app)
    assert app._object_action("hide")
    assert app._row_tags(app.object_model.row_identity(role_key(GROUP, ids))) == (True, False)
    assert app._row_tags(app.object_model.row_identity(role_key(CHILD, ids))) == (True, False)
    assert {ids.actor, ids.shape, ids.geometry} <= app.interaction.hidden
    child_scope = app.object_model.action_scope([role_key(CHILD, ids)])
    assert app._object_action("show", child_scope)
    assert app._row_tags(app.object_model.row_identity(role_key(CHILD, ids))) == (False, False)
    assert app._row_tags(app.object_model.row_identity(role_key(GROUP, ids))) == (True, False)
    assert ids.actor in app.interaction.hidden and not {ids.shape, ids.geometry} & app.interaction.hidden
    assert selection_and_publication(app) == before
    # Show selected still refers to the bucket, clearing its complete subtree.
    assert app._object_action("show")
    assert not app.interaction.hidden
    assert app._row_tags(app.object_model.row_identity(role_key(GROUP, ids))) == (False, False)
    assert selection_and_publication(app) == before


def test_alias_controls_keep_local_flags_separate_from_target_and_parent(app, tmp_path):
    assert app._display(*open_capture(tmp_path, "test_scene.ovd"))
    ids = test_scene_roles(app.session)
    choose(app, role_key(ALIAS, ids))
    before = selection_and_publication(app)
    assert app.selection.get_snapshot().paths() == [f"/objects/{ids.material}"]
    assert app._object_action("hide") and app._object_action("tag")
    assert app._row_tags(app.object_model.row_identity(role_key(ALIAS, ids))) == (True, True)
    assert app._row_tags(app.object_model.row_identity(f"object:{ids.material}")) == (False, False)
    assert app._row_tags(app.object_model.row_identity(f"group:object:{ids.shape}:materials")) == (False, False)
    assert not app.interaction.hidden and not app.interaction.transparent
    assert selection_and_publication(app) == before
    assert app.seek(2, snap=False)
    settle(app)
    assert selected_keys(app) == (role_key(ALIAS, ids),)
    assert app._row_tags(app.object_model.row_identity(role_key(ALIAS, ids))) == (True, True)


def test_hide_show_all_keep_explicit_mixed_rows_and_clear_native_and_local_flags(app, tmp_path):
    assert app._display(*open_capture(tmp_path, "test_scene.ovd"))
    ids = test_scene_roles(app.session)
    choose(app, role_key(GROUP, ids), role_key(ALIAS, ids))
    before = selection_and_publication(app)
    assert app._object_action("hide_all")
    assert app.interaction.hidden == app.interaction.known
    assert all(app._row_tags(app.object_model.row_identity(key))[0] for key in (role_key(GROUP, ids), role_key(ALIAS, ids)))
    assert app._object_action("show_all")
    assert not app.interaction.hidden
    assert not any(app._row_tags(app.object_model.row_identity(key))[0] for key in (role_key(GROUP, ids), role_key(ALIAS, ids)))
    assert selection_and_publication(app) == before


@pytest.mark.parametrize("source", ["viewport", "find"])
def test_external_selection_replaces_container_and_alias_rows_with_canonical_target(
        app, tmp_path, source):
    assert app._display(*open_capture(tmp_path, "test_scene.ovd"))
    ids = test_scene_roles(app.session)
    choose(app, role_key(GROUP, ids), role_key(ALIAS, ids))
    if source == "viewport":
        app.viewport = SimpleNamespace(screen_position_x=10, screen_position_y=20)
        app.window = SimpleNamespace(focused=True)
        app._mouse_pressed(30, 50, 0, 0)
        picked = []
        app.renderer.pick = lambda x, y: picked.append((x, y)) or ids.actor
        app._mouse_released(30, 50, 0)
        assert picked == [(20, 30)] and app.mouse_anchor is None
    else:
        framed = []
        app.frame_selected = lambda: framed.append(True)
        assert app._find_object(ids.actor)
        assert framed == []  # Find selects without moving the camera.
    settle(app)
    assert row_keys(app) == selected_keys(app) == (f"object:{ids.actor}",)
    assert app.interaction.sticky == (ids.actor,)
    assert app.presentation.inspected_ids == [ids.actor]


def test_history_navigation_replaces_tree_rows_without_creating_another_history_visit(app, tmp_path):
    assert app._display(*open_capture(tmp_path, "test_scene.ovd"))
    ids = test_scene_roles(app.session)
    app.selection.publish([f"/objects/{ids.actor}"], source="viewport")
    choose(app, role_key(GROUP, ids), role_key(ALIAS, ids))
    assert app.interaction.sticky == (ids.material,)
    history = tuple(app.interaction._history)
    app._interaction_action(app.interaction.back)
    settle(app)
    assert row_keys(app) == selected_keys(app) == (f"object:{ids.actor}",)
    app._interaction_action(app.interaction.forward)
    settle(app)
    assert row_keys(app) == selected_keys(app) == (f"object:{ids.material}",)
    assert tuple(app.interaction._history) == history


def test_query_record_and_container_selection_restore_their_frame_owned_metadata(app, tmp_path):
    assert app._display(*open_capture(tmp_path))
    query = app.object_model.canonical_item(-6)
    group = "group:object:6:scenequeries"
    choose(app, group, query.key)
    before = tuple(row.token for row in app.row_actions.selected)
    assert app.selected_record.query_id == -6
    assert app._object_action("hide")
    for frame in (2, 1):
        assert app.seek(frame, snap=False)
        settle(app)
        assert tuple(row.token for row in app.row_actions.selected) == before
        assert selected_keys(app) == (group, query.key)
        assert app.selected_record is app.object_model.canonical_item(-6)
        assert app.selected_record.query_id == -6
        assert app.presentation.frame == frame
        source, prop, index = (app.selected_record.source_instance_id,
                               app.selected_record.property_id, app.selected_record.element_index)
        assert app.presentation.array_record(source, prop, index).fields
        assert app._row_tags(app.object_model.row_identity(query.key)) == (True, False)


@pytest.mark.parametrize("action", ["hide", "show", "tag", "untag"])
def test_menu_scope_for_now_dead_actor_is_rejected_without_changing_existing_tags(
        app, tmp_path, action):
    assert app._display(*open_capture(tmp_path, "test_scene.ovd"))
    ids = test_scene_roles(app.session)
    choose(app, f"object:{ids.actor}")
    scope = app.object_model.action_scope([f"object:{ids.actor}"],
        action="transparency" if action in ("tag", "untag") else "visibility")
    assert app._object_action("hide") and app._object_action("tag")
    assert app.seek(3, snap=False)
    settle(app)
    assert ids.actor not in app.interaction.alive
    before = selection_and_publication(app), app.interaction.snapshot, app.row_actions.revision
    assert not app._object_action(action, scope)
    assert "Object contents changed. Reopen the object menu." in app.status_label.text
    assert (selection_and_publication(app), app.interaction.snapshot, app.row_actions.revision) == before
    # Rejecting an obsolete menu does not erase valid tags authored earlier.
    assert app.seek(1, snap=False)
    settle(app)
    assert ids.actor in app.interaction.hidden and ids.actor in app.interaction.transparent
    assert app._row_tags(app.object_model.row_identity(f"object:{ids.actor}")) == (True, True)


@pytest.mark.parametrize("action", ["hide", "hide_all", "show_all", "tag", "clear_tags"])
def test_reopened_capture_rejects_old_menu_scope_and_clears_row_flags(app, tmp_path, action):
    assert app._display(*open_capture(tmp_path, "test_scene.ovd"))
    ids = test_scene_roles(app.session)
    choose(app, role_key(GROUP, ids), role_key(ALIAS, ids))
    assert app._object_action("hide") and app._object_action("tag")
    scope = app.object_model.action_scope([role_key(GROUP, ids), role_key(ALIAS, ids)],
        action="transparency" if action in ("tag", "clear_tags") else "visibility")
    closed = []
    app.object_browser.close_context_menu = lambda: closed.append(True)
    # Same filename, distinct native session: old menu provenance is invalid.
    assert app._display(*open_capture(tmp_path / "replacement", "test_scene.ovd"))
    assert closed and not app.row_actions.selected
    assert not app.interaction.hidden and not app.interaction.transparent
    assert not any(app._row_tags(app.object_model.row_identity(key)) != (False, False)
                   for key in (role_key(GROUP, ids), role_key(ALIAS, ids)))
    before = selection_and_publication(app)
    assert not app._object_action(action, scope)
    assert "Capture or recording segment changed" in app.status_label.text
    assert selection_and_publication(app) == before
    assert not app.interaction.hidden and not app.interaction.transparent


def test_query_menu_rejects_a_retired_recording_segment_without_selection_or_native_request(
        app, tmp_path, monkeypatch):
    assert app._display(*open_capture(tmp_path))
    query = app.object_model.canonical_item(-6)
    choose(app, query.key)
    scope = app.object_model.action_scope([query.key])
    assert scope.segment_scoped and scope.selected_rows[0].query_id == -6
    # A PXD2 file has no resampling producer. Apply the same epoch notification
    # consumed by _refresh to isolate rejection of its real frame-owned row.
    app.row_actions.sync(app.object_model.capture_key, app.object_model.segment_epoch + 1)
    before = selection_and_publication(app)
    monkeypatch.setattr(app, "_request_presentation", fail_presentation)
    monkeypatch.setattr(app, "_refresh", fail_presentation)
    assert not app._object_action("hide", scope)
    assert "recording segment changed" in app.status_label.text
    assert selection_and_publication(app) == before
    assert not app.interaction.hidden


def test_new_capture_clears_tree_row_intent_and_rejects_the_previous_menu(app, tmp_path):
    assert app._display(*open_capture(tmp_path, "test_scene.ovd"))
    ids = test_scene_roles(app.session)
    choose(app, role_key(GROUP, ids), role_key(ALIAS, ids))
    scope = app.object_model.action_scope([role_key(GROUP, ids)])
    assert app._object_action("hide", scope)
    assert app._display(None, None)
    assert not app.row_actions.selected and not app.object_model._specs
    assert app.selection.get_snapshot().paths() == []
    assert not app.interaction.hidden and app.renderer.snapshot is None
    assert not app._object_action("hide", scope)
    assert "Capture or recording segment changed" in app.status_label.text


@pytest.mark.parametrize("action", ["hide_all", "show_all"])
def test_bulk_menu_rejects_old_segment_even_when_scope_has_only_positive_roots(app, tmp_path, action):
    from ovd_ui.array_settings import ArrayDisplaySettings
    # Keep this fixture deliberately free of transient pages. The historical
    # 10-item default now pages the physics root, which has separate epoch guards.
    app.object_model.set_array_settings(ArrayDisplaySettings(group_size=64))
    assert app._display(*open_capture(tmp_path, "test_scene.ovd"))
    ids = test_scene_roles(app.session)
    assert app._object_action("hide_all")
    scope = app.object_model.action_scope(app.object_model._root_keys)
    assert not scope.segment_scoped  # Visibility stopped before transient scene children.
    # Simulate the publication epoch boundary with the actual native root IDs.
    app.object_model._segment_epoch += 1
    app.row_actions.sync(app.object_model.capture_key, app.object_model.segment_epoch)
    assert app.row_actions.accepts(scope)  # Positive creation-ID scopes themselves remain valid.
    before = (selection_and_publication(app), app.interaction.snapshot, app.row_actions.revision)
    assert not app._object_action(action, scope)
    assert "recording segment changed" in app.status_label.text
    assert (selection_and_publication(app), app.interaction.snapshot, app.row_actions.revision) == before

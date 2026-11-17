# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Browser event/menu callbacks over real native models, without a desktop.

The public callbacks and Application action guards are production methods.
Only TreeView selection storage, deferred scheduling and Menu widgets are sinks.
Menu callbacks remain invocable after destroy to exercise lifecycle guards.
"""
from collections import deque
from types import SimpleNamespace

import pytest

from capture_semantics import test_scene_roles, role_key
from ovd_ui.widgets import object_browser
from ovd_ui.widgets.object_browser import ObjectBrowser
from test_application_presentation import app, open_capture


GROUP = "group:object:{scene}:pxrigiddynamic"
ALIAS = "reference:group:object:{shape}:materials:{material}"


class Deferred:
    def __init__(self): self.pending = deque()
    def call_soon(self, callback, *args): self.pending.append((callback, args))
    def drain(self):
        count = 0
        while self.pending:
            count += 1
            assert count <= 30, "Browser callback did not settle"
            callback, args = self.pending.popleft()
            callback(*args)


class TreeSink:
    def __init__(self, on_selection):
        self._selection = []
        self.on_selection = on_selection
        self.changes = []
        self.expanded = set()
    @property
    def selection(self): return list(self._selection)
    @selection.setter
    def selection(self, items):
        self._selection = list(items)
        self.changes.append(tuple(item.key for item in items))
        self.on_selection(self.selection)
    def clear_selection(self): self.selection = []
    def set_expanded(self, item, expanded, *_):
        if item is None:return
        if expanded: self.expanded.add(item.key)
        else: self.expanded.discard(item.key)
    def is_expanded(self, item): return item.key in self.expanded
    def dirty_widgets(self): pass


class MenuSink:
    def __init__(self, host, title, **_):
        self.host, self.title = host, title
        self.items = {}
        self.position = None
        self.destroy_count = 0
        self.shown = False
        self.shown_changed = None
        host.created.append(self)
    def __enter__(self): self.host.stack.append(self); return self
    def __exit__(self, *_): assert self.host.stack.pop() is self
    def set_shown_changed_fn(self, callback): self.shown_changed = callback
    def show_at(self, x, y):
        self.position = (x, y)
        self.set_shown(True)
    def set_shown(self, shown):
        self.shown = shown
        if self.shown_changed:self.shown_changed(shown)
    def destroy(self): self.destroy_count += 1


class MenuHost:
    def __init__(self): self.created, self.stack = [], []
    def menu(self, title, **kwargs): return MenuSink(self, title, **kwargs)
    def item(self, title, *, triggered_fn, enabled=True, **_):
        item = SimpleNamespace(trigger=triggered_fn, enabled=enabled)
        self.stack[-1].items[title] = item
        return item


@pytest.fixture
def callback_sinks(monkeypatch):
    deferred, menus = Deferred(), MenuHost()
    monkeypatch.setattr(object_browser, "asyncio", SimpleNamespace(get_event_loop=lambda: deferred))
    monkeypatch.setattr(object_browser, "ui", SimpleNamespace(
        Menu=menus.menu, MenuItem=menus.item, Separator=lambda: None, SimpleListModel=lambda _:None))
    return deferred, menus


def attach_browser(application):
    browser = ObjectBrowser.__new__(ObjectBrowser)
    browser.model = application.object_model
    browser.on_selection_changed = application._tree_selection
    browser._changing_group = False
    browser.grouping = SimpleNamespace(model=SimpleNamespace(get_item_value_model=lambda:
        SimpleNamespace(set_value=lambda _:None)))
    browser.on_activate = application.frame_selected
    browser.on_find_object = None
    browser.row_tags = application._row_tags
    browser.actions, browser.focuses, browser.frames = [], [], []
    def action(name, scope):
        accepted = application._object_action(name, scope)
        browser.actions.append((name, scope, accepted))
        return accepted
    browser.on_action = action
    browser.on_focus = lambda: browser.focuses.append(True)
    browser.tree = TreeSink(browser._tree_selection_changed)
    browser._tag_controls = {}
    browser._destroyed = False
    browser._menu_generation = 0
    browser.context_menu = None
    browser.find_dialog = browser.find_window = browser.filter_window = browser.menu = None
    browser.frame = SimpleNamespace(destroy=lambda: None)
    # Layout/search refresh is outside this callback-only test boundary.
    browser.refresh = lambda: None
    application.object_browser, application.tree = browser, browser.tree
    application.navigation = None
    application.renderer.frame_all = lambda **kwargs: browser.frames.append(
        (kwargs, tuple(application.selection.get_snapshot().paths())))
    return browser


def choose(browser, *keys):
    browser.tree.selection = [browser.model._item(key) for key in keys]


def selected(browser): return tuple(item.key for item in browser.tree.selection)


def open_menu(browser, deferred, key, modifiers=0):
    previous = browser.context_menu
    browser.row_pressed(key, 31, 57, 1, modifiers)
    assert browser.context_menu is previous  # Defer selection/menu changes until after drawing.
    deferred.drain()
    assert browser.context_menu is not None
    assert browser.context_menu.position == (31., 57.)
    return browser.context_menu


def test_context_popup_suppresses_rows_and_utilities_until_native_dismissal(
        app, tmp_path, callback_sinks):
    deferred, _ = callback_sinks
    assert app._display(*open_capture(tmp_path, "test_scene.ovd"))
    ids = test_scene_roles(app.session)
    browser = attach_browser(app)
    label, eye = SimpleNamespace(), SimpleNamespace()
    row = browser.model.row_identity(f"object:{ids.actor}")
    browser._set_row_tooltip((row.key, 0), label, f"Recorded body\nPxRigidDynamic · #{ids.actor}")
    browser._tag_tooltip(eye, 2, False, row)
    ordinary = label.tooltip, eye.tooltip
    popup = open_menu(browser, deferred, row.key)
    assert label.tooltip == eye.tooltip == ""
    popup.set_shown(False)  # Public Menu notification for Escape/outside click.
    assert (label.tooltip, eye.tooltip) == ordinary
    assert not browser._row_tooltips_suppressed


def test_popup_retains_latest_tooltip_text_and_suppresses_lazy_rows(
        app, tmp_path, callback_sinks):
    deferred, _ = callback_sinks
    assert app._display(*open_capture(tmp_path, "test_scene.ovd"))
    ids = test_scene_roles(app.session)
    browser = attach_browser(app)
    popup = open_menu(browser, deferred, f"object:{ids.actor}")
    eye, lazy_label = SimpleNamespace(), SimpleNamespace()
    row = browser.model.row_identity(f"object:{ids.actor}")
    browser._tag_tooltip(eye, 2, False, row)
    browser._tag_tooltip(eye, 2, True, row)  # Tag changes while the popup is shown.
    browser._set_row_tooltip((f"object:{ids.shape}", 0), lazy_label, f"Shape · #{ids.shape}")
    assert eye.tooltip == lazy_label.tooltip == ""
    popup.set_shown(False)
    assert eye.tooltip == "Show this subtree" and lazy_label.tooltip == f"Shape · #{ids.shape}"


def test_retired_menu_notifications_cannot_restore_tooltips_over_new_popup(
        app, tmp_path, callback_sinks):
    deferred, _ = callback_sinks
    assert app._display(*open_capture(tmp_path, "test_scene.ovd"))
    ids = test_scene_roles(app.session)
    browser = attach_browser(app)
    label = SimpleNamespace()
    browser._set_row_tooltip((f"object:{ids.actor}", 0), label, f"Body · #{ids.actor}")
    old = open_menu(browser, deferred, f"object:{ids.actor}")
    current = open_menu(browser, deferred, f"object:{ids.shape}")
    old.set_shown(False)
    assert label.tooltip == "" and browser._row_tooltips_suppressed
    browser.close_context_menu()
    assert label.tooltip == f"Body · #{ids.actor}"
    current.set_shown(True)
    assert label.tooltip == f"Body · #{ids.actor}" and not browser._row_tooltips_suppressed
    browser.destroy()
    old.set_shown(False)
    current.set_shown(False)
    assert label.tooltip == "" and not browser._row_tooltips


def test_tooltip_registry_replaces_rebuilt_cells_and_prunes_dead_native_rows(
        app, tmp_path, callback_sinks):
    assert app._display(*open_capture(tmp_path, "test_scene.ovd"))
    ids = test_scene_roles(app.session)
    browser = attach_browser(app)
    for _ in range(100):
        latest = SimpleNamespace()
        browser._set_row_tooltip((f"object:{ids.actor}", 0), latest, f"Body · #{ids.actor}")
    assert len(browser._row_tooltips) == 1
    assert browser._row_tooltips[(f"object:{ids.actor}", 0)][0] is latest
    browser.refresh_tags()
    app.adapter.session = app.session.presentation(frame=3)
    app.adapter.reload()
    app.object_model.populate(app.adapter)
    browser.refresh_tags()
    assert not browser._row_tooltips and latest.tooltip == ""


@pytest.mark.parametrize("kind,filename", [
    ("group", "test_scene.ovd"), ("page", "ue4_shooter.pxd2"),
    ("array", "cloth_physx34.pxd2")])
def test_group_page_and_query_array_double_click_do_not_frame_all(
        app, tmp_path, callback_sinks, kind, filename):
    deferred, _ = callback_sinks
    assert app._display(*open_capture(tmp_path, filename))
    browser = attach_browser(app)
    key = next(key for key, node in browser.model._specs.items() if node.kind == kind)
    choose(browser, key)
    before = selected(browser)
    browser.activate_row(key)
    deferred.drain()
    assert not browser.frames and selected(browser) == before


def test_reference_activation_is_deferred_and_reveals_canonical_target(app, tmp_path, callback_sinks):
    deferred, _ = callback_sinks
    assert app._display(*open_capture(tmp_path, "test_scene.ovd"))
    ids = test_scene_roles(app.session)
    browser = attach_browser(app)
    choose(browser, role_key(ALIAS, ids))
    browser.activate_row(role_key(ALIAS, ids))
    assert selected(browser) == (role_key(ALIAS, ids),) and not browser.frames
    deferred.drain()
    assert selected(browser) == (f"object:{ids.material}",)
    assert browser.frames == [({"selected": True}, (f"/objects/{ids.material}",))]
    assert f"object:{ids.physics}" in browser.tree.expanded


@pytest.mark.parametrize('target_class', ['PxShape', 'PxMaterial'])
def test_current_reference_menu_reveals_without_framing(app, tmp_path, callback_sinks, target_class):
    deferred, _ = callback_sinks
    assert app._display(*open_capture(tmp_path, "test_scene.ovd"))
    ids = test_scene_roles(app.session)
    browser = attach_browser(app)
    alias = next(node for node in browser.model._specs.values()
                 if node.kind == "reference" and node.class_name == target_class)
    choose(browser, alias.key)
    menu = open_menu(browser, deferred, alias.key)
    menu.items["Go to referenced object"].trigger()
    assert not browser.frames
    deferred.drain()
    assert selected(browser) == (f"object:{alias.instance_id}",)
    assert not browser.frames


@pytest.mark.parametrize("column,first,second", [(2, "hide", "show"), (3, "tag", "untag")])
def test_utility_control_targets_clicked_row_without_changing_mixed_selection(
        app, tmp_path, callback_sinks, column, first, second):
    assert app._display(*open_capture(tmp_path, "test_scene.ovd"))
    ids = test_scene_roles(app.session)
    browser = attach_browser(app)
    choose(browser, role_key(GROUP, ids), role_key(ALIAS, ids))
    before = selected(browser), app.interaction.sticky, tuple(app.interaction._history)
    row = browser.model.row_identity(f"object:{ids.shape}")
    browser.toggle_tag(row, column)
    name, scope, accepted = browser.actions[-1]
    assert accepted and name == first and scope.native_roots == (ids.shape,)
    assert tuple(r.key for r in scope.selected_rows) == (f"object:{ids.shape}",)
    browser.toggle_tag(row, column)
    assert browser.actions[-1][0] == second and browser.actions[-1][2]
    assert (selected(browser), app.interaction.sticky, tuple(app.interaction._history)) == before
    assert len(browser.focuses) == 2 and not browser.frames


def test_right_click_selected_row_keeps_mixed_selection_and_original_four_actions(
        app, tmp_path, callback_sinks):
    deferred, _ = callback_sinks
    assert app._display(*open_capture(tmp_path, "test_scene.ovd"))
    ids = test_scene_roles(app.session)
    browser = attach_browser(app)
    choose(browser, role_key(GROUP, ids), f"object:{ids.material}")
    before = len(browser.tree.changes)
    menu = open_menu(browser, deferred, role_key(GROUP, ids))
    assert selected(browser) == (role_key(GROUP, ids), f"object:{ids.material}") and len(browser.tree.changes) == before
    assert list(menu.items) == ["Hide selected", "Show selected", "Hide all", "Show all"]
    menu.items["Hide selected"].trigger()
    _, scope, accepted = browser.actions[-1]
    assert accepted and scope.native_roots == (ids.actor, ids.material)
    assert tuple(r.key for r in scope.selected_rows) == (role_key(GROUP, ids), f"object:{ids.material}")
    assert app.interaction.hidden == {ids.actor, ids.shape, ids.geometry, ids.material}
    assert selected(browser) == (role_key(GROUP, ids), f"object:{ids.material}") and not browser.frames


@pytest.mark.parametrize("modifiers", [0, 1, 2, 3])
def test_right_click_unselected_row_replaces_selection_unless_shift_or_control(
        app, tmp_path, callback_sinks, modifiers):
    deferred, _ = callback_sinks
    assert app._display(*open_capture(tmp_path, "test_scene.ovd"))
    ids = test_scene_roles(app.session)
    browser = attach_browser(app)
    choose(browser, f"object:{ids.material}")
    menu = open_menu(browser, deferred, role_key(GROUP, ids), modifiers)
    expected = (f"object:{ids.material}",) if modifiers else (role_key(GROUP, ids),)
    assert selected(browser) == expected
    menu.items["Hide selected"].trigger()
    _, scope, accepted = browser.actions[-1]
    assert accepted and tuple(r.key for r in scope.selected_rows) == expected
    assert scope.native_roots == ((ids.material,) if modifiers else (ids.actor,))


def test_menu_freezes_targets_across_selection_and_same_alive_frame_changes(app, tmp_path, callback_sinks):
    deferred, _ = callback_sinks
    assert app._display(*open_capture(tmp_path, "test_scene.ovd"))
    ids = test_scene_roles(app.session)
    browser = attach_browser(app)
    choose(browser, role_key(GROUP, ids))
    menu = open_menu(browser, deferred, role_key(GROUP, ids))
    choose(browser, f"object:{ids.material}")
    assert app.seek(2, snap=False)
    before = selected(browser), tuple(app.interaction._history), app.presentation.generation
    menu.items["Hide selected"].trigger()
    assert browser.actions[-1][2]
    assert browser.actions[-1][1].native_roots == (ids.actor,)
    assert app.interaction.hidden == {ids.actor, ids.shape, ids.geometry} and ids.material not in app.interaction.hidden
    assert (selected(browser), tuple(app.interaction._history), app.presentation.generation) == before


def test_open_menu_action_for_dead_actor_is_rejected_by_application_guard(app, tmp_path, callback_sinks):
    deferred, _ = callback_sinks
    assert app._display(*open_capture(tmp_path, "test_scene.ovd"))
    ids = test_scene_roles(app.session)
    browser = attach_browser(app)
    choose(browser, f"object:{ids.actor}")
    menu = open_menu(browser, deferred, f"object:{ids.actor}")
    assert app.seek(3, snap=False)
    assert ids.actor not in app.interaction.alive
    before = app.interaction.snapshot, app.row_actions.revision
    menu.items["Hide selected"].trigger()
    assert browser.actions[-1][2] is False
    assert "Object contents changed" in app.status_label.text
    assert (app.interaction.snapshot, app.row_actions.revision) == before


def test_capture_replacement_destroys_menu_and_retired_callbacks_cannot_touch_new_ids(
        app, tmp_path, callback_sinks):
    deferred, _ = callback_sinks
    assert app._display(*open_capture(tmp_path, "test_scene.ovd"))
    ids = test_scene_roles(app.session)
    browser = attach_browser(app)
    choose(browser, role_key(GROUP, ids))
    menu = open_menu(browser, deferred, role_key(GROUP, ids))
    assert app._display(*open_capture(tmp_path / "other", "test_scene.ovd"))
    assert menu.destroy_count == 1 and browser.context_menu is None
    for item in menu.items.values(): item.trigger()
    deferred.drain()
    assert not browser.actions and not app.interaction.hidden and not browser.frames


@pytest.mark.parametrize("retirement", ["close", "replace", "destroy"])
def test_retired_reference_menu_callbacks_cannot_activate_or_change_selection(
        app, tmp_path, callback_sinks, retirement):
    deferred, _ = callback_sinks
    assert app._display(*open_capture(tmp_path, "test_scene.ovd"))
    ids = test_scene_roles(app.session)
    browser = attach_browser(app)
    choose(browser, role_key(ALIAS, ids))
    menu = open_menu(browser, deferred, role_key(ALIAS, ids))
    assert "Go to referenced object" in menu.items
    if retirement == "replace": open_menu(browser, deferred, role_key(GROUP, ids))
    elif retirement == "close": browser.close_context_menu()
    else: browser.destroy()
    assert menu.destroy_count == 1
    before = selected(browser), app.interaction.snapshot
    for item in menu.items.values(): item.trigger()
    deferred.drain()
    assert not browser.actions and not browser.frames
    assert (selected(browser), app.interaction.snapshot) == before


@pytest.mark.parametrize("queued", ["activate", "context"])
def test_deferred_row_callbacks_do_not_run_after_browser_destroy(app, tmp_path, callback_sinks, queued):
    deferred, menus = callback_sinks
    assert app._display(*open_capture(tmp_path, "test_scene.ovd"))
    ids = test_scene_roles(app.session)
    browser = attach_browser(app)
    choose(browser, role_key(ALIAS, ids))
    if queued == "activate": browser.activate_row(role_key(ALIAS, ids))
    else: browser.row_pressed(role_key(ALIAS, ids), 31, 57, 1, 0)
    assert deferred.pending
    browser.destroy()
    deferred.drain()
    assert not browser.actions and not browser.frames and not menus.created
    assert selected(browser) == (role_key(ALIAS, ids),)


@pytest.mark.parametrize("retirement", ["close", "replace"])
def test_reference_menu_activation_queued_before_retirement_is_also_cancelled(
        app, tmp_path, callback_sinks, retirement):
    deferred, _ = callback_sinks
    assert app._display(*open_capture(tmp_path, "test_scene.ovd"))
    ids = test_scene_roles(app.session)
    browser = attach_browser(app)
    choose(browser, role_key(ALIAS, ids))
    menu = open_menu(browser, deferred, role_key(ALIAS, ids))
    menu.items["Go to referenced object"].trigger()
    assert deferred.pending and not browser.frames
    if retirement == "replace":
        # Open the replacement directly: the old activation is still queued.
        browser.show_context_menu(browser.model.row_identity(role_key(GROUP, ids)), 31, 57)
    else:
        browser.close_context_menu()
    before = selected(browser), app.interaction.snapshot
    deferred.drain()
    assert not browser.frames
    assert (selected(browser), app.interaction.snapshot) == before


def test_row_focus_does_not_take_over_left_click_selection_or_open_middle_click_menu(
        app, tmp_path, callback_sinks):
    deferred, menus = callback_sinks
    assert app._display(*open_capture(tmp_path, "test_scene.ovd"))
    ids = test_scene_roles(app.session)
    browser = attach_browser(app)
    choose(browser, f"object:{ids.material}")
    browser.row_pressed(role_key(GROUP, ids), 31, 57, 0, 0)
    assert browser.focuses == [True]
    browser.row_pressed(role_key(GROUP, ids), 31, 57, 2, 0)
    deferred.drain()
    assert browser.focuses == [True]
    assert selected(browser) == (f"object:{ids.material}",) and not menus.created


def test_deferred_activation_cannot_retarget_identical_ids_in_reopened_capture(app, tmp_path, callback_sinks):
    deferred, _ = callback_sinks
    assert app._display(*open_capture(tmp_path, "test_scene.ovd"))
    ids = test_scene_roles(app.session)
    browser = attach_browser(app)
    choose(browser, role_key(ALIAS, ids))
    browser.activate_row(role_key(ALIAS, ids))
    assert deferred.pending
    assert app._display(*open_capture(tmp_path / "replacement", "test_scene.ovd"))
    assert browser.model.row_identity(role_key(ALIAS, ids)) is not None  # Same row text/ID, different owner.
    before = selected(browser), app.interaction.snapshot
    deferred.drain()
    assert not browser.frames
    assert (selected(browser), app.interaction.snapshot) == before

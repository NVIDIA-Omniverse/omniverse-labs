# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Grouping switches retain explicit native rows across transient tree resets."""
from capture_semantics import test_scene_roles, multi_geom_roles, unique_instance_id, dynamic_geometry_owner
from types import SimpleNamespace
import os
from pathlib import Path
import subprocess
import sys

from gui_runtime import GUI_AVAILABLE
import pytest

from ovd_ui.widgets.object_browser import ObjectBrowser
from test_row_actions import capture


class Tree:
    """Public TreeView selection protocol, including projection-reset feedback."""
    def __init__(self, selection):
        self._selection = list(selection)
        self.notifications = []
        self.expansions = []

    def set_expanded(self, item, expanded, recursive):
        self.expansions.append((item.key if item else None, expanded, recursive))

    def set_selection_changed_fn(self, callback):
        self.callback = callback

    @property
    def selection(self):
        return self._selection

    @selection.setter
    def selection(self, items):
        self._selection = list(items)
        self.notifications.append(tuple(item.key for item in items))
        self.callback(list(items))


def make_browser(model, keys, monkeypatch):
    events = []
    browser = ObjectBrowser.__new__(ObjectBrowser)
    browser.model = model
    browser._changing_group = False
    browser.on_selection_changed = lambda items: events.append(tuple(item.key for item in items))
    browser.tree = Tree(model.restore_rows(model.row_identities(keys)))
    browser.tree.set_selection_changed_fn(browser._tree_selection_changed)
    # Native TreeView can clear selection both when its model changes and when
    # its rows/expansion are refreshed. Exercise both notifications explicitly.
    monkeypatch.setattr(model, "_item_changed", lambda _: setattr(browser.tree, "selection", []))
    browser.refresh = lambda: setattr(browser.tree, "selection", [])
    return browser, events


def grouping(index):
    return SimpleNamespace(get_item_value_model=lambda: SimpleNamespace(as_int=index))


def test_native_multiselection_survives_both_grouping_directions_and_publishes_once(tmp_path, monkeypatch):
    session, adapter, model = capture(tmp_path, "multi_geom.ovd")
    ids=multi_geom_roles(session)
    browser, events = make_browser(model, [f"object:{ids.physics}", f"object:{ids.gpu}"], monkeypatch)
    original = tuple(browser.tree.selection)
    for index in (1, 0):
        events.clear()
        browser.tree.notifications.clear()
        browser._group_changed(grouping(index), None)
        assert events == [(f"object:{ids.physics}", f"object:{ids.gpu}")]
        assert browser.tree.notifications[:2] == [(), ()]
        assert tuple(browser.tree.selection) == original
        assert not browser._changing_group
        assert browser.tree.expansions[0] == (None, True, False)
        assert browser.tree.expansions[1] == (None, True, False)
        assert all(expanded and not recursive for _, expanded, recursive in browser.tree.expansions)
        assert {key for key, _, _ in browser.tree.expansions if key} == {
            ancestor.key for item in original for ancestor in model.ancestors(item)}
        browser.tree.expansions.clear()
    # A later ordinary user click still reaches the application normally.
    browser.tree.selection = [model.canonical_item(ids.sphere)]
    assert events[-1] == (f"object:{ids.sphere}",)


def test_missing_container_and_alias_do_not_select_their_descendants(tmp_path, monkeypatch):
    session, adapter, model = capture(tmp_path)
    ids=test_scene_roles(session)
    keys = [f"group:object:{ids.scene}:pxrigiddynamic", f"reference:group:object:{ids.shape}:materials:{ids.material}"]
    browser, events = make_browser(model, keys, monkeypatch)
    browser._group_changed(grouping(1), None)
    assert events == [()] and browser.tree.selection == []
    assert model.canonical_item(ids.actor) is not None and model.canonical_item(ids.material) is not None


def test_mixed_selection_retains_only_valid_explicit_rows(tmp_path, monkeypatch):
    session, adapter, model = capture(tmp_path)
    ids=test_scene_roles(session)
    browser, events = make_browser(model,
        [f"group:object:{ids.scene}:pxrigiddynamic", f"object:{ids.actor}", f"reference:group:object:{ids.shape}:materials:{ids.material}"], monkeypatch)
    browser._group_changed(grouping(1), None)
    assert events == [(f"object:{ids.actor}",)]
    assert [item.instance_id for item in browser.tree.selection] == [ids.actor]


def test_unchanged_grouping_has_no_selection_churn_and_failure_releases_guard(tmp_path, monkeypatch):
    session, adapter, model = capture(tmp_path)
    ids=test_scene_roles(session)
    browser, events = make_browser(model, [f"object:{ids.actor}"], monkeypatch)
    browser._group_changed(grouping(0), None)
    assert not events and not browser.tree.notifications
    def fail(_):
        raise RuntimeError("Projection unavailable")
    monkeypatch.setattr(model, "set_grouping", fail)
    with pytest.raises(RuntimeError, match="Projection unavailable"):
        browser._group_changed(grouping(1), None)
    assert not browser._changing_group
    browser.tree.selection = [model.canonical_item(ids.material)]
    assert events == [(f"object:{ids.material}",)]


@pytest.mark.skipif(not GUI_AVAILABLE, reason="Requires an OVUI graphics display")
def test_native_tree_retains_selection_after_grouping_reparent_and_draw():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(root / "python/tests/object_browser_grouping_gui.py")],
        cwd=root, env=dict(os.environ, OVUIINSPECT_ENABLED="0"),
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Native tree grouping checks passed" in result.stdout

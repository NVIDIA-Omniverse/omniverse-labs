# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Passive selection composition with real public widgets, without a display.

These tests exercise native selected-state propagation and delegate routing.
Raster seams, alpha and computed layout still require the separate GUI proof.
"""
from capture_semantics import test_scene_roles, role_key, dynamic_geometry_owner
from types import SimpleNamespace

import omni.ui as ui
import pytest

from ovd_ui.models import ObjectItem, ObjectTreeModel
from ovd_ui.widgets.object_browser import ObjectBrowser, _ObjectDelegate, _selection_background
from test_application_presentation import app, open_capture


@pytest.fixture
def widgets(monkeypatch):
    result = []
    for kind in ("Frame", "HStack", "ZStack", "Placer", "Rectangle", "Triangle"):
        original = getattr(ui, kind)
        def record(*args, _original=original, _kind=kind, **kwargs):
            widget = _original(*args, **kwargs)
            result.append((_kind, widget, kwargs))
            return widget
        monkeypatch.setattr(ui, kind, record)
    return result


def rectangles(widgets):
    return [widget for kind, widget, _ in widgets if kind == "Rectangle"]


def assert_passive(widgets):
    for _, widget, _ in widgets:
        assert not widget.opaque_for_mouse_events
        for event in ("pressed", "released", "moved", "double_clicked", "hovered", "wheel"):
            assert not getattr(widget, f"has_mouse_{event}_fn")()


@pytest.mark.parametrize("ends", [{}, {"left": True}, {"right": True}])
def test_native_selection_propagates_through_passive_clipped_pieces(widgets, ends):
    # A native TreeView sets selected on the containing cell Stack. Do the
    # same here; there is no Python per-selection repaint or cached row flag.
    parent = ui.HStack()
    widgets.clear()
    try:
        with parent:
            background = _selection_background(**ends)
        assert background.horizontal_clipping and background.vertical_clipping
        assert_passive(widgets)
        painted = rectangles(widgets)
        assert painted and all(not widget.selected for widget in painted)
        for selected in (True, False, True, False):
            parent.selected = selected
            assert all(widget.selected == selected for widget in painted)
        # The public legacy CornerFlag values are deliberately never passed.
        assert all("corner_flag" not in kwargs for _, _, kwargs in widgets)
    finally:
        parent.destroy()


@pytest.mark.parametrize("actions", [False, True])
@pytest.mark.parametrize("key", ["object:{actor}", "group:object:{scene}:pxrigiddynamic",
                                "reference:group:object:{shape}:materials:{material}"])
def test_final_visible_column_closes_background_including_empty_utilities(app, tmp_path,
                                                                        monkeypatch, widgets,
                                                                        actions, key):
    assert app._display(*open_capture(tmp_path, "test_scene.ovd"))
    ids = test_scene_roles(app.session)
    key = role_key(key, ids)
    item = app.object_model._item(key)
    content, tags = [], []
    view = SimpleNamespace(on_action=(lambda *_: None) if actions else None,
                           build_tag=lambda node, column: tags.append((node, column)))
    delegate = _ObjectDelegate(view)
    # Text fitting remains the original content implementation. The routing
    # test only substitutes its font-dependent drawing, keeping native items.
    monkeypatch.setattr(delegate, "_build_content", lambda *args: content.append(args))
    expected_caps = {0: not actions, 1: False, 2: False, 3: actions}
    for column in range(4):
        parent = ui.HStack()
        widgets.clear()
        try:
            with parent:
                delegate.build_widget(app.object_model, item, column, 2, False)
            painted = rectangles(widgets)
            assert sum(widget.name == "object_selection_cap" for widget in painted) == expected_caps[column]
            assert bool(painted) == (column == 0 or actions and column in (2, 3))
            assert_passive(widgets)
            parent.selected = True
            assert all(widget.selected for widget in painted)
        finally:
            parent.destroy()
    assert len(content) == 1 and content[0][0] is item
    assert tags == ([(item, 2), (item, 3)] if actions else [])
    assert app.selection.get_snapshot().paths() == []


def test_native_branch_control_remains_separate_from_background(app, tmp_path, widgets):
    assert app._display(*open_capture(tmp_path, "test_scene.ovd"))
    ids = test_scene_roles(app.session)
    item = app.object_model._item(role_key("group:object:{scene}:pxrigiddynamic", ids))
    toggled = []
    delegate = _ObjectDelegate(SimpleNamespace(toggle_expanded=toggled.append))
    parent = ui.HStack()
    widgets.clear()
    try:
        with parent:
            delegate.build_branch(app.object_model, item, 0, 3, False)
        chevron, = [widget for kind, widget, _ in widgets if kind == "Triangle"]
        chevron.call_mouse_pressed_fn(0, 0, 1, 0)
        assert not toggled
        chevron.call_mouse_pressed_fn(0, 0, 0, 0)
        assert toggled == [item]
        assert all(not widget.has_mouse_pressed_fn() and not widget.opaque_for_mouse_events
                   for widget in rectangles(widgets))
        parent.selected = True
        assert all(widget.selected for widget in rectangles(widgets))
    finally:
        parent.destroy()


def test_empty_utility_builder_does_not_leave_a_hole_or_drop_the_right_cap(widgets):
    # Lazy cells can survive their previous projection until TreeView rebuilds.
    # The existing build_tag correctly emits no control for an unresolved row.
    view = ObjectBrowser.__new__(ObjectBrowser)
    view.on_action = lambda *_: pytest.fail("Decoration invoked a row action")
    view.model = ObjectTreeModel()
    item = ObjectItem("old-group", "Previous group", "")
    assert view.model.row_identity(item.key) is None
    delegate = _ObjectDelegate(view)
    parent = ui.HStack()
    widgets.clear()
    try:
        with parent:
            delegate.build_widget(view.model, item, 2, 1, False)
            delegate.build_widget(view.model, item, 3, 1, False)
        assert [widget.name for widget in rectangles(widgets)] == [
            "object_selection_middle", "object_selection_middle", "object_selection_cap"]
        assert_passive(widgets)
        parent.selected = True
        assert all(widget.selected for widget in rectangles(widgets))
    finally:
        parent.destroy()

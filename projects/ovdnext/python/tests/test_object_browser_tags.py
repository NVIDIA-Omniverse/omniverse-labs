# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Appearance refresh skips cached row walks during unchanged frame replay."""
from types import SimpleNamespace
from capture_semantics import test_scene_roles

import pytest

from ovd_ui.hierarchy import HierarchyNode
from ovd_ui.models import ObjectTreeModel
from ovd_ui.widgets import object_browser
from ovd_ui.widgets.object_browser import ObjectBrowser
from test_application_presentation import app, open_capture


def browser(model, tags):
    result = ObjectBrowser.__new__(ObjectBrowser)
    result.model, result.row_tags = model, tags
    result._tag_controls = {}
    result._destroyed = False
    return result


def control(browser, key, column=2):
    row = browser.model.row_identity(key)
    hidden, tagged = browser.row_tags(row)
    active = hidden if column == 2 else tagged
    widget = SimpleNamespace(alternate=active)
    browser._tag_controls[(key, column)] = (row, widget, active)
    return widget


def test_unchanged_appearance_skips_1000_cached_controls_and_model_ancestor_walks(monkeypatch):
    specs = {f"object:{i}": HierarchyNode(f"object:{i}", str(i), "PxRigidDynamic", i, "object")
             for i in range(1, 1001)}
    model = ObjectTreeModel()
    model.populate(SimpleNamespace(session=object(), hierarchy=lambda _: (specs, list(specs))))
    reads = []
    view = browser(model, lambda row: (reads.append(row.key) or False, False))
    for key in specs:
        control(view, key)
    signature = (frozenset(), frozenset(), 1)
    view.refresh_tags(signature)
    reads.clear()
    def unexpected(*_):
        pytest.fail("Unchanged tags traversed cached row identities")
    monkeypatch.setattr(model, "row_identity", unexpected)
    for _ in range(100):
        # New immutable containers with equal contents must reuse the cache.
        view.refresh_tags((frozenset(), frozenset(), 1))
    assert not reads and len(view._tag_controls) == 1000


def test_actual_native_tags_refresh_and_model_change_prunes_dead_controls(app, tmp_path):
    assert app._display(*open_capture(tmp_path, "test_scene.ovd"))
    actor = test_scene_roles(app.session).actor
    view = browser(app.object_model, app._row_tags)
    visible = control(view, f"object:{actor}")
    alpha = control(view, f"object:{actor}", 3)
    local = control(view, f"group:object:{actor}:shapes")
    app.object_browser = view
    app._refresh_row_tags()
    assert not visible.alternate and not alpha.alternate and not local.alternate
    app.interaction.set_visible([actor], False)
    app._refresh_row_tags()
    assert visible.alternate and not alpha.alternate
    app.interaction.set_transparent([actor], True)
    app._refresh_row_tags()
    assert alpha.alternate
    scope = app.object_model.action_scope([f"group:object:{actor}:shapes"])
    app.row_actions.apply(scope, False)
    app._refresh_row_tags()
    assert local.alternate
    # Same appearance, different actual native lifetime/projection.
    app.adapter.session = app.session.presentation(frame=3)
    app.adapter.reload()
    app.object_model.populate(app.adapter)
    app._refresh_row_tags()
    assert not view._tag_controls


def test_application_signature_ignores_frame_only_interaction_revision(app, tmp_path, monkeypatch):
    assert app._display(*open_capture(tmp_path, "test_scene.ovd"))
    actor = test_scene_roles(app.session).actor
    reads = []
    view = browser(app.object_model, lambda row: (reads.append(row.key) or app._row_tags(row)[0], False))
    control(view, f"object:{actor}")
    app.object_browser = view
    app._refresh_row_tags()
    reads.clear()
    signature = view._last_tag_signature
    # InteractionState intentionally advances its revision when the frame
    # supplies an ownership graph, even if its presentation tags are identical.
    for _ in range(3):
        app.interaction.update_frame(app.interaction.known, app.interaction.alive,
            children=app.interaction._children, owners=app.interaction._owners)
        app._refresh_row_tags()
    assert not reads and view._last_tag_signature == signature


def test_newly_built_control_reads_tags_even_when_global_signature_is_unchanged(app, tmp_path, monkeypatch):
    assert app._display(*open_capture(tmp_path, "test_scene.ovd"))
    actor = test_scene_roles(app.session).actor
    app.interaction.set_visible([actor], False)
    view = browser(app.object_model, app._row_tags)
    app.object_browser = view
    view.on_action = lambda *_: None
    app._refresh_row_tags()  # Cache established before this lazy row is built.
    class Scope:
        def __enter__(self): return self
        def __exit__(self, *_): pass
    monkeypatch.setattr(object_browser, "ui", SimpleNamespace(
        VStack=lambda **_: Scope(), Spacer=lambda **_: None))
    monkeypatch.setattr(object_browser, "icon_button", lambda *_, **__: SimpleNamespace())
    view.build_tag(app.object_model.canonical_item(actor), 2)
    row, widget, active = view._tag_controls[(f"object:{actor}", 2)]
    assert active and widget.alternate
    app._refresh_row_tags()
    assert widget.alternate


def test_refresh_without_signature_preserves_explicit_force_refresh_api(app, tmp_path):
    assert app._display(*open_capture(tmp_path, "test_scene.ovd"))
    actor = test_scene_roles(app.session).actor
    view = browser(app.object_model, app._row_tags)
    widget = control(view, f"object:{actor}")
    view.refresh_tags()
    app.interaction.set_visible([actor], False)
    view.refresh_tags()
    assert widget.alternate

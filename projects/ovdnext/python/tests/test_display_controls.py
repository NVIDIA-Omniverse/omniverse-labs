# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Visible tree headers and viewport controls use real OVUI widgets/native data."""
import numpy as np
import omni.ui as ui
import pytest

from ovd_ui.interaction import TransparencyMode
from ovd_ui.models import ObjectTreeModel
from ovd_ui.settings import Preferences
from ovd_ui.widgets.object_browser import ObjectBrowser
from test_application_presentation import app, open_capture
from test_visual_reset import OverlayModes


def test_visibility_and_transparency_have_separate_visible_headers(monkeypatch):
    browser = ObjectBrowser(ObjectTreeModel(), lambda _: None, on_action=lambda *_: None)
    host = ui.Frame()
    labels = []
    factory = ui.Label
    def record(*args, **kwargs):
        widget = factory(*args, **kwargs)
        labels.append(widget)
        return widget
    monkeypatch.setattr(ui, "Label", record)
    try:
        assert browser.tree.header_visible
        for column, title, explanation in [(2, "Vis", "subtree"), (3, "Trsp", "Tagged")]:
            with host:
                browser.delegate.build_header(column)
            assert labels[-1].text == title
            assert explanation in labels[-1].tooltip
    finally:
        browser.destroy()
        host.destroy()


@pytest.fixture
def controls(app, tmp_path):
    app.workflow.settings = Preferences(tmp_path / "preferences.json")
    app.overlay_modes = {}
    app.renderer.set_overlays = lambda snapshot: setattr(app.renderer, "overlays", OverlayModes(snapshot))
    assert app._display(*open_capture(tmp_path, "test_scene.ovd"))
    app.renderer.overlays = OverlayModes()
    actor = int(app.renderer.snapshot.object_ids[0])
    app._tree_selection([app.object_model.canonical_item(actor)])
    host = ui.Frame()
    with host:
        app._build_display_controls()
    yield app, actor
    app.transparency_slider.destroy()
    host.destroy()


@pytest.mark.parametrize("mode,tagged,alpha", [
    (TransparencyMode.NONE, True, 1.), (TransparencyMode.ALL, False, .25),
    (TransparencyMode.SELECTED, False, .25), (TransparencyMode.UNSELECTED, True, 1.),
    (TransparencyMode.TAGGED, False, 1.), (TransparencyMode.TAGGED, True, .25)])
def test_viewport_controls_apply_mode_without_changing_visibility_or_selection(controls, mode, tagged, alpha):
    app, actor = controls
    before = app.interaction.selected, app.presentation, app.frame
    scope = app.object_model.action_scope([f"object:{actor}"], action="transparency")
    assert app._object_action("tag" if tagged else "untag", scope)
    app.transparency_controls["transparency"].set_value(int(mode))
    app.transparency_controls["amount"].set_value(75.)
    assert app.interaction.transparency_mode == mode
    assert app.interaction.transparency_amount == .75
    np.testing.assert_allclose(app.renderer.instance_alpha, alpha)
    assert not app.renderer.hidden
    assert (app.interaction.selected, app.presentation, app.frame) == before
    assert Preferences(app.workflow.settings.path).get("viewport_display")["transparency_amount"] == .75


def test_viewport_controls_survive_options_close_and_restore_failed_changes(controls, monkeypatch):
    app, _ = controls
    app._destroy_view_options()
    assert app._set_transparency(TransparencyMode.TAGGED, .25)
    assert app.transparency_controls["transparency"].as_int == int(TransparencyMode.TAGGED)
    assert app.transparency_controls["amount"].as_float == 25.
    import ovd_ui.settings as settings
    def denied(*_): raise OSError("disk denied")
    monkeypatch.setattr(settings.os, "replace", denied)
    app.transparency_controls["amount"].set_value(90.)
    assert app.transparency_controls["amount"].as_float == 25.
    assert app.interaction.transparency_amount == .25

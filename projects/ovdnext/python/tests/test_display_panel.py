# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Embedded Display retains controls and respects the inspector's available space."""
from types import SimpleNamespace

import numpy as np
import omni.ui as ui
import pytest

from ovd_ui.camera import Camera, CameraController
from ovd_ui.settings import Preferences
from test_display_controls import app, controls


@pytest.fixture
def panel(controls):
    app, actor = controls
    app.window = ui.Window("Display panel test", width=1000, height=800)
    app.navigation = CameraController(Camera())
    app.inspector_column = SimpleNamespace(computed_height=700)
    app.palette_view = SimpleNamespace(show=lambda: None)
    host = ui.Frame()
    with host:
        app._build_display_panel()
    yield app, actor
    app._destroy_view_options()
    host.destroy()
    app.window.destroy()


def test_display_toggle_is_deferred_and_reuses_controls(panel):
    app, _ = panel
    before = app.presentation, app.interaction.selected, app.frame
    assert not app.view_panel.visible and not app.view_option_controls
    app.show_view_options()
    assert not app.view_panel.visible
    app._update_split_grips()
    assert app.view_panel.visible and ui.Workspace.get_window("Viewport options") is None
    models = app.view_option_controls.copy()
    models["gizmo_scale"].set_value(2.)
    app.show_view_options()
    app._update_split_grips()
    assert not app.view_panel.visible
    assert app._set_gizmo_scale(3.)
    app.show_view_options()
    app._update_split_grips()
    assert app.view_panel.visible and app.view_option_controls == models
    assert models["gizmo_scale"].as_float == 3.
    assert (app.presentation, app.interaction.selected, app.frame) == before
    assert Preferences(app.workflow.settings.path).get("soft_glass_display_open") is True


def test_display_height_shrinks_without_overwriting_preferred_height(panel):
    app, _ = panel
    app.show_view_options()
    app._update_split_grips()
    app._view_resized(400)
    app._update_split_grips()
    assert float(app.view_panel.height) == 400
    app.inspector_column.computed_height = 400
    app._update_split_grips()
    assert float(app.view_panel.height) <= 220
    assert app.workflow.settings.get("soft_glass_display_height") == 400
    app.inspector_column.computed_height = 700
    app._update_split_grips()
    assert float(app.view_panel.height) == 400


def test_contact_dropdown_applies_and_retains_selection_modes(panel):
    app, _ = panel
    app.show_view_options()
    app._update_split_grips()
    contact=app.view_option_controls[32]
    contact.set_value(1)
    assert app._refresh()
    owned=app.presentation
    for mode in (2,3,1):
        contact.set_value(mode)
        assert contact.as_int==mode==app._overlay_mode(32)
        assert Preferences(app.workflow.settings.path).get('viewport_display')['contacts']==mode
        assert app.presentation is owned and not app.presentation_dirty
    app.show_view_options()
    app._update_split_grips()
    app.show_view_options()
    app._update_split_grips()
    assert app.view_option_controls[32] is contact and contact.as_int==1


def test_family_controls_are_independent_and_new_modes_are_dropdowns(panel):
    app, _ = panel
    app.show_view_options()
    app._update_split_grips()
    controls=app.view_option_controls
    before=app._viewport_settings().scales
    controls['linear_velocity_scale'].set_value(7.)
    assert app._viewport_settings().scales=={**before,4:7.}
    assert app.overlay_scales[4]==7.
    for bit in (16,256):
        for mode in (2,3,0,1):
            controls[bit].set_value(mode)
            assert controls[bit].as_int==mode==app._overlay_mode(bit)
    assert Preferences(app.workflow.settings.path).get('viewport_display')['linear_velocity_scale']==7.
    assert app._set_viewport_settings(com_scale=.5)
    assert controls['com_scale'].as_float==.5 and controls['linear_velocity_scale'].as_float==7.


@pytest.mark.parametrize('name',['axes_scale','linear_velocity_scale','angular_velocity_scale','contact_scale','com_scale'])
def test_family_fields_accept_the_native_float_minimum_and_restore_invalid_values(panel,name):
    app, _ = panel
    app.show_view_options();app._update_split_grips()
    model=app.view_option_controls[name]
    model.set_value(float(np.float32(.0001)))
    assert getattr(app._viewport_settings(),name)==.0001
    accepted=app._viewport_settings()
    model.set_value(float('nan'))
    assert app._viewport_settings()==accepted
    assert model.as_float==pytest.approx(.0001)


def test_master_and_family_log_sliders_keep_other_controls_and_saved_values(panel,monkeypatch):
    app, _ = panel
    app.show_view_options();app._update_split_grips()
    models=app.view_option_controls
    sliders={s.model: s for s in app.view_sliders}
    assert 'master_scale' in models
    master=sliders[models['master_scale']]
    family=sliders[models['axes_scale']]
    assert master.position_model.as_float==.5 and family.position_model.as_float==.5
    master.position_model.set_value(1.)
    family.position_model.set_value(0.)
    assert app._viewport_settings().master_scale==pytest.approx(20.)
    assert app.overlay_scales[2]==pytest.approx(.05)
    assert app.overlay_scales[4]==1. and app._overlay_mode(2)==0
    master.position_model.set_value(.501)
    assert app._viewport_settings().master_scale==1.
    before=app._viewport_settings()
    def denied(*_):raise OSError('preferences read-only')
    monkeypatch.setattr('ovd_ui.settings.os.replace',denied)
    master.position_model.set_value(.8)
    assert app._viewport_settings()==before and master.position_model.as_float==.5
    saved=Preferences(app.workflow.settings.path).get('viewport_display')
    assert saved['master_scale']==1. and saved['axes_scale']==pytest.approx(.05)


def test_display_teardown_cancels_pending_resize_and_rebuild_restores_preferences(panel):
    app, _ = panel
    app.show_view_options()
    app._update_split_grips()
    app._view_resized(360)
    grip, slider = app.view_grip, app.view_sliders[0]
    grip.press(0, 100, 0, 0)
    grip.release(0, 50, 0)
    app._destroy_view_options()
    assert grip.closed and slider.closed
    app._update_split_grips()
    assert app.workflow.settings.get("soft_glass_display_height") == 360
    with ui.Frame():
        app._build_display_panel()
    app._update_split_grips()
    assert app.view_panel.visible and float(app.view_panel.height) == 360


@pytest.mark.parametrize("saved", [True, "wide", float("nan"), float("inf"), 10**400],
                         ids=["boolean", "text", "nan", "infinite", "huge"])
def test_invalid_saved_display_height_uses_a_bounded_default(panel, saved):
    app, _ = panel
    app._destroy_view_options()
    app.workflow.settings.values["soft_glass_display_height"] = saved
    with ui.Frame():
        app._build_display_panel()
    assert float(app.view_panel.height) == 320


def test_unwritable_preferences_do_not_disable_display_controls(panel, monkeypatch):
    app, _ = panel
    def denied(*_):raise OSError("preferences read-only")
    monkeypatch.setattr("ovd_ui.settings.os.replace", denied)
    app.show_view_options()
    app._update_split_grips()
    assert app.view_panel.visible and app.view_option_controls
    app._view_resized(360)
    app._update_split_grips()
    assert float(app.view_panel.height) == 360
    assert "preferences read-only" in app.error_message

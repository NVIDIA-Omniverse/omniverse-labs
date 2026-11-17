# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Toolbar and dialog share rendering state, including failed persistence."""
from types import SimpleNamespace

import omni.ui as ui

from ovd_ui.application import Application
from ovd_ui.render_settings import RenderingSettings, DisplayMode, FaceCulling
from ovd_ui.widgets.render_settings_view import RenderingSettingsView


def application(*, fail_save=False):
    app=Application.__new__(Application)
    app.syncing_render_controls=False
    app.render_settings=RenderingSettings()
    applied=[]
    app.renderer=SimpleNamespace(wireframe=True,set_settings=applied.append)
    saved=[]
    def save(key,value):
        if fail_save:raise OSError("Read-only preferences")
        saved.append((key,value))
    app.workflow=SimpleNamespace(settings=SimpleNamespace(set=save))
    app.render_settings_view=RenderingSettingsView(app.render_settings,app._render_settings_changed,
        preferences=app.workflow.settings)
    app.status_label=SimpleNamespace(text="")
    app.error_message=""
    class ReactiveModel:
        value=0
        def set_value(self,value):
            self.value=value
            app._set_wireframe(value)
    app.shading_control=SimpleNamespace(model=ReactiveModel())
    return app,applied,saved


def dialog_controls(view):
    """Real OVUI value models without a competing native Inspector window."""
    choices={"face_culling","planes","sleeping_objects","trigger_shapes","wireframe","background_mode"}
    for key,value in view.settings.to_dict().items():
        if key in choices:
            model=ui.SimpleIntModel()
            model=SimpleNamespace(get_item_value_model=lambda value=model:value)
        elif isinstance(value,bool):model=ui.SimpleBoolModel()
        elif isinstance(value,str):model=ui.SimpleStringModel()
        else:model=ui.SimpleFloatModel()
        view.controls[key]=SimpleNamespace(model=model)
    view.status=SimpleNamespace(text="")
    view._sync()
    return view.controls


def test_partial_dialog_mode_and_toolbar_share_one_canonical_setting():
    app,applied,saved=application()
    partial=RenderingSettings(wireframe=DisplayMode.SELECTED)
    app._render_settings_changed(partial)
    assert app.shading_control.model.value==-1
    assert app.renderer.wireframe is False
    assert applied[-1]==app.render_settings_view.settings==partial
    assert saved==[]
    app.shading_control.model.set_value(0)
    assert app.render_settings.wireframe==DisplayMode.NONE
    assert app.shading_control.model.value==0
    assert applied[-1]==app.render_settings_view.settings==app.render_settings
    assert saved[-1][1]["wireframe"]==DisplayMode.NONE


def test_failed_toolbar_preference_write_restores_the_displayed_mode():
    app,applied,saved=application(fail_save=True)
    app.shading_control.model.set_value(1)
    assert app.shading_control.model.value==0
    assert app.render_settings.wireframe==DisplayMode.NONE
    assert applied[-1]==app.render_settings
    assert saved==[]
    assert "Read-only preferences" in app.error_message==app.status_label.text


def test_toolbar_write_preserves_a_new_explicit_normal_culling_choice():
    app,applied,saved=application()
    app._render_settings_changed(RenderingSettings(face_culling=FaceCulling.NORMAL))
    app._set_wireframe(1)
    restored=RenderingSettings.from_preferences(saved[-1][1])
    assert restored.face_culling==1 and restored.wireframe==DisplayMode.ALL


def test_open_dialog_apply_preserves_toolbar_wireframe_and_other_drafts():
    app,applied,saved=application()
    view=app.render_settings_view
    controls=dialog_controls(view)
    controls["face_culling"].model.get_item_value_model().set_value(1)
    controls["anti_aliasing"].model.set_value(True)
    controls["geometry_scale"].model.set_value(2.5)
    controls["background_color"].model.set_value("invalid color")
    view._apply()
    assert view._status_error and saved==[]
    error=view.status.text

    app.shading_control.model.set_value(1)
    assert controls["wireframe"].model.get_item_value_model().as_int==DisplayMode.ALL
    assert controls["anti_aliasing"].model.as_bool
    assert controls["geometry_scale"].model.as_float==2.5
    assert controls["background_color"].model.as_string=="invalid color"
    assert view._status_error and view.status.text==error

    controls["background_color"].model.set_value("#112233")
    view._apply()
    assert not view._status_error
    assert app.render_settings.wireframe==DisplayMode.ALL
    assert app.render_settings.anti_aliasing and app.render_settings.geometry_scale==2.5
    assert applied[-1]==view.settings==app.render_settings
    assert saved[-1][1]==app.render_settings.to_preferences()
    assert RenderingSettings.from_preferences(saved[-1][1]).face_culling==1


def test_failed_toolbar_write_does_not_replace_an_open_dialog_draft():
    app,applied,saved=application(fail_save=True)
    controls=dialog_controls(app.render_settings_view)
    wireframe=controls["wireframe"].model.get_item_value_model()
    wireframe.set_value(DisplayMode.UNSELECTED)
    controls["anti_aliasing"].model.set_value(True)
    app.shading_control.model.set_value(1)
    assert wireframe.as_int==DisplayMode.UNSELECTED
    assert controls["anti_aliasing"].model.as_bool
    assert app.render_settings.wireframe==DisplayMode.NONE
    assert saved==[]

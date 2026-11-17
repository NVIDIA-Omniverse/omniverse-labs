# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Atomic General Settings acceptance using real preferences and OVUI models."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
import omni.ui as ui

from ovd_ui.array_settings import ArrayDisplaySettings
from ovd_ui.formatting import ValueFormat
from ovd_ui.general_settings import GeneralSettings, integer_text
from ovd_ui.render_settings import RenderingSettings, DisplayMode, FaceCulling
from ovd_ui.settings import Preferences
from ovd_ui.widgets.general_settings_view import GeneralSettingsView
from ovd_ui.widgets.palette_view import PaletteView


def make_view(tmp_path, candidate=GeneralSettings()):
    preferences = Preferences(tmp_path / "preferences.json")
    preferences.set_many({"future_setting": "keep", "array_window": 9, "theme": "light"})
    candidate.save(preferences)
    applied = []
    def accept(value):
        # Callback must see the whole durable candidate, never an earlier field.
        assert GeneralSettings.from_preferences(Preferences(preferences.path)) == value
        applied.append(value)
    view = GeneralSettingsView(preferences, accept)
    choices = {"face_culling", "planes", "sleeping_objects", "trigger_shapes", "wireframe", "background_mode"}
    for key, value in candidate.rendering.to_dict().items():
        if key in choices:
            model = ui.SimpleIntModel()
            model = SimpleNamespace(get_item_value_model=lambda m=model: m)
        elif isinstance(value, bool): model = ui.SimpleBoolModel()
        elif isinstance(value, str): model = ui.SimpleStringModel()
        else: model = ui.SimpleFloatModel()
        view.controls[key] = SimpleNamespace(model=model)
    for key in ("float_precision", "array_group_size", "max_array_to_load"):
        view.property_controls[key] = SimpleNamespace(model=ui.SimpleStringModel())
    choice = ui.SimpleIntModel()
    view.property_controls["float_format"] = SimpleNamespace(model=SimpleNamespace(get_item_value_model=lambda: choice))
    view.property_controls["show_hex"] = SimpleNamespace(model=ui.SimpleBoolModel())
    view.profile_color = SimpleNamespace(model=ui.SimpleStringModel())
    view.window = SimpleNamespace(visible=True)
    view.status = SimpleNamespace(text="")
    view._sync()
    return view, preferences, applied


@pytest.mark.parametrize("precision", (-1, 26, True, 3.5, "6"))
def test_invalid_precision_is_not_truncated(precision):
    with pytest.raises(ValueError):
        GeneralSettings(value_format=ValueFormat(precision=precision))


@pytest.mark.parametrize("text", ("2.5", "2e1", "-0", "", "１０", "true"))
def test_decimal_drafts_reject_noninteger_input(text):
    with pytest.raises(ValueError): integer_text(text, "Value")


def test_draft_accepts_bounded_leading_zeros_without_float_conversion():
    assert integer_text("+" + "0" * 5000 + "99999", "Value") == 99999


def test_read_old_preferences_migrates_in_memory_and_recovers_invalid_fields(tmp_path):
    p = Preferences(tmp_path / "preferences.json")
    p.set_many({"array_window": 9, "float_precision": 25, "ProfileFunctionColor": "green"})
    before = p.path.read_bytes()
    value = GeneralSettings.from_preferences(p)
    assert value.arrays.group_size == 10 and value.value_format.precision == 25
    assert p.get("array_group_size") is None and p.get("array_window") == 9
    assert p.path.read_bytes() == before
    p.set_many({"array_group_size": 0, "ProfileFunctionColor": False})
    value = GeneralSettings.from_preferences(p)
    assert value.arrays == ArrayDisplaySettings()
    assert value.profile_function_color == "#00FF00"
    assert value.value_format.precision == 25


@pytest.mark.parametrize("saved,expected", [(1, 0), (2, 2), (0, 0)])
def test_old_rendering_preferences_migrate_culling_without_resetting_other_settings(tmp_path, saved, expected):
    preferences = Preferences(tmp_path / "preferences.json")
    # Literal v0.2.9 payload: every rendering write included the implicit Normal.
    preferences.set_many({"rendering": {
        "anti_aliasing": True, "face_culling": saved, "geometry_scale": 2.,
        "planes": 0, "plane_scale": 1., "particle_size": 5., "show_particles": True,
        "draw_orbit_center": False, "sleeping_objects": 1, "trigger_shapes": 1,
        "wireframe": 3, "background_mode": "custom", "background_color": "#123456",
    }, "future_setting": "keep"})
    before = preferences.path.read_bytes()
    loaded = GeneralSettings.from_preferences(Preferences(preferences.path))
    assert loaded.rendering.face_culling == expected
    assert loaded.rendering.geometry_scale == 2. and loaded.rendering.anti_aliasing
    assert loaded.rendering.wireframe == 3 and loaded.rendering.background_color == "#123456"
    assert preferences.path.read_bytes() == before  # Reading alone never rewrites preferences.
    loaded.save(preferences)
    assert GeneralSettings.from_preferences(Preferences(preferences.path)) == loaded
    assert preferences.get("future_setting") == "keep"
    chosen = replace(loaded, rendering=replace(loaded.rendering, face_culling=FaceCulling.NORMAL))
    chosen.save(preferences)
    assert GeneralSettings.from_preferences(Preferences(preferences.path)).rendering.face_culling == 1


@pytest.mark.parametrize("schema", [2, 3])
def test_marked_normal_culling_is_never_treated_as_an_old_implicit_default(schema):
    value = GeneralSettings.from_preferences({"rendering": {"schema": schema, "face_culling": 1}})
    assert value.rendering.face_culling == 1


def test_defaults_changes_only_draft_cancel_preserves_all_accepted_values(tmp_path):
    candidate = GeneralSettings(RenderingSettings(geometry_scale=2.5, draw_orbit_center=True),
        ArrayDisplaySettings(9999, 0), ValueFormat(25, "scientific", True), "#123456")
    view, p, applied = make_view(tmp_path, candidate)
    before = p.path.read_bytes()
    view._defaults()
    assert view.accepted == candidate and view.settings == candidate.rendering
    assert view._candidate() == GeneralSettings()
    assert p.path.read_bytes() == before and applied == []
    view.close()
    assert not view.window.visible
    assert GeneralSettings.from_preferences(p) == candidate and applied == []


def test_apply_commits_all_tabs_once_and_preserves_unrelated_preferences(tmp_path, monkeypatch):
    view, p, applied = make_view(tmp_path)
    candidate = GeneralSettings(RenderingSettings(geometry_scale=2.5, draw_orbit_center=True),
        ArrayDisplaySettings(9999, 99999), ValueFormat(25, "scientific", True), "#123456")
    writes = []
    save = p.set_many
    def record(values):
        writes.append(values)
        save(values)
    monkeypatch.setattr(p, "set_many", record)
    view._fill(candidate)
    assert view._apply()
    assert view.accepted == candidate and applied == [candidate]
    assert writes == [candidate.to_preferences()]
    assert p.get("future_setting") == "keep" and p.get("theme") == "light"
    assert p.get("array_window") == 9


def test_validation_and_failed_write_keep_entire_draft_and_applied_state(tmp_path, monkeypatch):
    view, p, applied = make_view(tmp_path)
    before = p.path.read_bytes()
    view.controls["anti_aliasing"].model.set_value(True)
    view.property_controls["float_precision"].model.set_value("25")
    view.property_controls["array_group_size"].model.set_value("10.5")
    view.profile_color.model.set_value("#123456")
    assert not view._apply() and view._status_error
    assert "whole decimal" in view.status.text
    assert p.path.read_bytes() == before and applied == []
    view.property_controls["array_group_size"].model.set_value("20")
    def reject(_): raise OSError("destination is read-only")
    monkeypatch.setattr(p, "set_many", reject)
    assert not view._apply()
    assert "read-only" in view.status.text
    assert view.accepted == GeneralSettings()
    assert view._candidate().rendering.anti_aliasing
    assert view._candidate().value_format.precision == 25
    assert p.path.read_bytes() == before and applied == []


def test_toolbar_and_profile_updates_preserve_other_drafts_and_error(tmp_path):
    view, p, applied = make_view(tmp_path)
    view.property_controls["float_precision"].model.set_value("25")
    view.controls["background_color"].model.set_value("invalid color")
    assert not view._apply()
    message = view.status.text
    rendering = replace(view.accepted.rendering, wireframe=DisplayMode.ALL)
    p.set_many({"rendering": rendering.to_dict(), "ProfileFunctionColor": "#123456"})
    view.set_settings(rendering, fields=("wireframe",))
    view.set_profile_color("#123456")
    assert view.status.text == message and view._status_error
    assert view.property_controls["float_precision"].model.as_string == "25"
    assert view.controls["background_color"].model.as_string == "invalid color"
    view.controls["background_color"].model.set_value("#CCCCCC")
    assert view._apply()
    assert applied[-1].rendering.wireframe == DisplayMode.ALL
    assert applied[-1].profile_function_color == "#123456"
    assert applied[-1].value_format.precision == 25


def test_unrelated_palette_notification_keeps_unsaved_profile_draft(tmp_path):
    view, p, applied = make_view(tmp_path)
    view.profile_color.model.set_value("#123456")
    view.property_controls["float_precision"].model.set_value("invalid")
    assert not view._apply()
    error = view.status.text
    # Actor/gizmo edits and palette imports report the unchanged durable profile
    # value through the same callback. They must not overwrite another draft.
    view.set_profile_color(p.get("ProfileFunctionColor"))
    assert view.profile_color.model.as_string == "#123456"
    assert view._status_error and view.status.text == error
    assert view.accepted.profile_function_color == "#00FF00"
    assert applied == []


def test_external_profile_acceptance_updates_only_unedited_palette_control(tmp_path):
    p = Preferences(tmp_path / "preferences.json")
    # No palette read/reload is needed for the separately stored profile value.
    palette = PaletteView(None, p)
    channels = [ui.SimpleFloatModel() for _ in range(4)]
    picker = SimpleNamespace(model=SimpleNamespace(
        get_item_children=lambda: range(4), get_item_value_model=lambda item: channels[item]))
    field = SimpleNamespace(model=ui.SimpleStringModel())
    identity = ("profile", "ProfileFunctionColor")
    palette.controls[identity] = field, picker
    palette.sync_profile_color("#123456")
    assert field.model.as_string == "#123456"
    assert [m.as_float for m in channels] == pytest.approx([0x12/255, 0x34/255, 0x56/255, 1.])
    palette.pending[identity] = "#ABCDEF"
    field.model.set_value("#ABCDEF")
    palette.sync_profile_color("#987654")
    assert field.model.as_string == palette.pending[identity] == "#ABCDEF"

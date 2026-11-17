# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""General Settings adoption against real native captures and UI-only sinks.

The budget cases inject an exception at the Session boundary. Successful
publications, owned arrays, schema, geometry and lifetimes come from Session;
these tests do not synthesize a native out-of-memory capture or run a GPU.
"""
from collections import deque
from dataclasses import replace
import numpy as np
import pytest

from ovd_ui.array_settings import ArrayDisplaySettings
from ovd_ui.camera import Camera, CameraController
from ovd_ui.formatting import ValueFormat
from ovd_ui.general_settings import GeneralSettings
from ovd_ui.settings import Preferences
from ovd_ui.widgets.general_settings_view import GeneralSettingsView
from ovd_ui.widgets.object_browser import ObjectBrowser
from ovd_ui.widgets.palette_view import PaletteView

from test_application_presentation import app as base_app, open_capture


BUDGET_ERROR = "Presentation metadata/inspector data exceeds 128 MiB"


class ObservedSession:
    """Delegate every successful read; fail only explicitly queued calls."""
    def __init__(self, source):
        self.source = source
        self.calls = []
        self.failures = deque()

    def __getattr__(self, name):
        return getattr(self.source, name)

    def presentation(self, *args, **kwargs):
        self.calls.append((args, dict(kwargs)))
        if self.failures:
            failure = self.failures.popleft()
            if failure is not None:
                raise failure
        return self.source.presentation(*args, **kwargs)


@pytest.fixture
def app(base_app, tmp_path):
    app = base_app
    preferences = Preferences(tmp_path / "preferences.json")
    initial = GeneralSettings(arrays=ArrayDisplaySettings(10, 2000))
    initial.save(preferences)
    app.workflow.settings = preferences
    app.render_settings_view = GeneralSettingsView(preferences, app._general_settings_changed)
    app.render_settings = initial.rendering
    app.renderer.set_settings = lambda settings: setattr(app.renderer, "settings", settings)
    app.analysis_views = []
    app.palette_view = PaletteView(None, preferences)
    app.property_inspector.preferences = preferences
    app.property_inspector.set_array_settings(initial.arrays)
    app.object_model.set_array_settings(initial.arrays)
    app.navigation = CameraController(Camera())
    # Exercise the real reattachment method; only native widget surfaces are
    # sinks, as in the existing Application publication test fixture.
    browser = app.object_browser
    browser.model = app.object_model
    browser.tree = app.tree
    browser._changing_group = False
    browser.closed_menus = 0
    browser.close_context_menu = lambda: setattr(browser, "closed_menus", browser.closed_menus + 1)
    browser.restore_projection_selection = lambda rows: ObjectBrowser.restore_projection_selection(browser, rows)
    yield app
    app.selection_subscription = None


def opened(app, tmp_path, name="cloth_physx34.pxd2"):
    path, source = open_capture(tmp_path, name)
    observed = ObservedSession(source)
    assert app._display(path, observed)
    observed.calls.clear()
    return observed


def stable_state(app):
    """Actual displayed ownership, playhead/clip and camera; no timing fields."""
    return (app.session.capture_key, app.scene_id, app.frame,
            app.presentation.frame, app.playback.frame, app.playback.index,
            tuple((clip.name, clip.start, clip.stop) for clip in app.playback.clips),
            tuple(app.selection.get_snapshot().paths()), app.navigation.state())


def accept(app, candidate):
    # Match GeneralSettingsView's durable transaction-before-adoption contract.
    candidate.save(app.workflow.settings)
    app._general_settings_changed(candidate)


def test_repage_keeps_native_alias_query_selection_camera_clip_and_owned_record(app, tmp_path):
    source = opened(app, tmp_path)
    assert app.seek(30, snap=False)
    app._clip_action(app.playback.trim_end)
    assert app.seek(1, snap=False)
    model = app.object_model
    native = model.canonical_item(15)
    alias = model._item(next(key for key, node in model._specs.items()
                             if node.kind == "reference" and node.instance_id is not None))
    record = model.canonical_item(-6)
    app._tree_selection([native, alias, record])
    assert app._refresh()
    rows = model.row_identities([native.key, alias.key, record.key])
    assert len(rows) == 3 and rows[-1].query_id == -6
    before = stable_state(app)
    old_view = app.presentation
    old_record = old_view.array_record(record.source_instance_id, record.property_id, record.element_index)
    assert old_record.fields
    native_keys = {key for key, node in model._specs.items() if node.kind == "object"}
    closed = app.object_browser.closed_menus
    source.calls.clear()
    candidate = replace(GeneralSettings.from_preferences(app.workflow.settings),
                        arrays=ArrayDisplaySettings(20, 0))
    accept(app, candidate)
    assert source.calls == [] and app.presentation is old_view
    assert app.presentation_dirty and stable_state(app) == before
    assert [row.token for row in app.row_actions.selected] == [row.token for row in rows]
    assert [item.key for item in app.tree.selection] == [row.key for row in rows]
    assert app.selected_record.query_id == -6
    assert app.object_browser.closed_menus == closed + 1
    assert not app.selection_guard and not app.object_browser._changing_group
    assert {key for key, node in model._specs.items() if node.kind == "object"} == native_keys
    assert app._refresh() and len(source.calls) == 1
    assert stable_state(app) == before
    assert app.presentation.array_record(record.source_instance_id, record.property_id, record.element_index).fields
    assert old_view.array_record(record.source_instance_id, record.property_id, record.element_index).fields


def test_repage_rejects_old_ue4_page_without_clearing_explicit_native_or_alias(app, tmp_path):
    source = opened(app, tmp_path, "ue4_shooter.pxd2")
    scene = next(scene for scene in app.scenes if scene.stream_id == 686661696)
    assert app._refresh(scene_id=scene.id, frame=30)
    model = app.object_model
    native = model.canonical_item(1147)
    alias = model._item(next(key for key, node in model._specs.items()
                             if node.kind == "reference" and node.instance_id is not None))
    page = model._item(next(key for key, node in model._specs.items() if node.kind == "page"))
    app._tree_selection([native, alias, page])
    assert app._refresh()
    rows = model.row_identities([native.key, alias.key, page.key])
    assert len(rows) == 3 and rows[-1].kind == "page"
    before = stable_state(app)
    source.calls.clear()
    original = GeneralSettings.from_preferences(app.workflow.settings)
    accept(app, replace(original, arrays=ArrayDisplaySettings(20, 0)))
    assert source.calls == [] and stable_state(app) == before
    assert [row.token for row in app.row_actions.selected] == [row.token for row in rows[:-1]]
    assert [item.key for item in app.tree.selection] == [row.key for row in rows[:-1]]
    assert model.restore_rows([rows[-1]]) == ()
    assert app._refresh() and stable_state(app) == before
    # Returning to the old numeric group size cannot resurrect its old page.
    accept(app, original)
    assert model.restore_rows([rows[-1]]) == ()
    assert stable_state(app) == before


def test_format_only_apply_rebuilds_inspector_without_native_publication_or_repage(app, tmp_path):
    source = opened(app, tmp_path)
    app._tree_selection([app.object_model.canonical_item(15)])
    assert app._refresh()
    app.presentation_dirty = False
    before, view = stable_state(app), app.presentation
    revision = app.object_model.revision
    rebuilt = app.property_frame.rebuilds
    source.calls.clear()
    candidate = replace(GeneralSettings.from_preferences(app.workflow.settings),
                        value_format=ValueFormat(25, "scientific", True))
    accept(app, candidate)
    assert app.property_inspector.options == candidate.value_format
    assert app.property_frame.rebuilds == rebuilt + 1
    assert not app.presentation_dirty and source.calls == []
    assert app.presentation is view and app.object_model.revision == revision
    assert stable_state(app) == before


def optional_array(app, tmp_path):
    source = opened(app, tmp_path)
    app._tree_selection([app.object_model.canonical_item(18)])
    assert app._refresh()
    prop = app.adapter.property(18, "Restvalues")
    assert prop.type == "PvdF32" and app.presentation.property_size(18, prop.id) == 1052 * 4
    key = (18, prop.id)
    app.property_inspector.array_positions[key] = 10
    app.property_inspector.array_expanded.add(key)
    requests = app.property_inspector.read_requests(app.adapter, [18])
    assert requests == [(18, prop.id, 40, 40)]
    source.calls.clear()
    return source, prop, requests


def test_max_only_zero_suppresses_windows_without_repage_or_native_reference_loss(app, tmp_path):
    source, prop, requests = optional_array(app, tmp_path)
    before, view = stable_state(app), app.presentation
    revision = app.object_model.revision
    rows = app.row_actions.selected
    references = tuple(view.references(6, app.adapter.property(6, "Cloths").id))
    assert references
    accepted = GeneralSettings.from_preferences(app.workflow.settings)
    accept(app, replace(accepted, arrays=ArrayDisplaySettings(10, 0)))
    assert source.calls == [] and app.presentation is view
    assert app.object_model.revision == revision and app.row_actions.selected == rows
    assert app.property_inspector.read_requests(app.adapter, [18]) == []
    assert app.presentation_dirty and app._refresh()
    assert len(source.calls) == 1 and source.calls[0][1]["read_requests"] == []
    assert stable_state(app) == before
    assert tuple(app.presentation.references(6, app.adapter.property(6, "Cloths").id)) == references
    assert app.presentation.property_size(18, prop.id) == 1052 * 4
    assert requests[0][:2] not in app.property_inspector.array_expanded


def test_optional_budget_failure_retries_real_publication_once_and_does_not_loop(app, tmp_path):
    source, prop, requests = optional_array(app, tmp_path)
    previous = app.presentation
    before = stable_state(app)
    source.failures.append(ValueError(BUDGET_ERROR))
    app._request_presentation()
    assert app._refresh()
    assert len(source.calls) == 2
    first, retry = source.calls
    assert first[0] == retry[0] == (app.scene_id, 1)
    assert first[1]["read_requests"] == requests and retry[1]["read_requests"] == []
    assert {k: v for k, v in first[1].items() if k != "read_requests"} == {
        k: v for k, v in retry[1].items() if k != "read_requests"}
    assert app.presentation is not previous and app.presentation.ready and app.presentation.request_applied
    assert stable_state(app) == before
    actual = app.presentation.read(18, prop.id, offset=40, limit=40)
    assert actual == previous.read(18, prop.id, offset=40, limit=40)
    assert len(np.frombuffer(actual, dtype="<f4")) == 10
    assert app.property_inspector.read_requests(app.adapter, [18]) == []
    assert tuple(requests[0]) in app.property_inspector._array_denied
    app._request_presentation()
    assert app._refresh()
    assert len(source.calls) == 3 and source.calls[-1][1]["read_requests"] == []
    assert not app.presentation_dirty


@pytest.mark.parametrize("error", [ValueError("Unknown property ID"),
                                  ValueError(BUDGET_ERROR + "."), RuntimeError(BUDGET_ERROR)])
def test_other_errors_are_not_misclassified_as_optional_budget_failures(app, tmp_path, error):
    source, _prop, requests = optional_array(app, tmp_path)
    before, view = stable_state(app), app.presentation
    source.failures.append(error)
    with pytest.raises(type(error), match=str(error)):
        app._refresh()
    assert len(source.calls) == 1 and source.calls[0][1]["read_requests"] == requests
    assert app.presentation is view and stable_state(app) == before
    assert not app.property_inspector._array_denied


def test_core_budget_failure_without_optional_reads_is_not_retried(app, tmp_path):
    source = opened(app, tmp_path)
    before, view = stable_state(app), app.presentation
    source.failures.append(ValueError(BUDGET_ERROR))
    with pytest.raises(ValueError, match=BUDGET_ERROR):
        app._refresh()
    assert len(source.calls) == 1 and source.calls[0][1]["read_requests"] == []
    assert app.presentation is view and stable_state(app) == before


def test_second_core_failure_propagates_without_replacing_prior_owned_view(app, tmp_path):
    source, _prop, requests = optional_array(app, tmp_path)
    before, view = stable_state(app), app.presentation
    source.failures.extend((ValueError(BUDGET_ERROR), ValueError(BUDGET_ERROR)))
    with pytest.raises(ValueError, match=BUDGET_ERROR):
        app._refresh()
    assert len(source.calls) == 2
    assert source.calls[0][1]["read_requests"] == requests
    assert source.calls[1][1]["read_requests"] == []
    assert app.presentation is view and stable_state(app) == before
    assert not app.refreshing and not app.selection_guard

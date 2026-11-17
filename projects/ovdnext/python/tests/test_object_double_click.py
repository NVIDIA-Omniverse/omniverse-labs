# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native row callbacks select and center without moving the camera eye."""
import numpy as np
import omni.ui as ui
import pytest

from ovd_ui.widgets.object_browser import ObjectBrowser
from test_viewport_menu import app, menu_app, camera_state
from test_application_presentation import open_capture


@pytest.fixture
def object_app(menu_app, monkeypatch):
    application = menu_app
    host = ui.Frame()
    with host:
        browser = ObjectBrowser(application.object_model, application._tree_selection,
                                on_activate=application.center_selected)
    application.object_browser, application.tree = browser, browser.tree
    actor = int(application.renderer.snapshot.object_ids[0])
    bounds = application.renderer.snapshot.bounds[
        application.renderer.snapshot.object_ids == actor]
    center = (bounds[:, 0].min(0) + bounds[:, 1].max(0)) * .5
    application.renderer.camera.position = center + [7., 3., 30.]
    frames, native_frame = [], ui.Frame

    def record_frame(*args, **kwargs):
        frame = native_frame(*args, **kwargs)
        if frame.has_mouse_double_clicked_fn():
            frames.append(frame)
        return frame

    class HeadlessLabel(ui.Label):
        # Native font measurement needs a draw context; callbacks do not.
        @property
        def exact_content_width(self):
            return 0

    monkeypatch.setattr(ui, "Frame", record_frame)
    monkeypatch.setattr(ui, "Label", HeadlessLabel)
    application.actor, application.object_center = actor, center
    application.row_frames = frames
    yield application
    if not browser._destroyed:
        browser.destroy()
    host.destroy()


def drain(application):
    while application.queued:
        callback, args = application.queued.pop(0)
        callback(*args)


def select(application, item):
    application.tree.selection = [item]
    # No draw dispatch runs here: deliver the initial single-click notification.
    # Activation below receives no synthetic selection notification.
    application.object_browser._tree_selection_changed([item])


def row_frame(application, kind="object"):
    browser = application.object_browser
    item = next(browser.model._item(key) for key, node in browser.model._specs.items()
                if node.kind == kind and node.instance_id == application.actor)
    with browser.frame:
        browser.delegate.build_widget(browser.model, item, 0, 1, False)
    return application.row_frames[-1], item


@pytest.mark.parametrize("kind", ["object", "reference"])
@pytest.mark.parametrize("modifiers", [0, 1 << 30])
def test_plain_row_double_click_selects_target_and_centers_at_same_eye(object_app, kind, modifiers):
    application = object_app
    frame, item = row_frame(application, kind)
    camera = application.renderer.camera
    eye = camera.position.copy()
    radius = np.linalg.norm(eye - application.object_center)
    before = camera_state(application)
    select(application, item)  # A single TreeView selection only highlights.
    assert camera_state(application) == before
    assert application.renderer.selected == {application.actor}
    application.tree.clear_selection()
    application.object_browser._tree_selection_changed([])
    frame.call_mouse_double_clicked_fn(12, 15, 0, modifiers)
    assert camera_state(application) == before  # Activation waits until after draw.
    drain(application)
    assert application.interaction.selected == (application.actor,)
    assert [row.key for row in application.tree.selection] == [f"object:{application.actor}"]
    np.testing.assert_allclose(camera.position, eye)
    np.testing.assert_allclose(camera.target, application.object_center)
    assert camera.distance == pytest.approx(radius)


@pytest.mark.parametrize("button,modifiers", [(1, 0), (2, 0), (0, 1), (0, 2), (0, 4), (0, 8)])
def test_modified_or_nonleft_row_double_click_keeps_camera(object_app, button, modifiers):
    application = object_app
    frame, item = row_frame(application)
    select(application, item)
    before = camera_state(application), application.interaction.selected
    frame.call_mouse_double_clicked_fn(12, 15, button, modifiers)
    drain(application)
    assert (camera_state(application), application.interaction.selected) == before


@pytest.mark.parametrize("retirement", ["capture", "destroy"])
def test_queued_row_double_click_cannot_center_a_retired_row(object_app, tmp_path, retirement):
    application = object_app
    frame, item = row_frame(application)
    select(application, item)
    frame.call_mouse_double_clicked_fn(12, 15, 0, 0)
    if retirement == "capture":
        assert application._display(*open_capture(tmp_path / "replacement", "test_scene.ovd"))
    else:
        application.object_browser.destroy()
    before = camera_state(application), application.interaction.selected
    drain(application)
    assert (camera_state(application), application.interaction.selected) == before


def test_reference_menu_reveals_target_without_centering(object_app):
    application = object_app
    browser = application.object_browser
    _, alias = row_frame(application, "reference")
    select(application, alias)
    browser.show_context_menu(browser.model.row_identity(alias.key), 12, 15)
    action = next(item for item in browser.context_menu.items
                  if item.label == "Go to referenced object")
    before = camera_state(application)
    action.triggered_fn()
    drain(application)
    assert [row.key for row in application.tree.selection] == [f"object:{application.actor}"]
    assert application.interaction.selected == (application.actor,)
    assert camera_state(application) == before

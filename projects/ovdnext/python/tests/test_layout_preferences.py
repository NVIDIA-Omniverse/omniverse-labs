# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Production frame callbacks keep layout usable when preference writes fail."""
from types import SimpleNamespace

import omni.ui as ui
import pytest

from ovd_ui.application import Application
from ovd_ui.settings import Preferences
from test_application_presentation import live


@pytest.fixture
def layout(tmp_path, live):
    app = Application()
    app.workflow.settings = Preferences(tmp_path / "preferences.json")
    app.renderer = SimpleNamespace(provider=ui.ImageProvider(), close=lambda: None)
    app.build()
    app.session = app.workflow.live_session = live[0]
    app.capture = "current live capture"
    yield app
    app.close()
    app.window.destroy()


@pytest.mark.parametrize("boundary", ["browser", "inspector", "activity_resize", "activity_expand"])
@pytest.mark.parametrize("error_type", [OSError, ValueError])
def test_layout_write_failure_is_visible_and_next_frame_still_works(layout, monkeypatch, boundary, error_type):
    app = layout
    source, capture = app.session, app.capture
    preferences = app.workflow.settings
    original = preferences.set_many

    def denied(values):
        raise error_type("layout preferences unavailable")

    monkeypatch.setattr(preferences, "set_many", denied)
    if boundary == "activity_expand":
        app.activity.set_expanded(True)
        app.activity.poll()
        assert app.activity.expanded and app.activity.deck.visible
        assert float(app.activity.container.height) == 240
        key, expected = "soft_glass_activity_open", False
        retry = lambda: (app.activity.set_expanded(False), app.activity.poll())
    else:
        activity = boundary == "activity_resize"
        grip = app.activity.grip if activity else getattr(app, boundary + "_grip")
        # The geometry sink represents the intervening layout frame. The grip,
        # registered production callback, preferences and native session are real.
        target = SimpleNamespace(width=320., height=240., computed_width=320., computed_height=240.)
        grip.target = target
        point = lambda offset: (100, 100-offset) if activity else (100+offset*grip.direction, 100)
        def resize():
            grip.press(*point(0), 0, 0)
            grip.release(*point(20), 0)
            grip.update()
            target.computed_width = float(target.width)
            target.computed_height = float(target.height)
            grip.update()
        resize()
        assert not grip.awaiting_layout and not grip.persist_pending
        assert float(target.height if activity else target.width) == (260 if activity else 340)
        if activity:
            assert app.activity.saved_height == 260
        key = "soft_glass_activity_height" if activity else f"soft_glass_{boundary}_width"
        expected = 280 if activity else 360
        retry = resize
    assert "layout preferences unavailable" in app.error_message
    assert app.status_label.text == app.error_message
    assert app.session is app.workflow.live_session is source and app.capture == capture
    assert not source.info().ended
    assert preferences.get(key) is None
    monkeypatch.setattr(preferences, "set_many", original)
    retry()
    assert Preferences(preferences.path).get(key) == expected
    assert app.session is source and not source.info().ended

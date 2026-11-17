# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real owned presentation transitions with callback/toolbar sinks, no GPU."""
from types import SimpleNamespace
import omni.ui as ui

from test_application_presentation import app, open_capture, displayed
from test_analysis_availability import ControlledExecutor, facts
from ovd_ui.analysis_availability import AnalysisAvailability
from ovd_ui.widgets.workspace_tabs import WorkspaceTabs


def test_legacy_tabs_are_inapplicable_to_ovd_but_return_for_legacy_capture(app, tmp_path):
    attach(app)
    window = ui.Window("Analysis applicability test")
    with window.frame:
        app.workspace = WorkspaceTabs(window, lambda _: None, lambda _: None)
    try:
        assert app._display(*open_capture(tmp_path, "test_scene.ovd"))
        app._sync_workspace()
        assert not app.workspace.pages["profile"].applicable
        assert not app.workspace.select("profile")
        assert app._display(*open_capture(tmp_path))
        app._sync_workspace()
        assert app.workspace.pages["profile"].applicable
        assert app.workspace.select("profile")
        app.workspace.poll()
        assert app.workspace.active == "profile"
        assert app._display(*open_capture(tmp_path, "test_scene.ovd"))
        app._sync_workspace()
        app.workspace.poll()
        assert app.workspace.active == "render"
        assert app._display(None, None)
        app._sync_workspace()
        assert not app.workspace.pages["memory"].applicable
    finally:
        app.workspace.destroy()
        window.destroy()
        app.analysis_availability.close()


def attach(app):
    executor = ControlledExecutor()
    app.analysis_availability = AnalysisAvailability(executor=executor)
    app.analysis_menu_items = {}
    app.messages_button = SimpleNamespace(enabled=False, tooltip="")
    app.frame_graph_button = SimpleNamespace(enabled=False, tooltip="")
    return executor


def test_real_capture_switch_and_new_clear_enabledness_and_reject_old_menu(app, tmp_path):
    executor = attach(app)
    try:
        assert app._display(*open_capture(tmp_path))
        before = displayed(app)
        assert len(executor.calls) == 1 and not app.frame_graph_button.enabled
        executor.calls[0][0].set_result(facts())
        app.analysis_availability.poll(); app._update_analysis_controls()
        assert app.frame_graph_button.enabled and not app.messages_button.enabled
        key = app.analysis_availability.context.key
        called = []
        app._invoke_analysis_menu("profile", key, lambda: called.append(True))
        assert called == [True] and displayed(app) == before
        assert app._display(*open_capture(tmp_path, "test_scene.ovd"))
        assert app.messages_button.enabled and not app.frame_graph_button.enabled
        app._invoke_analysis_menu("profile", key, lambda: called.append(False))
        assert called == [True] and "Reopen View" in app.status_label.text
        app.show_profile(); app.show_diagnostic(True); app.show_frame_graph()
        assert "OVD" in app.status_label.text and len(executor.calls) == 1
        assert app._display(None, None)
        assert not app.messages_button.enabled and not app.frame_graph_button.enabled
        app.show_messages()
        assert "Open a capture" in app.status_label.text
    finally: app.analysis_availability.close()


def test_frame_change_does_not_scan_and_stale_completion_cannot_enable_toolbar(app, tmp_path):
    executor = attach(app)
    try:
        app._display(*open_capture(tmp_path))
        old_control = app.analysis_availability.pending[1]
        assert app.seek(3, snap=False)
        assert old_control.cancelled and len(executor.calls) == 1
        executor.calls[0][0].set_result(facts())
        assert not app.analysis_availability.poll()
        app._update_analysis_controls()
        assert not app.frame_graph_button.enabled
        before = displayed(app)
        app.show_diagnostic(False)
        assert "Checking" in app.status_label.text and len(executor.calls) == 2
        assert displayed(app) == before
        executor.calls[1][0].set_result(facts(errors=SimpleNamespace(known=True, available=False, reason="No log at this frame")))
        assert app.analysis_availability.poll()
        app.show_diagnostic(False)
        assert app.status_label.text == "No log at this frame" and len(executor.calls) == 2
        assert displayed(app) == before
    finally: app.analysis_availability.close()


def test_known_whole_file_facts_survive_next_without_background_scan(app, tmp_path):
    executor = attach(app)
    try:
        app._display(*open_capture(tmp_path))
        executor.calls[0][0].set_result(facts(memory_entries=144))
        app.analysis_availability.poll(); app._update_analysis_controls()
        assert app.seek(3,snap=False)
        assert app.frame_graph_button.enabled and len(executor.calls)==1
        assert app._guard_analysis("profile") and len(executor.calls)==1
        assert app._guard_analysis("memory") and len(executor.calls)==1
        assert not app._guard_analysis("errors") and len(executor.calls)==2
    finally: app.analysis_availability.close()

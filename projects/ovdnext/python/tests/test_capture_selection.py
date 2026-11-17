# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recording selection actions use the real native live target, not the viewport."""
import importlib
import struct
import os
from pathlib import Path
import sys

import pytest

from ovd_ui.capture_control import CaptureSelectionModel
from test_native_capture_control import (ACTOR, CONTEXT, SCENE, META, CATALOG, ack,
    catalog, live, metadata, no_reverse_bytes, receive, schema, wait_for)
from test_workflow import FIXTURE, producer, until, workflow
from gui_runtime import GUI_AVAILABLE, run_gui_process


def test_workflow_live_selection_survives_old_file_and_poll_hold_then_resets(workflow):
    workflow.startup()
    assert hasattr(workflow, "capture_selection"), "Workflow live selection binding is missing"
    model = workflow.capture_selection
    live_session = workflow.live_session
    assert model.session is live_session
    assert workflow.open_file(FIXTURE)
    with producer(workflow) as connection:
        connection.sendall(struct.pack("<IHHIII", 0x4f56444c, 1, 12, 0, 4, 0)
            + schema(1, "PxOmniPvdMetaData", META) + schema(2, "PxOmniPvdCaptureEntry", CATALOG)
            + metadata() + catalog())
        until(workflow, lambda: model.info.get("catalog_ready"))
        assert model.session is live_session and workflow.display_session is not live_session
        model.select(ACTOR, True)
        no_reverse_bytes(connection)
        release = workflow.suspend_live_polling()
        assert model.apply() == 1
        assert receive(connection, 36) == struct.pack("<IIQIIIQ", 0x3343564f, 36, 1, 0, 0, 1, ACTOR)
        connection.sendall(ack(1))
        until(workflow, lambda: model.info.get("state") == "applied")
        assert workflow.display_session is not live_session
        release()
        workflow.detach()
        assert model.session is None and model.selected_count == 0 and model.info == {}
    workflow.set_accept_connections(True)
    assert model.session is workflow.live_session and model.session is not live_session
    workflow.new()
    assert model.session is workflow.live_session and model.selected_count == 0


def panel(model):
    module = importlib.import_module("ovd_ui.widgets.capture_selection")
    return module.CaptureSelectionPanel(model)


def test_panel_actions_require_explicit_scope_and_keep_context_rows_unselectable():
    with live() as (session, connection):
        model = CaptureSelectionModel(); model.bind(session); model.poll()
        view = panel(model)
        assert not view.can_apply
        assert not view.apply() and "Select" in view.message
        assert not view.choose(CONTEXT, True)
        assert view.choose(ACTOR, True) and view.can_apply
        no_reverse_bytes(connection)
        assert view.apply()
        assert receive(connection, 36) == struct.pack("<IIQIIIQ", 0x3343564f, 36, 1, 0, 0, 1, ACTOR)
        connection.sendall(ack(1, 4))
        wait_for(lambda: session.capture_control_info()["state"] == "denied")
        view.refresh()
        assert "denied" in model.status
        view.set_scope(True); view.set_secondary(True)
        assert view.apply()
        assert receive(connection, 28) == struct.pack("<IIQIII", 0x3343564f, 28, 2, 1, 0, 0)
        view.close()
        assert not view.apply() and not view.refresh_catalog()


def test_retired_checkbox_cannot_select_a_reused_handle_in_a_new_generation_or_connection():
    with live() as (session, connection):
        model = CaptureSelectionModel(); model.bind(session); model.poll()
        view = panel(model)
        origin = (session, model.generation)
        assert view.choose(ACTOR, True, origin=origin)
        connection.sendall(b"".join(struct.pack("<BQQ", 10, 2, row) for row in (200, 201, 202)) + catalog(2))
        wait_for(lambda: session.capture_control_info()["generation"] == 2)
        assert not view.choose(ACTOR, True, origin=origin)
        assert model.selected_count == 0 and not view.can_apply
        assert view.choose(ACTOR, True, origin=(session, 2))
        with live() as (replacement, peer):
            model.bind(replacement); model.poll()
            assert not view.choose(ACTOR, True, origin=origin)
            assert model.selected_count == 0
            no_reverse_bytes(peer)
        no_reverse_bytes(connection)
        view.close()


@pytest.mark.parametrize("capabilities", [0, 4, 8])
def test_panel_unsupported_actions_send_no_bytes(capabilities):
    with live(capabilities, with_catalog=False) as (session, connection):
        model = CaptureSelectionModel(); model.bind(session); model.poll()
        view = panel(model)
        assert not view.can_apply
        view.set_scope(True)
        assert view.can_apply == bool(capabilities & 4)
        assert view.can_refresh == bool(capabilities & 8)
        if not view.can_apply:
            assert not view.apply()
        if not view.can_refresh:
            assert not view.refresh_catalog()
        no_reverse_bytes(connection)
        view.close()


@pytest.mark.skipif(not GUI_AVAILABLE, reason="Requires an owned OvUI graphics runtime")
def test_real_panel_clicks_target_live_connection(tmp_path):
    root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ, OVUIINSPECT_ENABLED="0", LOCALAPPDATA=str(tmp_path / "data"),
        XDG_DATA_HOME=str(tmp_path / "data"), XDG_CONFIG_HOME=str(tmp_path / "config"),
        XDG_CACHE_HOME=str(tmp_path / "cache"), OVDNEXT_CAPTURE_SCREENSHOT=str(tmp_path / "selection.png"))
    result = run_gui_process([sys.executable, "-B", str(root / "python/tests/capture_selection_gui.py")],
        cwd=root, env=environment, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Recording selection actual-click checks passed" in result.stdout
    assert "Evidence and recording selection resources closed" in result.stdout
    assert (tmp_path / "selection.png").is_file()

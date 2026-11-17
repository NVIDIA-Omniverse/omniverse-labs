# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persistent connection facts use the real TCP receiver and owned display.

Only widget/render sinks are synthetic; no GPU, window or network injection
into another application is needed for these controller/presentation checks.
"""
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
import socket

import pytest

from test_workflow import (workflow, producer, until, FIXTURE, HANDSHAKE,
                           PAYLOAD, ACTIVE_PAYLOAD)
from test_application_presentation import app, open_capture
from ovd_ui.recordings import RecordingLibrary
from ovd_ui.settings import Preferences
from ovd_ui.workflow import CaptureWorkflow


def test_application_preserves_all_startup_errors_in_one_poll(app,tmp_path):
    preferences_path=tmp_path/"preferences.json"
    preferences_path.write_text("invalid json")
    recordings_path=tmp_path/"recordings"
    recordings_path.write_text("not a directory")
    workflow=CaptureWorkflow(Preferences(preferences_path),RecordingLibrary(recordings_path))
    app.workflow=workflow
    try:
        # A repeated failure in the same poll should not drown out the others.
        workflow.open_file(tmp_path/"missing.ovd")
        workflow.open_file(tmp_path/"missing.ovd")
        app._poll_workflow()
        shown=app.status_label.text
        assert "Could not load preferences" in shown
        assert "Could not load recordings" in shown
        assert shown.count("Unable to open missing.ovd:")==1
        assert app.error_message==shown
    finally:
        workflow.shutdown()


def test_inactive_waiting_and_partial_receiving_are_distinct_owned_states(workflow):
    inactive=workflow.connection_status
    assert inactive.state=="inactive" and "not been started" in inactive.tooltip
    assert workflow.attach(workflow.port)
    waiting=workflow.connection_status
    assert waiting.state=="waiting" and str(workflow.port) in waiting.text
    assert not waiting.background and "requested" in waiting.tooltip
    with producer(workflow) as connection:
        connection.sendall(HANDSHAKE+PAYLOAD[:1024])
        until(workflow,lambda:workflow.connection_status.state=="receiving")
        receiving=workflow.connection_status
        assert not receiving.background and workflow.display_session is None
        assert workflow.live_session.info().connected
        assert waiting.state=="waiting" and inactive.state=="inactive"
        with pytest.raises(FrozenInstanceError):receiving.state="waiting"


def test_pausing_new_connections_preserves_actual_receiver_then_pauses(workflow):
    workflow.attach(workflow.port)
    with producer(workflow) as connection:
        connection.sendall(HANDSHAKE+ACTIVE_PAYLOAD)
        until(workflow,lambda:workflow.display_session is workflow.live_session)
        active=workflow.live_session
        workflow.set_accept_connections(False)
        status=workflow.connection_status
        assert status.state=="receiving" and not status.accepting
        assert "current producer continues" in status.tooltip
        assert not active.info().ended
    until(workflow,lambda:workflow.live_session is None)
    assert workflow.connection_status.state=="paused"
    assert workflow.display_session is active
    workflow.set_accept_connections(True)
    assert workflow.connection_status.state=="waiting"
    assert workflow.connection_status.background


def test_background_file_browsing_follow_and_detach_retain_correct_scope(workflow):
    workflow.attach(workflow.port)
    with producer(workflow) as connection:
        connection.sendall(HANDSHAKE+ACTIVE_PAYLOAD)
        until(workflow,lambda:workflow.display_session is workflow.live_session)
        active=workflow.live_session
        assert workflow.open_file(FIXTURE)
        assert workflow.live_session is active
        status=workflow.connection_status
        assert status.state=="receiving" and status.background
        assert "displayed capture is separate" in status.tooltip
        workflow.follow_latest()
        assert not workflow.connection_status.background
        workflow.detach()
        assert workflow.connection_status.state=="paused"
        assert workflow.display_session is active and active.info().ended
    workflow.new()
    assert workflow.connection_status.state=="paused" and workflow.display_session is None
    workflow.set_accept_connections(True)
    assert workflow.connection_status.state=="waiting" and not workflow.connection_status.background


def test_finished_recording_has_background_waiter_without_relabeling_it_receiving(workflow):
    workflow.attach(workflow.port)
    active=workflow.live_session
    with producer(workflow) as connection:connection.sendall(HANDSHAKE+ACTIVE_PAYLOAD)
    until(workflow,lambda:workflow.live_session is not active)
    assert workflow.display_session is active
    status=workflow.connection_status
    assert status.state=="waiting" and status.background
    assert "listening is requested in the background" in status.tooltip
    assert "live capture continues" not in status.tooltip


def test_native_listener_failure_retains_its_reason_then_clears_on_rearm(workflow):
    with socket.socket() as occupied:
        occupied.bind(("0.0.0.0",0));occupied.listen()
        workflow.attach(occupied.getsockname()[1])
        until(workflow,lambda:workflow.connection_status.state=="retrying")
        failed=workflow.connection_status
        assert failed.error and failed.error in " ".join(failed.tooltip.split())
        assert str(workflow.port) in failed.tooltip and "retry" in failed.tooltip
        # An unrelated failing open neither erases the listener failure nor
        # replaces it with a load error in the persistent transport tooltip.
        assert not workflow.open_file(FIXTURE.with_name("absent.ovd"))
        assert workflow.connection_status==failed
    until(workflow,lambda:workflow.live_session is not None,timeout=3)
    assert workflow.connection_status.state=="waiting" and not workflow.connection_status.error
    assert failed.error


def test_reading_status_never_reenters_native_session(workflow,monkeypatch):
    workflow.attach(workflow.port)
    with producer(workflow) as connection:
        connection.sendall(HANDSHAKE+ACTIVE_PAYLOAD)
        until(workflow,lambda:workflow.display_session is workflow.live_session)
        expected=workflow.connection_status
        def no_read(*_):raise AssertionError("Persistent status must use observed owned facts")
        with monkeypatch.context() as patch:
            patch.setattr(type(workflow.live_session),"info",no_read)
            for _ in range(100):assert workflow.connection_status==expected
    workflow.shutdown()
    assert workflow.connection_status.state=="closed"


def test_application_label_survives_native_seeks_and_does_not_replace_errors(app,workflow,tmp_path):
    app.workflow=workflow
    app.connection_label=SimpleNamespace(text="",tooltip="")
    assert workflow.open_file(FIXTURE)
    assert app._display(*open_capture(tmp_path))
    workflow.attach(workflow.port)
    workflow.follow_live=False
    with producer(workflow) as connection:
        connection.sendall(HANDSHAKE+ACTIVE_PAYLOAD)
        until(workflow,lambda:workflow.connection_status.state=="receiving")
        workflow.poll()  # The display here is explicitly the existing cloth view.
        app._sync_connection_status()
        text,tooltip=app.connection_label.text,app.connection_label.tooltip
        assert text==f"{workflow.port} · Receiving"
        assert "Displayed: PVD file" in tooltip and "background" in tooltip
        assert "cloth_physx34.pxd2" in tooltip
        assert app.seek(30,snap=False) and app.frame==30
        assert (app.connection_label.text,app.connection_label.tooltip)==(text,tooltip)
        app._report_error("A useful capture error")
        app._sync_connection_status()
        app._poll_workflow()
        assert app.status_label.text=="A useful capture error"
        assert app.connection_label.text==text
        assert "A useful capture error" not in app.connection_label.tooltip
        assert app.accept_button.tooltip.startswith(tooltip)
        # A workflow promotion can be queued before Application applies the
        # corresponding native view. Describe the actual shown cloth capture.
        workflow.follow_latest()
        # Receiving bytes does not imply a complete frame has been published.
        # Poll only the workflow, leaving Application on its owned cloth view.
        until(workflow,lambda:workflow.display_session is workflow.live_session)
        assert not workflow.connection_status.background
        app._sync_connection_status()
        assert app.frame==30 and "PVD file" in app.connection_label.tooltip
        assert "background" in app.connection_label.tooltip


def test_application_owned_live_mode_freezes_then_clears_with_new(app,workflow):
    app.workflow=workflow
    app.connection_label=SimpleNamespace(text="",tooltip="")
    workflow.attach(workflow.port)
    with producer(workflow) as connection:
        connection.sendall(HANDSHAKE+ACTIVE_PAYLOAD)
        until(workflow,lambda:workflow.display_session is workflow.live_session)
        assert app._display(workflow.display_path,workflow.display_session)
        assert "Displayed: OVD live capture" in app.connection_label.tooltip
        workflow.detach()
        app._sync_connection_status()
        assert app.connection_label.text=="Live paused"
        assert "OVD recording (frozen)" in app.connection_label.tooltip
    workflow.new()
    app._display(None,None)
    assert "No capture displayed" in app.connection_label.tooltip
    assert "recording" not in app.connection_label.tooltip


def test_tooltip_semantic_lines_preserve_all_facts_and_long_error_content():
    from ovd_ui.workflow import ConnectionStatus,wrap_connection_tooltip
    status=ConnectionStatus('receiving',5425,False,True)
    assert status.tooltip.splitlines()==[
        'Receiving a producer on TCP port 5425.',
        'The displayed capture is separate;',
        'live capture continues in the background.',
        'New connections are paused;',
        'the current producer continues.']
    error='Listener failed: '+('W'*200)+'; retry remains available.'
    failed=ConnectionStatus('retrying',5425,True,False,error)
    assert max(map(len,failed.tooltip.splitlines()))<=64
    assert ''.join(error.split()) in ''.join(failed.tooltip.split())
    filename='captured_'+('M'*240)+'.ovd'
    tooltip=wrap_connection_tooltip('Displayed: OVD file.\n'+filename+'\n'+status.tooltip)
    assert max(map(len,tooltip.splitlines()))<=64
    assert filename in ''.join(tooltip.splitlines())

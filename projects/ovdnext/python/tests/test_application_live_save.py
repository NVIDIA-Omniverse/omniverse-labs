# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A live reconnect must not retarget an open whole-capture save picker."""
from types import SimpleNamespace

import pytest

from ovd_ui.application import Application
from ovd_ui.widgets.file_browser import CaptureFileDialog
from test_workflow import workflow, producer, send_capture, until, HANDSHAKE, ACTIVE_PAYLOAD, PAYLOAD


@pytest.mark.parametrize('finish',['save','cancel','destroy'])
@pytest.mark.parametrize('ended_while_open',[False,True])
def test_save_picker_keeps_completed_source_while_next_producer_receives(workflow,tmp_path,monkeypatch,finish,ended_while_open):
    workflow.attach(workflow.port)
    first=workflow.live_session
    send_capture(workflow)
    until(workflow,lambda:workflow.live_session is not first)
    original=workflow.display_path
    assert original.read_bytes()==PAYLOAD
    app=Application.__new__(Application)
    app.workflow=workflow;app.session=first;app.capture=original;app.dialogs=[]
    destination=tmp_path/'first-copy.ovd'
    def show(dialog):
        dialog._dismissed=False
        dialog.filename_field=SimpleNamespace(model=SimpleNamespace(as_string=str(destination)))
    monkeypatch.setattr(CaptureFileDialog,'show',show)
    app.show_save()
    picker=app.dialogs[-1]
    second=workflow.live_session
    with producer(workflow) as connection:
        connection.sendall(HANDSHAKE+ACTIVE_PAYLOAD)
        until(workflow,lambda:second.info().completed_serial==2)
        if ended_while_open:
            import socket
            connection.sendall(PAYLOAD[len(ACTIVE_PAYLOAD):])
            connection.shutdown(socket.SHUT_WR)
            until(workflow,lambda:second.info().ended)
        assert workflow.display_session is first
        assert workflow.display_path==original and workflow.follow_live
        # A validation error must retain the same hold and destination picker.
        picker.filename_field.model.as_string=str(tmp_path/'missing'/'capture.ovd')
        picker.choose()
        assert not picker._dismissed and 'folder does not exist' in picker.error
        workflow.poll()
        assert workflow.display_session is first
        picker.filename_field.model.as_string=str(destination)
        if finish=='save':
            picker.choose()
            assert picker._dismissed and destination.read_bytes()==PAYLOAD
        else:
            getattr(picker,finish)()
            assert not destination.exists()
        until(workflow,lambda:workflow.display_session is second)
        assert workflow.accept_connections
        assert original.read_bytes()==PAYLOAD
        if ended_while_open:
            assert workflow.live_session is not second
        else:
            assert workflow.live_session is second and not second.info().ended
            connection.sendall(PAYLOAD[len(ACTIVE_PAYLOAD):])
    until(workflow,lambda:workflow.live_session is not second)
    assert workflow.display_path.read_bytes()==PAYLOAD


def test_nested_live_poll_holds_release_independently(workflow):
    workflow.attach(workflow.port)
    source=workflow.live_session
    first=workflow.suspend_live_polling()
    last=workflow.suspend_live_polling()
    send_capture(workflow)
    until(workflow,lambda:source.info().ended)
    first();first()
    workflow.poll()
    assert workflow.live_session is source and workflow.display_session is None
    last()
    until(workflow,lambda:workflow.display_session is source)
    assert workflow.live_session is not source and workflow.display_path.read_bytes()==PAYLOAD

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source ownership and bounded polling behind the embedded bottom workspace."""
from types import SimpleNamespace

from ovd_ui.analysis import CaptureMessages
from ovd_ui.application import Application


def test_unchanged_message_poll_keeps_owned_rows_and_reports_truncation_and_clear():
    row=SimpleNamespace(frame_id=1,type_name="eDEBUG_INFO",message="Recorded message",
                        file="example.cpp",line=12,type=0,handle=0)
    total=[1]
    source=SimpleNamespace(message_count=lambda:total[0],messages=lambda offset,count:[row]*count)
    adapter=CaptureMessages(source,limit=1)
    assert adapter.refresh()
    records=adapter.records
    assert not adapter.refresh() and adapter.records is records
    total[0]=2
    assert adapter.refresh() and adapter.truncated and adapter.total==2
    total[0]=0
    assert adapter.refresh() and not adapter.truncated and adapter.records==()
    assert not adapter.refresh()


def test_old_clip_controls_cannot_edit_a_replacement_capture():
    app=Application.__new__(Application)
    context=SimpleNamespace(capture_key=1,segment_epoch=0,scene_id=30)
    app.analysis_availability=SimpleNamespace(context=context)
    calls=[]
    app.playback=SimpleNamespace(split=object(),select=object())
    app._clip_action=lambda action,*args:calls.append((action,args))
    source=app._activity_source()
    app._activity_clip_action(source,"split")
    assert calls==[(app.playback.split,())]
    for field in ("capture_key","segment_epoch","scene_id"):
        previous=getattr(context,field)
        setattr(context,field,previous+1)
        app._activity_clip_action(source,"select",9)
        setattr(context,field,previous)
    app._activity_clip_action(None,"split")
    assert len(calls)==1

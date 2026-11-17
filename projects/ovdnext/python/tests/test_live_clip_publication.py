# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Clip edits wait for the same real TCP publication as the viewport/inspector."""
from copy import deepcopy
import struct

import pytest

from ovd_ui.playback import FrameMode
from test_application_presentation import (
    app,live,open_capture,displayed,wait_until,PAYLOAD,HANDSHAKE,FRAME2_END,
    ROOT,
)


@pytest.fixture
def repeated_live(app, live):
    source, connection = live
    payload = (ROOT / "test/data/resample_two_segment.ovd").read_bytes()
    offset = 0
    def publish(frame):
        nonlocal offset
        stop = struct.pack("<BQQ", 12, 2929590687920, frame)
        end = payload.index(stop) + len(stop)
        connection.sendall((HANDSHAKE if not offset else b"") + payload[offset:end])
        offset = end
        wait_until(lambda: source.info().completed_serial == frame)
    publish(2)
    app.workflow.live_session = source
    assert app._display(None, source)
    app.playback.set_mode(FrameMode.FULL)
    return publish


@pytest.mark.parametrize("operation", ["trim_end", "split"])
@pytest.mark.parametrize("follow", [False, True])
def test_repeated_live_publications_reuse_tail_and_preserve_edited_ranges(app, repeated_live, operation, follow):
    assert app.seek(2 if operation == "trim_end" else 1, snap=False)
    assert app._clip_action(getattr(app.playback, operation))
    app.workflow.follow_live = follow
    for frame in (3, 4, 5):
        repeated_live(frame)
        assert app._refresh(frame=frame if follow else 1)
        ranges = [(clip.start, clip.stop) for clip in app.playback.clips]
        if follow:
            assert ranges == [(1, 2), (0 if operation == "trim_end" else 2, frame+1)]
            assert app.playback.index == 1
        else:
            assert ranges == ([(1, 2)] if operation == "trim_end" else [(1, 2), (2, 3)])
            assert app.playback.index == 0
        assert app.frame == app.playback.frame == app.presentation.frame == (frame if follow else 1)


def test_follow_tail_survives_reselection_and_deletion_and_resets_on_capture_replacement(app, repeated_live, tmp_path):
    assert app.seek(2, snap=False)
    assert app._clip_action(app.playback.trim_end)
    app.workflow.follow_live = True
    repeated_live(3)
    assert app._refresh(frame=3) and len(app.playback.clips) == 2
    assert app._clip_action(app.playback.select, 0)
    app.workflow.follow_live = True
    repeated_live(4)
    assert app._refresh(frame=4)
    assert len(app.playback.clips) == 2 and app.playback.index == 1
    assert app._clip_action(app.playback.delete)
    assert len(app.playback.clips) == 1
    app.workflow.follow_live = True
    repeated_live(5)
    assert app._refresh(frame=5)
    assert [(c.start, c.stop) for c in app.playback.clips] == [(1, 2), (0, 6)]
    assert app.frame == app.playback.frame == app.presentation.frame == 5
    assert app._display(*open_capture(tmp_path))
    assert len(app.playback.clips) == 1 and app.playback.clip.stop == 772
    repeated_live(6)
    assert app._refresh()
    assert len(app.playback.clips) == 1 and app.playback.clip.stop == 772


@pytest.mark.parametrize("operation", ["trim_end", "split"])
def test_follow_reuses_renamed_tail_after_repeated_scrub_publications(app, repeated_live, operation):
    assert app.seek(2 if operation == "trim_end" else 1, snap=False)
    assert app._clip_action(getattr(app.playback, operation))
    for following, scrubbing in ((3, 4), (5, 6)):
        app.workflow.follow_live = True
        repeated_live(following)
        assert app._refresh(frame=following)
        assert app._clip_action(app.playback.rename, "My live range")
        assert app._clip_action(app.playback.select, 0)
        assert app.seek(1, snap=False)
        repeated_live(scrubbing)
        assert app._refresh()
        assert app.frame == app.playback.frame == app.presentation.frame == 1
        app.workflow.follow_live = True
        assert app._refresh(frame=scrubbing)
        assert [(c.start, c.stop) for c in app.playback.clips] == [
            (1, 2), (0 if operation == "trim_end" else 2, scrubbing + 1)]
        assert app.playback.index == 1 and app.playback.clip.name == "My live range"
        assert app.frame == app.playback.frame == app.presentation.frame == scrubbing


def test_manually_shortened_live_tail_stays_trimmed_when_follow_resumes(app, repeated_live):
    assert app.seek(2, snap=False)
    assert app._clip_action(app.playback.trim_end)
    app.workflow.follow_live = True
    repeated_live(3)
    assert app._refresh(frame=3)
    assert app.seek(2, snap=False)
    assert app._clip_action(app.playback.trim_end)
    repeated_live(4)
    assert app._refresh()
    app.workflow.follow_live = True
    assert app._refresh(frame=4)
    assert [(c.start, c.stop) for c in app.playback.clips] == [(1, 2), (0, 2), (0, 5)]
    assert app.seek(1, snap=False)
    repeated_live(5)
    assert app._refresh()
    app.workflow.follow_live = True
    assert app._refresh(frame=5)
    assert [(c.start, c.stop) for c in app.playback.clips] == [(1, 2), (0, 2), (0, 6)]
    assert app.frame == app.playback.frame == app.presentation.frame == 5


@pytest.mark.parametrize("operation", ["rename", "split"])
def test_live_tail_ownership_survives_deferred_clip_adoption(app, live, operation):
    source, connection = published_live(app, live, 2)
    assert app._clip_action(app.playback.trim_end)
    app.workflow.follow_live = True
    assert app._refresh(frame=2)
    assert app.seek(1, snap=False)
    make_partial(source, connection)
    displayed_playback = app.playback
    assert not app._clip_action(getattr(app.playback, operation), *(["Renamed tail"] if operation == "rename" else []))
    assert app.playback is displayed_playback and app.pending_playback is not None
    source.stop_live()
    assert app._refresh() and app.pending_playback is None
    app.workflow.follow_live = True
    assert app._refresh(frame=3)
    assert [(c.start, c.stop) for c in app.playback.clips] == (
        [(1, 2), (0, 4)] if operation == "rename" else [(1, 2), (0, 2), (2, 4)])
    assert app.frame == app.playback.frame == app.presentation.frame == 3


def published_live(app,live,frame):
    source,connection=live
    app.workflow.live_session=source
    connection.sendall(HANDSHAKE+PAYLOAD[:FRAME2_END])
    wait_until(lambda:source.info().completed_serial==2)
    assert app._display(None,source)
    app.playback.set_mode(FrameMode.FULL)
    assert app.seek(frame,snap=False)
    app.selection.publish(["/objects/1"],source="test")
    assert app._refresh()
    return source,connection


def make_partial(source,connection):
    connection.sendall(PAYLOAD[FRAME2_END:])
    # Frame 3 publishes before the fixture's final unpublished teardown.
    wait_until(lambda:source.info().recording_size==len(PAYLOAD)
               and source.info().unpublished_changes)


@pytest.mark.parametrize("operation,frame,expected_frame,expected_ranges,applied_ranges",[
    ("trim_start",2,2,[(2,3)],[(2,4)]),
    ("trim_end",2,1,[(1,2)],[(1,2)]),
    ("split",1,1,[(1,2),(2,3)],[(1,2),(2,3)]),
])
@pytest.mark.parametrize("panel_open",[False,True])
def test_clip_edit_keeps_displayed_state_until_partial_live_view_can_apply(
        app,live,operation,frame,expected_frame,expected_ranges,applied_ranges,panel_open):
    source,connection=published_live(app,live,frame)
    if not panel_open:del app.clips_view
    old=app.playback
    before=deepcopy(old.__dict__)
    visible=displayed(app)
    refreshes=app.clips_view.refreshes if panel_open else 0
    make_partial(source,connection)
    app.workflow.follow_live=True
    assert not app._clip_action(getattr(app.playback,operation))
    assert not app.workflow.follow_live
    staged=app.pending_playback
    assert staged is not old
    assert [(c.start,c.stop) for c in staged.clips]==expected_ranges
    assert app.playback is old and old.__dict__==before
    assert displayed(app)==visible
    assert app.pending_presentation==(app.scene_id,expected_frame)
    # A normal poll must neither leak the pending edit into the UI nor replay
    # it against the already-edited clone a second time.
    assert not app._refresh()
    assert app.pending_playback is staged and app.playback is old
    assert old.__dict__==before and displayed(app)==visible
    if panel_open:assert app.clips_view.refreshes==refreshes
    source.stop_live()
    assert app._refresh()
    assert app.pending_playback is None and app.pending_presentation is None
    assert app.frame==app.playback.frame==app.presentation.frame==expected_frame
    # Detach makes the already-published frame 3 safe to adopt. Tail extension
    # applies to the first clip only when its stop matched the old source end.
    assert [(c.start,c.stop) for c in app.playback.clips]==applied_ranges
    assert app.selection.get_snapshot().paths()==["/objects/1"]
    if panel_open:assert app.clips_view.refreshes==refreshes+1
    else:assert not hasattr(app,"clips_view")
    assert old.__dict__==before


def test_new_seek_supersedes_unpublished_clip_intent(app,live):
    source,connection=published_live(app,live,2)
    original=deepcopy(app.playback.__dict__)
    make_partial(source,connection)
    assert not app._clip_action(app.playback.trim_end)
    assert app.pending_playback.clip.stop==2
    assert not app.seek(1,snap=False)
    assert app.pending_playback.clip.stop==3
    assert app.playback.__dict__==original
    source.stop_live()
    assert app._refresh()
    assert app.playback.clip.start==1 and app.playback.clip.stop==4
    assert app.playback.frame==app.presentation.frame==1


def test_capture_replacement_discards_unpublished_clip_intent(app,live,tmp_path):
    source,connection=published_live(app,live,2)
    make_partial(source,connection)
    assert not app._clip_action(app.playback.trim_end)
    assert app._display(*open_capture(tmp_path))
    new_key=app.presentation.capture_key
    assert new_key!=source.capture_key and app.pending_playback is None
    assert app.playback.clip.start==1 and app.playback.clip.stop==772
    source.stop_live()
    assert app._refresh()
    assert app.presentation.capture_key==new_key
    assert app.playback.clip.start==1 and app.playback.clip.stop==772


@pytest.mark.parametrize("operation,args,expected_index,expected_ranges",[
    ("select",(1,),1,[(1,2),(2,3)]),
    ("delete",(),0,[(2,4)]),
])
def test_clip_selection_and_delete_share_the_same_deferred_adoption(
        app,live,operation,args,expected_index,expected_ranges):
    source,connection=published_live(app,live,1)
    assert app._clip_action(app.playback.split)
    before=deepcopy(app.playback.__dict__)
    visible=displayed(app)
    make_partial(source,connection)
    assert not app._clip_action(getattr(app.playback,operation),*args)
    assert app.playback.__dict__==before and displayed(app)==visible
    source.stop_live()
    assert app._refresh()
    assert app.playback.index==expected_index
    assert [(c.start,c.stop) for c in app.playback.clips]==expected_ranges
    assert app.playback.frame==app.presentation.frame==2


def test_second_clip_action_replaces_pending_intent_from_displayed_playback(app,live):
    source,connection=published_live(app,live,2)
    old=app.playback
    old_trim_start=old.trim_start
    before=deepcopy(old.__dict__)
    make_partial(source,connection)
    assert not app._clip_action(old.trim_end)
    assert app.pending_playback.clip.stop==2
    first_request=app.pending_playback
    # Resolve the method on a new deepcopy, not on the previously bound object
    # or the unapplied trim-end request. Latest intent uses what is displayed.
    assert not app._clip_action(old_trim_start)
    assert app.pending_playback is not first_request
    assert (app.pending_playback.clip.start,app.pending_playback.clip.stop)==(2,3)
    assert app.playback is old and old.__dict__==before
    assert first_request.clip.start==1 and first_request.clip.stop==2
    source.stop_live()
    assert app._refresh()
    assert (app.playback.clip.start,app.playback.clip.stop)==(2,4)
    assert app.frame==app.presentation.frame==2
    assert old.__dict__==before


def test_clip_row_passes_selected_index_and_pose_seek_does_not_refresh_rows(app,tmp_path):
    from ovd_ui.widgets.clips_view import ClipRow
    assert app._display(*open_capture(tmp_path))
    assert app.seek(30,snap=False)
    assert app._clip_action(app.playback.split)
    app._activity_source=lambda:(app.presentation.capture_key,app.presentation.segment_epoch,
                                 app.presentation.scene_id)
    clip=app.playback.clips[1]
    row=ClipRow(1,clip.name,clip.start,clip.stop)
    app._clip_row_action(app._activity_source(),"select",row)
    assert app.playback.index==1 and app.presentation.frame==31
    before=app.clips_view.refreshes
    assert app.seek(35,snap=False)
    assert app.clips_view.refreshes==before


@pytest.mark.parametrize("pending_no_op",[False,True])
def test_final_native_tail_extension_refreshes_panel_even_without_changed_pending_rows(app,live,pending_no_op):
    source,connection=published_live(app,live,1)
    before=app.clips_view.refreshes
    make_partial(source,connection)
    if pending_no_op:
        assert not app._clip_action(app.playback.trim_start)
        assert app.pending_playback.clip.start==app.playback.clip.start==1
        assert app.pending_playback.clip.stop==app.playback.clip.stop==3
    else:
        assert not app._refresh()
        assert app.pending_playback is None
    assert app.clips_view.refreshes==before and app.playback.clip.stop==3
    source.stop_live()
    assert app._refresh()
    assert app.playback.clip.stop==4
    assert app.clips_view.refreshes==before+1
    assert app.frame==app.playback.frame==app.presentation.frame==1

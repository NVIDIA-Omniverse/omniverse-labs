# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contact guidance follows the displayed frame, not pending playback intent."""
from copy import deepcopy

import pytest

from ovd_ui.playback import FrameMode
from test_application_presentation import (app, live, open_capture, displayed, wait_until,
    HANDSHAKE, PAYLOAD, FRAME2_END)


@pytest.mark.parametrize("mode", [1, 2, 3])
@pytest.mark.parametrize("playback_mode", [FrameMode.POST, FrameMode.FULL])
def test_explicit_contact_shortcut_uses_matching_pre_frame_and_preserves_selection(app, tmp_path, mode, playback_mode):
    app.overlay_mask = 32
    app.overlay_modes = {32: mode}
    assert app._display(*open_capture(tmp_path, "multi_geom.ovd"))
    actor = int(app.renderer.snapshot.object_ids[0])
    app.selection.publish([f"/objects/{actor}"], source="test")
    assert app._refresh()
    assert app.frame == 2
    app.playback.set_mode(playback_mode)
    before = displayed(app)
    assert app._contact_guidance() == (True, 1)
    assert displayed(app) == before
    app.playback.toggle()
    assert app._go_contact_pre()
    assert app.frame == app.presentation.frame == app.playback.frame == 1
    expected = FrameMode.FULL if playback_mode == FrameMode.FULL else FrameMode.PRE
    assert app.playback.mode == expected and not app.playback.playing
    assert app.selection.get_snapshot().paths() == [f"/objects/{actor}"]
    assert app._overlay_mode(32) == mode
    assert app.renderer.overlays.current_frame == 1
    assert app._contact_guidance() == (False, None)
    assert app.playback.step(1) == (2 if playback_mode == FrameMode.FULL else 3)


def test_guidance_ignores_pending_mode_and_never_escapes_trimmed_clip(app, tmp_path):
    assert app._contact_guidance() == (False, None)
    app.overlay_mask = 32
    assert app._display(*open_capture(tmp_path, "test_scene.ovd"))
    app.pending_playback = deepcopy(app.playback)
    app.pending_playback.set_mode(FrameMode.PRE)
    assert app._contact_guidance() == (True, 1)
    app.playback.clip.start = 2
    before = displayed(app)
    assert app._contact_guidance() == (True, None)
    assert not app._go_contact_pre() and displayed(app) == before
    app.pending_playback = None
    app.playback.clip.start = 0
    assert app.seek(0, snap=False)
    assert app._contact_guidance() == (False, None)
    assert app.seek(1, snap=False)
    assert app.playback.mode == FrameMode.POST
    assert app._contact_guidance() == (False, None)
    assert app.seek(2, snap=False)
    app.overlay_mask = 0
    assert app._contact_guidance() == (False, None)
    assert app._display(*open_capture(tmp_path))
    app.overlay_mask = 32
    assert app._contact_guidance() == (False, None)


@pytest.mark.parametrize("playback_mode", [FrameMode.POST, FrameMode.FULL])
def test_contact_shortcut_pauses_during_deferred_live_publication(app, live, playback_mode):
    source, connection = live
    app.overlay_mask = 32
    app.workflow.live_session = source
    connection.sendall(HANDSHAKE + PAYLOAD[:FRAME2_END])
    wait_until(lambda: source.info().completed_serial == 2)
    assert app._display(None, source)
    assert app.frame == 2
    app.playback.set_mode(playback_mode)
    before = displayed(app)
    connection.sendall(PAYLOAD[FRAME2_END:])
    wait_until(lambda: source.info().recording_size == len(PAYLOAD)
               and source.info().unpublished_changes)
    app.workflow.follow_live = True
    app.playback.toggle()
    assert not app._go_contact_pre()
    assert not app.playback.playing and not app.workflow.follow_live
    assert displayed(app) == before
    expected = FrameMode.FULL if playback_mode == FrameMode.FULL else FrameMode.PRE
    assert app.pending_playback.frame == 1 and app.pending_playback.mode == expected
    assert app._contact_guidance() == (True, 1)
    source.stop_live()
    assert app._refresh()
    assert app.frame == app.playback.frame == 1
    assert app.playback.mode == expected
    assert app._contact_guidance() == (False, None)

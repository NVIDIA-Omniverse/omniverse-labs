# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Only an actually adopted new recording segment clears scoped inspection."""
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from ovd_ui.playback import FrameMode
from ovd_ui.row_actions import RowIdentity
from test_application_presentation import HANDSHAKE, ROOT, live, wait_until
from test_visual_reset import app, base_app


@pytest.mark.parametrize("follow", [False, True])
def test_completed_display_segment_clears_selection_history_not_camera_or_policy(app, live, follow):
    source, connection = live
    payload = (ROOT / "test/data/resample_two_segment.ovd").read_bytes()
    first_end = 42117
    second_first_end = payload.index(struct.pack("<BQQ", 12, 2929590687920, 1), first_end) + 17
    connection.sendall(HANDSHAKE + payload[:first_end])
    wait_until(lambda: source.info().completed_serial == 120)
    app.workflow.live_session = source
    assert app._display(None, source)
    old_scene = app.scene_id
    old_segment = next(s.recording_segment for s in app.scenes if s.id == old_scene)
    app.playback.mode = FrameMode.PRE
    assert app.seek(119)
    actor = next(int(value) for value in app.presentation.geometry_snapshot().object_ids if value > 0)
    app.selection.publish([f"/objects/{actor}"], source="test")
    app.find_inspection = (source.capture_key, actor)
    app.frame_summary = True
    app.row_actions.select((RowIdentity(source.capture_key, "actor", "object", instance_id=actor),))
    app.interaction.set_visible([actor], False)
    camera, settings = app.navigation.camera, app.render_settings
    position, target = camera.position.copy(), camera.target.copy()
    assert app.interaction.can_back and app.selection.get_snapshot().paths()
    connection.sendall(payload[first_end:second_first_end])
    wait_until(lambda: source.info().completed_serial == 121)
    update = source.refresh_live()
    assert next(s.recording_segment for s in update.scenes if s.id == old_scene) == old_segment
    assert any(s.recording_segment != old_segment for s in update.scenes)
    # Ingestion advanced, but an inspector refresh still displays the old scene.
    assert app._refresh()
    assert app.selection.get_snapshot().paths() and app.interaction.can_back
    app.workflow.follow_live = follow
    app.workflow.poll = lambda: [SimpleNamespace(kind="frames", session=source, scenes=update.scenes,
        segment_changed=True, max_frame=update.max_frame, current_frame=update.current_frame)]
    app._poll_workflow()
    if follow:
        assert app.scene_id != old_scene
        assert not app.selection.get_snapshot().paths() and not app.interaction.sticky
        assert not app.selection_history and not app.interaction.can_back and not app.interaction.can_forward
        assert app.history_index == -1 and not app.row_actions.selected
        assert app.find_inspection is None and not app.frame_summary
    else:
        assert app.scene_id == old_scene and app.selection.get_snapshot().paths()
        assert app.interaction.can_back and app.find_inspection is not None
    assert actor in app.interaction.hidden
    assert app.playback.mode == FrameMode.PRE
    assert app.render_settings is settings
    np.testing.assert_array_equal(camera.position, position)
    np.testing.assert_array_equal(camera.target, target)

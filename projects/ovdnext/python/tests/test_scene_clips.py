# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scene changes retain capture clip edits through real native publications."""
import pytest
from pathlib import Path
import subprocess
from _pvd_native import Session

from test_application_presentation import app, open_capture, Value
from native_fixtures import executable


@pytest.mark.parametrize('current', [0, 1])
def test_scene_selection_retains_clips_and_updates_current_scene_range(app, tmp_path, current):
    assert app._display(*open_capture(tmp_path, 'ue4_shooter.pxd2'))
    assert app.seek(4, snap=False)
    assert app._clip_action(app.playback.split)
    assert app._clip_action(app.playback.select, 1)
    assert app.seek(8, snap=False)
    assert app._clip_action(app.playback.trim_end)
    assert app._clip_action(app.playback.select, 0)
    assert app.seek(2, snap=False)
    assert app._clip_action(app.playback.select, current)
    before = [(c.name, c.start, c.stop, c.frame) for c in app.playback.clips]
    target = next(i for i in range(len(app.scenes)) if i != app.scene_index)
    scene = app.scenes[target]
    app._select_scene(Value(target), None)
    assert app.scene_id == app.presentation.scene_id == scene.id
    assert len(app.playback.clips) == 2
    assert app.playback.index == current
    for index, clip in enumerate(app.playback.clips):
        expected = before[index]
        if index == current:
            expected = (expected[0], expected[1], scene.frame_count, expected[1])
        assert (clip.name, clip.start, clip.stop, clip.frame) == expected
    assert app.frame == app.playback.frame == app.presentation.frame == before[current][1]
    assert app.renderer.scene_id == scene.instance_id
    assert not app.pending_playback and not app.pending_presentation

    # Retaining edits within a capture must not transfer them to another file.
    assert app._display(*open_capture(tmp_path, 'cloth_physx34.pxd2'))
    assert [(c.name, c.start, c.stop) for c in app.playback.clips] == [('Capture', 1, 772)]


def test_shorter_native_scene_bounds_retained_clips_and_navigation(app, tmp_path):
    binary=executable('pvd_ingestion_fixture','PVD_INGESTION_FIXTURE')
    path=tmp_path/'different-scene-ranges.pxd2'
    subprocess.run([str(binary),str(path),'19','5'],check=True,capture_output=True,timeout=10)
    source=Session(str(path),str(tmp_path))
    assert [s.frame_count for s in source.scenes()]==[20,6]
    assert app._display(path,source)
    assert app.seek(10,snap=False) and app._clip_action(app.playback.split)
    assert app._clip_action(app.playback.select,1)
    assert app.seek(16,snap=False) and app._clip_action(app.playback.trim_start)
    app._select_scene(Value(1),None)
    assert app.scene_id==app.presentation.scene_id==200
    assert app.playback.index==1 and len(app.playback.clips)==2
    assert [(c.start,c.stop) for c in app.playback.clips]==[(1,6),(5,6)]
    assert app.frame==app.presentation.frame==app.playback.frame==5
    app._transport(app.playback.step,1)
    assert app.frame==5
    assert app._clip_action(app.playback.select,0)
    assert app.frame==app.presentation.frame==1

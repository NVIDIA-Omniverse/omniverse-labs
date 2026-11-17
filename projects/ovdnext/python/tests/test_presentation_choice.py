# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from _pvd_native import Session
from ovd_ui.presentation import default_scene_index


def test_ue4_opens_the_scene_that_owns_rendered_geometry(tmp_path):
    root=Path(__file__).resolve().parents[2]
    source=Session(str(root/"test/data/ue4_shooter.pxd2"),str(tmp_path))
    view=source.presentation(frame=1)
    scenes=view.scenes()
    geometry=view.geometry_snapshot()
    chosen=scenes[default_scene_index(scenes,geometry)]
    assert scenes[0].instance_id not in geometry.scene_ids
    assert chosen.instance_id==1143
    assert chosen.instance_id in geometry.scene_ids
    applied=source.presentation(chosen.id,1)
    assert applied.scene_id==chosen.id and applied.frame==1

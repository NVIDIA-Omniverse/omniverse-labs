# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from capture_semantics import test_scene_roles, unique_instance_id
from pathlib import Path

import numpy as np
from _pvd_native import Session
from ovd_ui.adapters import PvdDataAdapter

ROOT = Path(__file__).resolve().parents[2]


def test_native_geometry_and_pose_are_preserved(tmp_path):
    session=Session(str(ROOT / "test/data/test_scene.ovd"),str(tmp_path))
    session.seek(session.scenes()[0].id,1)
    adapter=PvdDataAdapter(session)
    objects,unsupported=adapter.draw_objects()
    assert not unsupported
    assert len(objects)==1
    sphere=objects[0]
    assert sphere.kind=="sphere"
    assert sphere.dimensions==(10.,)
    assert adapter.name(sphere.instance_id)=="sphere"
    np.testing.assert_allclose(sphere.matrix[:3,3],[420.,69.69,42.],rtol=1e-5)
    props={p.name:p for p in adapter.properties(sphere.instance_id)}
    assert adapter.value(sphere.instance_id,props['name'])=='sphere'
    assert adapter.value(sphere.instance_id,props['mass'])=='2'
    shape=unique_instance_id(session,'PxShape')
    assert adapter.value(sphere.instance_id,props['shapes'])==f'PxShape [{shape}]'


def test_end_of_capture_removes_dead_geometry(tmp_path):
    session=Session(str(ROOT / "test/data/test_scene.ovd"),str(tmp_path))
    session.seek(session.scenes()[0].id,1)
    adapter=PvdDataAdapter(session)
    assert adapter.draw_objects()[0]
    session.seek(session.scenes()[0].id,3)
    adapter.reload()
    assert not adapter.draw_objects()[0]

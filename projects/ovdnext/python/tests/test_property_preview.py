# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unknown property previews keep native totals without enlarging owned reads."""
from pathlib import Path
from types import SimpleNamespace

import pytest
from _pvd_native import Session

from ovd_ui.adapters import PvdDataAdapter
from ovd_ui.widgets import properties
from capture_semantics import unique_instance_id


ROOT=Path(__file__).resolve().parents[2]


def test_plane_inspector_explains_pose_based_normal_without_inventing_properties(tmp_path, monkeypatch):
    source = Session(str(ROOT / "test/data/multi_geom.ovd"), str(tmp_path))
    identity = unique_instance_id(source, "PxPlaneGeometry")
    view = source.presentation(frame=2, inspect_ids=[identity])
    adapter = PvdDataAdapter(view)
    assert not adapter.properties(identity)
    labels = []
    factory = properties.ui.Label
    def label(text, **kwargs):
        labels.append(text)
        return factory(text, **kwargs)
    monkeypatch.setattr(properties.ui, "Label", label)
    inspector = properties.PropertyInspector(lambda *_: None, lambda *_: None, {})
    frame = properties.ui.Frame()
    try:
        with frame:
            inspector.build(adapter, [identity])
        explanation = " ".join(labels)
        assert "local normal is +X" in explanation
        assert "localPose" in explanation and "globalPose" in explanation
        assert not inspector._rows
    finally:
        inspector.destroy()
        frame.destroy()


@pytest.mark.parametrize("name,total",[("SceneQueries.Hits",280),("SceneQueries.Raycasts",320)])
def test_real_cloth_unknown_array_preview_keeps_native_total(tmp_path,monkeypatch,name,total):
    source=Session(str(ROOT/"test/data/cloth_physx34.pxd2"),str(tmp_path))
    initial=source.presentation(frame=1)
    identity=next(scene.instance_id for scene in initial.scenes() if scene.id==initial.scene_id)
    view=source.presentation(frame=1,inspect_ids=[identity])
    adapter=PvdDataAdapter(view)
    prop=next(prop for prop in adapter.properties(identity) if prop.name==name)
    assert view.property_size(identity,prop.id)==total
    expected=view.read(identity,prop.id,limit=32).hex(" ")
    reads=[]
    class ObservedView:
        def read(self,*args,**kwargs):
            data=view.read(*args,**kwargs)
            reads.append((args,kwargs,len(data)))
            return data
        def __getattr__(self,name):return getattr(view,name)
    adapter.session=ObservedView()
    labels=[]
    monkeypatch.setattr(properties,"ui",SimpleNamespace(Label=lambda text,**_:labels.append(text)))
    inspector=properties.PropertyInspector(lambda *_:None,lambda *_:None,{})
    inspector._heading=lambda *_:None
    inspector._property(adapter,identity,prop)
    assert labels==[expected+f" … [32 of {total} bytes]"]
    assert reads==[((identity,prop.id),{"limit":256},256)]

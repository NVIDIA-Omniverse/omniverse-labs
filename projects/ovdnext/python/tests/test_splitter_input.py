# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gesture arbitration when two input sequences precede the next UI layout."""
from types import SimpleNamespace
import pytest
from ovd_ui.widgets import splitter


class Widget:
    def __init__(self,**kwargs):self.__dict__.update(kwargs)
    def __enter__(self):return self
    def __exit__(self,*_):pass


@pytest.mark.parametrize("direction",(1,-1))
def test_new_press_cancels_previous_unapplied_release_before_anchoring(monkeypatch,direction):
    # Only geometry layout is deferred here. Production callbacks and their
    # public update boundary handle two real user intents in one UI frame.
    monkeypatch.setattr(splitter,"ui",SimpleNamespace(ZStack=Widget,HStack=Widget,Spacer=Widget,Rectangle=Widget,
        InvisibleButton=Widget,Pixel=float))
    target=SimpleNamespace(width=320.,computed_width=320.)
    saved=[]
    grip=splitter.SplitGrip(target,saved.append,direction=direction)
    grip.press(100,0,0,0)
    grip.release(180,0,0)
    grip.press(200,0,0,0)
    grip.update()
    assert target.width==320. and saved==[]
    grip.move(210,0)
    grip.update()
    assert target.width==320+10*direction
    target.computed_width=target.width
    grip.update()
    assert saved==[]
    grip.release(220,0,0)
    grip.update()
    assert target.width==320+20*direction and saved==[]
    target.computed_width=target.width
    grip.update()
    assert saved==[320+20*direction]


@pytest.mark.parametrize("direction",(1,-1))
@pytest.mark.parametrize("axis",("x","y"))
def test_drag_remains_active_outside_hit_area_and_saves_only_after_release(monkeypatch,direction,axis):
    monkeypatch.setattr(splitter,"ui",SimpleNamespace(ZStack=Widget,HStack=Widget,VStack=Widget,Spacer=Widget,
        Rectangle=Widget,InvisibleButton=Widget,Pixel=float))
    extent="width" if axis=="x" else "height"
    target=SimpleNamespace(**{extent:320.,"computed_"+extent:320.})
    point=lambda p: (p,900) if axis=="x" else (900,p)
    saved=[]
    grip=splitter.SplitGrip(target,saved.append,direction=direction,axis=axis)
    assert not grip.rule.visible
    grip.hover(True)
    assert grip.rule.visible
    grip.press(*point(100),0,0)
    grip.hover(False)
    grip.move(*point(180))
    grip.update()
    setattr(target,"computed_"+extent,getattr(target,extent))
    grip.update()
    assert grip.rule.visible and grip.anchor is not None
    assert getattr(target,extent)==320+80*direction and saved==[]
    grip.release(*point(180),0)
    assert not grip.rule.visible
    grip.update()
    setattr(target,"computed_"+extent,getattr(target,extent))
    grip.update()
    assert saved==[320+80*direction]
    grip.destroy()
    grip.hover(True)
    grip.press(100,200,0,0)
    grip.move(400,200)
    grip.release(400,200,0)
    grip.update()
    assert saved==[320+80*direction]

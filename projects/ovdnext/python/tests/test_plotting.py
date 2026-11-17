# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PvdPlot's independent Qwt history, both-axis navigation and exact cursor."""
import pytest
from ovd_ui.plotting import Curve, PlotModel, PlotRange


def model():
    return PlotModel(range(10,111),[Curve('Value',tuple(float(i) for i in range(101)),'m')])


def test_rectangle_reverse_drag_stack_undo_redo_and_home():
    m=model();base=m.view
    assert m.rectangle(.8,.9,.2,.1)
    first=m.view
    assert first.left==30 and first.right==90
    assert first.bottom==pytest.approx(base.bottom+.1*base.height)
    assert first.top==pytest.approx(base.top-.1*base.height)
    m.rectangle(.25,.25,.75,.75);second=m.view
    assert m.history_index==2
    assert m.undo() and m.view==first
    assert m.redo() and m.view==second
    assert m.home() and m.view==base and m.history_index==0
    assert m.redo() and m.view==first
    m.rectangle(.3,.2,.6,.6)
    assert len(m.history)==3 and m.history[-1]!=second
    assert not m.redo()


def test_pan_and_centered_qwt_wheel_do_not_push_rectangle_history():
    m=model();base=m.view
    m.pan(.2,-.1)
    assert m.view.left==base.left-.2*base.width
    assert m.view.bottom==base.bottom-.1*base.height
    assert len(m.history)==1
    before=m.view;m.wheel(1)
    assert m.view.width==pytest.approx(before.width/.9)
    assert m.view.height==pytest.approx(before.height/.9)
    assert m.view.left+m.view.right==pytest.approx(before.left+before.right)
    m.home();assert m.view==base
    m.magnify(.9);assert m.view.width==pytest.approx(base.width*.9)
    assert not m.magnify(float('inf')) and not m.magnify(-1)


def test_exact_cursor_ties_missing_values_and_outside_capture():
    m=PlotModel((3,5,9),(Curve('a',(2.,None,8.),'ms'),Curve('b',(10.,20.,30.),'ms')))
    assert m.hover(1/6)==(('a',None,'ms'),('b',20.,'ms'))  # position4 tie goesforward5
    assert m.cursor_frame==5
    m.pan(1,0);assert m.hover(0) is None and m.cursor_frame is None
    m.home();assert m.hover(1)==(('a',8.,'ms'),('b',30.,'ms'))
    assert m.cursor_frame==9


def test_frame_range_validation_single_frame_and_all_missing():
    m=model();m.frame_range(30,50)
    assert (m.view.left,m.view.right)==(30,50)
    with pytest.raises(ValueError):m.frame_range(50,20)
    with pytest.raises(ValueError):m.frame_range(500,600)
    m.frame_range(40,40);assert m.view.left==39.5 and m.view.right==40.5
    m=PlotModel((8,),[Curve('missing',(None,))])
    assert not m.has_values and m.base.width==1
    assert m.display_paths(m.curves[0])==()
    assert m.hover(.5)==(('missing',None,''),)


def test_paths_preserve_original_samples_and_missing_gaps():
    m=PlotModel((0,1,2,3,4),[Curve('v',(1.,2.,None,4.,5.))])
    paths=m.display_paths(m.curves[0])
    assert len(paths)==2
    assert [x for path in paths for x,y in path]==[0,.25,.75,1]
    m.set_view(PlotRange(.5,3.5,1,5))
    paths=m.display_paths(m.curves[0])
    assert paths[0][0]==pytest.approx((0,.125))
    assert paths[1][-1]==pytest.approx((1,.875))


def test_large_curve_envelope_retains_narrow_spikes_and_does_not_bridge_gaps():
    values=[1.]*10000;values[155]=1000.;values[222]=None
    m=PlotModel(range(10000),[Curve('v',tuple(values))])
    paths=m.display_paths(m.curves[0],limit=100)
    assert sum(map(len,paths))<=400
    assert len(paths)==2
    assert not any(path[0][0]<222/9999<path[-1][0] for path in paths)
    assert max(y for path in paths for x,y in path)==pytest.approx((1000-m.base.bottom)/m.base.height)
    m.hover(155/9999);assert m.cursor_values()==(('v',1000.,''),)


def test_invalid_shapes_are_rejected_without_corrupting_zoom_history():
    with pytest.raises(ValueError):PlotModel((),[Curve('v',())])
    with pytest.raises(ValueError):PlotModel((1,1),[Curve('v',(1.,2.))])
    m=model();before=m.view
    assert not m.rectangle(.1,.1,.1,.1)
    assert not m.set_view(PlotRange(float('nan'),1,0,1),record=True)
    assert m.view==before and len(m.history)==1


def test_paths_clip_true_intersections_without_boundary_phantoms():
    curve=Curve('measurement',(0.,10.,0.))
    model=PlotModel((0,1,2),(curve,))
    model.set_view(PlotRange(0,2,4,6))
    paths=model.display_paths(curve)
    assert len(paths)==2
    assert paths[0][0]==pytest.approx((.2,0))
    assert paths[0][-1]==pytest.approx((.3,1))
    assert paths[1][0]==pytest.approx((.7,1))
    assert paths[1][-1]==pytest.approx((.8,0))
    model.set_view(PlotRange(0,2,20,30))
    assert model.display_paths(curve)==()


def test_paths_keep_spike_position_endpoint_and_missing_markers():
    curve=Curve('measurement',(0.,50.,None,9.))
    model=PlotModel((0,1,2,3),(curve,))
    model.magnify(1/.9)
    paths=model.display_paths(curve)
    r=model.view
    assert paths[0][1]==pytest.approx(((1-r.left)/r.width,(50-r.bottom)/r.height))
    assert len(paths[1])==1
    values=[0.]*10000
    values[713]=100.;values[-1]=80.
    large=Curve('large',tuple(values))
    model=PlotModel(range(10000),(large,))
    paths=model.display_paths(large,limit=20)
    assert any(x==pytest.approx(713/9999) and y==pytest.approx(105/110) for path in paths for x,y in path)
    assert paths[-1][-1][0]==1
    assert sum(map(len,paths))<=80
    assert paths[-1][-1][1]==pytest.approx(85/110)

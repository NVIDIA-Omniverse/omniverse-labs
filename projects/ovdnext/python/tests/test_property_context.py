# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Property menus copy owned display snapshots without additional native reads."""
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest
from _pvd_native import Session
from capture_semantics import unique_instance_id, dynamic_geometry_owner

from ovd_ui.adapters import PvdDataAdapter
from ovd_ui.widgets import properties
from ovd_ui.widgets.properties import DisplayedValue, PropertyInspector

ROOT=Path(__file__).resolve().parents[2]


class Widget:
    def __init__(self,kind,text="",**kwargs):
        self.kind,self.text,self.kwargs=kind,text,kwargs
        self.destroyed=False
        self.model=SimpleNamespace(set_value=lambda *_:None,add_value_changed_fn=lambda *_:None)
    def __enter__(self):return self
    def __exit__(self,*_):pass
    def set_build_fn(self,callback):self.build=callback
    def rebuild(self):self.build()
    def set_collapsed_changed_fn(self,callback):self.collapsed=callback
    def show_at(self,x,y):self.position=(x,y)
    def destroy(self):self.destroyed=True


@pytest.fixture
def display(monkeypatch):
    widgets=[]
    def factory(kind):
        def create(text="",**kwargs):
            widget=Widget(kind,text,**kwargs)
            widgets.append(widget)
            return widget
        return create
    names=("Label","Button","VStack","HStack","ZStack","Spacer","Rectangle",
           "CollapsableFrame","Frame","IntSlider","Menu","MenuItem")
    monkeypatch.setattr(properties,"ui",SimpleNamespace(Alignment=properties.ui.Alignment,
        **{name:factory(name) for name in names}))
    def find(text,kind="Label"):
        return next(widget for widget in reversed(widgets) if widget.kind==kind and widget.text==text)
    def open_menu(widget):
        widget.kwargs["mouse_pressed_fn"](43,57,1,0)
        assert find("Property value","Menu").position==(43,57)
        return {widget.text:widget.kwargs for widget in widgets[widgets.index(find("Property value","Menu"))+1:]
                if widget.kind=="MenuItem"}
    return SimpleNamespace(widgets=widgets,find=find,open_menu=open_menu)


def prepare(tmp_path,filename="test_scene.ovd",identity=None,frame=2):
    source=Session(str(ROOT/"test/data"/filename),str(tmp_path))
    source.presentation(frame=frame)
    if identity is None:
        identity=unique_instance_id(source,'PxRigidDynamic')
    elif identity == 'box':
        identity=dynamic_geometry_owner(source,2,frame)
    elif isinstance(identity,str):
        identity=unique_instance_id(source,identity)
    adapter=PvdDataAdapter(source.presentation(frame=frame,inspect_ids=[identity]))
    adapter.fixture_id=identity
    return source,adapter


class ObservedView:
    def __init__(self,view):self.view,self.reads,self.forbid=view,[],False
    def read(self,*args,**kwargs):
        assert not self.forbid,"Copy attempted to read native property data"
        result=self.view.read(*args,**kwargs)
        self.reads.append((args,kwargs,len(result)))
        return result
    def __getattr__(self,name):
        if self.forbid and name in ("property_size","string_value","references","named_values","array_record"):
            raise AssertionError("Copy attempted to read native metadata")
        return getattr(self.view,name)


def test_scalar_snapshot_menu_copy_and_plot_keep_exact_display_and_metadata(tmp_path,display):
    source,adapter=prepare(tmp_path)
    identity = adapter.fixture_id
    observed=ObservedView(adapter.session)
    adapter.session=observed
    copied,plotted=[],[]
    inspector=PropertyInspector(lambda *_:None,lambda *args:plotted.append(args),
        {"show_hex":True},on_copy_text=copied.append)
    inspector.capture_key=source.capture_key
    prop=adapter.property(identity,"environmentID")
    inspector._property(adapter,identity,prop)
    value=display.find("4294967295 (0xffffffff)")
    assert value.kwargs["explicit_hover"]
    menu=display.open_menu(value)
    assert set(menu)=={"Copy value","Plot value"}
    # Menu payload remains a string even if preferences and native frame change.
    inspector.preferences["show_hex"]=False
    source.presentation(frame=3)
    observed.forbid=True
    assert menu["Copy value"]["triggered_fn"]()
    assert copied==["4294967295 (0xffffffff)"]
    assert menu["Plot value"]["triggered_fn"]()
    assert plotted==[(identity,prop)]


def test_compound_parent_copies_immediate_children_and_child_copies_only_itself(tmp_path,display):
    source,adapter=prepare(tmp_path)
    identity = adapter.fixture_id
    copied=[]
    inspector=PropertyInspector(lambda *_:None,lambda *_:None,{},on_copy_text=copied.append)
    prop=adapter.property(identity,"globalPose")
    inspector._property(adapter,identity,prop)
    children=[widget.text for widget in display.widgets
              if widget.kind=="Label" and widget.kwargs.get("name")=="mono_small"]
    assert children==["421.548340, 48.335281, 48.839565","0.253113, 0.015109, 0.023405, 0.967036"]
    parent=display.open_menu(display.find("globalPose"))
    assert set(parent)=={"Copy value"}
    parent["Copy value"]["triggered_fn"]()
    assert copied==[children[0]+"; "+children[1]+"; "]
    display.open_menu(display.find(children[0]))["Copy value"]["triggered_fn"]()
    assert copied[-1]==children[0]
    # Immediate-child copy must not silently become recursive flattening.
    nested=DisplayedValue("outer",(DisplayedValue("child",(DisplayedValue("grandchild"),)),))
    assert nested.copy_text=="child; "


def test_real_cloth_array_menu_copies_only_loaded_window_and_survives_rebuild(tmp_path,display):
    source,adapter=prepare(tmp_path,"cloth_physx34.pxd2",16,1)
    prop=adapter.property(16,"ParticleIndices")
    assert prop.id==583
    view=source.presentation(frame=1,inspect_ids=[16],read_requests=[(16,prop.id,3810*4,40)])
    observed=ObservedView(view)
    adapter.session=observed
    copied=[]
    inspector=PropertyInspector(lambda *_:None,lambda *_:None,{"array_group_size":10,"max_array_to_load":99999},on_copy_text=copied.append)
    inspector.array_positions[(16,prop.id)]=3810
    inspector._property(adapter,16,prop)
    frame=next(widget for widget in display.widgets if widget.kind=="Frame")
    frame.rebuild()
    assert observed.reads==[((16,583),{"offset":15240,"limit":40},40)]
    menu=display.open_menu(display.find("ParticleIndices"))
    # Real native values already verified independently in Inspector85–94.
    expected="".join(f"{value}; " for value in struct.unpack("<10I",view.read(16,583,offset=15240,limit=40)))
    inspector.array_positions[(16,prop.id)]=0
    frame.rebuild()
    assert observed.reads[-1]==((16,583),{"offset":0,"limit":40},40)
    observed.forbid=True
    assert menu["Copy value"]["triggered_fn"]()
    assert copied==[expected]
    assert all(read[2]==40 for read in observed.reads)


def test_unknown_preview_copy_retains_truncation_marker_and_native_total(tmp_path,display):
    _,adapter=prepare(tmp_path,"cloth_physx34.pxd2",6,1)
    copied=[]
    inspector=PropertyInspector(lambda *_:None,lambda *_:None,{},on_copy_text=copied.append)
    prop=adapter.property(6,"SceneQueries.Raycasts")
    observed=ObservedView(adapter.session)
    adapter.session=observed
    inspector._property(adapter,6,prop)
    preview=next(widget for widget in display.widgets if "[32 of 320 bytes]" in widget.text)
    observed.forbid=True
    display.open_menu(preview)["Copy value"]["triggered_fn"]()
    assert copied==[preview.text]
    assert observed.reads==[((6,prop.id),{"limit":256},256)]


def test_capture_replacement_retains_copy_text_but_rejects_stale_plot(tmp_path,display):
    source,adapter=prepare(tmp_path)
    identity = adapter.fixture_id
    copied,plotted,errors=[],[],[]
    inspector=PropertyInspector(lambda *_:None,lambda *args:plotted.append(args),{},
        on_copy_text=copied.append,on_error=errors.append)
    inspector.capture_key=source.capture_key
    inspector._property(adapter,identity,adapter.property(identity,"mass"))
    menu=display.open_menu(display.find("mass"))
    other_source,other=prepare(tmp_path/"other")
    inspector.build(other,[other.fixture_id])
    assert inspector.capture_key==other_source.capture_key!=source.capture_key
    assert menu["Copy value"]["triggered_fn"]() and copied==["2.000000"]
    assert not menu["Plot value"]["triggered_fn"]() and not plotted
    assert errors==["The capture changed. Open the current property's menu to plot it."]


def test_copy_failures_and_destroyed_or_superseded_menu_callbacks(tmp_path,display):
    _,adapter=prepare(tmp_path)
    identity = adapter.fixture_id
    copied,errors=[],[]
    def fail(text):raise RuntimeError("Desktop clipboard unavailable")
    inspector=PropertyInspector(lambda *_:None,lambda *_:None,{},on_copy_text=fail,on_error=errors.append)
    inspector._property(adapter,identity,adapter.property(identity,"name"))
    menu=display.open_menu(display.find("sphere"))
    assert set(menu)=={"Copy value"}
    action=menu["Copy value"]["triggered_fn"]
    assert not action()
    assert errors==["Unable to copy value: Desktop clipboard unavailable"]
    inspector.on_copy_text=copied.append
    next_action=display.open_menu(display.find("sphere"))["Copy value"]["triggered_fn"]
    assert not action() and not copied
    assert next_action() and copied==["sphere"] and not inspector.copy_error
    window=inspector.menu
    inspector.destroy()
    assert window.destroyed and inspector.menu is None
    assert not next_action() and copied==["sphere"]


@pytest.mark.parametrize("clear_capture",[False,True])
def test_empty_selection_capture_change_invalidates_open_plot_but_keeps_copy(tmp_path,display,clear_capture):
    source,adapter=prepare(tmp_path)
    identity = adapter.fixture_id
    copied,plotted,errors=[],[],[]
    inspector=PropertyInspector(lambda *_:None,lambda *args:plotted.append(args),{},
        on_copy_text=copied.append,on_error=errors.append)
    inspector.build(adapter,[identity])
    menu=display.open_menu(display.find("mass"))
    other_source,other=prepare(tmp_path/"replacement")
    inspector.build(None if clear_capture else other,[])
    assert inspector.capture_key==(None if clear_capture else other_source.capture_key)
    assert menu["Copy value"]["triggered_fn"]() and copied==["2.000000"]
    assert not menu["Plot value"]["triggered_fn"]() and not plotted
    assert errors==["The capture changed. Open the current property's menu to plot it."]


def test_empty_recorded_string_copies_the_visible_dash_placeholder(tmp_path,display):
    _,adapter=prepare(tmp_path,"multi_geom.ovd",'PxParticleBuffer',1)
    identity=adapter.fixture_id
    prop=adapter.property(identity,"name")
    assert adapter.session.string_value(identity,prop.id)==""
    copied=[]
    inspector=PropertyInspector(lambda *_:None,lambda *_:None,{},on_copy_text=copied.append)
    inspector._property(adapter,identity,prop)
    display.open_menu(display.find("-"))["Copy value"]["triggered_fn"]()
    assert copied==["-"]


@pytest.mark.parametrize("kind,value,expected",[("PvdU64",2**64-1,"18446744073709551615"),
    ("PvdI64",-(2**63),"-9223372036854775808")])
def test_64_bit_scalar_menu_never_coerces_through_float(display,kind,value,expected):
    copied=[]
    inspector=PropertyInspector(lambda *_:None,lambda *_:None,{},on_copy_text=copied.append)
    inspector._value(kind,struct.pack("<Q" if kind=="PvdU64" else "<q",value))
    display.open_menu(display.find(expected))["Copy value"]["triggered_fn"]()
    assert copied==[expected]

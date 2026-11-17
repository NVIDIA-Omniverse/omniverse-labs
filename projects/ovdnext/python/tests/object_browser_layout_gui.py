# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real OVUI TreeView resize regression, using an owned native capture view."""
from capture_semantics import test_scene_roles, multi_geom_roles, unique_instance_id, dynamic_geometry_owner
from gui_runtime import isolate_gui_output, hide_owned_window
from pathlib import Path
import tempfile

import omni.ui as ui
from _pvd_native import Session
from ovd_ui import style
from ovd_ui.adapters import PvdDataAdapter
from ovd_ui.models import ObjectTreeModel
from ovd_ui.widgets import object_browser as browser_module
from ovd_ui.widgets.object_browser import ObjectBrowser, _ObjectDelegate

ROOT=Path(__file__).resolve().parents[2]


class RecordingDelegate(_ObjectDelegate):
    """Retain the public widgets constructed by the production delegate."""
    def __init__(self,browser):
        super().__init__(browser)
        self.rows={}
        self.build_count=0

    def build_widget(self,model,item,column_id,level,expanded):
        self.build_count+=1
        label_factory,frame_factory=ui.Label,ui.Frame
        record={}
        def label(*args,**kwargs):
            result=label_factory(*args,**kwargs)
            if kwargs.get("name")=="object_name":record["label"]=result
            if kwargs.get("name")=="mono_small":record["identity"]=result
            return result
        def frame(*args,**kwargs):
            result=frame_factory(*args,**kwargs)
            # The passive selection caps are also clipped Frames. Measure the
            # original interactive text allocation, not a decorative half-cap.
            if kwargs.get("horizontal_clipping") and kwargs.get("mouse_pressed_fn"):
                record["frame"]=result
            return result
        ui.Label,ui.Frame=label,frame
        try:super().build_widget(model,item,column_id,level,expanded)
        finally:ui.Label,ui.Frame=label_factory,frame_factory
        if record and item is not None:self.rows[item.key]=record


async def verify():
    with tempfile.TemporaryDirectory() as temporary:
        source=Session(str(ROOT / "test/data/multi_geom.ovd"),temporary)
        adapter=PvdDataAdapter(source.presentation(frame=1))
        model=ObjectTreeModel();model.populate(adapter)
        window=ui.Window("Object row regression",width=640,height=700,
            flags=ui.WINDOW_FLAGS_NO_TITLE_BAR)
        with window.frame:
            outer=ui.Frame(width=264,height=660,horizontal_clipping=True,vertical_clipping=True)
            with outer:
                browser_module._ObjectDelegate=RecordingDelegate
                try:browser=ObjectBrowser(model,lambda _items:None,on_action=lambda *_:None)
                finally:browser_module._ObjectDelegate=_ObjectDelegate
        delegate=browser.delegate
        identity=unique_instance_id(adapter.session,'PxArticulationReducedCoordinate')
        target=model.canonical_item(identity)
        assert target.class_name=="PxArticulationReducedCoordinate"
        for _ in range(3):await ui.next_frame()
        browser.select_item(target)
        assert not browser.clear_search_host.visible
        snapshots=[]
        for width in (264,420,200,140,420):
            outer.width=ui.Pixel(width)
            window.frame.rebuild()
            for _ in range(10):await ui.next_frame()
            row=delegate.rows[target.key]
            assert model.canonical_item(identity) is target
            assert browser.tree.selection==[target]
            assert abs(outer.computed_width-width)<1
            budget=max(0,row["frame"].computed_width-row["identity"].computed_width-12)
            actual=row["label"].computed_width
            assert actual<=budget+1,(width,actual,budget,str(row["label"].width))
            assert row["label"].text=="PxArticulationReducedCoordinate"
            assert row["label"].elided_text
            assert row["frame"].horizontal_clipping
            # The production eye/tag columns must remain visible after the
            # name gives up space, including widths below the app minimum.
            controls=[browser._tag_controls[(target.key,column)][1].button for column in (2,3)]
            assert all(control.computed_width>0 for control in controls)
            assert controls[0].screen_position_x+controls[0].computed_width<=controls[1].screen_position_x+1
            assert controls[1].screen_position_x+controls[1].computed_width<=outer.screen_position_x+width+1
            snapshots.append((width,actual,row["frame"].computed_width))
        assert snapshots[1][1]>=snapshots[2][1]>=snapshots[3][1]
        assert snapshots[1][1]>snapshots[3][1] and snapshots[3][1]==0
        assert snapshots[1][2]>snapshots[2][2]>snapshots[3][2]
        assert snapshots[1][1:]==snapshots[4][1:]
        # A style refresh must not freeze the old truncated measurement.
        style.set_theme("light")
        for _ in range(8):await ui.next_frame()
        assert abs(delegate.rows[target.key]["label"].computed_width-snapshots[1][1])<1
        builds=delegate.build_count
        for _ in range(10):await ui.next_frame()
        assert delegate.build_count==builds,"Unchanged rows must not rebuild continuously"
        browser.search_model.set_value("PxRigidStatic")
        for _ in range(6):await ui.next_frame()
        assert browser.clear_search_host.visible
        searched_width=browser.search_field.computed_width
        browser.search_model.set_value("")
        for _ in range(6):await ui.next_frame()
        assert not browser.clear_search_host.visible
        assert browser.search_field.computed_width>searched_width
        print("Native tree resize checks passed",snapshots,flush=True)
        window.destroy()
        for _ in range(2):await ui.next_frame()


if __name__=="__main__":
    errors=[]
    async def run():
        try:await verify()
        except Exception as error:errors.append(error)
        finally:ui.shutdown()
    isolate_gui_output()
    ui.init("Object row regression",width=640,height=700,max_fps=60)
    hide_owned_window()
    style.startup("dark")
    ui.run(run())
    if errors:raise errors[0]

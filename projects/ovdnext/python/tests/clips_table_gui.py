# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native OVUI model/layout/menu lifecycle, not a screenshot-driven user flow."""
from gui_runtime import isolate_gui_output, hide_owned_window
import omni.ui as ui

from ovd_ui import style
from ovd_ui.playback import Playback
from ovd_ui.widgets.clips_view import ClipsView
from ovd_ui.widgets.workspace_tabs import PageHost
from types import SimpleNamespace


async def verify():
    window=ui.Window('Clip table regression',width=850,height=400)
    with window.frame:frame=ui.Frame()
    owner=SimpleNamespace(active='clips',expanded=True,window=window)
    host=PageHost(owner,'clips',frame)
    playback=Playback();source=(1,0,99);calls=[]
    def action(scope,kind,row):
        calls.append((scope,kind,row.index))
        if kind!='save':
            getattr(playback,kind)(row.index) if kind in ('select','delete') else getattr(playback,kind)()
            view.refresh(playback,source,'')
    view=ClipsView(action,host)
    async def frames():
        for _ in range(5):
            view.poll();await ui.next_frame()
    view.refresh(playback,None,'Open a capture before exporting.')
    await frames()
    assert not view.model.items and not any(b.enabled for b in view.buttons.values())
    playback.reset(12);view.refresh(playback,source,'');await frames()
    assert view.table.headers==('Name','Start Frame','End Frame','Frame Count')
    assert all(button is None for button in view.table.buttons)
    assert view.model.items[0].row.values==('Capture','1','11','11')
    only=view.model.items[0].row
    view.show_context_menu(only,100,100);await frames()
    assert view.menu_actions==('save',) and not view.buttons['delete'].enabled
    view.close_menu()
    playback.seek(4);view._current_action('split');await frames()
    assert [i.row.values for i in view.model.items]==[('Capture_1','1','4','4'),('Capture_2','5','11','7')]
    first,second=view.model.items
    view.tree.selection=[second];await frames()
    assert playback.index==1 and playback.frame==5 and second.current
    playback.seek(8);view.refresh(playback,source,'');await frames()
    assert view.model.items==[first,second]
    view.tree.selection=[first];await frames()
    assert playback.frame==1
    view.tree.selection=[second];await frames()
    assert playback.frame==8
    # Right-click menu payload can address the other row without selection.
    view.show_context_menu(first.row,100,100);await frames()
    assert view.menu_actions==('save','delete')
    generation=view.menu_generation
    view._menu_action('save',first.row,generation)
    assert calls[-1]==(source,'save',0) and playback.index==1 and playback.frame==8
    view.close_menu()
    grip=view.table.grips[0];grip.press(200,100,0,0);grip.release(290,100,0)
    await frames();widths=view.table.displayed
    playback.seek(9);view.refresh(playback,source,'');await frames()
    assert view.model.items==[first,second] and view.table.displayed==widths
    style.startup('light');await frames()
    assert view.table.displayed==widths and [float(x) for x in view.tree.column_widths]==list(widths)
    owner.active='messages';view.poll();owner.active='clips';await frames()
    assert view.table.displayed==widths and view.tree.selection==[second]
    view.show_context_menu(first.row,100,100);generation=view.menu_generation
    previous=len(calls)
    # An identical-looking replacement source must invalidate both open menus
    # and delegate callbacks belonging to its predecessor.
    source=(2,0,99);view.refresh(playback,source,'');await frames()
    view._menu_action('delete',first.row,generation)
    view.show_context_menu(first.row,100,100)
    assert len(calls)==previous and view.menu is None and len(playback.clips)==2
    first=view.model.items[0]
    view._action('delete',first.row);await frames()
    assert playback.index==0 and playback.frame==9 and len(view.model.items)==1
    assert not view.buttons['delete'].enabled
    playback.seek(10);view._current_action('trim_start');await frames()
    playback.seek(11);view._current_action('trim_end');await frames()
    assert view.model.items[0].row.values==('Capture_2','10','10','1')
    assert not any(b.enabled for b in view.buttons.values())
    view.refresh(playback,source,'OVD clip export is unsupported')
    view.show_context_menu(view.model.items[0].row,100,100);await frames()
    previous=len(calls);view._menu_action('save',view.model.items[0].row,view.menu_generation)
    assert len(calls)==previous
    playback.reset();view.refresh(playback,None,'Open a capture before exporting.');await frames()
    assert not view.model.items and not view.tree.selection and view.menu is None
    view.destroy();window.destroy()
    print('Native clip table checks passed',flush=True)


if __name__=='__main__':
    errors=[]
    async def run():
        try:await verify()
        except Exception as error:errors.append(error)
        finally:ui.shutdown()
    isolate_gui_output()
    ui.init('Clip table regression',width=850,height=400,max_fps=60)
    hide_owned_window()
    style.startup('dark');ui.run(run())
    if errors:raise errors[0]

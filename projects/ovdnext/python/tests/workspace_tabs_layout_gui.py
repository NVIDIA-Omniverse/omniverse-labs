# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native OVUI lifecycle regression, separate from screenshot-first application QA."""
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import omni.ui as ui
from ovd_ui import style
from ovd_ui.analysis import HistoryRequest, ScalarSeries
from ovd_ui.widgets.analysis_views import PlotView
from ovd_ui.widgets.workspace_tabs import FIXED_TABS, WorkspaceTabs, page_surface
from gui_runtime import isolate_gui_output


class Job:
    def __init__(self):
        self.future=Future()
        self.cancelled=0
        self.progress=(0,3)
    def cancel(self):self.cancelled+=1


async def verify():
    window=ui.Window('Workspace lifecycle regression',width=920,height=720)
    views=[]
    disposed=[]
    def dispose(view):
        disposed.append(view)
        views.remove(view)
        view.destroy()
    def activate(page):
        if page.key=='profile' and page.view is None and page.enabled:
            page.view=graph(page.host)
            page.source=tabs.source
    def graph(host):
        view=PlotView(Job(),'Recorded values',lambda _:None,page_surface,host=host)
        views.append(view)
        return view
    with window.frame:
        with ui.HStack():
            center=ui.VStack(width=604,spacing=8)
            with center:
                tabs=WorkspaceTabs(window,activate,dispose)
                with tabs.render_frame:ui.Label('Render surface')
            ui.Spacer()
    states={key:SimpleNamespace(enabled=True,reason='') for key in ('profile','memory','errors')}
    source=(1,0,Path('/capture.pxd2'),3,1)
    tabs.sync(source,states)
    async def frames(count=6):
        for _ in range(count):
            tabs.poll()
            for view in tuple(views):view.poll()
            await ui.next_frame()
    await frames()
    assert tuple(tabs.pages)==tuple(key for key,_ in FIXED_TABS)
    assert not any(tabs.close(key) for key,_ in FIXED_TABS)
    # Unchecked tabs can be selected without pretending native data is ready.
    states['profile']=SimpleNamespace(enabled=False,reason='Unchecked',requestable=True)
    tabs.sync(source,states);await frames()
    assert tabs.buttons['profile'].enabled and tabs.select('profile')
    await frames()
    assert tabs.active=='profile' and tabs.pages['profile'].view is None and not views
    states['profile']=SimpleNamespace(enabled=False,reason='No profile events',requestable=False)
    tabs.sync(source,states);await frames()
    assert not tabs.buttons['profile'].enabled and not tabs.select('profile')
    states['profile']=SimpleNamespace(enabled=True,reason='')
    tabs.sync(source,states)
    assert tabs.select('profile')
    await frames()
    fixed=tabs.pages['profile'].view
    assert fixed is not None and fixed.job.cancelled==0
    tabs.select('render');await frames()
    assert not fixed.window.visible and not fixed.finished and fixed.job.cancelled==0
    series=ScalarSeries(HistoryRequest(Path('/capture.pxd2'),3,1,1),'Value','float',(1,2,3),(2.,6.,4.))
    fixed.job.future.set_result(series)
    await frames()
    assert fixed.finished and fixed.model.has_values and fixed.job.cancelled==0
    tabs.select('profile');await frames()
    assert tabs.pages['profile'].view is fixed
    first=tabs.add('Independent graph A',graph);await frames()
    first.job.future.set_result(series);await frames()
    first.window.focus();await frames()
    assert first.window.keyboard_focus
    second=tabs.add('Independent graph B',graph);await frames()
    assert first.job.cancelled==0 and not first.window.visible
    second.job.future.set_result(series);await frames()
    first.model.frame_range(1,2)
    saved=first.model.view
    assert first.model.view!=second.model.view
    tabs.select(first.window.key);await frames()
    assert first.model.view==saved and not second.destroyed
    assert tabs.tab_scroll.scroll_x>0  # Active overflowed tab is revealed.
    center.width=ui.Pixel(240);await frames()
    assert first.window.frame.computed_width<=240
    assert first.window.content.computed_width>=480
    button=tabs.buttons[first.window.key]
    assert button.screen_position_x+button.computed_width+22<=tabs.tab_scroll.screen_position_x+tabs.tab_scroll.computed_width+2
    assert first.window.scroll.scroll_x_max>0
    style.startup('light');await frames()
    assert first.model.view==saved
    tabs.close(second.window.key);await frames()
    assert second.destroyed and second.job.cancelled==1
    tabs.close(first.window.key);await frames()
    assert first.destroyed and first.job.cancelled==1 and tabs.active=='threads'
    # A changed frame disposes stale fixed data but keeps generated histories.
    third=tabs.add('Pending graph C',graph);await frames()
    tabs.sync((*source[:-1],2),states);await frames()
    assert fixed.destroyed and fixed.job.cancelled==1 and third.job.cancelled==0
    # Replacing the capture cancels even hidden/pending graphs and returns Render.
    tabs.sync((2,0,Path('/new.pxd2'),4,1),states);await frames()
    assert third.destroyed and third.job.cancelled==1 and tabs.active=='render'
    assert tuple(tabs.pages)==tuple(key for key,_ in FIXED_TABS)
    # OVD has no legacy analysis streams; hide controls without deleting pages.
    tabs.select('threads');await frames()
    ovd_source=(3,0,Path('/capture.ovd'),4,2)
    with patch.object(tabs,'bar',wraps=tabs.bar) as bar:
        tabs.sync(ovd_source,states,analysis_applicable=False);await frames()
        assert bar.rebuild.call_count==1,bar.rebuild.call_count
        assert tuple(tabs.buttons)==('render',) and not tabs.select('profile')
        assert tabs.active=='render' and tuple(tabs.pages)==tuple(key for key,_ in FIXED_TABS)
        tabs.sync(ovd_source,states,analysis_applicable=False);await frames()
        assert bar.rebuild.call_count==1,bar.rebuild.call_count
    graph_view=tabs.add('OVD property graph',graph);await frames()
    assert set(tabs.buttons)=={'render',graph_view.window.key}
    tabs.close(graph_view.window.key);await frames()
    assert tabs.active=='render'
    legacy_source=(4,0,Path('/legacy.pxd2'),4,1)
    with patch.object(tabs,'bar',wraps=tabs.bar) as bar:
        tabs.sync(legacy_source,states);await frames()
        assert bar.rebuild.call_count==1,bar.rebuild.call_count
        assert tuple(tabs.buttons)==tuple(key for key,_ in FIXED_TABS)
        tabs.sync(legacy_source,states);await frames()
        assert bar.rebuild.call_count==1,bar.rebuild.call_count
    tabs.destroy();await frames()
    assert not views and len(disposed)==5
    window.destroy()
    print('Native workspace tab checks passed',flush=True)


if __name__=='__main__':
    errors=[]
    async def run():
        try:await verify()
        except Exception as error:errors.append(error)
        finally:ui.shutdown()
    isolate_gui_output()
    ui.init('Workspace lifecycle regression',width=920,height=720,max_fps=60)
    style.startup('dark')
    ui.run(run())
    if errors:raise errors[0]

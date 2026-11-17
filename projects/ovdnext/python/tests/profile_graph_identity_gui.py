# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native OVUI graph source/lifetime regression; separate from Inspector QA."""
from gui_runtime import isolate_gui_output, hide_owned_window
from pathlib import Path
import math
import tempfile
import time
from types import SimpleNamespace

import omni.ui as ui
import _pvd_native as native
from ovd_ui import style
from ovd_ui.application import Application
from ovd_ui.profiling import DiagnosticsTasks, EventKey
from ovd_ui.widgets.workspace_tabs import WorkspaceTabs

ROOT = Path(__file__).resolve().parents[2]


async def verify():
    with tempfile.TemporaryDirectory() as temporary:
        app = Application.__new__(Application)
        app.capture = ROOT/'test/data/ue4_shooter.pxd2'
        app.session = native.Session(str(app.capture), temporary)
        app.scene_id = app.session.scenes()[-1].id
        original_scene = app.scene_id
        app.presentation = app.session.presentation(app.scene_id, 30)
        app.workflow = SimpleNamespace(settings={}, display_session=app.session,
                                       display_data_path=app.capture)
        app.playback = SimpleNamespace(clip=SimpleNamespace(start=30, stop=35))
        app.analysis_views = []
        app.diagnostics_tasks = DiagnosticsTasks()
        jumps = []
        app._analysis_frame = lambda session, scene, frame: jumps.append((session, scene, frame))
        window = ui.Window('Graph identity regression', width=1280, height=800)
        app.status_label = ui.Label('')
        def activate(page):
            if page.key in ('profile','threads') and page.view is None:
                page.view=app._create_profile(page.host,page.key=='threads')
                page.source=app.workspace.source
                app.analysis_views.append(page.view)
        with window.frame:
            with ui.VStack():
                app.workspace=WorkspaceTabs(window, activate, app._dispose_analysis)
        states={key:SimpleNamespace(enabled=True,reason='') for key in ('profile','memory','errors')}
        app.workspace.sync((1,0,app.capture,original_scene,30),states)

        async def frames(count=5):
            for _ in range(count):
                app.workspace.poll()
                for view in tuple(app.analysis_views):view.poll()
                await ui.next_frame()

        async def loaded(view):
            deadline=time.monotonic()+15
            while not view.finished:
                assert time.monotonic()<deadline, 'Native graph timed out'
                await frames(1)
            await frames()

        try:
            app.workspace.select('profile');await frames()
            profile=app.workspace.pages['profile'].view
            await loaded(profile)
            row=next(row for row in profile.model.rows()
                     if row.name=='Basic.checkResults' and row.context_id==original_scene and row.thread_id==1380)
            event_item=profile.tree_model.leaves[EventKey.of(row)]
            assert event_item.parent.values[0].as_string=='Thread 1380'
            context_name=profile.model.snapshot.context_names[original_scene]
            assert context_name=='PxScene id: 1143'
            assert event_item.values[0].as_string==f'Basic.checkResults [{context_name}]'
            assert all(profile.tree.is_expanded(root) for root in profile.tree_model.roots)
            app.workspace.select('threads');await frames()
            threads=app.workspace.pages['threads'].view;await loaded(threads)
            task_item=threads.tree_model.leaves[EventKey.of(row)]
            assert task_item.values[0].as_string=='Thread 1380'
            assert task_item.parent.values[0].as_string==context_name
            assert all(threads.tree.is_expanded(root) for root in threads.tree_model.roots)
            expected={EventKey.of(item) for item in threads.model.rows()
                      if (item.context_id,item.thread_id)==(original_scene,1380)}
            assert len(expected)>1
            threads._select_tree([None,task_item,task_item.parent]);await frames()
            assert threads.model.selected=={EventKey.of(item) for item in threads.model.rows()
                                           if item.context_id==original_scene}
            threads._select_tree([task_item]);await frames()
            assert threads.model.selected==expected and threads.tree.selection==[task_item]
            assert threads.activate_found(threads.model.snapshot,EventKey.of(row));await frames()
            task_item=threads.tree_model.leaves[EventKey.of(row)]
            assert threads.model.selected=={EventKey.of(row)} and threads.tree.selection==[task_item]
            # Sorting preserves the exact Find key even though its visible
            # context/thread row aggregates several events. Collapsed siblings
            # stay collapsed after the model invalidates its row widgets.
            collapsed=next(root for root in threads.tree_model.roots if root is not task_item.parent)
            threads.tree.set_expanded(collapsed,False,False)
            for column in (2,2,3,0):
                threads.pending_sort=column;await frames()
                assert threads.model.selected=={EventKey.of(row)}
                assert threads.tree.selection==[task_item]
                assert not threads.tree.is_expanded(collapsed)
            table=threads.table
            table.grips[0].pending_width=420.
            await frames()
            assert table.displayed[0]==420.
            assert tuple(cell.computed_width for cell in table.cells)==table.displayed
            assert tuple(float(width) for width in table.tree.column_widths)==table.displayed
            assert table.tree.computed_width==sum(table.displayed)
            assert table.header.computed_width==table.content.computed_width
            assert all(button.computed_height==table.header.computed_height for button in table.buttons), (
                [button.computed_height for button in table.buttons],table.header.computed_height)
            assert all(button.computed_width==width-style.layout.pvd_splitter_hit_width
                       for button,width in zip(table.buttons,table.displayed))
            # A hidden aggregate member remains individually recoverable in
            # both views. Find reopens the group without showing its intervals.
            for view in (threads,profile):
                app.workspace.select('threads' if view is threads else 'profile');await frames()
                assert view.activate_found(view.model.snapshot,EventKey.of(row));await frames()
                view.visibility(False);await frames()
                hidden=view.tree_model.hidden_root
                assert hidden and not view.tree.is_expanded(hidden)
                assert view.tree_model.leaves[EventKey.of(row)].parent is hidden
                assert view.model.selected=={EventKey.of(row)}
                assert view.activate_found(view.model.snapshot,EventKey.of(row));await frames()
                assert view.tree.is_expanded(view.tree_model.hidden_root)
                assert EventKey.of(row) in view.model.hidden
                view.visibility(True);await frames()
                assert view.tree_model.hidden_root is None
                assert view.model.selected=={EventKey.of(row)}
            # Find may leave the timeline scrolled to a later lane. Hiding
            # everything except that event must rebuild the shorter visible
            # canvas immediately, then restore all lane widgets on Show.
            for view in (profile,threads):
                app.workspace.select('profile' if view is profile else 'threads');await frames()
                assert view.activate_found(view.model.snapshot,EventKey.of(row));await frames()
                view.visibility(False,True);await frames()
                assert len(view._visible_lanes())==1
                assert [lane for lane,_ in view.bands]==view._visible_lanes()
                assert view.scroll.scroll_y==0
                hidden=view.tree_model.hidden_root
                view._select_tree([hidden]);await frames()
                assert len(view.model.selected)>1
                view.visibility(True);await frames()
                assert not view.model.hidden and view.tree_model.hidden_root is None
                assert [lane for lane,_ in view.bands]==view._visible_lanes()
                assert len(view.bands)>1
            app.workspace.select('profile');await frames()
            profile.model.select([]);profile.selection_changed();await frames()
            assert profile.model.selected==frozenset()
            app.workspace.select('profile');await frames()
            profile.model.select([EventKey.of(row)])
            profile.plot(False);await frames()
            current=app.analysis_views[-1];await loaded(current)
            app.workspace.select('profile');await frames()
            profile.plot(True);await frames()
            total=app.analysis_views[-1];await loaded(total)
            assert current.title=='Thread 1380 · Basic.checkResults'
            assert total.title=='All threads · Basic.checkResults'
            assert current.series.request.thread_id==1380 and total.series.request.thread_id is None
            assert current.series.frames==total.series.frames==tuple(range(30,35))
            for index, frame in enumerate(current.series.frames):
                assert app.session.seek(original_scene,frame)
                snapshot=native.profile_snapshot(app.session,original_scene)
                rows=[item for item in snapshot.summaries if
                      (item.zone_id,item.event_id,item.context_id)==(row.zone_id,row.event_id,row.context_id)]
                for graph, items in ((current,[item for item in rows if item.thread_id==1380]),(total,rows)):
                    assert math.isclose(graph.series.values[index],sum(item.total_ms for item in items),abs_tol=1e-15)
            assert total.series.values[0]>current.series.values[0]>0
            for graph in (current,total):
                page=app.workspace.pages[graph.window.key]
                assert graph.title_label.text==page.title==graph.title
                assert page.description==graph.source_label
                assert f'Scene {original_scene}' in page.tooltip and 'Frames 30–34' in page.tooltip
                assert 'Context '+str(row.context_id) in page.tooltip
            # A later refresh of the originating view must not retarget old graphs.
            app.workspace.select('profile');await frames()
            app.scene_id=app.session.scenes()[0].id
            assert app.scene_id!=original_scene
            app.presentation=app.session.presentation(app.scene_id,31)
            profile.on_refresh(profile);await loaded(profile)
            for graph in (current,total):
                graph.on_frame(32)
                assert jumps[-1]==(app.session,original_scene,32)
                assert f'Scene {original_scene}' in graph.source_label
                app.workspace.select(graph.window.key);await frames()
                assert graph.window.visible and graph.model.has_values
            style.startup('light');await frames()
            assert total.title_label.text==total.title and total.model.has_values
            app.workspace.close(current.window.key);await frames()
            assert current.destroyed and not total.destroyed
        finally:
            app.workspace.destroy()
            app.diagnostics_tasks.close()
            window.destroy()
        assert not app.analysis_views
    print('Native profile graph identity checks passed',flush=True)


if __name__=='__main__':
    errors=[]
    async def run():
        try:await verify()
        except Exception as error:errors.append(error)
        finally:ui.shutdown()
    isolate_gui_output()
    ui.init('Graph identity regression',width=1280,height=800,max_fps=60)
    hide_owned_window()
    style.startup('dark')
    ui.run(run())
    if errors:raise errors[0]

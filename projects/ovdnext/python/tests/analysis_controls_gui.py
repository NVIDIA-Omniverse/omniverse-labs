# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native OVUI control lifetime checks; distinct from screenshot-first GUI QA."""
from gui_runtime import isolate_gui_output, hide_owned_window
from concurrent.futures import Future
from pathlib import Path
import tempfile
from types import SimpleNamespace

import omni.ui as ui
import _pvd_native as native
from ovd_ui import style
from ovd_ui.profiling import EventKey
from ovd_ui.widgets.diagnostic_tables import ErrorStreamView, MemoryView
from ovd_ui.widgets.profile_view import ProfileView
from ovd_ui.widgets.workspace_tabs import WorkspaceTabs, page_surface

ROOT = Path(__file__).resolve().parents[2]


class Job:
    def __init__(self):
        self.future = Future()
        self.progress = (0, 1)
        self.cancelled = 0

    def cancel(self):
        self.cancelled += 1


async def verify():
    with tempfile.TemporaryDirectory() as temporary:
        cloth = native.Session(str(ROOT/'test/data/cloth_physx34.pxd2'), temporary)
        scene = cloth.scenes()[0].id
        profile, memory = [], []
        for frame in (2, 3):
            assert cloth.seek(scene, frame)
            profile.append(native.profile_snapshot(cloth, scene))
            memory.append(native.memory_snapshot(cloth, scene))
        shooter = native.Session(str(ROOT/'test/data/ue4_shooter.pxd2'), temporary)
        errors = []
        for frame in (920, 921):
            assert shooter.seek(shooter.scenes()[-1].id, frame)
            errors.append(native.error_snapshot(shooter, shooter.scenes()[-1].id))
        del cloth, shooter

    for key, data in (('profile', profile), ('threads', profile), ('memory', memory), ('errors', errors)):
        window = ui.Window('Analysis controls regression', width=1280, height=800)
        views = []
        def dispose(view):
            views.remove(view)
            view.destroy()
        def activate(page):
            if page.key != key or not page.enabled or page.view is not None:
                return
            job = Job()
            if key in ('profile', 'threads'):
                view = ProfileView(job, page_surface, host=page.host, mode=int(key=='threads'))
            else:
                view = (MemoryView if key=='memory' else ErrorStreamView)(job, surface=page_surface, host=page.host)
            page.view, page.source = view, tabs.source
            if page.controls:
                view.restore_workspace_state(page.controls)
            views.append(view)
        with window.frame:
            with ui.VStack():
                tabs = WorkspaceTabs(window, activate, dispose)
        states = {name: SimpleNamespace(enabled=True, reason='') for name in ('profile', 'memory', 'errors')}
        source = (1, 0, Path('/capture.pxd2'), 3, data[0].current_frame)
        async def frames():
            for _ in range(5):
                tabs.poll()
                for view in tuple(views):
                    view.poll()
                await ui.next_frame()
        tabs.sync(source, states)
        tabs.select(key)
        await frames()
        page = tabs.pages[key]
        first = page.view
        first.job.future.set_result(data[0])
        await frames()
        if key in ('profile', 'threads'):
            event = EventKey.of(data[0].summaries[0])
            first.criterion.model.get_item_value_model().set_value(3)
            first.search_field.model.set_value(str(event.event_id))
            first.model.hidden.add(event)
            first.model.select([event])
            first.model.mark(0, data[0].duration_ms/2)
            # Release/sort intent may precede the next layout poll. A frame
            # replacement must carry those controls, not stale measurements.
            first.table.grips[0].pending_width=240.
            first.pending_sort=3
        else:
            first.search.model.set_value('Np' if key=='memory' else 'CCD')
            first._sort('file')
            first._sort('file')
            if key=='memory':
                first._mode(True)
                first._sort('line')
        wanted = first.get_workspace_state()
        tabs.sync((*source[:-1], data[1].current_frame), states)
        await frames()
        second = page.view
        assert second is not first and first.job.cancelled==1
        assert second.get_workspace_state()==wanted
        # The new frame cannot expose the previous rows or export its data.
        if key in ('profile', 'threads'):
            assert second.model.snapshot is None and not second.model.selected
            assert second.model.marked_range is None and not second.find_button.enabled
        else:
            assert second.data is None and not second.table.items and not second.export_button.enabled
        second.job.future.set_result(data[1])
        await frames()
        assert second.get_workspace_state()==wanted
        state = second.get_inspector_state()
        assert state['frame']==data[1].current_frame
        if key in ('profile', 'threads'):
            assert state['search_by']=='event_id' and state['hidden']==1
            assert not state['selected'] and state['marked_ms'] is None
            assert second.criterion.model.get_item_value_model().as_int==3
            assert second.sort_column==3 and second.sort_descending
            assert second.table.displayed[0]==240.
        else:
            assert second.search.model.as_string==wanted['text']
            assert second.export_options()['text']==wanted['text']
            if key=='memory':
                assert second.outstanding and second.sort=='line'
                second._mode(False)
                assert (second.sort, second.reverse)==('file', True)
                second._mode(True)
                assert (second.sort, second.reverse)==('line', True)
        # A frame with no corresponding stream keeps controls but no result.
        availability_key = 'profile' if key=='threads' else key
        states[availability_key] = SimpleNamespace(enabled=False, reason='No recorded data')
        tabs.sync((*source[:-1], 0), states)
        await frames()
        assert page.view is None and page.controls==wanted, (key, page.controls, wanted)
        states[availability_key] = SimpleNamespace(enabled=True, reason='')
        tabs.sync((*source[:-1], data[1].current_frame), states)
        await frames()
        pending = page.view
        assert pending.get_workspace_state()==wanted
        # A late completion of the disposed job cannot publish into its replacement.
        tabs.sync((*source[:-1], data[1].current_frame+1), states)
        await frames()
        pending.job.future.set_result(data[1])
        await frames()
        assert pending.job.cancelled==1 and not page.view.finished
        assert page.view.get_workspace_state()==wanted
        # Each component of capture/segment/path/scene invalidates saved controls,
        # including when the old page currently has no view.
        for replacement in (None, (2,0,source[2],3,1), (2,1,source[2],3,1),
                            (2,1,Path('/replacement.pxd2'),3,1), (2,1,Path('/replacement.pxd2'),4,1)):
            page.controls = dict(wanted)
            states[availability_key] = SimpleNamespace(enabled=False, reason='Unavailable')
            tabs.sync(replacement, states)
            await frames()
            assert page.controls=={} and page.view is None
        tabs.destroy()
        window.destroy()
        await ui.next_frame()
    print('Native analysis control checks passed', flush=True)


if __name__=='__main__':
    errors=[]
    async def run():
        try:
            await verify()
        except Exception as error:
            errors.append(error)
        finally:
            ui.shutdown()
    isolate_gui_output()
    ui.init('Analysis controls regression', width=1280, height=800, max_fps=60)
    hide_owned_window()
    style.startup('dark')
    ui.run(run())
    if errors:
        raise errors[0]

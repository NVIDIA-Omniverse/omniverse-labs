# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stable recording rows over the existing capture lifecycle and file library."""
from datetime import datetime
import math
from pathlib import Path
import struct
import time

import omni.ui as ui

from ..camera import normalize_key
from ..style import layout as fl
from .table_columns import ColumnWidths,ResizableTable
from .workspace_tabs import page_surface


def recorded_time(value):
    """Qt's local yyyy-MM-dd HH:mm:ss.zzz, at filesystem millisecond precision."""
    seconds,millis=divmod(math.floor(value*1000),1000)
    return datetime.fromtimestamp(seconds).strftime('%Y-%m-%d %H:%M:%S')+f'.{millis:03d}'


def recorded_size(value):
    # Match the original float-based binary-unit sizeprettyprint.hpp; the
    # tooltip preserves the exact integer even when the display rounds it.
    number,=struct.unpack('f',struct.pack('f',value))
    unit='bytes'
    for candidate in ('kb','Mb','Gb','Tb'):
        if number<1024:break
        number/=1024
        unit=candidate
    return f'{number:.2f} {unit}'


def recording_key(entry):
    return getattr(entry,'id',entry.path)


def recording_name(entry):
    return entry.path.name if entry.path else f'Memory capture {entry.id}'


class RecordingItem(ui.AbstractItem):
    def __init__(self,entry):
        super().__init__()
        self.entry=entry
        self.values=[ui.SimpleStringModel('') for _ in range(4)]
        self.displayed=self.selected=self.alternate=False

    def update(self,entry,displayed,alternate):
        changed=(self.entry,self.displayed,self.alternate)!=(entry,displayed,alternate)
        self.entry,self.displayed,self.alternate=entry,displayed,alternate
        values=(recorded_time(entry.recorded_at),entry.status,recorded_size(entry.size_bytes),recording_name(entry))
        for model,value in zip(self.values,values):
            if model.as_string!=value:model.set_value(value)
        return changed


class RecordingModel(ui.AbstractItemModel):
    def __init__(self):
        super().__init__();self.items=[];self.cache={}

    def update(self,entries,display_path):
        items=[];changed=False
        for index,entry in enumerate(entries):
            key=recording_key(entry)
            item=self.cache.get(key) or RecordingItem(entry)
            changed=item.update(entry,key==display_path,bool(index%2)) or changed
            items.append(item)
        self.cache={recording_key(item.entry):item for item in items}
        if items!=self.items:
            self.items=items;self._item_changed(None)
            changed=True
        return changed

    def get_item_children(self,item=None):return self.items if item is None else []
    def get_item_value_model_count(self,item=None):return 4
    def get_item_value_model(self,item,column_id=0):return item.values[column_id]
    def can_item_have_children(self,item=None):return item is None


class RecordingDelegate(ui.AbstractItemDelegate):
    def build_branch(self,*_):pass
    def build_widget(self,model,item,column_id,level,expanded):
        if item is None:return
        entry=item.entry
        text=item.values[column_id].as_string
        tooltip=(str(entry.path) if entry.path else 'Retained in memory; Save as to keep an OVD file') if column_id==3 else f'{entry.size_bytes:,} bytes' if column_id==2 else text
        if item.displayed:tooltip='Displayed capture\n'+tooltip
        with ui.ZStack(height=fl.pvd_row):
            ui.Rectangle(name='recording_alternate' if item.alternate else 'recording_row',
                selected=item.selected,opaque_for_mouse_events=False)
            with ui.HStack(spacing=0):
                ui.Spacer(width=4)
                ui.Label(text,height=fl.pvd_row,name='recording_displayed' if item.displayed else '',
                    elided_text=True,explicit_hover=True,tooltip=tooltip)
                ui.Spacer(width=4)


class RecordingsView:
    def __init__(self,workflow,on_open,on_delete,on_folder,host,*,on_save=None,folder_status='',folder_pending=False):
        self.workflow,self.on_open,self.on_delete=workflow,on_open,on_delete
        self.on_save=on_save
        self.window=host;host.minimum_width=700;host.minimum_height=280
        self.model=RecordingModel();self.delegate=RecordingDelegate()
        self.last_refresh=0.;self.snapshot=None;self.selecting=False;self.closed=False;self.table_focus=False
        self.columns=ColumnWidths((190,88,84,338),(170,70,70,180),(420,320,300,1800),stretch=3)
        with page_surface(host,'Recordings'):
            with ui.HStack(height=fl.pvd_control,spacing=8):
                self.keep=ui.CheckBox(width=22)
                self.keep.model.set_value(workflow.keep_recordings)
                self.keep.model.add_value_changed_fn(lambda m:workflow.set_keep_recordings(m.as_bool)
                    if m.as_bool!=workflow.keep_recordings else None)
                ui.Label('Auto-save to disk',width=130,explicit_hover=True,
                    tooltip='Captures stay in memory until removed or the app closes. Also save each useful completed capture to disk; empty or rejected connections stay in memory only. A change applies to connections that have not started streaming.')
                ui.Spacer()
                self.follow=ui.Button('Follow latest',width=108,clicked_fn=self.follow_latest)
                ui.Button('Refresh',width=76,clicked_fn=self.reload)
                self.folder_button=ui.Button('Open folder',name='outline',width=104,enabled=not folder_pending,
                    explicit_hover=True,tooltip=str(workflow.recordings.directory),clicked_fn=on_folder)
                self.save=ui.Button('Save as',width=76,enabled=False,clicked_fn=self.save_selected)
                self.delete=ui.Button('Delete',width=76,enabled=False,explicit_hover=True,clicked_fn=self.delete_selected)
            self.table=ResizableTable(self.model,self.delegate,('Recorded','Status','Size','Capture'),self.columns,None)
            self.table.set_sort(None,False)
            self.tree=self.table.tree;self.tree.root_visible=False
            self.tree.set_selection_changed_fn(self._select)
            self.tree.set_mouse_double_clicked_fn(lambda *_:self.activate())
            self.count=ui.Label('',name='muted',height=20)
            folder_status=folder_status or str(workflow.recordings.directory)
            self.folder_label=ui.Label(folder_status,name='muted',height=24,
                elided_text=True,explicit_hover=True,tooltip=folder_status)
            with ui.CollapsableFrame('Recent files',collapsed=True,height=0) as self.recent_section:
                with ui.ScrollingFrame() as self.recent_scroll:
                    self.recents=ui.Frame(build_fn=self._build_recents)
        self.recent_section.set_collapsed_changed_fn(self._recent_collapsed)
        host.scroll.vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED
        host.set_key_pressed_fn(self._key)
        self.refresh()

    def _recent_collapsed(self,collapsed):
        # Share remaining space between the two scrollable lists. A fixed
        # recent-list height can otherwise consume the entire recording table.
        self.recent_section.height=ui.Pixel(0) if collapsed else ui.Fraction(1)
        self.window.minimum_height=280 if collapsed else 380

    def _build_recents(self):
        with ui.VStack(height=0,spacing=8):
            for path in self.workflow.refresh_recent_files():
                ui.Button(Path(path).name,height=fl.pvd_control,explicit_hover=True,tooltip=path,
                    clicked_fn=lambda p=path:self.on_open(Path(p)))
            ui.Button('Clear recent files',height=fl.pvd_control,
                clicked_fn=lambda:(self.workflow.clear_recent(),self.recents.rebuild()))

    def _select(self,items):
        if self.selecting:return
        selected=[item for item in items if item is not None][-1:]
        if list(items)!=selected:
            self.selecting=True
            try:self.tree.selection=selected
            finally:self.selecting=False
        for item in self.model.items:item.selected=item in selected
        self.tree.dirty_widgets()
        self._update_actions()

    def _entries(self):
        sessions=self.workflow.session_recordings
        live=self.workflow.live_recording
        paths={entry.path for entry in sessions if entry.path is not None}
        if live is not None and live.path is not None:paths.add(live.path)
        entries=(*sessions,*(entry for entry in self.workflow.recordings.entries if entry.path not in paths))
        return (() if live is None else (live,))+tuple(sorted(entries,key=lambda entry:entry.recorded_at,reverse=True))

    def _update_actions(self):
        entry=self.selected_entry()
        memory=entry is not None and hasattr(entry,'id')
        self.delete.enabled=self.can_delete()
        self.delete.text='Remove' if memory else 'Delete'
        self.delete.tooltip=('Remove this capture from memory. Saved files are kept, and files in the recording library remain listed.'
            if memory else '')
        self.save.enabled=bool(memory and entry.status=='Complete' and self.on_save)

    def selected_entry(self):
        selected=[item for item in self.tree.selection if item is not None]
        if not selected:return None
        key=recording_key(selected[-1].entry)
        # Refreshes and live finalization can run between selection and input.
        return next((entry for entry in self._entries() if recording_key(entry)==key),None)

    def can_delete(self):
        entry=self.selected_entry()
        return bool(entry is not None and entry.status!='Live' and
            (hasattr(entry,'id') or entry.path not in (self.workflow.display_path,self.workflow.active_recording_path)))

    def activate(self):
        entry=self.selected_entry()
        if entry is not None:
            if entry.status=='Live':self.follow_latest()
            elif hasattr(entry,'id'):self.workflow.select_session_recording(entry.id)
            else:self.on_open(entry.path)

    def save_selected(self):
        entry=self.selected_entry()
        if entry is not None and hasattr(entry,'id') and entry.status=='Complete' and self.on_save:
            self.on_save(entry.id)

    def delete_selected(self):
        if self.can_delete():
            entry=self.selected_entry()
            if hasattr(entry,'id'):self.workflow.remove_session_recording(entry.id)
            else:self.on_delete(entry.path)

    def _key(self,key,modifiers,pressed):
        if (pressed and self.table_focus and self.window.focused and not (modifiers&15)
                and ui.Menu.get_current() is None and normalize_key(key) in (257,335)):self.activate()

    def focus_at(self,x,y):
        tree=self.tree
        self.table_focus=(tree.screen_position_x<=x<tree.screen_position_x+tree.computed_width
            and tree.screen_position_y<=y<tree.screen_position_y+tree.computed_height)

    def reload(self):
        self.workflow.reload_recordings();self.refresh();self.recents.rebuild()

    def follow_latest(self):
        self.workflow.follow_latest();self.refresh(select_displayed=True)

    def refresh(self,*,select_displayed=False):
        if self.closed:return
        workflow=self.workflow
        entries=self._entries()
        display_key=workflow.displayed_session_recording_id or workflow.display_path
        live=workflow.live_recording
        if live is not None and workflow.display_session is live.session:display_key=live.id
        snapshot=(entries,display_key,workflow.active_recording_path,workflow.follow_live,workflow.keep_recordings)
        if snapshot==self.snapshot and not select_displayed:return
        select_displayed=select_displayed or self.snapshot is None or self.snapshot[1]!=display_key
        self.snapshot=snapshot
        selected=[item for item in self.tree.selection if item is not None]
        changed=self.model.update(entries,display_key)
        retained=([item for item in self.model.items if recording_key(item.entry)==display_key]
            if select_displayed else [item for item in selected if item in self.model.items])
        self.tree.selection=retained
        if changed:self.tree.dirty_widgets()
        self._update_actions()
        self.follow.enabled=not workflow.follow_live
        self.keep.model.set_value(workflow.keep_recordings)
        self.count.text=f'{len(entries):,} recordings · Newest first' if entries else 'Recordings appear here when a producer connects.'

    def poll(self):
        if self.closed or not self.window.visible:return
        # The activity panel may be shorter than its preferred height. Keep
        # rows usable and scroll the page instead of collapsing the table.
        minimum=220 if self.recent_section.collapsed else 320
        height=max(minimum,self.window.scroll.computed_height)
        if float(self.window.content.height)!=height:
            self.window.content.height=ui.Pixel(height)
        self.table.update()
        if time.monotonic()-self.last_refresh>=.25:
            self.last_refresh=time.monotonic();self.refresh()

    def get_inspector_state(self):
        return {'visible':self.window.visible,'widths':self.table.displayed,
            'selected':str(recording_key(self.selected_entry())) if self.selected_entry() else None,
            'delete_enabled':self.delete.enabled,'follow_enabled':self.follow.enabled,'count':len(self.model.items),
            'rows':[{'path':str(i.entry.path) if i.entry.path is not None else None,
                     'id':getattr(i.entry,'id',None),'key':str(recording_key(i.entry)),
                     'recorded_at':i.entry.recorded_at,'size_bytes':i.entry.size_bytes,
                     'status':i.entry.status,'values':[m.as_string for m in i.values],'displayed':i.displayed}
                    for i in self.model.items[:200]]}

    def destroy(self):
        if self.closed:return
        self.closed=True;self.table.destroy();self.window.destroy()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read-only clip rows with source-scoped actions over applied playback state."""
from dataclasses import dataclass

import omni.ui as ui

from ..style import layout as fl
from .table_columns import ColumnWidths, ResizableTable
from .workspace_tabs import page_surface


@dataclass(frozen=True)
class ClipRow:
    index: int
    name: str
    start: int
    stop: int

    @property
    def signature(self):return self.name,self.start,self.stop

    @property
    def values(self):return self.name,str(self.start),str(max(self.start,self.stop-1)),str(self.stop-self.start)


class ClipItem(ui.AbstractItem):
    def __init__(self,row):
        super().__init__()
        self.row=row
        self.values=[ui.SimpleStringModel(value) for value in row.values]
        self.current=False


class ClipModel(ui.AbstractItemModel):
    def __init__(self):
        super().__init__();self.items=[];self.current=0

    def update(self,rows,current,*,reset=False):
        cached={} if reset else {item.row:item for item in self.items}
        items=[cached.get(row) or ClipItem(row) for row in rows]
        changed=items!=self.items or current!=self.current
        self.current=current
        for item in items:item.current=item.row.index==current
        if items!=self.items:
            self.items=items;self._item_changed(None)
        return changed

    def get_item_children(self,item=None):return self.items if item is None else []
    def get_item_value_model_count(self,item=None):return 4
    def get_item_value_model(self,item,column_id=0):return item.values[column_id]
    def can_item_have_children(self,item=None):return item is None


class ClipDelegate(ui.AbstractItemDelegate):
    def __init__(self,on_context):
        super().__init__();self.on_context=on_context

    def build_branch(self,*_):pass

    def build_widget(self,model,item,column_id,level,expanded):
        if item is None:return
        row=item.row;text=row.values[column_id]
        with ui.ZStack(height=fl.pvd_row):
            ui.Rectangle(name='recording_alternate' if row.index%2 else 'recording_row',
                selected=item.current,opaque_for_mouse_events=False)
            with ui.HStack(spacing=0):
                ui.Spacer(width=4)
                ui.Label(text,height=fl.pvd_row,name='recording_displayed' if item.current else '',
                    alignment=ui.Alignment.LEFT_CENTER if column_id==0 else ui.Alignment.RIGHT_CENTER,
                    elided_text=True,explicit_hover=True,tooltip=text,
                    mouse_pressed_fn=lambda x,y,button,modifiers,r=row:
                        self.on_context(r,x,y) if button==1 else None)
                ui.Spacer(width=4)


class ClipsView:
    def __init__(self,on_action,host):
        self.on_action,self.window=on_action,host
        self.closed=self.refreshing=False
        self.source=None;self.snapshot=None;self.export_reason='Open a capture before exporting.'
        self.menu=None;self.menu_generation=0;self.menu_actions=()
        self.model=ClipModel();self.delegate=ClipDelegate(self.show_context_menu)
        self.columns=ColumnWidths((200,100,100,110),(90,86,86,96),(1600,400,400,400))
        host.minimum_height=160
        with page_surface(host,'Clips'):
            self.table=ResizableTable(self.model,self.delegate,
                ('Name','Start Frame','End Frame','Frame Count'),self.columns,None)
            self.table.set_sort(None,False)
            self.tree=self.table.tree;self.tree.root_visible=False
            self.tree.set_selection_changed_fn(self._select)
            self.status=ui.Label('Open a capture to work with clips.',name='muted',height=20)
            with ui.HStack(height=fl.pvd_control,spacing=fl.pvd_spacing):
                self.buttons={action:ui.Button(title,enabled=False,clicked_fn=lambda a=action:self._current_action(a))
                    for action,title in [('split','Split here'),('trim_start','Trim start'),
                                         ('trim_end','Trim end'),('delete','Delete clip')]}

    def refresh(self,playback,source,export_reason):
        if self.closed:return
        rows=tuple(ClipRow(i,c.name,c.start,c.stop) for i,c in enumerate(playback.clips))
        snapshot=(source,rows,playback.index,export_reason)
        if snapshot==self.snapshot:return
        self.close_menu()
        self.snapshot,self.source,self.export_reason=snapshot,source,export_reason
        self.refreshing=True
        try:
            # A replacement capture can have identical rows; never reuse its
            # predecessor's item identities or retained row callbacks.
            reset=source!=getattr(self,'model_source',None)
            self.model_source=source
            changed=self.model.update(rows,playback.index,reset=reset)
            self._select_current()
            if changed:self.tree.dirty_widgets()
        finally:self.refreshing=False
        current=next((r for r in rows if r.index==playback.index),None)
        editable=bool(source is not None and current and current.stop-current.start>1)
        for action in ('split','trim_start','trim_end'):self.buttons[action].enabled=editable
        self.buttons['delete'].enabled=bool(source is not None and len(rows)>1)
        self.status.text=(f'{len(rows):,} clip'+('s' if len(rows)!=1 else '')+
            ' · Right-click a row to save or delete it') if rows else 'Open a capture to work with clips.'

    def _select_current(self):
        selected=[item for item in self.model.items if item.current]
        if self.tree.selection!=selected:
            previous=self.refreshing;self.refreshing=True
            try:self.tree.selection=selected
            finally:self.refreshing=previous

    def _select(self,items):
        if self.closed or self.refreshing:return
        selected=[item for item in items if item in self.model.items]
        if selected:self._action('select',selected[-1].row)
        # Failed/deferred native publication must retain the applied clip.
        self._select_current()

    def _current_action(self,action):
        row=next((item.row for item in self.model.items if item.current),None)
        if row is not None:self._action(action,row)

    def _action(self,action,row):
        if self.closed or self.source is None or not any(i.row is row for i in self.model.items):return
        return self.on_action(self.source,action,row)

    def show_context_menu(self,row,x,y):
        if self.closed or not self.window.visible or not any(i.row is row for i in self.model.items):return
        self.close_menu()
        generation=self.menu_generation
        self.menu=ui.Menu('Clip',tearable=False)
        actions=['save']
        with self.menu:
            ui.MenuItem('Save as',enabled=not self.export_reason,
                explicit_hover=True,tooltip=self.export_reason,
                triggered_fn=lambda:self._menu_action('save',row,generation))
            if len(self.model.items)>1:
                actions.append('delete')
                ui.MenuItem('Delete',triggered_fn=lambda:self._menu_action('delete',row,generation))
        self.menu_actions=tuple(actions)
        self.menu.show_at(x,y)

    def _menu_action(self,action,row,generation):
        if self.closed or generation!=self.menu_generation:return
        if action=='save' and self.export_reason:return
        return self._action(action,row)

    def close_menu(self):
        self.menu_generation+=1
        if self.menu is not None:self.menu.destroy();self.menu=None
        self.menu_actions=()

    def poll(self):
        if self.closed:return
        if not self.window.visible:
            if self.menu is not None:self.close_menu()
            return
        self.table.update()

    def get_inspector_state(self):
        return {'visible':self.window.visible,'source':self.source,'current':self.model.current,
            'widths':self.table.displayed,'export_reason':self.export_reason,'menu_actions':self.menu_actions,
            'rows':[{'index':i.row.index,'values':i.row.values,'current':i.current} for i in self.model.items[:200]],
            'buttons':{key:button.enabled for key,button in self.buttons.items()}}

    def destroy(self):
        if self.closed:return
        self.closed=True;self.close_menu();self.table.destroy();self.window.destroy()

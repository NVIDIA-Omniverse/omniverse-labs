# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native event statistics and scoped task timelines in reusable OVUI views."""
import math

import omni.ui as ui

from ..profiling import EventKey,ProfileModel,thread_name
from ..profile_find import activate_profile_event
from ..camera import normalize_key
from ..palette import PROFILE_FUNCTION_DEFAULT,parse_color,MAGENTA
from ..style import layout as fl
from .hover import HoverOverlay
from .profile_find import FindProfileDialog
from .table_columns import ColumnWidths,ResizableTable


def profile_window_rect(host_size,rect=None):
    """Fit the floating panel to logical host pixels without moving it needlessly.

    At the supported 1280×800 workspace this yields 1248×768. Smaller hosts still
    bound the window, but usable toolbar width is a separate content constraint.
    None/invalid host measurements retain the preferred initial dimensions.
    """
    preferred=(1340.,820.)
    try:
        host_width,host_height=map(float,host_size)
        valid=all(math.isfinite(value) and value>0 for value in (host_width,host_height))
    except (TypeError,ValueError):valid=False
    if not valid:return rect or (0.,0.,*preferred)
    margin=min(2*float(fl.pvd_spacing),host_width/4,host_height/4)
    available=(max(1.,host_width-2*margin),max(1.,host_height-2*margin))
    if rect is None:
        width,height=(min(w,a) for w,a in zip(preferred,available))
        return ((host_width-width)/2,(host_height-height)/2,width,height)
    x,y,width,height=map(float,rect)
    width,height=min(width,available[0]),min(height,available[1])
    return (max(margin,min(x,host_width-width-margin)),
            max(margin,min(y,host_height-height-margin)),width,height)


class ProfileItem(ui.AbstractItem):
    def __init__(self,label,key=None,row=None):
        super().__init__()
        self.key,self.row=key,row
        self.keys=[key] if key is not None else []
        self.tooltip=label
        self.children=[]
        self.parent=None
        self.count=0
        self.total_ms=0.
        self.percent=0.
        self.hidden=False
        self.values=[ui.SimpleStringModel(label),ui.SimpleStringModel(str(row.count) if row else ""),
                     ui.SimpleStringModel(f"{row.total_ms:.4g}" if row else ""),ui.SimpleStringModel("")]


class ProfileTree(ui.AbstractItemModel):
    def __init__(self):
        super().__init__()
        self.roots=[]
        self.leaves={}
        self.hidden_root=None

    def populate(self,rows,zones,mode=0,*,hidden=(),context_names=None):
        """Qt Profile is thread → event/context; Thread Profiler is context → thread.

        A task-thread row represents several exact event keys. Keep that mapping
        separate from the visible hierarchy so selecting a chart interval or Find
        result can still target one event without selecting the entire thread.
        """
        groups={}
        self.roots=[]
        self.leaves={}
        self.hidden_root=None
        hidden=set(hidden)
        context_names=context_names or {}
        for row in sorted(rows,key=lambda row:(row.context_id if mode else row.thread_id,
                row.thread_id if mode else row.name,row.context_id,row.zone_id,row.event_id)):
            key=EventKey.of(row)
            context=context_names.get(row.context_id,f"Context {row.context_id}")
            detail=f"{context} · Context {row.context_id}" if row.context_id in context_names else context
            is_hidden=key in hidden
            if is_hidden:
                if self.hidden_root is None:
                    self.hidden_root=ProfileItem('Hidden events')
                    self.hidden_root.hidden=True
                    self.hidden_root.tooltip='Hidden statistics remain selectable. Show selected reveals their intervals.'
                    self.roots.insert(0,self.hidden_root)
                parent=self.hidden_root
            else:
                root_key=(row.context_id,) if mode else (row.thread_id,)
                if root_key not in groups:
                    groups[root_key]=ProfileItem(context if mode else thread_name(row.thread_id))
                    if mode:groups[root_key].tooltip=detail
                    self.roots.append(groups[root_key])
                parent=groups[root_key]
            if mode and not is_hidden:
                item_key=(row.context_id,row.thread_id)
                if item_key not in groups:
                    item=groups[item_key]=ProfileItem(thread_name(row.thread_id))
                    item.parent=parent
                    item.tooltip=f"{thread_name(row.thread_id)} · {detail}"
                    parent.children.append(item)
                leaf=groups[item_key]
                leaf.keys.append(key)
            else:
                leaf=ProfileItem(f"{row.name} [{context}]",key,row)
                leaf.parent=parent
                leaf.hidden=is_hidden
                leaf.tooltip=(f"{row.name}\n{thread_name(row.thread_id)} · {detail}\n"
                              f"{zones.get(row.zone_id,'')} · Zone {row.zone_id} · Event {row.event_id}")
                parent.children.append(leaf)
            self.leaves[key]=leaf
            for item in (parent,leaf):
                item.count+=row.count
                item.total_ms+=row.total_ms
                item.percent+=row.percent
        for root in self.roots:
            for item in (root,*root.children):
                item.keys=tuple(item.keys)
                item.values[1].set_value(str(item.count))
                item.values[2].set_value(f"{item.total_ms:.4g}")
                item.values[3].set_value(f"{item.percent:.2f}")
        self._item_changed(None)

    def sort(self,column,descending=False):
        def value(item):
            return item.values[0].as_string if column==0 else (item.count,item.total_ms,item.percent)[column-1]
        self.roots.sort(key=value,reverse=descending)
        if self.hidden_root:
            self.roots.remove(self.hidden_root)
            self.roots.insert(0,self.hidden_root)
        for root in self.roots:root.children.sort(key=value,reverse=descending)
        self._item_changed(None)

    def get_item_children(self,item=None):return self.roots if item is None else item.children
    def get_item_value_model_count(self,item=None):return 4
    def get_item_value_model(self,item,column_id=0):return item.values[column_id] if item else None


class ProfileDelegate(ui.AbstractItemDelegate):
    def __init__(self,view):
        super().__init__()
        self.view=view

    def build_branch(self,model,item,column_id,level,expanded):
        if column_id or item is None:return
        with ui.HStack(width=level*12+16,height=24):
            ui.Spacer(width=level*12)
            if item.children:
                with ui.VStack(width=16,spacing=0):
                    ui.Spacer()
                    with ui.HStack(height=8,spacing=0):
                        ui.Spacer()
                        triangle=ui.Triangle(name="object_chevron",width=8,height=8,explicit_hover=True,
                            alignment=ui.Alignment.CENTER_BOTTOM if expanded else ui.Alignment.RIGHT_CENTER)
                        ui.Spacer()
                    ui.Spacer()
                triangle.set_mouse_pressed_fn(lambda x,y,b,m:self.view.tree.set_expanded(item,not expanded,False) if b==0 else None)

    def build_widget(self,model,item,column_id,level,expanded):
        if item is not None:
            text=item.values[column_id].as_string
            ui.Label(text,name='muted' if item.hidden else '',height=24,elided_text=True,explicit_hover=True,
                tooltip=item.tooltip if column_id==0 else text)


class ProfileView:
    def __init__(self,job,surface,*,on_refresh=None,on_plot=None,function_color=PROFILE_FUNCTION_DEFAULT,
                 capture_key=None,source_label="",host_size=None,host=None,mode=0):
        self.job,self.surface,self.on_refresh,self.on_plot=job,surface,on_refresh,on_plot
        self.capture_key,self.source_label=capture_key,source_label
        self.host_size=host_size
        self.host_extent=host_size() if host_size else None
        self.model=ProfileModel()
        self.tree_model=ProfileTree()
        self.finished=False
        self.find_ready=False
        self.find_dialog=None
        self.menu=None
        self.found_key=None
        self.query=""
        self.search_by="name"
        self.lane_page=0
        self.page_size=40
        self.dirty=False
        self.set_function_color(function_color)
        self.drag=None
        self.mode=mode
        self.tree_guard=False
        self.restore_tree=0
        self.restore_expansion=None
        self.sort_column,self.sort_descending=0,False
        self.pending_sort=None
        self.columns=ColumnWidths((146,52,74,76),(100,44,60,60),(1200,300,400,300))
        self.search_editing=False
        self.mouse_x=None
        self.bands=[]
        self.chart_width=0
        x,y,width,height=profile_window_rect(self.host_extent)
        self.window=host or ui.Window("Profiler",width=width,height=height)
        if self.host_extent is not None:self.window.setPosition(x,y)
        self.window.set_key_pressed_fn(self._key)
        with surface(self.window,"Profiler"):
            with ui.HStack(height=fl.pvd_control,spacing=fl.pvd_spacing):
                self.progress=ui.Label("Loading native profile events…",name="muted",elided_text=True,explicit_hover=True)
                self.cancel=ui.Button("Cancel",width=80,clicked_fn=job.cancel)
                self.find_button=ui.Button("Find events",width=110,enabled=False,
                    explicit_hover=True, tooltip="Find profile events in the loaded result (Alt+F)",clicked_fn=self.show_find)
                ui.Button("Refresh",width=80,enabled=on_refresh is not None,clicked_fn=lambda:on_refresh(self))
            with ui.HStack(height=fl.pvd_control,spacing=fl.pvd_spacing):
                search=self.search_field=ui.StringField(explicit_hover=True, tooltip="Filter the timeline by recorded event or native identifier")
                search.model.add_begin_edit_fn(lambda _:setattr(self,"search_editing",True))
                search.model.add_end_edit_fn(lambda _:setattr(self,"search_editing",False))
                search.model.add_value_changed_fn(self._search)
                criterion=self.criterion=ui.ComboBox(0,"Event name","Context ID","Thread ID","Event ID","Zone ID",width=140)
                criterion.model.add_item_changed_fn(self._criterion)
            with ui.HStack(height=fl.pvd_control,spacing=fl.pvd_spacing):
                if host is None:
                    mode_control=ui.ComboBox(self.mode,"Profile events","Thread tasks",width=190)
                    mode_control.model.add_item_changed_fn(self._mode)
                ui.Button("Fit",width=60,clicked_fn=lambda:self.action(self.model.reset_zoom))
                ui.Button("Zoom marked",width=110,clicked_fn=lambda:self.action(self.model.zoom_to_mark))
                ui.Button("Select results",width=110,clicked_fn=self.select_results)
                actions=ui.Button("Actions",width=85)
                actions.set_clicked_fn(lambda:self.context_menu(actions.screen_position_x,
                    actions.screen_position_y+actions.computed_height))
                ui.Spacer()
            with ui.HStack(spacing=fl.pvd_spacing):
                with ui.VStack(width=ui.Fraction(5) if host else 430,spacing=0):
                    self.delegate=ProfileDelegate(self)
                    self.table=ResizableTable(self.tree_model,self.delegate,
                        ("Event / context" if not self.mode else "Context / thread","Count","Total ms","% frame"),
                        self.columns,lambda column:setattr(self,'pending_sort',column))
                    self.table.set_sort(self.sort_column,self.sort_descending)
                    self.tree=self.table.tree
                    self.tree.set_selection_changed_fn(self._select_tree)
                    self.tree.set_mouse_released_fn(lambda x,y,b,m:self.context_menu(x,y) if b==1 else None)
                with ui.VStack(width=ui.Fraction(3) if host else ui.Fraction(1),spacing=4):
                    self.range_label=ui.Label("Recorded frame interval",name="muted",height=24,elided_text=True,explicit_hover=True)
                    with ui.ScrollingFrame(horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF) as scroll:
                        self.scroll=scroll
                        with ui.ZStack(height=80) as canvas:
                            self.canvas=canvas
                            self.chart=ui.Frame(build_fn=self._build_chart)
                            self.hover=HoverOverlay(self.hovered)
                            self.input=ui.InvisibleButton(explicit_hover=True)
                            self.input.set_mouse_pressed_fn(self.press)
                            self.input.set_mouse_moved_fn(self.move)
                            self.input.set_mouse_released_fn(self.release)
                            self.input.set_mouse_wheel_fn(self.wheel)
                            self.input.set_mouse_double_clicked_fn(lambda x,y,b,m:self.double(self.chart,x,m))
                    with ui.HStack(height=fl.pvd_control,spacing=fl.pvd_spacing):
                        ui.Button("Previous",width=70,clicked_fn=lambda:self.page(-1))
                        self.page_label=ui.Label("",name="muted",elided_text=True,explicit_hover=True)
                        ui.Button("Next",width=55,clicked_fn=lambda:self.page(1))
            self.selection_label=ui.Label("Select an event to see its count, duration and average.",name="muted",height=38,word_wrap=True)
            ui.Label("Wheel: zoom · Drag: pan · Shift+drag: mark range · Ctrl+double-click: marker · Home: fit · Enter: zoom marked",name="muted",height=36,word_wrap=True)

    def get_workspace_state(self):
        sort=(self.sort_column,self.sort_descending)
        if self.pending_sort is not None:
            sort=(self.pending_sort,not self.sort_descending if self.pending_sort==self.sort_column else self.pending_sort!=0)
        return {"query":self.query, "search_by":self.search_by,
                "hidden":frozenset(self.model.hidden),"sort":sort,
                "columns":self.table.get_state()}

    def restore_workspace_state(self,state):
        self.model.hidden=set(state["hidden"])
        self.sort_column,self.sort_descending=state['sort']
        self.columns.restore(state['columns'])
        self.table.set_sort(self.sort_column,self.sort_descending)
        criterion=("name","context_id","thread_id","event_id","zone_id").index(state["search_by"])
        self.criterion.model.get_item_value_model().set_value(criterion)
        self.search_field.model.set_value(state["query"])

    def set_job(self,job):
        if self.find_dialog:self.find_dialog.invalidate()
        self.found_key=None
        self.find_button.enabled=False
        self.find_ready=False
        self.job.cancel()
        self.job=job
        self.finished=False
        self.cancel.enabled=True
        self.cancel.set_clicked_fn(job.cancel)

    def _sync_host_size(self):
        if getattr(self,"host_size",None) is None:return
        extent=self.host_size()
        if extent==self.host_extent:return
        self.host_extent=extent
        window=self.window
        current=(window.position_x,window.position_y,window.width,window.height)
        x,y,width,height=profile_window_rect(extent,current)
        if (width,height)!=current[2:]:window.width,window.height=width,height
        if (x,y)!=current[:2]:window.setPosition(x,y)

    @property
    def can_find(self):
        return self.window.visible and self.find_ready and self.model.supported

    def show_find(self):
        if not self.can_find:return False
        if self.find_dialog is None:self.find_dialog=FindProfileDialog(self)
        return self.find_dialog.show()

    def activate_found(self,snapshot,key):
        if not self.can_find or not activate_profile_event(self.model,snapshot,key):return False
        # The dialog states this explicit reveal action before activation.
        self.search_field.model.set_value("")
        self.query=""
        self.found_key=key
        self.rebuild_table()
        self._reveal_found_lane()
        self.dirty=True
        return True

    def _visible_lanes(self):
        lanes=list(self.model.lanes)
        lanes.sort(key=lambda lane:(lane.context_id,lane.thread_id,lane.zone_id,lane.depth) if self.mode
            else (lane.thread_id,lane.context_id,lane.zone_id,lane.depth))
        keys={EventKey.of(row) for row in self.rows()}
        return [lane for lane in lanes if any(EventKey.of(scope) in keys for scope in self.model.scopes(lane))]

    def _reveal_found_lane(self):
        if self.found_key is None:return
        for index,lane in enumerate(self._visible_lanes()):
            if any(EventKey.of(scope)==self.found_key for scope in self.model.scopes(lane)):
                self.lane_page=index//self.page_size
                self.scroll.scroll_y=(index%self.page_size)*52
                break

    def set_function_color(self,value):
        try:rgba=parse_color(value)
        except ValueError:rgba=MAGENTA
        self.function_color=ui.color(*rgba)
        # Keep recorded event names readable when users choose bright or dark colors.
        light=sum(weight*component for weight,component in zip((.2126,.7152,.0722),rgba))>.45
        self.function_text=ui.color("#111827" if light else "#FFFFFF")
        self.dirty=True

    def get_inspector_state(self):
        rows=[row for row in self.model.rows(include_hidden=True) if EventKey.of(row) in self.model.selected]
        return {"kind":"profile","visible":self.window.visible,"range_ms":self.model.view_range,
                "marked_ms":self.model.marked_range,"marker_ms":self.model.marker_ms,
                "selected":[{"name":row.name,"count":row.count,"total_ms":row.total_ms,
                             "zone":row.zone_id,"event":row.event_id,"context":row.context_id,"thread":row.thread_id} for row in rows],
                "hidden":len(self.model.hidden),"lanes":len(self.model.lanes),"mode":self.mode,
                "tree_grouping":"context/thread" if self.mode else "thread/event",
                "sort_column":self.sort_column,"sort_descending":self.sort_descending,
                "column_widths":self.table.displayed,
                "context_names":getattr(self.model.snapshot,'context_names',{}),
                "hidden_rows":len(self.tree_model.hidden_root.children) if self.tree_model.hidden_root else 0,
                "drawn_lanes":[{"zone":lane.zone_id,"context":lane.context_id,"thread":lane.thread_id,"depth":lane.depth} for lane,_ in self.bands],
                "chart_scroll_y":self.scroll.scroll_y,
                "hidden_group_expanded":self.tree.is_expanded(self.tree_model.hidden_root) if self.tree_model.hidden_root else False,
                "inline_filter":self.query,"search_by":self.search_by,
                "frame":self.model.snapshot.current_frame if self.model.snapshot else None,
                "find_visible":bool(self.find_dialog and self.find_dialog.window.visible),
                "find_results":len(self.find_dialog.model.items) if self.find_dialog else 0}

    def poll(self):
        self._sync_host_size()
        self.table.update()
        if self.pending_sort is not None:
            column,self.pending_sort=self.pending_sort,None
            self.sort_descending=not self.sort_descending if column==self.sort_column else column!=0
            self.sort_column=column
            self.restore_expansion=[(root,self.tree.is_expanded(root)) for root in self.tree_model.roots]
            self.tree_guard=True
            try:self.tree_model.sort(column,self.sort_descending)
            finally:self.tree_guard=False
            self.table.set_sort(column,self.sort_descending)
            self.restore_tree=2
        if getattr(self.progress,"tooltip",None)!=self.progress.text:self.progress.tooltip=self.progress.text
        if not self.finished and self.job.future.done():
            self.finished=True
            self.cancel.enabled=False
            try:
                if self.find_dialog:self.find_dialog.invalidate()
                self.found_key=None
                self.model.load(self.job.future.result())
                self.find_ready=True
                snapshot=self.model.snapshot
                self.progress.text=(f"Frame {snapshot.current_frame} · {len(snapshot.scopes):,} measured intervals · {snapshot.duration_ms:.5g} ms"
                    if snapshot.supported else snapshot.reason)
                if snapshot.diagnostics:self.progress.text+=" · "+"; ".join(snapshot.diagnostics[:2])
                self.rebuild_table()
                self.dirty=True
            except Exception as error:self.progress.text=f"Profile unavailable: {error}"
            self.find_button.enabled=self.can_find
        if not self.window.visible and self.find_dialog and self.find_dialog.window.visible:
            self.find_dialog.invalidate()
        if self.restore_tree:
            self.restore_tree-=1
            if not self.restore_tree:
                for root in self.tree_model.roots:self.tree.set_expanded(root,True,True)
                self.selection_changed()
                if self.restore_expansion is not None:
                    for root,expanded in self.restore_expansion:self.tree.set_expanded(root,expanded,False)
                    self.restore_expansion=None
        if self.dirty:
            self.dirty=False
            # A filtered/hidden result can be shorter than the old scroll offset.
            # OVUI lazily rebuilds only visible Frames, so clamp the offset and
            # size the Frame itself before asking it to populate the new lanes.
            height=max(80,min(self.page_size,len(self._visible_lanes()))*52+30)
            self.canvas.height=self.chart.height=ui.Pixel(height)
            self.scroll.scroll_y=min(self.scroll.scroll_y,max(0,height-self.scroll.computed_height))
            self.chart.rebuild()
        width=max(1,int(self.scroll.computed_width)-fl.pvd_scrollbar)
        if width!=self.chart_width:
            self.chart_width=width
            self.canvas.width=ui.Pixel(width)
            self.dirty=True

    def rows(self):
        # Visibility affects intervals, not the availability of statistics.
        # The tree puts matching hidden rows in its separate recovery group.
        return self.model.search(self.query,self.search_by) if self.query else self.model.rows(include_hidden=True)

    def rebuild_table(self):
        zones={zone.id:zone.name for zone in self.model.snapshot.zones} if self.model.snapshot else {}
        self.tree_guard=True
        try:
            self.tree_model.populate(self.rows(),zones,self.mode,hidden=self.model.hidden,
                context_names=getattr(self.model.snapshot,'context_names',{}))
            self.tree_model.sort(self.sort_column,self.sort_descending)
            for root in self.tree_model.roots:self.tree.set_expanded(root,True,True)
        finally:self.tree_guard=False
        self.selection_changed()
        self.restore_expansion=None
        self.restore_tree=2

    def _search(self,model):
        self.found_key=None
        self.query=model.as_string
        self.lane_page=0
        self.rebuild_table();self.dirty=True

    def _criterion(self,model,_):
        self.found_key=None
        self.search_by=("name","context_id","thread_id","event_id","zone_id")[model.get_item_value_model().as_int]
        self.rebuild_table();self.dirty=True

    def _mode(self,model,_):
        self.mode=model.get_item_value_model().as_int
        self.table.headers=("Context / thread" if self.mode else "Event / context",*self.table.headers[1:])
        self.table.set_sort(self.sort_column,self.sort_descending)
        self.rebuild_table()
        self.lane_page=0;self._reveal_found_lane();self.dirty=True

    def _select_tree(self,items):
        if self.tree_guard or self.restore_tree:return
        stack=[item for item in items if item is not None];keys=[]
        while stack:
            item=stack.pop()
            keys.extend(item.keys)
            stack.extend(item.children)
        self.model.select(keys);self.selection_changed()

    def select_results(self):
        self.model.select([EventKey.of(row) for row in self.rows()]);self.selection_changed()

    def selection_changed(self):
        rows=[row for row in self.model.rows(include_hidden=True) if EventKey.of(row) in self.model.selected]
        count=sum(row.count for row in rows);total=sum(row.total_ms for row in rows)
        self.selection_label.text=f"{len(rows):,} event keys · {count:,} calls · {total:.6g} ms total · {total/count if count else 0:.6g} ms average · {sum(row.percent for row in rows):.4g}% of interval"
        if self.model.selected & self.model.hidden:
            self.selection_label.text+=" · Hidden events: Show selected reveals their intervals."
        self.tree_guard=True
        try:
            selected=list(dict.fromkeys(self.tree_model.leaves[key] for key in self.model.selected if key in self.tree_model.leaves))
            for item in selected:
                parent=item.parent
                ancestors=[]
                while parent:
                    ancestors.append(parent)
                    parent=parent.parent
                for parent in reversed(ancestors):self.tree.set_expanded(parent,True,False)
            self.tree.selection=selected
        finally:self.tree_guard=False
        self.dirty=True

    def visibility(self,visible,inverse=False):
        self.model.set_selected_visibility(visible,inverse=inverse)
        self.rebuild_table();self.dirty=True
        if not visible and self.tree_model.hidden_root:
            self.tree.set_expanded(self.tree_model.hidden_root,False,False)
            self.restore_expansion=[(self.tree_model.hidden_root,False)]

    def show_all(self):
        self.model.hidden.clear();self.rebuild_table();self.dirty=True

    def action(self,fn,*args):fn(*args);self.dirty=True

    def page(self,step):
        self.lane_page=max(0,min(max(0,(len(self.model.lanes)-1)//self.page_size),self.lane_page+step));self.dirty=True

    def _build_chart(self):
        self.bands=[]
        with ui.VStack(height=0,spacing=4):
            if not self.model.view_range:
                ui.Label(self.model.reason or "No recorded profile interval",name="muted",height=60)
                return
            low,high=self.model.view_range
            self.range_label.text=f"{low:.6g} – {high:.6g} ms · {high-low:.6g} ms shown"
            if self.model.marked_range:
                first,last=self.model.marked_range
                self.range_label.text+=f" · Marked {last-first:.6g} ms ({first:.6g} – {last:.6g})"
            if self.model.marker_ms is not None:self.range_label.text+=f" · Marker {self.model.marker_ms:.6g} ms"
            self.range_label.tooltip=self.range_label.text
            lanes=self._visible_lanes()
            keys={EventKey.of(row) for row in self.rows()}
            self.lane_page=min(self.lane_page,max(0,(len(lanes)-1)//self.page_size))
            start=self.lane_page*self.page_size
            self.page_label.text=f"{start+1 if lanes else 0}–{min(start+self.page_size,len(lanes))} / {len(lanes)}"
            self.page_label.tooltip=f"Lanes {self.page_label.text}"
            budget=4000
            for lane in lanes[start:start+self.page_size]:
                context=getattr(self.model.snapshot,'context_names',{}).get(lane.context_id,f"Context {lane.context_id}")
                label=f"{context} · {thread_name(lane.thread_id)} · depth {lane.depth}"
                ui.Label(label,name="muted",height=20,elided_text=True,explicit_hover=True,
                    tooltip=f"{label}\nContext {lane.context_id}")
                with ui.ZStack(height=24,width=self.chart_width) as band:
                    self.bands.append((lane,band))
                    ui.Rectangle(name="profile_lane")
                    for scope in self.model.scopes(lane):
                        key=EventKey.of(scope)
                        if key not in keys or scope.end_ms<low or scope.start_ms>high:continue
                        if budget<=0:break
                        budget-=1
                        left=max(0.,(scope.start_ms-low)/(high-low)*100)
                        width=max(.08,(min(high,scope.end_ms)-max(low,scope.start_ms))/(high-low)*100)
                        with ui.Placer(offset_x=self.chart_width*left/100):
                            with ui.ZStack(width=self.chart_width*width/100,height=20):
                                selected=key in self.model.selected
                                ui.Rectangle(name="profile_selected" if selected else "profile_scope",
                                    style={} if selected else {"background_color":self.function_color})
                                if width>8:ui.Label(scope.name,name="profile_text",elided_text=True,
                                    style={} if selected else {"color":self.function_text})
                    if self.model.marked_range:
                        first,last=self.model.marked_range
                        with ui.Placer(offset_x=self.chart_width*max(0,(first-low)/(high-low))):
                            ui.Rectangle(width=self.chart_width*max(0,(min(high,last)-max(low,first))/(high-low)),height=24,
                                         name="profile_range",enabled=False,opaque_for_mouse_events=False)
                    if self.model.marker_ms is not None and low<=self.model.marker_ms<=high:
                        with ui.Placer(offset_x=self.chart_width*(self.model.marker_ms-low)/(high-low)):
                            ui.Rectangle(width=1,height=24,name="profile_marker",enabled=False,opaque_for_mouse_events=False)
            if budget<=0:ui.Label("4,000 visible intervals shown; zoom or filter to inspect more.",name="warning",height=26)

    def time_at(self,band,x):
        low,high=self.model.view_range
        return low+(high-low)*max(0,min(1,(x-band.screen_position_x)/max(band.computed_width,1)))

    def hit(self,x,y):
        if not self.model.view_range:return None
        for lane,band in self.bands:
            if band.screen_position_y<=y<band.screen_position_y+band.computed_height:
                return self.model.hit_test(lane,self.time_at(self.chart,x))
        return None

    def press(self,x,y,button,modifiers):
        if button==0 and self.model.view_range:self.drag=(x,self.time_at(self.chart,x),self.model.view_range,modifiers)

    def move(self,x,y,*_):
        self.mouse_x=x
        scope=self.hit(x,y)
        self.input.tooltip=(f"{scope.name}\n{scope.start_ms:.8g}–{scope.end_ms:.8g} ms\nDuration {scope.duration_ms:.8g} ms" if scope else "")
        if self.drag:
            origin,start,view,mods=self.drag
            if mods & 1:self.model.mark(start,self.time_at(self.chart,x))
            else:
                self.model.view_range=view
                self.model.pan(-(x-origin)/max(self.chart.computed_width,1)*(view[1]-view[0]))
            self.dirty=True

    def hovered(self,x,y):
        self.mouse_x=self.input.screen_position_x+x*self.input.computed_width
        scope=self.hit(self.mouse_x,self.input.screen_position_y+y*self.input.computed_height)
        self.input.tooltip=(f"{scope.name}\n{scope.start_ms:.8g}–{scope.end_ms:.8g} ms\nDuration {scope.duration_ms:.8g} ms" if scope else "")

    def release(self,x,y,button,modifiers):
        if button==1:
            scope=self.hit(x,y)
            if scope and EventKey.of(scope) not in self.model.selected:self.pick(EventKey.of(scope),0,0)
            self.context_menu(x,y)
        if self.drag and abs(x-self.drag[0])<3 and not self.drag[3] & 1:
            scope=self.hit(x,y)
            if scope:self.pick(EventKey.of(scope),button,modifiers)
        self.drag=None

    def context_menu(self,x,y):
        if self.menu:self.menu.destroy()
        self.menu=ui.Menu("Profile events",tearable=False)
        with self.menu:
            ui.MenuItem("Plot current thread",enabled=len(self.model.selected)==1,triggered_fn=lambda:self.plot(False))
            ui.MenuItem("Plot all threads",enabled=len(self.model.selected)==1,triggered_fn=lambda:self.plot(True))
            ui.MenuItem("Hide selected events",triggered_fn=lambda:self.visibility(False))
            ui.MenuItem("Show selected events",triggered_fn=lambda:self.visibility(True))
            ui.MenuItem("Hide other events",triggered_fn=lambda:self.visibility(False,True))
            ui.MenuItem("Show other events",triggered_fn=lambda:self.visibility(True,True))
            ui.MenuItem("Show all events",triggered_fn=self.show_all)
        self.menu.show_at(x,y)

    def wheel(self,x,y,modifiers):
        if not self.model.view_range:return
        if modifiers & 2:self.scroll.scroll_y=max(0,min(self.scroll.scroll_y_max,self.scroll.scroll_y-y*40))
        elif modifiers & 1:self.action(self.model.pan,-y*(self.model.view_range[1]-self.model.view_range[0])*.1)
        else:
            anchor=self.time_at(self.chart,self.mouse_x) if self.mouse_x is not None else None
            self.action(self.model.zoom,math.exp(-y*.1),anchor)

    def double(self,band,x,modifiers):
        if not self.model.view_range:return
        if modifiers & 2:self.action(self.model.set_marker,self.time_at(band,x))
        elif self.model.marked_range:self.action(self.model.zoom_to_mark)

    def pick(self,key,button,modifiers):
        if button==0 and not modifiers & 1:
            self.model.select([key],toggle=bool(modifiers & 2));self.selection_changed()

    def _key(self,key,mods,pressed):
        if not pressed or not self.window.focused:return
        key=normalize_key(key)
        if key==70 and mods&15==4:
            self.show_find()
            return
        if self.search_editing:return
        if key==268:self.action(self.model.reset_zoom)
        elif key in (257,335):self.action(self.model.zoom_to_mark)
        elif key==256:
            self.action(self.model.cancel_mark)
            if not self.mode:self.model.select([]);self.selection_changed()

    def plot(self,all_threads):
        keys=list(self.model.selected)
        if len(keys)!=1:
            self.selection_label.text="Select one event/context/thread row to plot its history."
            return
        if self.on_plot:self.on_plot(keys[0],all_threads)

    def destroy(self):
        self.job.cancel()
        self.table.destroy()
        if self.menu:self.menu.destroy()
        if self.find_dialog:self.find_dialog.destroy()
        self.hover.destroy()
        self.window.destroy()

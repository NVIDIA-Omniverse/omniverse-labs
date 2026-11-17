# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OVUI views for owned native messages and asynchronous scalar measurements."""
import math
import time
from pathlib import Path

import omni.ui as ui
from omni.ui_scene import scene as sc

from ..analysis import CaptureMessages, MessageSeverity, ScalarSeries, export_series_csv,export_frame_timing_csv,export_messages_csv
from ..profiling import ProfileSeries,export_profile_csv
from ..style import layout as fl
from omni.ui import color as cl
from ..plotting import Curve, PlotModel
from ..graph_keyboard import KeyboardCrosshair, accepted_rectangle
from ..camera import normalize_key
from .diagnostic_tables import _surface
from .hover import HoverOverlay
from .file_browser import CaptureFileDialog
from .table_columns import ColumnWidths,ResizableTable


class MessageItem(ui.AbstractItem):
    def __init__(self, record):
        super().__init__()
        self.record = record
        self.values = [ui.SimpleStringModel(text) for text in
                       (str(record.frame), record.type_name or "(unknown)", record.text, record.location)]
        self.selected=False


class MessageModel(ui.AbstractItemModel):
    def __init__(self):
        super().__init__()
        self.items=[]
        self.cache={}

    def set_records(self, records):
        items=[]
        for record in records:
            item=self.cache.get(record.index)
            items.append(item if item is not None and item.record==record else MessageItem(record))
        self.cache={item.record.index:item for item in items}
        if items!=self.items:
            self.items=items
            self._item_changed(None)

    def get_item_children(self, item=None):
        return self.items if item is None else []

    def get_item_value_model_count(self, item=None):
        return 4

    def get_item_value_model(self, item, column_id=0):
        return item.values[column_id] if item else None


class MessageDelegate(ui.AbstractItemDelegate):
    def build_branch(self,*_):
        pass

    def build_widget(self,model,item,column_id,level,expanded):
        if item is not None:
            text=item.values[column_id].as_string
            role="message_"+item.record.severity.name.lower()
            with ui.ZStack(height=fl.pvd_row):
                ui.Rectangle(name=role,selected=item.selected,opaque_for_mouse_events=False)
                with ui.HStack(spacing=0):
                    ui.Spacer(width=4)
                    ui.Label(text,height=fl.pvd_row,name=role if column_id==1 else "mono" if column_id==0 else "",
                        elided_text=True,explicit_hover=True,tooltip=text)
                    ui.Spacer(width=4)


class MessagesView:
    def __init__(self, session, on_frame, surface, *, host=None):
        self.adapter=CaptureMessages(session)
        self.on_frame=on_frame
        self.model=MessageModel()
        self.delegate=MessageDelegate()
        self.severity=None
        self.text=""
        self.sort_column=0
        self.descending=False
        self.pending_sort=None
        self.columns=ColumnWidths((64,160,280,200),(48,100,160,120),(240,520,1600,1200),stretch=2)
        self.export_dialog=None
        self.window=host or ui.Window("Capture messages",width=1120,height=600)
        self.last_refresh=0.
        if host is not None:
            host.minimum_width=700
            host.minimum_height=240
        with surface(self.window,"Capture messages"):
            with ui.HStack(height=fl.pvd_control,spacing=fl.pvd_spacing):
                combo=ui.ComboBox(0,"All severities","Errors","Warnings","Information",width=170)
                combo.model.add_item_changed_fn(self._severity)
                self.search=search=ui.StringField(explicit_hover=True, tooltip="Search message text, type or source location")
                search.model.add_value_changed_fn(self._search)
                ui.Button("Refresh",width=90,clicked_fn=self.refresh)
                ui.Button("Export CSV",width=120,clicked_fn=self.export)
            self.table=ResizableTable(self.model,self.delegate,("Frame","Type","Message","Location"),self.columns,
                lambda column:setattr(self,"pending_sort",column),numeric_columns={0})
            self.table.set_sort(self.sort_column,self.descending)
            self.tree=self.table.tree
            self.tree.root_visible=False
            self.tree.set_selection_changed_fn(self._select)
            self.tree.set_mouse_double_clicked_fn(self._jump)
            self.count=ui.Label("",name="muted",height=24,elided_text=True,explicit_hover=True,
                tooltip="Double-click a message to inspect its recorded frame. CSV exports the filtered and sorted rows.")
        self.refresh()

    def poll(self):
        if self.pending_sort is not None:
            column,self.pending_sort=self.pending_sort,None
            self.descending=not self.descending if column==self.sort_column else False
            self.sort_column=column
            self.table.set_sort(column,self.descending)
            self._filter()
        if self.window.visible:self.table.update()
        if self.window.visible and time.monotonic()-self.last_refresh>=.25:
            self.last_refresh=time.monotonic()
            if self.adapter.refresh():self._filter()

    def get_inspector_state(self):
        return {"kind":"messages","visible":self.window.visible,"filter":self.text,
                "severity":int(self.severity) if self.severity is not None else None,
                "total":self.adapter.total,"shown":len(self.model.items),
                "sort":self.sort_key,"descending":self.descending,"widths":self.table.displayed,
                "counts":{severity.name.lower():count for severity,count in self.adapter.counts.items()},
                "rows":[{"index":item.record.index,"frame":item.record.frame,"type":item.record.type_name,
                         "text":item.record.text,"location":item.record.location,"severity":item.record.severity.name}
                        for item in self.model.items[:200]],
                "selected":[{"index":item.record.index,"frame":item.record.frame,
                             "text":item.record.text} for item in self.tree.selection if item is not None]}

    @property
    def sort_key(self):
        return ("frame","type_name","text","location")[self.sort_column]

    def _select(self,items):
        selected={item.record.index for item in items if item is not None}
        for item in self.model.items:item.selected=item.record.index in selected
        self.tree.dirty_widgets()

    def _severity(self, model, _):
        self.severity=(None,MessageSeverity.ERROR,MessageSeverity.WARNING,MessageSeverity.INFO)[model.get_item_value_model().as_int]
        self._filter()

    def _search(self, model):
        self.text=model.as_string
        self._filter()

    def _filter(self):
        selected=[item for item in self.tree.selection if item is not None]
        self.model.set_records(self.adapter.filtered(self.severity,self.text,sort=self.sort_key,reverse=self.descending))
        retained=set(self.model.items)
        self.tree.selection=[item for item in selected if item in retained]
        counts=self.adapter.counts
        self.count.text=f"{len(self.model.items):,} shown / {self.adapter.total:,} total   ·   {counts[MessageSeverity.ERROR]} errors   ·   {counts[MessageSeverity.WARNING]} warnings   ·   {counts[MessageSeverity.INFO]} info"
        if self.adapter.truncated:
            self.count.text+=" · Message display limit reached"

    def refresh(self):
        self.last_refresh=time.monotonic()
        if self.adapter.refresh() or not self.count.text:self._filter()

    def _jump(self,*_):
        selected=[item for item in self.tree.selection if item is not None]
        if selected:
            record=selected[-1].record
            if record.frame:
                self.on_frame(record.frame)

    def export(self):
        if self.export_dialog:self.export_dialog.destroy()
        records=self.adapter.filtered(self.severity,self.text,sort=self.sort_key,reverse=self.descending)
        self.export_dialog=CaptureFileDialog(lambda p:export_messages_csv(records,p,overwrite=p.exists()),
                mode="save",initial_path=Path.home()/"messages.csv",
                extensions=(".csv",),default_extension=".csv",title="Export messages")
        self.export_dialog.show()

    def destroy(self):
        if self.export_dialog:
            self.export_dialog.destroy()
        self.table.destroy()
        self.window.destroy()


class PlotView:
    """Owned curves with Qwt zoom history and a portable keyboard crosshair."""
    def __init__(self, job, title, on_frame, surface=None, *, host=None, source_label=""):
        self.job,self.on_frame,self.title=job,on_frame,title
        self.source_label=source_label
        self.series=self.model=None
        self.finished=self.destroyed=False
        self.start=self.stop=0
        self.export_dialog=self.context_menu=None
        self.drag=None
        self.keyboard=KeyboardCrosshair()
        self.keyboard_navigation_flags=None
        self.editing_fields=set()
        self.chart_scene=None
        self.chart_signature=None
        self.window=host or ui.Window(title,width=1100,height=680)
        self.window.set_key_pressed_fn(self._key)
        self.window.set_focused_changed_fn(lambda focused:None if focused else self._cancel_keyboard())
        with (surface or _surface)(self.window,title):
            self.title_label=ui.Label(title,height=24,elided_text=True,explicit_hover=True,tooltip=title)
            if source_label:
                with ui.VStack(height=40,spacing=0):
                    for line in source_label.splitlines():
                        ui.Label(line,height=20,name="muted",elided_text=True,explicit_hover=True,tooltip=source_label)
            with ui.HStack(height=fl.pvd_control,spacing=fl.pvd_spacing):
                self.progress=ui.Label("Loading capture measurements…",name="muted",elided_text=True)
                self.cancel_button=ui.Button("Cancel",width=85,clicked_fn=job.cancel)
                self.export_button=ui.Button("Export CSV",width=115,enabled=False,clicked_fn=self.export)
            self.legend=ui.Frame(height=24,build_fn=self._build_legend)
            with ui.HStack(spacing=fl.pvd_spacing):
                with ui.VStack(width=78):
                    self.y_top=ui.Label("",height=24,name="muted")
                    ui.Spacer()
                    self.y_bottom=ui.Label("",height=24,name="muted")
                self.canvas=ui.ZStack()
                with self.canvas:
                    self.input=ui.InvisibleButton(explicit_hover=True)
                    ui.Rectangle(name="plot_background",style={"background_color":cl.pvd_field})
                    self.curve_frame=ui.Frame(build_fn=self._build_curves,opaque_for_mouse_events=False)
                    self.overlay=ui.Frame(build_fn=self._build_overlay,opaque_for_mouse_events=False)
                    self.hover_overlay=HoverOverlay(self._passive_hover)
            self.input.set_mouse_pressed_fn(self._press)
            self.input.set_mouse_moved_fn(self._move)
            self.input.set_mouse_released_fn(self._release)
            self.input.set_mouse_double_clicked_fn(self._double)
            self.input.set_mouse_wheel_fn(self._wheel)
            with ui.HStack(height=20):
                ui.Spacer(width=86)
                self.x_left=ui.Label("",name="muted")
                ui.Label("Recorded frame",name="muted",alignment=ui.Alignment.CENTER)
                self.x_right=ui.Label("",name="muted",alignment=ui.Alignment.RIGHT_CENTER)
            with ui.HStack(height=fl.pvd_control,spacing=fl.pvd_spacing):
                ui.Label("From",width=44)
                self.start_model=ui.SimpleIntModel(0)
                ui.IntField(self.start_model,width=90)
                self.start_model.add_begin_edit_fn(lambda _:self._field_edit("start",True))
                self.start_model.add_end_edit_fn(lambda _:self._field_edit("start",False))
                ui.Label("To",width=24)
                self.stop_model=ui.SimpleIntModel(0)
                ui.IntField(self.stop_model,width=90)
                self.stop_model.add_begin_edit_fn(lambda _:self._field_edit("stop",True))
                self.stop_model.add_end_edit_fn(lambda _:self._field_edit("stop",False))
                ui.Button("Zoom to range",width=130,clicked_fn=self.zoom)
            with ui.HStack(height=fl.pvd_control,spacing=fl.pvd_spacing):
                self.undo_button=ui.Button("Undo zoom",width=105,enabled=False,clicked_fn=lambda:self._action("undo"))
                self.redo_button=ui.Button("Redo zoom",width=105,enabled=False,clicked_fn=lambda:self._action("redo"))
                ui.Button("Fit",width=60,clicked_fn=self.fit)
            self.cursor=ui.Label("Hover for exact values; double-click to inspect the frame.",height=38,name="muted",word_wrap=True)
            ui.Label("Mouse: left pan · right rectangle · wheel / +/- zoom · Ctrl+right / Home fit · Shift+O/I undo/redo\nArrows: graph crosshair (OS pointer stays put) · Return: start / finish rectangle · Escape: cancel",height=38,name="muted",word_wrap=True,
                explicit_hover=True, tooltip="Arrows move one logical pixel. Hold for 400ms to repeat five pixels every 50ms, paced by UI polling. The operating-system pointer does not move.")

    def get_inspector_state(self):
        return {"kind":"plot","title":self.title,"source_label":self.source_label,"visible":self.window.visible,
                "finished":self.finished,"has_values":bool(self.model and self.model.has_values),
                "range":vars(self.model.view) if self.model else None,
                "keyboard_cursor":self.keyboard.visible}

    def poll(self):
        if self.destroyed:return
        if not self.window.visible and not getattr(self.window,"embedded",False):
            self._cancel_keyboard()
            self.job.cancel();self.finished=True;return
        self._poll_keyboard()
        signature=(self.canvas.computed_width,self.canvas.computed_height,
                   ui.ColorStore.find("pvd_accent"),ui.ColorStore.find("pvd_warning"))
        if self.model and signature!=self.chart_signature:
            self._draw_curves();self.overlay.rebuild()
        if self.finished:return
        done,total=self.job.progress
        self.progress.text=f"Sampling {done:,} / {total:,} frames" if total else "Loading capture measurements…"
        if not self.job.future.done():return
        self.finished=True
        self.cancel_button.enabled=False
        try:
            self.series=self.job.future.result()
            if isinstance(self.series,(ScalarSeries,ProfileSeries)):
                curves=(Curve(self.series.name,tuple(self.series.values),self.series.unit),)
            else:
                curves=(Curve("Inner frame",tuple(self.series.inner_ms),"ms"),Curve("Total frame",tuple(self.series.total_ms),"ms"))
            self.model=PlotModel(self.series.frames,curves)
        except Exception as error:
            self.progress.text=f"Analysis unavailable: {error or type(error).__name__}"
            return
        self.progress.text=f"{len(self.series.frames):,} recorded frames"
        if isinstance(self.series,ScalarSeries):self.progress.text+=f" · {self.series.missing:,} missing values"
        self.legend.rebuild()
        self.export_button.enabled=True
        self.curve_frame.rebuild()
        self._sync()
        if not self.model.has_values:self.cursor.text="This range has no recorded values; gaps remain empty in the graph and CSV."

    def _build_legend(self):
        if not self.model:return
        with ui.HStack(spacing=fl.pvd_spacing):
            for i,curve in enumerate(self.model.curves):
                ui.Label(curve.name+(f" ({curve.unit})" if curve.unit else ""),
                         style={"color":cl.pvd_accent if i==0 else cl.pvd_warning})

    def _build_curves(self):
        if not self.model:return
        projection=[1,0,0,0, 0,1,0,0, 0,0,2e-7,0, 0,0,1,1]
        self.chart_scene=sc.SceneView(sc.CameraModel(projection,sc.Matrix44.get_translation_matrix(0,0,-1)),
            aspect_ratio_policy=sc.AspectRatioPolicy.STRETCH,opaque_for_mouse_events=False)
        self._draw_curves()

    def _draw_curves(self):
        if not self.chart_scene or not self.model:return
        width,height=max(1,self.canvas.computed_width),max(1,self.canvas.computed_height)
        self.chart_signature=(self.canvas.computed_width,self.canvas.computed_height,
                              ui.ColorStore.find("pvd_accent"),ui.ColorStore.find("pvd_warning"))
        self.chart_scene.scene.clear()
        with self.chart_scene.scene:
            for i,curve in enumerate(self.model.curves):
                color=cl.pvd_accent if i==0 else cl.pvd_warning
                vertices=[]
                isolated=[]
                for path in self.model.display_paths(curve,limit=2048):
                    if len(path)==1:isolated.append((path[0][0]*2-1,path[0][1]*2-1,0))
                    for a,b in zip(path,path[1:]):
                        dx,dy=(b[0]-a[0])*width,(b[1]-a[1])*height
                        length=math.hypot(dx,dy)
                        if length<1e-9:continue
                        nx,ny=-dy/length*.7/width,dx/length*.7/height
                        # Separate native quads avoid ImGui's acute polyline
                        # joins thinning a narrow recorded spike into dots.
                        vertices.extend((max(-1,min(1,x*2-1)),max(-1,min(1,y*2-1)),0)
                            for x,y in ((a[0]+nx,a[1]+ny),(a[0]-nx,a[1]-ny),
                                        (b[0]-nx,b[1]-ny),(b[0]+nx,b[1]+ny)))
                if vertices:sc.PolygonMesh(vertices,[color]*len(vertices),[4]*(len(vertices)//4),list(range(len(vertices))))
                if isolated:sc.Points(isolated,colors=[color],sizes=[3.])

    def _build_overlay(self):
        if not self.model:return
        r=self.model.view
        projection=[1,0,0,0, 0,1,0,0, 0,0,2e-7,0, 0,0,1,1]
        scene=sc.SceneView(sc.CameraModel(projection,sc.Matrix44.get_translation_matrix(0,0,-1)),
            aspect_ratio_policy=sc.AspectRatioPolicy.STRETCH,opaque_for_mouse_events=False)
        with scene.scene:
            # Native scene guides never participate in widget layout. In
            # particular, a cursor at the last frame must not grow the window.
            for fraction in (.25,.5,.75):
                sc.Line((-1,fraction*2-1,0),(1,fraction*2-1,0),color=cl.pvd_border,thickness=1.)
            frame=self.model.cursor_frame
            if frame is not None and r.left<=frame<=r.right:
                x=(frame-r.left)/r.width*2-1
                sc.Line((x,-1,0),(x,1,0),color=cl.pvd_muted,thickness=1.)
                for _,value,_ in self.model.cursor_values():
                    if value is not None and math.isfinite(value) and r.bottom<=value<=r.top:
                        y=(value-r.bottom)/r.height*2-1
                        sc.Line((-1,y,0),(1,y,0),color=cl.pvd_muted,thickness=1.)
            if self.drag and self.drag[0]==1:
                _,x0,y0,x1,y1,_=self.drag
                left,right=sorted(max(-1,min(1,x*2-1)) for x in (x0,x1))
                bottom,top=sorted(max(-1,min(1,1-y*2)) for y in (y0,y1))
                corners=((left,bottom,0),(left,top,0),(right,top,0),(right,bottom,0),(left,bottom,0))
                for first,last in zip(corners,corners[1:]):
                    sc.Line(first,last,color=cl.pvd_accent,thickness=1.)
            if self.keyboard.visible and self.keyboard.position is not None:
                width,height=max(1,self.canvas.computed_width),max(1,self.canvas.computed_height)
                px,py=self.keyboard.position
                x,y=px/width*2-1,1-py/height*2
                # Raw two-dimensional pixels, independent of the snapped frame
                # and recorded-value guides above. All marks ignore hit testing.
                sc.Line((max(-1,x-10/width),y,0),(min(1,x+10/width),y,0),color=cl.pvd_accent,thickness=2.)
                sc.Line((x,max(-1,y-10/height),0),(x,min(1,y+10/height),0),color=cl.pvd_accent,thickness=2.)
                if self.keyboard.active:
                    ax,ay=self.keyboard.anchor
                    a,b=ax/width*2-1,1-ay/height*2
                    corners=((a,b,0),(a,y,0),(x,y,0),(x,b,0),(a,b,0))
                    for first,last in zip(corners,corners[1:]):
                        sc.Line(first,last,color=cl.pvd_accent,thickness=1.)

    def _sync(self):
        if not self.model:return
        r=self.model.view
        self.start,self.stop=r.left,r.right
        self.start_model.set_value(math.ceil(r.left));self.stop_model.set_value(math.floor(r.right))
        self.x_left.text=f"{r.left:.6g}";self.x_right.text=f"{r.right:.6g}"
        self.y_top.text=f"{r.top:.6g}";self.y_bottom.text=f"{r.bottom:.6g}"
        self.undo_button.enabled=self.model.history_index>0
        self.redo_button.enabled=self.model.history_index<len(self.model.history)-1
        self._draw_curves()
        self.overlay.rebuild()

    def _action(self,name,*args):
        self._cancel_keyboard()
        if self.model:
            getattr(self.model,name)(*args);self._sync()

    def fit(self):self._action("home")

    def zoom(self):
        if not self.model:return
        self._cancel_keyboard()
        try:self.model.frame_range(self.start_model.as_int,self.stop_model.as_int)
        except ValueError as error:self.cursor.text=str(error);return
        self._sync()

    def _fraction(self,x,y):
        return ((x-self.canvas.screen_position_x)/max(1,self.canvas.computed_width),
                (y-self.canvas.screen_position_y)/max(1,self.canvas.computed_height))

    def _hover(self,x,y):
        if not self.model:return
        xf,yf=self._fraction(x,y)
        values=self.model.hover(xf)
        if values is None:self.cursor.text="Outside the recorded frame range."
        else:
            texts=[f"{name}: {value:.8g} {unit}" if value is not None and math.isfinite(value) else f"{name}: no recorded value" for name,value,unit in values]
            self.cursor.text=f"Frame {self.model.cursor_frame:,} · "+" · ".join(texts)
        self.overlay.rebuild()

    def _passive_hover(self,xf,yf):
        if not self.model or self.drag:return
        changed=self.keyboard.pointer(xf*self.canvas.computed_width,yf*self.canvas.computed_height)
        if changed:self._sync_keyboard_navigation()
        if self.keyboard.visible:
            if changed:self._keyboard_hover()
            return
        self._hover(self.canvas.screen_position_x+xf*self.canvas.computed_width,
                    self.canvas.screen_position_y+yf*self.canvas.computed_height)

    def _press(self,x,y,button,modifiers):
        if not self.model:return
        if button not in (0,1):return
        self._cancel_keyboard()
        self.editing_fields.clear()
        self._hover(x,y)
        xf,yf=self._fraction(x,y)
        if button==1 and modifiers&2:
            self.drag=(3,xf,yf,xf,yf,self.model.view)
        else:self.drag=(button,xf,yf,xf,yf,self.model.view)

    def _move(self,x,y,modifiers,buttons):
        if self.drag:
            button,x0,y0,_,_,origin=self.drag
            xf,yf=self._fraction(x,y)
            self.drag=(button,x0,y0,xf,yf,origin)
            if button==0:self.model.pan(xf-x0,yf-y0,origin=origin);self._sync()
            else:self.overlay.rebuild()
        self._hover(x,y)

    def _release(self,x,y,button,modifiers):
        if not self.drag:return
        kind,x0,y0,_,_,origin=self.drag
        self.drag=None
        xf,yf=self._fraction(x,y)
        if kind==3:self.model.home()
        elif kind==1:
            width,height=self.canvas.computed_width,self.canvas.computed_height
            rectangle=accepted_rectangle((x0*width,y0*height),(xf*width,yf*height))
            if rectangle is not None:
                self._commit_rectangle(rectangle,origin)
            else:
                self.context_menu=ui.Menu("Graph",tearable=False)
                with self.context_menu:ui.MenuItem("Save to CSV",triggered_fn=self.export)
                self.context_menu.show_at(x,y)
        self._sync()
        self._hover(x,y)

    def _double(self,x,y,button,modifiers):
        if button==0 and self.model:
            self._cancel_keyboard()
            self.drag=None;self._hover(x,y)
            if self.model.cursor_frame is not None:self.on_frame(self.model.cursor_frame)

    def _wheel(self,x,y,modifiers):
        if not (modifiers & 15):self._action("wheel",y)

    def _key(self,key,modifiers,pressed):
        key=normalize_key(key)
        if not pressed:
            self.keyboard.release(key)
            return
        if not self._keyboard_allowed():
            self._cancel_keyboard()
            return
        chord=modifiers & 15
        if chord:self.keyboard.stop_repeat()
        if chord==0 and key in (*KeyboardCrosshair.ARROWS,257,335):
            if self.drag:return
            self.keyboard.configure(self.canvas.computed_width,self.canvas.computed_height)
            if key in KeyboardCrosshair.ARROWS:self.keyboard.arrow(key,time.monotonic())
            elif self.keyboard.active:
                rectangle,origin=self.keyboard.finish()
                if rectangle is not None:self._commit_rectangle(rectangle,origin)
                self._sync()
            else:self.keyboard.begin(self.model.view)
            self._keyboard_hover()
        elif chord==0 and key==268:self.fit()
        elif chord==1 and key==79:self._action("undo")
        elif chord==1 and key==73:self._action("redo")
        # Keep '=' / Shift+'=' and keypad aliases alongside the Qwt magnifier.
        elif (key==61 and chord in (0,1)) or (key==334 and chord==0):self._action("magnify",.9)
        elif chord==0 and key in (45,333):self._action("magnify",1/.9)
        elif chord==0 and key==256:
            self._cancel_keyboard()
            if self.drag:self.drag=None;self.overlay.rebuild()

    def _commit_rectangle(self,rectangle,origin):
        left,top,right,bottom=rectangle
        width,height=max(1,self.canvas.computed_width),max(1,self.canvas.computed_height)
        self.model.rectangle(left/width,top/height,right/width,bottom/height,origin=origin)

    def _keyboard_allowed(self):
        return (not self.destroyed and self.model is not None and self.window.visible
            and self.window.focused and not self.editing_fields and ui.Menu.get_current() is None
            and not (self.export_dialog and self.export_dialog.window and self.export_dialog.window.visible))

    def _cancel_keyboard(self):
        if self.keyboard.cancel():
            self._clear_keyboard_label()
            self.overlay.rebuild()
        self._sync_keyboard_navigation()

    def _clear_keyboard_label(self):
        if self.cursor.text.startswith("Keyboard cursor "):
            self.cursor.text=self.cursor.text.split(" · ",1)[-1]

    def _sync_keyboard_navigation(self):
        # Match the workspace's public no-navigation-input flag while arrows
        # own the crosshair; restore the previous bit when a field takes focus.
        flag=1<<16
        if self.keyboard.visible and self.keyboard_navigation_flags is None:
            self.keyboard_navigation_flags=self.window.flags & flag
            self.window.flags|=flag
        elif not self.keyboard.visible and self.keyboard_navigation_flags is not None:
            self.window.flags=(self.window.flags & ~flag)|self.keyboard_navigation_flags
            self.keyboard_navigation_flags=None

    def _field_edit(self,name,editing):
        if editing:
            self.editing_fields.add(name)
            self._cancel_keyboard()
        else:self.editing_fields.discard(name)

    def _keyboard_hover(self):
        if not self.keyboard.visible or self.keyboard.position is None:return
        self._sync_keyboard_navigation()
        x,y=self.keyboard.position
        self._hover(self.canvas.screen_position_x+x,self.canvas.screen_position_y+y)
        self.cursor.text=f"Keyboard cursor {x}, {y} px · "+self.cursor.text

    def _poll_keyboard(self):
        changed=self.keyboard.configure(self.canvas.computed_width,self.canvas.computed_height)
        if not self._keyboard_allowed():
            self._cancel_keyboard()
        elif self.keyboard.repeat(time.monotonic()):self._keyboard_hover()
        if changed:
            self._clear_keyboard_label()
            self._sync_keyboard_navigation()
            self.overlay.rebuild()

    def export(self):
        if self.series is None:return
        self._cancel_keyboard()
        if self.export_dialog:self.export_dialog.destroy()
        series=self.series
        exporter=export_series_csv if isinstance(series,ScalarSeries) else export_profile_csv if isinstance(series,ProfileSeries) else export_frame_timing_csv
        self.export_dialog=CaptureFileDialog(lambda p:exporter(series,p,overwrite=p.exists()),
            mode="save",initial_path=Path.home()/"capture-history.csv",extensions=(".csv",),
            default_extension=".csv",title="Export graph data")
        self.export_dialog.show()

    def destroy(self):
        if self.destroyed:return
        self._cancel_keyboard()
        self.destroyed=True;self.job.cancel()
        self.hover_overlay.destroy()
        if self.chart_scene:self.chart_scene.scene.clear()
        if self.export_dialog:self.export_dialog.destroy()
        if self.context_menu:self.context_menu.hide()
        self.window.destroy()

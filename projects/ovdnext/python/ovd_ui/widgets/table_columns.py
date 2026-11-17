# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A shared-width table header with deferred, captured OVUI column resize grips."""
import math
import omni.ui as ui

from ..style import layout as fl
from .splitter import SplitGrip


class ColumnWidths:
    def __init__(self, defaults, minimums, maximums, *, stretch=0):
        self.defaults=tuple(defaults)
        self.minimums,self.maximums=tuple(minimums),tuple(maximums)
        self.stretch=stretch
        self.reset()

    def reset(self):
        self.widths=self.defaults
        self.manual=False

    def layout(self, available):
        widths=list(self.widths)
        if not self.manual and math.isfinite(available):
            index=self.stretch
            widths[index]=min(self.maximums[index],widths[index]+max(0,available-sum(widths)))
        return tuple(widths)

    def resize(self,index,width,displayed):
        if not math.isfinite(width):return
        widths=list(self.widths if self.manual else displayed)
        widths[index]=max(self.minimums[index],min(self.maximums[index],width))
        self.widths=tuple(widths)
        self.manual=True

    def state(self):return {'widths':self.widths,'manual':self.manual}

    def restore(self,state):
        self.widths=tuple(state['widths'])
        self.manual=state['manual']


class _ColumnTarget:
    """Adapt a column to the same queued grip contract as the side panels."""
    def __init__(self,table,index):self.table,self.index=table,index

    @property
    def computed_width(self):return self.table.displayed[self.index]

    @property
    def width(self):return ui.Pixel(self.computed_width)

    @width.setter
    def width(self,value):
        self.table.columns.resize(self.index,float(value),self.table.displayed)


class ResizableTable:
    def __init__(self,model,delegate,headers,columns,on_sort,*,numeric_columns=None):
        self.columns,self.headers=columns,tuple(headers)
        self.numeric_columns=set(range(1,len(headers))) if numeric_columns is None else set(numeric_columns)
        self.displayed=columns.widths
        self.cells,self.buttons,self.placers,self.grips=[],[],[],[]
        self.labels,self.sort_marks,self.sort_targets=[],[],[]
        self.closed=False
        total=sum(self.displayed)
        # Both header and body scroll horizontally together. Reserve the same
        # vertical scrollbar width beside the header and beside the rows.
        with ui.ScrollingFrame(vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF) as outer:
            self.outer=outer
            with ui.VStack(width=total+fl.pvd_scrollbar,spacing=0) as content:
                self.content=content
                with ui.ZStack(width=total+fl.pvd_scrollbar,height=fl.pvd_control) as header:
                    self.header=header
                    with ui.HStack(spacing=0):
                        for index,(title,width) in enumerate(zip(self.headers,self.displayed)):
                            with ui.Frame(width=width,horizontal_clipping=True) as cell:
                                self.cells.append(cell)
                                with ui.ZStack():
                                    ui.Rectangle(style_type_name_override='TreeView.Header',opaque_for_mouse_events=False)
                                    # Keep the sort hit target clear of the two
                                    # half-grips that straddle each column edge.
                                    with ui.Placer(offset_x=fl.pvd_splitter_hit_width/2,
                                            width=width-fl.pvd_splitter_hit_width,height=fl.pvd_control) as target:
                                        self.sort_targets.append(target)
                                        button=ui.InvisibleButton(height=fl.pvd_control,
                                            explicit_hover=True,tooltip=self._sort_tooltip(index),
                                            clicked_fn=lambda i=index:on_sort(i)) if on_sort is not None else None
                                        self.buttons.append(button)
                                    with ui.HStack(spacing=0,opaque_for_mouse_events=False):
                                        self.labels.append(ui.Label(title,name='muted',alignment=ui.Alignment.CENTER,
                                            elided_text=True,opaque_for_mouse_events=False))
                                        with ui.VStack(width=16,spacing=0,opaque_for_mouse_events=False):
                                            ui.Spacer()
                                            with ui.HStack(height=4,spacing=0,opaque_for_mouse_events=False):
                                                ui.Spacer()
                                                self.sort_marks.append(ui.Triangle(width=6,height=4,name='object_chevron',
                                                    opaque_for_mouse_events=False))
                                                ui.Spacer()
                                            ui.Spacer()
                    for index in range(len(headers)):
                        with ui.Placer(width=fl.pvd_splitter_hit_width,height=fl.pvd_control,
                                offset_x=sum(self.displayed[:index+1])-fl.pvd_splitter_hit_width/2) as placer:
                            self.placers.append(placer)
                            grip=SplitGrip(_ColumnTarget(self,index),lambda width:None,
                                minimum=columns.minimums[index],maximum=columns.maximums[index])
                            grip.widget.tooltip=f'Drag to resize {self.headers[index]}'
                            self.grips.append(grip)
                with ui.ScrollingFrame(horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
                        vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON):
                    self.tree=ui.TreeView(model,delegate=delegate,header_visible=False,explicit_hover=True,
                        width=total,column_widths=[ui.Pixel(width) for width in self.displayed],
                        columns_resizable=False,auto_scroll_selection=True)

    def update(self):
        if self.closed:return
        for grip in self.grips:grip.update()
        widths=self.columns.layout(max(0,self.outer.computed_width-fl.pvd_scrollbar))
        if widths==self.displayed:return
        self.displayed=widths
        total=sum(widths)
        self.content.width=ui.Pixel(total+fl.pvd_scrollbar)
        self.header.width=ui.Pixel(total+fl.pvd_scrollbar)
        self.tree.width=ui.Pixel(total)
        self.tree.column_widths=[ui.Pixel(width) for width in widths]
        for cell,width in zip(self.cells,widths):cell.width=ui.Pixel(width)
        for target,width in zip(self.sort_targets,widths):target.width=ui.Pixel(width-fl.pvd_splitter_hit_width)
        for index,placer in enumerate(self.placers):
            placer.offset_x=ui.Pixel(sum(widths[:index+1])-fl.pvd_splitter_hit_width/2)

    def _sort_tooltip(self,index):
        return f'Sort numerically by {self.headers[index]}' if index in self.numeric_columns else f'Sort by {self.headers[index]}'

    def set_sort(self,column,descending):
        for index,(label,mark,title) in enumerate(zip(self.labels,self.sort_marks,self.headers)):
            label.text=title
            mark.visible=column==index
            mark.alignment=ui.Alignment.CENTER_BOTTOM if descending else ui.Alignment.CENTER_TOP
            if self.buttons[index] is not None:self.buttons[index].tooltip=self._sort_tooltip(index)
            self.grips[index].widget.tooltip=f'Drag to resize {title}'

    def get_state(self):
        # A release and a frame step can be queued before update() applies the
        # final width. Snapshot that intent without mutating a drawing widget.
        state=self.columns.state()
        pending=[(i,grip.pending_width) for i,grip in enumerate(self.grips) if grip.pending_width is not None]
        if pending:
            widths=list(state['widths'] if state['manual'] else self.displayed)
            for index,width in pending:widths[index]=width
            state={'widths':tuple(widths),'manual':True}
        return state

    def destroy(self):
        if self.closed:return
        self.closed=True
        for grip in self.grips:grip.destroy()

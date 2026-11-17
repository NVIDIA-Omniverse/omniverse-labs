# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Responsive ownership explorer over ObjectTreeModel; no renderer dependencies."""
import asyncio
from dataclasses import dataclass
import re

import omni.ui as ui
from _pvd_native import SnapshotDataUnavailable

from ..camera import normalize_key
from ..models import ObjectItem
from ..style import layout as fl, style_dialog
from .icon_button import icon_button

_MAX_VISUAL_INDENT = 6
_SEARCH_FIELDS = (("name", "type", "id"), ("name",), ("type",), ("id",))


def signed_decimal(text, bits=32):
    """QString's decimal integer domain, without float/hex/substring coercion."""
    text=str(text).strip()
    if not re.fullmatch(r"[+-]?[0-9]+",text):return None
    # Avoid Python's large-integer text limit without rejecting leading zeros.
    digits=text.lstrip("+-").lstrip("0") or "0"
    if len(digits)>len(str(1<<(bits-1))):return None
    value=int(digits)*(-1 if text.startswith("-") else 1)
    return value if -(1<<(bits-1))<=value<(1<<(bits-1)) else None


@dataclass(frozen=True)
class FoundObject:
    identity: int
    label: str
    class_name: str
    alive: bool


def find_objects(adapter, expression, mode=0):
    """Original Name/ID/Type lookup over enumerated native instances.

    Name and Type are case-sensitive literal substrings, including spaces.
    Name reads recorded strings only; lowercase OVD ``name`` is accepted too.
    Query-record IDs belong to the separate hierarchy filter, not this lookup.
    """
    if mode not in (0,1,2):raise ValueError("Unknown object search mode")
    if adapter is None:return []
    expression=str(expression)
    instances=adapter.instances.values()
    if mode==1:
        identity=signed_decimal(expression)
        instance=adapter.instances.get(identity)
        instances=() if instance is None else (instance,)
    rows=[]
    for instance in instances:
        class_name=adapter.class_name(instance.id)
        label=class_name
        if mode==0:
            prop=adapter.property(instance.id,"Name")
            if prop is None or prop.name not in ("Name","name") or prop.type!="StringHandle":continue
            try:label=adapter.session.string_value(instance.id,prop.id)
            except (SnapshotDataUnavailable,IndexError):continue
            if not label or expression not in label:continue
        elif mode==2 and expression not in class_name:continue
        rows.append(FoundObject(instance.id,label,class_name,instance.alive))
    return rows


def find_sort_key(row, column=0):
    value=str(row.identity) if column==0 else row.label
    number=signed_decimal(value,64)
    return (0,number) if number is not None else (1,value.encode("utf-16-be","surrogatepass"))


class _FindItem(ui.AbstractItem):
    def __init__(self, result):
        super().__init__()
        self.result=result
        self.values=[ui.SimpleStringModel(str(result.identity)),ui.SimpleStringModel(result.label)]


class _FindModel(ui.AbstractItemModel):
    def __init__(self):
        super().__init__()
        self.items=[]
        self.headers=[ui.SimpleStringModel(label) for label in ("ID","Name / Type")]
        self.column=0
        # QTreeView's enabled sorting starts at column 0, descending.
        self.reverse=True

    def set_rows(self, rows):
        self.items=[_FindItem(row) for row in rows]
        self.sort()

    def sort(self, column=None):
        if column is not None:
            self.reverse=not self.reverse if column==self.column else False
            self.column=column
        self.items.sort(key=lambda item:find_sort_key(item.result,self.column),reverse=self.reverse)
        self._item_changed(None)

    def get_item_children(self,item=None):return self.items if item is None else []
    def can_item_have_children(self,item=None):return item is None and bool(self.items)
    def get_item_value_model_count(self,item=None):return 2
    def get_item_value_model(self,item,column_id=0):return item.values[column_id] if item else self.headers[column_id]


class _FindDelegate(ui.AbstractItemDelegate):
    def __init__(self,dialog):
        super().__init__()
        self.dialog=dialog

    def build_branch(self,*_):pass

    def build_header(self,column_id):
        label=("ID","Name / Type")[column_id]
        ui.Button(label,name="ghost",height=fl.pvd_control,
            clicked_fn=lambda:self.dialog.sort(column_id),explicit_hover=True, tooltip=f"Sort by {label}; click again to reverse")

    def build_widget(self,model,item,column_id,level,expanded):
        if item is None:return
        result=item.result
        with ui.HStack(height=fl.pvd_row,spacing=fl.pvd_spacing):
            ui.Label(item.values[column_id].as_string,
                name="muted" if not result.alive else "mono" if column_id==0 else "",
                elided_text=True,tooltip=f"{result.label}\n{result.class_name} · ID {result.identity}"+
                    ("\nUnavailable at the current frame" if not result.alive else ""),
                explicit_hover=True,mouse_double_clicked_fn=lambda x,y,button,mods:
                    self.dialog.activate(item) if button==0 else None)
            if column_id==1 and not result.alive:
                ui.Label("Unavailable",name="muted",width=fl.pvd_button_width)


class FindObjectsDialog:
    """Find results are independent of the sidebar filter and main selection."""
    def __init__(self,browser):
        self.browser=browser
        self.model=_FindModel()
        self.delegate=_FindDelegate(self)
        self.expression=ui.SimpleStringModel("")
        self.mode=0
        self.focus="expression"
        self.capture_key=None
        self._selecting=False
        self.window=ui.Window("Find objects",width=fl.pvd_find_width,
            height=fl.pvd_find_height+fl.pvd_row*6,
            flags=ui.WINDOW_FLAGS_NO_DOCKING|ui.WINDOW_FLAGS_NO_TITLE_BAR|ui.WINDOW_FLAGS_MODAL)
        style_dialog(self.window)
        self.window.set_key_pressed_fn(self.key)
        with self.window.frame:
            with ui.ZStack():
                ui.Rectangle(name="dialog")
                with ui.HStack():
                    ui.Spacer(width=fl.pvd_padding)
                    with ui.VStack(spacing=fl.pvd_spacing):
                        ui.Spacer(height=fl.pvd_padding)
                        ui.Label("Find objects",name="title",height=fl.pvd_control)
                        with ui.HStack(height=fl.pvd_control,spacing=fl.pvd_spacing):
                            self.field=ui.StringField(self.expression,
                                explicit_hover=True, tooltip="Name and Type use case-sensitive text; ID uses an exact signed decimal integer.")
                            self.criteria=ui.ComboBox(0,"By Name","By ID","By Type",width=fl.pvd_button_width)
                            self.criteria.model.add_item_changed_fn(self._mode_changed)
                            ui.Button("Find",name="primary",width=fl.pvd_button_width,clicked_fn=self.search)
                        self.status=ui.Label("",name="muted",height=fl.pvd_row,elided_text=True)
                        with ui.ScrollingFrame(horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF):
                            self.tree=ui.TreeView(self.model,delegate=self.delegate,header_visible=True,
                                root_visible=True,columns_resizable=True,
                                column_widths=[ui.Pixel(fl.pvd_button_width),ui.Fraction(1)],
                                auto_scroll_selection=True)
                            self.tree.set_selection_changed_fn(self._selection_changed)
                        with ui.HStack(height=fl.pvd_control,spacing=fl.pvd_spacing):
                            ui.Spacer()
                            self.select_button=ui.Button("Select",name="primary",width=fl.pvd_button_width,
                                enabled=False,clicked_fn=self.activate)
                            ui.Button("Cancel",width=fl.pvd_button_width,clicked_fn=self.close)
                        ui.Spacer(height=fl.pvd_padding)
                    ui.Spacer(width=fl.pvd_padding)
        self.expression.add_begin_edit_fn(lambda _model:self._focus_expression())

    def _focus_expression(self):
        self.focus="expression"
        self.window.flags=getattr(self.window,"flags",0)&~(1<<16)

    def _focus_results(self):
        self.focus="results"
        # TreeView has no public keyboard-focus setter in this wheel. Own its
        # Up/Down/Return events and suppress concurrent ImGui button navigation.
        self.window.flags=getattr(self.window,"flags",0)|(1<<16)

    def show(self):
        self.expression.set_value("")
        self.criteria.model.get_item_value_model().set_value(0)
        self.mode=0
        self.model.column,self.model.reverse=0,True
        self.clear()
        self.status.text="Search recorded names, native IDs or class names."
        self._focus_expression()
        self.window.visible=True
        self.window.focus()
        self.field.focus_keyboard()

    def clear(self):
        self.tree.clear_selection()
        self.model.set_rows([])
        self.select_button.enabled=False
        self.capture_key=None

    def _mode_changed(self,model,_item):
        self.mode=model.get_item_value_model().as_int
        self.clear()
        self.status.text="Press Find to search by "+("name.","exact signed decimal ID.","class name.")[self.mode]
        self._focus_expression()

    def search(self):
        adapter=self.browser.model._adapter
        rows=find_objects(adapter,self.expression.as_string,self.mode)
        self.clear()
        self.capture_key=getattr(adapter.session,"capture_key",id(adapter.session)) if adapter else None
        self.model.set_rows(rows)
        self.select_button.enabled=len(rows)==1
        self.status.text=f"{len(rows):,} result{'s' if len(rows)!=1 else ''}"
        if self.mode==1 and signed_decimal(self.expression.as_string) is None:
            self.status.text="Enter a signed decimal ID between -2147483648 and 2147483647."
        self._focus_results()
        self.field.focus_keyboard(False)

    def _selection_changed(self,items):
        if self._selecting:return
        self._selecting=True
        try:
            if len(items)>1:self.tree.selection=[items[-1]]
        finally:self._selecting=False
        self.select_button.enabled=bool(items) or len(self.model.items)==1
        if items:self._focus_results()

    def sort(self,column):
        self.model.sort(column)
        self._focus_results()

    def activate(self,item=None):
        item=item or (self.tree.selection[-1] if self.tree.selection else
                      self.model.items[0] if len(self.model.items)==1 else None)
        if item is None:return False
        adapter=self.browser.model._adapter
        current=getattr(adapter.session,"capture_key",id(adapter.session)) if adapter else None
        if adapter is None or current!=self.capture_key or item.result.identity not in adapter.instances:
            self.status.text="The capture changed. Search again before selecting."
            return False
        if self.browser.activate_found(item.result.identity) is False:return False
        self.close()
        return True

    def key(self,key,modifiers,pressed):
        if not pressed:return
        key=normalize_key(key)
        if key==256:self.close()
        elif key==258 and self.focus=="results":
            self._focus_expression()
            self.field.focus_keyboard()
        elif key in (257,335):
            if self.focus=="expression":self.search()
            else:self.activate()
        elif key in (264,265) and self.focus=="results" and self.model.items:
            items=self.model.items
            selected=self.tree.selection[-1] if self.tree.selection else None
            index=items.index(selected) if selected in items else (-1 if key==264 else len(items))
            self.tree.selection=[items[max(0,min(len(items)-1,index+(1 if key==264 else -1)))]]

    def close(self):
        self.window.visible=False

    def destroy(self):self.window.destroy()


def row_text(item):
    """Recorded name first; class/identity remain legible on their own line."""
    title = re.sub(r" \[id:-?\d+\]$", "", item.values[0].as_string)
    if item.kind == "reference":
        title = title.removeprefix("↗ ")
    identity = item.instance_id if item.instance_id is not None else item.query_id
    detail = item.class_name
    if item.kind == "reference":
        detail = f"Reference · {detail}"
    if identity is not None:
        detail += f" · #{identity}"
    elif item.kind in ("array", "array_page"):
        detail += (f" · {item.array_error}" if item.array_error else f" · {item.element_count:,} records")
    elif not detail:
        detail = "Property group" if item.kind == "group" else "Page"
    return title, detail


def _selection_background(*, left=False, right=False):
    """Passive, disjoint pieces inheriting the native cell's selected state.

    Clip a half of an ordinary all-corners rectangle for each end. The pinned
    public CornerFlag values are not compatible with its current fill backend;
    no corner flags or framework internals are needed here. Clipping also keeps
    an unusually narrow cell from enlarging the native column allocation.
    """
    radius = fl.pvd_row_radius
    with ui.Frame(height=fl.pvd_object_row, horizontal_clipping=True,
                  vertical_clipping=True, opaque_for_mouse_events=False) as background:
        with ui.HStack(height=fl.pvd_object_row, spacing=0, opaque_for_mouse_events=False):
            if left:
                with ui.Frame(width=radius, height=fl.pvd_object_row, horizontal_clipping=True,
                              vertical_clipping=True, opaque_for_mouse_events=False):
                    ui.Rectangle(name="object_selection_cap", width=radius*2,
                                 height=fl.pvd_object_row, opaque_for_mouse_events=False)
            ui.Rectangle(name="object_selection_middle", height=fl.pvd_object_row,
                         opaque_for_mouse_events=False)
            if right:
                with ui.Frame(width=radius, height=fl.pvd_object_row, horizontal_clipping=True,
                              vertical_clipping=True, opaque_for_mouse_events=False):
                    with ui.Placer(offset_x=-radius, opaque_for_mouse_events=False):
                        ui.Rectangle(name="object_selection_cap", width=radius*2,
                                     height=fl.pvd_object_row, opaque_for_mouse_events=False)
    return background


class _ObjectDelegate(ui.AbstractItemDelegate):
    def __init__(self, browser):
        super().__init__()
        self.browser = browser
        self._row_sizes = {}
        self._refresh_pending = False

    def build_header(self, column_id):
        if column_id == 1 or column_id > 1 and self.browser.on_action is None:return
        label, tooltip = (("Scene", "Select objects without moving the camera"), ("", ""),
            ("Vis", "Show or hide this subtree"),
            ("Trsp", "Transparency tag: applies only in Tagged mode"))[column_id]
        ui.Label(label, name="muted", height=fl.pvd_control,
            alignment=ui.Alignment.LEFT_CENTER if column_id == 0 else ui.Alignment.CENTER,
            explicit_hover=True, tooltip=tooltip)

    def _apply_fitted_rows(self):
        self._refresh_pending=False
        self.browser.tree.dirty_widgets()

    def _remember_fit(self,key,title,natural,width):
        previous=self._row_sizes.get(key)
        self._row_sizes[key]=(title,natural,width)
        if len(self._row_sizes)>4096:
            oldest=next(iter(self._row_sizes))
            self._row_sizes.pop(oldest)
        if previous is None or previous[0]!=title or abs(previous[2]-width)>.5:
            # TreeView caches each node's layout independently of child dirty
            # flags. Start rebuilt cells at their known final width, once for
            # all rows fitted in this event-loop turn.
            if not self._refresh_pending:
                self._refresh_pending=True
                asyncio.get_event_loop().call_soon(self._apply_fitted_rows)

    def build_branch(self, model, item, column_id, level, expanded):
        if column_id or item is None:
            return
        indent = fl.pvd_object_indent * min(max(0,level-1), _MAX_VISUAL_INDENT)
        with ui.ZStack(width=indent + fl.pvd_object_branch, height=fl.pvd_object_row,
                       opaque_for_mouse_events=False):
            _selection_background(left=True)
            with ui.HStack(width=indent + fl.pvd_object_branch, height=fl.pvd_object_row):
                ui.Spacer(width=indent)
                with ui.VStack(width=fl.pvd_object_branch, height=fl.pvd_object_row,
                               spacing=0, opaque_for_mouse_events=False):
                    ui.Spacer(opaque_for_mouse_events=False)
                    if model.can_item_have_children(item):
                        with ui.HStack(height=fl.pvd_object_chevron, spacing=0,
                                       opaque_for_mouse_events=False):
                            ui.Spacer(opaque_for_mouse_events=False)
                            chevron = ui.Triangle(name="object_chevron", width=fl.pvd_object_chevron,
                                height=fl.pvd_object_chevron, explicit_hover=True,
                                alignment=ui.Alignment.CENTER_BOTTOM if expanded else ui.Alignment.RIGHT_CENTER)
                            chevron.set_mouse_pressed_fn(lambda x, y, button, modifiers, node=item:
                                self.browser.toggle_expanded(node) if button == 0 else None)
                            ui.Spacer(opaque_for_mouse_events=False)
                    ui.Spacer(opaque_for_mouse_events=False)

    def build_widget(self, model, item, column_id, level, expanded):
        if item is None:
            return
        if column_id in (2, 3):
            if self.browser.on_action is not None:
                # Empty group utilities still occupy these native columns.
                with ui.ZStack(height=fl.pvd_object_row, opaque_for_mouse_events=False):
                    _selection_background(right=column_id==3)
                    self.browser.build_tag(item, column_id)
            return
        if column_id:
            return
        title, detail = row_text(item)
        if item.kind=="reference":detail+="\nDouble-click to reveal the referenced object."
        # TreeView has already subtracted the branch/indent width. Keep this
        # remaining cell allocation independent of the previous label width;
        # otherwise a fixed text width can prevent the panel from shrinking.
        with ui.ZStack(height=fl.pvd_object_row, opaque_for_mouse_events=False):
            _selection_background(right=self.browser.on_action is None)
            self._build_content(item, title, detail)

    def _build_content(self, item, title, detail):
        with ui.Frame(height=fl.pvd_object_row, horizontal_clipping=True,
                      vertical_clipping=True, explicit_hover=True,
                      opaque_for_mouse_events=False,
                      mouse_pressed_fn=lambda x,y,button,mods:
                          self.browser.row_pressed(item.key,x,y,button,mods),
                      mouse_double_clicked_fn=lambda x,y,button,mods:
                          self.browser.activate_row(item.key) if button==0 and not mods & 15 else None) as row:
            # Parent TreeView/Frame hover guards do not propagate to this
            # cached tooltip-bearing group. Tag icons guard their own button.
            with ui.HStack(height=fl.pvd_object_row, spacing=6, explicit_hover=True) as content:
                self.browser._set_row_tooltip((item.key,0),content,f"{title}\n{detail}")
                name_label=ui.Label(title, name="object_name", height=fl.pvd_object_row,
                         alignment=ui.Alignment.LEFT_CENTER, elided_text=False)
                natural_width=name_label.exact_content_width
                name_label.elided_text=True
                name_label.text=""
                name_label.text=title
                identity=item.instance_id if item.instance_id is not None else item.query_id
                if identity is not None:
                    cached=self._row_sizes.get(item.key)
                    if cached is not None and cached[0]==title:
                        natural_width=cached[1]
                        name_label.width=ui.Pixel(cached[2])
                    label=f"#{identity}"
                    identity_width=min(112,len(label)*7)
                    ui.Label(label,name="mono_small",width=identity_width,height=fl.pvd_object_row,
                        alignment=ui.Alignment.LEFT_CENTER,elided_text=True)
                    ui.Spacer()
                    def fit_name():
                        available=max(0,row.computed_width-identity_width-12)
                        self._remember_fit(item.key,title,natural_width,min(natural_width,available))
                    def measure_name():
                        nonlocal natural_width
                        # Measure in the label's draw context, after inherited
                        # style and the window's actual font size have resolved.
                        name_label.elided_text=False
                        # Elision changes do not clear this wheel's text cache.
                        name_label.text=""
                        name_label.text=title
                        natural_width=name_label.exact_content_width
                        name_label.elided_text=True
                        name_label.text=""
                        name_label.text=title
                        _=name_label.exact_content_width
                        fit_name()
                    # Once text has a pixel width, its own size need not change
                    # when the parent grows. Observe the allocated cell instead.
                    row.set_computed_content_size_changed_fn(fit_name)
                    name_label.set_computed_content_size_changed_fn(measure_name)


class ObjectBrowser:
    """Construct inside the parent's UI scope; retain until shutdown.

    `tree` remains the native OVUI TreeView. Selection callbacks receive the
    original ObjectItems, including reference and typed-array metadata. Call
    `refresh()` after model.populate; `show_find()` opens independent Find
    results. `on_find_object(native_id)` optionally handles explicit inspection
    of objects outside their current lifetime; return False to keep Find open.
    Hosts dispatch keys to `search_key` before their text-input guard and provide
    `search_window_focused`; only the active sidebar expression handles Enter.
    """
    def __init__(self, model, on_selection_changed, *, on_activate=None,on_find_object=None,
                 on_action=None, row_tags=None, on_focus=None,search_window_focused=None):
        self.model = model
        self.on_selection_changed = on_selection_changed
        self._changing_group = False
        self.on_activate = on_activate
        self.on_find_object=on_find_object
        self.on_action=on_action
        self.on_focus=on_focus
        self.row_tags=row_tags or (lambda _row:(False,False))
        self._tag_controls={}
        self._row_tooltips={}
        self._row_tooltips_suppressed=False
        self.context_menu=None
        self._menu_generation=0
        self._destroyed=False
        self.search_model = ui.SimpleStringModel(model.filter)
        # Filter options shares search_model, but its editing must not grant
        # sidebar keyboard ownership. Keep only the sidebar editor separate.
        self.search_editor_model = ui.SimpleStringModel(model.filter)
        self.search_window_focused=search_window_focused or (lambda:False)
        self._search_editing=False
        self._search_refocus=False
        self.search_editor_model.add_value_changed_fn(self._search_edited)
        self.search_editor_model.add_begin_edit_fn(self._begin_search_edit)
        self.search_editor_model.add_end_edit_fn(self._end_search_edit)
        self.match_case = ui.SimpleBoolModel(False)
        self.whole_word = ui.SimpleBoolModel(False)
        self.fields_index = 0
        self.find_window = None
        self.find_dialog = None
        self.filter_window = None
        self.menu = None
        self.tree = None
        self._matches = []
        self._last_search_signature = None
        self._last_projection = None
        self.delegate = _ObjectDelegate(self)
        self.frame = ui.Frame()
        with self.frame:
            self._build()
        self.search_model.add_value_changed_fn(self._search_changed)

    def _build(self):
        with ui.VStack(spacing=fl.pvd_spacing):
            with ui.HStack(height=fl.pvd_field_height, spacing=fl.pvd_spacing):
                with ui.ZStack():
                    self.search_background=ui.Rectangle(name="hierarchy_search",opaque_for_mouse_events=False)
                    with ui.HStack(opaque_for_mouse_events=False):
                        # AbstractField only supports symmetric padding. Keep
                        # its native 4px padding and account for it in the gap,
                        # so empty and edited text share the CSS's 34px inset.
                        with ui.ZStack(width=fl.pvd_search_inset+fl.pvd_search_icon_size+
                                             fl.pvd_spacing-fl.pvd_compact_padding):
                            self.search_prefix=ui.InvisibleButton(clicked_fn=self._focus_search)
                            with ui.HStack(opaque_for_mouse_events=False):
                                ui.Spacer(width=fl.pvd_search_inset)
                                with ui.VStack(width=fl.pvd_search_icon_size,opaque_for_mouse_events=False):
                                    ui.Spacer()
                                    ui.Circle(width=fl.pvd_search_icon_size,height=fl.pvd_search_icon_size,
                                        radius=fl.pvd_search_icon_radius,size_policy=ui.CircleSizePolicy.FIXED,
                                        name="hierarchy_search",opaque_for_mouse_events=False)
                                    ui.Spacer()
                                ui.Spacer(width=fl.pvd_spacing-fl.pvd_compact_padding)
                        with ui.ZStack():
                            with ui.VStack():
                                ui.InvisibleButton(clicked_fn=self._focus_search,
                                    height=(fl.pvd_field_height-fl.pvd_search_editor_height)/2)
                                # Keep backend navigation decoration inside the
                                # editor; the themed surround owns focus chrome.
                                with ui.Frame(height=fl.pvd_search_editor_height,
                                              horizontal_clipping=True,vertical_clipping=True):
                                    self.search_field=ui.StringField(self.search_editor_model,
                                        name="hierarchy_search",height=fl.pvd_search_editor_height,
                                        explicit_hover=True, tooltip="Find recorded names, classes or native IDs. Enter: next match; Shift+Enter: previous match.")
                                ui.InvisibleButton(clicked_fn=self._focus_search,
                                    height=(fl.pvd_field_height-fl.pvd_search_editor_height)/2)
                            with ui.HStack(opaque_for_mouse_events=False):
                                ui.Spacer(width=fl.pvd_compact_padding)
                                self.search_hint=ui.Label("Search hierarchy",name="hierarchy_search",
                                    visible=not bool(self.search_model.as_string),opaque_for_mouse_events=False)
                        self.search_suffix=ui.InvisibleButton(clicked_fn=self._focus_search,
                            width=fl.pvd_search_inset-fl.pvd_compact_padding)
                with ui.Frame(width=fl.pvd_control,visible=bool(self.search_model.as_string)) as self.clear_search_host:
                    icon_button("close",clicked_fn=lambda:self.search_model.set_value(""),
                                tooltip="Clear object search")
            with ui.HStack(height=fl.pvd_control, spacing=fl.pvd_spacing):
                self.grouping = ui.ComboBox(0 if self.model.grouping == "scene" else 1,
                                           "Scene hierarchy", "Class groups")
                self.grouping.model.add_item_changed_fn(self._group_changed)
                ui.Button("Collapse",width=64,clicked_fn=self.collapse_all,explicit_hover=True, tooltip="Collapse expanded branches")
                self.options_button=icon_button("dots",name="ghost",width=28,clicked_fn=self.show_menu,
                          tooltip="Find objects (Ctrl+F) and hierarchy filter options")
            with ui.ScrollingFrame(horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF):
                self.tree = ui.TreeView(self.model, delegate=self.delegate, header_visible=True,name="objects",
                    column_widths=[ui.Fraction(1), ui.Pixel(0),
                                   ui.Pixel(max(32,fl.pvd_object_action_width) if self.on_action else 0),
                                   ui.Pixel(max(32,fl.pvd_object_action_width) if self.on_action else 0)],
                    min_column_widths=[ui.Pixel(0)]*4,
                    columns_resizable=False, expand_on_branch_click=False,
                    auto_scroll_selection=True)
                self.tree.set_selection_changed_fn(self._tree_selection_changed)
            with ui.HStack(height=20,spacing=4):
                self.summary=ui.Label("Native scene hierarchy",name="muted",elided_text=True)
                self.previous_button=icon_button("previous",name="ghost",width=24,
                    clicked_fn=lambda:self.navigate(True),tooltip="Previous match (Shift+Enter in search)")
                self.next_button=icon_button("next",name="ghost",width=24,
                    clicked_fn=self.navigate,tooltip="Next match (Enter in search)")
        self.refresh()

    def build_tag(self,item,column):
        if self.on_action is None:return
        identity=self.model.row_identity(item.key)
        if identity is None:return
        hidden,tagged=self.row_tags(identity)
        active=hidden if column==2 else tagged
        icon="visible" if column==2 else "opaque"
        with ui.VStack(width=fl.pvd_object_action_width,height=fl.pvd_object_row):
            ui.Spacer()
            control=icon_button(icon,alternate_icon="hidden" if column==2 else "transparent",
                name="quiet",size=fl.pvd_object_action_icon,
                height=fl.pvd_object_action_width,width=fl.pvd_object_action_width,
                clicked_fn=lambda:self.toggle_tag(identity,column))
            ui.Spacer()
        control.alternate=active
        self._tag_controls[(item.key,column)]=(identity,control,active)
        self._tag_tooltip(control,column,active,identity)

    def _tag_tooltip(self,control,column,active,identity):
        local=identity.kind in ("reference","array","array_page","array_element")
        if column==2:
            text=("Show" if active else "Hide")+(" this row" if local else " this subtree")
        else:
            text=("Clear" if active else "Set")+" transparency tag · Tagged objects mode"
        if local:text+="\nLocal row flag; referenced geometry stays unchanged."
        self._set_row_tooltip((identity.key,column),control,text)

    def _set_row_tooltip(self,key,widget,text):
        # Keep one current widget per row/column, including lazily built rows.
        # Static tooltips must be cleared: OVUI caches dynamic tooltip frames.
        if not hasattr(self,"_row_tooltips"):self._row_tooltips={}
        self._row_tooltips[key]=(widget,text)
        widget.tooltip="" if getattr(self,"_row_tooltips_suppressed",False) else text

    def _suppress_row_tooltips(self,suppressed):
        self._row_tooltips_suppressed=suppressed
        for key,(widget,text) in tuple(getattr(self,"_row_tooltips",{}).items()):
            if self._destroyed or key[0] not in self.model._specs:
                widget.tooltip=""
                self._row_tooltips.pop(key,None)
            else:
                widget.tooltip="" if suppressed else text

    def _prune_row_tooltips(self):
        revision=(self.model.capture_key,self.model.revision)
        if revision==getattr(self,"_tooltip_model_revision",None):return
        self._tooltip_model_revision=revision
        for key,(widget,_text) in tuple(getattr(self,"_row_tooltips",{}).items()):
            if key[0] not in self.model._specs:
                widget.tooltip=""
                self._row_tooltips.pop(key,None)

    def _context_menu_shown(self,generation,shown):
        if not self._destroyed and generation==self._menu_generation:
            self._suppress_row_tooltips(shown)

    def refresh_tags(self,signature=None):
        """Refresh existing controls only when supplied appearance/model state changes.

        None retains the uncached refresh API. Newly built controls already
        read current tags; model revision changes still prune stale controls.
        """
        current=(signature,self.model.revision) if signature is not None else None
        if current is not None and current==getattr(self,"_last_tag_signature",None):return
        self._prune_row_tooltips()
        for key,(identity,control,previous) in tuple(self._tag_controls.items()):
            if not self._current_row(identity):
                self._tag_controls.pop(key,None)
                continue
            hidden,tagged=self.row_tags(identity)
            column=key[1]
            active=hidden if column==2 else tagged
            if active==previous:continue
            control.alternate=active
            self._tag_tooltip(control,column,active,identity)
            self._tag_controls[key]=(identity,control,active)
        self._last_tag_signature=current

    def toggle_tag(self,identity,column):
        if not self._current_row(identity):return
        if self.on_focus:self.on_focus()
        hidden,tagged=self.row_tags(identity)
        action=("show" if hidden else "hide") if column==2 else ("untag" if tagged else "tag")
        scope=self.model.action_scope([identity.key],action="visibility" if column==2 else "transparency")
        self.on_action(action,scope)

    def _current_row(self,identity):
        row=self.model.row_identity(identity.key) if identity is not None else None
        return not self._destroyed and row is not None and row.token==identity.token

    def activate_row(self,key,*,menu_generation=None):
        if menu_generation is not None and menu_generation!=self._menu_generation:return
        identity=self.model.row_identity(key)
        if identity is None or identity.kind not in ("object","reference","array_element"):return
        if identity.instance_id is None and identity.query_id is None:return
        def activate():
            if menu_generation is not None and menu_generation!=self._menu_generation:return
            if not self._current_row(identity):return
            target=(self.model.canonical_item(identity.instance_id) if identity.kind=="reference"
                    else self.model._item(identity.key))
            if target is None:return
            self.select_item(target)
            # Programmatic TreeView selection notifies on its next draw.
            self.on_selection_changed([target])
            if self.on_activate and menu_generation is None:self.on_activate()
        asyncio.get_event_loop().call_soon(activate)

    def row_pressed(self,key,x,y,button,modifiers):
        if button in (0,1) and self.on_focus:self.on_focus()
        if button!=1 or self.on_action is None:return
        identity=self.model.row_identity(key)
        # Selection callbacks and row refreshes must run after TreeView draw.
        asyncio.get_event_loop().call_soon(self.show_context_menu,identity,x,y,modifiers)

    def show_context_menu(self,identity,x,y,modifiers=0):
        if not self._current_row(identity):return
        selected=[item for item in self.tree.selection if item is not None]
        if not modifiers&3 and identity.key not in {item.key for item in selected}:
            item=self.model.restore_rows([identity])[0]
            self.tree.selection=[item]
            selected=[item]
        scope=self.model.action_scope([item.key for item in selected])
        # All actions are also capture-scoped: an old popup cannot modify a
        # replacement recording that happens to reuse the same native IDs.
        all_scope=self.model.action_scope(self.model._root_keys)
        self.close_context_menu()
        generation=self._menu_generation
        self.context_menu=ui.Menu("Object visibility",tearable=False)
        self.context_menu.set_shown_changed_fn(lambda shown:self._context_menu_shown(generation,shown))
        def apply(action,target):
            if not self._destroyed and generation==self._menu_generation:
                self.on_action(action,target)
                if self.on_focus:self.on_focus()
        with self.context_menu:
            ui.MenuItem("Hide selected",enabled=bool(selected),triggered_fn=lambda:apply("hide",scope))
            ui.MenuItem("Show selected",enabled=bool(selected),triggered_fn=lambda:apply("show",scope))
            ui.Separator()
            ui.MenuItem("Hide all",triggered_fn=lambda:apply("hide_all",all_scope))
            ui.MenuItem("Show all",triggered_fn=lambda:apply("show_all",all_scope))
            if identity.kind=="reference":
                ui.Separator()
                ui.MenuItem("Go to referenced object",triggered_fn=lambda:
                    self.activate_row(identity.key,menu_generation=generation) if self._current_row(identity) else None)
        # Popup hover in public OVUI still reaches the widgets underneath it.
        # Suppress before the first popup draw; the shown callback restores on
        # outside-click/Escape dismissal without rebuilding the tree.
        self._suppress_row_tooltips(True)
        self.context_menu.show_at(float(x),float(y))

    def close_context_menu(self):
        self._menu_generation+=1
        if self.context_menu is not None:
            self.context_menu.destroy()
            self.context_menu=None
        self._suppress_row_tooltips(False)

    def _tree_selection_changed(self, items):
        if not self._changing_group:
            self.on_selection_changed(items)

    def _group_changed(self, model, _item):
        grouping="scene" if model.get_item_value_model().as_int == 0 else "class"
        if grouping==self.model.grouping:return
        identities=self.model.row_identities(self.tree.selection)
        self._changing_group=True
        try:
            self.model.set_grouping(grouping)
            restored=self.model.restore_rows(identities)
            self.restore_projection_selection(restored)
        finally:
            self._changing_group=False
        self.on_selection_changed(list(restored))

    def restore_projection_selection(self, restored):
        """Reattach a changed hierarchy and reveal only selected ancestor paths."""
        was_changing=self._changing_group
        self._changing_group=True
        try:
            ancestors={}
            for item in restored:
                for ancestor in self.model.reveal_ancestors(item):
                    ancestors.setdefault(ancestor.key,ancestor)
            # Native TreeView removal can erase cache entries for stable items
            # that have moved from an old descendant to a new root. Empty its
            # nodes through the public model API before attaching the projection.
            # No intermediate model is drawn or published to the application.
            try:
                self.tree.model=ui.SimpleListModel([])
                self.tree.set_expanded(None,True,False)
            finally:
                self.tree.model=self.model
            # set_expanded creates immediate children synchronously.
            # Then walk only the selected paths, so even previously collapsed
            # class groups have native nodes before selection is restored.
            self.tree.set_expanded(None,True,False)
            for ancestor in ancestors.values():
                self.tree.set_expanded(ancestor,True,False)
            self._last_search_signature = None
            self.refresh()
            self.tree.selection=list(restored)
        finally:
            self._changing_group=was_changing

    def _search_edited(self, model):
        if not self._destroyed and model.as_string!=self.search_model.as_string:
            self.search_model.set_value(model.as_string)

    def _focus_search(self):
        # Public click delivery occurs on release. Requesting keyboard focus
        # from an overlapping background's mouse press is lost by InputText.
        if not self._destroyed:self.search_field.focus_keyboard()

    def _begin_search_edit(self, _model):
        self._search_editing=not self._destroyed
        if (background:=getattr(self,"search_background",None)) is not None:
            background.selected=self._search_editing

    def _end_search_edit(self, _model):
        self._search_editing=False
        if (background:=getattr(self,"search_background",None)) is not None:
            background.selected=False
        # Native InputText ends editing after Enter. Restore it on the following
        # draw so repeated navigation needs no click and retains the expression.
        restore,self._search_refocus=self._search_refocus,False
        if restore and not self._destroyed and self.search_window_focused():
            self.search_field.focus_keyboard()

    def search_key(self,key,modifiers,pressed):
        """Consume exact Enter/Shift+Enter from the focused search expression.

        A field key callback alone is hover-gated in public OVUI. The host's
        focused-window dispatcher supplies delivery even after the pointer moves.
        """
        key=normalize_key(key)
        chord=modifiers & 15
        if (not pressed or key not in (257,335) or chord not in (0,1)
                or self._destroyed or not self._search_editing
                or not self.search_window_focused()):return False
        self.navigate(backwards=chord==1)
        self._search_refocus=True
        return True

    def _search_changed(self, _model=None):
        if self._destroyed:return
        if self.search_editor_model.as_string!=self.search_model.as_string:
            self.search_editor_model.set_value(self.search_model.as_string)
        self.search_hint.visible=not bool(self.search_model.as_string)
        self.clear_search_host.visible=bool(self.search_model.as_string)
        self.model.set_filter(self.search_model.as_string, match_case=self.match_case.as_bool,
                              whole_word=self.whole_word.as_bool, fields=_SEARCH_FIELDS[self.fields_index])
        self._last_search_signature = None
        self.refresh()
        if self._matches:
            self.select_item(self._matches[0])

    def refresh(self):
        if self.tree is None:
            return
        signature = (self.model.revision, self.model.filter, self.fields_index,
                     self.match_case.as_bool, self.whole_word.as_bool)
        if signature == self._last_search_signature:
            return
        self._matches = self.model.find_matches() if self.model.filter else []
        self._last_search_signature = signature
        count = self.model.object_count
        self.summary.text = (f"{len(self._matches):,} matches · {count:,} objects" if self.model.filter
                             else f"{count:,} objects · native ownership")
        self.previous_button.enabled = self.next_button.enabled = bool(self._matches)
        if self._last_projection is not self.model._specs:
            self._last_projection = self.model._specs
            for item in self.model.expanded_items():
                self.tree.set_expanded(item, True, False)

    def select_item(self, item):
        if item.instance_id is not None:
            self.ensure_native_visible([item.instance_id])
        self.model.reveal_ancestors(item, lambda ancestor: self.tree.set_expanded(ancestor, True, False))
        self.tree.selection = [item]

    def ensure_native_visible(self, identities):
        """Explicit inspection of compacted shape data opens the native class view."""
        if not any(f'object:{identity}' in self.model.hidden_keys for identity in identities):
            return
        was_changing = self._changing_group
        self._changing_group = True
        try:
            self.model.ensure_native_visible(identities)
            self.grouping.model.get_item_value_model().set_value(1)
            self.restore_projection_selection([])
        finally:
            self._changing_group = was_changing

    def navigate(self, backwards=False):
        current = self.tree.selection[-1] if self.tree.selection else None
        item = self.model.previous_match(current) if backwards else self.model.next_match(current)
        if item is not None:
            self.select_item(item)

    def toggle_expanded(self, item):
        expanded = not self.tree.is_expanded(item)
        self.model.remember_expanded(item, expanded)
        self.tree.set_expanded(item, expanded, False)

    def collapse_all(self):
        # Recursive TreeView collapse enumerates descendants, including unopened
        # query pages. Only materialized rows can hold native expansion state.
        for item in tuple(dict.values(self.model.cache)):
            if self.tree.is_expanded(item):
                self.tree.set_expanded(item, False, False)
        self.model.expanded_keys.clear()

    def show_menu(self):
        self.menu=ui.Menu("Object search",tearable=False)
        with self.menu:
            ui.MenuItem("Find objects (Ctrl+F)",triggered_fn=self.show_find)
            ui.MenuItem("Filter options",triggered_fn=self.show_filter_options)
        button=self.options_button.button
        self.menu.show_at(button.screen_position_x,button.screen_position_y+button.computed_height)

    def show_filter_options(self):
        if self.filter_window is None:
            self.filter_window = ui.Window("Filter options", width=fl.pvd_find_width,
                height=fl.pvd_find_height,
                flags=ui.WINDOW_FLAGS_NO_RESIZE | ui.WINDOW_FLAGS_NO_SCROLLBAR)
            style_dialog(self.filter_window)
            with self.filter_window.frame:
                with ui.ZStack():
                    ui.Rectangle(name="dialog")
                    with ui.HStack():
                        ui.Spacer(width=fl.pvd_padding)
                        with ui.VStack(spacing=fl.pvd_spacing):
                            ui.Spacer(height=fl.pvd_spacing)
                            ui.Label("Filter options", name="section", height=fl.pvd_control)
                            ui.StringField(self.search_model, height=fl.pvd_control,
                                explicit_hover=True, tooltip="Names, classes and native IDs support multiple search terms")
                            with ui.HStack(height=fl.pvd_control, spacing=fl.pvd_spacing):
                                ui.Label("Search in", width=fl.pvd_button_width)
                                fields = ui.ComboBox(self.fields_index, "Name, class and ID", "Recorded name", "Class name", "Native ID")
                                fields.model.add_item_changed_fn(self._fields_changed)
                            with ui.HStack(height=fl.pvd_control, spacing=fl.pvd_spacing):
                                ui.CheckBox(self.match_case, width=fl.pvd_object_branch)
                                ui.Label("Match case")
                                ui.CheckBox(self.whole_word, width=fl.pvd_object_branch)
                                ui.Label("Whole words")
                            ui.Label("Terms are combined. Matching objects retain their parent context.",
                                     name="muted", height=fl.pvd_control, word_wrap=True)
                            with ui.HStack(height=fl.pvd_control, spacing=fl.pvd_spacing):
                                ui.Spacer()
                                ui.Button("Close", width=fl.pvd_button_width, clicked_fn=self._close_filter)
                                ui.Button("Apply", name="primary", width=fl.pvd_button_width,
                                          clicked_fn=self._search_changed)
                            ui.Spacer(height=fl.pvd_spacing)
                        ui.Spacer(width=fl.pvd_padding)
        self.filter_window.visible = True

    def _fields_changed(self, model, _item):
        self.fields_index = model.get_item_value_model().as_int

    def _close_filter(self):
        self.filter_window.visible = False

    def show_find(self):
        if self.find_dialog is None:
            self.find_dialog=FindObjectsDialog(self)
            self.find_window=self.find_dialog.window
        self.find_dialog.show()

    def activate_found(self,identity):
        if self.on_find_object:
            return self.on_find_object(identity)
        item=self.model.canonical_item(identity)
        if item is not None:
            self.select_item(item)
            if self.on_activate:self.on_activate()
        else:
            adapter=self.model._adapter
            item=ObjectItem(f"object:{identity}",adapter.name(identity),adapter.class_name(identity),identity)
            self.on_selection_changed([item])

    def destroy(self):
        self._destroyed=True
        self._search_editing=self._search_refocus=False
        self.close_context_menu()
        self._tag_controls.clear()
        if self.find_dialog is not None:
            self.find_dialog.destroy()
            self.find_dialog = None
            self.find_window = None
        if self.filter_window is not None:self.filter_window.destroy()
        if self.menu is not None:self.menu.destroy()
        self.frame.destroy()

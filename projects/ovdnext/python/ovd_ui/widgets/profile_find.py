# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Themed, single-result Find dialog over its profiler's owned snapshot."""
from itertools import count
import omni.ui as ui

from ..camera import normalize_key
from ..profile_find import CRITERIA, find_profile_events, profile_find_sort_key, unsigned_decimal
from ..style import layout as fl, style_dialog

_DIALOG_IDS = count(1)


class ProfileFindItem(ui.AbstractItem):
    def __init__(self, result):
        super().__init__()
        self.result = result
        self.values = [ui.SimpleStringModel(value) for value in result.cells]


class ProfileFindModel(ui.AbstractItemModel):
    def __init__(self):
        super().__init__()
        self.items = []
        self.headers = [ui.SimpleStringModel(name) for name in ("Name", "Context", "Thread", "ID")]
        self.column, self.reverse = 0, True  # QTreeView's initial sort

    def set_rows(self, rows):
        self.items = [ProfileFindItem(row) for row in rows]
        self.sort()

    def sort(self, column=None):
        if column is not None:
            self.reverse = not self.reverse if column == self.column else False
            self.column = column
        self.items.sort(key=lambda item: profile_find_sort_key(item.result, self.column), reverse=self.reverse)
        self._item_changed(None)

    def get_item_children(self, item=None): return self.items if item is None else []
    def can_item_have_children(self, item=None): return item is None and bool(self.items)
    def get_item_value_model_count(self, item=None): return 4
    def get_item_value_model(self, item, column_id=0):
        return item.values[column_id] if item else self.headers[column_id]


class ProfileFindDelegate(ui.AbstractItemDelegate):
    def __init__(self, dialog):
        super().__init__()
        self.dialog = dialog

    def build_branch(self, *_): pass

    def build_header(self, column_id):
        name = self.dialog.model.headers[column_id].as_string
        ui.Button(name, name="ghost", height=fl.pvd_control,
            clicked_fn=lambda: self.dialog.sort(column_id), explicit_hover=True, tooltip=f"Sort by {name}; click again to reverse")

    def build_widget(self, model, item, column_id, level, expanded):
        if item is None: return
        result, key = item.result, item.result.key
        context=(result.context_name+'\n') if result.context_name else ''
        ui.Label(item.values[column_id].as_string, name="mono" if column_id else "",
            height=fl.pvd_row, elided_text=True, explicit_hover=True,
            tooltip=f"{result.name}\n{context}Zone {key.zone_id} · Context {key.context_id}\nThread {key.thread_id} · Event {key.event_id}",
            mouse_double_clicked_fn=lambda x, y, button, mods: self.dialog.activate(item) if button == 0 else None)


class FindProfileDialog:
    def __init__(self, view):
        self.view = view
        self.model = ProfileFindModel()
        self.delegate = ProfileFindDelegate(self)
        self.expression = ui.SimpleStringModel("")
        self.mode, self.focus, self.snapshot = 0, "expression", None
        self._selecting = False
        self.window = ui.Window(f"Find profile events##pvd-profile-find-{next(_DIALOG_IDS)}", width=fl.pvd_find_width + 160,
            height=fl.pvd_find_height + fl.pvd_row * 7,
            flags=ui.WINDOW_FLAGS_NO_DOCKING | ui.WINDOW_FLAGS_NO_TITLE_BAR | ui.WINDOW_FLAGS_MODAL)
        style_dialog(self.window)
        self.window.set_key_pressed_fn(self.key)
        with self.window.frame:
            with ui.ZStack():
                ui.Rectangle(name="dialog")
                with ui.HStack():
                    ui.Spacer(width=fl.pvd_padding)
                    with ui.VStack(spacing=fl.pvd_spacing):
                        ui.Spacer(height=fl.pvd_padding)
                        ui.Label("Find profile events", name="title", height=fl.pvd_control)
                        self.scope_label = ui.Label("", name="muted", height=fl.pvd_row, elided_text=True, explicit_hover=True)
                        with ui.HStack(height=fl.pvd_control, spacing=fl.pvd_spacing):
                            self.field = ui.StringField(self.expression,
                                explicit_hover=True, tooltip="Names are case-sensitive. IDs accept unsigned 64-bit decimal integers.")
                            self.criteria = ui.ComboBox(0, *CRITERIA, width=fl.pvd_button_width + 30)
                            self.criteria.model.add_item_changed_fn(self._mode_changed)
                            ui.Button("Find", name="primary", width=fl.pvd_button_width, clicked_fn=self.search)
                        self.status = ui.Label("", name="muted", height=fl.pvd_row, elided_text=True)
                        with ui.ScrollingFrame(horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF):
                            self.tree = ui.TreeView(self.model, delegate=self.delegate, header_visible=True,
                                root_visible=True, columns_resizable=True,
                                column_widths=[ui.Fraction(1), ui.Pixel(155), ui.Pixel(110), ui.Pixel(55)],
                                auto_scroll_selection=True)
                            self.tree.set_selection_changed_fn(self._selection_changed)
                        ui.Label("Selecting clears the inline filter. Hidden events stay hidden.", name="muted", height=fl.pvd_row)
                        with ui.HStack(height=fl.pvd_control, spacing=fl.pvd_spacing):
                            ui.Spacer()
                            self.select_button = ui.Button("Select", name="primary", width=fl.pvd_button_width,
                                enabled=False, clicked_fn=self.activate)
                            ui.Button("Cancel", width=fl.pvd_button_width, clicked_fn=self.close)
                        ui.Spacer(height=fl.pvd_padding)
                    ui.Spacer(width=fl.pvd_padding)
        self.expression.add_begin_edit_fn(lambda _: self._focus_expression())

    def _focus_expression(self):
        self.focus = "expression"
        self.window.flags = getattr(self.window, "flags", 0) & ~(1 << 16)

    def _focus_results(self):
        self.focus = "results"
        # Same public TreeView logical keyboard ownership as Find objects.
        self.window.flags = getattr(self.window, "flags", 0) | (1 << 16)

    def show(self):
        if not self.view.can_find: return False
        self.expression.set_value("")
        self.criteria.model.get_item_value_model().set_value(0)
        self.mode = 0
        self.model.column, self.model.reverse = 0, True
        self.clear()
        self.scope_label.text = (f"Loaded profile result · Frame {self.view.model.snapshot.current_frame} · "
                                 f"{self.view.source_label}")
        self.scope_label.tooltip = self.scope_label.text
        self.status.text = "Search this result's recorded events, including hidden events."
        self._focus_expression()
        self.window.visible = True
        self.window.focus()
        self.field.focus_keyboard()
        return True

    def clear(self):
        self.tree.clear_selection()
        self.model.set_rows([])
        self.select_button.enabled = False
        self.snapshot = None

    def invalidate(self):
        self.clear()
        self.window.visible = False

    def _mode_changed(self, model, _item):
        self.mode = model.get_item_value_model().as_int
        self.clear()
        self.status.text = f"Press Find to search by {CRITERIA[self.mode].lower()}."
        self._focus_expression()

    def search(self):
        self.clear()
        if not self.view.can_find:
            self.status.text = "The profile result changed. Reopen Find after loading finishes."
            return
        self.snapshot = self.view.model.snapshot
        self.model.set_rows(find_profile_events(self.snapshot, self.expression.as_string, self.mode))
        count = len(self.model.items)
        self.select_button.enabled = count == 1
        self.status.text = f"{count:,} result{'s' if count != 1 else ''}"
        if self.mode and unsigned_decimal(self.expression.as_string) is None:
            self.status.text = "Enter a decimal ID between 0 and 18446744073709551615."
        self._focus_results()
        self.field.focus_keyboard(False)

    def _selection_changed(self, items):
        if self._selecting: return
        self._selecting = True
        try:
            if len(items) > 1: self.tree.selection = [items[-1]]
        finally: self._selecting = False
        self.select_button.enabled = bool(items) or len(self.model.items) == 1
        if items: self._focus_results()

    def sort(self, column):
        self.model.sort(column)
        self._focus_results()

    def activate(self, item=None):
        item = item or (self.tree.selection[-1] if self.tree.selection else
                        self.model.items[0] if len(self.model.items) == 1 else None)
        if item is None: return False
        if item not in self.model.items or not self.view.activate_found(self.snapshot, item.result.key):
            self.status.text = "The profile result changed. Search again before selecting."
            return False
        self.close()
        return True

    def key(self, key, modifiers, pressed):
        if not pressed or not self.window.focused: return
        key = normalize_key(key)
        if key == 256: self.close()
        elif key == 258 and self.focus == "results":
            self._focus_expression()
            self.field.focus_keyboard()
        elif key in (257, 335):
            if self.focus == "expression": self.search()
            else: self.activate()
        elif key in (264, 265) and self.focus == "results" and self.model.items:
            items = self.model.items
            selected = self.tree.selection[-1] if self.tree.selection else None
            index = items.index(selected) if selected in items else (-1 if key == 264 else len(items))
            self.tree.selection = [items[max(0, min(len(items) - 1, index + (1 if key == 264 else -1)))]]

    def close(self):
        self.window.visible = False
        self.clear()
        if self.view.window.visible: self.view.window.focus()

    def destroy(self):
        self.snapshot = None
        self.window.destroy()

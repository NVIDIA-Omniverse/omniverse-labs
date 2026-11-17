# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Nonmodal recorder scope controls, independent of local object selection."""
from contextlib import contextmanager
import omni.ui as ui

from ..style import layout as fl, style_dialog
from .segmented import SegmentedControl


@contextmanager
def _surface(window):
    with window.frame:
        with ui.ZStack():
            ui.Rectangle(name="dialog")
            with ui.HStack():
                ui.Spacer(width=fl.pvd_padding)
                with ui.VStack():
                    ui.Spacer(height=fl.pvd_padding)
                    yield
                    ui.Spacer(height=fl.pvd_padding)
                ui.Spacer(width=fl.pvd_padding)


class _Row(ui.AbstractItem):
    def __init__(self, entry):
        super().__init__()
        self.entry = entry
        self.values = [ui.SimpleStringModel(text) for text in
            ("", entry["name"] or "(unnamed)", entry["type_name"], str(entry["target_handle"]))]


class _Rows(ui.AbstractItemModel):
    def __init__(self):
        super().__init__(); self.items = []

    def update(self, entries):
        self.items = [_Row(entry) for entry in entries]  # One native page, at most128.
        self._item_changed(None)

    def get_item_children(self, item=None): return self.items if item is None else []
    def get_item_value_model_count(self, item=None): return 4
    def get_item_value_model(self, item, column_id=0): return item.values[column_id]
    def can_item_have_children(self, item=None): return item is None


class _Delegate(ui.AbstractItemDelegate):
    def __init__(self, panel):
        super().__init__(); self.panel = panel

    def build_branch(self, *_): pass

    def build_header(self, column_id):
        ui.Label(("", "Name", "Family", "Handle")[column_id], name="muted", height=fl.pvd_row)

    def build_widget(self, model, item, column_id, level, expanded):
        if item is None: return
        row, panel = item.entry, self.panel
        target = row["target_handle"]
        if column_id == 0:
            check = ui.CheckBox(width=22, height=fl.pvd_row, enabled=row["can_select"] and not panel.model.all,
                tooltip="Record this target" if row["can_select"] else "Context only; select an owning supported object")
            check.model.set_value(panel.model.is_selected(target))
            origin = (panel.model.session, panel.model.generation)
            check.model.add_value_changed_fn(lambda value: panel.choose(target, value.as_bool, origin=origin))
            panel._checks[target] = check
        elif column_id == 1:
            ui.Button(item.values[1].as_string, name="quiet", height=fl.pvd_row,
                tooltip="Browse catalog children; does not select or send a request",
                clicked_fn=lambda: panel.page(0, target))
        else:
            ui.Label(item.values[column_id].as_string, height=fl.pvd_row, elided_text=True,
                explicit_hover=True, tooltip=item.values[column_id].as_string)


class CaptureSelectionPanel:
    def __init__(self, model):
        self.model = model
        self.window = None
        self.message = ""
        self.supersede = self._closed = self._syncing = False
        self._identity = self._rows_key = None
        self._checks = {}

    @property
    def can_apply(self):
        return (not self._closed and self.model.info.get("can_apply", False)
                and (self.model.all or self.model.selected_count > 0)
                and (self.model.info.get("state") != "unknown" or self.supersede))

    @property
    def can_refresh(self):
        return (not self._closed and self.model.info.get("can_refresh", False)
                and (self.model.info.get("state") != "unknown" or self.supersede))

    def _action(self, action):
        if self._closed: return False
        try:
            action()
            self.message = ""
        except (ValueError, RuntimeError, OSError) as error:
            self.message = str(error)
            self.refresh()
            return False
        self.refresh()
        return True

    def choose(self, target, checked, *, origin=None):
        def select():
            self.model.poll()
            if origin is not None and origin != (self.model.session, self.model.generation):
                raise ValueError("Catalog changed; choose a row from the current generation")
            self.model.select(target, checked)
        return self._action(select)
    def apply(self): return self._action(lambda: self.model.apply(self.supersede))
    def refresh_catalog(self): return self._action(lambda: self.model.refresh_catalog(self.supersede))

    def set_scope(self, all_permitted):
        if not self._syncing:
            self.model.all = bool(all_permitted)
            self.refresh()

    def set_secondary(self, enabled):
        if not self._syncing:
            self.model.secondary = bool(enabled)
            self.refresh()

    def page(self, offset=0, parent=None):
        return self._action(lambda: self.model.load_page(max(0, offset),
            self.model.parent if parent is None else parent, self.prefix.model.as_string))

    def show(self):
        if self._closed: return
        if self.window is None: self._build()
        self.window.visible = True
        self.window.focus()
        self.refresh()

    def hide(self):
        if self.window is not None: self.window.visible = False

    def close(self):
        self._closed = True
        if self.window is not None: self.window.destroy()
        self.window = None
        self._checks.clear()

    def refresh(self):
        if self._closed: return
        self.model.poll()
        info = self.model.info
        identity = (self.model.session, info.get("request_id"), info.get("state"))
        if identity != self._identity:
            self.supersede = False
            if self._identity is not None and identity[0] is not self._identity[0]: self.message = ""
            self._identity = identity
        if self.window is None: return
        self._syncing = True
        try:
            self.scope.model.set_value(int(self.model.all))
            self.policy.model.set_value(int(self.model.secondary))
            self.supersede_check.model.set_value(self.supersede)
            self.supersede_row.visible = info.get("state") == "unknown"
            self.apply_button.enabled, self.refresh_button.enabled = self.can_apply, self.can_refresh
            self.scope.buttons[0].enabled = bool(info.get("capabilities", 0) & 8)
            self.status.text = self.message or self.model.status
            self.identity.text = (f"Live producer · epoch {info.get('capture_epoch', 0)} · "
                f"catalog {self.model.generation} · completed segment {info.get('segment_transition_serial', 0)}")
            actual = ("all permitted" if info.get("actual_all") else "producer selection")
            self.actual.text = (f"Actual: {actual}, " + ("primary + secondary" if info.get("actual_secondary") else "primary")
                if info.get("policy_known") else "Actual: no recording policy published")
            used, limits = info.get("used", {}), info.get("limits", {})
            maximum = max((used.get(key, 0) / value for key, value in limits.items() if value), default=0)
            self.budget.text = f"Most-used recording budget: {maximum:.1%} · Apply reserves space for another segment"
            page = self.model.page
            self.count.text = (f"{self.model.selected_count:,} selected · {page['total']:,} matching rows · "
                f"page {self.model.offset // 128 + 1}" + (" · context children" if self.model.parent else ""))
            self.previous.enabled = self.model.offset > 0
            self.next.enabled = self.model.offset + 128 < page["total"]
            key = (id(self.model.session), id(page), self.model.all,
                   tuple(self.model.is_selected(row["target_handle"]) for row in page["entries"]))
            if key != self._rows_key:
                self._checks.clear()
                self.rows.update(page["entries"])
                self.tree.dirty_widgets()
                self._rows_key = key
        finally:
            self._syncing = False

    def _supersede(self, model):
        if not self._syncing:
            self.supersede = model.as_bool
            self.refresh()

    def _build(self):
        self.window = ui.Window("Recording selection", width=780, height=640,
            flags=ui.WINDOW_FLAGS_NO_DOCKING | ui.WINDOW_FLAGS_NO_TITLE_BAR)
        style_dialog(self.window)
        with _surface(self.window):
            with ui.VStack(spacing=fl.pvd_spacing):
                with ui.HStack(height=fl.pvd_control):
                    ui.Label("Recording selection", name="title")
                    ui.Button("Close", width=80, clicked_fn=self.hide)
                self.identity = ui.Label("", height=24, name="muted")
                ui.Label("Targets the live producer, even while an older capture is displayed.", height=24)
                with ui.HStack(height=fl.pvd_control, spacing=8):
                    self.scope = SegmentedControl(((0, "Selected", 100), (1, "All permitted", 128)),
                        on_changed=self.set_scope, height=fl.pvd_control)
                    self.policy = SegmentedControl(((0, "Primary", 100), (1, "+ Secondary", 128)),
                        on_changed=self.set_secondary, height=fl.pvd_control)
                self.actual = ui.Label("", height=24, name="muted")
                with ui.HStack(height=fl.pvd_control, spacing=8):
                    ui.Label("Name prefix", width=85)
                    self.prefix = ui.StringField()
                    ui.Button("Find", width=64, clicked_fn=lambda: self.page())
                    ui.Button("All rows", width=80, clicked_fn=lambda: self.page(0, 0))
                    self.refresh_button = ui.Button("Refresh catalog", width=132, clicked_fn=self.refresh_catalog)
                self.rows, self.delegate = _Rows(), _Delegate(self)
                with ui.ScrollingFrame():
                    self.tree = ui.TreeView(self.rows, delegate=self.delegate, root_visible=False,
                        header_visible=True, column_widths=[ui.Pixel(28), ui.Fraction(1), ui.Pixel(230), ui.Pixel(150)])
                with ui.HStack(height=fl.pvd_control, spacing=8):
                    self.count = ui.Label("")
                    self.previous = ui.Button("Previous", width=85, clicked_fn=lambda: self.page(self.model.offset - 128))
                    self.next = ui.Button("Next", width=65, clicked_fn=lambda: self.page(self.model.offset + 128))
                self.budget = ui.Label("", height=24, name="muted")
                self.status = ui.Label("", height=48, word_wrap=True)
                with ui.HStack(height=fl.pvd_control, spacing=8) as self.supersede_row:
                    self.supersede_check = ui.CheckBox(width=22)
                    self.supersede_check.model.add_value_changed_fn(self._supersede)
                    ui.Label("Supersede unknown request; the earlier request may still execute")
                with ui.HStack(height=fl.pvd_control, spacing=8):
                    ui.Label("Apply restarts recording. The producer may narrow or deny the request.", name="muted")
                    self.apply_button = ui.Button("Apply", name="primary", width=90, clicked_fn=self.apply)

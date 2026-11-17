# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native diagnostic tables with asynchronous, private-session refresh.

MemoryView(job, on_refresh=None, surface=None) and ErrorStreamView(...) own a
floating .window (.widget alias). Call poll() on the UI thread; it returns True
once when a completed job is adopted. set_job(job) cancels the old request and
preserves the last useful table while loading. on_refresh() may return a new
AnalysisJob, or call set_job itself. destroy() cancels work and closes dialogs.
The caller owns its DiagnosticsTasks/ErrorTasks and closes that service later.

Memory callers should explicitly choose DiagnosticsRequest(max_events=1_000_000)
for dense Apex captures. Snapshots are complete or rejected on budget overflow;
errors expose a clearly labelled bounded prefix with the exact original count.
CSV exports the visible filtered/sorted table, including complete source paths.
"""
from contextlib import contextmanager
from concurrent.futures import CancelledError
from pathlib import Path

import omni.ui as ui

from .. import style
from ..style import layout as fl
from ..error_stream import ErrorStreamModel
from ..profiling import MemoryModel
from .file_browser import CaptureFileDialog


@contextmanager
def _surface(window, title):
    style.style_dialog(window)
    window.flags |= ui.WINDOW_FLAGS_NO_TITLE_BAR
    with window.frame:
        with ui.ZStack():
            ui.Rectangle(name="dialog")
            with ui.VStack(spacing=fl.pvd_spacing):
                ui.Spacer(height=fl.pvd_spacing)
                with ui.HStack():
                    ui.Spacer(width=fl.pvd_padding)
                    with ui.VStack(spacing=fl.pvd_spacing):
                        with ui.HStack(height=fl.pvd_control):
                            ui.Label(title, name="title")
                            ui.Button("Close", width=90, name="quiet", clicked_fn=lambda: setattr(window, "visible", False))
                        yield
                    ui.Spacer(width=fl.pvd_padding)
                ui.Spacer(height=fl.pvd_spacing)


def _basename(value):
    return value.replace("\\", "/").rsplit("/", 1)[-1]


class _Item(ui.AbstractItem):
    def __init__(self, key):
        super().__init__()
        self.key = key
        self.values = []
        self.tooltips = []

    def update(self, values, tooltips):
        if len(self.values) != len(values):
            self.values = [ui.SimpleStringModel(value) for value in values]
        else:
            for model, value in zip(self.values, values):
                if model.as_string != value:
                    model.set_value(value)
        self.tooltips = tooltips


class _Table(ui.AbstractItemModel):
    def __init__(self):
        super().__init__()
        self.items, self.cache = [], {}
        self.columns = 0

    def set_rows(self, rows, columns):
        self.columns = columns
        self.items = []
        for key, values, tooltips in rows:
            item = self.cache.get(key)
            if item is None:
                item = self.cache[key] = _Item(key)
            item.update(values, tooltips)
            self.items.append(item)
        self._item_changed(None)

    def reset(self):
        self.items, self.cache = [], {}
        self._item_changed(None)

    def get_item_children(self, item=None):
        return self.items if item is None else []

    def can_item_have_children(self, item=None):
        return item is None and bool(self.items)

    def get_item_value_model_count(self, item=None):
        return self.columns

    def get_item_value_model(self, item, column_id=0):
        return item.values[column_id] if item else None


class _Delegate(ui.AbstractItemDelegate):
    def build_branch(self, *args):
        pass

    def build_widget(self, model, item, column_id, level, expanded):
        if item and column_id < len(item.values):
            ui.Label(item.values[column_id].as_string, height=fl.pvd_row,
                     explicit_hover=True, tooltip=item.tooltips[column_id], elided_text=True)


class _DiagnosticView:
    title = "Diagnostics"
    export_name = "diagnostics.csv"

    def __init__(self, job, on_refresh=None, surface=None, *, host=None):
        self.job, self.on_refresh = job, on_refresh
        self.finished, self.destroyed = False, False
        self.data = self.adapter = None
        self.data_request = None
        self.text = ""
        self.search_editing = False
        self.sort, self.reverse = self.default_sort
        self.export_dialog = None
        self.table = _Table()
        self.table.columns = len(self.columns())
        self.delegate = _Delegate()
        self.window = host or ui.Window(self.title, width=1260, height=720)
        self.widget = self.window
        with (surface or _surface)(self.window, self.title):
            self._build_options()
            with ui.HStack(height=fl.pvd_control, spacing=fl.pvd_spacing):
                ui.Label("Search", width=56, name="muted")
                self.search = ui.StringField(explicit_hover=True, tooltip="Filter this table; exported CSV uses the same filter and numeric sort")
                self.search.model.add_value_changed_fn(self._search)
                self.search.model.add_begin_edit_fn(lambda _:setattr(self,"search_editing",True))
                self.search.model.add_end_edit_fn(lambda _:setattr(self,"search_editing",False))
                ui.Button("Clear", width=70, clicked_fn=lambda: self.search.model.set_value(""))
            with ui.HStack(height=fl.pvd_control, spacing=fl.pvd_spacing):
                self.refresh_button = ui.Button("Refresh frame", width=125, enabled=on_refresh is not None, clicked_fn=self.refresh)
                self.cancel_button = ui.Button("Cancel", width=80, clicked_fn=lambda: self.job.cancel())
                self.export_button = ui.Button("Export CSV", width=115, enabled=False, clicked_fn=self.export)
            self.summary = ui.Label("Loading recorded measurements…", height=50, word_wrap=True)
            with ui.ScrollingFrame(vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF):
                with ui.VStack(width=1120 if host else ui.Fraction(1), spacing=0):
                    self.header = ui.Frame(height=fl.pvd_control, build_fn=self._build_headers)
                    with ui.ScrollingFrame(horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF):
                        self.tree = ui.TreeView(self.table, delegate=self.delegate, header_visible=False,
                                                column_widths=self.widths(), columns_resizable=False)
            self.count = ui.Label("", height=22, name="muted")
            self.notice = ui.Label("", height=44, word_wrap=True, name="warning")

    def _build_options(self):
        pass

    def get_workspace_state(self):
        return {"text":self.text, "sort":self.sort, "reverse":self.reverse}

    def restore_workspace_state(self,state):
        self.sort,self.reverse=state["sort"],state["reverse"]
        self.header.rebuild()
        self.search.model.set_value(state["text"])

    def get_inspector_state(self):
        return {"kind":"diagnostic", "title":self.title, "visible":self.window.visible,
                "finished":self.finished, "frame":self.data.current_frame if self.data is not None else None,
                "rows":len(self.table.items), "filter":self.text,
                "sort":self.sort, "descending":self.reverse}

    def _build_headers(self):
        with ui.HStack(spacing=0):
            for (name, key), width in zip(self.columns(), self.widths()):
                arrow = " (desc)" if self.reverse else " (asc)"
                ui.Button(name + (arrow if self.sort == key else ""), width=width,
                          explicit_hover=True, tooltip=f"Sort numerically by {name}" if key not in ("file", "type", "message") else f"Sort by {name}",
                          clicked_fn=lambda k=key: self._sort(k))

    def _sort(self, key):
        self.reverse = not self.reverse if key == self.sort else key not in ("file", "type", "message", "index")
        self.sort = key
        self.header.rebuild()
        self._filter()

    def _search(self, model):
        self.text = model.as_string
        self._filter()

    def set_job(self, job):
        if self.destroyed:
            job.cancel()
            return
        if self.job is not job:
            self.job.cancel()
        self.job, self.finished = job, False
        self.cancel_button.enabled = True
        self.notice.text = "Refreshing; the table still shows the previous snapshot." if self.data else "Loading capture…"

    def refresh(self):
        if self.on_refresh:
            try:
                job = self.on_refresh()
                if job is not None:
                    self.set_job(job)
            except Exception as error:
                self.notice.text = f"Refresh unavailable: {error}"

    def poll(self):
        if self.finished or self.destroyed:
            return False
        if not self.window.visible and not getattr(self.window,"embedded",False):
            self.job.cancel()
            self.finished = True
            return False
        if not self.job.future.done():
            if not self.data:
                self.notice.text = "Reading recorded events in a background session…"
            return False
        self.finished = True
        self.cancel_button.enabled = False
        try:
            result = self.job.future.result()
            if not result.supported:
                raise ValueError(result.reason)
        except CancelledError:
            self.notice.text = "Refresh cancelled; previous snapshot retained." if self.data else "Analysis cancelled."
            return False
        except Exception as error:
            self.notice.text = f"Analysis unavailable: {error or type(error).__name__}"
            if not self.data:
                self.summary.text = "No measurements loaded."
            return False
        self.data = result
        self.data_request = getattr(self.job, "request", None)
        self.adapter = self.make_adapter(result)
        self.table.reset()
        self._filter()
        self.export_button.enabled = True
        return True

    def export(self):
        if not self.adapter:
            return
        if self.export_dialog:
            self.export_dialog.destroy()
        # Freeze the chosen table and snapshot while the save dialog is open.
        adapter, options = self.adapter, self.export_options()
        def save(path):
            try:
                adapter.export_csv(path, overwrite=path.exists(), **options)
                self.notice.text = f"Saved {path.name}"
            except Exception as error:
                self.notice.text = f"CSV export failed: {error}"
                raise
        self.export_dialog = CaptureFileDialog(save, mode="save", initial_path=Path.home()/self.export_name,
                                                extensions=(".csv",), default_extension=".csv", title=f"Export {self.title.lower()}")
        self.export_dialog.show()

    def export_options(self):
        return dict(text=self.text, sort=self.sort, reverse=self.reverse)

    def destroy(self):
        if self.destroyed:
            return
        self.destroyed = True
        self.job.cancel()
        if self.export_dialog:
            self.export_dialog.destroy()
        self.window.destroy()


class MemoryView(_DiagnosticView):
    title = "Memory"
    export_name = "memory.csv"
    default_sort = ("allocated_bytes", True)
    make_adapter = MemoryModel

    def __init__(self, job, on_refresh=None, surface=None, *, host=None):
        self.outstanding = False
        self._sort_modes = {False: ("allocated_bytes", True), True: ("bytes", True)}
        super().__init__(job, on_refresh, surface, host=host)

    def _build_options(self):
        with ui.HStack(height=fl.pvd_control, spacing=fl.pvd_spacing):
            self.activity_button = ui.Button("Frame activity", width=160, name="primary", clicked_fn=lambda: self._mode(False))
            self.outstanding_button = ui.Button("Recorded outstanding", width=200, clicked_fn=lambda: self._mode(True))
            ui.Label("Byte counts from the native capture", name="muted",elided_text=True)

    def get_workspace_state(self):
        modes=dict(self._sort_modes)
        modes[self.outstanding]=self.sort,self.reverse
        return dict(super().get_workspace_state(), outstanding=self.outstanding,
                    sort_modes=modes)

    def restore_workspace_state(self,state):
        self._mode(state["outstanding"])
        self._sort_modes=dict(state["sort_modes"])
        super().restore_workspace_state(state)

    def get_inspector_state(self):
        return dict(super().get_inspector_state(), outstanding=self.outstanding)

    def _mode(self, outstanding):
        if outstanding == self.outstanding:
            return
        self._sort_modes[self.outstanding] = self.sort, self.reverse
        self.outstanding = outstanding
        self.sort, self.reverse = self._sort_modes[outstanding]
        self.activity_button.name = "" if outstanding else "primary"
        self.outstanding_button.name = "primary" if outstanding else ""
        self.table.reset()
        self.tree.column_widths = self.widths()
        self.header.rebuild()
        self._filter()

    def columns(self):
        if self.outstanding:
            return (("Source", "file"), ("Line", "line"), ("Type", "type"), ("Recorded index", "index"), ("Outstanding B", "bytes"))
        return (("Source", "file"), ("Line", "line"), ("Type", "type"), ("Allocated B", "allocated_bytes"),
                ("Allocations", "allocation_count"), ("Freed B", "deallocated_bytes"), ("Frees", "deallocation_count"), ("Churn B", "churn_bytes"))

    def widths(self):
        if self.outstanding:
            return [ui.Fraction(2), ui.Pixel(76), ui.Fraction(2), ui.Pixel(145), ui.Pixel(170)]
        return [ui.Fraction(2), ui.Pixel(66), ui.Fraction(2), ui.Pixel(125), ui.Pixel(110), ui.Pixel(120), ui.Pixel(80), ui.Pixel(115)]

    def _filter(self):
        if not self.adapter:
            return
        data = self.data
        budget = getattr(self.data_request, "max_events", None)
        budget_text = f" / {budget:,} event limit" if budget else " events"
        prefix = f"Frame {data.current_frame:,} · {data.raw_event_count:,}{budget_text} · "
        if self.outstanding:
            records = self.adapter.outstanding(self.text, sort=self.sort, reverse=self.reverse)
            known = [r.bytes for r in data.outstanding if r.bytes is not None]
            measured = f"{sum(known):,} B across {len(known):,} available entries" if known else "Unavailable"
            self.summary.text = prefix + f"Recorded outstanding: {measured}\nCaptured signed 32-bit totals; unavailable values stay blank in CSV."
            self.notice.text = f"{len(data.outstanding)-len(known):,} entries have no recorded value. These totals are separate from this frame's allocations and frees."
            if any(value < 0 for value in known):
                self.notice.text += " Negative recorded totals may reflect signed 32-bit overflow."
            rows = []
            for r in records:
                values = (_basename(r.file), f"{r.line:,}", r.type, str(r.index), "Unavailable" if r.bytes is None else f"{r.bytes:,}")
                rows.append(((r.file, r.type, r.line, r.index), values, (r.file, str(r.line), r.type, str(r.index), values[-1])))
            total = len(data.outstanding)
        else:
            records = self.adapter.rows(self.text, sort=self.sort, reverse=self.reverse)
            t = data.total
            self.summary.text = prefix + f"Allocated {t.allocated_bytes:,} B ({t.allocation_count:,}) · Freed {t.deallocated_bytes:,} B ({t.deallocation_count:,}) · Churn {t.churn_bytes:,} B\nFrees include only allocations matched within this frame."
            self.notice.text = " · ".join(data.diagnostics) or "All recorded events in this frame were processed."
            rows = []
            for r in records:
                values = (_basename(r.file), f"{r.line:,}", r.type, f"{r.allocated_bytes:,}", f"{r.allocation_count:,}", f"{r.deallocated_bytes:,}", f"{r.deallocation_count:,}", f"{r.churn_bytes:,}")
                rows.append(((r.file, r.type, r.line), values, (r.file, *values[1:])))
            total = len(data.rows)
        self.table.set_rows(rows, len(self.columns()))
        self.count.text = f"{len(records):,} / {total:,} rows · Click a column to sort · CSV exports the displayed table"

    def export_options(self):
        return dict(**super().export_options(), outstanding=self.outstanding)


class ErrorStreamView(_DiagnosticView):
    title = "Error Stream"
    export_name = "error-stream.csv"
    default_sort = ("index", False)
    make_adapter = ErrorStreamModel

    def columns(self):
        return (("Index", "index"), ("Code", "code"), ("Message", "message"), ("Source", "file"), ("Line", "line"))

    def widths(self):
        return [ui.Pixel(75), ui.Pixel(95), ui.Fraction(3), ui.Fraction(2), ui.Pixel(90)]

    def _filter(self):
        if not self.adapter:
            return
        data = self.data
        records = self.adapter.rows(self.text, sort=self.sort, reverse=self.reverse)
        self.summary.text = f"Frame {data.current_frame:,} · {data.total:,} recorded errors · First legacy error log {data.log_instance_id}" if data.has_log else f"Frame {data.current_frame:,} · This capture has no recorded legacy Error Stream."
        rows = []
        for r in records:
            values = (str(r.index), str(r.code), r.message, _basename(r.file), str(r.line))
            rows.append((r.index, values, (*values[:3], r.file, values[-1])))
        self.table.set_rows(rows, len(self.columns()))
        self.count.text = f"{len(records):,} shown / {len(data.records):,} loaded / {data.total:,} recorded · CSV exports the displayed table"
        self.notice.text = (f"Display limit reached: first {len(data.records):,} records only. " if data.truncated else "") + "These records have no individual frame or timestamp. OVD Capture messages are shown separately."

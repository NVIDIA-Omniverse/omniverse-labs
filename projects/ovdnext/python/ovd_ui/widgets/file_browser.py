# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture open/save workflow, independent of the native session and renderer.

The current published ovui-widgets wheel does not include its content browser.
This PVD surface composes OVUI's virtualized TreeView and confirmation service;
it can be replaced by FilePickerDialog without changing callers' Path callback.
Validation remains separate from widgets so it can be used by drops/recents too.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from itertools import count
import os
from pathlib import Path
from typing import Callable

import omni.ui as ui
from ovui_widgets.common.dialogs import confirm_dialog

from ..camera import normalize_key
from ..style import layout as fl, style_dialog

CAPTURE_EXTENSIONS = (".ovd", ".pxd2", ".pxd3", ".pvd")
_SORT_OPTIONS = ("Name", "Modified: newest first", "Modified: oldest first")
_DIALOG_IDS = count()


@dataclass(frozen=True)
class FileEntry:
    path: Path
    directory: bool
    size: int = 0
    modified: float = 0


def scan_directory(directory: Path, extensions=CAPTURE_EXTENSIONS,
                   search: str = "", show_hidden: bool = False) -> list[FileEntry]:
    """List local capture files and navigable folders, folders first."""
    extensions = tuple(extension.lower() for extension in extensions)
    entries = []
    with os.scandir(directory) as listing:
        for entry in listing:
            if not show_hidden and entry.name.startswith("."):
                continue
            try:
                is_directory = entry.is_dir()
                if not is_directory and (not entry.is_file() or
                        (extensions and Path(entry.name).suffix.lower() not in extensions)):
                    continue
                if search.casefold() not in entry.name.casefold():
                    continue
                info = None if is_directory else entry.stat()
                entries.append(FileEntry(Path(entry.path), is_directory,
                                         info.st_size if info else 0, info.st_mtime if info else 0))
            except OSError:
                # A disappearing entry should not hide the rest of a folder.
                continue
    return sorted(entries, key=lambda entry: (not entry.directory, entry.path.name.casefold()))


def validate_capture_path(value: str | Path, directory: Path, *, mode="open",
                          extensions=CAPTURE_EXTENSIONS,
                          default_extension=".pxd2") -> Path:
    """Return an absolute local path or a readable, actionable ValueError.

    This function never writes, creates, or truncates a file. Existing save
    targets must pass an additional overwrite decision in the UI/controller.
    """
    if mode not in ("open", "save"):
        raise ValueError("File mode must be open or save.")
    raw = str(value).strip()
    if not raw:
        raise ValueError("Choose a capture or enter a file name.")
    if "\x00" in raw:
        raise ValueError("The file name contains an invalid character.")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = directory / path
    if mode == "save" and not path.suffix:
        path = path.with_suffix(default_extension)
    try:
        path = path.resolve()
        if path.is_dir():
            raise ValueError("Choose a file inside this folder.")
        suffixes = tuple(extension.lower() for extension in extensions)
        if suffixes and path.suffix.lower() not in suffixes:
            formats = ", ".join(extension[1:].upper() for extension in suffixes)
            raise ValueError(f"Choose a supported capture file: {formats}.")
        if mode == "open":
            if not path.is_file():
                raise ValueError(f"File not found: {path.name}")
            if not os.access(path, os.R_OK):
                raise ValueError(f"This capture cannot be read: {path.name}")
        elif not path.parent.is_dir():
            raise ValueError("The destination folder does not exist. Choose an existing folder.")
        elif not os.access(path.parent, os.W_OK):
            raise ValueError("The destination folder is not writable.")
        elif path.exists() and not os.access(path, os.W_OK):
            raise ValueError(f"This file cannot be replaced: {path.name}")
    except OSError as error:
        raise ValueError(f"Cannot access this path: {error.strerror or error}") from error
    return path


def _format_size(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024


class _FileItem(ui.AbstractItem):
    def __init__(self, entry):
        super().__init__()
        self.entry = entry
        try:
            modified = "" if entry.directory else datetime.fromtimestamp(entry.modified).strftime("%Y-%m-%d %H:%M")
        except (OSError, OverflowError, ValueError):
            modified = "Unknown"
        self.values = [ui.SimpleStringModel(entry.path.name),
                       ui.SimpleStringModel("Folder" if entry.directory else entry.path.suffix[1:].upper()),
                       ui.SimpleStringModel("" if entry.directory else _format_size(entry.size)),
                       ui.SimpleStringModel(modified)]


class _FileModel(ui.AbstractItemModel):
    def __init__(self):
        super().__init__()
        self.items = []
        self.sort_order = 0
        self.headers = [ui.SimpleStringModel(name) for name in ("Name", "Kind", "Size", "Modified")]

    def update(self, entries):
        self.items = [_FileItem(entry) for entry in entries]
        self.sort(self.sort_order)

    def sort(self, order):
        self.sort_order = order
        self.items.sort(key=lambda item: item.entry.path.name.casefold())
        if order:
            self.items.sort(key=lambda item: item.entry.modified, reverse=order == 1)
        self.items.sort(key=lambda item: not item.entry.directory)
        self._item_changed(None)

    def get_item_children(self, item=None):
        return self.items if item is None else []

    def get_item_value_model_count(self, item=None):
        return len(self.headers)

    def get_item_value_model(self, item, column_id=0):
        return item.values[column_id] if item else self.headers[column_id]


class _FileDelegate(ui.AbstractItemDelegate):
    def build_header(self, column_id):
        ui.Label(("Name", "Kind", "Size", "Modified")[column_id], name="muted", height=fl.pvd_control)

    def build_widget(self, model, item, column_id, level, expanded):
        if item:
            ui.Label(model.get_item_value_model(item, column_id).as_string,
                     height=fl.pvd_row, explicit_hover=True, tooltip=str(item.entry.path),
                     name="" if column_id == 0 else "muted")


class CaptureFileDialog:
    """Reusable capture browser. Call ``show`` inside a running OVUI app.

    ``on_selected(Path)`` is invoked only for a valid target and, in save mode,
    after explicit overwrite confirmation. Exceptions leave the browser open
    with the error visible. Async operations can call ``report_error`` later.
    Retain the instance until the application closes and call ``destroy`` then.
    Directory enumeration runs off the UI thread; stale navigation is discarded.
    """
    def __init__(self, on_selected: Callable[[Path], object], *, mode="open",
                 initial_path=None, title=None, extensions=CAPTURE_EXTENSIONS,
                 default_extension=".pxd2", on_cancel=None, on_closed=None, sort_order=0, filter_label=None):
        if mode not in ("open", "save"):
            raise ValueError("File mode must be open or save.")
        self.mode = mode
        self.on_selected, self.on_cancel = on_selected, on_cancel
        self.on_closed = on_closed
        self.extensions = tuple(extensions)
        self.csv_mode = tuple(ext.lower() for ext in self.extensions) == (".csv",)
        self.filter_label = filter_label
        self.default_extension = default_extension
        self.title = title or ("Open capture" if mode == "open" else "Save capture as")
        # ImGui identifies windows by title. A picker can have the same visible
        # title as its parent or another picker without sharing popup state.
        self.window_id = f"{self.title}##pvd-file-{next(_DIALOG_IDS)}"
        initial = Path(initial_path or Path.cwd()).expanduser().absolute()
        self.directory = initial if initial.is_dir() else initial.parent
        self.filename = "" if initial.is_dir() else initial.name
        self.search = ""
        self.filter_index = 0
        self.sort_order = sort_order if type(sort_order) is int and 0 <= sort_order < len(_SORT_OPTIONS) else 0
        self.error = ""
        self.window = None
        self.confirmation = None
        self._task = None
        self._generation = 0
        self._dismissed = True
        self._entries = []

    @property
    def visible(self):
        return bool(self.window and self.window.visible)

    @visible.setter
    def visible(self, value):
        self.show() if value else self.hide()

    def show(self, path=None):
        if self.window is None:
            self._build()
        self._dismissed = False
        self.window.visible = True
        if path is not None:
            path = Path(path).expanduser().absolute()
            if not path.is_dir():
                self.filename = path.name
                self.filename_field.model.set_value(self.filename)
                path = path.parent
            self.directory = path
        self.navigate(self.directory)

    def hide(self):
        was_open = not self._dismissed
        self._dismissed = True
        if self.window:
            self.window.visible = False
        if was_open and self.on_closed:
            self.on_closed()

    def destroy(self):
        self.hide()
        if self.confirmation:
            self.confirmation.cancel()
            self.confirmation = None
        self._generation += 1
        if self._task:
            self._task.cancel()
            self._task = None
        if self.window:
            self.window.destroy()
            self.window = None

    def cancel(self):
        if self._dismissed:
            return
        self.hide()
        if self.on_cancel:
            self.on_cancel()

    def report_error(self, message):
        self.error = str(message)
        if self.window:
            self.error_label.text = self.error
            self.error_label.tooltip = self.error
            self.error_label.visible = bool(self.error)

    def snapshot(self):
        return {"visible": self.visible, "mode": self.mode,
                "directory": str(self.directory),
                "filename": self.filename_field.model.as_string if self.window else self.filename,
                "error": self.error, "entries": len(self._entries),
                "pending": bool(self._task and not self._task.done())}

    def navigate(self, path):
        path = Path(path).expanduser()
        if not path.is_absolute():
            path = self.directory / path
        self._generation += 1
        generation = self._generation
        if self._task:
            self._task.cancel()
        self.report_error("")
        self.summary.text = "Reading folder…"
        self._task = asyncio.ensure_future(self._read_directory(path, generation))

    async def _read_directory(self, path, generation):
        extensions = self.extensions if self.filter_index == 0 else (self.extensions[self.filter_index - 1],)
        try:
            resolved = path.resolve()
            entries = await asyncio.to_thread(scan_directory, resolved, extensions, self.search)
        except (OSError, ValueError) as error:
            if generation == self._generation and self.window:
                self.report_error(f"Cannot open folder: {getattr(error, 'strerror', None) or error}")
                self.summary.text = "Choose another folder or edit the location above."
            return
        if generation != self._generation or self.window is None:
            return
        self.directory, self._entries = resolved, entries
        self.path_field.model.set_value(str(resolved))
        self.model.update(entries)
        files = sum(not entry.directory for entry in entries)
        noun = self.filter_label or ("CSV files" if self.csv_mode else "captures")
        self.summary.text = f"{files} {noun} · {len(entries) - files} folders" if entries else f"No {noun} match this folder and filter."
        self.up_button.enabled = resolved.parent != resolved

    def _search_changed(self, model):
        self.search = model.as_string
        self.navigate(self.directory)

    def _filter_changed(self, model, _item=None):
        self.filter_index = model.get_item_value_model().as_int
        self.navigate(self.directory)

    def _selection_changed(self, items):
        if items:
            entry = items[0].entry
            self.filename = entry.path.name
            self.filename_field.model.set_value(self.filename)

    def _double_clicked(self, *_args):
        self.choose()

    def choose(self):
        value = self.filename_field.model.as_string.strip()
        self.filename = value
        try:
            candidate = Path(value).expanduser()
            if not candidate.is_absolute():
                candidate = self.directory / candidate
            if value and candidate.is_dir():
                self.filename_field.model.set_value("")
                self.filename = ""
                self.navigate(candidate)
                return
            default_extension = (self.extensions[self.filter_index - 1]
                                 if self.filter_index else self.default_extension)
            path = validate_capture_path(value, self.directory, mode=self.mode,
                                         extensions=self.extensions, default_extension=default_extension)
        except (ValueError, OSError) as error:
            self.report_error(error)
            return
        self.report_error("")
        if self.mode == "save" and path.exists():
            # Capture the validated path; editing the filename behind this
            # confirmation can never redirect its eventual callback.
            self.confirmation = confirm_dialog(
                "Replace capture?", f"{path.name}\n\nA file with this name already exists.",
                on_confirm=lambda: self._accept(path), confirm_label="Replace", cancel_label="Keep file")
            if self.confirmation:
                style_dialog(self.confirmation._window)
                self._build_confirmation(self.confirmation, path)
        else:
            self._accept(path)

    @staticmethod
    def _build_confirmation(dialog, path):
        # Released OVUI renders native modal backgrounds independently of
        # Window styles. Paint the body explicitly while retaining the OVUI
        # service's public confirm/cancel callbacks and window lifetime guard.
        dialog._window.height = fl.pvd_control * 7
        dialog._window.flags |= ui.WINDOW_FLAGS_NO_SCROLLBAR | ui.WINDOW_FLAGS_NO_TITLE_BAR
        dialog._window.set_key_pressed_fn(lambda key,modifiers,pressed:
            dialog.cancel() if pressed and normalize_key(key)==256 else None)
        dialog._window.frame.clear()
        with dialog._window.frame:
            with ui.ZStack():
                ui.Rectangle(name="dialog")
                with ui.VStack(spacing=fl.pvd_spacing):
                    ui.Spacer(height=fl.pvd_spacing)
                    with ui.HStack():
                        ui.Spacer(width=fl.pvd_padding)
                        with ui.VStack(spacing=fl.pvd_spacing):
                            with ui.HStack(height=fl.pvd_control):
                                ui.Label("Replace capture?",name="title")
                                ui.Button("Close",name="quiet",width=fl.pvd_control*2,clicked_fn=dialog.cancel)
                            ui.Label(path.name, height=fl.pvd_row, elided_text=True,
                                     explicit_hover=True, tooltip=str(path))
                            ui.Label("A file with this name already exists.", name="muted", height=fl.pvd_row)
                            ui.Spacer()
                            with ui.HStack(height=fl.pvd_control, spacing=fl.pvd_spacing):
                                ui.Spacer()
                                ui.Button("Keep file", width=fl.pvd_button_width, clicked_fn=dialog.cancel)
                                ui.Button("Replace", name="primary", width=fl.pvd_button_width, clicked_fn=dialog.confirm)
                        ui.Spacer(width=fl.pvd_padding)
                    ui.Spacer(height=fl.pvd_spacing)

    def _accept(self, path):
        try:
            self.on_selected(path)
        except Exception as error:
            self.report_error(error)
            return
        self.hide()

    def _key(self, key, modifiers, pressed):
        if not pressed or self._dismissed:
            return
        key=normalize_key(key)
        if key == 256:
            self.cancel()
        elif key in (257, 335) and not modifiers & 15:
            self.choose()

    def _build(self):
        flags = (ui.WINDOW_FLAGS_MODAL | ui.WINDOW_FLAGS_NO_DOCKING |
                 ui.WINDOW_FLAGS_NO_SCROLLBAR | ui.WINDOW_FLAGS_NO_TITLE_BAR)
        self.window = ui.Window(self.window_id, width=fl.pvd_dialog_width,
                                height=fl.pvd_dialog_height, flags=flags)
        style_dialog(self.window)
        self.window.set_key_pressed_fn(self._key)
        self.model = _FileModel()
        self.model.sort_order = self.sort_order
        self.delegate = _FileDelegate()
        with self.window.frame:
            with ui.ZStack():
                ui.Rectangle(name="dialog")
                with ui.VStack(spacing=fl.pvd_spacing):
                    ui.Spacer(height=fl.pvd_spacing)
                    with ui.HStack():
                        ui.Spacer(width=fl.pvd_padding)
                        with ui.VStack(spacing=fl.pvd_spacing):
                            with ui.HStack(height=fl.pvd_control):
                                ui.Label(self.title, name="title")
                                ui.Button("Close", name="quiet", width=fl.pvd_button_width, clicked_fn=self.cancel)
                            ui.Label(f"Browse {self.filter_label} files" if self.filter_label else "Choose a folder for CSV data" if self.csv_mode else "Browse physics recordings", name="muted", height=fl.pvd_row)
                            with ui.HStack(height=fl.pvd_control, spacing=fl.pvd_spacing):
                                self.up_button = ui.Button("Up", width=fl.pvd_control * 2,
                                                           clicked_fn=lambda: self.navigate(self.directory.parent))
                                ui.Button("Home", width=fl.pvd_control * 2,
                                          clicked_fn=lambda: self.navigate(Path.home()))
                                self.path_field = ui.StringField(explicit_hover=True, tooltip="Folder location; click Go to navigate")
                                ui.Button("Go", width=fl.pvd_control * 2,
                                          clicked_fn=lambda: self.navigate(self.path_field.model.as_string))
                            with ui.HStack(height=fl.pvd_control, spacing=fl.pvd_spacing):
                                ui.Label("Search", name="muted", width=fl.pvd_control * 2)
                                self.search_field = ui.StringField(explicit_hover=True, tooltip="Filter file and folder names")
                                self.search_field.model.add_value_changed_fn(self._search_changed)
                                names = [self.filter_label or ("CSV files" if self.csv_mode else "All captures")] + [ext[1:].upper() for ext in self.extensions]
                                self.filter_combo = ui.ComboBox(0, *names, width=fl.pvd_inspector)
                                self.filter_combo.model.add_item_changed_fn(self._filter_changed)
                            with ui.HStack(height=fl.pvd_control, spacing=fl.pvd_spacing):
                                ui.Label("Sort", name="muted", width=fl.pvd_control * 2)
                                self.sort_combo = ui.ComboBox(self.model.sort_order, *_SORT_OPTIONS)
                                self.sort_combo.model.add_item_changed_fn(lambda model, _: self.model.sort(model.get_item_value_model().as_int))
                            with ui.ScrollingFrame():
                                self.tree = ui.TreeView(self.model, header_visible=True,
                                    column_widths=[ui.Fraction(1), ui.Pixel(72), ui.Pixel(80), ui.Pixel(150)],
                                    delegate=self.delegate,
                                    columns_resizable=True, root_visible=False,
                                    mouse_double_clicked_fn=self._double_clicked)
                                self.tree.set_selection_changed_fn(self._selection_changed)
                            self.summary = ui.Label("", name="muted", height=fl.pvd_row)
                            with ui.HStack(height=fl.pvd_control, spacing=fl.pvd_spacing):
                                ui.Label("File name", width=fl.pvd_button_width, name="muted")
                                self.filename_field = ui.StringField()
                                self.filename_field.model.set_value(self.filename)
                            self.error_label = ui.Label("", name="error", height=fl.pvd_row, explicit_hover=True,
                                                        word_wrap=True, visible=False)
                            with ui.HStack(height=fl.pvd_control, spacing=fl.pvd_spacing):
                                ui.Label(" / ".join(ext[1:].upper() for ext in self.extensions), name="muted")
                                ui.Button("Cancel", width=fl.pvd_button_width, clicked_fn=self.cancel)
                                self.primary_button=ui.Button(("Open " if self.mode=="open" else "Save ")+self.filter_label if self.filter_label else ("Open CSV" if self.mode == "open" else "Save CSV") if self.csv_mode else ("Open capture" if self.mode == "open" else "Save capture"),
                                          name="primary", width=fl.pvd_button_width * 1.4, clicked_fn=self.choose)
                        ui.Spacer(width=fl.pvd_padding)
                    ui.Spacer(height=fl.pvd_spacing)

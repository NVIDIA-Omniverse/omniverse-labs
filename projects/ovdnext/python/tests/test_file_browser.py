# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from datetime import datetime
import os
from types import SimpleNamespace

import pytest

from ovd_ui.widgets.file_browser import scan_directory, validate_capture_path
from ovd_ui.widgets.file_browser import CaptureFileDialog, FileEntry, _FileModel


def test_modified_sort_keeps_folders_first_and_preserves_selected_items(tmp_path):
    for name, modified in (("a-old.ovd", 100), ("z-new.OVD", 300), ("b-tie.ovd", 300)):
        path = tmp_path / name
        path.write_bytes(b"capture")
        os.utime(path, (modified, modified))
    (tmp_path / "Scenes").mkdir()
    model = _FileModel()
    model.update(scan_directory(tmp_path))
    selected = model.items[1]
    assert selected.entry.path.name == "a-old.ovd"
    model.sort(1)
    assert [item.entry.path.name for item in model.items] == ["Scenes", "b-tie.ovd", "z-new.OVD", "a-old.ovd"]
    assert model.items[-1] is selected
    assert selected.entry.modified == 100
    assert model.get_item_value_model_count() == 4
    assert model.get_item_value_model(selected, 3).as_string
    model.sort(2)
    assert [item.entry.path.name for item in model.items] == ["Scenes", "a-old.ovd", "b-tie.ovd", "z-new.OVD"]
    model.sort(1)
    model.update(scan_directory(tmp_path, search="ovd"))
    assert [item.entry.path.name for item in model.items] == ["b-tie.ovd", "z-new.OVD", "a-old.ovd"]
    model.sort(0)
    assert [item.entry.path.name for item in model.items] == ["a-old.ovd", "b-tie.ovd", "z-new.OVD"]


def test_capture_enumeration_is_case_insensitive_and_keeps_folders(tmp_path):
    for name in ("body.OVD", "cloth.pxd2", "world.PXD3", "legacy.pvd", "notes.txt", ".hidden.ovd"):
        (tmp_path / name).write_text("data")
    (tmp_path / "Scenes").mkdir()
    entries = scan_directory(tmp_path)
    assert [entry.path.name for entry in entries] == ["Scenes", "body.OVD", "cloth.pxd2", "legacy.pvd", "world.PXD3"]
    assert entries[0].directory
    assert entries[1].size == 4
    assert [entry.path.name for entry in scan_directory(tmp_path, search="BODY")] == ["body.OVD"]
    assert [entry.path.name for entry in scan_directory(tmp_path, extensions=(".ovd",))] == ["Scenes", "body.OVD"]


@pytest.mark.parametrize("saved", [None, -1, 3, "newest", [], True])
def test_invalid_saved_sort_falls_back_to_name(saved, tmp_path):
    browser = CaptureFileDialog(lambda _: None, initial_path=tmp_path, sort_order=saved)
    try:
        browser._build()
        browser.model.update([FileEntry(tmp_path / "z.ovd", False, modified=2),
                              FileEntry(tmp_path / "a.ovd", False, modified=1)])
        assert [item.entry.path.name for item in browser.model.items] == ["a.ovd", "z.ovd"]
        assert browser.sort_combo.model.get_item_value_model().as_int == 0
    finally:
        browser.destroy()


@pytest.mark.parametrize("modified", [-1, -11644473600, 253402300800, 1e20])
def test_unformattable_date_does_not_hide_capture(modified, tmp_path):
    model = _FileModel()
    model.update([FileEntry(tmp_path / "old.ovd", False, 7, modified)])
    assert model.items[0].entry.path.name == "old.ovd"
    assert model.get_item_value_model(model.items[0], 0).as_string == "old.ovd"
    try:
        expected = datetime.fromtimestamp(modified).strftime("%Y-%m-%d %H:%M")
    except (OSError, OverflowError, ValueError):
        expected = "Unknown"
    assert model.get_item_value_model(model.items[0], 3).as_string == expected


def test_open_validation_preserves_supported_legacy_formats_and_relative_paths(tmp_path):
    for suffix in (".ovd", ".pxd2", ".pxd3", ".pvd", ".OVD"):
        capture = tmp_path / f"capture{suffix}"
        capture.write_bytes(b"capture")
        assert validate_capture_path(capture.name, tmp_path) == capture
    with pytest.raises(ValueError, match="File not found"):
        validate_capture_path("missing.ovd", tmp_path)
    with pytest.raises(ValueError, match="supported capture"):
        validate_capture_path("scene.usd", tmp_path)
    with pytest.raises(ValueError, match="inside this folder"):
        validate_capture_path(tmp_path, tmp_path)


def test_save_validation_is_non_destructive_and_obeys_export_formats(tmp_path):
    assert validate_capture_path("copy", tmp_path, mode="save") == tmp_path / "copy.pxd2"
    assert not (tmp_path / "copy.pxd2").exists()
    existing = tmp_path / "recording.pxd2"
    existing.write_bytes(b"preserve this")
    assert validate_capture_path(existing, tmp_path, mode="save") == existing
    assert existing.read_bytes() == b"preserve this"
    with pytest.raises(ValueError, match="destination folder does not exist"):
        validate_capture_path("new_folder/output.pxd2", tmp_path, mode="save")
    with pytest.raises(ValueError, match="supported capture"):
        validate_capture_path("copy.ovd", tmp_path, mode="save", extensions=(".pxd2", ".pxd3"))
    with pytest.raises(ValueError, match="Choose a capture"):
        validate_capture_path("", tmp_path)
    with pytest.raises(ValueError, match="invalid character"):
        validate_capture_path("broken\0.ovd", tmp_path)


def test_symlink_targets_are_resolved_before_overwrite_decision(tmp_path):
    from filesystem_capabilities import symlink
    original = tmp_path / "original.pxd2"
    original.write_bytes(b"original")
    alias = tmp_path / "alias.pxd2"
    symlink(alias, original)
    result = validate_capture_path(alias, tmp_path, mode="save")
    assert result == original
    assert result.exists()
    assert original.read_bytes() == b"original"


def test_overwrite_callback_requires_confirmation_and_freezes_target(tmp_path, monkeypatch):
    original = tmp_path / "recording.pxd2"
    original.write_bytes(b"capture")
    accepted = []
    prompts = []
    monkeypatch.setattr("ovd_ui.widgets.file_browser.confirm_dialog", lambda *args, **kwargs: prompts.append(kwargs))
    browser = CaptureFileDialog(accepted.append, mode="save", initial_path=original)
    browser.filename_field = SimpleNamespace(model=SimpleNamespace(as_string=original.name))
    browser.choose()
    assert not accepted
    assert original.read_bytes() == b"capture"
    browser.filename_field.model.as_string = "unconfirmed.pxd2"
    prompts[0]["on_confirm"]()
    assert accepted == [original]


def test_rejected_callback_keeps_recoverable_error(tmp_path):
    capture = tmp_path / "broken.ovd"
    capture.write_bytes(b"invalid")
    def reject(_path):
        raise RuntimeError("The recording header is invalid.")
    browser = CaptureFileDialog(reject, initial_path=capture)
    browser.filename_field = SimpleNamespace(model=SimpleNamespace(as_string=capture.name))
    browser._dismissed = False
    browser.choose()
    assert browser.error == "The recording header is invalid."
    assert not browser._dismissed


@pytest.mark.parametrize('key',[256,526])
def test_escape_dismisses_picker_once_on_press_in_both_key_spaces(tmp_path,key):
    cancelled=[]
    browser=CaptureFileDialog(lambda *_:pytest.fail('Cancel must not select'),initial_path=tmp_path,
                              on_cancel=lambda:cancelled.append(True))
    browser._dismissed=False
    browser._key(key,0,False)
    assert not browser._dismissed and not cancelled
    browser._key(key,0,True)
    assert browser._dismissed and cancelled==[True]
    browser._key(key,0,False)
    browser._key(key,0,True)
    assert cancelled==[True]


@pytest.mark.parametrize('key',[257,525,335,627])
def test_enter_submits_once_on_press_and_ignores_modified_keys(tmp_path,key):
    source=tmp_path/'capture.ovd';source.write_bytes(b'capture')
    accepted=[]
    browser=CaptureFileDialog(accepted.append,initial_path=source)
    browser.filename_field=SimpleNamespace(model=SimpleNamespace(as_string=source.name))
    browser._dismissed=False
    browser._key(key,0,False)
    assert not accepted and not browser._dismissed
    for modifiers in [1,2,4,8]:browser._key(key,modifiers,True)
    assert not accepted and not browser._dismissed
    browser._key(key,0,True)
    assert accepted==[source] and browser._dismissed
    browser._key(key,0,False)
    browser._key(key,0,True)
    assert accepted==[source]

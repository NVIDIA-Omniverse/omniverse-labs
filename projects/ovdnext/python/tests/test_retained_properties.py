# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retained rows must show current owned values and release obsolete readers."""
import gc
import struct
from types import SimpleNamespace
import weakref

import pytest
from _pvd_native import SnapshotDataUnavailable

from capture_semantics import unique_instance_id
from ovd_ui.adapters import PvdDataAdapter
from ovd_ui.array_settings import ArrayDisplaySettings
from ovd_ui.widgets.properties import PropertyInspector
from test_property_context import display, prepare, ObservedView
from test_array_lifetime import recording


def restrict(adapter, identity, names):
    props = [adapter.property(identity, name) for name in names]
    adapter.schema[adapter.instances[identity].class_id] = props
    return props


def inspector(**kwargs):
    return PropertyInspector(lambda *_: None, lambda *_: None, {}, **kwargs)


def test_native_pose_updates_while_unchanged_scalar_widget_is_retained(tmp_path, display):
    source, adapter = prepare(tmp_path, "multi_geom.ovd", "box", 30)
    identity = adapter.fixture_id
    _, pose = restrict(adapter, identity, ["environmentID", "globalPose"])
    target = inspector()
    target.build(adapter, [identity], retain=True)
    unchanged = display.find("4294967295")
    old_pose = display.find("Translation")
    frames = {key: row[0] for key, row in target._rows.items()}
    # Adjacent post/pre publications repeat the pose; the next post changes it.
    repeated = PvdDataAdapter(source.presentation(frame=31, inspect_ids=[identity]))
    restrict(repeated, identity, ["environmentID", "globalPose"])
    target.build(repeated, [identity], retain=True)
    assert display.find("Translation") is old_pose
    following = PvdDataAdapter(source.presentation(frame=32, inspect_ids=[identity]))
    restrict(following, identity, ["environmentID", "globalPose"])
    before = len(display.widgets)
    target.build(following, [identity], retain=True)
    assert display.find("4294967295") is unchanged
    assert display.find("Translation") is old_pose
    assert {key: row[0] for key, row in target._rows.items()} == frames
    values = struct.unpack("<7f", following.session.read(identity, pose.id))
    expected = ", ".join(f"{value:.6f}" for value in values[4:])
    assert display.find(expected).tooltip == expected
    assert len(display.widgets) == before
    assert adapter.session.read(identity, pose.id) != following.session.read(identity, pose.id)


class ValueView(ObservedView):
    def __init__(self, view, prop, raw=None, names=None, unavailable=False):
        super().__init__(view)
        self.prop, self.raw, self.names, self.unavailable = prop, raw, names, unavailable

    def read(self, identity, prop, **kwargs):
        if prop == self.prop and self.unavailable:
            raise SnapshotDataUnavailable("Not published")
        return self.raw if prop == self.prop and self.raw is not None else super().read(identity, prop, **kwargs)

    def named_values(self, prop):
        return self.names if prop == self.prop and self.names is not None else self.view.named_values(prop)


def test_equal_formatted_text_still_refreshes_after_exact_byte_change(tmp_path, display):
    _, adapter = prepare(tmp_path)
    identity = adapter.fixture_id
    prop, = restrict(adapter, identity, ["mass"])
    view = adapter.session
    adapter.session = ValueView(view, prop.id, struct.pack("<f", 1.0))
    target = inspector()
    target.build(adapter, [identity], retain=True)
    original = display.find("1.000000")
    before = target._rows[prop.id][1]
    adapter.session = ValueView(view, prop.id, struct.pack("<I", 0x3f800001))
    target.build(adapter, [identity], retain=True)
    assert display.find("1.000000") is original
    assert target._rows[prop.id][1] != before


def test_updated_compound_copy_uses_current_text_and_open_menu_keeps_old_text(tmp_path, display):
    source, adapter = prepare(tmp_path, "multi_geom.ovd", "box", 30)
    identity = adapter.fixture_id
    prop, = restrict(adapter, identity, ["globalPose"])
    copied = []
    target = inspector(on_copy_text=copied.append)
    target.build(adapter, [identity], retain=True)
    heading = display.find("globalPose")
    values = target._value_rows[prop.id]
    translation = values.values[0][0]
    old = translation.text
    menu = display.open_menu(translation)
    following = PvdDataAdapter(source.presentation(frame=32, inspect_ids=[identity]))
    restrict(following, identity, ["globalPose"])
    target.build(following, [identity], retain=True)
    assert target._value_rows[prop.id] is values and translation.text != old
    assert translation.tooltip == translation.text
    assert menu["Copy value"]["triggered_fn"]() and copied[-1] == old
    menu = display.open_menu(heading)
    assert menu["Copy value"]["triggered_fn"]()
    assert copied[-1] == "".join(label.text + "; " for label, _entry in values.values)
    menu = display.open_menu(translation)
    assert menu["Copy value"]["triggered_fn"]() and copied[-1] == translation.text


def test_short_compound_payload_rebuilds_and_complete_values_recover(tmp_path, display):
    _, adapter = prepare(tmp_path, "multi_geom.ovd", "box", 30)
    identity = adapter.fixture_id
    prop, = restrict(adapter, identity, ["globalPose"])
    view = adapter.session
    target = inspector()
    target.build(adapter, [identity], retain=True)
    old = target._value_rows[prop.id]
    adapter.session = ValueView(view, prop.id, raw=b"\x01\x02")
    target.build(adapter, [identity], retain=True)
    assert prop.id not in target._value_rows
    adapter.session = view
    target.build(adapter, [identity], retain=True)
    assert target._value_rows[prop.id] is not old
    assert len(target._value_rows[prop.id].values) == 2


def test_named_metadata_refreshes_even_when_native_enum_bytes_match(tmp_path, display):
    _, adapter = prepare(tmp_path)
    identity = adapter.fixture_id
    prop, = restrict(adapter, identity, ["type"])
    view = adapter.session
    value = int.from_bytes(view.read(identity, prop.id), "little")
    target = inspector()
    adapter.session = ValueView(view, prop.id, names=[("Original name", value)])
    target.build(adapter, [identity], retain=True)
    first = display.find(f"Original name ({value})")
    adapter.session = ValueView(view, prop.id, names=[("Updated name", value)])
    target.build(adapter, [identity], retain=True)
    assert display.find(f"Updated name ({value})") is not first


def test_unavailable_value_recovers_and_retained_scalars_do_not_keep_reader(tmp_path, display):
    _, adapter = prepare(tmp_path)
    identity = adapter.fixture_id
    prop, = restrict(adapter, identity, ["mass"])
    view = adapter.session
    observed = ValueView(view, prop.id, unavailable=True)
    adapter.session = observed
    target = inspector()
    target.build(adapter, [identity], retain=True)
    assert display.find("Waiting for a published property value…")
    reference = weakref.ref(observed)
    adapter.session = view
    target.build(adapter, [identity], retain=True)
    assert display.find("2.000000")
    del observed
    gc.collect()
    assert reference() is None
    current = ObservedView(view)
    reference = weakref.ref(current)
    adapter.session = current
    target.build(adapter, [identity], retain=True)
    del current, adapter
    gc.collect()
    assert reference() is None, "Unchanged scalar rows must not retain their previous native reader"


@pytest.mark.parametrize("change", ["capture", "segment", "scene", "name", "class", "schema", "format", "arrays"])
def test_layout_metadata_invalidates_rows(tmp_path, display, change):
    _, adapter = prepare(tmp_path)
    identity = adapter.fixture_id
    prop, = restrict(adapter, identity, ["mass"])
    target = inspector()
    target.build(adapter, [identity], retain=True)
    original = target._rows[prop.id][0]
    if change == "capture":
        _, adapter = prepare(tmp_path / "second")
        identity = adapter.fixture_id
        restrict(adapter, identity, ["mass"])
    elif change in ("segment", "scene"):
        session = ObservedView(adapter.session)
        name = "segment_epoch" if change == "segment" else "scene_id"
        setattr(session, name, getattr(session, name) + 1)
        adapter.session = session
    elif change == "name":
        adapter.names[identity] = "Renamed actor"
    elif change == "class":
        adapter.classes = adapter.classes.copy()
        adapter.classes[adapter.instances[identity].class_id] = SimpleNamespace(name="Renamed class")
    elif change == "schema":
        fields = {name: getattr(prop, name) for name in ("id", "name", "type", "native_type", "semantic", "array", "field_count")}
        fields["name"] = "Renamed property"
        adapter.schema[adapter.instances[identity].class_id] = [SimpleNamespace(**fields)]
    elif change == "format":
        target.preferences["float_precision"] = 9
    else:
        target.array_settings = ArrayDisplaySettings(20, 500)
    target.build(adapter, [identity], retain=True)
    assert target._rows[prop.id][0] is not original


def test_capture_change_keeps_open_copy_snapshot_but_rejects_old_plot(tmp_path, display):
    _, adapter = prepare(tmp_path)
    identity = adapter.fixture_id
    restrict(adapter, identity, ["mass"])
    copied, plotted = [], []
    target = PropertyInspector(lambda *_: None, lambda *args: plotted.append(args), {}, on_copy_text=copied.append)
    target.build(adapter, [identity], retain=True)
    menu = display.open_menu(display.find("mass"))
    _, other = prepare(tmp_path / "other")
    restrict(other, other.fixture_id, ["mass"])
    target.build(other,[other.fixture_id], retain=True)
    assert menu["Copy value"]["triggered_fn"]() and copied == ["2.000000"]
    assert not menu["Plot value"]["triggered_fn"]() and not plotted


def test_reference_rows_refresh_target_names_and_freeze_old_array_builders(tmp_path, display):
    _, adapter = prepare(tmp_path)
    identity = adapter.fixture_id
    prop, = restrict(adapter, identity, ["shapes"])
    shape = unique_instance_id(adapter.session, 'PxShape')
    original_name = adapter.name(shape)
    target = inspector()
    target.build(adapter, [identity], retain=True)
    old_content = next(w for w in display.widgets if w.kind == "Frame" and hasattr(w, "build"))
    old_content.rebuild()
    assert display.find(f"{original_name} [{shape}]", "Button")
    adapter.names[shape] = "Renamed shape"
    start = len(display.widgets)
    target.build(adapter, [identity], retain=True)
    fresh = next(w for w in display.widgets[start:] if w.kind == "Frame" and hasattr(w, "build"))
    assert fresh is not old_content
    fresh.rebuild()
    assert display.find(f"Renamed shape [{shape}]", "Button")
    old_content.rebuild()
    assert display.widgets[-1].text == f"{original_name} [{shape}]"


def test_owned_string_changes_refresh_without_recreating_other_scalar_rows(tmp_path, display):
    _, adapter = prepare(tmp_path)
    identity = adapter.fixture_id
    string, _ = restrict(adapter, identity, ["name", "mass"])
    target = inspector()
    class StringView(ObservedView):
        def string_value(self, *_):return self.value
    observed = StringView(adapter.session)
    adapter.session = observed
    observed.value = "First recorded string"
    target.build(adapter, [identity], retain=True)
    old_scalar = display.find("2.000000")
    observed.value = "Next recorded string"
    target.build(adapter, [identity], retain=True)
    assert display.find("Next recorded string")
    assert display.find("2.000000") is old_scalar


@pytest.mark.parametrize("death", [False, True])
def test_owned_array_refresh_and_rewind_keep_current_windows_and_guard_old_input(tmp_path, display, death):
    source, identity, prop, request, target, requests = recording(tmp_path, death=death)

    def refresh(frame):
        view = source.presentation(frame=frame, inspect_ids=[identity], read_requests=[request], allow_unavailable_requests=True)
        adapter = PvdDataAdapter(view)
        restrict(adapter, identity, ["variable"])
        start = len(display.widgets)
        target.build(adapter, [identity], retain=True)
        created = display.widgets[start:]
        for widget in created:
            if widget.kind == "Frame" and hasattr(widget, "build"):
                widget.rebuild()
        return created

    first = refresh(1)
    old_content = next(w for w in first if w.kind == "Frame" and hasattr(w, "build"))
    old_section = next(w for w in first if w.kind == "CollapsableFrame")
    old_row = target._rows[prop.id][0]
    assert display.find("18000.000000")
    refresh(2)
    if death:
        assert display.find("Not alive at this frame") and not target._rows
    else:
        assert target._rows[prop.id][0] is old_row
        assert display.find("1 elements", "CollapsableFrame")
    # Old callbacks must not mutate the new generation's expansion/request state.
    before = (target.array_expanded.copy(), target.array_positions.copy(), len(requests))
    old_section.collapsed(False)
    assert (target.array_expanded, target.array_positions, len(requests)) == before
    # A retained old builder may still render only its own immutable bytes.
    old_content.rebuild()
    assert display.find("18000.000000")
    target.array_positions[(identity, prop.id)] = 18000
    target.array_expanded.add((identity, prop.id))
    refresh(1 if death else 3)
    assert display.find("18000.000000" if death else "118000.000000")
    target.destroy()
    assert not target._rows and target._rows_key is None

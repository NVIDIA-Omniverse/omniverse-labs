# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One capture backend must supply coherent inspector, geometry and gizmos.

These exercise the real native module. Reintroducing an inspection-only
secondary store, confusing empty samples with absent ones, or retaining engine
borrows in a published view breaks the corresponding observable contract.
"""
import gc
from pathlib import Path
import struct

import numpy as np
import pytest

from _pvd_native import Session, SnapshotDataUnavailable
from ovd_ui.adapters import PvdDataAdapter
from test_property_context import display


ROOT = Path(__file__).resolve().parents[2]


def instance_named_class(view, name):
    classes = {cls.id: cls.name for cls in view.classes()}
    return next(item for item in view.instances() if item.alive and classes[item.class_id] == name)


def properties(view, instance):
    return {prop.name: prop for prop in view.properties(instance.class_id)}


def test_properties_geometry_and_gizmos_share_an_owned_dom_publication(tmp_path):
    source = Session(str(ROOT / "test/data/test_scene.ovd"), str(tmp_path))
    initial = source.presentation(frame=1)
    actor = instance_named_class(initial, "PxRigidDynamic")
    shape = instance_named_class(initial, "PxShape")
    view = source.presentation(frame=1, inspect_ids=[actor.id], overlay_mask=2 | 4 | 8 | 256)
    adapter = PvdDataAdapter(view)
    props = properties(view, actor)

    assert view.data_backend == "pvddom"
    assert adapter.name(actor.id) == "sphere"
    assert struct.unpack("<f", adapter.raw(actor.id, "mass")) == (2.0,)
    assert adapter.references(actor.id, "shapes") == [shape.id]
    assert view.property_availability(actor.id, props["mass"].id) == "present"
    with pytest.raises((IndexError, ValueError)):
        view.property_availability(actor.id, -1)

    geometry = view.geometry_snapshot()
    np.testing.assert_allclose(geometry.transforms[0, :3, 3], [420, 69.69, 42], rtol=1e-6)
    np.testing.assert_array_equal(geometry.dimensions, [[10, 10, 10]])
    assert geometry.object_ids.tolist() == [actor.id]
    overlays = view.overlay_snapshot()
    # One shape: three actor axes, two velocity arrows, three COM cross lines.
    assert overlays.ready
    assert sorted(overlays.categories.tolist()) == [2, 2, 2, 4, 8, 256, 256, 256]
    assert set(overlays.object_ids.tolist()) == {actor.id}
    assert np.isfinite(overlays.positions).all()
    assert not overlays.positions.flags.writeable
    old_pose = view.read(actor.id, props["globalPose"].id)
    old_overlay = overlays.positions.copy()

    later = source.presentation(frame=2, inspect_ids=[actor.id])
    assert later.read(actor.id, props["globalPose"].id) != old_pose
    source.presentation(frame=3)
    del source, initial, later
    gc.collect()
    assert view.read(actor.id, props["globalPose"].id) == old_pose
    np.testing.assert_array_equal(view.overlay_snapshot().positions, old_overlay)
    np.testing.assert_allclose(view.geometry_snapshot().transforms[0, :3, 3], [420, 69.69, 42], rtol=1e-6)


def test_pxd2_owned_publication_identifies_pvd_backend_after_session_close(tmp_path):
    source = Session(str(ROOT / "test/data/cloth_physx34.pxd2"), str(tmp_path))
    view = source.presentation(frame=1)
    assert view.ready and view.data_backend == "pvd"

    del source
    gc.collect()
    assert view.data_backend == "pvd"


def presence_capture(tmp_path):
    """Independent OmniPVD 0.4 stream: an empty SET and a later scalar SET."""
    context, cls = 0x1200FFEECCAA, 0x4321
    attrs = {"emptyText": 0x10091, "laterValue": 0x10131,
             "lateEmpty": 0x10141, "stableValue": 0x10151}

    def string(text):
        raw = text.encode("utf-8")
        return struct.pack("<H", len(raw)) + raw

    def declare(name, data_type):
        return struct.pack("<BIIHI", 3, cls, attrs[name], data_type, 1) + string(name)

    def value(name, raw):
        return struct.pack("<BQQBII", 6, context, context, 1, attrs[name], len(raw)) + raw

    payload = struct.pack("<III", 0, 4, 0)
    payload += struct.pack("<BII", 1, cls, 0) + string("PxScene")
    payload += (declare("emptyText", 10) + declare("laterValue", 6)
                + declare("lateEmpty", 10) + declare("stableValue", 6))
    payload += struct.pack("<BQIQ", 9, context, cls, context) + string("Presence scene")
    payload += struct.pack("<BQQ", 11, context, 1)
    payload += value("emptyText", b"")
    payload += value("stableValue", struct.pack("<I", 9))
    payload += struct.pack("<BQQ", 12, context, 1)
    payload += struct.pack("<BQQ", 11, context, 2)
    payload += value("lateEmpty", b"")
    payload += value("laterValue", struct.pack("<I", 7))
    payload += value("stableValue", struct.pack("<I", 9))
    payload += struct.pack("<BQQ", 12, context, 2)
    path = tmp_path / "presence.ovd"
    path.write_bytes(payload)
    return Session(str(path), str(tmp_path))


def test_dom_sample_presence_distinguishes_empty_absent_and_unrequested(tmp_path):
    source = presence_capture(tmp_path)
    initial = source.presentation(frame=1)
    scene = instance_named_class(initial, "PxScene")
    props = properties(initial, scene)
    assert initial.property_availability(scene.id, props["laterValue"].id) == "not_projected"
    with pytest.raises(SnapshotDataUnavailable):
        initial.read(scene.id, props["laterValue"].id)
    first = source.presentation(frame=1, inspect_ids=[scene.id])
    assert first.property_availability(scene.id, props["emptyText"].id) == "present"
    assert first.property_availability(scene.id, props["laterValue"].id) == "absent_at_frame"
    assert first.string_value(scene.id, props["emptyText"].id) == ""
    assert first.read(scene.id, props["emptyText"].id) == b""
    # Raw API compatibility: availability carries the distinction, not invented bytes.
    assert first.read(scene.id, props["laterValue"].id) == b""
    second = source.presentation(frame=2, inspect_ids=[scene.id])
    assert second.property_availability(scene.id, props["laterValue"].id) == "present"
    assert second.read(scene.id, props["laterValue"].id) == struct.pack("<I", 7)
    assert first.property_availability(scene.id, props["laterValue"].id) == "absent_at_frame"
    assert first.data_backend == second.data_backend == "pvddom"


def test_inspector_labels_missing_dom_samples_and_recovers(tmp_path, display):
    from ovd_ui.widgets.properties import PropertyInspector

    source = presence_capture(tmp_path)
    scene = instance_named_class(source.presentation(frame=1), "PxScene")
    first = source.presentation(frame=1, inspect_ids=[scene.id])
    prop = properties(first, scene)["laterValue"]
    inspector = PropertyInspector(lambda *_: None, lambda *_: None, {})
    inspector._build_property_row(PvdDataAdapter(first), scene.id, prop)
    assert display.find("No recorded value at this frame").text == "No recorded value at this frame"

    second = source.presentation(frame=2, inspect_ids=[scene.id])
    inspector._build_property_row(PvdDataAdapter(second), scene.id, prop)
    assert display.find("7").text == "7"


def test_retained_inspector_rebuilds_when_empty_string_availability_changes(tmp_path, display):
    from ovd_ui.widgets.properties import PropertyInspector

    source = presence_capture(tmp_path)
    scene = instance_named_class(source.presentation(frame=1), "PxScene")
    first = source.presentation(frame=1, inspect_ids=[scene.id])
    second = source.presentation(frame=2, inspect_ids=[scene.id])
    props = properties(first, scene)
    late_empty = props["lateEmpty"]
    stable = props["stableValue"]
    assert first.property_availability(scene.id, late_empty.id) == "absent_at_frame"
    assert second.property_availability(scene.id, late_empty.id) == "present"

    def retained_adapter(view):
        adapter = PvdDataAdapter(view)
        adapter.schema[adapter.instances[scene.id].class_id] = [late_empty, stable]
        return adapter

    inspector = PropertyInspector(lambda *_: None, lambda *_: None, {})
    inspector.build(retained_adapter(first), [scene.id], retain=True)

    before = len(display.widgets)
    inspector.build(retained_adapter(second), [scene.id], retain=True)
    forward_labels = [widget.text for widget in display.widgets[before:] if widget.kind == "Label"]
    assert "lateEmpty" in forward_labels
    assert "-" in forward_labels
    assert stable.name not in forward_labels

    before = len(display.widgets)
    inspector.build(retained_adapter(first), [scene.id], retain=True)
    backward_labels = [widget.text for widget in display.widgets[before:] if widget.kind == "Label"]
    assert "lateEmpty" in backward_labels
    assert "No recorded value at this frame" in backward_labels
    assert stable.name not in backward_labels

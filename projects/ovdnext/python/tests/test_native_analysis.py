# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Atomic native analysis batches retain file semantics and restore the session."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import struct

import numpy as np
import pytest
import _pvd_native
from _pvd_native import Session
from test_native_geometry import animated_capture  # shared pinned-writer fixture

ROOT = Path(__file__).resolve().parents[2]


def test_profile_cross_thread_id_is_exported_by_value():
    assert _pvd_native.CROSS_THREAD_ID == 99999789


def actor_mass(s):
    classes = {c.id: c.name for c in s.classes()}
    actor = next(i for i in s.instances() if i.alive and classes[i.class_id] == "PxRigidDynamic")
    prop = next(p for p in s.properties(actor.class_id) if p.name.lower() == "mass")
    return actor, prop


def test_history_has_owned_typed_gaps_and_restores_frame(tmp_path):
    s = Session(str(ROOT / "test/data/test_scene.ovd"), str(tmp_path))
    scene = s.scenes()[0]
    s.seek(scene.id, 1)
    actor, prop = actor_mass(s)
    before = s.geometry_snapshot()
    history = s.scalar_history(scene.id, actor.id, prop.id, 0, 4)
    assert history.property_name == "mass" and history.type == "PvdF32"
    assert history.frames.dtype == np.uint32 and history.values.dtype == np.float64
    assert history.valid.tolist() == [0, 1, 1, 0]
    np.testing.assert_allclose(history.values[1:3], [2, 2])
    assert np.isnan(history.values[[0, 3]]).all()
    assert s.scenes()[0].current_frame == 1
    np.testing.assert_array_equal(s.geometry_snapshot().transforms, before.transforms)
    with ThreadPoolExecutor(max_workers=3) as pool:
        histories = list(pool.map(lambda _: s.scalar_history(scene.id, actor.id, prop.id, 0, 4), range(8)))
    assert all(h.valid.tolist() == [0, 1, 1, 0] for h in histories)
    del s
    assert not history.values.flags.writeable
    np.testing.assert_allclose(history.values[1:3], [2, 2])


def test_history_finds_legacy_instance_born_after_range_start(tmp_path):
    path = ROOT / "test/data/cloth_physx34.pxd2"
    display = Session(str(path), str(tmp_path))
    scene = display.scenes()[0]
    display.seek(scene.id, 1)
    actor, prop = actor_mass(display)
    s = Session(str(path), str(tmp_path))
    assert s.scenes()[0].current_frame == 0
    assert s.instances() == []
    history = s.scalar_history(scene.id, actor.id, prop.id, 0, 4)
    assert history.valid.tolist() == [0, 1, 1, 1]
    np.testing.assert_allclose(history.values[1:], [15.26813793] * 3)
    assert s.scenes()[0].current_frame == 0 and s.instances() == []
    with pytest.raises(IndexError):
        s.scalar_history(scene.id, actor.id, 999999, 0, 4)
    assert s.scenes()[0].current_frame == 0 and s.instances() == []
    with pytest.raises(IndexError):
        s.scalar_history(scene.id, 999999, prop.id, 0, 4)
    assert s.scenes()[0].current_frame == 0 and s.instances() == []


def test_declared_numeric_arrays_never_become_scalar_when_empty_or_short(tmp_path, animated_capture):
    s = Session(str(animated_capture), str(tmp_path))
    scene = s.scenes()[0]
    classes = {c.id: c.name for c in s.classes()}
    sample = next(i for i in s.instances() if classes[i.class_id] == "AnalysisSample")
    props = {p.name: p for p in s.properties(sample.class_id)}
    assert props["variable"].array and props["variable"].field_count == 0
    assert props["scalar"].field_count == 1 and not props["scalar"].array
    assert props["pair"].field_count == 2
    for frame, byte_count in [(1, 0), (2, 4)]:
        s.seek(scene.id, frame)
        assert s.property_size(sample.id, props["variable"].id) == byte_count
        with pytest.raises(ValueError, match="numeric scalar"):
            s.scalar_history(scene.id, sample.id, props["variable"].id, frame, frame + 1)
        with pytest.raises(ValueError, match="numeric scalar"):
            s.scalar_history(scene.id, sample.id, props["pair"].id, frame, frame + 1)
        assert s.scenes()[0].current_frame == frame
    h = s.scalar_history(scene.id, sample.id, props["scalar"].id, 1, 4)
    np.testing.assert_array_equal(h.values, [10, 20, 30])
    assert h.valid.tolist() == [1, 1, 1]
    assert s.scenes()[0].current_frame == 2


@pytest.mark.parametrize("class_name,property_name,type_name,format_code", [
    ("PxPhysics", "TolerancesScale.Length", "PxF32", "f"),
    ("PxPhysics", "Version.Major", "PxU32", "I"),
    ("PxMaterial", "Flags", "PxU16", "H"),
    ("PxRigidDynamic", "DominanceGroup", "PxU8", "B"),
])
def test_legacy_px_scalar_history_matches_recorded_values_and_restores_frame(
        tmp_path, class_name, property_name, type_name, format_code):
    s = Session(str(ROOT / "test/data/ue4_shooter.pxd2"), str(tmp_path))
    scene = s.scenes()[-1]
    s.seek(scene.id, 921)
    classes = {c.id: c.name for c in s.classes()}
    obj = next(i for i in s.instances() if i.alive and classes[i.class_id] == class_name)
    prop = next(p for p in s.properties(obj.class_id) if p.name == property_name)
    assert prop.type == type_name and not prop.array
    history = s.scalar_history(scene.id, obj.id, prop.id, 1, 4)
    assert s.scenes()[-1].current_frame == 921
    assert history.type == type_name and history.property_name == property_name
    assert history.frames.tolist() == [1, 2, 3] and history.valid.tolist() == [1, 1, 1]
    for frame, value in zip(history.frames, history.values):
        s.seek(scene.id, int(frame))
        expected, = struct.unpack("<" + format_code, s.read(obj.id, prop.id))
        assert value == expected


def test_legacy_px_composite_history_remains_unsupported(tmp_path):
    s = Session(str(ROOT / "test/data/ue4_shooter.pxd2"), str(tmp_path))
    scene = s.scenes()[-1]
    s.seek(scene.id, 921)
    classes = {c.id: c.name for c in s.classes()}
    obj = next(i for i in s.instances() if i.alive and classes[i.class_id] == "PxRigidDynamic")
    prop = next(p for p in s.properties(obj.class_id) if p.name == "GlobalPose")
    assert prop.type == "PxTransform"
    with pytest.raises(ValueError, match="numeric scalar"):
        s.scalar_history(scene.id, obj.id, prop.id, 1, 4)
    assert s.scenes()[-1].current_frame == 921


def test_message_pages_copy_actual_native_record_and_timing_is_format_specific(tmp_path):
    s = Session(str(ROOT / "test/data/testRecordMessage.ovd"), str(tmp_path))
    assert s.message_count() == 1
    msg, = s.messages(0, 1)
    assert msg.frame_id == 0 and msg.type_name == "eOUT_OF_MEMORY"
    assert msg.message == "Test Error Message"
    assert s.messages(1, 1) == [] and s.messages(0, 0) == []
    with pytest.raises(ValueError):
        s.messages(0, 1001)
    assert not s.frame_timing().supported
    del s
    assert msg.message == "Test Error Message"
    s = Session(str(ROOT / "test/data/cloth_physx34.pxd2"), str(tmp_path))
    s.seek(s.scenes()[0].id, 2)
    timing = s.frame_timing()
    assert timing.supported
    assert timing.inner_ms == pytest.approx(6.491027224645894)
    assert timing.total_ms == pytest.approx(55.76889291221413)
    assert s.message_count() == 0 and s.messages() == []


def test_empty_legacy_reference_lists_do_not_call_reader_with_null_output(tmp_path):
    s = Session(str(ROOT / "test/data/cloth_physx34.pxd2"), str(tmp_path))
    s.seek(s.scenes()[0].id, 1)
    physics = next(i for i in s.instances() if i.id == 4)
    props = [p for p in s.properties(physics.class_id) if p.type == "ObjectRef"]
    values = {p.name: s.references(physics.id, p.id) for p in props}
    assert values["Scenes"] == [6]
    assert any(value == [] for value in values.values())

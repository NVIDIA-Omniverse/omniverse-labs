# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source identities and stored sample stamps on the actual pinned DOM reader."""
from pathlib import Path
import struct
import subprocess

import pytest
from _pvd_native import Session
from native_fixtures import executable

ROOT = Path(__file__).resolve().parents[2]


def opened(tmp_path, name="test_scene.ovd"):
    return Session(str(ROOT / "test/data" / name), str(tmp_path))


def rows(session, size=2):
    result, offset = [], 0
    while True:
        page = session.evidence_metadata(offset, size)
        assert page.capture_key == session.capture_key
        result.extend(page.rows)
        if page.next_offset is None:
            assert len(result) == page.total
            return result, page.domain
        assert page.next_offset > offset
        offset = page.next_offset


def test_source_namespace_and_scopes_do_not_seek_the_display(tmp_path):
    session = opened(tmp_path)
    scene = session.scenes()[0]
    session.presentation(scene.id, 1)
    identities, domain = rows(session)
    actor = next(row for row in identities if row.class_name == "PxRigidDynamic")
    assert (actor.object_uid, actor.api_handle, actor.recording_segment) == (8, 1342902134704, 0)
    assert (actor.scene_instance_id, actor.scene_context, actor.scene_object_uid) == (scene.instance_id, 1342900227584, 3)
    assert actor.supported and -(2**31) <= actor.instance_id < 2**31
    assert domain.supported and domain.frame_scopes_complete
    assert (domain.min_frame, domain.max_frame) == (1, 3)
    scope, = domain.scopes
    assert (scope.segment, scope.context, scope.scene_uid) == (0, 1342900227584, 3)
    assert (scope.first, scope.last, scope.count) == (1, 3, 3)
    assert scope.scene_mapped and scope.monotonic and scope.consecutive and not scope.scene_recreated
    assert session.scenes()[0].current_frame == 1


def test_reused_handle_keeps_distinct_source_segments_and_stored_values(tmp_path):
    session = opened(tmp_path, "resample_two_segment.ovd")
    identities, domain = rows(session)
    actors = {row.object_uid: row for row in identities if row.object_uid in (11, 22)}
    assert set(actors) == {11, 22}
    assert actors[11].api_handle == actors[22].api_handle == 2929591180976
    assert (actors[11].recording_segment, actors[22].recording_segment) == (0, 1)
    assert actors[11].scene_instance_id != actors[22].scene_instance_id
    assert domain.segment_maxima == [121, 121]
    expected = {11: (0.0, -50.16350173950195, -100.0),
                22: (-1.0197088718414307, 23.702146530151367, -71.46412658691406)}
    for uid, actor in actors.items():
        scene = next(s for s in session.scenes() if s.instance_id == actor.scene_instance_id)
        cls = next(i.class_id for i in session.instances() if i.id == actor.instance_id)
        prop = next(p.id for p in session.properties(cls) if p.name == "linearVelocity")
        plain = session.presentation(scene.id, 2)
        assert plain.evidence_identity(actor.instance_id) is None
        view = session.presentation(scene.id, 2, inspect_ids=[actor.instance_id],
                                    read_requests=[(actor.instance_id, prop, 0, 12)], include_evidence=True)
        assert view.request_applied and view.segment_epoch == 1
        frozen = view.evidence_identity(actor.instance_id)
        assert (frozen.object_uid, frozen.recording_segment) == (uid, actor.recording_segment)
        stamp = view.evidence_sample_stamp(actor.instance_id, prop)
        assert stamp.supported and stamp.has_sample and stamp.frame == 2
        assert struct.unpack("<3f", view.read(actor.instance_id, prop)) == expected[uid]
        session.presentation(scene.id, 3)
        assert view.evidence_sample_stamp(actor.instance_id, prop).frame == 2
        assert view.evidence_identity(actor.instance_id).object_uid == uid


def test_half_open_lifetime_and_page_validation(tmp_path):
    session = opened(tmp_path)
    identities, _ = rows(session)
    actor = next(row for row in identities if row.object_uid == 8)
    # The checked-in inspector fixture starts membership at recorded frame 1.
    assert [(life.first, life.stop) for life in actor.lifespans] == [(1, 3)]
    scene = session.scenes()[0]
    view = session.presentation(scene.id, 3, include_evidence=True, inspect_ids=[actor.instance_id])
    assert actor.instance_id in view.unavailable_ids
    assert view.evidence_identity(actor.instance_id) is None
    for limit in (0, 513):
        with pytest.raises(ValueError, match="512"):
            session.evidence_metadata(0, limit)


def test_legacy_capture_metadata_is_explicitly_unsupported(tmp_path):
    session = opened(tmp_path, "cloth_physx34.pxd2")
    page = session.evidence_metadata(0, 1)
    assert not page.domain.supported
    assert all(not row.supported for row in page.rows)


def generated(tmp_path, mode):
    path = tmp_path / "scope.ovd"
    subprocess.run([str(executable("pvd_geometry_fixture", "PVD_GEOMETRY_FIXTURE")),
                    str(path), "--evidence-" + mode], check=True, timeout=30)
    return Session(str(path), str(tmp_path))


def test_large_source_handles_and_carried_vs_missing_stamps(tmp_path):
    session = generated(tmp_path, "high")
    identities, domain = rows(session)
    actor = next(row for row in identities if row.class_name == "PxRigidDynamic")
    assert actor.api_handle == 9007199254741003
    assert actor.scene_context == 9007199254740993
    assert actor.lifespans[-1].stop == 0
    cls = next(i.class_id for i in session.instances() if i.id == actor.instance_id)
    props = {p.name: p.id for p in session.properties(cls)}
    view = session.presentation(actor.scene_instance_id, 2, inspect_ids=[actor.instance_id],
                                read_requests=[(actor.instance_id, props[name], 0, 12)
                                               for name in ("linearVelocity", "angularVelocity")], include_evidence=True)
    stored = view.evidence_sample_stamp(actor.instance_id, props["linearVelocity"])
    missing = view.evidence_sample_stamp(actor.instance_id, props["angularVelocity"])
    assert stored.has_sample and stored.frame == 1  # Carried, never request frame 2.
    assert missing.supported and not missing.has_sample
    assert domain.scopes[0].context == 9007199254740993


@pytest.mark.parametrize("mode,first,last,count,monotonic,consecutive,mapped,recreated", [
    ("gap", 1, 3, 2, True, False, True, False),
    ("reset", 1, 1, 3, False, False, True, False),
    ("min", 3, 4, 2, True, True, True, False),
    ("unmapped", 1, 2, 2, True, True, False, False),
    ("recreated", 1, 2, 2, True, True, True, True),
])
def test_nonstandard_counter_scopes_are_reported_without_guessing(tmp_path, mode, first, last, count,
                                                                monotonic, consecutive, mapped, recreated):
    session = generated(tmp_path, mode)
    _, domain = rows(session)
    scope, = domain.scopes
    assert (scope.first, scope.last, scope.count) == (first, last, count)
    assert (scope.monotonic, scope.consecutive, scope.scene_mapped, scope.scene_recreated) == (
        monotonic, consecutive, mapped, recreated)


def test_scope_overflow_preserves_normal_loading_without_partial_proof(tmp_path):
    session = generated(tmp_path, "overflow")
    _, domain = rows(session)
    assert not domain.frame_scopes_complete and len(domain.scopes) == 4096
    view = session.presentation(frame=1)
    assert view.ready and view.request_applied


def test_metadata_byte_budget_rejects_a_page_without_breaking_playback(tmp_path):
    session = generated(tmp_path, "budget")
    with pytest.raises(ValueError, match="Evidence page exceeds 8 MiB"):
        session.evidence_metadata(0, 512)
    page = session.evidence_metadata(0, 1)
    assert len(page.rows) == 1
    assert session.presentation(frame=1).request_applied

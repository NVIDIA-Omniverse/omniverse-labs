# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Content, namespace, stored-value and async adoption proof on real captures."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
import subprocess
from threading import Event
import pytest

from _pvd_native import Session
from ovd_ui.evidence import parse_evidence
from ovd_ui.ingestion import load_capture
from native_fixtures import executable

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).parent / "fixtures/evidence"


def prepared(tmp_path, filename="resample_two_segment.ovd", case="v1-two-segment-reused-handle"):
    loaded = load_capture(ROOT / "test/data" / filename, temporary_directory=tmp_path)
    sdk = json.loads((FIXTURES / (case + ".json")).read_bytes())
    proof = loaded.verified_payload
    doc = dict(format="ovdnext.timeline-evidence", version=1,
               captures={role: dict(sha256=proof.sha256, byteLength=str(proof.byte_length)) for role in ("left", "right")},
               findings=[dict(id="pair", kind="comparison", evidence=sdk)])
    return loaded, Session(str(proof.path), str(tmp_path)), doc


def resolve(document, proof, session):
    from ovd_ui.evidence_binding import resolve_bindings
    return resolve_bindings(parse_evidence(json.dumps(document).encode()), proof, session, lambda: False)


def test_reused_handle_yields_only_the_two_real_role_points(tmp_path):
    loaded, private, doc = prepared(tmp_path)
    result = resolve(doc, loaded.verified_payload, private)
    assert result.state == "matching-both" and result.roles == ("left", "right")
    assert len(result.markers) == 2
    left, right = result.markers
    assert (left.role, right.role, left.frame, right.frame) == ("left", "right", 2, 2)
    assert left.scene_id != right.scene_id and left.instance_id != right.instance_id
    assert left.identity["recording_segment"] == 0 and right.identity["recording_segment"] == 1
    assert left.identity["api_handle"] == right.identity["api_handle"] == 2929591180976
    assert result.reasons[("pair", "left")] == ()


def test_missing_mismatched_and_changed_payload_never_make_markers(tmp_path):
    loaded, private, doc = prepared(tmp_path)
    missing = resolve(doc, None, private)
    assert missing.state == "no-finalized-payload" and not missing.markers
    for role in doc["captures"]:
        doc["captures"][role]["sha256"] = "b" * 64
    mismatch = resolve(doc, loaded.verified_payload, private)
    assert mismatch.state == "mismatch" and not mismatch.markers
    loaded.data_path.write_bytes(b"Changed private payload")
    changed = resolve(doc, loaded.verified_payload, private)
    assert changed.state == "changed-payload" and not changed.markers


def test_invalid_namespace_has_no_fallback_to_a_reused_handle(tmp_path):
    loaded, private, doc = prepared(tmp_path)
    sdk = doc["findings"][0]["evidence"]
    sdk["leftIdentity"]["objectId"] = sdk["leftSample"]["identity"]["objectId"] = "999999"
    result = resolve(doc, loaded.verified_payload, private)
    assert [m.role for m in result.markers] == ["right"]
    assert "namespace" in " ".join(result.reasons[("pair", "left")]).lower()


def test_measurement_values_are_checked_against_native_samples(tmp_path):
    loaded, private, doc = prepared(tmp_path, "test_scene.ovd", "v1-equal-identical-roles")
    sdk = doc["findings"][0]["evidence"]
    sdk["leftVelocity"][0] = sdk["rightVelocity"][0] = 100
    result = resolve(doc, loaded.verified_payload, private)
    assert not result.markers
    assert "value" in " ".join(result.reasons[("pair", "left")]).lower()


def test_first_dead_frame_does_not_select_an_object(tmp_path):
    loaded, private, doc = prepared(tmp_path, "test_scene.ovd", "v1-equal-identical-roles")
    sdk = doc["findings"][0]["evidence"]
    sdk["alignment"].update(startFrame="3", endFrame="3")
    for role in ("left", "right"):
        sample = sdk[role + "Sample"]
        sample["requested"]["recordedFrame"] = sample["source"]["recordedFrame"] = "3"
        sample["sampleTimestamp"] = "3"
    result = resolve(doc, loaded.verified_payload, private)
    assert not result.markers and "lifetime" in " ".join(result.reasons[("pair", "left")]).lower()


def settle(controller):
    deadline = time.monotonic() + 10
    while controller.pending and time.monotonic() < deadline:
        controller.poll();time.sleep(.005)
    assert not controller.pending


def test_controller_requires_explicit_role_and_rejects_bad_import_atomically(tmp_path):
    from ovd_ui.evidence_binding import DisplayContext, EvidenceController
    loaded, private, doc = prepared(tmp_path)
    context = DisplayContext(loaded.session.capture_key, loaded.session.info().segment_epoch, loaded.verified_payload)
    path = tmp_path / "evidence.json";path.write_text(json.dumps(doc))
    controller = EvidenceController()
    try:
        controller.import_file(path, context);settle(controller)
        original = controller.selected_set
        controller.select("pair")
        assert controller.selected_marker is None
        controller.select("pair", "right")
        assert controller.selected_marker.role == "right"
        path.write_text('{"version":1,"version":2}')
        controller.import_file(path, context);settle(controller)
        assert controller.selected_set == original and controller.selected_marker.role == "right"
        assert controller.error
        controller.sync(replace(context, capture_key=context.capture_key+1, payload=None))
        settle(controller)
        assert controller.selected_marker is None
    finally:
        controller.close()


def test_controller_storage_limit_and_duplicate_content_selection(tmp_path):
    from ovd_ui.evidence_binding import DisplayContext, EvidenceController
    sdk = json.loads((FIXTURES / "v1-equal-identical-roles.json").read_bytes())
    context, controller, first = DisplayContext(), EvidenceController(), None
    path = tmp_path / "set.json"
    try:
        for number in range(5):
            sdk["message"] = str(number)
            path.write_text(json.dumps(sdk));controller.import_file(path, context);settle(controller)
            first = first or controller.selected_set
        assert len(controller.sets) == 4 and controller.error
        sdk["message"] = "0";path.write_text(json.dumps(sdk))
        controller.import_file(path, context);settle(controller)
        assert len(controller.sets) == 4 and controller.selected_set == first and not controller.error
        controller.sync(replace(context, capture_key=1));settle(controller)
        assert controller.selected_set == first  # Rebinding is not a set selection.
    finally:
        controller.close()


def test_stale_import_cannot_replace_existing_selection(tmp_path, monkeypatch):
    from ovd_ui import evidence_binding as binding
    controller, context = binding.EvidenceController(), binding.DisplayContext()
    sdk = json.loads((FIXTURES / "v1-equal-identical-roles.json").read_bytes())
    path = tmp_path / "set.json";path.write_text(json.dumps(sdk))
    entered, release = Event(), Event()
    try:
        controller.import_file(path, context);settle(controller)
        original = controller.selected_set;controller.select("comparison", "right")
        read = binding._read_json
        def delayed(source):
            entered.set();assert release.wait(5)
            return read(source)
        monkeypatch.setattr(binding, "_read_json", delayed)
        sdk["message"] = "A separate valid set";path.write_text(json.dumps(sdk))
        controller.import_file(path, context)
        assert entered.wait(5)
        controller.sync(replace(context, capture_key=99))
        release.set();settle(controller)
        assert len(controller.sets) == 1 and controller.selected_set == original
        assert controller.selection == ("comparison", "right", None)
    finally:
        release.set();controller.close()


def test_final_navigation_reads_only_the_applied_frozen_source(tmp_path):
    from ovd_ui.evidence_binding import validate_marker_snapshot
    loaded, private, doc = prepared(tmp_path)
    marker = resolve(doc, loaded.verified_payload, private).markers[0]
    read = marker.reads[0]
    view = loaded.session.presentation(marker.scene_id, marker.frame, inspect_ids=[marker.instance_id],
        read_requests=[(read.instance_id, read.property_id, 0, read.field_count * 4)], include_evidence=True)
    assert not validate_marker_snapshot(view, marker)
    loaded.session.presentation(marker.scene_id, 3)
    assert not validate_marker_snapshot(view, marker)
    assert validate_marker_snapshot(loaded.session.presentation(marker.scene_id, 3), marker)


def energy_set(tmp_path):
    path = tmp_path / "energy.ovd"
    subprocess.run([str(executable("pvd_geometry_fixture", "PVD_GEOMETRY_FIXTURE")), str(path), "--evidence-energy"], check=True)
    loaded = load_capture(path, temporary_directory=tmp_path)
    private = Session(str(loaded.data_path), str(tmp_path))
    identities = private.evidence_metadata().rows
    actor = next(row for row in identities if row.class_name == "PxRigidDynamic")
    scene = next(row for row in identities if row.instance_id == actor.scene_instance_id)
    def identity(row):
        data = json.loads((FIXTURES / "v1-equal-identical-roles.json").read_bytes())["leftIdentity"]
        data.update(objectId=str(row.object_uid), externalHandle=str(row.api_handle), recordingSegment=str(row.recording_segment), className=row.class_name)
        return data
    def frame(n):
        return dict(recordingSegment="0", context=str(actor.scene_context), recordedFrame=str(n))
    observations, supporting = [], []
    for attribute, value, units, row in [("mass", [2], "kg", actor), ("massSpaceInertiaTensor", [1, 2, 3], "kg*m^2", actor),
        ("cMassLocalPose", [0, 0, 0, 1, 0, 0, 1], "m", actor), ("gravity", [0, 0, -10], "m/s^2", scene),
        ("energyReferencePosition", [0, 0, 0], "m", actor)]:
        observations.append(dict(id=attribute, role="left", identity=identity(row), attribute=attribute, units=units,
            requested=frame(1), source=None, sampleState="Carried", origin="producer-log" if attribute == "energyReferencePosition" else "native-recorded",
            value=value, note="Explicit reference" if attribute == "energyReferencePosition" else "Recorded initialization"))
        supporting.append(attribute)
    points = []
    for n in (1, 2):
        references = []
        for attribute, value, units in [("globalPose", [0, 0, 0, 1, 1, 2, 3], "m"),
                                         ("linearVelocity", [2, 0, 0], "m/s"), ("angularVelocity", [0, 0, 1], "rad/s")]:
            key = attribute + str(n);references.append(key)
            observations.append(dict(id=key, role="left", identity=identity(actor), attribute=attribute, units=units,
                requested=frame(n), source=frame(n), sampleState="Exact", origin="native-recorded", value=value, note="Recorded sample"))
        points.append(dict(frame=frame(n), phase="unknown", valueJ=85.5,
                           components=dict(translationalJ=4, rotationalJ=1.5, gravitationalJ=80), observations=references))
    window = dict(id="energy", kind="scalar-window", role="left", identity=identity(actor), anchor=frame(1),
        measure=dict(name="mechanicalEnergy.relativeHandoverChange", units="fraction", value=0, reference=0, tolerance=.01, status="Observed"),
        points=points, supporting=supporting, assumptions="Test analytic policy; no solver verdict", message="Constant recorded energy")
    doc = dict(format="ovdnext.timeline-evidence", version=1,
               captures={"left": dict(sha256=loaded.verified_payload.sha256, byteLength=str(loaded.verified_payload.byte_length))},
               observations=observations, findings=[window])
    return loaded, private, doc


def test_scalar_points_require_actual_static_initialization_and_exact_dynamic_samples(tmp_path):
    loaded, private, doc = energy_set(tmp_path)
    result = resolve(doc, loaded.verified_payload, private)
    assert [m.frame for m in result.markers] == [1, 2]
    assert all(len(m.reads) == 7 for m in result.markers)
    assert any(read.source_frame == 0 for read in result.markers[0].reads)
    assert result.reasons[("energy", "left")] == ()


@pytest.mark.parametrize("change", [lambda d: d["observations"][0].update(origin="producer-log"),
    lambda d: d["observations"][5].update(source=None),
    lambda d: d["observations"][0].update(value=None),
    lambda d: d["findings"][0]["points"][0].update(valueJ=99),
    lambda d: d["findings"][0]["measure"].update(value=.5)])
def test_unverified_or_contradictory_scalar_inputs_never_mark_frames(tmp_path, change):
    loaded, private, doc = energy_set(tmp_path);change(doc)
    assert not resolve(doc, loaded.verified_payload, private).markers


@pytest.mark.parametrize("mode,frame_number", [("gap", 1), ("reset", 1), ("min", 3), ("unmapped", 1), ("recreated", 1), ("overflow", 1)])
def test_unsupported_native_scopes_never_guess_a_frame_relation(tmp_path, mode, frame_number):
    path = tmp_path / "scope.ovd"
    subprocess.run([str(executable("pvd_geometry_fixture", "PVD_GEOMETRY_FIXTURE")), str(path), "--evidence-" + mode], check=True)
    loaded = load_capture(path, temporary_directory=tmp_path)
    private = Session(str(loaded.data_path), str(tmp_path))
    actor = next(r for r in private.evidence_metadata().rows if r.class_name == "PxRigidDynamic")
    sdk = json.loads((FIXTURES / "v1-equal-identical-roles.json").read_bytes())
    sdk["alignment"].update(startFrame=str(frame_number), endFrame=str(frame_number), leftContext=str(actor.scene_context), rightContext=str(actor.scene_context))
    for role in ("left", "right"):
        sdk[role + "Identity"].update(objectId=str(actor.object_uid), externalHandle=str(actor.api_handle), recordingSegment=str(actor.recording_segment))
        sample = sdk[role + "Sample"]
        sample["identity"] = dict(sdk[role + "Identity"])
        for field in ("requested", "source"):
            sample[field].update(context=str(actor.scene_context), recordedFrame=str(frame_number))
        sdk[role + "Velocity"] = [1, 2, 3]
    doc = dict(format="ovdnext.timeline-evidence", version=1,
               captures={r: dict(sha256=loaded.verified_payload.sha256, byteLength=str(loaded.verified_payload.byte_length)) for r in ("left", "right")},
               findings=[dict(id="scope", kind="comparison", evidence=sdk)])
    result = resolve(doc, loaded.verified_payload, private)
    assert not result.markers
    assert any(word in " ".join(result.reasons[("scope", "left")]).lower() for word in ("scope", "domain", "lifetime", "namespace"))


def test_changed_payload_after_native_reads_discards_every_marker(tmp_path):
    loaded, private, doc = prepared(tmp_path)
    class ChangedDuringRead:
        def __getattr__(self, name):
            return getattr(private, name)
        def presentation(self, *args, **kwargs):
            view = private.presentation(*args, **kwargs)
            loaded.data_path.write_bytes(b"Changed after the first hash")
            return view
    result = resolve(doc, loaded.verified_payload, ChangedDuringRead())
    assert result.state == "changed-payload" and not result.markers


def test_aggregate_storage_budget_rejects_new_set_without_dropping_existing(tmp_path):
    from ovd_ui.evidence_binding import DisplayContext, EvidenceController
    controller, context = EvidenceController(), DisplayContext()
    sdk = json.loads((FIXTURES / "v1-equal-identical-roles.json").read_bytes())
    sdk["futureSamples"] = [None] * 30000
    path = tmp_path / "set.json"
    try:
        for number in range(4):
            sdk["message"] = str(number);path.write_text(json.dumps(sdk))
            controller.import_file(path, context);settle(controller)
        assert len(controller.sets) == 3 and controller.error
        assert sum(stored.evidence.node_count for stored in controller.sets.values()) <= 100000
    finally:
        controller.close()


def test_changed_private_payload_is_an_explicit_rebind_state(tmp_path):
    from ovd_ui.evidence_binding import DisplayContext, EvidenceController
    loaded, private, doc = prepared(tmp_path)
    context = DisplayContext(loaded.session.capture_key, loaded.session.info().segment_epoch, loaded.verified_payload)
    path = tmp_path / "evidence.json";path.write_text(json.dumps(doc))
    controller = EvidenceController()
    try:
        controller.import_file(path, context);settle(controller)
        original = controller.selected_set
        loaded.data_path.write_bytes(b"Changed private payload")
        controller.sync(replace(context, capture_key=context.capture_key+1));settle(controller)
        assert controller.selected_set == original
        assert controller.sets[original].binding.state == "changed-payload"
        assert not controller.markers
    finally:
        controller.close()

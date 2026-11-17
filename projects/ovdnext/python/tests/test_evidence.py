# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real inspector records and bounded, immutable viewer evidence parsing."""
import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures/evidence"


def parse(data):
    from ovd_ui.evidence import parse_evidence
    return parse_evidence(data if isinstance(data, bytes) else json.dumps(data).encode())


def record(name="v1-equal-identical-roles"):
    return json.loads((FIXTURES / (name + ".json")).read_bytes())


def envelope(document=None):
    return {"format": "ovdnext.timeline-evidence", "version": 1,
            "captures": {role: {"sha256": "a" * 64, "byteLength": "1000"} for role in ("left", "right")},
            "findings": [{"id": "sample", "kind": "comparison", "evidence": document or record()}]}


@pytest.mark.parametrize("name,status", [
    ("v1-equal-identical-roles", "Equal"), ("v2-equal-identical-roles", "Equal"),
    ("v2-equal-orientation", "Equal"), ("v1-different-offset", "Incomparable"),
    ("v2-different-offset", "Incomparable"), ("v1-incomparable-identity", "Incomparable"),
    ("v2-incomparable-attribute", "Incomparable"),
    ("v1-two-segment-reused-handle", "Different"), ("v2-two-segment-reused-handle", "Different"),
])
def test_real_producer_outputs_keep_actual_status_and_one_representative_pair(name, status):
    result = parse((FIXTURES / (name + ".json")).read_bytes())
    assert len(result.findings) == 1
    sdk = result.findings[0].data["evidence"]
    assert sdk["status"] == status and sdk["viewerImportSupported"] is False
    assert sdk["comparedFrames"] == record(name)["comparedFrames"]
    assert not result.captures["left"].sha256
    with pytest.raises(TypeError):
        sdk["status"] = "Equal"


def test_envelope_preserves_integer_namespace_completion_and_text_replacement():
    sdk = record("v2-two-segment-reused-handle")
    sdk["textWasReplaced"] = True
    sdk["leftSelector"]["objectId"] = "18446744073709551615"
    result = parse(envelope(sdk))
    saved = result.findings[0].data["evidence"]
    assert saved["leftSelector"]["objectId"] == "18446744073709551615"
    assert saved["leftIdentity"]["externalHandle"] == saved["rightIdentity"]["externalHandle"]
    assert (saved["leftIdentity"]["recordingSegment"], saved["rightIdentity"]["recordingSegment"]) == ("0", "1")
    assert saved["leftFrameCompleted"] is True and saved["textWasReplaced"] is True
    assert result.captures["left"].sha256 == "a" * 64
    assert result.captures["left"].byte_length == 1000


@pytest.mark.parametrize("bad", ["01", "-0", "+1", "1e2", " 1", "18446744073709551616", 8, True])
def test_noncanonical_or_numeric_source_ids_are_rejected(bad):
    sdk = record();sdk["leftIdentity"]["objectId"] = bad
    with pytest.raises(ValueError):
        parse(sdk)


@pytest.mark.parametrize("bad", ["-0", "+1", "01", "9223372036854775808", "-9223372036854775809", 0, False])
def test_noncanonical_or_overflowing_signed_offsets_are_rejected(bad):
    sdk = record();sdk["alignment"]["rightFrameOffset"] = bad
    with pytest.raises(ValueError):
        parse(sdk)


@pytest.mark.parametrize("change", [lambda d: d.update(version=True), lambda d: d.update(version=3),
    lambda d: d.update(status="Passed"), lambda d: d.update(tolerance=True),
    lambda d: d["leftSample"].update(dataType=True), lambda d: d["alignment"].update(frameStride="0")])
def test_unknown_versions_enums_and_bool_numeric_fields_are_rejected(change):
    sdk = record();change(sdk)
    with pytest.raises(ValueError):
        parse(sdk)


@pytest.mark.parametrize("payload", [b"x" * (4 * 1024 * 1024 + 1), b"[" * 33 + b"0" + b"]" * 33,
    b'{"a":{"b":1,"b":2}}', b'{"x":NaN}', b'{"x":Infinity}', b'{"x":1e9999}',
    b'{"x":' + b"1" * 1025 + b'}', b'\xef\xbb\xbf{}', b'{"x":"\xff"}'],
    ids=["oversize", "depth", "duplicate", "nan", "infinity", "overflow", "number-token", "bom", "utf8"])
def test_malformed_and_resource_bounded_inputs_are_rejected(payload):
    with pytest.raises(ValueError):
        parse(payload)


def test_limits_apply_to_uninterpreted_producer_fields():
    for extra in ("x" * 65537, [None] * 50001, ["x" * 64000] * 17):
        sdk = record();sdk["futureMetadata"] = extra
        with pytest.raises(ValueError):
            parse(sdk)
    sdk = record();sdk["futureMetadata"] = {"text": "brackets [ and { inside text", "number": 1}
    assert parse(sdk).findings[0].data["evidence"]["futureMetadata"]["number"] == 1


def test_contradictory_measurements_stay_readable_with_a_non_navigation_reason():
    sdk = record();sdk["leftVelocity"] = None
    result = parse(envelope(sdk))
    assert result.findings[0].reasons
    assert result.findings[0].data["evidence"]["leftVelocity"] is None


def test_capture_metadata_has_required_text_types_and_missing_equal_is_qualified():
    sdk = record();sdk["leftCapture"]["fileHash"] = 42
    with pytest.raises(ValueError):
        parse(sdk)
    sdk = record();sdk.update(hasEvidence=False, distance=None, leftVelocity=None, rightVelocity=None)
    assert parse(sdk).findings[0].reasons


@pytest.mark.parametrize("change", [lambda d: d.update(status="Equal", distance=0),
    lambda d: d["leftSample"]["source"].update(recordedFrame="1")])
def test_forged_metric_or_exact_source_fields_cannot_become_verified(change):
    sdk = record("v1-two-segment-reused-handle");change(sdk)
    assert parse(envelope(sdk)).findings[0].reasons


def test_strict_envelope_hashes_and_unique_findings():
    for change in (lambda d: d.update(unknown=True), lambda d: d["captures"]["left"].update(sha256="A" * 64),
                   lambda d: d["findings"].append(d["findings"][0]),
                   lambda d: d["findings"].extend([dict(d["findings"][0], id=str(i)) for i in range(256)])):
        doc = envelope();change(doc)
        with pytest.raises(ValueError):
            parse(doc)
    doc = envelope();doc["findings"][0]["evidence"]["leftCapture"]["fileHash"] = "sha256:" + "b" * 64
    with pytest.raises(ValueError, match="hash"):
        parse(doc)


def scalar_set():
    identity = record()["leftIdentity"]
    frame = {"recordingSegment": "0", "context": "1342900227584", "recordedFrame": "2"}
    observation = dict(id="pose", role="left", identity=identity, attribute="globalPose", units="m",
                       requested=frame, source=frame, sampleState="Exact", origin="native-recorded",
                       value=[0, 0, 0, 1, 1, 2, 3], note="Recorded test input")
    window = dict(id="window", kind="scalar-window", role="left", identity=identity, anchor=frame,
                  measure=dict(name="mechanicalEnergy.relativeHandoverChange", units="fraction", value=None,
                               reference=0, tolerance=.01, status="Inconclusive"),
                  points=[dict(frame=dict(frame), phase="post", valueJ=85.5,
                               components=dict(translationalJ=4, rotationalJ=1.5, gravitationalJ=80), observations=["pose"])],
                  supporting=[], assumptions="Qualified analytic evidence", message="One evaluated point")
    return dict(format="ovdnext.timeline-evidence", version=1,
                captures={"left": {"sha256": "a" * 64, "byteLength": "1000"}},
                observations=[observation], findings=[window])


def test_scalar_windows_preserve_only_enumerated_points_and_observations():
    result = parse(scalar_set())
    assert tuple(result.observations) == ("pose",)
    point, = result.findings[0].data["points"]
    assert point["valueJ"] == 85.5 and point["observations"] == ("pose",)


@pytest.mark.parametrize("change", [lambda d: d["findings"][0]["supporting"].append("missing"),
    lambda d: d["observations"].append(d["observations"][0]),
    lambda d: d["observations"][0].update(role="right"),
    lambda d: d["findings"][0]["anchor"].update(recordedFrame="3"),
    lambda d: d["findings"][0].update(supporting=["pose"] * 8193),
    lambda d: d["findings"][0]["points"].extend([d["findings"][0]["points"][0]] * 256)])
def test_scalar_window_reference_role_anchor_and_point_limits(change):
    doc = scalar_set();change(doc)
    with pytest.raises(ValueError):
        parse(doc)

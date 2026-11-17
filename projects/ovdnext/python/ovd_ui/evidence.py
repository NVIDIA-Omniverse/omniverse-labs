# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded immutable evidence records. Parsing supplies no navigation authority."""
from dataclasses import dataclass
import hashlib
import json
import math
import re
from types import MappingProxyType

MAX_BYTES = 4 * 1024 * 1024
_UINT = re.compile(r"(?:0|[1-9][0-9]{0,19})\Z", re.ASCII)
_SINT = re.compile(r"(?:0|-?[1-9][0-9]{0,18})\Z", re.ASCII)
_HASH = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_ID = re.compile(r"[A-Za-z0-9_.:-]{1,64}\Z", re.ASCII)


class EvidenceError(ValueError):
    pass


@dataclass(frozen=True)
class RoleBinding:
    sha256: str | None
    byte_length: int | None
    phase_mapping: object = None


@dataclass(frozen=True)
class Finding:
    id: str
    kind: str
    data: object
    reasons: tuple = ()


@dataclass(frozen=True)
class Observation:
    id: str
    role: str
    data: object


@dataclass(frozen=True)
class EvidenceSet:
    digest: str
    findings: tuple
    captures: object
    observations: object
    node_count: int
    raw: object


def _require(condition, message):
    if not condition:
        raise EvidenceError(message)


def _fields(value, required, optional=(), *, strict=False):
    _require(type(value) is dict, "Expected a JSON object")
    required, optional = set(required.split()), set(optional.split()) if isinstance(optional, str) else set(optional)
    _require(required <= value.keys(), "Missing evidence fields: " + ", ".join(sorted(required - value.keys())))
    _require(not strict or value.keys() <= required | optional, "Unknown envelope fields")
    return value


def uint(value):
    _require(type(value) is str and _UINT.fullmatch(value) and int(value) <= 2**64 - 1,
             "Expected a canonical uint64 decimal string")
    return int(value)


def _offset(value):
    _require(type(value) is str and _SINT.fullmatch(value) and -2**63 <= int(value) < 2**63,
             "Expected a canonical int64 offset string")


def _text(value, limit=65536):
    _require(type(value) is str and len(value.encode("utf-8")) <= limit, "Evidence text exceeds its limit or is invalid")


def _enum(value, choices):
    _require(type(value) is str and value in choices.split(), "Unknown evidence enum")


def _boolean(value):
    _require(type(value) is bool, "Expected a JSON boolean")


def _integer(value, choices):
    _require(type(value) is int and value in choices, "Unsupported numeric version/type/count")


def _number(value, *, nullable=False, nonnegative=False):
    if value is None and nullable:
        return
    _require(type(value) in (int, float) and math.isfinite(value), "Expected a finite JSON number")
    _require(not nonnegative or value >= 0, "Expected a nonnegative measurement")


def _vector(value, counts, *, nullable=True):
    if value is None and nullable:
        return
    _require(type(value) is list and len(value) in counts, "Invalid measurement component count")
    for component in value:
        _number(component)


def _frame(value, *, strict=False):
    _fields(value, "recordingSegment context recordedFrame", strict=strict)
    for key in ("recordingSegment", "context", "recordedFrame"):
        uint(value[key])


def _identity(value, *, nullable=False):
    if value is None and nullable:
        return
    _fields(value, "objectId externalHandle recordingSegment className name lifetimeStart lifetimeEnd lifetimeStartCommand lifetimeEndCommand")
    for key in ("objectId", "externalHandle", "recordingSegment", "lifetimeStart", "lifetimeStartCommand"):
        uint(value[key])
    for key in ("lifetimeEnd", "lifetimeEndCommand"):
        if value[key] is not None:
            uint(value[key])
    _text(value["className"]);_text(value["name"])


def _sample(value):
    _fields(value, "identity requested source hasSourceFrame rawContext rawFrame commandOrdinal sampleTimestamp age preMembership state dataType")
    _identity(value["identity"]);_frame(value["requested"])
    if value["source"] is not None:
        _frame(value["source"])
    for key in ("rawContext", "commandOrdinal", "sampleTimestamp"):
        uint(value[key])
    for key in ("rawFrame", "age"):
        if value[key] is not None:
            uint(value[key])
    _boolean(value["hasSourceFrame"]);_boolean(value["preMembership"])
    _enum(value["state"], "Exact Carried Missing");_integer(value["dataType"], range(256))


def _sdk(value):
    _fields(value, "format version viewerImportSupported status severity color message assumptions comparedFrames metric units tolerance alignment leftSelector rightSelector leftIdentity rightIdentity leftCapture rightCapture hasEvidence distance leftSample rightSample textWasReplaced")
    _require(value["format"] == "physx.ovd.inspection-evidence", "Unsupported evidence format")
    _integer(value["version"], (1, 2))
    _enum(value["status"], "Equal Different Incomparable");_enum(value["severity"], "info warning error")
    _require(type(value["color"]) is str and re.fullmatch(r"#[0-9A-Fa-f]{6}", value["color"]), "Invalid evidence color")
    for key in ("message", "assumptions", "metric", "units"):
        _text(value[key])
    for key in ("viewerImportSupported", "hasEvidence", "textWasReplaced"):
        _boolean(value[key])
    uint(value["comparedFrames"])
    _number(value["tolerance"], nullable=True, nonnegative=True)
    _number(value["distance"], nullable=True, nonnegative=True)
    alignment = _fields(value["alignment"], "leftSegment rightSegment leftContext rightContext startFrame endFrame frameStride rightFrameOffset requireExactSamples")
    for key in ("leftSegment", "rightSegment", "leftContext", "rightContext", "startFrame", "endFrame", "frameStride"):
        uint(alignment[key])
    _require(uint(alignment["frameStride"]) > 0, "Frame stride must be positive")
    _offset(alignment["rightFrameOffset"]);_boolean(alignment["requireExactSamples"])
    for role in ("left", "right"):
        selector = _fields(value[role + "Selector"], "objectId name className")
        uint(selector["objectId"]);_text(selector["name"]);_text(selector["className"])
        _identity(value[role + "Identity"], nullable=True)
        capture = _fields(value[role + "Capture"], "sourcePath fileHash sdkVersion metadata")
        for key in ("sourcePath", "fileHash", "sdkVersion", "metadata"):
            _text(capture[key])
        _sample(value[role + "Sample"])
    if value["version"] == 1:
        _fields(value, "leftVelocity rightVelocity")
        for role in ("left", "right"):
            _vector(value[role + "Velocity"], (3,))
        values = [value[role + "Velocity"] for role in ("left", "right")]
    else:
        _fields(value, "measure componentCount leftValue rightValue leftFrameCompleted rightFrameCompleted")
        measure = _fields(value["measure"], "attributeName distanceKind units")
        _text(measure["attributeName"]);_text(measure["units"])
        _enum(measure["distanceKind"], "Vec3Euclidean TransformPositionEuclidean TransformOrientationRadians")
        _integer(value["componentCount"], (0, 3, 4))
        for role in ("left", "right"):
            _vector(value[role + "Value"], (value["componentCount"],))
            if value[role + "FrameCompleted"] is not None:
                _boolean(value[role + "FrameCompleted"])
        values = [value[role + "Value"] for role in ("left", "right")]
    reasons = []
    measured = value["hasEvidence"]
    if value["status"] == "Incomparable":
        reasons.append("No comparable recorded samples")
    elif not measured:
        reasons.append("Reported status has no measured evidence")
    if (measured and (None in values or value["distance"] is None or value["tolerance"] is None
                     or not uint(value["comparedFrames"]) or value["status"] == "Incomparable")
            or not measured and (any(v is not None for v in values) or value["distance"] is not None)):
        reasons.append("Contradictory measurement fields")
    if measured and value["distance"] is not None and value["tolerance"] is not None:
        if (value["status"] == "Equal") != (value["distance"] <= value["tolerance"]):
            reasons.append("Reported status contradicts distance and tolerance")
        if all(item is not None for item in values):
            try:
                orientation = value["version"] == 2 and value["measure"]["distanceKind"] == "TransformOrientationRadians"
                if orientation:
                    normalized = []
                    for quaternion in values:
                        scale = max(map(abs, quaternion))
                        _require(scale > 0 and len(quaternion) == 4, "Invalid comparison quaternion")
                        q = [component / scale for component in quaternion];length = math.hypot(*q)
                        normalized.append([component / length for component in q])
                    a, b = normalized
                    if sum(x*y for x, y in zip(a, b)) < 0:
                        b = [-x for x in b]
                    computed = 4 * math.atan2(math.dist(a, b), math.hypot(*(x+y for x, y in zip(a, b))))
                else:
                    _require(all(len(item) == 3 for item in values), "Invalid vector comparison width")
                    computed = math.dist(*values)
                if not math.isclose(computed, value["distance"], rel_tol=1e-9, abs_tol=1e-12):
                    reasons.append("Reported distance contradicts retained values")
            except (ValueError, OverflowError):
                reasons.append("Retained values do not support the declared metric")
    for role in ("left", "right"):
        sample = value[role + "Sample"]
        if measured and (sample["state"] == "Missing" or sample["hasSourceFrame"] != (sample["source"] is not None)):
            reasons.append("Incomplete sample provenance")
        if measured:
            requested, source = sample["requested"], sample["source"]
            if (sample["state"] == "Exact" and (source is None or sample["preMembership"]
                    or any(source[key] != requested[key] for key in ("recordingSegment", "context", "recordedFrame")))):
                reasons.append("Exact sample source contradicts requested frame")
            identity = value[role + "Identity"]
            if identity is None or any(identity[key] != sample["identity"][key]
                                       for key in ("objectId", "externalHandle", "recordingSegment", "className")):
                reasons.append("Sample identity contradicts resolved incarnation")
            if requested["recordingSegment"] != alignment[role + "Segment"] or requested["context"] != alignment[role + "Context"]:
                reasons.append("Requested sample contradicts alignment scope")
    if measured:
        left = uint(value["leftSample"]["requested"]["recordedFrame"])
        right = uint(value["rightSample"]["requested"]["recordedFrame"])
        if not (uint(alignment["startFrame"]) <= left <= uint(alignment["endFrame"])) or (
                left - uint(alignment["startFrame"])) % uint(alignment["frameStride"]) or right != left + int(alignment["rightFrameOffset"]):
            reasons.append("Retained pair contradicts requested alignment")
    return tuple(dict.fromkeys(reasons))


def _freeze(value):
    if type(value) is dict:
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if type(value) is list:
        return tuple(_freeze(v) for v in value)
    return value


def _binding(value):
    _fields(value, "sha256 byteLength", "sdkVersion provenance phaseMapping", strict=True)
    _require(type(value["sha256"]) is str and _HASH.fullmatch(value["sha256"]), "Invalid capture hash")
    _require(uint(value["byteLength"]) <= 512 * 1024 * 1024, "Capture binding exceeds snapshot limit")
    if value.get("sdkVersion") is not None:
        _text(value["sdkVersion"], 128)
    if "provenance" in value:
        _text(value["provenance"])
    phase = value.get("phaseMapping")
    if phase is not None:
        _fields(phase, "kind firstPostFrame firstCapturedStep stepSeconds provenance", strict=True)
        _require(phase["kind"] == "sdk-pre-post", "Unknown phase mapping")
        uint(phase["firstPostFrame"]);uint(phase["firstCapturedStep"])
        _number(phase["stepSeconds"]);_require(phase["stepSeconds"] > 0, "Phase interval must be positive")
        _text(phase["provenance"])
    return RoleBinding(value["sha256"], uint(value["byteLength"]), _freeze(phase))


def _identifier(value):
    _require(type(value) is str and _ID.fullmatch(value), "Invalid evidence identifier")


def _observation(value, captures):
    _fields(value, "id role identity attribute units requested source sampleState origin value note", strict=True)
    _identifier(value["id"]);_require(value["role"] in captures, "Observation role has no capture binding")
    _identity(value["identity"]);_frame(value["requested"], strict=True)
    if value["source"] is not None:
        _frame(value["source"], strict=True)
    _enum(value["sampleState"], "Exact Carried Missing");_enum(value["origin"], "native-recorded producer-log")
    _text(value["attribute"], 128);_text(value["units"], 64);_text(value["note"])
    _vector(value["value"], range(1, 8))
    return Observation(value["id"], value["role"], _freeze(value))


def _window(value, captures, observations):
    _fields(value, "id kind role identity anchor measure points supporting assumptions message", strict=True)
    role = value["role"]
    _require(role in captures, "Window role has no capture binding")
    _identity(value["identity"]);_frame(value["anchor"], strict=True)
    _text(value["assumptions"]);_text(value["message"])
    measure = _fields(value["measure"], "name units value reference tolerance status", strict=True)
    _text(measure["name"], 128);_text(measure["units"], 64)
    for key in ("value", "reference", "tolerance"):
        _number(measure[key], nullable=True, nonnegative=key == "tolerance")
    _enum(measure["status"], "Observed Warning Inconclusive")
    _require(type(value["points"]) is list and 0 < len(value["points"]) <= 256, "Window point limit exceeded")
    _require(type(value["supporting"]) is list, "Invalid supporting references")
    refs = list(value["supporting"])
    for point in value["points"]:
        _fields(point, "frame phase valueJ components observations", strict=True)
        _frame(point["frame"], strict=True);_enum(point["phase"], "pre post unknown")
        _number(point["valueJ"], nullable=True)
        components = _fields(point["components"], "translationalJ rotationalJ gravitationalJ", strict=True)
        for number in components.values():
            _number(number, nullable=True)
        _require(type(point["observations"]) is list, "Invalid point references")
        refs.extend(point["observations"])
    _require(len(refs) <= 8192, "Observation reference limit exceeded")
    _require(any(point["frame"] == value["anchor"] for point in value["points"]), "Anchor is not an evaluated point")
    for reference in refs:
        _identifier(reference)
        _require(reference in observations and observations[reference].role == role, "Unresolved observation reference/role")
        identity = observations[reference].data["identity"]
        _require(identity["recordingSegment"] == value["identity"]["recordingSegment"], "Observation segment differs")
        same_body = all(identity[key] == value["identity"][key] for key in ("objectId", "externalHandle", "className"))
        scene = identity["className"] == "PxScene" and identity["externalHandle"] == value["anchor"]["context"]
        _require(same_body or scene, "Observation belongs to a different body/scene")
    reasons = []
    if measure["name"] != "mechanicalEnergy.relativeHandoverChange" or measure["units"] != "fraction" or measure["reference"] != 0:
        reasons.append("Unsupported analytic metric/units/reference")
    if any(measure[key] is None for key in ("value", "reference", "tolerance")):
        reasons.append("Analytic measurement is inconclusive")
    return tuple(reasons), len(refs), len(value["points"])


def _decode(data):
    _require(type(data) is bytes and len(data) <= MAX_BYTES, "Evidence JSON exceeds 4 MiB")
    text = data.decode("utf-8")
    _require(not text.startswith("\ufeff"), "Evidence UTF-8 must not have a BOM")
    depth, quoted, escaped = 0, False, False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1;_require(depth <= 32, "Evidence nesting exceeds 32")
        elif char in "]}":
            depth -= 1
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    def numeric(token, conversion):
        _require(len(token) <= 1024, "JSON numeric token limit exceeded")
        result = conversion(token)
        if type(result) is float:
            _require(math.isfinite(result), "Nonfinite JSON number")
        return result
    def constant(_):
        raise EvidenceError("Nonfinite JSON token")
    root = json.loads(text, object_pairs_hook=pairs, parse_constant=constant,
                      parse_int=lambda s: numeric(s, int), parse_float=lambda s: numeric(s, float))
    stack, nodes, text_bytes = [root], 0, 0
    while stack:
        value = stack.pop();nodes += 1
        if type(value) is dict:
            nodes += len(value);stack.extend(value.values())
            for key in value:
                _text(key);text_bytes += len(key.encode("utf-8"))
        elif type(value) is list:
            stack.extend(value)
        elif type(value) is str:
            _text(value);text_bytes += len(value.encode("utf-8"))
        _require(nodes <= 50000 and text_bytes <= 1024 * 1024, "Evidence node/text budget exceeded")
    return root, nodes


def parse_evidence(data: bytes) -> EvidenceSet:
    try:
        root, nodes = _decode(data)
        _fields(root, "format version")
        observations, bindings, findings = {}, {}, []
        if root["format"] == "physx.ovd.inspection-evidence":
            reasons = _sdk(root)
            for role in ("left", "right"):
                supplied = root[role + "Capture"]["fileHash"]
                digest = supplied[7:] if type(supplied) is str and supplied.startswith("sha256:") else ""
                bindings[role] = RoleBinding(digest if _HASH.fullmatch(digest) else None, None)
            findings.append(Finding("comparison", "comparison", _freeze({"evidence": root}), reasons))
        else:
            _fields(root, "format version captures findings", "observations provenance", strict=True)
            _require(root["format"] == "ovdnext.timeline-evidence", "Unsupported evidence format")
            _integer(root["version"], (1,))
            _fields(root["captures"], "left", "right", strict=True)
            bindings = {role: _binding(value) for role, value in root["captures"].items()}
            if "provenance" in root:
                _text(root["provenance"])
            _require(type(root.get("observations", [])) is list, "Invalid observations")
            for value in root.get("observations", []):
                observation = _observation(value, bindings)
                _require(observation.id not in observations, "Duplicate observation ID")
                observations[observation.id] = observation
            _require(type(root["findings"]) is list and len(root["findings"]) <= 256, "Finding limit exceeded")
            seen, references, points = set(), 0, 0
            for value in root["findings"]:
                _fields(value, "id kind");_identifier(value["id"])
                _require(value["id"] not in seen, "Duplicate finding ID");seen.add(value["id"])
                if value["kind"] == "comparison":
                    _fields(value, "id kind evidence", strict=True)
                    reasons = _sdk(value["evidence"])
                    _require("right" in bindings, "Comparison requires both capture roles")
                    for role in ("left", "right"):
                        supplied = value["evidence"][role + "Capture"]["fileHash"]
                        _require(not supplied or supplied == "sha256:" + bindings[role].sha256, "Nested capture hash disagrees")
                elif value["kind"] == "scalar-window":
                    reasons, count, number = _window(value, bindings, observations)
                    references += count;points += number
                else:
                    raise EvidenceError("Unknown finding kind")
                _require(references <= 8192 and points <= 4096, "Evidence reference/point budget exceeded")
                findings.append(Finding(value["id"], value["kind"], _freeze(value), reasons))
        return EvidenceSet(hashlib.sha256(data).hexdigest(), tuple(findings), MappingProxyType(bindings),
                           MappingProxyType(observations), nodes, _freeze(root))
    except EvidenceError:
        raise
    except (ValueError, TypeError, KeyError, UnicodeError, RecursionError, OverflowError) as error:
        raise EvidenceError("Invalid evidence JSON: " + str(error)) from None

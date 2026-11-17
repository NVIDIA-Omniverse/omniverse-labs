# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Off-thread content proof and owned marker data; no display engine mutation."""
from concurrent.futures import CancelledError, ThreadPoolExecutor
from dataclasses import dataclass, field
import hashlib
import math
import os
from pathlib import Path
import stat
import struct
import tempfile
from threading import Event
from types import MappingProxyType

from _pvd_native import Session
from .evidence import EvidenceError, MAX_BYTES, parse_evidence, uint
from .ingestion import MAX_VERIFIED_OVD_BYTES, _identity, submit_capture
from .evidence_energy import mechanical_energy


@dataclass(frozen=True)
class DisplayContext:
    capture_key: int = 0
    segment_epoch: int = 0
    payload: object = None
    ovd: bool = True
    live: bool = False
    scene_id: int = 0
    clip: tuple = ()

    @property
    def binding_key(self):
        return (self.capture_key, self.segment_epoch, self.payload.sha256 if self.payload else None, self.ovd, self.live)


@dataclass(frozen=True)
class ReadCheck:
    instance_id: int
    property_id: int
    type_code: int
    field_count: int
    source_frame: int
    values: tuple
    identity: object
    projection: str = "vector"


@dataclass(frozen=True)
class Marker:
    set_digest: str
    finding_id: str
    role: str
    scene_id: int
    frame: int
    instance_id: int
    identity: object
    reads: tuple
    status: str


@dataclass(frozen=True)
class BindingResult:
    state: str
    roles: tuple = ()
    markers: tuple = ()
    reasons: object = field(default_factory=lambda: MappingProxyType({}))
    message: str = ""


def _check(cancelled):
    if cancelled():
        raise CancelledError("Evidence binding cancelled")


def _need(condition, message):
    if not condition:
        raise EvidenceError(message)


def verify_payload(payload, cancelled):
    _need(payload is not None and payload.byte_length <= MAX_VERIFIED_OVD_BYTES, "No finalized bounded OVD payload")
    _check(cancelled)
    before = payload.path.stat()
    _need(stat.S_ISREG(before.st_mode) and _identity(before) == payload.file_identity, "Changed payload identity")
    digest, count = hashlib.sha256(), 0
    with payload.path.open("rb") as stream:
        _need(_identity(os.fstat(stream.fileno())) == payload.file_identity, "Changed payload identity")
        while True:
            _check(cancelled)
            block = stream.read(1024 * 1024)
            if not block:
                break
            count += len(block);_need(count <= payload.byte_length, "Changed payload length")
            digest.update(block)
        _need(_identity(os.fstat(stream.fileno())) == payload.file_identity, "Changed payload identity")
    _need(_identity(payload.path.stat()) == payload.file_identity and count == payload.byte_length
          and digest.hexdigest() == payload.sha256, "Changed payload content")


_IDENTITY_FIELDS = ("object_uid", "api_handle", "recording_segment", "class_name",
                    "scene_instance_id", "scene_context", "scene_object_uid")


def identity_signature(row):
    return MappingProxyType({key: getattr(row, key) for key in _IDENTITY_FIELDS})


def _alive(row, frame):
    return any(life.first <= frame and (life.stop == 0 or frame < life.stop) for life in row.lifespans)


def _namespace(session, cancelled):
    _need(session.info().ovd and not session.info().live, "Unsupported native capture format/domain")
    identities, offset, domain = [], 0, None
    while True:
        _check(cancelled)
        page = session.evidence_metadata(offset, 512)
        _need(page.total <= 100000 and len(page.domain.scopes) <= 4096, "Native evidence metadata limit exceeded")
        domain = page.domain
        identities.extend(page.rows)
        if page.next_offset is None:
            _need(len(identities) == page.total, "Incomplete native metadata index")
            break
        _need(offset < page.next_offset <= page.total, "Invalid native metadata cursor")
        offset = page.next_offset
    by_uid, by_instance, classes = {}, {}, {}
    for row in identities:
        by_uid.setdefault(row.object_uid, []).append(row);by_instance[row.instance_id] = row
    for offset in range(0, len(identities), 512):
        _check(cancelled)
        classes.update((item.id, item.class_id) for item in session.instances(offset, 512))
    return domain, by_uid, by_instance, classes, {scene.instance_id: scene for scene in session.scenes()}


def _row(identity, frame, index):
    domain, by_uid, by_instance, _, scenes = index
    candidates = [row for row in by_uid.get(uint(identity["objectId"]), ()) if row.supported
                  and row.api_handle == uint(identity["externalHandle"])
                  and row.recording_segment == uint(identity["recordingSegment"])
                  and row.class_name == identity["className"]
                  and row.scene_context == uint(frame["context"])]
    _need(len(candidates) == 1, "No unique source namespace match")
    row = candidates[0]
    _need(_alive(row, uint(frame["recordedFrame"])), "Object lifetime excludes requested frame")
    scene = by_instance.get(row.scene_instance_id)
    _need(scene is not None and scene.class_name == "PxScene" and scene.object_uid == row.scene_object_uid
          and scene.api_handle == row.scene_context and scene.recording_segment == row.recording_segment,
          "Source scene namespace is unsupported")
    n = uint(frame["recordedFrame"])
    _need(uint(frame["recordingSegment"]) == row.recording_segment, "Source segment namespace differs")
    scopes = [scope for scope in domain.scopes if (scope.segment, scope.context, scope.scene_uid) ==
              (row.recording_segment, row.scene_context, row.scene_object_uid)]
    _need(domain.supported and domain.frame_scopes_complete and domain.min_frame == 1 and len(scopes) == 1,
          "Unsupported native frame domain")
    scope = scopes[0]
    _need(scope.scene_mapped and scope.monotonic and scope.consecutive and not scope.scene_recreated and scope.first == 1,
          "Frame scope is unmapped, gapped, reset or recreated")
    _need(0 < n < 2**32 and scope.first <= n <= scope.last and n <= domain.max_frame,
          "Frame is outside the proved source scope")
    if domain.saw_segment_metadata:
        _need(row.recording_segment < len(domain.segment_maxima) and n <= domain.segment_maxima[row.recording_segment],
              "Frame is outside its recording segment")
    else:
        _need(row.recording_segment == 0, "Source segment metadata is unavailable")
    selected = scenes.get(row.scene_instance_id)
    _need(selected is not None and n < selected.frame_count, "Frame is outside its native scene")
    return row, selected


def _property(session, row, index, attribute, count, type_code=None):
    props = [p for p in session.properties(index[3][row.instance_id]) if p.name == attribute]
    _need(len(props) == 1, "Missing or ambiguous recorded attribute")
    prop = props[0]
    native_code = 9 if prop.type == "PvdF64" else 8 if prop.type in (
        "PvdF32", "PxVec2", "PxVec3", "PxVec4", "PxBounds3", "PxTransform", "PxMat3") else None
    _need(native_code is not None and (type_code is None or type_code == native_code)
          and prop.field_count == count, "Unsupported recorded value type/component count")
    return prop, native_code


def _decoded(view, check):
    width, code = (4, "f") if check.type_code == 8 else (8, "d")
    _need(view.property_size(check.instance_id, check.property_id) == check.field_count * width, "Recorded value byte size differs")
    raw = view.read(check.instance_id, check.property_id, 0, check.field_count * width)
    _need(len(raw) == check.field_count * width, "Recorded value is missing")
    values = struct.unpack("<" + str(check.field_count) + code, raw)
    _need(all(math.isfinite(value) for value in values), "Recorded value is nonfinite")
    return values[4:7] if check.projection == "position" else values[:4] if check.projection == "orientation" else values


def validate_marker_snapshot(view, marker):
    """Final check reads only frozen Presentation data, never its engine."""
    try:
        _need(view.ready and view.request_applied and view.frame == marker.frame and view.scene_id == marker.scene_id,
              "Evidence request was not applied exactly")
        for identity_id, expected in [(marker.instance_id, marker.identity)] + [(r.instance_id, r.identity) for r in marker.reads]:
            row = view.evidence_identity(identity_id)
            _need(row is not None and all(getattr(row, key) == value for key, value in expected.items()),
                  "Displayed source identity changed")
        for check in marker.reads:
            stamp = view.evidence_sample_stamp(check.instance_id, check.property_id)
            _need(stamp is not None and stamp.supported and stamp.has_sample and stamp.frame == check.source_frame,
                  "Displayed source sample changed or is missing")
            _need(_decoded(view, check) == check.values, "Stored native value does not match imported value")
        return ""
    except (EvidenceError, ValueError, IndexError, RuntimeError) as error:
        return str(error)


def _comparison(evidence, finding, role, session, index):
    sdk = finding.data["evidence"]
    _need(not finding.reasons and sdk["hasEvidence"], "; ".join(finding.reasons) or "No retained measured pair")
    sample = sdk[role + "Sample"]
    _need(sample["state"] in ("Exact", "Carried") and sample["hasSourceFrame"] and sample["source"] is not None,
          "Source frame is unknown or missing")
    requested, source = sample["requested"], sample["source"]
    row, scene = _row(sdk[role + "Identity"], requested, index)
    _need(source["recordingSegment"] == requested["recordingSegment"] and source["context"] == requested["context"]
          and 0 < uint(source["recordedFrame"]) <= uint(requested["recordedFrame"])
          and _alive(row, uint(source["recordedFrame"])), "Source frame/lifetime cannot be mapped")
    _row(sdk[role + "Identity"], source, index)
    projection, count, attribute = "vector", 3, "linearVelocity"
    expected = sdk[role + "Velocity"] if sdk["version"] == 1 else sdk[role + "Value"]
    if sdk["version"] == 2:
        attribute = sdk["measure"]["attributeName"]
        if sdk["measure"]["distanceKind"] != "Vec3Euclidean":
            count = 7
            projection = "position" if sdk["measure"]["distanceKind"] == "TransformPositionEuclidean" else "orientation"
    _need(expected is not None and sample["dataType"] in (8, 9), "Unsupported or missing typed values")
    prop, code = _property(session, row, index, attribute, count, sample["dataType"])
    check = ReadCheck(row.instance_id, prop.id, code, count, uint(source["recordedFrame"]), tuple(expected), identity_signature(row), projection)
    marker = Marker(evidence.digest, finding.id, role, scene.id, uint(requested["recordedFrame"]), row.instance_id,
                    identity_signature(row), (check,), sdk["status"])
    view = session.presentation(scene.id, marker.frame, inspect_ids=[row.instance_id],
                                read_requests=[(row.instance_id, prop.id, 0, count * (4 if code == 8 else 8))], include_evidence=True)
    reason = validate_marker_snapshot(view, marker)
    _need(not reason, reason)
    return (marker,)


_ENERGY_FIELDS = {
    "mass": (1, "kg"), "massSpaceInertiaTensor": (3, "kg*m^2"), "cMassLocalPose": (7, "m"),
    "gravity": (3, "m/s^2"), "globalPose": (7, "m"), "linearVelocity": (3, "m/s"), "angularVelocity": (3, "rad/s"),
}
_STATIC_ENERGY = ("mass", "massSpaceInertiaTensor", "cMassLocalPose", "gravity")


def _observation_check(observation, frame, session, index, *, static=False):
    value = observation.data
    attribute = value["attribute"]
    count, units = _ENERGY_FIELDS[attribute]
    _need(value["origin"] == "native-recorded" and value["units"] == units and value["value"] is not None
          and len(value["value"]) == count, "Missing recorded energy input or incompatible units")
    row, _ = _row(value["identity"], frame, index)
    _row(value["identity"], value["requested"], index)
    source = value["source"]
    if source is None:
        _need(static and value["sampleState"] == "Carried", "Dynamic sample has no exact source frame")
        source_frame = 0  # Confirmed below by the actual frozen native stamp.
    else:
        _row(value["identity"], source, index)
        source_frame = uint(source["recordedFrame"])
        _need(source["context"] == frame["context"] and source["recordingSegment"] == frame["recordingSegment"]
              and source_frame <= uint(frame["recordedFrame"]), "Energy input source is outside the point scope")
        _need(value["sampleState"] in ("Exact", "Carried"), "Missing energy input")
        if value["sampleState"] == "Exact":
            _need(source == value["requested"], "Exact energy source contradicts its request")
    if not static:
        _need(value["sampleState"] == "Exact" and value["requested"] == frame and source == frame,
              "Dynamic energy inputs require exact point samples")
    prop, code = _property(session, row, index, attribute, count)
    return ReadCheck(row.instance_id, prop.id, code, count, source_frame, tuple(value["value"]), identity_signature(row))


def _scalar_window(evidence, finding, role, session, index, cancelled):
    window = finding.data
    _need(role == window["role"], "Finding belongs to the other capture role")
    _need(not finding.reasons, "; ".join(finding.reasons))
    setup = {}
    for reference in window["supporting"]:
        observation = evidence.observations[reference]
        attribute = observation.data["attribute"]
        if attribute in (*_STATIC_ENERGY, "energyReferencePosition"):
            _need(attribute not in setup, "Ambiguous energy setup observations")
            setup[attribute] = observation
    _need(all(attribute in setup for attribute in (*_STATIC_ENERGY, "energyReferencePosition")), "Missing recorded energy setup/reference")
    reference = setup["energyReferencePosition"].data
    _need(reference["origin"] == "producer-log" and reference["source"] is None and reference["units"] == "m"
          and reference["value"] is not None and len(reference["value"]) == 3,
          "Energy reference must be an explicit analytic choice, not a claimed captured fact")
    points = sorted(window["points"], key=lambda point: uint(point["frame"]["recordedFrame"]))
    _need(len(points) >= 2 and len({uint(point["frame"]["recordedFrame"]) for point in points}) == len(points),
          "Energy window needs distinct before/after points")
    markers, energies = [], []
    for point in points:
        _check(cancelled)
        frame = point["frame"]
        row, scene = _row(window["identity"], frame, index)
        selected = {attribute: setup[attribute] for attribute in _STATIC_ENERGY}
        for ref in point["observations"]:
            observation = evidence.observations[ref]
            attribute = observation.data["attribute"]
            if attribute in ("globalPose", "linearVelocity", "angularVelocity"):
                _need(attribute not in selected, "Ambiguous dynamic energy observations")
                selected[attribute] = observation
        _need(all(attribute in selected for attribute in _ENERGY_FIELDS), "Missing dynamic recorded energy inputs")
        for attribute, observation in selected.items():
            expected_uid = row.scene_object_uid if attribute == "gravity" else row.object_uid
            _need(uint(observation.data["identity"]["objectId"]) == expected_uid, "Energy observation belongs to another body/scene")
        checks = tuple(_observation_check(observation, frame, session, index, static=attribute in _STATIC_ENERGY)
                       for attribute, observation in selected.items())
        marker = Marker(evidence.digest, finding.id, role, scene.id, uint(frame["recordedFrame"]), row.instance_id,
                        identity_signature(row), checks, window["measure"]["status"])
        view = session.presentation(scene.id, marker.frame, inspect_ids=sorted({check.instance_id for check in checks}),
            read_requests=[(check.instance_id, check.property_id, 0, check.field_count * (4 if check.type_code == 8 else 8)) for check in checks],
            include_evidence=True)
        reason = validate_marker_snapshot(view, marker)
        _need(not reason, reason)
        values = {attribute: observation.data["value"] for attribute, observation in selected.items()}
        energy = mechanical_energy(values["mass"][0], values["massSpaceInertiaTensor"], values["cMassLocalPose"],
                                   values["globalPose"], values["linearVelocity"], values["angularVelocity"], values["gravity"], reference["value"])
        expected = (point["components"]["translationalJ"], point["components"]["rotationalJ"], point["components"]["gravitationalJ"])
        _need(point["valueJ"] is not None and all(value is not None for value in expected), "Energy point is inconclusive")
        _need(all(math.isclose(a, b, rel_tol=1e-8, abs_tol=max(energy.roundoff, 1e-12)) for a, b in zip(energy.components, expected))
              and math.isclose(energy.total, point["valueJ"], rel_tol=1e-8, abs_tol=max(energy.roundoff, 1e-12)),
              "Energy calculation contradicts its verified recorded inputs")
        markers.append(marker);energies.append(energy)
    before, after = energies[0], energies[-1]
    _need(before.total > max(0, 10 * before.roundoff), "Energy window reference is too small")
    fraction = (after.total-before.total)/before.total
    measure = window["measure"]
    _need(math.isclose(fraction, measure["value"], rel_tol=1e-8, abs_tol=1e-10), "Relative energy change contradicts recorded points")
    warning = fraction > measure["tolerance"]
    _need((measure["status"] == "Warning") == warning and measure["status"] != "Inconclusive", "Analytic status contradicts supplied gain threshold")
    return tuple(markers)


def resolve_bindings(evidence, payload, private_session, cancelled=lambda: False):
    if payload is None:
        return BindingResult("no-finalized-payload", message="Save and open a finalized OVD capture to bind evidence")
    try:
        verify_payload(payload, cancelled)
    except (OSError, EvidenceError) as error:
        return BindingResult("changed-payload", message=str(error))
    roles = tuple(role for role, binding in evidence.captures.items() if binding.sha256 == payload.sha256
                  and (binding.byte_length is None or binding.byte_length == payload.byte_length))
    if not roles:
        missing = not any(binding.sha256 for binding in evidence.captures.values())
        return BindingResult("missing-binding" if missing else "mismatch", message="Capture content binding missing" if missing else "Capture hashes/lengths do not match")
    state = "matching-both" if len(roles) == 2 else "matching-" + roles[0]
    reasons, markers = {}, []
    try:
        index = _namespace(private_session, cancelled)
    except (EvidenceError, ValueError, RuntimeError) as error:
        return BindingResult("unsupported-format/domain", roles, message=str(error))
    for finding in evidence.findings:
        for role in roles:
            _check(cancelled)
            try:
                if finding.kind == "comparison":
                    markers.extend(_comparison(evidence, finding, role, private_session, index))
                else:
                    markers.extend(_scalar_window(evidence, finding, role, private_session, index, cancelled))
                reasons[(finding.id, role)] = ()
            except (EvidenceError, ValueError, IndexError, RuntimeError) as error:
                reasons[(finding.id, role)] = (str(error),)
    try:
        verify_payload(payload, cancelled)
    except (OSError, EvidenceError) as error:
        return BindingResult("changed-payload", message=str(error))
    return BindingResult(state, roles, tuple(markers), MappingProxyType(reasons))


def _read_json(path):
    path = Path(path)
    before = path.stat()
    _need(stat.S_ISREG(before.st_mode) and before.st_size <= MAX_BYTES, "Evidence must be a regular JSON file <=4 MiB")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        _need(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), "Evidence must be a regular JSON file")
        data = stream.read(MAX_BYTES + 1)
        _need(_identity(before) == _identity(os.fstat(stream.fileno())) == _identity(path.stat()), "Evidence file changed while reading")
    return parse_evidence(data)


@dataclass
class StoredSet:
    evidence: object
    binding: BindingResult


class EvidenceController:
    def __init__(self):
        self.sets, self.selected_set, self.selection = {}, None, None
        self.context, self.error, self.revision = DisplayContext(), "", 0
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="evidence")
        self._jobs, self._generation, self._closed = [], 0, False

    @property
    def pending(self):
        return bool(self._jobs)

    def _cancel_jobs(self):
        self._generation += 1
        for future, event, _, _ in self._jobs:
            event.set();future.cancel()

    def sync(self, context):
        changed = context.binding_key != self.context.binding_key
        self.context = context
        if changed:
            self._cancel_jobs();self.revision += 1
            for stored in self.sets.values():
                stored.binding = BindingResult("pending")
                self._submit(stored.evidence, None)

    def _submit(self, evidence, path):
        context, generation, event, known = self.context, self._generation, Event(), set(self.sets)
        def work():
            _check(event.is_set)
            parsed = _read_json(path) if path is not None else evidence
            _check(event.is_set)
            if path is not None and parsed.digest in known:
                return parsed, None
            if not context.capture_key or not context.payload or context.live:
                result = BindingResult("no-capture" if not context.capture_key else "no-finalized-payload", message="Save and open a finalized OVD capture to bind evidence")
            elif not context.ovd:
                result = BindingResult("unsupported-format/domain", message="Evidence navigation supports OVD captures")
            else:
                try:
                    verify_payload(context.payload, event.is_set)
                except (OSError, EvidenceError) as error:
                    result = BindingResult("changed-payload", message=str(error))
                else:
                    with tempfile.TemporaryDirectory(prefix="evidence-", dir=context.payload.path.parent) as temporary:
                        private = Session(str(context.payload.path), temporary)
                        try:
                            result = resolve_bindings(parsed, context.payload, private, event.is_set)
                        finally:
                            private = None
            return parsed, result
        if context.payload:
            future = submit_capture(self._executor, context.payload.path, work)
        else:
            future = self._executor.submit(work)
        self._jobs.append((future, event, generation, path is not None))

    def import_file(self, path, context):
        if self._closed:
            return
        self.sync(context)
        self._submit(None, path)

    def poll(self):
        remaining, changed = [], False
        for future, event, generation, is_import in self._jobs:
            if not future.done():
                remaining.append((future, event, generation, is_import));continue
            try:
                parsed, binding = future.result()
                if event.is_set() or generation != self._generation:
                    continue
                existing = self.sets.get(parsed.digest)
                if existing is not None:
                    if binding is not None:
                        existing.binding = binding
                else:
                    _need(len(self.sets) < 4 and parsed.node_count + sum(s.evidence.node_count for s in self.sets.values()) <= 100000,
                          "Stored evidence limit reached; remove a set before importing")
                    self.sets[parsed.digest] = StoredSet(parsed, binding)
                if is_import and self.selected_set != parsed.digest:
                    self.selected_set, self.selection = parsed.digest, None
                self.error = "";self.revision += 1;changed = True
            except CancelledError:
                pass
            except (OSError, ValueError, RuntimeError) as error:
                if not event.is_set() and generation == self._generation:
                    self.error = str(error);self.revision += 1;changed = True
        self._jobs = remaining
        return changed

    def select_set(self, digest):
        if digest in self.sets and digest != self.selected_set:
            self.selected_set, self.selection = digest, None;self.revision += 1

    def select(self, finding_id, role=None, frame=None):
        stored = self.sets.get(self.selected_set)
        if stored is not None and any(f.id == finding_id for f in stored.evidence.findings) and role in (None, "left", "right"):
            self.selection = finding_id, role, frame;self.revision += 1

    @property
    def markers(self):
        return tuple(marker for stored in self.sets.values() for marker in stored.binding.markers)

    @property
    def selected_marker(self):
        stored = self.sets.get(self.selected_set)
        if stored is None or self.selection is None:
            return None
        finding_id, role, frame = self.selection
        if role is None and len(stored.binding.roles) == 1:
            role = stored.binding.roles[0]
        if role is None:
            return None
        finding = next(f for f in stored.evidence.findings if f.id == finding_id)
        if frame is None and finding.kind == "scalar-window":
            frame = uint(finding.data["anchor"]["recordedFrame"])
        return next((m for m in stored.binding.markers if m.finding_id == finding_id and m.role == role
                     and (frame is None or m.frame == frame)), None)

    def remove(self, digest):
        self._cancel_jobs()
        self.sets.pop(digest, None)
        if self.selected_set == digest:
            self.selected_set, self.selection = next(iter(self.sets), None), None
        for stored in self.sets.values():
            if stored.binding.state == "pending":
                self._submit(stored.evidence, None)
        self.revision += 1

    def close(self):
        self._closed = True;self._cancel_jobs()
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._jobs.clear();self.sets.clear()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strict evidence preflight for Python/Qt completed application updates.

This module imports no frontend, graphics toolkit or native capture reader.
See compare_applications.md for the producer contract and quality review format.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from collections import Counter

ABS_TOL = REL_TOL = 1e-5
MISSING = object()
DISPLAY_MODES = ("none", "all", "selected", "unselected")
OVERLAYS = ("bounds", "contacts", "joints", "scene_queries", "coordinate_system", "velocity")


def number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def integer(value):
    return type(value) is int


def vector(length, *, nullable=False):
    return lambda value: isinstance(value, list) and len(value) == length and all(
        number(item) or nullable and item is None for item in value)


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def same(a, b):
    if number(a) and number(b):
        return math.isclose(a, b, rel_tol=REL_TOL, abs_tol=ABS_TOL)
    if type(a) is not type(b):
        return False
    if isinstance(a, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[key], b[key]) for key in a)
    return a == b


def preview(value):
    """Keep mismatched large native snapshots out of the diagnostic JSON."""
    if isinstance(value, (list, tuple)):
        if len(value) > 24:
            return {"length": len(value), "first": [preview(v) for v in value[:4]]}
        return [preview(v) for v in value]
    if isinstance(value, dict):
        if len(value) > 12:
            return {"length": len(value), "first": {str(k): preview(v) for k, v in list(value.items())[:4]}}
        return {str(k): preview(v) for k, v in value.items()}
    return value


class Audit:
    def __init__(self):
        self.issues = []
        self.issue_count = 0
        self.notes = []

    def issue(self, path, reason, **values):
        self.issue_count += 1
        if len(self.issues) < 100:
            self.issues.append(dict(path=path, reason=reason, **{k: preview(v) for k, v in values.items()}))

    def pair(self, path, a=MISSING, b=MISSING, valid=None, *, compare=True, exact=False):
        good = True
        for side, value in (("python", a), ("qt", b)):
            if value is MISSING:
                self.issue(path, "missing evidence", side=side)
                good = False
            elif valid and not valid(value):
                self.issue(path, "invalid or unsupported evidence", side=side, value=value)
                good = False
        if good and compare and not ((a == b) if exact else same(a, b)):
            self.issue(path, "mismatch", python=a, qt=b)
            return False
        return good


def value(case, report, key):
    return case.get(key, report.get(key, MISSING))


def legacy_settings(case, report, frontend):
    """Expose only recorded, unambiguous old fields; never fill defaults."""
    effective = value(case, report, "effective_settings")
    if isinstance(effective, dict):
        return effective
    raw = case.get("settings", {})
    if not isinstance(raw, dict):
        return {}
    result = {}
    names = ({key: key for key in ("face_culling", "planes", "geometry_scale", "plane_scale",
        "particle_size", "show_particles", "sleeping_objects", "trigger_shapes", "wireframe")}
        if frontend == "python" else {"face_culling": "faceCulling", "planes": "drawPlanes",
        "geometry_scale": "geomScale", "particle_size": "particleSize", "sleeping_objects": "sleepingObjects",
        "trigger_shapes": "triggerShapes", "wireframe": "wireframe", "transparency": "transparent",
        "transparency_alpha": "transparencyAmount"})
    for key, old in names.items():
        if old not in raw:
            continue
        item = raw[old]
        choices = DISPLAY_MODES if key in ("sleeping_objects", "trigger_shapes", "wireframe", "transparency") else (
            ("none", "normal", "inverse") if key == "face_culling" else
            ("auto", "none", "basis", "scene") if key == "planes" else None)
        if key == "face_culling" and frontend == "qt" and type(item) is bool:
            item = int(item)
        if choices and integer(item) and 0 <= item < len(choices):
            item = choices[item]
        result[key] = item
    if frontend == "qt" and isinstance(report.get("sizes"), dict):
        if "samples" in report["sizes"]:
            result["actual_msaa_samples"] = report["sizes"]["samples"]
    return result


def settings(audit, a, b):
    validators = {
        "face_culling": lambda v: v in ("none", "normal", "inverse"),
        "planes": lambda v: v in ("auto", "none", "basis", "scene"),
        "background_rgba": lambda v: vector(4)(v) and all(0 <= x <= 1 for x in v),
        "geometry_scale": lambda v: number(v) and v > 0,
        "plane_scale": lambda v: number(v) and v > 0,
        "particle_size": lambda v: number(v) and v >= 0,
        "show_particles": lambda v: type(v) is bool,
        "actual_msaa_samples": lambda v: integer(v) and v >= 0,
        "transparency_alpha": lambda v: number(v) and 0 <= v <= 1,
        **{key: lambda v: v in DISPLAY_MODES for key in
           ("sleeping_objects", "trigger_shapes", "wireframe", "transparency")}}
    for key, valid in validators.items():
        audit.pair("effective_settings." + key, a.get(key, MISSING), b.get(key, MISSING), valid,
                   exact=key == "actual_msaa_samples")
    for key in (*OVERLAYS, "orbit_center"):
        first, second = a.get("overlays", {}), b.get("overlays", {})
        first = first if isinstance(first, dict) else {}
        second = second if isinstance(second, dict) else {}
        valid = (lambda v: type(v) is bool) if key == "orbit_center" else lambda v: v in DISPLAY_MODES
        audit.pair("effective_settings.overlays." + key,
                   first.get(key, MISSING), second.get(key, MISSING), valid)


def ids(audit, path, a, b):
    valid = lambda v: isinstance(v, list) and all(integer(x) for x in v) and len(v) == len(set(v))
    if not audit.pair(path, a, b, valid, compare=False):
        return False
    return audit.pair(path, sorted(a), sorted(b), exact=True)


INSPECTOR_POLICY_FIELDS = ("coverage", "refresh_policy", "group_size", "max_array_to_load")


def inspector_workload(audit, path, data, side, workload, declared_policy):
    """Accept only observed closed arrays; backend read mechanisms may differ."""
    if not isinstance(data, dict):
        audit.issue(path, "missing inspector workload evidence", side=side)
        return {}
    validators = {
        "schema_version": lambda v: integer(v) and v == 1,
        "complete": lambda v: v is True,
        "coverage": lambda v: v == "closed-property-arrays",
        "refresh_policy": lambda v: v in ("current-frame", "frozen-at-pass-start"),
        "group_size": lambda v: integer(v) and 10 <= v <= 9999,
        "max_array_to_load": lambda v: integer(v) and 0 <= v <= 99999,
        "expanded_property_ranges": lambda v: isinstance(v, list) and not v,
        "array_controls": lambda v: isinstance(v, list),
        "backend_policy": lambda v: isinstance(v, dict) and bool(v),
        "diagnostics": lambda v: isinstance(v, list) and not v,
    }
    for key, valid in validators.items():
        if key not in data or not valid(data[key]):
            audit.issue(path + "." + key, "missing, malformed or unsupported inspector evidence",
                        side=side, value=data.get(key))
    refresh = data.get("refresh_policy")
    controls = data.get("array_controls")
    if isinstance(controls, list):
        for index, control in enumerate(controls):
            if not isinstance(control, dict) or control.get("expanded") is not False:
                audit.issue(f"{path}.array_controls[{index}]",
                            "missing closed-control observation or unsupported expansion", side=side)
            elif "observed_expansion_states" in control:
                states = control["observed_expansion_states"]
                if not isinstance(states, list) or any(v is not False for v in states):
                    audit.issue(f"{path}.array_controls[{index}].observed_expansion_states",
                                "native expansion observations contradict closed-array coverage", side=side)
    if refresh == "frozen-at-pass-start":
        if workload != "normal-playback" or declared_policy != refresh:
            audit.issue(path + ".refresh_policy", "frozen inspection requires explicit normal-playback policy", side=side)
    elif declared_policy is not MISSING and declared_policy != refresh:
        audit.issue(path + ".refresh_policy", "inspection declarations disagree", side=side)
    reads = data.get("optional_reads")
    if not isinstance(reads, dict):
        audit.issue(path + ".optional_reads", "missing observed read mechanism", side=side)
    else:
        mechanism = "owned-presentation" if side == "python" else "direct-native"
        if reads.get("mechanism") != mechanism:
            audit.issue(path + ".optional_reads.mechanism", "unsupported frontend read mechanism", side=side)
        for key in ("read_requests", "record_requests", "publication_count"):
            item = reads.get(key, MISSING)
            valid = (item is None if side == "qt" else
                     integer(item) and item >= 0 if key == "publication_count" else
                     isinstance(item, list) and not item)
            if not valid:
                audit.issue(path + ".optional_reads." + key,
                            "missing or unsupported observed reads for closed-array workload", side=side)
    return data


RECORD_KEYS = ("object_id", "shape_id", "geometry_id", "kind", "element_index")
RECORD_FIELDS = {
    **{key: integer for key in (*RECORD_KEYS, "scene_id", "actor_class", "shape_flags",
                               "actor_flags", "particle_radius_mode")},
    "kind": lambda v: integer(v) and 1 <= v <= 7,
    "element_index": lambda v: integer(v) and v >= 0,
    "sleeping": lambda v: type(v) is bool,
    "double_sided": lambda v: type(v) is bool,
    "transform": vector(16), "dimensions": vector(3), "bounds": vector(6, nullable=True),
    "override_color": lambda v: v is None or vector(4)(v),
    "resource_key": lambda v: v is None or isinstance(v, str),
}


def native_geometry(audit, path, data, side, scene_id):
    """Validate each source independently, including unmatched sample frames."""
    if not isinstance(data, dict):
        audit.issue(path, "missing native source geometry", side=side)
        return {}, {}
    for key, expected in (("schema_version", 1), ("complete", True), ("diagnostics", [])):
        if type(data.get(key)) is not type(expected) or data.get(key) != expected:
            audit.issue(path + "." + key, "missing or incomplete source evidence", side=side)
    resources, records = data.get("resources"), data.get("records")
    if not isinstance(resources, list) or not isinstance(records, list):
        audit.issue(path, "missing records/resources lists", side=side)
        return {}, {}
    resource_map = {}
    for resource in resources:
        if not isinstance(resource, dict) or not all(
                key in resource for key in ("key", "source_id", "vertex_count", "index_count",
                                             "vertices_sha256", "indices_sha256")):
            audit.issue(path + ".resources", "incomplete resource", side=side)
            continue
        valid = (all(integer(resource[key]) for key in ("source_id", "vertex_count", "index_count"))
            and resource["vertex_count"] >= 0 and resource["index_count"] >= 0
            and digest(resource["vertices_sha256"]) and digest(resource["indices_sha256"]))
        key = resource["key"]
        expected = f'{resource["source_id"]}:{resource["vertices_sha256"]}:{resource["indices_sha256"]}'
        if not valid or key != expected or key in resource_map:
            audit.issue(path + ".resources", "invalid or duplicate canonical resource", side=side)
            continue
        resource_map[key] = resource
    record_map = {}
    for index, record in enumerate(records):
        prefix = f"{path}.records[{index}]"
        if not isinstance(record, dict):
            audit.issue(prefix, "invalid native record", side=side)
            continue
        valid = True
        for field, validator in RECORD_FIELDS.items():
            if field not in record or not validator(record[field]):
                audit.issue(prefix + "." + field, "missing or invalid native field", side=side)
                valid = False
        if not valid:
            continue
        key = tuple(record[k] for k in RECORD_KEYS)
        if key in record_map:
            audit.issue(prefix, "duplicate native record identity", side=side, identity=key)
            continue
        record_map[key] = record
        if integer(scene_id) and record["scene_id"] > 0 and record["scene_id"] != scene_id:
            audit.issue(prefix + ".scene_id", "record belongs to a different owning scene", side=side)
        if record["resource_key"] is not None and record["resource_key"] not in resource_map:
            audit.issue(prefix, "missing referenced canonical resource", side=side, identity=key)
        if record["kind"] in (5, 6, 7) and record["resource_key"] is None:
            audit.issue(prefix + ".resource_key", "mesh/line/point source requires owned native resource", side=side)
    return record_map, resource_map


def compare_geometry(audit, path, a, b):
    audit.pair(path + ".resources", a[1], b[1], exact=True)
    audit.pair(path + ".record_identities", sorted(a[0]), sorted(b[0]), exact=True)
    for key in sorted(a[0].keys() & b[0].keys()):
        first, second = a[0][key], b[0][key]
        for field in RECORD_FIELDS:
            audit.pair(f"{path}.records[{key}].{field}", first[field], second[field],
                       exact=field not in ("transform", "dimensions", "bounds", "override_color"))


def declared_images(case, report):
    result = set()
    for name in ("screenshot_sha256", "viewport_screenshot_sha256"):
        if digest(case.get(name)):
            result.add(case[name])
    for container in (case, report):
        images = container.get("images")
        if isinstance(images, dict):
            result.update(v for v in images.values() if digest(v))
    return result


def quality_evidence(audit, review, python_hash, qt_hash, condition, directory, images):
    if not isinstance(review, dict):
        audit.issue("visual_quality", "missing local review of the exact pair's screenshots")
        return
    expected = {"schema_version": 1, "accepted": True, "condition": condition,
                "python_sha256": python_hash, "qt_sha256": qt_hash}
    for key, target in expected.items():
        current = review.get(key, MISSING)
        if (current is MISSING or type(current) is not type(target) or current != target
                or key.endswith("sha256") and not digest(current)):
            audit.issue("visual_quality." + key, "missing, rejected or stale review evidence")
    for key in ("scope", "reviewer"):
        if not isinstance(review.get(key), str) or not review[key].strip():
            audit.issue("visual_quality." + key, "missing review context")
    artifacts = review.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) < 2:
        audit.issue("visual_quality.artifacts", "review must reference images from both frontends")
        return
    verified = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str) or not digest(artifact.get("sha256")):
            audit.issue("visual_quality.artifacts", "invalid artifact reference")
            continue
        path = directory / artifact["path"]
        try:
            data = path.read_bytes()
        except OSError:
            data = b""
        if hashlib.sha256(data).hexdigest() != artifact["sha256"]:
            audit.issue("visual_quality.artifacts", "missing or changed reviewed image", artifact=str(path))
        elif not (data.startswith(b"\x89PNG\r\n\x1a\n") or data.startswith(b"\xff\xd8\xff")):
            audit.issue("visual_quality.artifacts", "artifact is not a PNG/JPEG image", artifact=str(path))
        elif artifact["sha256"] not in images[0] | images[1]:
            audit.issue("visual_quality.artifacts", "image is not declared by either compared report", artifact=str(path))
        else:
            verified.add(artifact["sha256"])
    for side, declared in zip(("python", "qt"), images):
        if not verified & declared:
            audit.issue("visual_quality.artifacts", "no verified reviewed image from this report", side=side)


def compare(python, qt, *, python_case=None, review=None, python_hash=None, qt_hash=None, review_directory=Path(".")):
    audit = Audit()
    if not isinstance(python, dict) or not isinstance(qt, dict):
        raise ValueError("Reports must be JSON objects")
    captures = python.get("captures", [])
    if not isinstance(captures, list) or not all(isinstance(case, dict) and isinstance(case.get("job"), dict) for case in captures):
        raise ValueError("Python captures must be objects with job metadata")
    if not isinstance(qt.get("job", {}), dict):
        raise ValueError("Qt job metadata must be an object")
    if python_case is None:
        candidates = [case for case in captures if case.get("job", {}).get("capture_sha256") == qt.get("capture_sha256")]
        if len(candidates) > 1:
            selection = "actor" if qt.get("job", {}).get("selected_id", 0) else "none"
            candidates = [case for case in candidates if case.get("job", {}).get("selection") == selection]
        if len(candidates) == 1:
            first = candidates[0]
        elif len(captures) == 1:
            first = captures[0]
        else:
            return {"comparable": False, "timing_ratios": None,
                    "issues": [{"path": "case", "reason": "ambiguous or missing Python case; supply --python-case"}]}
    elif not 0 <= python_case < len(captures):
        raise ValueError("--python-case is outside captures[]")
    else:
        first = captures[python_case]
    pair = ((first, python), (qt, qt))
    def field(key):
        return [value(case, report, key) for case, report in pair]
    def dictionaries(key):
        values = field(key)
        audit.pair(key, *values, lambda v: isinstance(v, dict), compare=False)
        return [v if isinstance(v, dict) else {} for v in values]
    jobs = [first.get("job", {}), qt.get("job", {})]
    audit.pair("capture_sha256", jobs[0].get("capture_sha256", MISSING),
               qt.get("capture_sha256", jobs[1].get("capture_sha256", MISSING)), digest, exact=True)
    audit.pair("scene_instance_id", *(job.get("scene_instance_id", MISSING) for job in jobs), integer, exact=True)
    ids(audit, "job.excluded_actor_ids", *(job.get("excluded_actor_ids", MISSING) for job in jobs))
    frame_sequences = [job.get("frames", MISSING) for job in jobs]
    audit.pair("frame_sequence", *frame_sequences,
        lambda v: isinstance(v, list) and bool(v) and all(integer(f) and f >= 0 for f in v), exact=True)
    audit.pair("workload", *field("workload"), lambda v: v in ("normal-playback", "paused-next"))
    for key in ("warmup_passes", "measured_passes"):
        audit.pair(key, *field(key), lambda v: integer(v) and v >= (1 if key == "measured_passes" else 0), exact=True)
    effective = [legacy_settings(case, report, frontend) for (case, report), frontend in zip(pair, ("python", "qt"))]
    settings(audit, *effective)
    inspectors = [inspector_workload(audit, "inspector_workload", case.get("inspector_workload"), side,
                  value(case, report, "workload"), value(case, report, "inspection_policy"))
                  for side, (case, report) in zip(("python", "qt"), pair)]
    for name in INSPECTOR_POLICY_FIELDS:
        audit.pair("inspector_workload." + name, *(data.get(name, MISSING) for data in inspectors), exact=True)
    policy = dictionaries("completion_policy")
    required = {"mode": "completed-update", "pre_sample_finish": True, "viewport_phase_finish": False,
        "post_ui_finish": True, "post_ui_context": "composition", "post_ui_context_verified": True}
    for key, expected in required.items():
        audit.pair("completion_policy." + key, *(v.get(key, MISSING) for v in policy),
                   lambda v: type(v) is type(expected) and v == expected)
    audit.pair("completion_policy.swap_interval_requested", *(v.get("swap_interval_requested", MISSING) for v in policy), integer, exact=True)
    gpu = dictionaries("gpu")
    for key in ("renderer", "vendor", "driver_version", "api"):
        audit.pair("gpu." + key, *(v.get(key, MISSING) for v in gpu), lambda v: isinstance(v, str) and bool(v.strip()))
    for side, evidence in zip(("python", "qt"), gpu):
        renderer = str(evidence.get("renderer", "")).lower()
        if any(name in renderer for name in ("llvmpipe", "softpipe", "software rasterizer", "swiftshader")):
            audit.issue("gpu.renderer", "hardware rendering required for this benchmark", side=side)
        if evidence.get("api", "glx") != "glx":
            audit.issue("gpu.api", "current comparison requires observed desktop GLX", side=side)
    surface = dictionaries("surface")
    condition = surface[1].get("condition", qt.get("condition"))
    audit.pair("surface.condition", *(v.get("condition", MISSING) for v in surface),
               lambda v: v in ("same-window", "same-geometry-fbo"))
    geometry_equal = surface[0].get("geometry_fbo_pixels", MISSING) == surface[1].get("geometry_fbo_pixels", None)
    for key in ("window_logical", "window_pixels", "geometry_fbo_pixels"):
        audit.pair("surface." + key, *(v.get(key, MISSING) for v in surface),
            lambda v: vector(2)(v) and all(x > 0 and (key == "window_logical" or integer(x)) for x in v),
            compare=(key == "geometry_fbo_pixels") == (condition == "same-geometry-fbo"), exact=True)
    audit.pair("surface.device_pixel_ratio", *(v.get("device_pixel_ratio", MISSING) for v in surface), lambda v: number(v) and v > 0)
    audit.pair("surface.gl_viewport", *(v.get("gl_viewport", MISSING) for v in surface),
        lambda v: vector(4)(v) and all(integer(x) for x in v) and min(v[2:]) > 0,
        compare=condition == "same-geometry-fbo", exact=True)
    for side, evidence in zip(("python", "qt"), surface):
        extent = evidence.get("geometry_fbo_pixels")
        if vector(2)(extent) and evidence.get("gl_viewport") != [0, 0, *extent]:
            audit.issue("surface.gl_viewport", "geometry viewport does not cover its actual framebuffer", side=side)
        logical, pixels, scale = (evidence.get(k) for k in ("window_logical", "window_pixels", "device_pixel_ratio"))
        if vector(2)(logical) and vector(2)(pixels) and number(scale):
            if not same([v * scale for v in logical], pixels):
                audit.issue("surface.device_pixel_ratio", "logical dimensions times scale differ from physical window", side=side)
    audit.pair("view_matrix", *field("view_matrix"), vector(16))
    audit.pair("projection_matrix", *field("projection_matrix"), vector(16), compare=condition == "same-geometry-fbo")
    if condition == "same-window":
        # Different aspect ratios legitimately change the projection. They do
        # not justify a different lens or clipping plane.
        for name in ("fov", "near"):
            a = first.get("camera", {})
            a = a.get(name, jobs[0].get(name, MISSING)) if isinstance(a, dict) else MISSING
            audit.pair("camera." + name, a, jobs[1].get(name, MISSING), lambda v: number(v) and v > 0)
    if (condition == "same-window" and not geometry_equal
            and all(vector(2)(evidence.get("geometry_fbo_pixels")) for evidence in surface)):
        audit.notes.append("Same window has unequal geometry pixels/projections: whole-application layout cost only, not equivalent viewport work.")
    if surface[0].get("window_pixels") != surface[1].get("window_pixels"):
        audit.notes.append("Whole-window composition pixel counts differ; this is not the same-window condition.")
    samples = [case.get("samples") if isinstance(case.get("samples"), list) else [] for case, _ in pair]
    maps = []
    geometries = {}
    for side, rows, frames, (case, report) in zip(("python", "qt"), samples, frame_sequences, pair):
        mapped = {}
        repeats = value(case, report, "measured_passes")
        selected_id = case.get("selected_id", case.get("job", {}).get("selected_id", MISSING))
        if not integer(selected_id) or selected_id < 0:
            audit.issue("selected_id", "missing declared nonnegative selection target", side=side)
        expected_selection = [selected_id] if integer(selected_id) and selected_id > 0 else []
        selection_mode = case.get("job", {}).get("selection")
        if selection_mode is not None and selection_mode != ("actor" if expected_selection else "none"):
            audit.issue("job.selection", "declared selection mode disagrees with selected target", side=side)
        if not isinstance(rows, list) or not rows:
            audit.issue("samples", "missing measured samples", side=side)
            rows = []
        for index, row in enumerate(rows):
            prefix = f"samples[{index}]"
            if not isinstance(row, dict):
                audit.issue(prefix, "sample must be an object", side=side)
                continue
            observed_inspector = inspector_workload(audit, prefix + ".inspector_workload",
                row.get("inspector_workload"), side, value(case, report, "workload"),
                value(case, report, "inspection_policy"))
            case_inspector = inspectors[0 if side == "python" else 1]
            for name in (*INSPECTOR_POLICY_FIELDS, "backend_policy"):
                if observed_inspector.get(name, MISSING) != case_inspector.get(name, MISSING):
                    audit.issue(prefix + ".inspector_workload." + name,
                                "per-sample inspector policy changed from case", side=side)
            geometries[id(row)] = native_geometry(audit, prefix + ".source_geometry",
                row.get("source_geometry"), side, case.get("job", {}).get("scene_instance_id"))
            submission = row.get("submitted_geometry")
            if ("submitted_shape_mapping_supported" in row
                    and row["submitted_shape_mapping_supported"] is not True):
                audit.issue(prefix + ".submitted_shape_mapping_supported", "producer reports unsupported native shape mapping", side=side)
            if not isinstance(submission, dict):
                audit.issue(prefix + ".submitted_geometry", "missing authoritative submission mapping evidence", side=side)
            else:
                for name, expected in (("schema_version", 1), ("complete", True),
                                       ("mapping_supported", True), ("diagnostics", [])):
                    if type(submission.get(name)) is not type(expected) or submission.get(name) != expected:
                        audit.issue(prefix + ".submitted_geometry." + name,
                                    "missing, unsupported or incomplete submitted geometry mapping", side=side)
            for name in ("selected_native_ids", "hidden_native_ids", "submitted_native_ids", "submitted_shape_ids"):
                current = row.get(name)
                if not isinstance(current, list) or not all(integer(v) for v in current) or len(current) != len(set(current)):
                    audit.issue(prefix + "." + name, "missing authoritative unique native ID list", side=side)
            if row.get("selected_native_ids") != expected_selection:
                audit.issue(prefix + ".selected_native_ids", "measured selection differs from declared target", side=side)
            if expected_selection:
                inspection = row.get("inspector_frame")
                if not integer(inspection) or inspection < 0:
                    audit.issue(prefix + ".inspector_frame", "missing valid inspected native frame", side=side)
                elif integer(row.get("frame")) and inspection > row["frame"]:
                    audit.issue(prefix + ".inspector_frame", "inspection is ahead of the measured native frame", side=side)
                elif (inspection != row.get("frame")
                      or observed_inspector.get("refresh_policy") == "frozen-at-pass-start"):
                    # Original Qt normal playback freezes the selected property
                    # panel. A producer must explicitly declare and bind that
                    # workload; arbitrary stale observations are not equivalent.
                    policy = value(case, report, "inspection_policy")
                    starts = value(case, report, "pass_starts")
                    matching = [v for v in starts if isinstance(v, dict) and v.get("repetition") == row.get("repetition")] if isinstance(starts, list) else []
                    start = matching[0] if len(matching) == 1 else None
                    if (observed_inspector.get("refresh_policy") != "frozen-at-pass-start"
                            or value(case, report, "workload") != "normal-playback" or policy != "frozen-at-pass-start"
                            or not start or start.get("inspector_frame") != inspection):
                        audit.issue(prefix + ".inspector_frame", "stale inspection lacks a declared matching pass-start freeze", side=side)
                    elif not any("frozen inspection" in note for note in audit.notes):
                        audit.notes.append("Normal playback includes frozen inspection at each declared pass start; it does not measure selected-property refresh on each frame.")
            source = geometries[id(row)][0].values()
            source_objects = {record["object_id"] for record in source}
            source_shapes = {record["shape_id"] for record in source if record["shape_id"] > 0}
            for name, known in (("submitted_native_ids", source_objects), ("submitted_shape_ids", source_shapes)):
                observed = row.get(name)
                if isinstance(observed, list) and all(integer(v) for v in observed) and not set(observed) <= known:
                    audit.issue(prefix + "." + name, "submitted identity absent from native source witness", side=side)
            visible = row.get("visible_shape_records")
            if not isinstance(visible, list) or not all(vector(4)(v) and all(integer(x) for x in v) for v in visible):
                audit.issue(prefix + ".visible_shape_records", "missing authoritative submitted native geometry rows", side=side)
            else:
                submitted = Counter(tuple(v) for v in visible)
                available = Counter(tuple(record[k] for k in RECORD_KEYS[:4]) for record in source)
                if submitted - available:
                    audit.issue(prefix + ".visible_shape_records", "submitted rows exceed native source occurrences", side=side)
                for name, expected in (("submitted_native_ids", {v[0] for v in visible}),
                                       ("submitted_shape_ids", {v[1] for v in visible if v[1] > 0})):
                    reported = row.get(name)
                    if isinstance(reported, list) and all(integer(v) for v in reported) and set(reported) != expected:
                        audit.issue(prefix + "." + name, "IDs disagree with submitted row witness", side=side)
            timing = "completed_iteration_ms" if side == "python" else "completed_update_ms"
            if not number(row.get(timing)) or row[timing] <= 0:
                audit.issue(prefix + "." + timing, "missing positive completed-update timing", side=side)
            if not integer(row.get("repetition")) or not integer(row.get("frame")):
                audit.issue(f"samples[{index}]", "missing explicit repetition/raw frame", side=side)
                continue
            key = (row["repetition"], row["frame"])
            if key in mapped:
                audit.issue("samples", "duplicate repetition/raw frame", side=side, identity=key)
            mapped[key] = row
        if isinstance(frames, list) and integer(repeats) and repeats >= 1:
            expected = [(r, frame) for r in range(repeats) for frame in frames]
            if list(mapped) != expected:
                audit.issue("samples", "measured sequence differs from declared repetitions/frames", side=side)
        maps.append(mapped)
    audit.pair("samples.identities", list(maps[0]), list(maps[1]), exact=True)
    for key in sorted(maps[0].keys() & maps[1].keys()):
        a, b = maps[0][key], maps[1][key]
        prefix = f"samples[{key[0]},{key[1]}]"
        for name in ("selected_native_ids", "submitted_native_ids", "submitted_shape_ids"):
            ids(audit, prefix + "." + name, a.get(name, MISSING), b.get(name, MISSING))
        if (a.get("hidden_native_ids") != b.get("hidden_native_ids")
                and isinstance(jobs[0].get("excluded_actor_ids"), list)
                and jobs[0]["excluded_actor_ids"] == jobs[1].get("excluded_actor_ids")
                and a.get("submitted_native_ids") == b.get("submitted_native_ids")
                and not any("cascade tag storage" in note for note in audit.notes)):
            audit.notes.append("Requested excluded actor roots and submitted actor IDs match, but raw hidden IDs differ in cascade tag storage. Raw tag equality is not required. Equivalent visible scope requires complete authoritative submitted shape/geometry mappings and matching native row multiplicities; missing mapping evidence still blocks ratios.")
        av, bv = a.get("visible_shape_records"), b.get("visible_shape_records")
        if all(isinstance(rows, list) and all(vector(4)(v) and all(integer(x) for x in v) for v in rows) for rows in (av, bv)):
            audit.pair(prefix + ".visible_shape_records", sorted(av), sorted(bv), exact=True)
        if a.get("selected_native_ids") or b.get("selected_native_ids"):
            audit.pair(prefix + ".inspector_frame", a.get("inspector_frame", MISSING), b.get("inspector_frame", MISSING), integer, exact=True)
        compare_geometry(audit, prefix + ".source_geometry", geometries[id(a)], geometries[id(b)])
        for name in ("effective_settings", "view_matrix", "projection_matrix", "surface", "gpu", "completion_policy"):
            for side, row, (case, report) in zip(("python", "qt"), (a, b), pair):
                if name in row and not same(row[name], value(case, report, name)):
                    audit.issue(prefix + "." + name, "per-sample evidence changed from case", side=side)
    quality_evidence(audit, review, python_hash, qt_hash, condition, Path(review_directory),
                     [declared_images(case, report) for case, report in pair])
    ratios = None
    if not audit.issue_count:
        a = statistics.median(row["completed_iteration_ms"] for row in samples[0])
        b = statistics.median(row["completed_update_ms"] for row in samples[1])
        if not number(a) or not number(b) or b <= 0 or not number(a / b):
            audit.issue("timing_ratios", "completed timing aggregate is not finite and positive")
        else:
            ratios = {"python_median_ms": a, "qt_median_ms": b, "python_over_qt": a / b,
                  "scope": "matched geometry pixels, different frontend composition" if condition == "same-geometry-fbo"
                           else "same-window whole-application layout cost; viewport work may differ"}
    return {"comparable": not audit.issue_count, "condition": condition,
        "equivalent_viewport_work": bool(not audit.issue_count and condition == "same-geometry-fbo"),
        "timing_ratios": ratios, "issue_count": audit.issue_count, "issues": audit.issues,
        "issues_truncated": audit.issue_count > len(audit.issues), "notes": audit.notes,
        "tolerances": {"absolute": ABS_TOL, "relative": REL_TOL},
        "diagnostics": {"effective_settings": dict(zip(("python", "qt"), effective)),
            "inspector_workload": {side: {"case": preview(data),
                "samples": [preview(row.get("inspector_workload")) for row in rows if isinstance(row, dict)]}
                for side, data, rows in zip(("python", "qt"), inspectors, samples)},
            "visibility": {side: {"requested_excluded_actor_ids": job.get("excluded_actor_ids"),
                "samples": [{key: row.get(key) for key in ("repetition", "frame", "hidden_native_ids",
                             "submitted_native_ids", "submitted_shape_ids", "submitted_geometry")}
                            for row in rows if isinstance(row, dict)]}
                for side, job, rows in zip(("python", "qt"), jobs, samples)},
            "synchronization_descriptions": [report.get("synchronization", report.get("scope")) for _, report in pair],
            "submitted_work": [{k: row[k] for k in ("repetition", "frame", "draw_calls", "draw_objects", "render_queue_counts") if k in row}
                               for rows in samples for row in rows if isinstance(row, dict)]},
        "limitation": "Passing applies only to the recorded condition; no feature parity, statistical significance or application FPS claim."}


def read_json(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON value {value}")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key}")
            result[key] = value
        return result
    def finite(value):
        result = float(value)
        if not number(result):
            reject(value)
        return result
    return json.loads(path.read_text(), parse_constant=reject, parse_float=finite, object_pairs_hook=unique)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("python", type=Path)
    parser.add_argument("qt", type=Path)
    parser.add_argument("--python-case", type=int)
    parser.add_argument("--quality-evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite existing comparison evidence")
    try:
        result = compare(read_json(args.python), read_json(args.qt), python_case=args.python_case,
            review=read_json(args.quality_evidence) if args.quality_evidence else None,
            python_hash=hashlib.sha256(args.python.read_bytes()).hexdigest(),
            qt_hash=hashlib.sha256(args.qt.read_bytes()).hexdigest(),
            review_directory=args.quality_evidence.parent if args.quality_evidence else Path("."))
    except (ValueError, TypeError, KeyError, OSError) as error:
        result = {"comparable": False, "timing_ratios": None,
                  "issues": [{"path": "input", "reason": str(error)}]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"comparable": result["comparable"], "timing_ratios": result["timing_ratios"],
                      "issues": result.get("issue_count", len(result["issues"])), "report": str(args.output)}))
    return 0 if result["comparable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

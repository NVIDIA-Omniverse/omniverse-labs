# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Observed inspector workload for closed-array application comparisons.

This module never asks the inspector to plan reads. read_requests() mutates its
budget/request state; the recorder must instead supply actual publication calls
and actual _array build observations. Expanded windows are diagnostic intent,
not proof of rendered bytes, and deliberately make this coverage incomplete.
"""
from ovd_ui.array_settings import MAX_ARRAY_WINDOW_BYTES, VISIBLE_ARRAY_WINDOW


def publication_requests(args, kwargs):
    """Copy exact public Session.presentation optional arguments before the call."""
    def argument(name, position):
        values = kwargs.get(name, args[position] if len(args) > position else ())
        return [list(value) for value in values]
    # Session.presentation(scene, frame, inspect_ids, read_requests, record_requests, ...).
    return dict(read_requests=argument("read_requests", 3),
                record_requests=argument("record_requests", 4))


def array_control(inspector, identity, prop, count, *, stride=0, key=None):
    """Observe a control just built by the production PropertyInspector._array."""
    key = (identity, prop.id) if key is None else key
    policy = inspector.array_settings
    allowed = policy.allows_expansion(count)
    start, length = policy.window(count, inspector.array_positions.get(key, 0),
                                  element_bytes=stride or 1)
    expanded = bool(allowed and count > 0 and length > 0 and key in inspector.array_expanded)
    return dict(key=list(key), total=count, expanded=expanded, expansion_allowed=allowed,
                requested_first=start if expanded else None,
                requested_count=length if expanded else None)


def inspector_workload(app, controls, publications):
    """Serialize policy plus already observed work; no GUI/native reads occur here."""
    policy = app.property_inspector.array_settings
    observed = controls is not None
    controls = [] if controls is None else controls
    expanded = [dict(item) for item in controls if item["expanded"]]
    reads = [list(value) for call in publications for value in call["read_requests"]]
    records = [list(value) for call in publications for value in call["record_requests"]]
    diagnostics = []
    if not observed:
        diagnostics.append("No actual property inspector build was observed")
    if app.object_model.array_settings != policy:
        diagnostics.append("Hierarchy and property array policies disagree")
    if expanded:
        diagnostics.append("Expanded property arrays are outside this benchmark coverage")
    if reads or records:
        diagnostics.append("Optional native windows/records are outside closed-array coverage")
    return dict(schema_version=1, complete=not diagnostics, diagnostics=diagnostics,
                coverage="closed-property-arrays", refresh_policy="current-frame",
                group_size=policy.group_size, max_array_to_load=policy.max_array_to_load,
                expanded_property_ranges=expanded, array_controls=list(controls),
                optional_reads=dict(mechanism="owned-presentation", read_requests=reads,
                    record_requests=records, publication_count=len(publications)),
                backend_policy=dict(array_grouping="bounded-logical",
                    maximum_leaf_elements=VISIBLE_ARRAY_WINDOW, maximum_window_bytes=MAX_ARRAY_WINDOW_BYTES,
                    threshold_scope="all-supported-property-expansion",
                    copy_policy="owned-previews-and-explicit-windows"))

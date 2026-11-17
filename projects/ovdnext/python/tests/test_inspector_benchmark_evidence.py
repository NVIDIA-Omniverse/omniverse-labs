# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evidence serialization and actual native request observation; no GUI launch."""
import ast
import importlib.util
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace as State

import pytest

from ovd_ui.array_settings import ArrayDisplaySettings


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("inspector_evidence", ROOT / "python/benchmarks/inspector_evidence.py")
evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)


def app(policy=ArrayDisplaySettings()):
    inspector = State(array_settings=policy, array_positions={}, array_expanded=set())
    # Calling the production planner again would change budgets. Evidence must
    # never depend on it, even when a control is already expanded.
    def unexpected(*args):
        raise AssertionError("Evidence called the stateful read planner")
    inspector.read_requests = unexpected
    return State(property_inspector=inspector, object_model=State(array_settings=policy))


def test_closed_array_position_is_not_claimed_as_an_expanded_window():
    target = app()
    target.property_inspector.array_positions[(15, 649)] = 30
    control = evidence.array_control(target.property_inspector, 15, State(id=649), 654, stride=16)
    result = evidence.inspector_workload(target, [control], [])
    assert result["complete"] and result["expanded_property_ranges"] == []
    assert result["array_controls"][0]["requested_first"] is None
    assert result["group_size"] == 10 and result["max_array_to_load"] == 1000


@pytest.mark.parametrize("group,total,stride,first,count", [(10, 654, 16, 30, 10), (9999, 200, 16, 37, 64),
                                                          (10, 654, 3 * 1024 * 1024, 37, 2)])
def test_expanded_intent_uses_bounded_policy_but_rejects_unverified_visible_bytes(group,total,stride,first,count):
    target = app(ArrayDisplaySettings(group, 1000))
    target.property_inspector.array_positions[(15, 649)] = 37
    target.property_inspector.array_expanded.add((15, 649))
    control = evidence.array_control(target.property_inspector, 15, State(id=649), total, stride=stride)
    result = evidence.inspector_workload(target, [control], [])
    assert not result["complete"]
    assert result["expanded_property_ranges"][0]["requested_first"] == first
    assert result["expanded_property_ranges"][0]["requested_count"] == count


def test_zero_threshold_suppresses_nonempty_control_even_with_retained_intent():
    target = app(ArrayDisplaySettings(10, 0))
    target.property_inspector.array_expanded.add((15, 649))
    control = evidence.array_control(target.property_inspector, 15, State(id=649), 654, stride=16)
    result = evidence.inspector_workload(target, [control], [])
    assert result["complete"] and not control["expanded"] and not control["expansion_allowed"]
    assert result["max_array_to_load"] == 0


def test_missing_actual_ui_build_and_mismatched_policy_are_incomplete():
    target = app()
    assert not evidence.inspector_workload(target, None, [])["complete"]
    target.object_model.array_settings = ArrayDisplaySettings(20, 1000)
    assert not evidence.inspector_workload(target, [], [])["complete"]


def test_exact_positional_and_keyword_request_arguments_are_owned_copies():
    reads = [[15, 649, 64, 16]]
    first = evidence.publication_requests((6, 1, [], reads, [(6, 401, 4)]), {})
    second = evidence.publication_requests((6, 1), {"read_requests": reads, "record_requests": [(6, 401, 4)]})
    reads[0][2] = 999
    assert first == second == dict(read_requests=[[15, 649, 64, 16]], record_requests=[[6, 401, 4]])


def recorder(native):
    # Compile the actual benchmark Recorder without importing/launching OVUI or
    # make_job. Only its public Session wrapper is exercised below.
    path = ROOT / "python/benchmarks/application_playback.py"
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "Recorder")
    namespace = dict(_pvd_native=native, publication_requests=evidence.publication_requests, perf_counter=perf_counter)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    result = namespace["Recorder"]()
    result.current = {}
    result.wrap_native_presentation()
    return result


def test_actual_native_window_and_record_requests_are_observed_without_replanning(tmp_path):
    native = pytest.importorskip("_pvd_native")
    session = native.Session(str(ROOT / "test/data/cloth_physx34.pxd2"), str(tmp_path))
    scene = session.scenes()[0].id
    rec = recorder(native)
    try:
        view = session.presentation(scene, 1, read_requests=[(15, 649, 64, 16)], record_requests=[(6, 401, 4)])
        assert view.has_read(15, 649, 64, 16) and len(view.read(15, 649, 64, 16)) == 16
        assert view.array_record(6, 401, 4).fields
        result = evidence.inspector_workload(app(), [], rec.current["presentation_requests"])
        assert not result["complete"]
        assert result["optional_reads"] == dict(mechanism="owned-presentation", publication_count=1,
            read_requests=[[15, 649, 64, 16]], record_requests=[[6, 401, 4]])
    finally:
        rec.restore()
        del session


def test_actual_native_budget_rejection_then_core_retry_retains_failed_optional_attempt(tmp_path):
    native = pytest.importorskip("_pvd_native")
    session = native.Session(str(ROOT / "test/data/cloth_physx34.pxd2"), str(tmp_path))
    scene = session.scenes()[0].id
    rec = recorder(native)
    try:
        with pytest.raises(ValueError, match="8 MiB"):
            session.presentation(scene, 1, read_requests=[(15, 649, 0, 8 * 1024 * 1024 + 1)])
        view = session.presentation(scene, 1)
        assert view.ready and view.request_applied and view.frame == 1
        result = evidence.inspector_workload(app(), [], rec.current["presentation_requests"])
        assert result["optional_reads"]["publication_count"] == 2
        assert result["optional_reads"]["read_requests"] == [[15, 649, 0, 8 * 1024 * 1024 + 1]]
        assert not result["complete"]  # A successful empty retry cannot disguise extra work.
    finally:
        rec.restore()
        del session

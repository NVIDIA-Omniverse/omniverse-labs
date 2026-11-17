# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact source-copy semantics and real/synthetic GC observation without a GUI."""
import gc
import copy
import importlib.util
from pathlib import Path
import textwrap
import sys

import pytest
from test_application_presentation import app


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("refresh_cost_evidence", ROOT / "python/benchmarks/refresh_cost_evidence.py")
evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)
SOURCE = ROOT / "python/ovd_ui/application.py"


def graph_functions(original):
    generated, proof = evidence.instrument_graph(original)
    start = original.index(b"            children=dict(native_children)")
    stop = original.index(b"            query_ids=[record.query_id")
    added = sum(len(bytes.fromhex(edit["addition_hex"])) for edit in proof["insertions"])
    functions, calls = [], []
    for body in (original[start:stop], generated[start:stop + added]):
        namespace = {evidence.BEGIN: lambda: calls.append("begin"), evidence.END: lambda: calls.append("end")}
        source = "def graph(snapshot,native_children,native_owners):\n" + textwrap.indent(textwrap.dedent(body.decode()), "    ") + "    return children,owners\n"
        exec(compile(source, "<actual-application-graph-span>", "exec"), namespace)
        functions.append(namespace["graph"])
    return functions, calls, generated, proof


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_instrumentation_reverses_every_byte_and_preserves_ast(newline):
    original = SOURCE.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", newline)
    _, _, generated, proof = graph_functions(original)
    restored = generated
    for edit in reversed(proof["insertions"]):
        offset, addition = edit["offset"], bytes.fromhex(edit["addition_hex"])
        assert restored[offset:offset + len(addition)] == addition
        restored = restored[:offset] + restored[offset + len(addition):]
    assert restored == original
    assert proof["ast_equivalence_verified"] and proof["byte_reversal_verified"]


@pytest.mark.parametrize("name", ["multi_geom.ovd", "ue4_shooter.pxd2"])
def test_actual_graph_span_matches_on_owned_native_scene_and_preserves_adapter(name,tmp_path):
    native = pytest.importorskip("_pvd_native")
    from ovd_ui.adapters import PvdDataAdapter
    source = native.Session(str(ROOT / "test/data" / name), str(tmp_path))
    # The first PXD2 stream is not necessarily the populated physics scene.
    from capture_semantics import populated_scene
    scene = populated_scene(source, 30)
    view = source.presentation(scene.id, 30)
    scene = next(item for item in view.scenes() if item.id == scene.id)
    snapshot = view.geometry_snapshot()
    assert sum(owner == scene.instance_id for owner in snapshot.scene_ids) >= (1000 if name.endswith("pxd2") else 8)
    adapter = PvdDataAdapter(view)
    children, owners = adapter.ownership_graph()
    before = copy.deepcopy((children, owners))
    functions, calls, _, _ = graph_functions(SOURCE.read_bytes())
    expected = functions[0](snapshot, children, owners)
    actual = functions[1](snapshot, children, owners)
    assert actual == expected and calls == ["begin", "end"]
    assert (children, owners) == before
    assert next(item for item in source.scenes() if item.id == scene.id).current_frame == view.frame == 30


def test_exceptions_propagate_unchanged_and_never_claim_a_complete_span():
    functions, calls, _, _ = graph_functions(SOURCE.read_bytes())
    class Broken:
        def __iter__(self):
            raise RuntimeError("native membership fixture failure")
    for function in functions:
        with pytest.raises(RuntimeError, match="native membership fixture failure"):
            function(None, Broken(), {})
    assert calls == ["begin"]


def previous_graph(snapshot, native_children, native_owners):
    """Frozen pre-optimization membership rules from Application at 39e9b3f."""
    children={key:set(value) for key,value in native_children.items()}
    owners={key:set(value) for key,value in native_owners.items()}
    if snapshot is not None:
        for actor,shape,geometry,scene in zip(snapshot.object_ids,snapshot.shape_ids,snapshot.geometry_ids,snapshot.scene_ids):
            if shape>0:children.setdefault(int(actor),set()).add(int(shape))
            if geometry>0 and shape>0:children.setdefault(int(shape),set()).add(int(geometry))
            if scene>0:
                for identity in (actor,shape,geometry):
                    if identity>0:owners.setdefault(int(identity),set()).add(int(scene))
    return children,owners


@pytest.mark.parametrize('filename',['multi_geom.ovd','ue4_shooter.pxd2','cloth_physx34.pxd2',
                                    'test_scene.ovd','resample_two_segment.ovd'])
def test_shared_adjacencies_match_previous_rules_across_real_native_frames(filename,tmp_path):
    native=pytest.importorskip('_pvd_native')
    from ovd_ui.adapters import PvdDataAdapter
    session=native.Session(str(ROOT/'test/data'/filename),str(tmp_path))
    functions,_,_,_=graph_functions(SOURCE.read_bytes())
    adapter=None
    for scene in session.scenes():
        for frame in sorted({1,min(2,scene.frame_count-1),min(30,scene.frame_count-1),min(31,scene.frame_count-1)}):
            if frame>=scene.frame_count:continue
            view=session.presentation(scene.id,frame)
            assert view.ready and view.request_applied
            if adapter is None:adapter=PvdDataAdapter(view)
            else:adapter.session=view;adapter.reload()
            children,owners=adapter.ownership_graph()
            assert all(type(v) is tuple for v in children.values())
            assert all(type(v) is frozenset for v in owners.values())
            before=copy.deepcopy((children,owners))
            snapshot=view.geometry_snapshot()
            expected=previous_graph(snapshot,children,owners)
            actual=functions[0](snapshot,children,owners)
            assert tuple({k:set(v) for k,v in graph.items()} for graph in actual)==expected
            assert (children,owners)==before


def test_geometry_supplements_missing_edges_without_mutating_shared_owners():
    from types import SimpleNamespace
    functions,_,_,_=graph_functions(SOURCE.read_bytes())
    children={1:(2,),2:(3,),10:(1,)}
    owners={1:frozenset((10,)),2:frozenset((10,)),3:frozenset((10,))}
    snapshot=SimpleNamespace(object_ids=[1,1,1,-2],shape_ids=[2,4,2,0],
                             geometry_ids=[3,5,3,-1],scene_ids=[10,11,12,-1])
    expected=previous_graph(snapshot,children,owners)
    before=copy.deepcopy((children,owners))
    actual=functions[0](snapshot,children,owners)
    assert tuple({k:set(v) for k,v in graph.items()} for graph in actual)==expected
    assert (children,owners)==before
    # Unchanged memberships are safely shared; additions replace one value.
    assert actual[0][2] is children[2] and actual[0][1] is not children[1]
    assert actual[1][1]==frozenset((10,11,12))


def test_many_shapes_and_repeated_geometry_preserve_every_native_membership():
    from types import SimpleNamespace
    functions,_,_,_=graph_functions(SOURCE.read_bytes())
    original=tuple(range(100,1100))
    shapes=list(range(100,1600))*2
    children={1:original}
    owners={1:frozenset((10,))}
    snapshot=SimpleNamespace(object_ids=[1]*len(shapes),shape_ids=shapes,
                             geometry_ids=[0]*len(shapes),scene_ids=[10]*len(shapes))
    actual=functions[0](snapshot,children,owners)
    assert tuple({k:set(v) for k,v in graph.items()} for graph in actual)==previous_graph(snapshot,children,owners)
    assert children[1] is original and len(children[1])==1000


def test_changed_ambiguous_or_already_instrumented_source_is_rejected():
    original = SOURCE.read_bytes()
    generated, _ = evidence.instrument_graph(original)
    with pytest.raises(ValueError, match="already occur"):
        evidence.instrument_graph(generated)
    with pytest.raises(ValueError, match="absent or duplicated"):
        evidence.instrument_graph(original.replace(b"query_ids=[record.query_id", b"query_ids=list(record.query_id"))
    with pytest.raises(ValueError, match="statement boundary"):
        evidence.instrument_graph(original.replace(b"            owners=dict(native_owners)", b"            unexpected=1\n            owners=dict(native_owners)"))


def clocked(max_events=200000):
    now = [0.]
    observer = evidence.GcObservation(clock=lambda: now[0], max_events=max_events)
    def event(at, phase, generation=0):
        now[0] = at
        observer._event(phase, dict(generation=generation, collected=3 if phase == "stop" else 0, uncollectable=0))
    return observer, event


def test_gc_overlap_clips_cross_boundary_collections_and_does_not_sum_nested_phases():
    observer, event = clocked()
    event(.5, "start"); event(2., "stop")
    event(3., "start", 2); event(6., "stop", 2)
    result = observer.window(1., 5., [("graph", 1.5, 3.5), ("refresh", 1., 4.), ("ui_wait", 4., 5.)])
    assert result["complete"] and result["overlap_ms"] == pytest.approx(3000)
    assert result["overlap_by_phase_ms"] == dict(graph=1000., refresh=2000., ui_wait=1000.)
    assert [(c["generation"], c["overlap_ms"]) for c in result["collections"]] == [(0, 1000.), (2, 2000.)]


def test_incomplete_unpaired_and_dropped_gc_events_are_explicit():
    observer, event = clocked(2)
    event(1., "start"); event(2., "stop")
    event(3., "start"); event(4., "stop")
    result = observer.window(0., 5.)
    assert not result["complete"] and result["dropped_events"] == 2
    assert len(observer.summary()["events"]) == 2 and not observer.summary()["complete"]
    observer, event = clocked()
    event(1., "start")
    assert observer.window(0., 2.)["unfinished_collections"]
    assert not observer.summary()["complete"]
    observer, event = clocked()
    event(1., "stop")
    assert observer.summary()["protocol_errors"] == 1 and not observer.summary()["complete"]


def test_real_gc_callbacks_preserve_controls_and_other_observers():
    before, callbacks = evidence.GcObservation.controls(), list(gc.callbacks)
    observer = evidence.GcObservation()
    extra = lambda phase, info: None
    observer.start()
    gc.callbacks.append(extra)
    try:
        cycle = []; cycle.append(cycle); del cycle
        gc.collect()  # Test-only trigger; the benchmark never forces collection.
        observer.stop()
        assert evidence.GcObservation.controls() == before
        assert gc.callbacks == callbacks + [extra]
        summary = observer.summary()
        assert summary["complete"] and summary["normal_gc_unchanged"]
        assert any(event["phase"] == "stop" and event["collected"] > 0 for event in summary["events"])
    finally:
        observer.stop()
        gc.callbacks.remove(extra)


def test_changed_gc_controls_reject_attribution_without_restoring_or_overriding_them(monkeypatch):
    observer, _ = clocked()
    changed = {**observer.before, "thresholds": [1, 2, 3]}
    monkeypatch.setattr(observer, "controls", lambda: changed)
    assert not observer.window(0., 1.)["complete"]
    assert not observer.summary()["normal_gc_unchanged"]


def test_actual_instrumented_application_refresh_records_exactly_one_native_graph_span(app,tmp_path,monkeypatch):
    native = pytest.importorskip("_pvd_native")
    from test_inspector_benchmark_evidence import recorder
    rec = recorder(native)
    rec.gc_observation = evidence.GcObservation()
    rec.current = None  # Initial display/setup is outside the sampled update.
    generated, _ = evidence.instrument_graph(SOURCE.read_bytes())
    copied = tmp_path / "application_observed.py"
    copied.write_bytes(generated)
    spec = importlib.util.spec_from_file_location("ovd_ui.test_cost_application", copied)
    module = importlib.util.module_from_spec(spec)
    module._pvd_benchmark_graph_begin = rec.graph_begin
    module._pvd_benchmark_graph_end = rec.graph_end
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(app, "_refresh", module.Application._refresh.__get__(app))
    try:
        path = ROOT / "test/data/multi_geom.ovd"
        source = native.Session(str(path), str(tmp_path / "capture"))
        assert app._display(path, source)
        rec.current = {}
        assert app._refresh(frame=2)
        assert app.frame == app.presentation.frame == 2
        assert rec.current["graph_preparation_ms_calls"] == 1
        assert rec.current["native_presentation_ms_calls"] == 1
        interval = next(i for i in rec.current["_phase_intervals"] if i[0] == "graph_preparation_ms")
        assert interval[2] >= interval[1] and rec.graph_started is None
        assert app.renderer.snapshot is app.presentation.geometry_snapshot()
    finally:
        rec.restore()


@pytest.mark.parametrize("observer_enabled", [False, True])
def test_both_modes_record_owned_gc_controls_without_changing_them(observer_enabled,monkeypatch):
    before, callbacks = evidence.GcObservation.controls(), list(gc.callbacks)
    def forbidden_collection(*args, **kwargs):
        raise AssertionError("Evidence must never force GC")
    monkeypatch.setattr(gc, "collect", forbidden_collection)
    observer = evidence.GcObservation() if observer_enabled else None
    try:
        if observer:
            observer.start()
            observer.stop()
        result = evidence.gc_control_evidence(before)
        assert result["unchanged"] and result["before"] == result["after"] == before
        assert gc.callbacks == callbacks
        before["thresholds"][0] += 1
        assert result["before"] == result["after"]  # Owned, not borrowed input.
    finally:
        if observer:
            observer.stop()


def test_controls_only_evidence_detects_external_change_without_restoring_it():
    before, callbacks = evidence.GcObservation.controls(), list(gc.callbacks)
    thresholds = gc.get_threshold()
    changed = (thresholds[0] + 123, *thresholds[1:])
    try:
        gc.set_threshold(*changed)  # Test-only external change.
        result = evidence.gc_control_evidence(before)
        assert not result["unchanged"] and result["after"]["thresholds"] == list(changed)
        assert gc.get_threshold() == changed and gc.callbacks == callbacks
    finally:
        gc.set_threshold(*thresholds)

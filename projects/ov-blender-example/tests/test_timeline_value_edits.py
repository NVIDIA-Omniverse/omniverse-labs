# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace
from types import SimpleNamespace

from ovrtx_blender_example import timeline_value_edits as timeline
from ovrtx_blender_example.interactive_edit_planner import InteractiveEdit, EditShape, DataAuthority, edit_location


def _edit(value=1.0):
    return InteractiveEdit(shape=EditShape.VALUE, data_authority=DataAuthority.VIEW,
        **edit_location(usd_prim_path="/World/Animated", usd_attribute="inputs:exposure",
            provenance={"blender_id_kind": "LIGHT", "blender_session_uid": 4}), value=value)


def test_timeline_cache_deduplicates_only_accepted_values_and_resets_generation():
    cache = timeline.TimelineValueCache()
    edit = _edit()
    assert cache.changed((1, 0), (edit,)) == (edit,)
    assert cache.changed((1, 0), (edit,)) == (edit,)
    cache.accepted(edit)
    assert cache.changed((1, 0), (edit,)) == ()
    changed = _edit(2.0)
    assert cache.changed((1, 0), (changed,)) == (changed,)
    assert cache.changed((2, 0), (edit,)) == (edit,)


def test_timeline_sampling_has_no_selection_or_geometry_and_keeps_actual_depsgraph(monkeypatch):
    obj = SimpleNamespace(type="MESH", material_slots=(), animation_data=object())
    graph = object()
    seen = {}
    def build(proxy, **kwargs):
        seen.update(proxy=proxy, kwargs=kwargs)
        return (_edit(), replace(_edit(), shape=EditShape.TOPOLOGY),
                replace(_edit(), data_authority=DataAuthority.SIM))
    monkeypatch.setattr(timeline, "build_interactive_edits_from_depsgraph", build)
    edits = timeline.evaluated_value_edits(SimpleNamespace(objects=(obj,), world=None), graph, object())
    assert len(edits) == 1
    assert edits[0].provenance["evaluation_reason"] == "timeline"
    assert "selection_resolution" not in edits[0].provenance
    assert seen["proxy"].evaluated_depsgraph is graph
    assert seen["proxy"].updates[0].id is obj
    assert seen["proxy"].updates[0].is_updated_geometry is False


def test_timeline_skips_static_objects_but_keeps_constraints_drivers_and_animated_parents(monkeypatch):
    static = SimpleNamespace(type="MESH", material_slots=())
    parent = SimpleNamespace(type="EMPTY", animation_data=object(), material_slots=())
    child = SimpleNamespace(type="MESH", parent=parent, material_slots=())
    constrained = SimpleNamespace(type="MESH", constraints=(object(),), material_slots=())
    driven_light = SimpleNamespace(type="LIGHT", data=SimpleNamespace(animation_data=object()), material_slots=())
    seen = []
    def build(proxy, **kwargs):
        seen.extend(update.id for update in proxy.updates)
        return ()
    monkeypatch.setattr(timeline, "build_interactive_edits_from_depsgraph", build)
    timeline.evaluated_value_edits(SimpleNamespace(objects=(static,parent,child,constrained,driven_light), world=None), object(), object())
    assert all(candidate is not static for candidate in seen)
    assert {id(candidate) for candidate in seen} == {id(parent),id(child),id(constrained),id(driven_light)}

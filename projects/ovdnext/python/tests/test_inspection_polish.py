# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regressions from inspecting the April pyramid capture in the release viewer."""
from pathlib import Path
import struct

import pytest
from _pvd_native import Session

from ovd_ui.adapters import PvdDataAdapter
from ovd_ui.models import ObjectTreeModel
from test_property_context import display
from test_application_presentation import app

ROOT = Path(__file__).resolve().parents[2]
PYRAMID = ROOT / 'test/data/2026_04_01_15_21_33_0000019f4952a142.ovd'


def pyramid_view(tmp_path):
    session = Session(str(PYRAMID), str(tmp_path))
    scene = session.scenes()[0]
    session.seek(scene.id, 111)
    view = session.presentation(scene.id, 111, inspect_ids=[scene.instance_id], overlay_mask=32)
    return session, scene, PvdDataAdapter(view)


@pytest.mark.parametrize('name,count,stride', [
    ('pairsContactNormals', 584, 12), ('pairsContactPoints', 584, 12),
    ('pairsContactCounts', 146, 4), ('pairsContactSeparations', 584, 4),
])
def test_populated_contact_arrays_have_real_record_counts(tmp_path, name, count, stride):
    session, scene, adapter = pyramid_view(tmp_path)
    nodes, _ = adapter.hierarchy()
    prop = adapter.property(scene.instance_id, name)
    node = next(n for n in nodes.values() if n.source_instance_id == scene.instance_id and n.property_id == prop.id)
    assert node.byte_count == count * stride
    assert node.element_count == count
    assert node.element_size == stride


def test_contact_vector_tree_record_reads_one_bounded_vector(tmp_path):
    session, scene, adapter = pyramid_view(tmp_path)
    model = ObjectTreeModel(page_size=1000)
    model.populate(adapter)
    prop = adapter.property(scene.instance_id, 'pairsContactNormals')
    node = next(n for n in model._specs.values() if n.source_instance_id == scene.instance_id and n.property_id == prop.id)
    assert node.element_count == 584
    raw = adapter.array_record(scene.instance_id, prop.id, 0, node.element_size)
    assert len(raw) == 12
    assert sum(v*v for v in struct.unpack('<3f', raw)) == pytest.approx(1)


def test_primitive_record_requests_raw_window_not_legacy_structure():
    from types import SimpleNamespace
    from ovd_ui.array_layout import inspection_requests
    item = SimpleNamespace(class_name='PxVec3', source_instance_id=3,
                           property_id=77, element_index=7, element_size=12)
    assert inspection_requests(item) == ([(3, 77, 84, 12)], [])
    item.class_name = 'Contact'
    assert inspection_requests(item) == ([], [(3, 77, 7)])


def test_dangling_material_reference_is_visible_without_a_fake_object(tmp_path):
    session, scene, adapter = pyramid_view(tmp_path)
    nodes, _ = adapter.hierarchy()
    shapes = [i.id for i in adapter.instances.values() if adapter.class_name(i.id) == 'PxShape']
    assert len(shapes) == 57
    for shape in shapes:
        group = nodes[f'group:object:{shape}:materials']
        assert len(group.children) == 1
        missing = nodes[group.children[0]]
        assert missing.kind == 'unresolved'
        assert 'material' in missing.name.lower()
        assert missing.instance_id is None
        assert missing.source_instance_id == shape
        assert missing.element_index == 0


def test_unresolved_row_never_requests_a_legacy_record():
    from types import SimpleNamespace
    from ovd_ui.array_layout import inspection_requests
    item = SimpleNamespace(kind='unresolved', class_name='Material', source_instance_id=70,
                           property_id=12, element_index=0, element_size=0)
    assert inspection_requests(item) == ([], [])


def test_single_shape_tree_is_actor_then_geometry_and_keeps_material_owner(tmp_path):
    session, scene, adapter = pyramid_view(tmp_path)
    original, _ = adapter.hierarchy()
    model = ObjectTreeModel(page_size=1000)
    model.populate(adapter)
    for actor in (i.id for i in adapter.instances.values() if adapter.class_name(i.id) == 'PxRigidDynamic'):
        shape, = adapter.references(actor, 'shapes')
        geometry, = adapter.references(shape, 'geom')
        children = model.get_item_children(model.canonical_item(actor))
        assert [c.instance_id for c in children] == [geometry]
        assert children[0].class_name == 'PxBoxGeometry'
        materials = model.get_item_children(children[0])
        assert [m.key for m in materials] == [f'group:object:{shape}:materials']
        assert model.row_identity(materials[0]).owner_id == shape
        assert model.action_scope([f'object:{actor}']).native_roots == (actor,)
        assert model.object_count == 175
        assert original[f'object:{geometry}'].parent_key == f'group:object:{shape}:geom'


def test_multiple_shapes_keep_one_shapes_group_and_shared_shape_stays_reference():
    from ovd_ui.hierarchy import HierarchyNode as N
    from ovd_ui.compact_hierarchy import compact_scene_hierarchy
    nodes = {}
    def add(key, cls='', ident=None, kind='group', parent=None, children=(), target=None):
        nodes[key] = N(key, key, cls, ident, kind, parent, target, children=list(children))
    add('a', 'PxRigidDynamic', 1, 'object', children=['shapes'])
    add('shapes', parent='a', children=['s1','s2']); nodes['shapes'].name='shapes'
    for index in (1,2):
        add(f's{index}', 'PxShape', 10+index, 'object', 'shapes', [f'g{index}'])
        add(f'g{index}', parent=f's{index}', children=[f'b{index}']); nodes[f'g{index}'].name='geom'
        add(f'b{index}', 'PxBoxGeometry', 20+index, 'object', f'g{index}')
    add('other', 'PxRigidStatic', 2, 'object', children=['shared'])
    add('shared', parent='other', children=['alias']); nodes['shared'].name='shapes'
    add('alias', 'PxShape', 11, 'reference', 'shared', target='s1')
    projected, hidden = compact_scene_hierarchy(nodes)
    assert projected['a'].children == ['shapes']
    assert projected['shapes'].name == 'Shapes'
    assert projected['shapes'].children == ['b1','b2']
    assert projected['b1'].parent_key == 'shapes'
    assert projected['other'].children == ['alias']
    assert projected['alias'].kind == 'reference'
    assert projected['alias'].instance_id == 11  # Shared shape is still a jump, not new ownership.
    assert projected['alias'].target_key == 's1'
    assert 'PxBoxGeometry' in projected['alias'].name
    assert {'s1','s2','g1','g2','shared'} <= hidden
    assert nodes['shapes'].children == ['s1','s2']


def test_unresolved_inspector_explains_missing_target_and_links_real_shape(tmp_path, display):
    from ovd_ui.widgets.properties import PropertyInspector
    session, scene, adapter = pyramid_view(tmp_path)
    model = ObjectTreeModel(); model.populate(adapter)
    missing = next(n for n in model._specs.values() if n.kind == 'unresolved')
    selected = []
    inspector = PropertyInspector(selected.append, lambda *_: None, {})
    inspector.build(adapter, [], record=model._item(missing.key))
    assert any('target is unavailable' in w.text for w in display.widgets)
    display.find('Inspect shape properties', 'Button').kwargs['clicked_fn']()
    assert selected == [missing.source_instance_id]


def test_geometry_inspector_keeps_shape_properties_one_click_away(tmp_path, display):
    from ovd_ui.widgets.properties import PropertyInspector
    session, scene, adapter = pyramid_view(tmp_path)
    shape = next(i.id for i in adapter.instances.values() if adapter.class_name(i.id) == 'PxShape')
    geometry, = adapter.references(shape, 'geom')
    adapter = PvdDataAdapter(session.presentation(frame=111, inspect_ids=[geometry]))
    selected = []
    PropertyInspector(selected.append, lambda *_: None, {}).build(adapter, [geometry])
    display.find(f'Inspect shape #{shape}', 'Button').kwargs['clicked_fn']()
    assert selected == [shape]


def test_explicit_hidden_shape_inspection_switches_to_class_view(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from test_object_browser_grouping import make_browser
    session, scene, adapter = pyramid_view(tmp_path)
    model = ObjectTreeModel(); model.populate(adapter)
    shape = next(i.id for i in adapter.instances.values() if adapter.class_name(i.id) == 'PxShape')
    browser, events = make_browser(model, [], monkeypatch)
    indices = []
    browser.grouping = SimpleNamespace(model=SimpleNamespace(get_item_value_model=lambda:
        SimpleNamespace(set_value=indices.append)))
    browser.ensure_native_visible([shape])
    browser.select_item(model.canonical_item(shape))
    assert model.grouping == 'class' and indices == [1]
    assert browser.tree.selection[0].instance_id == shape
    assert not model.hidden_keys


def test_selected_contact_vector_is_published_and_inspectable(app, tmp_path, display):
    from ovd_ui.widgets.properties import PropertyInspector
    source = Session(str(PYRAMID), str(tmp_path))
    assert app._display(PYRAMID, source)
    assert app._refresh(frame=111)
    group = next(n for n in app.object_model._specs.values() if n.name == 'pairsContactNormals')
    item = app.object_model._item(group.key)
    while item.kind != 'array_element':
        item = app.object_model.get_item_children(item)[-1]
    app._tree_selection([item])
    assert app._refresh()
    assert app.selected_record.element_index == 583
    raw = app.adapter.array_record(item.source_instance_id, item.property_id, 583, 12)
    assert len(raw) == 12
    assert sum(v*v for v in struct.unpack('<3f', raw)) == pytest.approx(1)
    PropertyInspector(lambda *_:None, lambda *_:None, {}).build(app.adapter, [], record=app.selected_record)
    assert not any('Waiting' in w.text or 'unavailable' in w.text for w in display.widgets)


@pytest.mark.parametrize('type_name,size,error', [('PvdF32',7,'Incomplete recorded array'),
                                                ('UnknownRecord',12,'Element layout unavailable')])
def test_nonempty_invalid_contact_layout_has_an_explicit_diagnostic(type_name, size, error):
    from types import SimpleNamespace as NS
    from ovd_ui.hierarchy import build_hierarchy
    from ovd_ui.widgets.object_browser import row_text
    prop = NS(name='pairsContactNormals', id=12, type=type_name, array=True, semantic='')
    adapter = NS(ovd=True, instances={1:NS(id=1, alive=True, physics_id=-1, scene_id=-1)},
        class_name=lambda _: 'PxScene', name=lambda _: 'Scene', properties=lambda _: [prop],
        session=NS(property_size=lambda *_:size), classes={}, _tree_array_sizes={}, hierarchy_diagnostics=[])
    nodes, roots = build_hierarchy(adapter)
    from test_tree_model import SpecificationAdapter
    model = ObjectTreeModel(); model.populate(SpecificationAdapter(nodes, roots))
    item = model._item('group:object:1:pairscontactnormals')
    assert item.array_error == error
    assert error in row_text(item)[1] and '0 records' not in row_text(item)[1]


def test_scene_inspector_uses_same_contact_vector_layout_as_tree(tmp_path, display):
    from ovd_ui.widgets.properties import PropertyInspector
    source, scene, adapter = pyramid_view(tmp_path)
    prop = adapter.property(scene.instance_id, 'pairsContactNormals')
    inspector = PropertyInspector(lambda *_:None, lambda *_:None, {}, on_request=lambda:None)
    inspector._property(adapter, scene.instance_id, prop)
    assert display.find('584 elements', 'CollapsableFrame')


def test_mixed_material_slots_distinguish_resolved_missing_null_and_empty(tmp_path, monkeypatch):
    from capture_semantics import test_scene_roles
    source = Session(str(ROOT / 'test/data/test_scene.ovd'), str(tmp_path))
    source.presentation(frame=1)
    ids = test_scene_roles(source)
    adapter = PvdDataAdapter(source)
    original = adapter.references
    slots = [ids.material, -1, 0, ids.material]
    monkeypatch.setattr(adapter, 'references', lambda identity, name:
        slots if identity == ids.shape and name.casefold() == 'materials' else original(identity, name))
    for expected_slots, missing_count, resolved_count in ((slots,1,1),([],0,0),([ids.material],0,1)):
        slots = expected_slots
        adapter._hierarchies.clear()
        nodes, _ = adapter.hierarchy()
        rows = [nodes[k] for k in nodes[f'group:object:{ids.shape}:materials'].children]
        assert sum(row.kind == 'unresolved' for row in rows) == missing_count
        assert sum(row.instance_id == ids.material for row in rows) == resolved_count
        assert all(row.instance_id != 0 for row in rows)


def test_compacted_shapes_remain_searchable_by_name_and_native_id(tmp_path):
    source, scene, adapter = pyramid_view(tmp_path)
    shape = next(i.id for i in adapter.instances.values() if adapter.class_name(i.id) == 'PxShape')
    adapter.names[shape] = 'Unique contact patch shape'
    model = ObjectTreeModel(); model.populate(adapter)
    assert f'object:{shape}' in model.hidden_keys
    for text, fields in ((f'id:{shape}', ('id',)), ('Unique contact patch shape', ('name',))):
        model.set_filter(text, fields=fields, whole_word=True)
        assert [item.instance_id for item in model.find_matches()] == [shape]
    assert model.ensure_native_visible([shape])
    assert model.canonical_item(shape).key in model._visible_keys


@pytest.mark.parametrize('array', [True, False])
def test_container_actions_include_existing_pages_and_records_without_expansion(tmp_path, array):
    from ovd_ui.row_actions import RowActionState
    source, scene, adapter = pyramid_view(tmp_path)
    model = ObjectTreeModel(); model.populate(adapter)
    group = next(n for n in model._specs.values()
                 if n.name == ('pairsContactNormals' if array else 'PxRigidDynamic'))
    node = model._item(group.key)
    if array:
        while node.kind != 'array_element':
            node = model.get_item_children(node)[0]
    local_keys = {n.key for n in model._specs.values() if n.kind in ('page','array_page','array_element')
                  and group.key in {p.key for p in model.ancestors(n.key)}}
    assert local_keys
    before = len(model._specs), len(model.cache)
    state = RowActionState(); state.sync(model.capture_key, model.segment_epoch)
    for action, enabled in (('visibility', False), ('transparency', True)):
        scope = model.action_scope([group.key], action=action)
        assert local_keys <= set(scope.local_keys)
        assert state.apply(scope, enabled)
        for key in local_keys:
            assert (state.is_hidden if action == 'visibility' else state.is_transparent)(model.row_identity(key))
    assert (len(model._specs), len(model.cache)) == before


def test_retained_geometry_link_refreshes_when_published_shape_owner_changes(tmp_path, display):
    from dataclasses import replace
    from ovd_ui.widgets.properties import PropertyInspector
    source, scene, initial = pyramid_view(tmp_path)
    shapes = [i.id for i in initial.instances.values() if initial.class_name(i.id) == 'PxShape']
    first, second = shapes[:2]
    geometry, = initial.references(first, 'geom')
    selected = []
    inspector = PropertyInspector(selected.append, lambda *_:None, {})
    previous_key = None
    for frame, owner in ((111, first), (113, second), (111, first)):
        adapter = PvdDataAdapter(source.presentation(frame=frame, inspect_ids=[geometry]))
        # Drive the owned hierarchy boundary independently from scalar values.
        # Geometry/schema/identity survive; only the published owner changes.
        nodes, _ = adapter.hierarchy()
        key = f'object:{geometry}'
        nodes[key] = replace(nodes[key], parent_key=f'group:object:{owner}:geom')
        start = len(display.widgets)
        inspector.build(adapter, [geometry], retain=True)
        assert inspector._rows_key != previous_key
        previous_key = inspector._rows_key
        links = [w for w in display.widgets[start:] if w.kind == 'Button' and w.text.startswith('Inspect shape')]
        assert len(links) == 1 and links[0].text == f'Inspect shape #{owner}'
        links[0].kwargs['clicked_fn']()
    assert selected == [first, second, first]

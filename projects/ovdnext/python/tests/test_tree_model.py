# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ownership, lifetime and bounded UI enumeration against real PVD captures."""
from capture_semantics import test_scene_roles, multi_geom_roles, unique_instance_id, dynamic_geometry_owner
from pathlib import Path
from types import SimpleNamespace

import pytest
from _pvd_native import Session

from ovd_ui.adapters import PvdDataAdapter
from ovd_ui.hierarchy import HierarchyNode, build_hierarchy
from ovd_ui.models import ObjectTreeModel

ROOT = Path(__file__).resolve().parents[2]


def capture(tmp_path, filename, frame=1):
    session = Session(str(ROOT / "test/data" / filename), str(tmp_path))
    session.seek(session.scenes()[0].id, frame)
    adapter = PvdDataAdapter(session)
    model = ObjectTreeModel()
    model.populate(adapter)
    return session, adapter, model


def object_ancestors(model, instance_id):
    return [item.instance_id for item in model.ancestors(instance_id)
            if item.kind == "object"]


def assert_tree_integrity(model, adapter):
    specs = model._specs
    visited = set()
    stack = list(model._root_keys)
    while stack:
        key = stack.pop()
        assert key not in visited, f"Cycle or repeated canonical row: {key}"
        visited.add(key)
        for child in specs[key].children:
            assert specs[child].parent_key == key
            stack.append(child)
        if specs[key].kind == "reference":
            assert not specs[key].children
            assert specs[key].target_key == f"object:{specs[key].instance_id}"
    assert visited == set(specs) - model.hidden_keys
    assert {node.instance_id for node in specs.values() if node.kind == "object"} == {
        item.id for item in adapter.instances.values() if item.alive}


def test_native_scene_shape_geometry_and_material_alias(tmp_path):
    session, adapter, model = capture(tmp_path, "test_scene.ovd")
    ids=test_scene_roles(session)
    # The collapsed model allocates root widgets, not one widget per instance.
    assert len(model.cache) == len(model.roots)
    assert object_ancestors(model, ids.actor) == [ids.physics, ids.scene]
    assert object_ancestors(model, ids.shape) == [ids.physics, ids.scene, ids.actor]
    assert object_ancestors(model, ids.geometry) == [ids.physics, ids.scene, ids.actor]
    assert object_ancestors(model, ids.material) == [ids.physics]
    aliases = [node for node in model._specs.values()
               if node.kind == "reference" and node.instance_id == ids.material]
    assert len(aliases) == 1
    assert f"object:{ids.shape}" in aliases[0].parent_key
    sphere = model.canonical_item(ids.actor)
    assert sphere.values[0].as_string == f"sphere [id:{ids.actor}]"
    assert sphere.values[1].as_string == "PxRigidDynamic"
    assert sphere.renderable
    children, owners = adapter.ownership_graph()
    assert children[ids.actor] == (ids.shape,)
    assert children[ids.shape] == (ids.geometry,)
    assert owners[ids.geometry] == frozenset((ids.scene,))
    assert owners[ids.material] == frozenset()  # Shared factory material has no scene owner.
    assert_tree_integrity(model, adapter)


def test_articulation_links_follow_inbound_joint_parent_links(tmp_path):
    session, adapter, model = capture(tmp_path, "multi_geom.ovd")
    ids=multi_geom_roles(session)
    assert object_ancestors(model, ids.leaf) == [ids.physics, ids.scene, ids.articulation, ids.root, ids.middle1, ids.middle2]
    assert object_ancestors(model, ids.leaf_joint) == [ids.physics, ids.scene, ids.articulation, ids.root, ids.middle1, ids.middle2, ids.leaf]
    assert object_ancestors(model, ids.root) == [ids.physics, ids.scene, ids.articulation]
    assert not any(node.kind == "reference" and node.instance_id == ids.leaf
                   and f"object:{ids.articulation}" in node.parent_key for node in model._specs.values())
    children = model.get_item_children(model.canonical_item(ids.middle2))
    assert children[-1].instance_id == ids.leaf
    assert_tree_integrity(model, adapter)


@pytest.mark.parametrize("filename,frame", [
    ("multiscene_rigid.ovd", 0), ("multiscene_rigid.ovd", 1),
    ("resample_two_segment.ovd", 1), ("resample_two_segment.ovd", 45)])
def test_multiscene_and_recording_segments_keep_native_ownership(tmp_path, filename, frame):
    _, adapter, model = capture(tmp_path, filename, frame)
    scenes = [i for i in adapter.instances.values()
              if i.alive and adapter.class_name(i.id) == "PxScene"]
    assert len(scenes) == (1 if filename == "multiscene_rigid.ovd" and frame == 1 else 2)
    for scene in scenes:
        assert object_ancestors(model, scene.id) == [scene.physics_id]
    for actor in adapter.instances.values():
        if actor.alive and adapter.class_name(actor.id) in ("PxRigidDynamic", "PxRigidStatic"):
            if actor.scene_id in adapter.instances and adapter.instances[actor.scene_id].alive:
                assert object_ancestors(model, actor.id)[:2] == [actor.physics_id, actor.scene_id]
                for shape in adapter.references(actor.id, "shapes"):
                    assert actor.id in object_ancestors(model, shape)
    assert_tree_integrity(model, adapter)


@pytest.mark.skipif(not hasattr(Session, "query_records"), reason="Native query-record API is an optional binding capability")
def test_legacy_cloth_ownership_and_native_typed_query_records(tmp_path):
    _, adapter, model = capture(tmp_path, "cloth_physx34.pxd2")
    assert object_ancestors(model, 15) == [4, 6]
    assert object_ancestors(model, 16) == [4]
    assert any(n.kind == "reference" and n.instance_id == 16 and "object:15" in n.parent_key
               for n in model._specs.values())
    arrays = [n for n in model._specs.values() if n.kind == "array"]
    assert {(n.class_name, n.element_count, n.element_size) for n in arrays} == {
        ("PvdRaycast", 5, 64), ("PvdSqHit", 5, 56)}
    query = model.cache[next(n.key for n in arrays if n.class_name == "PvdRaycast")]
    records = model.get_item_children(query)
    assert len(records) == 5
    record = records[0]
    assert record.kind == "array_element"
    assert record.instance_id is None
    assert record.query_id == -6
    assert model.canonical_item(-6) is record
    assert model.canonical_item(-70).element_index == 4
    assert model.canonical_item(-5).class_name == "PvdSqHit"
    assert model.canonical_item(-9999) is None
    assert object_ancestors(model, -6) == [4, 6]
    assert len(adapter.session.array_record(record.source_instance_id, record.property_id, 0).fields) == 9
    model.set_filter("id:-54", fields=("id",), whole_word=True)
    assert [item.query_id for item in model.find_matches()] == [-54]
    model.filter = ""
    assert len(adapter.array_record(record.source_instance_id, record.property_id,
                                   record.element_index, record.element_size)) == 64
    with pytest.raises(IndexError):
        adapter.array_record(record.source_instance_id, record.property_id, 5, 64)
    assert not adapter.hierarchy_diagnostics
    assert_tree_integrity(model, adapter)


def test_widget_identity_and_expansion_survive_frame_death_and_rewind(tmp_path):
    session, adapter, model = capture(tmp_path, "test_scene.ovd")
    ids=test_scene_roles(session)
    original = model.canonical_item(ids.actor)
    expanded = model.reveal_ancestors(original)
    schema = adapter.schema[adapter.instances[ids.actor].class_id]
    for frame in (2, 3, 1):
        session.seek(session.scenes()[0].id, frame)
        adapter.reload()
        model.populate(adapter)
        assert adapter.schema[adapter.instances[ids.actor].class_id] is schema
        if frame == 3:
            assert model.canonical_item(ids.actor) is None
        else:
            assert model.canonical_item(ids.actor) is original
    assert {item.key for item in expanded} <= model.expanded_keys
    assert {item.key for item in model.expanded_items()} == {item.key for item in expanded}


def test_recursive_search_navigation_reveal_and_class_projection(tmp_path):
    session, adapter, model = capture(tmp_path, "multi_geom.ovd")
    ids=multi_geom_roles(session)
    model.set_filter("PxArticulationLink")
    matches = model.find_matches()
    assert {item.instance_id for item in matches} == {ids.leaf, ids.middle2, ids.middle1, ids.root}
    assert model.next_match(matches[-1]) is matches[0]
    assert model.previous_match(matches[0]) is matches[-1]
    assert model.next_match().instance_id == ids.root
    model.set_filter(f"id:{ids.sphere}", whole_word=True)
    assert [item.instance_id for item in model.find_matches()] == [ids.sphere]
    assert model.canonical_item(ids.sphere).key in model._visible_keys
    assert all(item.key in model._visible_keys for item in model.ancestors(ids.sphere))
    assert model.canonical_item(ids.leaf).key not in model._visible_keys
    revealed = []
    model.reveal_ancestors(ids.leaf, revealed.append)
    assert [item.key for item in revealed] == [item.key for item in model.ancestors(ids.leaf)]
    assert model.canonical_item(ids.leaf).key in model._visible_keys
    model.set_filter("pxrigiddynamic", fields=("type",), match_case=True)
    assert not model.find_matches()
    model.set_filter("sphere", fields=("name",))
    assert any("sphere" in item.values[0].as_string.casefold() for item in model.find_matches())
    original = model.canonical_item(ids.sphere)
    model.filter = ""
    model.set_grouping("class")
    assert model.canonical_item(ids.sphere) is original
    assert model.ancestors(ids.sphere)[0].key.startswith("class:")
    assert_tree_integrity(model, adapter)


class SpecificationAdapter:
    def __init__(self, specs, roots):
        self.session = object()
        self._schema_epoch = 0
        self.specs, self.roots = specs, roots

    def hierarchy(self, grouping):
        return self.specs, self.roots


def test_large_reference_lists_and_typed_arrays_materialize_in_bounded_pages():
    root = HierarchyNode("object:1", "Scene", "PxScene", 1, "object")
    specs = {root.key: root}
    for index in range(2, 5002):
        key = f"object:{index}"
        specs[key] = HierarchyNode(key, f"Actor {index}", "PxRigidDynamic", index,
                                   "object", root.key)
        root.children.append(key)
    query = HierarchyNode("queries", "Raycasts", "PvdRaycast", kind="array",
                          parent_key=root.key, source_instance_id=1, property_id=22,
                          element_count=20000, element_size=64)
    specs[query.key] = query
    root.children.append(query.key)
    model = ObjectTreeModel(page_size=64)
    model.populate(SpecificationAdapter(specs, [root.key]))
    assert len(model.cache) == 1
    assert len(root.children) == 5001  # Projection never mutates adapter cache.
    assert all(len(node.children) <= 64 for node in model._specs.values())
    last = model.canonical_item(5001)
    assert model.ancestors(last)[0].instance_id == 1
    assert len(model.cache) < 10
    pages = model.get_item_children(model.cache[query.key])
    assert len(pages) <= 64
    while pages[0].kind == "array_page":
        pages = model.get_item_children(pages[0])
    assert len(pages) == 64
    assert pages[0].element_index == 0
    assert pages[-1].element_index == 63
    assert len(model.cache) < 200
    model.set_filter("PvdRaycast", fields=("type",))
    assert len(model.get_item_children(model.cache[pages[0].parent.key])) == 64


def test_deep_hierarchy_reveal_is_iterative():
    specs = {}
    for index in range(1, 1501):
        key = f"object:{index}"
        specs[key] = HierarchyNode(key, f"Link {index}", "PxArticulationLink", index,
            "object", f"object:{index-1}" if index > 1 else None,
            children=[f"object:{index+1}"] if index < 1500 else [])
    model = ObjectTreeModel()
    model.populate(SpecificationAdapter(specs, ["object:1"]))
    assert len(model.cache) == 1
    item = model.canonical_item(1500)
    assert item.parent.instance_id == 1499
    assert len(model.reveal_ancestors(item)) == 1499
    adapter = model._adapter
    adapter._schema_epoch += 1
    model.populate(adapter)
    assert model.canonical_item(1500) is item


def test_immutable_presentations_share_capture_identity_only_with_their_source():
    node = HierarchyNode("object:1", "Actor", "PxRigidDynamic", 1, "object")
    adapter = SpecificationAdapter({node.key: node}, [node.key])
    adapter.session = SimpleNamespace(capture_key=101)
    model = ObjectTreeModel()
    model.populate(adapter)
    item = model.canonical_item(1)
    adapter.session = SimpleNamespace(capture_key=101)
    model.populate(adapter)
    assert model.canonical_item(1) is item
    adapter.session = SimpleNamespace(capture_key=202)
    model.populate(adapter)
    assert model.canonical_item(1) is not item


def test_malformed_reference_cycles_become_explicit_leaf_aliases():
    prop = SimpleNamespace(name="Children", type="ObjectRef", semantic="children", array=True)
    adapter = SimpleNamespace(
        instances={i: SimpleNamespace(id=i, class_id=1, alive=True, physics_id=-1, scene_id=-1)
                   for i in (1, 2, 3)},
        class_name=lambda _: "Node", name=lambda i: f"Node {i}",
        properties=lambda _: [prop], references=lambda i, _: {1: [2], 2: [3], 3: [1]}[i])
    specs, roots = build_hierarchy(adapter)
    assert roots == ["object:1"]
    assert len([n for n in specs.values() if n.kind == "object"]) == 3
    aliases = [n for n in specs.values() if n.kind == "reference"]
    assert len(aliases) == 1
    assert aliases[0].instance_id == 1
    assert aliases[0].children == []


def test_exact_native_revision_reuses_projection_and_invalidates_on_change(tmp_path):
    native = Session(str(ROOT / "test/data/test_scene.ovd"), str(tmp_path))
    native.seek(native.scenes()[0].id, 1)
    actor = test_scene_roles(native).actor

    class VersionedSession:
        revision = 1

        def hierarchy_revision(self):
            return self.revision

        def __getattr__(self, name):
            return getattr(native, name)

    session = VersionedSession()
    adapter = PvdDataAdapter(session)
    model = ObjectTreeModel()
    model.populate(adapter)
    original = model.canonical_item(actor)
    projection = model._specs
    # A native exact comparison permits reuse with no graph/UI traversal.
    adapter.reload()
    model.populate(adapter)
    assert model._specs is projection
    assert model.canonical_item(actor) is original
    native.seek(native.scenes()[0].id, 3)
    session.revision += 1
    adapter.reload()
    model.populate(adapter)
    assert model._specs is not projection
    assert model.canonical_item(actor) is None


@pytest.mark.skipif(not hasattr(Session, "hierarchy_changes"), reason="Native topology-delta API is an optional binding capability")
def test_unreal_query_geometry_delta_matches_full_projection_and_keeps_actor_rows(tmp_path):
    session, adapter, model = capture(tmp_path, "ue4_shooter.pxd2", frame=30)
    assert hasattr(session, "hierarchy_changes")
    actor_id = next(i.id for i in adapter.instances.values() if i.alive and adapter.class_name(i.id) == "PxRigidStatic")
    actor = model.canonical_item(actor_id)
    adapter.ownership_graph()
    for frame in (31, 32, 30):
        old_keys = set(model._specs)
        session.seek(session.scenes()[0].id, frame)
        adapter.reload()
        assert adapter._hierarchy_patch is not None
        model.populate(adapter)
        assert model.canonical_item(actor_id) is actor
        # The optimization must be exactly equivalent to ownership arbitration,
        # including replaced leaf IDs and their SceneQueries.GeometryList group.
        reference = PvdDataAdapter(session)
        assert adapter.hierarchy() == reference.hierarchy()
        assert adapter.ownership_graph() == reference.ownership_graph()
        reference_model = ObjectTreeModel()
        reference_model.populate(reference)
        assert model._specs == reference_model._specs
        assert len(old_keys - set(model._specs)) == 1  # The transient query geometry.
        assert_tree_integrity(model, adapter)

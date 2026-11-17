# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact, conservative branch updates for transient scene-query geometry.

Legacy PhysX captures replace query backing geometries each frame. The native
delta proves every other object's topology/name unchanged. Any broader edit
falls back to the full ownership projection; there is no fixture-specific ID.
"""
from dataclasses import replace


def patch_query_geometry(adapter, previous_instances, delta):
    if delta.schema_changed or delta.query_records_changed:
        return None
    removed = set(delta.removed_ids)
    changed = set(delta.changed_ids)
    added = {key for key in changed if key not in previous_instances or not previous_instances[key].alive}
    existing = changed - added
    if not removed or len(added) != len(removed) or len(existing) != 1:
        return None
    source = next(iter(existing))
    if adapter.class_name(source) != "PxScene":
        return None
    old_source, new_source = previous_instances[source], adapter.instances[source]
    if (old_source.scene_id, old_source.physics_id) != (new_source.scene_id, new_source.physics_id):
        return None
    for key in removed:
        instance = previous_instances.get(key)
        if instance is None:
            return None
        cls = adapter.classes[instance.class_id].name
        if not cls.startswith("Px") or not cls.endswith("Geometry"):
            return None
    if any(not adapter.class_name(key).startswith("Px") or not adapter.class_name(key).endswith("Geometry")
           for key in added):
        return None
    name_prop = adapter.property(source, "name")
    name = (adapter.session.string_value(source, name_prop.id)
            if name_prop and name_prop.type == "StringHandle" else "") or adapter.class_name(source)
    if name != adapter.names.get(source):
        return None

    refreshed_refs = {}
    replacements = {}
    for prop, lower, _ in adapter.hierarchy_properties(source):
        if prop.type != "ObjectRef":
            if adapter.session.property_size(source, prop.id) != adapter._tree_array_sizes.get((source, prop.id), 0):
                return None
            continue
        refs = tuple(adapter.session.references(source, prop.id))
        old = adapter._reference_cache.get((source, prop.id))
        if old is None:
            return None
        refreshed_refs[(source, prop.id)] = refs
        if refs == old:
            continue
        if (not lower.startswith(("scenequeries.", "batchedqueries.")) or
                not lower.endswith(".geometrylist") or len(old) != len(refs)):
            return None
        for before, after in zip(old, refs):
            if before == after:
                continue
            if before not in removed or after not in added:
                return None
            if before in replacements and replacements[before] != after:
                return None
            replacements[before] = after
    if set(replacements) != removed or set(replacements.values()) != added:
        return None
    for key in added:
        for prop, _, _ in adapter.hierarchy_properties(key):
            if prop.type != "ObjectRef":
                continue
            refs = tuple(adapter.session.references(key, prop.id))
            if any(ref in adapter.instances and adapter.instances[ref].alive for ref in refs):
                return None
            refreshed_refs[(key, prop.id)] = refs

    nodes, roots = adapter._hierarchies["scene"]
    old_keys = {f"object:{key}" for key in removed}
    if any(node.kind == "reference" and node.target_key in old_keys for node in nodes.values()):
        return None
    updates, deleted, renamed = {}, set(), {}
    for before, after in replacements.items():
        old_key, new_key = f"object:{before}", f"object:{after}"
        node = nodes.get(old_key)
        if node is None or node.parent_key is None or node.class_name != adapter.class_name(after):
            return None
        parent = nodes[node.parent_key]
        cursor = parent
        while cursor.kind != "object" and cursor.parent_key:
            cursor = nodes[cursor.parent_key]
        if cursor.instance_id != source:
            return None
        # A query backing geometry may expose empty schema wrappers. References
        # to live meshes/shapes require the general ownership arbitration.
        if any(nodes[key].kind != "group" or nodes[key].children for key in node.children):
            return None
        if previous_instances[before].class_id != adapter.instances[after].class_id:
            return None
        children = []
        for child_key in node.children:
            new_child = child_key.replace(old_key, new_key, 1)
            updates[new_child] = replace(nodes[child_key], key=new_child, parent_key=new_key,
                                         source_instance_id=after)
            children.append(new_child)
            deleted.add(child_key)
            renamed[child_key] = new_child
        updates[new_key] = replace(node, key=new_key, instance_id=after,
            name=f"{adapter.name(after)} [id:{after}]", children=children)
        parent = updates.get(parent.key, parent)
        updates[parent.key] = replace(parent, children=[new_key if key == old_key else key for key in parent.children])
        deleted.add(old_key)
        renamed[old_key] = new_key

    # Validation above is read-only. Publish the complete small change once.
    for key in deleted:
        nodes.pop(key)
    nodes.update(updates)
    for key in removed:
        adapter.names.pop(key, None)
    for key in tuple(adapter._reference_cache):
        if key[0] in removed | changed:
            adapter._reference_cache.pop(key)
    adapter._reference_cache.update(refreshed_refs)
    if adapter._ownership is not None:
        children, owners = adapter._ownership
        children[source] = tuple(replacements.get(key, key) for key in children[source])
        for before, after in replacements.items():
            children[after] = children.pop(before)
            owners[after] = owners.pop(before)
    return nodes, deleted, updates, renamed

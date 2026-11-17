# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source-faithful, cycle-safe projection of native PVD references into a tree.

Rules follow PVDNext/instanceview.cpp's scene buckets, canonical reference
leaves, articulation link ownership, metadata placement, and query groups.
This module contains no OVUI objects and never interprets capture handles.
"""
from dataclasses import dataclass, field
from .array_layout import array_layout


RENDERABLE_CLASSES = frozenset(("PxRigidDynamic", "PxRigidStatic", "PxCloth", "PxShape",
    "PxAggregate", "PxArticulationLink", "PxPBDParticleSystem"))
SCENE_MEMBER_CLASSES = RENDERABLE_CLASSES - {"PxShape", "PxArticulationLink"} | {
    "PxArticulation", "PxArticulationReducedCoordinate", "PxConstraint", "PxJoint",
    "PxParticleSystem", "PxParticleFluid", "PxDeformableVolume", "PxDeformableSurface"}


@dataclass
class HierarchyNode:
    key: str
    name: str
    class_name: str = ""
    instance_id: int | None = None
    kind: str = "group"
    parent_key: str | None = None
    target_key: str | None = None
    renderable: bool = False
    children: list[str] = field(default_factory=list)
    source_instance_id: int | None = None
    property_id: int | None = None
    element_index: int | None = None
    element_size: int = 0
    element_count: int = 0
    byte_count: int = 0
    query_id: int | None = None
    segment_scoped: bool = False
    array_error: str = ''


def build_hierarchy(adapter, grouping="scene"):
    alive = {key: item for key, item in adapter.instances.items() if item.alive}
    class_names = {key: adapter.class_name(key) for key in alive}
    nodes = {}
    for instance_id, instance in alive.items():
        cls = class_names[instance_id]
        key = f"object:{instance_id}"
        nodes[key] = HierarchyNode(key, f"{adapter.name(instance_id)} [id:{instance_id}]",
            cls, instance_id, "object", renderable=cls in RENDERABLE_CLASSES)

    groups = {}

    def group(source, label):
        lookup = (source, label.casefold())
        if lookup in groups:
            return groups[lookup]
        parent_key = f"object:{source}"
        for component in label.split("."):
            key = f"group:{parent_key}:{component.casefold()}"
            if key not in nodes:
                nodes[key] = HierarchyNode(key, component, parent_key=parent_key,
                    source_instance_id=source,
                    segment_scoped=label.startswith(("SceneQueries.", "BatchedQueries.")))
                nodes[parent_key].children.append(key)
            parent_key = key
        groups[lookup] = parent_key
        return parent_key

    if grouping == "class":
        roots = []
        for instance_id, instance in alive.items():
            cls = class_names[instance_id]
            key = f"class:{instance.class_id}"
            if key not in nodes:
                nodes[key] = HierarchyNode(key, cls)
                roots.append(key)
            obj = nodes[f"object:{instance_id}"]
            obj.parent_key = key
            nodes[key].children.append(obj.key)
        return nodes, sorted(roots, key=lambda key: nodes[key].name.casefold())

    # A priority selects each object's one canonical home independently of
    # traversal/expansion order. Factory lists therefore never steal actors
    # from their scenes, and shared resources appear as explicit jump leaves.
    homes = {}
    edges = []
    arrays = []
    unresolved_materials = []
    relation_groups = set()
    articulation_links = {}
    physics_ids = [key for key in alive if class_names[key] == "PxPhysics"]

    def offer(child, parent, label, priority):
        if child not in alive or parent not in alive or child == parent:
            return
        if class_names[child] == "PxPhysics":
            return
        candidate = (priority, parent, label)
        if child not in homes or candidate < homes[child]:
            homes[child] = candidate

    for source in alive:
        cls = class_names[source]
        properties = (adapter.hierarchy_properties(source) if hasattr(adapter, "hierarchy_properties")
                      else ((prop, prop.name.casefold(), prop.semantic.casefold()) for prop in adapter.properties(source)))
        for prop, prop_lower, semantic in properties:
            if prop.type != "ObjectRef":
                if (prop.array and cls == "PxScene" and
                    (prop.name.startswith(("SceneQueries.", "BatchedQueries.")) or "contact" in prop_lower) and
                    not any(skip in prop_lower for skip in (".poselist", ".filterdatalist"))):
                    size = adapter.session.property_size(source, prop.id)
                    adapter._tree_array_sizes[(source, prop.id)] = size
                    if size:
                        arrays.append((source, prop, size))
                continue
            refs = adapter.references(source, prop.name)
            valid = tuple(dict.fromkeys(ref for ref in refs if ref in alive))
            if cls == 'PxShape' and prop_lower == 'materials':
                unresolved_materials.extend((source, prop, index) for index, ref in enumerate(refs)
                                            if ref not in alive and ref != 0)
            if semantic == "parents":
                for parent in valid:
                    if prop_lower == "scene":
                        offer(source, parent, cls, 12)
                    elif prop_lower == "physics":
                        offer(source, parent, "scenes" if cls == "PxScene" else cls, 10)
                continue
            if cls in ("PxArticulationReducedCoordinate", "PxArticulation") and prop_lower == "links":
                articulation_links[source] = valid
                continue
            if cls == "PxArticulationLink" and prop_lower == "inboundjoint":
                for target in valid:
                    offer(target, source, "", 2)
                    edges.append((source, "", target))
                continue
            extra_reference = cls == "PxShape" and prop_lower in ("geom", "geometry", "materials")
            if semantic not in ("", "children") and not extra_reference:
                continue
            if cls == "PxScene" and "." not in prop.name:
                for target in valid:
                    label = class_names[target]
                    offer(target, source, label, 20)
                    edges.append((source, label, target))
                continue
            label = prop.name
            relation_groups.add((source, label, getattr(prop, "id", None)))
            priority = 60 if cls == "PxPhysics" else 45
            if cls == "PxShape":
                priority = 30 if prop_lower in ("geom", "geometry") else 75
            elif prop_lower == "shapes":
                priority = 25 if cls != "PxPhysics" else 60
            if cls == "PxPhysics" and prop_lower == "scenes":
                priority = 15
            for target in valid:
                offer(target, source, label, priority)
                edges.append((source, label, target))

    # Native ownership includes recording segment identity; forward lists can
    # omit a scene snapshot or reuse the same capture handle in another scene.
    for instance_id, instance in alive.items():
        cls = class_names[instance_id]
        physics = instance.physics_id
        scene = instance.scene_id
        if cls == "PxScene" and physics in alive:
            offer(instance_id, physics, "scenes", 1)
        elif getattr(instance, "scene_joint", False) and scene in alive:
            offer(instance_id, scene, "Joints", 11)
        elif cls in SCENE_MEMBER_CLASSES and scene in alive:
            offer(instance_id, scene, cls, 11)
        elif cls == "PxOmniPvdMetaData":
            # Metadata has no ownership property in older OVD schemas. The
            # legacy instance view puts it under the first physics object.
            owner = physics if physics in physics_ids else physics_ids[0] if physics_ids else None
            if owner is not None:
                offer(instance_id, owner, "", 10)

    for articulation, links in articulation_links.items():
        link_set = set(links)
        for link in links:
            joint = next(iter(adapter.references(link, "inboundJoint")), None)
            parent = next(iter(adapter.references(joint, "parentLink")), None) if joint in alive else None
            if parent not in link_set or parent == link:
                # PhysX 3 captures may record Parent directly on the link.
                parent = next(iter(adapter.references(link, "parent")), None)
            owner = parent if parent in link_set and parent != link else articulation
            offer(link, owner, "", 1)
            edges.append((owner, "", link))

    # Break malformed ownership cycles once, keeping their original edges as
    # reference leaves. Iterative coloring also handles very deep articulations.
    finished = set()
    for start in alive:
        path, positions = [], {}
        cursor = start
        while cursor in homes and cursor not in finished:
            if cursor in positions:
                cycle = path[positions[cursor]:]
                del homes[min(cycle)]
                break
            positions[cursor] = len(path)
            path.append(cursor)
            cursor = homes[cursor][1]
        finished.update(path)

    def attach(source, label, target, alias=False):
        parent_key = group(source, label) if label else f"object:{source}"
        obj_key = f"object:{target}"
        if alias:
            key = f"reference:{parent_key}:{target}"
            if key in nodes:
                return
            original = nodes[obj_key]
            nodes[key] = HierarchyNode(key, f"↗ {original.name}", original.class_name,
                target, "reference", parent_key, obj_key, original.renderable)
        else:
            key = obj_key
            nodes[key].parent_key = parent_key
        # Canonical homes are assigned once; alias keys are deduplicated above.
        # Membership scans here become quadratic for large scene actor buckets.
        nodes[parent_key].children.append(key)

    for source, label, property_id in sorted(relation_groups, key=lambda value: value[:2]):
        nodes[group(source, label)].property_id = property_id
    for child, (_, parent, label) in homes.items():
        attach(parent, label, child)
    for source, label, target in edges:
        home = homes.get(target)
        if home is None or (home[1], home[2].casefold()) != (source, label.casefold()):
            attach(source, label, target, alias=True)

    # Preserve recorded slots even when their target cannot be resolved. They
    # are diagnostics, not native objects and must never become selectable IDs.
    for source, prop, index in unresolved_materials:
        parent = group(source, prop.name)
        key = f'unresolved:{parent}:{index}'
        nodes[key] = HierarchyNode(key, f'Unresolved material [{index}]', 'Material',
            kind='unresolved', parent_key=parent, source_instance_id=source,
            property_id=prop.id, element_index=index, segment_scoped=True)
        nodes[parent].children.append(key)

    # Scalar query/contact arrays are represented by their actual native
    # record type/stride. OVUI creates record rows lazily and pages large arrays.
    for source, prop, byte_count in arrays:
        key = group(source, prop.name)
        node = nodes[key]
        node.kind = "array"
        node.segment_scoped = True
        parent_key = node.parent_key
        while parent_key and nodes[parent_key].kind != "object":
            nodes[parent_key].segment_scoped = True
            parent_key = nodes[parent_key].parent_key
        node.class_name, node.element_size = array_layout(adapter, prop)
        node.source_instance_id = source
        node.property_id = prop.id
        node.byte_count = byte_count
        if node.element_size and byte_count % node.element_size == 0:
            node.element_count = byte_count // node.element_size
        else:
            node.array_error = ('Incomplete recorded array' if node.element_size else 'Element layout unavailable')
            adapter.hierarchy_diagnostics.append(f"{prop.name}: {node.array_error} ({byte_count} bytes)")

    roots = [node.key for node in nodes.values() if node.parent_key is None and node.kind == "object"]
    roots.sort(key=lambda key: (nodes[key].class_name != "PxPhysics", nodes[key].class_name != "PxScene", nodes[key].instance_id))
    # Sort only flattened scene buckets. Property wrappers preserve schema
    # grouping, and articulation children follow shape/joint/link ordering.
    for source in alive:
        node = nodes[f"object:{source}"]
        if node.class_name == "PxScene":
            node.children.sort(key=lambda key: nodes[key].name.casefold())
        elif node.class_name == "PxPhysics":
            node.children.sort(key=lambda key: nodes[key].name.casefold() != "scenes")
        elif node.class_name == "PxArticulationLink":
            node.children.sort(key=lambda key: (nodes[key].class_name == "PxArticulationLink",
                                                nodes[key].class_name.startswith("PxArticulationJoint")))
    return nodes, roots

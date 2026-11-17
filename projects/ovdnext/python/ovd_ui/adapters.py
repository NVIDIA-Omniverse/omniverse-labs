# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PVD data adapters. Native IDs and native reference resolution stay authoritative."""
from dataclasses import dataclass
import struct

import numpy as np


@dataclass
class DrawObject:
    instance_id: int
    kind: str
    matrix: np.ndarray
    dimensions: tuple


def pose_matrix(data):
    matrix = np.eye(4, dtype="f4")
    if len(data) != 28:
        return matrix
    x, y, z, w, px, py, pz = struct.unpack("<7f", data)
    matrix[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                      [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                      [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    matrix[:3, 3] = (px, py, pz)
    return matrix


class PvdDataAdapter:
    def __init__(self, session):
        self.session = session
        self.classes = {}
        self.instances = {}
        self.schema = {}
        self.names = {}
        self._property_lookup = {}
        self._hierarchy_properties = {}
        self._reference_cache = {}
        self._class_signature = None
        self._schema_epoch = None
        self.revision = 0
        self.hierarchy_diagnostics = []
        self._hierarchies = {}
        self._ownership = None
        self._query_records = None
        self._query_ids = {}
        self._query_locations = {}
        self._topology_revision = None
        self._hierarchy_patch = None
        self._tree_array_sizes = {}
        self._capture_key = None
        self.reload()

    def reload(self):
        previous_instances = self.instances
        previous_revision = self._topology_revision
        self._hierarchy_patch = None
        self.classes = {c.id: c for c in self.session.classes()}
        info = self.session.info()
        capture_key = getattr(self.session, "capture_key", id(self.session))
        capture_changed = capture_key != self._capture_key
        self._capture_key = capture_key
        self.ovd = info.ovd
        signature = tuple((c.id, c.base_id, c.name) for c in self.classes.values())
        epoch = getattr(info, "segment_epoch", 0)
        # A native revision must compare current names, lifetimes, ownership,
        # schema, references and synthetic record identities exactly. Older
        # bindings fall back to a fresh projection; never assume static refs.
        revision_fn = getattr(self.session, "hierarchy_revision", None)
        topology_revision = revision_fn() if revision_fn else object()
        topology_changed = (capture_changed or topology_revision != self._topology_revision or
                            signature != self._class_signature or epoch != self._schema_epoch)
        self._topology_revision = topology_revision
        if capture_changed or signature != self._class_signature or epoch != self._schema_epoch or getattr(info, "live", False):
            self.schema.clear()
            self._property_lookup.clear()
            self._hierarchy_properties.clear()
        self._class_signature, self._schema_epoch = signature, epoch
        count = info.instance_count
        self.instances = {i.id: i for start in range(0, count, 10000)
                          for i in self.session.instances(start, 10000)}
        if topology_changed and not capture_changed and isinstance(previous_revision, int) and "scene" in self._hierarchies:
            changes_fn = getattr(self.session, "hierarchy_changes", None)
            if changes_fn:
                from .incremental import patch_query_geometry
                self._hierarchy_patch = patch_query_geometry(self, previous_instances, changes_fn(previous_revision))
                if self._hierarchy_patch is not None:
                    topology_changed = False
                    self._hierarchies.pop("class", None)
        if topology_changed:
            self.names.clear()
            self._reference_cache.clear()
            self._hierarchies.clear()
            self._ownership = None
            self._query_records = None
            self._query_ids.clear()
            self._query_locations.clear()
            self.hierarchy_diagnostics.clear()
            self._tree_array_sizes.clear()
        self.revision += 1

    def properties(self, instance_id):
        instance = self.instances[instance_id]
        if instance.class_id not in self.schema:
            self.schema[instance.class_id] = self.session.properties(instance.class_id)
        return self.schema[instance.class_id]

    def property(self, instance_id, name):
        instance = self.instances.get(instance_id)
        if instance is None:
            return None
        if instance.class_id not in self._property_lookup:
            self._property_lookup[instance.class_id] = {
                prop.name.casefold(): prop for prop in self.properties(instance_id)}
        return self._property_lookup[instance.class_id].get(name.casefold())

    def class_name(self, instance_id):
        instance = self.instances.get(instance_id)
        cls = self.classes.get(instance.class_id) if instance else None
        return cls.name if cls else "Unknown class"

    def hierarchy_properties(self, instance_id):
        """Cache relevant schema projection, never per-frame reference values."""
        instance = self.instances[instance_id]
        if instance.class_id not in self._hierarchy_properties:
            is_scene = self.class_name(instance_id) == "PxScene"
            self._hierarchy_properties[instance.class_id] = tuple(
                (prop, prop.name.casefold(), prop.semantic.casefold())
                for prop in self.properties(instance_id)
                if prop.type == "ObjectRef" or (is_scene and prop.array and
                    (prop.name.startswith(("SceneQueries.", "BatchedQueries.")) or "contact" in prop.name.casefold()) and
                    not any(skip in prop.name.casefold() for skip in (".poselist", ".filterdatalist"))))
        return self._hierarchy_properties[instance.class_id]

    def raw(self, instance_id, name):
        prop = self.property(instance_id, name)
        return self.session.read(instance_id, prop.id) if prop else b""

    def references(self, instance_id, name):
        prop = self.property(instance_id, name)
        if not prop or prop.type != "ObjectRef" or not self.instances[instance_id].alive:
            return []
        key = (instance_id, prop.id)
        if key not in self._reference_cache:
            # Empty legacy lists must never reach engine getPropertyValue with
            # a null destination. Native versions also guard this boundary.
            # OVD unique-lists have no raw property payload; references() asks
            # the native adapter to materialize their current membership.
            size = self.session.property_size(instance_id, prop.id) if not self.ovd else 1
            self._reference_cache[key] = tuple(self.session.references(instance_id, prop.id)) if size else ()
        return list(self._reference_cache[key])

    def hierarchy(self, grouping="scene"):
        """Return native-identity node specifications and root keys."""
        if grouping not in ("scene", "class"):
            raise ValueError("Tree grouping must be scene or class")
        if grouping not in self._hierarchies:
            from .hierarchy import build_hierarchy
            self._hierarchies[grouping] = build_hierarchy(self, grouping)
        return self._hierarchies[grouping]

    def array_record(self, source_instance_id, property_id, index, element_size):
        """Read one bounded synthetic tree record without fabricating an ID."""
        if index < 0 or element_size <= 0 or element_size > 8 * 1024 * 1024:
            raise ValueError("Invalid array record range")
        size = self.session.property_size(source_instance_id, property_id)
        offset = index * element_size
        if offset + element_size > size:
            raise IndexError("Array record is outside the captured property")
        return self.session.read(source_instance_id, property_id, offset=offset, limit=element_size)

    def query_records(self):
        """Native synthetic identities; Python never reproduces type counters."""
        if self._query_records is None:
            query_records = getattr(self.session, "query_records", None)
            self._query_records = query_records() if not self.ovd and query_records else []
            self._query_ids = {record.query_id: record for record in self._query_records}
            self._query_locations = {(record.scene_instance_id, record.property_id, record.element_index): record
                                     for record in self._query_records}
        return self._query_records

    def query_record(self, query_id):
        self.query_records()
        return self._query_ids.get(query_id)

    def query_identity(self, source_instance_id, property_id, element_index):
        self.query_records()
        record = self._query_locations.get((source_instance_id, property_id, element_index))
        return record.query_id if record else None

    def ownership_graph(self):
        """Return canonical child IDs and owning-scene sets for presentation tags.

        Property wrappers are transparent. Explicit reference aliases never
        become ownership edges; a shared material cannot pull its users into a
        hide/selection cascade. The scene projection applies native ownership
        before forward lists, including recording segment identity.
        Child tuples and owner frozensets are immutable; consumers may shallow
        copy the mappings and replace individual values to supplement edges.
        """
        if self._ownership is None:
            nodes, roots = self.hierarchy("scene")
            children = {node.instance_id: [] for node in nodes.values() if node.kind == "object"}
            owners = {}
            stack = [(key, None, None) for key in roots]
            while stack:
                key, parent, scene = stack.pop()
                node = nodes[key]
                if node.kind == "reference":
                    continue
                if node.kind == "object":
                    object_id = node.instance_id
                    if parent is not None:
                        children[parent].append(object_id)
                    if node.class_name == "PxScene":
                        scene = object_id
                    owners[object_id] = frozenset((scene,)) if scene is not None else frozenset()
                    parent = object_id
                stack.extend((child, parent, scene) for child in reversed(node.children))
            self._ownership = ({key: tuple(value) for key, value in children.items()}, owners)
        return self._ownership

    def name(self, instance_id):
        if instance_id not in self.names:
            instance = self.instances[instance_id]
            prop = self.property(instance_id, "name")
            availability = getattr(self.session, "property_availability", None)
            available = not (prop and availability
                             and availability(instance_id, prop.id) != "present")
            name = self.session.string_value(instance_id, prop.id) if prop and prop.type == "StringHandle" and instance.alive and available else ""
            self.names[instance_id] = name or self.classes[instance.class_id].name
        return self.names[instance_id]

    def value(self, instance_id, prop, precision=6):
        if not self.instances[instance_id].alive:
            return "Not alive at this frame"
        if prop.type == "ObjectRef":
            refs = self.references(instance_id, prop.name)
            return ", ".join(f"{self.name(i)} [{i}]" if i in self.instances else f"Unresolved [{i}]" for i in refs[:16]) or "—"
        if prop.type == "StringHandle":
            return self.session.string_value(instance_id, prop.id) or "—"
        size = self.session.property_size(instance_id, prop.id)
        raw = self.session.read(instance_id, prop.id, limit=128)
        if not raw:
            return "—"
        floating = prop.type in ("PvdF32", "PxVec3", "PxQuat", "PxTransform", "PxBounds3", "PxMat33", "PxMat44")
        if floating and len(raw) % 4 == 0:
            values = struct.unpack(f"<{len(raw)//4}f", raw)
            text = ", ".join(f"{value:.{precision}g}" for value in values[:16])
        elif prop.type == "PvdF64" and len(raw) % 8 == 0:
            text = ", ".join(f"{value:.{precision}g}" for value in struct.unpack(f"<{len(raw)//8}d",raw)[:16])
        elif len(raw) in (1, 2, 4, 8):
            text = str(int.from_bytes(raw, "little", signed="I" in prop.type))
        else:
            text = raw[:32].hex(" ")
        return text + (f" … [{size:,} bytes]" if size > 128 else "")

    def draw_objects(self):
        result = []
        unsupported = set()
        for actor in self.instances.values():
            if not actor.alive:
                continue
            actor_pose = self.raw(actor.id, "globalPose")
            if not actor_pose:
                continue
            for shape_id in self.references(actor.id, "shapes"):
                shape = self.instances.get(shape_id)
                if not shape or not shape.alive:
                    continue
                geometry_ids = self.references(shape_id, "geom") or self.references(shape_id, "geometry")
                for geometry_id in geometry_ids:
                    geometry = self.instances.get(geometry_id)
                    if not geometry or not geometry.alive:
                        continue
                    kind = self.classes[geometry.class_id].name
                    matrix = pose_matrix(actor_pose) @ pose_matrix(self.raw(shape_id, "localPose"))
                    if kind == "PxSphereGeometry":
                        radius = self.raw(geometry_id, "radius")
                        if len(radius) == 4:
                            result.append(DrawObject(actor.id, "sphere", matrix, struct.unpack("<f", radius)))
                    elif kind == "PxBoxGeometry":
                        extents = self.raw(geometry_id, "halfExtents")
                        if len(extents) == 12:
                            result.append(DrawObject(actor.id, "box", matrix, struct.unpack("<3f", extents)))
                    else:
                        unsupported.add(kind)
        return result, sorted(unsupported)

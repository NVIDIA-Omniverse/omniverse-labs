# SPDX-License-Identifier: Apache-2.0
"""Selection-independent evaluated timeline values, on Blender's main thread."""
from dataclasses import replace
from types import SimpleNamespace

from .blender_interactive_edit_builders import build_interactive_edits_from_depsgraph
from .interactive_edit_planner import EditShape, DataAuthority


def evaluated_value_edits(scene, depsgraph, resolver, *, material_owners=None):
    """Sample existing typed values only; topology stays in depsgraph routing."""
    objects = tuple(getattr(scene, "objects", ()))
    materials = {id(material): material for material in (material_owners or ())}
    updates = []
    for obj in objects:
        if str(getattr(obj, "type", "")) != "CAMERA" and _animated_object(obj):
            updates.append(SimpleNamespace(id=obj, is_updated_geometry=False))
        if material_owners is None:
            for slot in getattr(obj, "material_slots", ()):
                material = getattr(slot, "material", None)
                if material is not None:
                    materials[id(material)] = material
    # Only animated materials require sampling each frame. Ordinary edits
    # already arrive through the depsgraph callback.
    for material in materials.values():
        tree = getattr(material, "node_tree", None)
        if getattr(material, "animation_data", None) or getattr(tree, "animation_data", None):
            updates.append(SimpleNamespace(id=material, is_updated_geometry=False))
    world = getattr(scene, "world", None)
    if world is not None and (getattr(world, "animation_data", None)
                             or getattr(getattr(world, "node_tree", None), "animation_data", None)):
        updates.append(SimpleNamespace(id=world, is_updated_geometry=False))
    graph = SimpleNamespace(updates=updates, evaluated_depsgraph=depsgraph)
    edits = build_interactive_edits_from_depsgraph(graph, usd_prim_resolver=resolver,
        light_objects=objects, materials=tuple(materials.values()), worlds=(world,) if world else ())
    return tuple(replace(edit, provenance={**edit.provenance, "evaluation_reason": "timeline"})
                 for edit in edits if edit.shape == EditShape.VALUE and edit.data_authority == DataAuthority.VIEW)


def _animated_object(obj):
    """Keep constraints and animated parent chains without static edit builds."""
    seen = set()
    current = obj
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if (getattr(current, "animation_data", None) or getattr(current, "constraints", ())
                or getattr(getattr(current, "data", None), "animation_data", None)):
            return True
        current = getattr(current, "parent", None)
    return False


class TimelineValueCache:
    def __init__(self):
        self.context = None
        self.values = {}
        self.material_context = None
        self.material_owners = ()

    def materials_for(self, scene, context):
        if context != self.material_context:
            result = {}
            for obj in getattr(scene, "objects", ()):
                for slot in getattr(obj, "material_slots", ()):
                    material = getattr(slot, "material", None)
                    if material is not None:
                        result[id(material)] = material
            self.material_owners = tuple(result.values())
            self.material_context = context
        return self.material_owners

    def changed(self, context, edits):
        if context != self.context:
            self.context = context
            self.values.clear()
        return tuple(edit for edit in edits if self.values.get(self.key(edit)) != repr(edit.value))

    @staticmethod
    def key(edit):
        return (edit.provenance.get("blender_id_kind"), edit.provenance.get("blender_session_uid"),
                edit.usd_prim_path, edit.usd_attribute)

    def accepted(self, edit):
        self.values[self.key(edit)] = repr(edit.value)

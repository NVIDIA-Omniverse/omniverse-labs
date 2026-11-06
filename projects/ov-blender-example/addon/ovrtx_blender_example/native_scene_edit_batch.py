# SPDX-License-Identifier: Apache-2.0
"""Coalesce native RNA selection/variant recalls into precise generation IDs.

No USD authoring or runtime calls happen here. Ordinary interactive value edits
retain their normal depsgraph/scheduler route. This explicit batch boundary is
for native tools performing many datablock assignments in one operation.
"""
from contextlib import contextmanager

_generation_snapshots = {}


def _snapshot(scene):
    from . import scene_generation as generation
    from .material_graph_fingerprint import _graph_signature, world_graph_signature
    from .object_visibility import scene_visibility
    values = {}
    visibility = {generation.blender_id(obj,'OBJECT'): value for obj,value in scene_visibility(scene)}
    for obj in getattr(scene,'objects',()):
        kind = str(getattr(obj,'type',''))
        if kind not in generation.SUPPORTED_OBJECT_TYPES and kind != 'EMPTY':
            continue
        identity = generation.blender_id(obj,'OBJECT')
        data = getattr(obj,'data',None)
        mesh_inputs = generation._mesh_render_inputs(obj,data,include_elements=False) if kind == 'MESH' else ()
        instance = getattr(obj,'instance_collection',None)
        values[identity] = (kind,int(getattr(data,'session_uid',0) or 0),
            tuple(int(getattr(slot.material,'session_uid',0) or 0) for slot in getattr(obj,'material_slots',())),
            int(getattr(instance,'session_uid',0) or 0),visibility.get(identity,False),
            bool(getattr(obj,'visible_shadow',True)),repr(mesh_inputs))
        for slot in getattr(obj,'material_slots',()):
            material = slot.material
            identity = None if material is None else generation.blender_id(material,'MATERIAL')
            if identity is not None and identity not in values:
                values[identity] = _graph_signature(material,native_principled_values=False)
    world = getattr(scene,'world',None)
    if world is not None:
        values[generation.blender_id(world,'WORLD')] = world_graph_signature(world)
    return values


def remember_generation_snapshot(scene, generation):
    """Bind an RNA snapshot to a source candidate; acceptance remains the owner’s."""
    key = (int(scene.session_uid),generation.number,generation.digest)
    if key not in _generation_snapshots:
        _generation_snapshots[key] = _snapshot(scene)
    from .scene_generation_sessions import _owners
    owner = _owners.get(int(scene.session_uid))
    retained = {key}
    if owner is not None:
        for item in (owner.current_generation,getattr(owner,'pending_generation',None)):
            if item is not None:
                retained.add((int(scene.session_uid),item.number,item.digest))
    for previous in tuple(_generation_snapshots):
        if previous[0] == key[0] and previous not in retained:
            del _generation_snapshots[previous]


def _accepted_snapshot(scene):
    from .scene_generation_sessions import _owners
    owner = _owners.get(int(scene.session_uid))
    generation = None if owner is None else owner.current_generation
    if generation is None:
        return None
    return _generation_snapshots.get((int(scene.session_uid),generation.number,generation.digest))


def _queue_difference(scene, before, after):
    affected = {identity for identity in set(before)|set(after)
                if before.get(identity) != after.get(identity)}
    if affected:
        from .scene_generation_sessions import mark_scene_dirty
        if not mark_scene_dirty(scene,affected):
            raise RuntimeError('Native edit batch generation admission failed')
    return affected


def refresh_native_scene(scene):
    """Admit deferred native edits against the last accepted source snapshot."""
    before = _accepted_snapshot(scene)
    # A scene without an accepted source will be exported from current RNA on
    # its first capture. No invented predecessor or unscoped dirty request.
    if before is None:
        return set()
    return _queue_difference(scene,before,_snapshot(scene))


@contextmanager
def native_scene_edit_batch(scene):
    """Queue actual native edits, preserving failed-candidate predecessor state."""
    before = _accepted_snapshot(scene)
    if before is None:
        before = _snapshot(scene)
    yield
    _queue_difference(scene,before,_snapshot(scene))

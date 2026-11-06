# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Main-thread, immutable physics authoring observations; never exports or sends RPCs."""
from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any


@dataclass(frozen=True)
class PhysicsSnapshot:
    # (source kind, session uid, bone name, property path, explicit opinion, value)
    leaves: tuple[tuple, ...]


@dataclass(frozen=True)
class PhysicsChanges:
    restart: bool = False
    structural: bool = False
    redraw: bool = False
    pending_anchor: bool = False
    body_input: bool = False
    changed_sources: frozenset[tuple[str, int]] = frozenset()


def _uid(value: Any) -> int:
    return int(getattr(value, "session_uid", 0) or 0)


def _value(value: Any) -> Any:
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    return tuple(_value(item) for item in value)


def _leaves(group: Any, prefix: str = ""):
    """Read every registered leaf, including default values and sparse flags.

    Unset ID pointers must not be read: Blender can materialize an explicit
    null opinion merely by evaluating their getter.
    """
    if group is None:
        return
    for prop in group.bl_rna.properties:
        name = prop.identifier
        if name in {"rna_type"}:
            continue
        path = prefix + name
        explicit = bool(group.is_property_set(name))
        if prop.type == "POINTER":
            if prop.fixed_type.identifier.startswith("Ov"):
                yield from _leaves(getattr(group, name), path + ".")
            else:
                target = getattr(group, name) if explicit else None
                yield path, explicit, (_uid(target), getattr(target, "name_full", ""))
        elif prop.type == "COLLECTION":
            collection = getattr(group, name)
            yield path + ".__count__", explicit, len(collection)
            for index, item in enumerate(collection):
                yield from _leaves(item, f"{path}[{index}].")
        else:
            yield path, explicit, _value(getattr(group, name))


_render_init_capture = threading.local()


class render_init_capture:
    """Admit captures on Blender's render job thread during ``render_init``.

    Blender 5.3 runs ``render_init`` on the job thread; the final render
    request (including its stage export) is built there.
    """

    def __enter__(self):
        _render_init_capture.active = True
        return self

    def __exit__(self, *_exc):
        _render_init_capture.active = False


def _authored(group: Any) -> bool:
    """Inspect persisted RNA opinions without walking every registered default.

    Reading default nested groups can create empty containers. Such pointer
    containers are not opinions; their stored descendants still determine
    participation. Explicit default/null leaves and empty explicit collections
    preserve the original full-leaf predicate's semantics.
    """
    if group is None:
        return False
    keys = getattr(group, 'keys', None)
    if not callable(keys):
        return any(explicit and not path.endswith('show_indicator')
                   or path.endswith('.__count__') and value
                   for path, explicit, value in _leaves(group))
    for name in keys():
        prop = group.bl_rna.properties.get(name)
        if prop is None or name == 'rna_type':
            continue
        if prop.type == 'POINTER' and prop.fixed_type.identifier.startswith('Ov'):
            if _authored(getattr(group, name)):
                return True
        elif prop.type == 'COLLECTION':
            if group.is_property_set(name) or len(getattr(group, name)):
                return True
        elif name != 'show_indicator' and group.is_property_set(name):
            return True
    return False


#: Scene ``ov`` settings that are not physics authoring (never restart physics).
NON_PHYSICS_SCENE_SETTINGS = frozenset({"sync_playback_to_ovrtx_frames"})


def scene_authoring_observation(scene: Any) -> PhysicsSnapshot:
    """Constant-size scene controls, including unset/reset and unit scale."""
    if (threading.current_thread() is not threading.main_thread()
            and not getattr(_render_init_capture, 'active', False)):
        raise RuntimeError('physics authoring observations require Blender\'s main thread')
    leaves = [('SCENE', _uid(scene), '', path, explicit, value)
              for path, explicit, value in _leaves(getattr(scene, 'ov', None))
              if path not in NON_PHYSICS_SCENE_SETTINGS]
    leaves.append(('SCENE', _uid(scene), '', 'unit_scale', True,
                   float(getattr(getattr(scene, 'unit_settings', None), 'scale_length', 1))))
    return PhysicsSnapshot(tuple(sorted(leaves)))


def capture(scene: Any) -> PhysicsSnapshot:
    if (threading.current_thread() is not threading.main_thread()
            and not getattr(_render_init_capture, "active", False)):
        raise RuntimeError("physics authoring snapshots require Blender's main thread")
    leaves = list(scene_authoring_observation(scene).leaves)
    def add(kind, uid, bone, path, explicit, value):
        leaves.append((kind, uid, bone, path, explicit, value))
    def group(kind, owner, bone, value):
        for path, explicit, value in _leaves(value):
            add(kind, _uid(owner), bone, path, explicit, value)
    materials = {}
    mesh_sources = set()
    legacy_materials = set()
    for obj in scene.objects:
        physics = getattr(getattr(obj, "ov", None), "physics", None)
        if physics is None:
            continue
        # Defaults attached by registration do not make a render Object a
        # physics participant. Preserve ordinary camera/light/topology handling.
        # Explicit default-valued opinions still count, as do newly authored
        # leaves on an Object that was absent from the previous observation.
        bones = tuple(getattr(getattr(obj, "pose", None), "bones", ()))
        legacy = any(str(key).startswith(("pxr:usd:physics_", "pxr:physics:")) for key in obj.keys())
        for material in getattr(getattr(obj, "data", None), "materials", ()):
            if material is not None and _authored(material.ov.physics):
                materials[_uid(material)] = material
        if not (_authored(physics) or legacy or any(_authored(bone.ov.physics) for bone in bones)):
            continue
        group("OBJECT", obj, "", obj.ov)
        add("OBJECT", _uid(obj), "", "membership", True,
            (obj.type, _uid(obj.data), _uid(obj.parent), obj.name_full,
             tuple(sorted(collection.name_full for collection in obj.users_collection))))
        if physics.simready_version == 0:
            for key in obj.keys():
                if str(key).startswith(("pxr:usd:physics_", "pxr:physics:")):
                    add("OBJECT", _uid(obj), "", "simready." + key, True, _value(obj[key]))
            if any(str(key).startswith("pxr:usd:physics_") for key in obj.keys()):
                for constraint in obj.constraints:
                    add("OBJECT", _uid(obj), "", "simready_constraint." + constraint.name,
                        True, (constraint.type, _uid(getattr(constraint, "target", None)), constraint.influence))
                legacy_materials.update(_uid(m) for m in getattr(getattr(obj, "data", None), "materials", ()) if m is not None)
        joint = physics.joint
        if joint.persistent_id or physics.rigid_body.schema_opinion == "APPLY":
            add("OBJECT", _uid(obj), "", "handle_pose" if joint.persistent_id else "body_pose",
                True, tuple(tuple(float(v) for v in row) for row in obj.matrix_world))
        if physics.rigid_body.schema_opinion == "APPLY":
            add("OBJECT", _uid(obj), "", "body_basis", True,
                tuple(tuple(float(v) for v in row) for row in obj.matrix_basis))
            add("OBJECT", _uid(obj), "", "body_scale", True,
                tuple(float(v) for v in obj.matrix_world.to_scale()))
        targets = [obj]
        for pose_bone in getattr(getattr(obj, "pose", None), "bones", ()):
            group("OBJECT", obj, pose_bone.name, pose_bone.ov)
            group("OBJECT", obj, pose_bone.name + ":definition", pose_bone.bone.ov)
            bone = pose_bone.bone
            add("OBJECT", _uid(obj), pose_bone.name, "rest_topology", True,
                (bone.parent.name if bone.parent else "", bone.inherit_scale,
                 bool(bone.use_connect), bool(bone.use_inherit_rotation),
                 tuple(float(v) for v in bone.head_local),
                 tuple(float(v) for v in bone.tail_local),
                 tuple(tuple(float(v) for v in row) for row in bone.matrix_local)))
            targets.append(pose_bone)
            if pose_bone.ov.physics.rigid_body.schema_opinion == "APPLY":
                add("OBJECT", _uid(obj), pose_bone.name, "body_basis", True,
                    tuple(tuple(float(v) for v in row) for row in pose_bone.matrix_basis))
                add("OBJECT", _uid(obj), pose_bone.name, "body_pose", True,
                    tuple(tuple(float(v) for v in row) for row in obj.matrix_world @ pose_bone.matrix))
        for target in targets:
            p = target.ov.physics
            for record in p.colliders:
                if "mesh_source" in record.bl_rna.properties and record.is_property_set("mesh_source"):
                    mesh_sources.add(_uid(record.mesh_source))
                if record.is_property_set("physics_material"):
                    material = record.physics_material
                    if material is not None:
                        materials[_uid(material)] = material
        for material in getattr(getattr(obj, "data", None), "materials", ()):
            if material is not None:
                materials[_uid(material)] = material
    for material in materials.values():
        group("MATERIAL", material, "", material.ov)
        if _uid(material) in legacy_materials:
            for key in material.keys():
                if str(key).startswith("pxr:usd:physics_"):
                    add("MATERIAL", _uid(material), "", "simready." + key, True, _value(material[key]))
    for obj in scene.objects:
        if _uid(obj) in mesh_sources:
            add("OBJECT", _uid(obj), "", "collider_source_pose", True,
                tuple(tuple(float(v) for v in row) for row in obj.matrix_world))
    return PhysicsSnapshot(tuple(sorted(leaves)))


_LEGACY_PHYSICS_KEYS = ("pxr:usd:physics_", "pxr:physics:")
_NO_SYSTEM_PROPERTIES = object()


def _stored_physics(owner: Any) -> Any:
    """``owner``'s raw stored ``ov.physics`` IDProperty group, without RNA wrappers.

    Returns None when nothing is stored and ``_NO_SYSTEM_PROPERTIES`` when
    the owner exposes no ``bl_system_properties_get`` (older Blender, fakes).
    Never creates storage (``do_create`` stays False).
    """
    get = getattr(owner, 'bl_system_properties_get', None)
    if not callable(get):
        return _NO_SYSTEM_PROPERTIES
    group = get()
    for name in ('ov', 'physics'):
        if group is None:
            return None
        group = group.get(name)
    return group


def _stored_leaf(values: Any) -> bool:
    """Any stored non-group value below ``values`` (a superset of ``_authored``)."""
    for name, value in values.items():
        if isinstance(value, dict):
            if _stored_leaf(value):
                return True
        elif name != 'show_indicator':
            return True
    return False


def _owner_authored(owner: Any) -> bool:
    """``_authored(owner.ov.physics)``, with a cheap raw-storage pre-filter.

    Nested groups created merely by reading defaults hold no leaves, so the
    common physics-free owner is rejected from raw storage alone. Any stored
    leaf is confirmed through the exact RNA predicate (stale keys of
    unregistered properties are not opinions).
    """
    stored = _stored_physics(owner)
    if stored is None:
        return False
    if stored is not _NO_SYSTEM_PROPERTIES and not _stored_leaf(stored.to_dict()):
        return False
    return _authored(getattr(getattr(owner, 'ov', None), 'physics', None))


def object_participates(obj: Any, data_materials: dict | None = None) -> bool:
    """Whether ``obj`` has stored physics opinions that ``capture`` would observe.

    Same predicate as ``capture``'s participant test (Object, pose bones, legacy
    keys, materials of its data), ordered cheapest first. ``data_materials``
    caches the material verdict per data-block for one batch, so objects
    sharing a mesh read its materials once. Unguarded: callers are main-thread
    observers that already passed a guarded observation.
    """
    if _owner_authored(obj):
        return True
    if any(str(key).startswith(_LEGACY_PHYSICS_KEYS) for key in obj.keys()):
        return True
    pose = getattr(obj, 'pose', None)
    if pose is not None and any(_owner_authored(bone) for bone in pose.bones):
        return True
    data = getattr(obj, 'data', None)
    if data is None:
        return False
    key = getattr(data, 'session_uid', None) if data_materials is not None else None
    if key is not None and key in data_materials:
        return data_materials[key]
    authored = any(material is not None and _owner_authored(material)
                   for material in getattr(data, 'materials', ()))
    if key is not None:
        data_materials[key] = authored
    return authored


def _physics_path(path: str) -> bool:
    return path.startswith('ov.') or '.ov.' in path


def _action_fcurves(action: Any):
    layers = getattr(action, 'layers', None)
    if layers is None:
        return tuple(getattr(action, 'fcurves', ()))
    return tuple(curve for layer in layers for strip in layer.strips
                 for bag in strip.channelbags for curve in bag.fcurves)


# Animation data-path verdicts. Drivers: (owner uid) -> (driver count, verdict),
# re-read when the count changes. Actions: (action uid) -> verdict, read once
# and forgotten when depsgraph_update_post reports the Action edited
# (``forget_animation_paths``). Bounded: cleared when it grows large.
_DRIVER_PATHS: dict = {}
_ACTION_PATHS: dict = {}


def forget_animation_paths(action_uids) -> None:
    """Drop cached Action verdicts for edited Actions (latch-only bookkeeping)."""
    for uid in action_uids:
        _ACTION_PATHS.pop(uid, None)


def animation_targets_physics(owner: Any) -> bool:
    """Whether ``owner``'s Action or drivers animate an ``ov`` physics path.

    Reads only the owner's AnimData. Driver paths are re-read when the
    owner's driver count changes; an Action's curves are read once until an
    edit of that Action is reported. Raw curve edits that report nothing are
    left to idle discovery, which sees the written-back value.
    """
    animation = getattr(owner, 'animation_data', None)
    if animation is None:
        return False
    for cache in (_DRIVER_PATHS, _ACTION_PATHS):
        if len(cache) > 65536:
            cache.clear()
    drivers = animation.drivers
    count = len(drivers)
    if count:
        uid = _uid(owner)
        cached = _DRIVER_PATHS.get(uid)
        if cached is None or cached[0] != count:
            cached = _DRIVER_PATHS[uid] = (count, any(_physics_path(c.data_path) for c in drivers))
        if cached[1]:
            return True
    action = animation.action
    if action is None:
        return False
    uid = _uid(action)
    verdict = _ACTION_PATHS.get(uid)
    if verdict is None:
        verdict = _ACTION_PATHS[uid] = any(_physics_path(c.data_path) for c in _action_fcurves(action))
    return verdict


class RenderOnlyDiscovery:
    """Bound idle discovery of raw RNA edits that do not publish callbacks.

    Raw Python collection edits (``colliders.add()``, ``joint.axes.add()``),
    ``property_unset`` and animation write-back call no RNA ``update`` and
    tag no depsgraph update, so no Blender hook reports them. Discovery probes
    objects round-robin from the 0.2 s main-thread timer: every call probes at
    least ``MIN_BATCH`` objects (or the whole roster, if smaller) and continues
    while ``TIME_BUDGET_S`` lasts, up to one pass, so a full pass over a large
    scene takes a few ticks rather than N/128. The roster is refreshed at the
    end of every pass, so Objects linked while the depsgraph handler was
    skipped are probed within the next pass.

    Created only after a complete physics-free observation. Normal authoring
    callbacks and depsgraph changes discard it; explicit admission still does
    a complete capture. Retained RNA references live only until that next
    observation or scene teardown, and freed references force re-observation.
    """
    MIN_BATCH = 128
    TIME_BUDGET_S = 0.004

    def __init__(self, scene):
        self.scene = scene
        self.objects = tuple(scene.objects)
        self.index = 0
        self.passes = 0
        self.last_probed = 0

    def changed(self, *, min_batch: int | None = None, time_budget_s: float | None = None,
                clock=time.perf_counter) -> bool:
        """Probe the next Objects; True as soon as one participates.

        One call probes at most one pass (each roster Object once, crossing
        the end of the roster at most once), at least ``min_batch`` Objects,
        and otherwise stops when the time budget is spent.
        """
        min_batch = self.MIN_BATCH if min_batch is None else min_batch
        deadline = clock() + (self.TIME_BUDGET_S if time_budget_s is None else time_budget_s)
        data_materials: dict = {}
        probed = 0
        self.last_probed = 0
        try:
            start, before = self.index, len(self.objects)
            limit = max(1, before)
            while probed < limit:
                if self.index >= len(self.objects):
                    # Pass complete: refresh the roster so Objects linked
                    # without an observed depsgraph update join the next pass.
                    self.passes += 1
                    self.objects = tuple(self.scene.objects)
                    self.index = 0
                    # Continue up to where this call started, plus additions.
                    limit = probed + max(0, min(len(self.objects), start + len(self.objects) - before))
                    if probed >= limit:
                        return False
                obj = self.objects[self.index]
                self.index += 1
                probed += 1
                self.last_probed = probed
                if object_participates(obj, data_materials):
                    return True
                if probed >= min_batch and clock() >= deadline:
                    return False
            return False
        except ReferenceError:
            return True


def classify(before: PhysicsSnapshot, after: PhysicsSnapshot) -> PhysicsChanges:
    old = {row[:4]: row[4:] for row in before.leaves}
    new = {row[:4]: row[4:] for row in after.leaves}
    changed = {key for key in old.keys() | new.keys() if old.get(key) != new.get(key)}
    redraw = pending = body = restart = structural = False
    sources = set()
    for key in changed:
        path = key[3]
        if path.endswith("show_indicator") or path.startswith(("indicator_", "indicators_")):
            redraw = True
        elif path == "handle_pose":
            pending = redraw = True
        elif path in {"body_pose", "body_basis"}:
            body = True  # existing body-input transport retains ownership/reset rules
        else:
            restart = True
            sources.add(key[:2])
            if (key not in old or key not in new or path in {"membership", "rest_topology", "unit_scale", "body_scale"}
                or any(token in path for token in ("endpoint", "persistent_id", "collider", "schema_opinion", "mesh_source"))
                or path.endswith(("joint.type", "joint.axis", "rigid_body.role"))):
                structural = True
    return PhysicsChanges(restart, structural, redraw, pending, body, frozenset(sources))


def absorb_presentation(before: PhysicsSnapshot, after: PhysicsSnapshot) -> PhysicsSnapshot:
    """Advance only transient transform observations; never swallow authoring edits."""
    transient = {row[:4]: row for row in after.leaves if row[3] in {"body_pose", "body_basis", "body_scale", "handle_pose", "collider_source_pose"}}
    return PhysicsSnapshot(tuple(transient.get(row[:4], row) for row in before.leaves))

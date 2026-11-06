# SPDX-License-Identifier: Apache-2.0
"""Bounded structural projection plans; stable USD identities after reparenting.

World-pose packets change the rendered hierarchy's effective result without
namespace relocation or re-export. Blender retains the authored hierarchy.
Only acknowledged plans mutate cached parent/child indices. Unsupported lifetime,
instance, schema and physics changes return to scoped population.
"""
from dataclasses import dataclass, field
import math


def _identity(obj):
    from .scene_generation import blender_id
    return blender_id(obj, 'OBJECT')


def _key(generation):
    return generation.number, generation.digest


def _parent(obj):
    parent = getattr(obj, 'parent', None)
    return _identity(parent) if parent is not None else None


def _shape_controls(obj):
    return (str(obj.name), str(obj.type), getattr(getattr(obj,'data',None),'session_uid',None),
            str(getattr(obj,'instance_type','NONE')), bool(getattr(obj,'rigid_body',None)),
            bool(getattr(obj,'rigid_body_constraint',None)))


@dataclass
class StructureContract:
    generation_key: tuple
    objects: dict
    paths: dict
    parents: dict
    children: dict
    controls: dict
    eligible: frozenset
    revision: int = 0


@dataclass(frozen=True)
class ReparentPlan:
    values: tuple = ()
    fallback: tuple = ()
    handled: frozenset = frozenset()
    affected: frozenset = frozenset()
    parent_changes: tuple = ()
    parent_session_uids: dict = field(default_factory=dict)
    generation_key: tuple = ()
    predecessor_revision: int = 0


def capture_contract(stage, generation, objects, *, children=None):
    """Use the already captured object map; no scene traversal or evaluation."""
    paths,parents,controls,eligible = {},{},{},set()
    own_children = children is None
    if own_children:
        children = {}
    for identity,obj in objects.items():
        parent = _parent(obj)
        parents[identity] = parent
        if own_children and parent is not None:
            children.setdefault(parent,set()).add(identity)
        controls[identity] = _shape_controls(obj)
        mapping = generation.blender_prim_paths.get(identity)
        if mapping is None:
            continue
        paths[identity] = str(mapping.object_path)
        if obj.type not in {'MESH','LIGHT','CAMERA','EMPTY'} or controls[identity][3] != 'NONE':
            continue
        prim = stage.GetPrimAtPath(mapping.schema_path)
        if not prim:
            continue
        family = str(prim.GetTypeName())
        # A mesh owner with mixed GN/instance output does not have one Mesh leaf.
        if obj.type == 'MESH' and family != 'Mesh':
            continue
        if obj.type == 'EMPTY' and family not in {'Xform','Scope'}:
            continue
        eligible.add(identity)
    return StructureContract(_key(generation),objects,paths,parents,children,controls,frozenset(eligible))


def descendant_closure(children, roots, *, parent_changes=()):
    """Walk only affected descendants, with proposed edges applied virtually."""
    touched = {}
    for identity, old, new in parent_changes:
        if old is not None:
            touched.setdefault(old,set(children.get(old,()))).discard(identity)
        if new is not None:
            touched.setdefault(new,set(children.get(new,()))).add(identity)
    seen = set(roots)
    pending = list(roots)
    while pending:
        parent=pending.pop()
        for child in touched.get(parent,children.get(parent,())):
            if child not in seen:
                seen.add(child);pending.append(child)
    return frozenset(seen)


def plan_reparent(generation, changed_identities, contract, *, depsgraph, pending=None):
    """Plan current effective poses, including coalesced A→B→A reversals.

    `pending` is the last unacknowledged ReparentPlan, if any. It preserves the
    need to send a reversal even when the final parent equals the accepted one.
    Nominated IDs and cached descendant edges bound all work.
    """
    def reject(reason):
        return ReparentPlan(fallback=(reason,),generation_key=contract.generation_key,
                            predecessor_revision=contract.revision)
    if _key(generation) != contract.generation_key:
        return reject('structure_generation_changed')
    if pending is not None and (pending.generation_key != contract.generation_key or
                                pending.predecessor_revision != contract.revision):
        return reject('structure_pending_plan_stale')
    nominated=set(changed_identities)
    pending_roots={change[0] for change in pending.parent_changes} if pending else set()
    nominated.update(pending_roots)
    changes=[]
    for identity in nominated:
        obj=contract.objects.get(identity)
        if obj is None:
            return reject('structure_owner_not_populated')
        try:
            parent=_parent(obj)
            previous=contract.parents.get(identity)
            if parent==previous and identity not in pending_roots:
                continue
            if identity not in contract.eligible or _shape_controls(obj)!=contract.controls[identity]:
                return reject('structure_schema_or_namespace_changed')
            if parent is not None and parent not in contract.paths:
                return reject('structure_new_parent_not_populated')
            if parent is not None and any(contract.controls[parent][4:]):
                return reject('structure_physics_requires_population')
            changes.append((identity,previous,parent))
        except (ReferenceError,AttributeError):
            return reject('structure_owner_removed')
    roots={change[0] for change in changes}
    if not roots:
        return ReparentPlan(generation_key=contract.generation_key,predecessor_revision=contract.revision)
    affected=descendant_closure(contract.children,roots,parent_changes=changes)
    # A Blender cycle is normally impossible; explicit contracts can also come
    # from tests or future adapters, so guard only the nominated parent chains.
    proposed={identity:new for identity,old,new in changes}
    for root in roots:
        visited={root};parent=proposed.get(root,contract.parents.get(root))
        while parent is not None:
            if parent in visited:
                return reject('structure_parent_cycle')
            visited.add(parent);parent=proposed.get(parent,contract.parents.get(parent))
    rows=[]
    try:
        for identity in affected:
            obj=contract.objects.get(identity)
            if obj is None or identity not in contract.eligible:
                return reject('structure_descendant_requires_population')
            if _shape_controls(obj)!=contract.controls[identity]:
                return reject('structure_schema_or_namespace_changed')
            if any(contract.controls[identity][4:]):
                return reject('structure_physics_requires_population')
            evaluated=obj.evaluated_get(depsgraph)
            matrix=evaluated.matrix_world
            pose=tuple(tuple(float(matrix[j][i]) for j in range(4)) for i in range(4))
            if not all(math.isfinite(value) for row in pose for value in row):
                return reject('structure_nonfinite_pose')
            path=contract.paths[identity]
            rows.extend((dict(prim_path=path,attribute_name='omni:resetXformStack',value_type='Bool',value=True),
                         dict(prim_path=path,attribute_name='omni:xform',value_type='Matrix4d',value=pose)))
    except (ReferenceError,AttributeError,ValueError,TypeError):
        return reject('structure_evaluated_pose_unavailable')
    sessions={identity:int(getattr(getattr(contract.objects[identity],'parent',None),'session_uid',0))
              for identity,old,new in changes}
    return ReparentPlan(tuple(rows),(),frozenset(roots),affected,tuple(changes),sessions,
                        contract.generation_key,contract.revision)


def acknowledge(contract, plan):
    """Call only after the worker acknowledges the complete effective-pose packet."""
    if plan.fallback or plan.generation_key!=contract.generation_key or plan.predecessor_revision!=contract.revision:
        raise ValueError('Cannot acknowledge stale or rejected structure plan')
    for identity,old,new in plan.parent_changes:
        if old is not None:
            contract.children.setdefault(old,set()).discard(identity)
        if new is not None:
            contract.children.setdefault(new,set()).add(identity)
        contract.parents[identity]=new
    if plan.parent_changes:
        contract.revision+=1

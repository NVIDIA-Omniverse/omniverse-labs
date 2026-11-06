# SPDX-License-Identifier: Apache-2.0
"""Render visibility on existing leaf Mesh/Light prims, without scene rescans.

The capture runs once after population. Plan reads nominated object handles and
emits the standard USD visibility token. It never authors derived visibility.
Collection/layer membership notifications remain the coordinator's fallback;
object viewport hiding is a visibility value under the existing export policy.
Instances and prim lifetime changes are deliberately excluded.
Adopt next_contract only after the native packet is acknowledged.
"""
from dataclasses import dataclass, replace, field
from . import object_visibility

LIGHT_TYPES = frozenset({'SphereLight', 'DistantLight', 'RectLight', 'DiskLight', 'CylinderLight', 'DomeLight'})

@dataclass(frozen=True)
class State:
    path: str
    token: str
    controls: tuple
    render_enabled: bool

@dataclass(frozen=True)
class VisibilityContract:
    generation_key: tuple
    states: dict
    handles: dict
    view_layer: object = None
    render_eligibility: dict = field(default_factory=dict)

@dataclass(frozen=True)
class VisibilityPlan:
    values: tuple
    fallback: tuple
    handled: frozenset
    next_contract: VisibilityContract


def _identity(obj):
    from .scene_generation import blender_id
    return blender_id(obj, 'OBJECT')


def _key(generation):
    return generation.number, generation.digest


def _eye_hidden(obj, layer):
    getter = getattr(obj, 'hide_get', None)
    return bool(getter(view_layer=layer) if layer is not None else getter()) if callable(getter) else False


def _controls(obj, layer):
    return (getattr(getattr(obj, 'parent', None), 'session_uid', None),
            getattr(getattr(obj, 'data', None), 'session_uid', None),
            str(obj.type), str(getattr(obj, 'instance_type', 'NONE')),
            bool(getattr(obj, 'hide_viewport', False)), _eye_hidden(obj, layer))


def _viewport_visible(obj, layer):
    if bool(getattr(obj, 'hide_viewport', False)) or _eye_hidden(obj, layer):
        return False
    getter = getattr(obj, 'visible_get', None)
    return bool(getter(view_layer=layer) if layer is not None else getter()) if callable(getter) else True


def capture_contract(stage, scene, generation, *, view_layer=None):
    layer = object_visibility.scene_view_layer(scene, view_layer=view_layer)
    enabled = set()
    def visit(collection, hidden=False):
        hidden = hidden or bool(getattr(collection, 'hide_render', False))
        if not hidden:
            enabled.update(int(obj.session_uid) for obj in collection.objects)
        for child in collection.children:
            visit(child, hidden)
    root = getattr(scene, 'collection', None)
    if root is not None:
        visit(root)
    states, handles, eligibility = {}, {}, {}
    for obj in scene.objects:
        if obj.type not in {'MESH', 'LIGHT'} or getattr(obj, 'instance_type', 'NONE') != 'NONE':
            continue
        identity = _identity(obj)
        eligibility[identity] = root is None or int(obj.session_uid) in enabled
        mapping = generation.blender_prim_paths.get(identity)
        if mapping is None:
            continue
        prim = stage.GetPrimAtPath(mapping.schema_path)
        if not prim or str(prim.GetTypeName()) not in ({'Mesh'} if obj.type == 'MESH' else LIGHT_TYPES):
            continue
        attribute = prim.GetAttribute('visibility')
        token = str(attribute.Get()) if attribute else ''
        if token not in {'inherited', 'invisible'}:
            continue
        states[identity] = State(str(mapping.schema_path), token, _controls(obj, layer),
                                 root is None or int(obj.session_uid) in enabled)
        handles[identity] = obj
    return VisibilityContract(_key(generation), states, handles, layer, eligibility)


def plan(scene, generation, changed_identities, contract, *, depsgraph=None):
    if _key(generation) != contract.generation_key:
        return VisibilityPlan((), ('visibility_generation_changed',), frozenset(), contract)
    rows, reasons, handled = [], [], set()
    states = dict(contract.states)
    for identity in set(changed_identities):
        state = contract.states.get(identity)
        if state is None:
            reasons.append('visibility_target_not_supported')
            continue
        original = contract.handles[identity]
        try:
            controls = _controls(original, contract.view_layer)
            if controls[:4] != state.controls[:4]:
                raise ValueError('visibility_structure_changed')
            evaluated = original.evaluated_get(depsgraph) if depsgraph is not None else original
            is_visible = (state.render_enabled and _viewport_visible(original, contract.view_layer)
                          and not bool(evaluated.hide_render))
            token = 'inherited' if is_visible else 'invisible'
            if token != state.token:
                rows.append(dict(prim_path=state.path, attribute_name='visibility', value_type='Token', value=token))
            if token != state.token or controls != state.controls:
                states[identity] = replace(state, token=token, controls=controls)
            handled.add(identity)
        except (ReferenceError, ValueError, AttributeError, TypeError) as exc:
            reasons.append(str(exc))
    if reasons:
        return VisibilityPlan((), tuple(dict.fromkeys(reasons)), frozenset(), contract)
    return VisibilityPlan(tuple(rows), (), frozenset(handled), replace(contract, states=states))

# SPDX-License-Identifier: Apache-2.0
"""Conservative, stage-cached admission for native material updates.

No USD exports or stage opens occur in plan(). The owner must adopt next_contract
only after the whole native batch succeeds, and recapture after any population.
"""
from dataclasses import dataclass, replace, field
from typing import Any, Mapping
from .material_value_conversion import usd_attribute_values
from .usd_value_edit_support import PRINCIPLED_PREVIEW_SURFACE_VALUE_SPECS
from .material_graph_fingerprint import material_graph_signature
from .material_usd_prim import _material_prim_index_from_prims, resolve_material_usd_prim


@dataclass(frozen=True)
class MaterialState:
    graph: str
    values: tuple
    targets: Mapping[str, str]


@dataclass(frozen=True)
class BindingState:
    path: str
    target: str
    control: tuple
    face_slots: tuple | None


@dataclass(frozen=True)
class MaterialContract:
    generation_key: tuple
    materials: Mapping[Any, MaterialState]
    bindings: Mapping[Any, BindingState]
    material_paths: Mapping[Any, str]
    material_handles: Mapping[Any, Any] = field(default_factory=dict)
    object_handles: Mapping[Any, Any] = field(default_factory=dict)
    mesh_users: Mapping[Any, tuple] = field(default_factory=dict)
    path_users: Mapping[str, tuple] = field(default_factory=dict)
    effective_slots: Mapping[Any, tuple] = field(default_factory=dict)


@dataclass(frozen=True)
class MaterialPlan:
    values: tuple
    fallback: tuple[str, ...]
    handled: frozenset
    next_contract: MaterialContract


def _helpers():
    from .scene_generation import blender_id, effective_object_materials, object_export_materials, _mesh_shading_inputs
    return blender_id, effective_object_materials, object_export_materials, _mesh_shading_inputs


def _scene_materials(scene):
    identity, _, exported, _ = _helpers()
    result = {}
    # Include mapped unused materials when Blender is available, permitting a
    # later binding to an already exported target without relying on names.
    try:
        import bpy
        materials = list(bpy.data.materials)
    except ImportError:
        materials = list(getattr(scene, 'materials', ()))
    for obj in scene.objects:
        materials.extend(exported(obj))
    for material in materials:
        if material is not None:
            result[identity(material, 'MATERIAL')] = material
    return result


def _key(generation):
    return generation.number, generation.digest


def _values(material):
    return tuple((v.name, v.value_type, v.value) for v in usd_attribute_values(material))


def _control(obj):
    # Remove only material identities/link modes from the accepted cheap mesh
    # controls. Everything else (modifiers, parents, normals controls, physics,
    # names, visibility, data identity) remains an admission proof.
    _, _, _, shading = _helpers()
    signature = list(shading(obj))
    signature[3] = ()
    # SimReady carries material UID even when a slot has no physics opinions.
    # Preserve identities for authored physics materials: changing those needs
    # regeneration of material:binding:physics, outside the visual value route.
    physics = list(signature[4])
    if physics:
        physics[1] = tuple((0, entry[1], entry[2])
                           if entry and not entry[1] and entry[2] == 'INHERIT'
                           else entry for entry in physics[1])
        signature[4] = tuple(physics)
    render = list(signature[8]); render[6] = (); signature[8] = tuple(render)
    return tuple(signature)


def _single_target(obj, paths):
    identity, slots, _, _ = _helpers()
    materials = slots(obj)
    if len(materials) != 1 or materials[0] is None:
        return ''
    return paths.get(identity(materials[0], 'MATERIAL'), '')


def capture_contract(stage, scene, generation, *, material_path_overrides=None):
    """Capture from the owner's already opened, accepted composed USD stage."""
    identity, _, _, _ = _helpers()
    overrides = material_path_overrides or {}
    material_paths = {key: overrides.get(mapping.schema_path, mapping.schema_path)
                      for key, mapping in generation.blender_prim_paths.items()
                      if key.kind == 'MATERIAL' and str(stage.GetPrimAtPath(
                          overrides.get(mapping.schema_path, mapping.schema_path)).GetTypeName()) == 'Material'}
    materials = {}
    live_materials = _scene_materials(scene)
    for key, material in live_materials.items():
        if key not in material_paths:
            continue
        values = _values(material); targets = {}
        # Known identity permits a bounded material-subtree walk. Never build
        # whole-stage N-material x N-shader candidate tables for every input.
        from pxr import Usd
        path = material_paths[key]
        prims = tuple(Usd.PrimRange(stage.GetPrimAtPath(path)))
        for name, _, _ in values:
            index = _material_prim_index_from_prims(
                prims, usd_attribute=name, schema_default_material_paths=(path,))
            resolved = resolve_material_usd_prim(material, index, usd_attribute=name,
                                                 known_prim_path=path)
            target = getattr(resolved, 'value', None)
            if target is not None and not target.connected:
                targets[name] = target.prim_path
        preview = {'inputs:' + n for n, _ in PRINCIPLED_PREVIEW_SURFACE_VALUE_SPECS.values()}
        if any(name not in preview for name in targets):
            values = tuple(v for v in values if v[0] not in preview)
        elif targets:
            values = tuple(v for v in values if v[0] in preview)
        fingerprints = getattr(generation, 'material_graph_fingerprints', {})
        graph = fingerprints[key] if key in fingerprints else material_graph_signature(material)
        materials[key] = MaterialState(graph, values, targets)
    bindings = {}
    for obj in scene.objects:
        key = identity(obj, 'OBJECT'); mapping = generation.blender_prim_paths.get(key)
        if mapping is None or obj.type != 'MESH' or tuple(getattr(obj, 'modifiers', ())):
            continue
        target = _single_target(obj, material_paths)
        if not target:
            continue
        prim = stage.GetPrimAtPath(mapping.schema_path)
        # Native relationship editing cannot synthesize/remove geometry subsets
        # or safely rewrite inherited/collection binding precedence.
        if any(str(child.GetTypeName()) == 'GeomSubset' for child in prim.GetChildren()):
            continue
        rel = prim.GetRelationship('material:binding')
        if not rel or tuple(str(p) for p in rel.GetTargets()) != (target,):
            continue
        if any(str(r.GetName()) != 'material:binding' and str(r.GetName()).startswith('material:binding')
               for r in prim.GetRelationships()):
            continue
        # Uniform exported binding + one effective slot establishes initial
        # ownership. Read current polygon slots only if a binding edit occurs.
        bindings[key] = BindingState(mapping.schema_path, target, _control(obj), None)
    objects = {identity(obj, 'OBJECT'): obj for obj in scene.objects}
    mesh_users = {}; path_users = {}
    for key, obj in objects.items():
        if obj.type == 'MESH':
            mesh_users.setdefault(identity(obj.data, 'MESH'), []).append(key)
    for key, obj in objects.items():
        mapping = generation.blender_prim_paths.get(key)
        if obj.type == 'MESH' and mapping is not None:
            path_users.setdefault(mapping.schema_path, []).append(key)
    return MaterialContract(_key(generation), materials, bindings, material_paths,
                            live_materials, objects,
                            {k: tuple(v) for k, v in mesh_users.items()},
                            {k: tuple(v) for k, v in path_users.items()},
                            {k: effective_slot_signature(o) for k, o in objects.items()
                             if o.type == 'MESH'})


def plan(scene, generation, changed_identities, contract, *, update_provenance=None):
    """Return atomic rows or explicit fallback; never partially handle a batch.

    OBJECT/MESH notifications require explicit shading-only provenance so a
    simultaneous geometry/transform change cannot be swallowed by this route.
    """
    identity, _, _, _ = _helpers()
    affected = frozenset(changed_identities); errors = []; rows = []
    if contract.generation_key != _key(generation):
        return MaterialPlan((), ('stale material contract',), frozenset(), contract)
    material_states = dict(contract.materials); binding_states = dict(contract.bindings)
    live_materials = contract.material_handles
    objects = contract.object_handles
    touched_paths = set()
    for key in affected:
        if key.kind == 'MATERIAL':
            old = contract.materials.get(key); material = live_materials.get(key)
            if old is None or material is None:
                errors.append('material missing from accepted generation'); continue
            if material_graph_signature(material) != old.graph:
                errors.append('material graph changed'); continue
            previous = {n: (t, v) for n, t, v in old.values}
            values = tuple(v for v in _values(material) if v[0] in previous)
            if set(previous) != {n for n, _, _ in values}:
                errors.append('material value schema changed'); continue
            for name, value_type, value in values:
                if previous[name] == (value_type, value):
                    continue
                path = old.targets.get(name)
                if path is None:
                    errors.append('unsupported or ambiguous shader input: ' + name); continue
                rows.append(dict(prim_path=path, attribute_name=name, value_type=value_type, value=value))
            material_states[key] = replace(old, values=values)
        elif key.kind in {'OBJECT', 'MESH'}:
            provenance = (update_provenance or {}).get(key)
            if (provenance is None or provenance.geometry is not False
                    or provenance.transform is not False or provenance.shading is not True):
                errors.append('binding edit lacks shading-only provenance'); continue
            owners = ([objects[key]] if key in objects else
                      [objects[k] for k in contract.mesh_users.get(key, ())])
            if not owners:
                errors.append('binding owner missing'); continue
            for obj in owners:
                owner = identity(obj, 'OBJECT'); old = contract.bindings.get(owner)
                if old is None:
                    errors.append('binding requires subsets or uncaptured owner'); continue
                if _control(obj) != old.control:
                    errors.append('binding geometry or export controls changed'); continue
                if old.face_slots is None:
                    valid_faces = all(int(p.material_index) == 0 for p in obj.data.polygons)
                else:
                    valid_faces = tuple(int(p.material_index) for p in obj.data.polygons) == old.face_slots
                if not valid_faces:
                    errors.append('binding geometry or export controls changed'); continue
                target = _single_target(obj, contract.material_paths)
                if not target:
                    errors.append('binding target not already exported or slots changed'); continue
                binding_states[owner] = replace(old, target=target)
                touched_paths.add(old.path)
        else:
            errors.append('notification outside material category: ' + key.kind)
    # A shared USD mesh cannot acquire contradictory per-object overrides.
    path_targets = {}
    for key in {key for path in touched_paths for key in contract.path_users.get(path, ())}:
        state = binding_states.get(key)
        if state is None:
            errors.append('shared USD mesh includes uncaptured binding owner'); continue
        # Inspect only affected aliases, including those omitted from notifications.
        obj = objects.get(key)
        current = _single_target(obj, contract.material_paths) if obj is not None else ''
        if current != state.target:
            errors.append('unnotified binding owner changed')
        path_targets.setdefault(state.path, set()).add(state.target)
    for targets in path_targets.values():
        if len(targets) != 1:
            errors.append('shared USD mesh has conflicting object bindings')
    for key in {key for path in touched_paths for key in contract.path_users.get(path, ())}:
        state = binding_states.get(key)
        if state is None:
            continue
        if state.target != contract.bindings[key].target:
            rows.append(dict(prim_path=state.path, attribute_name='material:binding',
                             value_type='Relationship', value=[state.target]))
    if errors:
        return MaterialPlan((), tuple(dict.fromkeys(errors)), frozenset(), contract)
    unique = {(row['prim_path'], row['attribute_name']): row for row in rows}
    slots = dict(contract.effective_slots)
    for key in changed_effective_slots(contract, affected):
        slots[key] = effective_slot_signature(contract.object_handles[key])
    proposed = replace(contract, materials=material_states, bindings=binding_states, effective_slots=slots)
    return MaterialPlan(tuple(unique.values()), (), affected, proposed)


def capture_composed_contract(source_stage, scene, generation, presentation_layer=None, *, already_composed=False):
    """Compose translator material overlay on its original root without exports.

    Call immediately after baseline translation, before subsequent authoring.
    Anonymous layers/stage are temporary and never saved or passed to runtime.
    """
    if presentation_layer is None:
        return capture_contract(source_stage, scene, generation)
    from pxr import Sdf, Usd
    overlay = Sdf.Layer.CreateAnonymous('direct-material-overlay.usda')
    if not overlay.ImportFromString('#usda 1.0\n' + presentation_layer.layer_body):
        raise ValueError('invalid material presentation overlay')
    if already_composed:
        stage = source_stage
    else:
        root = Sdf.Layer.CreateAnonymous('direct-material-contract.usda')
        root.subLayerPaths = [overlay.identifier, source_stage.GetRootLayer().identifier]
        stage = Usd.Stage.Open(root, sessionLayer=source_stage.GetSessionLayer())
    overrides = {}
    for record in presentation_layer.diagnostics.get('materials', ()):
        # MaterialX layers rebind to generated materials and record a mapping
        # per material; the USD Preview repair layer patches materials in place
        # and lists only their names, so it has no path overrides.
        if not isinstance(record, Mapping):
            continue
        generated = record.get('generated_material_path', '')
        source = record.get('identity', {}).get('material_path', '')
        if not source:
            source = record.get('identity', {}).get('source_usd_path', '')
        if generated and source:
            overrides[source] = generated
    return capture_contract(stage, scene, generation, material_path_overrides=overrides)


def effective_slot_signature(obj):
    """Effective DATA/OBJECT assignments, independent of mesh.data.materials.

    Captured for every original mesh owner, including unsupported modifier or
    subset owners. A slot assignment must not disappear behind geometry fallback.
    """
    identity, effective, _, _ = _helpers()
    slots = getattr(obj, 'material_slots', None)
    materials = effective(obj)
    return tuple((str(getattr(slots[i], 'link', 'DATA')) if slots is not None else 'DATA',
                  identity(material, 'MATERIAL') if material is not None else None)
                 for i, material in enumerate(materials))


def changed_effective_slots(contract, changed_identities):
    """Return original OBJECT IDs whose effective slot state differs from ACK.

    Coordinator must route these owners through the material binding planner and
    respect its fallback before accepting mesh-only rows. Geometry extraction of
    original data.materials cannot represent OBJECT overrides. No scene scan.
    """
    owners = set()
    for key in changed_identities:
        if key.kind == 'OBJECT' and key in contract.effective_slots:
            owners.add(key)
        elif key.kind == 'MESH':
            owners.update(contract.mesh_users.get(key, ()))
    return frozenset(key for key in owners if key in contract.effective_slots
                     and effective_slot_signature(contract.object_handles[key]) != contract.effective_slots[key])

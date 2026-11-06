# SPDX-License-Identifier: Apache-2.0
"""Current Blender projection compiled to native values before USD reconciliation.

One state per accepted generation; no edit archive. Unsupported output rejoins
population with every directly changed owner so stale bootstrap USD cannot win.
"""
from dataclasses import dataclass, field
from types import SimpleNamespace
import time
from .direct_projection_ack import ProjectionAck

# Renderer population/prototype/hierarchy behavior is proven in the isolated zoo.
INSTANCE_GRAPHS_ENABLED = True


@dataclass
class DirectState:
    generation_key: tuple
    stage: object
    layers: tuple
    objects: dict
    data_users: dict
    materials: object
    presentation: tuple
    mesh_contracts: dict = field(default_factory=dict)
    dirty: set = field(default_factory=set)
    values: dict = field(default_factory=dict)
    revision: int = 0
    applied_revision: int = -1
    diagnostics: dict = field(default_factory=dict)
    lights: object = None
    cameras: object = None
    controls: dict = field(default_factory=dict)
    node_users: dict = field(default_factory=dict)
    has_physics: bool = False
    scene_uid: int = 0
    pending_materials: object = None
    pending_lights: object = None
    children: dict = field(default_factory=dict)
    ack: ProjectionAck = field(default_factory=ProjectionAck)
    visibility: object = None
    pending_visibility: object = None
    pending_graphs: dict = field(default_factory=dict)
    structure: object = None
    pending_structure: object = None
    instance_sources: dict = field(default_factory=dict)
    pending_instances: dict = field(default_factory=dict)
    mapping_overrides: dict = field(default_factory=dict)
    instance_reference_users: dict = field(default_factory=dict)
    instance_token: object = None
    pending_controls: dict = field(default_factory=dict)
    visibility_only_ids: set = field(default_factory=set)


def _key(generation):
    return generation.number, generation.digest


def initialize(owner, scene, generation, request):
    """Capture contracts once, on the main thread before any following edit."""
    current = getattr(owner, '_direct_state', None)
    if current is not None and current.generation_key == _key(generation):
        return current
    from pxr import Sdf, Usd
    from .scene_generation import blender_id
    from . import material_direct_updates
    root = Sdf.Layer.CreateAnonymous('direct-contract.usda')
    layers = []
    for contribution in (request.material_scene_layer, request.light_scene_layer):
        if contribution is not None:
            layer = Sdf.Layer.CreateAnonymous('direct-presentation.usda')
            body = contribution.layer_body
            layer.ImportFromString(body if body.lstrip().startswith('#usda') else '#usda 1.0\n' + body)
            layers.append(layer)
    root.subLayerPaths = [layer.identifier for layer in layers] + [generation.usd_path]
    stage = Usd.Stage.Open(root)
    objects = {blender_id(obj, 'OBJECT'): obj for obj in scene.objects}
    users = {}
    for identity, obj in objects.items():
        data = getattr(obj, 'data', None)
        if data is not None:
            users.setdefault(int(data.session_uid), set()).add(identity)
    state = DirectState(_key(generation), stage, (root, *layers), objects, users,
                        material_direct_updates.capture_composed_contract(stage, scene, generation, request.material_scene_layer,
                                                                         already_composed=True),
                        (request.material_scene_layer, request.light_scene_layer))
    if current is not None and current.diagnostics:
        state.diagnostics = dict(current.diagnostics)
    try:
        from . import direct_light_updates
        state.lights = direct_light_updates.capture_contract(stage, scene, generation)
    except ImportError:
        pass
    owner._direct_state = state
    # One bootstrap classification includes native Physics/Physx authorship,
    # which may have no matching Blender rigid_body property. No edit-time scan.
    state.has_physics = any(any(str(schema).startswith(('Physics', 'Physx'))
        for schema in prim.GetAppliedSchemas()) for prim in stage.TraverseAll())
    state.scene_uid = int(scene.session_uid)
    from . import direct_object_updates
    for key, obj in objects.items():
        state.has_physics |= bool(getattr(obj, 'rigid_body', None))
        if obj.parent is not None:
            state.children.setdefault(blender_id(obj.parent, 'OBJECT'), set()).add(key)
        state.controls[key] = direct_object_updates.controls(obj)
        for mod in getattr(obj, 'modifiers', ()):
            tree = getattr(mod, 'node_group', None)
            if tree is not None: state.node_users.setdefault(int(tree.session_uid), set()).add(key)
    try:
        from . import direct_camera_updates
        state.cameras = direct_camera_updates.capture_contract(stage, scene, generation)
    except ImportError:
        pass
    from . import direct_visibility_updates
    state.visibility = direct_visibility_updates.capture_contract(stage, scene, generation)
    from . import direct_structure_updates
    state.structure = direct_structure_updates.capture_contract(stage, generation, state.objects, children=state.children)
    return state


def presentation(owner, generation):
    state = getattr(owner, '_direct_state', None)
    from . import scene_generation_sessions
    return state.presentation if (state is not None and state.generation_key == _key(generation)
        and state.scene_uid in scene_generation_sessions._camera_capture_owners) else None


def require_projection_for_other_consumer(owner, scene):
    from . import scene_generation_sessions
    state = getattr(owner, '_direct_state', None)
    if state is None or not state.dirty:
        return
    uid = int(scene.session_uid)
    if uid not in scene_generation_sessions._camera_capture_owners:
        scene_generation_sessions._dirty.setdefault(uid, set()).update(state.dirty)


def fallback_ids(owner, affected):
    state = getattr(owner, '_direct_state', None)
    return set(affected) | (state.dirty if state is not None else set())


def _pose(obj, path, graph):
    evaluated = obj.evaluated_get(graph)
    # The existing runtime transform contract takes row-vector world matrices
    # with reset boundaries, and handles inherited/instance fanout centrally.
    matrix = tuple(tuple(float(evaluated.matrix_world[j][i]) for j in range(4)) for i in range(4))
    return [dict(prim_path=path, attribute_name='omni:resetXformStack', value_type='Bool', value=True),
            dict(prim_path=path, attribute_name='omni:xform', value_type='Matrix4d', value=matrix)]


def visibility_only_owners(state, generation, object_ids, affected, provenance):
    """Prove visibility separately from Mesh contents, using nominated handles.

    Blender may flag object geometry when hiding it. Never use that exception
    for a dirty Mesh, changed geometry controls, unknown provenance, or a
    transform-dependent modifier. No vertex reads or scene traversal.
    """
    from .scene_generation import blender_id, _mesh_shading_inputs
    from . import direct_visibility_updates as visibility
    contract = state.pending_visibility or state.visibility
    result = set()
    if contract is None:
        return result
    for key in object_ids:
        obj = state.objects.get(key)
        before = getattr(generation, 'mesh_shading_inputs', {}).get(key)
        old = contract.states.get(key)
        note = provenance.get(key)
        if obj is None or obj.type != 'MESH' or before is None or old is None or note is None:
            continue
        if note.geometry is None or note.transform is None or note.shading is None:
            continue
        # Modifiers can depend on another owner's geometry/pose while their
        # own RNA inputs stay unchanged. A simultaneous hide must not mask it.
        if (note.geometry or note.transform) and getattr(obj, 'modifiers', ()):
            continue
        mesh_id = blender_id(obj.data, 'MESH')
        mesh_note = provenance.get(mesh_id)
        if mesh_id in affected and (mesh_note is None or mesh_note.geometry is not False):
            continue
        current_controls = visibility._controls(obj, contract.view_layer)
        if current_controls[:4] != old.controls[:4]:
            continue
        current = _mesh_shading_inputs(obj)
        if before[:6] + before[8:] != current[:6] + current[8:]:
            continue
        # A geometry-only notification is not itself proof of a visibility edit.
        token = 'inherited' if (old.render_enabled and visibility._viewport_visible(obj, contract.view_layer)
                                and not obj.hide_render) else 'invisible'
        if token != old.token or current_controls != old.controls or before[6:8] != current[6:8]:
            result.add(key)
    return result


def plan(owner, scene, generation, affected, provenance):
    state = getattr(owner, '_direct_state', None)
    if state is not None:
        state.visibility_only_ids.clear()
    if state is None or state.generation_key != _key(generation):
        return False
    from . import scene_generation_sessions
    # This lane has one serialized capture owner. Viewport/physics producers
    # keep their existing scheduler until their direct-packet admission is tested.
    if int(getattr(scene, 'session_uid', 0)) not in scene_generation_sessions._camera_capture_owners:
        return False
    started = time.perf_counter()
    from .scene_generation import (blender_id, _affected_object_ids,
                                  _proven_non_geometry_objects, _empty_evaluated_mesh_proofs)
    from . import material_direct_updates, direct_mesh_updates
    import bpy
    affected = set(affected)
    from . import direct_object_updates
    owner_changes = direct_object_updates.plan(state.objects, state.controls, state.data_users,
        state.pending_controls, {key for key in affected if key.kind == 'OBJECT'})
    reasons = []
    rows = []
    def reject():
        state.diagnostics = {'route': 'population_fallback',
            'fallback_reasons': tuple(dict.fromkeys(reasons)),
            'affected_count': len(affected), 'value_count': 0,
            'owner_lookup_count': owner_changes.examined,
            'compile_ms': (time.perf_counter()-started)*1000}
        return False
    pending_materials = state.pending_materials if state.pending_materials is not None else state.materials
    pending_lights = state.pending_lights if state.pending_lights is not None else state.lights
    visibility_ids = set()
    graph_updates = ()
    instance_sources = {}
    if any(key.kind in {'PHYSICS', 'COLLECTION', 'SCENE'} for key in affected):
        reasons.append('membership_or_physics_notification')
    if len(scene.objects) != generation.scene_object_count:
        reasons.append('object_membership_changed')
    object_ids = set()
    material_ids = set()
    light_ids = set()
    camera_ids = set()
    if reasons:
        return reject()
    for key in affected:
        if state.lights is not None and key.kind in {'NODETREE', 'IMAGE'} and key in state.lights.users:
            light_ids.add(key); continue
        if key.kind == 'OBJECT': object_ids.add(key)
        elif key.kind == 'MATERIAL': material_ids.add(key)
        elif key.kind == 'WORLD': light_ids.add(key)
        elif key.kind in {'MESH', 'CURVE', 'LIGHT', 'CAMERA', 'NODETREE'}:
            users = owner_changes.users.get(key.session_uid, state.data_users.get(key.session_uid)) or state.node_users.get(key.session_uid)
            if users: object_ids.update(users)
            elif key.kind != 'NODETREE': reasons.append('unmapped_data_owner:' + key.kind + ':' + str(key.session_uid))
            else: reasons.append('unmapped_node_tree')
        else: reasons.append('unsupported_notification:' + key.kind)
    for key in affected:
        object_ids.update(state.instance_reference_users.get(key, ()))
        for pending_source in state.pending_instances.values():
            if key in pending_source.reference_identities:
                object_ids.add(pending_source.owner_identity)
    state.visibility_only_ids = visibility_only_owners(state, generation, object_ids, affected, provenance)
    if reasons:
        return reject()
    if state.has_physics and any(getattr(state.objects.get(key), 'type', '') not in {'CAMERA', 'LIGHT'} for key in object_ids):
        reasons.append('affected_physics_consumer_requires_population')
        return reject()
    if material_ids:
        result = material_direct_updates.plan(scene, generation, material_ids, pending_materials,
                                              update_provenance=provenance)
        if result.fallback:
            from . import material_graph_updates
            compiled = material_graph_updates.plan(scene, generation, material_ids, pending_materials)
            if compiled.fallback:
                reasons.extend(compiled.fallback)
                return reject()
            graph_updates = compiled.updates
            pending_materials = compiled.next_contract
        else:
            rows.extend(result.values); pending_materials=result.next_contract
    graph = bpy.context.evaluated_depsgraph_get()
    reparent = None
    if state.structure is not None:
        from . import direct_structure_updates
        reparent = direct_structure_updates.plan_reparent(generation, object_ids, state.structure,
            depsgraph=graph, pending=state.pending_structure)
        if reparent.fallback:
            reasons.extend(reparent.fallback)
            return reject()
        rows.extend(reparent.values)
        object_ids.update(reparent.affected)
    ancestors = [key for key in object_ids if key not in provenance or provenance[key].transform is not False]
    descendants = set()
    while ancestors:
        for child in state.children.get(ancestors.pop(), ()):
            if child not in descendants:
                descendants.add(child); ancestors.append(child)
    object_ids.update(descendants)
    # Parent updates may nominate mesh descendants even when the initial event
    # was a light/camera; physics admission belongs after that closure too.
    if state.has_physics and any(getattr(state.objects.get(key), 'type', '') not in {'CAMERA', 'LIGHT'} for key in object_ids):
        reasons.append('affected_physics_consumer_requires_population')
        return reject()
    changed_bindings = material_direct_updates.changed_effective_slots(pending_materials, object_ids)
    nominated = SimpleNamespace(objects=tuple(state.objects[key] for key in object_ids if key in state.objects))
    proven = _proven_non_geometry_objects(nominated, generation, affected, object_ids, provenance or {})
    proven.update(state.visibility_only_ids)
    proven.difference_update(owner_changes.swapped)
    empty = _empty_evaluated_mesh_proofs(nominated, object_ids - proven)
    for key in object_ids:
        obj = state.objects.get(key)
        active_instance = state.pending_instances.get(key) or state.instance_sources.get(key)
        mapping = active_instance.mapping if active_instance is not None else state.mapping_overrides.get(key, generation.blender_prim_paths.get(key))
        if obj is None or mapping is None:
            reasons.append('object_not_populated'); continue
        try:
            if obj.name != mapping.blender_id_name:
                reasons.append('object_namespace_changed'); continue
            controls = direct_object_updates.controls(obj)
            old_controls = state.controls.get(key)
            if reparent is not None and key in reparent.handled and old_controls is not None:
                old_controls = (controls[0], *old_controls[1:])
            if controls != old_controls:
                # Visibility and a Mesh datablock replacement do not alter the
                # existing owner/schema identity. Geometry is compiled below.
                if (old_controls is not None and controls[0] == old_controls[0]
                        and controls[2] == old_controls[2] and controls[5:] == old_controls[5:]
                        and (controls[1] == old_controls[1] or obj.type == 'MESH')
                        and obj.type in {'MESH', 'LIGHT'} and state.visibility is not None):
                    visibility_ids.add(key)
                else:
                    reasons.append('object_structure_or_visibility_changed'); continue
            if state.visibility is not None and key in state.visibility.states:
                visibility_ids.add(key)
            if obj.type == 'LIGHT':
                light_ids.add(key); continue
            if obj.type == 'CAMERA':
                if state.cameras is not None:
                    camera_ids.add(key); continue
                # Active camera parameters are carried in the RenderRequest.
                # Other camera changes must not silently disappear.
                if obj != scene.camera:
                    reasons.append('inactive_camera_changed'); continue
                rows.extend(_pose(obj, mapping.object_path, graph)); continue
            if obj.type not in {'MESH', 'EMPTY'}:
                reasons.append('unsupported_object_type:' + obj.type); continue
            if str(getattr(obj, 'instance_type', 'NONE')) != 'NONE':
                reasons.append('instance_owner_requires_population'); continue
            note = (provenance or {}).get(key)
            instance_candidate = obj.type == 'MESH' and (active_instance is not None or
                (key not in proven and any(mod.type == 'NODES' for mod in getattr(obj, 'modifiers', ()))))
            if instance_candidate and INSTANCE_GRAPHS_ENABLED:
                from . import direct_instance_updates, direct_visibility_updates
                if active_instance is not None and note is not None and note.geometry is False and note.shading is False:
                    if note.transform is not False: rows.extend(_pose(obj, mapping.object_path, graph))
                    continue
                visibility_contract = state.pending_visibility if state.pending_visibility is not None else state.visibility
                old_visibility = visibility_contract.states.get(key) if visibility_contract is not None else None
                render_enabled = (old_visibility.render_enabled if old_visibility is not None else
                    (visibility_contract.render_eligibility.get(key, True) if visibility_contract is not None else True))
                layer = visibility_contract.view_layer if visibility_contract is not None else None
                visible = render_enabled and direct_visibility_updates._viewport_visible(obj, layer)
                source = direct_instance_updates.compile_owner(obj, graph, generation,
                    projection_path=active_instance.projection_path if active_instance is not None else None,
                    render_visible=visible, allow_mesh_only=active_instance is not None)
                if not source.fallback:
                    instance_sources[key] = source
                    visibility_ids.discard(key)
                    continue
                if source.fallback != ('owner_has_no_instance_component',) or active_instance is not None:
                    reasons.extend(source.fallback)
                    continue
            binding_changed = False
            if obj.type == 'MESH' and key in changed_bindings:
                from .scene_generation import SceneUpdateProvenance
                bind = material_direct_updates.plan(scene, generation, {key}, pending_materials,
                    update_provenance={key: SceneUpdateProvenance(False, False, True)})
                if bind.fallback:
                    reasons.extend(bind.fallback)
                    continue
                rows.extend(bind.values)
                pending_materials = bind.next_contract
                binding_changed = bool(bind.values)
            if obj.type == 'MESH' and key not in proven:
                if empty.get(key) and empty.get(key) == generation.empty_evaluated_mesh_proofs.get(key):
                    if note is None or note.transform is not False:
                        rows.extend(_pose(obj, mapping.object_path, graph))
                    continue
                # Material assignment is independent from vertex data. Ask its
                # category proof before material-slot changes reach mesh export.
                if not binding_changed or note is None or note.geometry is not False:
                    with direct_mesh_updates.evaluated_mesh(obj, graph) as mesh:
                        contract = state.mesh_contracts.get(key)
                        if contract is None:
                            material_paths = tuple(generation.blender_prim_paths.get(blender_id(getattr(m, 'original', m), 'MATERIAL')).schema_path
                                if m and blender_id(getattr(m, 'original', m), 'MATERIAL') in generation.blender_prim_paths else '' for m in mesh.materials)
                            contract = direct_mesh_updates.contract_from_usd(state.stage.GetPrimAtPath(mapping.schema_path), mesh,
                                material_paths=material_paths, unit_scale=1.0)
                            state.mesh_contracts[key] = contract
                        if binding_changed and not contract.subsets:
                            from dataclasses import replace
                            contract = replace(contract, material_slots=direct_mesh_updates.material_slot_ids(mesh))
                        patch = direct_mesh_updates.build_mesh_patch(mesh, contract)
                    if not patch.supported:
                        reasons.append(patch.reason); continue
                    rows.extend(direct_mesh_updates.wire_values(patch))
            if note is None or note.transform is not False:
                rows.extend(_pose(obj, mapping.object_path, graph))
        except (ValueError, ReferenceError, RuntimeError) as error:
            reasons.append(str(error))
    if light_ids:
        if state.lights is None:
            reasons.append('light_direct_contract_unavailable')
        else:
            from . import direct_light_updates
            result = direct_light_updates.plan(scene, generation, light_ids, pending_lights, depsgraph=graph)
            reasons.extend(result.fallback); rows.extend(result.values); pending_lights=result.next_contract
    if camera_ids:
        from . import direct_camera_updates
        result = direct_camera_updates.plan(scene, generation, camera_ids, state.cameras, depsgraph=graph)
        reasons.extend(result.fallback); rows.extend(result.values)
    pending_visibility = state.pending_visibility if state.pending_visibility is not None else state.visibility
    if owner_changes.swapped and pending_visibility is not None:
        from dataclasses import replace
        states = dict(pending_visibility.states)
        for key in owner_changes.swapped:
            if key in states:
                old = states[key]
                states[key] = replace(old, controls=(old.controls[0], owner_changes.controls[key][1], *old.controls[2:]))
        pending_visibility = replace(pending_visibility, states=states)
    if reparent is not None and reparent.handled and pending_visibility is not None:
        from dataclasses import replace
        states = dict(pending_visibility.states)
        for key in reparent.handled:
            if key in states:
                old = states[key]
                parent_uid = reparent.parent_session_uids[key] or None
                states[key] = replace(old, controls=(parent_uid, *old.controls[1:]))
        pending_visibility = replace(pending_visibility, states=states)
    if visibility_ids:
        from . import direct_visibility_updates
        result = direct_visibility_updates.plan(scene, generation, visibility_ids, pending_visibility, depsgraph=graph)
        reasons.extend(result.fallback); rows.extend(result.values); pending_visibility = result.next_contract
    if instance_sources:
        from dataclasses import replace
        from . import direct_visibility_updates
        if pending_visibility is not None:
            states = dict(pending_visibility.states)
            handles = dict(pending_visibility.handles)
            for key, source in instance_sources.items():
                obj = state.objects[key]
                old = states.get(key)
                enabled = old.render_enabled if old is not None else pending_visibility.render_eligibility.get(key, True)
                evaluated = obj.evaluated_get(graph)
                visible = enabled and direct_visibility_updates._viewport_visible(obj, pending_visibility.view_layer) and not bool(evaluated.hide_render)
                states[key] = direct_visibility_updates.State(source.projection_path,
                    'inherited' if visible else 'invisible', direct_visibility_updates._controls(obj, pending_visibility.view_layer), enabled)
                handles[key] = obj
            pending_visibility = replace(pending_visibility, states=states, handles=handles)
        # The inline current source already contains these owners' current
        # poses/visibility. Never replay the old owner's visible token after it.
        replaced = {source.hidden_source_path for source in instance_sources.values()}
        projection_roots = {source.projection_path for source in instance_sources.values()}
        rows = [row for row in rows if not (row['prim_path'] in replaced and row['attribute_name']=='visibility')
                and not any(row['prim_path']==root or row['prim_path'].startswith(root+'/') for root in projection_roots)]
    unique = {}
    for row in rows:
        target = (row['prim_path'], row['attribute_name'])
        old = unique.get(target)
        if old is not None:
            import numpy as np
            if old['value_type'] != row['value_type'] or not np.array_equal(old['value'], row['value']):
                reasons.append('conflicting_shared_native_target:' + '.'.join(target))
        unique[target] = row
    state.diagnostics = {'route': 'direct_values' if not reasons else 'population_fallback',
        'fallback_reasons': tuple(dict.fromkeys(reasons)), 'affected_count':len(affected),
        'owner_lookup_count': owner_changes.examined, 'mesh_swap_count':len(owner_changes.swapped),
        'visibility_only_count':len(state.visibility_only_ids),
        'value_count':len(rows), 'material_graph_count':len(graph_updates), 'instance_graph_count':len(instance_sources), 'reparent_count':len(reparent.handled) if reparent else 0, 'compile_ms':(time.perf_counter()-started)*1000}
    if reasons: return False
    for update in graph_updates:
        root = update['material_path']
        stale = [key for key in state.values if key[0] == root or key[0].startswith(root + '/')]
        for key in stale: state.values.pop(key)
        state.ack.forget(stale)
        state.pending_graphs[root] = update
    for key, source in instance_sources.items():
        root = source.projection_path
        stale = [target for target in state.values if target[0]==root or target[0].startswith(root+'/') or
                 target == (source.hidden_source_path,'visibility')]
        for target in stale: state.values.pop(target)
        state.ack.forget(stale)
        state.pending_instances[key] = source
    state.values.update(unique)
    for target in unique:
        state.ack.changed(target)
    state.pending_materials = pending_materials
    state.pending_lights = pending_lights
    state.pending_visibility = pending_visibility
    state.pending_structure = reparent
    state.pending_controls.update(owner_changes.controls)
    state.dirty.update(object_ids | material_ids)
    if state.lights is not None:
        for key in light_ids:
            state.dirty.update((key,) if key in state.lights.handles else state.lights.users.get(key, ()))
    state.revision += 1
    return True


def apply(owner, generation, controller):
    state = getattr(owner, '_direct_state', None)
    if state is None or state.generation_key != _key(generation): return {}
    from .ovrtx_value_updates import OvrtxAttributeValue
    client, simulation_id = controller._active()
    token_fn = getattr(client, 'direct_projection_token', None)
    token = token_fn(simulation_id) if callable(token_fn) else None
    ticket = state.ack.select(state.values, token)
    values = tuple(OvrtxAttributeValue(r['prim_path'], r['attribute_name'], r['value'], r['value_type'])
                   for r in ticket.rows)
    started = time.perf_counter()
    instance_updates = dict(state.pending_instances)
    if state.instance_sources and state.instance_token != token:
        instance_updates = {**state.instance_sources, **instance_updates}
    instance_count = len(instance_updates)
    if instance_count:
        client.update_instance_graphs(simulation_id, tuple(dict(owner_id=int(key.session_uid),
            projection_path=source.projection_path, hidden_source_path=source.hidden_source_path,
            layer_body=source.layer_text) for key, source in instance_updates.items()))
    graph_count = len(state.pending_graphs)
    if graph_count:
        client.update_material_graphs(simulation_id, tuple(state.pending_graphs.values()))
    if values:
        from .generation_runtime_adapters import _replay_ovrtx_values
        from .runtime_scheduler import RuntimeTickStatus
        result = controller.apply_runtime_updates(lambda port, _: _replay_ovrtx_values(port, (), values))
        if result.status == RuntimeTickStatus.FAILED:
            raise RuntimeError(result.skipped_reason or 'direct_native_update_failed')
    if not state.ack.acknowledge(ticket, token_fn(simulation_id) if callable(token_fn) else None):
        # Native receipt alone cannot certify skipped projection values if the
        # worker/population changed mid-packet. Preserve pending contracts; a
        # subsequent stable capture must replay the current projection.
        raise RuntimeError('direct_native_projection_ack_uncertain')
    if state.pending_materials is not None: state.materials = state.pending_materials
    if state.pending_lights is not None: state.lights = state.pending_lights
    if state.pending_visibility is not None: state.visibility = state.pending_visibility
    state.pending_materials = state.pending_lights = None
    state.pending_visibility = None
    if state.pending_structure is not None:
        from . import direct_structure_updates
        direct_structure_updates.acknowledge(state.structure, state.pending_structure)
        for key, parent_uid in state.pending_structure.parent_session_uids.items():
            state.controls[key] = (parent_uid, *state.controls[key][1:])
    state.pending_structure = None
    from . import direct_object_updates
    direct_object_updates.acknowledge(state.objects, state.controls, state.data_users,
                                     state.pending_controls, state.structure)
    state.pending_controls.clear()
    state.pending_graphs.clear()
    for key, source in state.pending_instances.items():
        old = state.instance_sources.get(key)
        for dependency in old.reference_identities if old is not None else ():
            state.instance_reference_users.get(dependency, set()).discard(key)
        for dependency in source.reference_identities:
            state.instance_reference_users.setdefault(dependency, set()).add(key)
        state.instance_sources[key] = source
        state.mapping_overrides[key] = source.mapping
        state.mesh_contracts.pop(key, None)
        if state.structure is not None:
            state.structure.paths[key] = source.projection_path
            state.structure.eligible = state.structure.eligible | {key}
    state.pending_instances.clear()
    state.instance_token = token_fn(simulation_id) if callable(token_fn) else None
    state.applied_revision = state.revision
    return {**state.diagnostics, 'current_value_count':len(state.values), 'submitted_value_count':len(values), 'submitted_material_graphs':graph_count, 'submitted_instance_graphs':instance_count, 'revision':state.revision,
            'apply_ms':(time.perf_counter()-started)*1000}

# SPDX-License-Identifier: Apache-2.0
"""Bounded light/World typed updates. Adopt next_contract only after native ack.

Initial population supplies schema and original-ID handles once. plan never
opens/traverses USD or scans scene.objects. Resource replacement is a candidate
until GPU zoo proves refresh; same-path dirty images explicitly fall back.
"""
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from . import light_value_conversion as lights
from . import world_dome_conversion as worlds

@dataclass(frozen=True)
class State:
    path: str
    family: str
    form: str
    schema: frozenset
    values: tuple
    unsupported: dict = field(default_factory=dict)
    pose_path: str = ''
    pose_schema: frozenset = frozenset()

@dataclass(frozen=True)
class LightContract:
    generation_key: tuple
    states: dict
    handles: dict
    users: dict
    world_id: object = None
    world_graph: tuple = ()

@dataclass(frozen=True)
class LightPlan:
    values: tuple
    fallback: tuple
    handled: frozenset
    next_contract: LightContract

def _identity(obj, kind):
    from .scene_generation import blender_id
    return blender_id(obj, kind)

def _key(generation):
    return generation.number, generation.digest

def _schema(prim):
    return frozenset(str(a.GetName()) for a in prim.GetAttributes()) if prim else frozenset()

def _graph(world):
    tree=getattr(world, 'node_tree', None)
    if not getattr(world, 'use_nodes', False): return ('flat',)
    return (tuple((getattr(n,'name',id(n)),getattr(n,'type',''),getattr(n,'blend_type',None),getattr(n,'use_clamp',None),getattr(n,'projection',None),getattr(n,'vector_type',None),getattr(n,'is_active_output',None)) for n in tree.nodes),
            tuple((getattr(l.from_node,'name',id(l.from_node)),l.from_socket.name,getattr(l.to_node,'name',id(l.to_node)),l.to_socket.name) for l in tree.links))

def _pose(obj):
    matrix=obj.matrix_world
    # Blender matrices are column-vector; USD Gf matrices use row-vector.
    return tuple(tuple(float(matrix[j][i]) for j in range(4)) for i in range(4))

def _light_values(obj):
    attrs=tuple((a.name,a.value_type,a.value) for a in lights.usd_attribute_values(obj))
    return attrs

def _world_values(world):
    description=worlds.environment_texture_dome(world)
    if description:
        # Existing initial converter resolves packed images and endpoint MixRGB.
        return (('inputs:intensity','Float',description['intensity']),
                ('inputs:color','Color3f',description['color']),
                ('inputs:texture:file','Asset',description['texture_file']),
                ('xformOp:transform','Matrix4d',worlds.environment_rotation_matrix(description['rotation_z'])))
    attrs=worlds.usd_attribute_values(world)
    if not attrs: raise ValueError('unsupported_world_graph')
    return tuple((a.name,a.value_type,a.value) for a in attrs)

def capture_contract(stage, scene, generation):
    states={};handles={};users={}
    for obj in scene.objects:
        if getattr(obj,'type','')!='LIGHT': continue
        identity=_identity(obj,'OBJECT');mapping=generation.blender_prim_paths.get(identity)
        if mapping is None:continue
        path=mapping.schema_path;prim=stage.GetPrimAtPath(path)
        if not prim:continue
        handles[identity]=obj
        dataid=_identity(obj.data,'LIGHT');users.setdefault(dataid,[]).append(identity)
        form=lights.authored_light_form(obj.data.type,getattr(obj.data,'shape',''))
        values=_light_values(obj)
        posepath=mapping.object_path;poseprim=stage.GetPrimAtPath(posepath)
        poseschema=_schema(poseprim)
        values+=(('xformOp:transform','Matrix4d',_pose(obj)),)
        states[identity]=State(path,str(prim.GetTypeName()),form,_schema(prim),values,
                              lights.unsupported_field_values(obj),posepath,poseschema)
    world=getattr(scene,'world',None);wid=None;graph=()
    if world is not None:
        wid=_identity(world,'WORLD');handles[wid]=world;graph=_graph(world)
        tree=getattr(world,'node_tree',None)
        if tree is not None:
            users[_identity(tree,'NODETREE')]=[wid]
            for node in tree.nodes:
                image=getattr(node,'image',None)
                if image is not None:users.setdefault(_identity(image,'IMAGE'),[]).append(wid)
        prim=stage.GetPrimAtPath(worlds.DEFAULT_DOME_OWNER_PATH)
        if prim and str(prim.GetTypeName())=='DomeLight':
            try:values=_world_values(world)
            except ValueError:values=()
            states[wid]=State(worlds.DEFAULT_DOME_OWNER_PATH,'DomeLight','WORLD',_schema(prim),values)
    return LightContract(_key(generation),states,handles,{k:tuple(v) for k,v in users.items()},wid,graph)

def plan(scene, generation, changed_identities, contract, *, depsgraph=None):
    if _key(generation)!=contract.generation_key:
        return LightPlan((),('light_generation_changed',),frozenset(),contract)
    targets={};handled=set();reasons=[];rows=[];states=dict(contract.states)
    for identity in changed_identities:
        owners=(identity,) if identity in contract.handles else contract.users.get(identity,())
        if owners:handled.add(identity)
        for owner in owners:targets[owner]=True
    for identity in targets:
        original=contract.handles[identity];state=contract.states.get(identity)
        if state is None:reasons.append('light_target_absent_from_population');continue
        try:
            if identity==contract.world_id:
                if getattr(scene,'world',None) is not original:raise ValueError('world_assignment_changed')
                if _graph(original)!=contract.world_graph:raise ValueError('world_graph_changed')
                for node in getattr(getattr(original,'node_tree',None),'nodes',()):
                    image=getattr(node,'image',None)
                    if image is not None and getattr(image,'is_dirty',False):raise ValueError('world_same_path_image_content_requires_resource_refresh')
                values=_world_values(original)
            else:
                evaluated=original.evaluated_get(depsgraph) if depsgraph is not None else original
                form=lights.authored_light_form(evaluated.data.type,getattr(evaluated.data,'shape',''))
                if form!=state.form:raise ValueError('light_form_changed')
                if lights.unsupported_field_changes(evaluated,state.unsupported):raise ValueError('unsupported_light_field_changed')
                values=_light_values(evaluated)
                pose=_pose(evaluated)
                values+=(('xformOp:transform','Matrix4d',pose),)
            previous={name:(typ,value) for name,typ,value in state.values}
            for name,typ,value in values:
                if previous.get(name)==(typ,value):continue
                pose=name=='xformOp:transform' and identity!=contract.world_id;schema=state.pose_schema if pose else state.schema
                transform=name=='xformOp:transform'
                if name not in schema and not (pose and 'xformOpOrder' in schema):raise ValueError('missing_light_attribute:'+name)
                rows.append({'prim_path':state.pose_path if pose else state.path,'attribute_name':'omni:xform' if transform else name,'value_type':typ,'value':value})
                if transform:rows.append({'prim_path':state.pose_path if pose else state.path,'attribute_name':'omni:resetXformStack','value_type':'Bool','value':True})
            states[identity]=replace(state,values=values)
        except (ValueError,AttributeError,TypeError) as exc:reasons.append(str(exc))
    if reasons:return LightPlan((),tuple(reasons),frozenset(),contract)
    return LightPlan(tuple(rows),(),frozenset(handled),replace(contract,states=states))

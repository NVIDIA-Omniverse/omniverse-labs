# SPDX-License-Identifier: Apache-2.0
"""Current small structural source, for runtime-owned prototype/hierarchy setup.

This is an explicit experimental transport, not an automatic Blender route.
The supplied inline chunk is authoritative for its nominated subtrees. It may
reference already-populated USD prims. No files or historical overlays are kept.
"""
import time

MAX_SOURCE_BYTES = 8 * 1024 * 1024

class InlineStructurePopulation:
    def __init__(self, owner):
        self.owner = owner
        self.handle = None
        self.text = ''
        self.roots = ()

    def replace(self, text, roots, *, ordinal):
        if self.owner.failed:
            raise RuntimeError('Failed structural population requires scene reload')
        if text == self.text and tuple(roots) == self.roots:
            return False
        api,stage=self.owner.population,self.owner.stage
        candidate = api.add_usd_reference_from_string(stage,text,'/') if roots else None
        try:
            if self.handle is not None:
                api.remove_usd(stage,self.handle)
            api.apply_usd_changes(stage,ordinal=ordinal)
        except Exception:
            self.owner.failed=True
            raise
        self.handle,self.text,self.roots=candidate,text,tuple(roots)
        self.owner.ordinal=max(self.owner.ordinal,ordinal)
        return True

    def clear(self):
        if self.handle is not None:
            self.owner.population.remove_usd(self.owner.stage,self.handle)
        self.handle,self.text,self.roots=None,'',()


def apply(backend, params):
    try:
        from .direct_native_structure import _path, _bounded_list, _under
    except ImportError:
        from direct_native_structure import _path, _bounded_list, _under
    if params.get('experimental') is not True or set(params)!={'experimental','inline_population'}:
        raise ValueError('Inline structure requires an isolated experimental packet')
    packet=params['inline_population']
    if not isinstance(packet,dict) or set(packet)-{'usda','owned_roots','superseded_attributes'} or not {'usda','owned_roots'} <= set(packet):
        raise ValueError('Inline population requires usda and owned_roots')
    text=packet['usda']
    if not isinstance(text,str) or not text.startswith('#usda 1.0') or len(text.encode('utf-8'))>MAX_SOURCE_BYTES:
        raise ValueError('Inline structure requires bounded USDA text')
    roots=tuple(_path(path) for path in _bounded_list(packet['owned_roots'],'owned_roots'))
    if len(set(roots))!=len(roots) or any(_under(a,b) for a in roots for b in roots if a!=b):
        raise ValueError('Owned roots must be distinct nonoverlapping subtrees')
    if not roots and text.strip()!='#usda 1.0':
        raise ValueError('An empty root list only removes the current inline structure')
    superseded=set()
    for item in _bounded_list(packet.get('superseded_attributes',[]),'superseded_attributes'):
        if not isinstance(item,dict) or set(item)!={'prim_path','attribute_name'}:
            raise ValueError('Superseded attributes require prim_path and attribute_name')
        path=_path(item['prim_path']);name=item['attribute_name']
        if not isinstance(name,str) or not name or not any(_under(path,root) for root in roots):
            raise ValueError('Superseded attribute must belong to a retained owned root')
        superseded.add((path,name))
    owner=getattr(backend,'population_owner',None)
    if owner is None:
        raise RuntimeError('Inline structure requires populated scene ownership')
    backend._require_idle_native_render()
    current=getattr(owner,'structure',None)
    if current is None:
        current=owner.structure=InlineStructurePopulation(owner)
    old_roots=current.roots
    started=time.perf_counter()
    # Replaying a few current edited properties is independent of scene size.
    # The incoming source names any values that intentionally supersede them.
    retained={target:row for target,row in backend.desired_values.items()
        if any(_under(target[0],root) for root in roots) and target not in superseded}
    try:
        from .direct_native_structure import _columns
        from .renderer_values import prepare_value,tensor_value
    except ImportError:
        from direct_native_structure import _columns
        from renderer_values import prepare_value,tensor_value
    encoded=[]
    with backend.ovstage.PathDictionary(backend.stage) as paths:
        for target,row in retained.items():
            schema=_columns(backend,paths,target[0],[target[1]]).get(target[1])
            if schema is None:
                raise ValueError('Retained inline property has no native schema: '+str(target))
            data,array,semantic=tensor_value(prepare_value(row),backend.ovstage,paths,
                schema['dtype'],existing_semantic=schema['semantic'],existing_is_array=schema['is_array'],existing_width=schema['width'])
            encoded.append((*target,data,array,semantic))
    backend.ordinal+=1
    try:
        changed=current.replace(text,roots,ordinal=backend.ordinal)
        population_ms=(time.perf_counter()-started)*1000
        with backend.ovstage.PathDictionary(backend.stage) as paths:
            for boundary in (True,False):
                backend._write_phase(paths,[row for row in encoded if (row[1]=='omni:resetXformStack')==boundary])
        backend.stage.advance_write_floor(backend.ordinal).wait()
        backend.renderer.update_from_stage(backend.ordinal)
        backend.renderer.reset(time=backend.simulation_time_ns/1e9)
    except Exception:
        backend.experimental_structure_uncertain=True
        raise
    # Remove obsolete subtree overrides; keep unaffected source owners and the
    # accepted overrides replayed on retained subtrees.
    backend.desired_values={target:value for target,value in backend.desired_values.items()
        if target not in superseded and (target in retained or
            not any(_under(target[0],root) for root in (*old_roots,*roots)))}
    try:
        from .native_pose_fanout import successor_fanout
    except ImportError:
        from native_pose_fanout import successor_fanout
    # The stage survives: untouched synthetic world inputs keep their identity.
    backend.native_pose_fanout=successor_fanout(backend.native_pose_fanout,(*old_roots,*roots))
    backend.sample_count=0
    return dict(experimental=True,production_enabled=False,structural_updated=changed,
        native_stage_id=backend.native_stage_id,ordinal=backend.ordinal,
        population_update='current_inline_structure',retained_overlay_handles=int(current.handle is not None),
        owned_roots=list(roots),replayed_attribute_count=len(encoded),source_projection_stale=True,
        timings=dict(population_ms=population_ms,total_ms=(time.perf_counter()-started)*1000))

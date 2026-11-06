# SPDX-License-Identifier: Apache-2.0
"""Batch changed owner instance sources; one current reference per live owner.

No cumulative source merge, source file, full-scene scan, or edit archive.
The producer copies current evaluated prototype data into each nominated chunk.
"""
from dataclasses import dataclass
import hashlib
import time

@dataclass(frozen=True)
class Entry:
    handle: int
    projection_path: str
    hidden_source_path: str
    digest: str
    text: str = ''

@dataclass(frozen=True)
class Update:
    owner_id: int
    projection_path: str
    hidden_source_path: str
    text: str
    digest: str


def _under(path,root):return path==root or path.startswith(root+'/')

class InstanceGraphPopulation:
    def __init__(self,owner):
        self.owner=owner
        self.entries={}

    def prepare(self,updates):
        if not isinstance(updates,(list,tuple)) or not updates:
            raise ValueError('Instance graph updates require a nonempty owner list')
        result=[];seen=set()
        for item in updates:
            if not isinstance(item,dict) or set(item)!={'owner_id','projection_path','hidden_source_path','layer_body'}:
                raise ValueError('Instance graph requires owner_id, projection_path, hidden_source_path, layer_body')
            identity=item['owner_id'];path=item['projection_path'];hidden=item['hidden_source_path'];text=item['layer_body']
            if type(identity) is not int or identity<=0 or identity in seen:
                raise ValueError('Instance graph owner IDs must be distinct positive integers')
            if path!='/World/_DirectGeometry/Object_'+str(identity):
                raise ValueError('Instance graph projection path must match its stable owner ID')
            if not isinstance(hidden,str) or not hidden.startswith('/') or hidden=='/' or _under(path,hidden) or _under(hidden,path):
                raise ValueError('Instance graph must hide a distinct existing owner')
            if not isinstance(text,str) or not text.startswith('#usda 1.0\n'):
                raise ValueError('Instance graph requires current inline USDA')
            previous=self.entries.get(identity)
            if previous is not None and (previous.projection_path,previous.hidden_source_path)!=(path,hidden):
                raise ValueError('Instance graph paths changed within one source generation')
            seen.add(identity)
            result.append(Update(identity,path,hidden,text,hashlib.sha256(text.encode('utf-8')).hexdigest()))
        return tuple(result)

    def apply(self,updates,*,ordinal):
        if self.owner.failed:raise RuntimeError('Instance graph population requires scene reload')
        changed=tuple(item for item in updates if self.entries.get(item.owner_id) is None or self.entries[item.owner_id].digest!=item.digest)
        if not changed:return ()
        api,stage=self.owner.population,self.owner.stage
        candidates={}
        try:
            # Parse every candidate while all preceding references remain live.
            for item in changed:candidates[item.owner_id]=api.add_usd_reference_from_string(stage,item.text,'/')
        except Exception:
            for handle in candidates.values():api.remove_usd(stage,handle)
            raise
        try:
            for item in changed:
                previous=self.entries.get(item.owner_id)
                if previous is not None:api.remove_usd(stage,previous.handle)
            api.apply_usd_changes(stage,ordinal=ordinal)
        except Exception:
            self.owner.failed=True
            raise
        for item in changed:
            self.entries[item.owner_id]=Entry(candidates[item.owner_id],item.projection_path,item.hidden_source_path,item.digest,item.text)
        self.owner.ordinal=max(self.owner.ordinal,ordinal)
        return changed

    def clear(self):
        for entry in self.entries.values():self.owner.population.remove_usd(self.owner.stage,entry.handle)
        self.entries.clear()


def apply(backend,params):
    if set(params)!={'updates'}:raise ValueError('Instance graph packet requires updates only')
    backend._require_idle_native_render()
    owner=getattr(backend,'population_owner',None)
    if owner is None:raise RuntimeError('Instance graphs require populated scene ownership')
    graphs=getattr(owner,'instance_graphs',None)
    if graphs is None:graphs=owner.instance_graphs=InstanceGraphPopulation(owner)
    updates=graphs.prepare(params['updates'])
    try:
        from .direct_native_structure import inspect
    except ImportError:
        from direct_native_structure import inspect
    # Validate nominated old owner identities only. No recursive stage query.
    paths=list(dict.fromkeys(item.hidden_source_path for item in updates))
    for offset in range(0,len(paths),256):
        cols=inspect(backend,dict(prim_paths=paths[offset:offset+256],attributes=['usd-prim-type']))['columns']
        if any('usd-prim-type' not in cols.get(path,{}) for path in paths[offset:offset+256]):
            raise ValueError('Instance graph original owner is absent')
    started=time.perf_counter();backend.ordinal+=1
    try:
        changed=graphs.apply(updates,ordinal=backend.ordinal)
        population_ms=(time.perf_counter()-started)*1000
        backend.stage.advance_write_floor(backend.ordinal).wait()
        backend.renderer.update_from_stage(backend.ordinal)
        backend.renderer.reset(time=backend.simulation_time_ns/1e9)
    except Exception:
        backend.experimental_structure_uncertain=True
        raise
    roots=tuple(item.projection_path for item in changed)
    hidden={(item.hidden_source_path,'visibility') for item in changed}
    backend.desired_values={target:row for target,row in backend.desired_values.items()
        if target not in hidden and not any(_under(target[0],root) for root in roots)}
    try:
        from .native_pose_fanout import successor_fanout
    except ImportError:
        from native_pose_fanout import successor_fanout
    # The stage survives: untouched synthetic world inputs keep their identity.
    backend.native_pose_fanout=successor_fanout(backend.native_pose_fanout,roots);backend.sample_count=0
    return dict(instance_graph_updated=True,owner_count=len(updates),changed_owner_count=len(changed),
        retained_owner_count=len(graphs.entries),native_stage_id=backend.native_stage_id,ordinal=backend.ordinal,
        population_ms=population_ms,total_ms=(time.perf_counter()-started)*1000,source_projection_stale=True)

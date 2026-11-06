from pathlib import Path
from types import SimpleNamespace as NS
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'runtime'))
from instance_graph_updates import InstanceGraphPopulation

def fixture():
    events=[];sequence=[0]
    def add(stage,text,target):
        if 'BAD' in text:raise ValueError('bad source')
        sequence[0]+=1;events.append(('add',sequence[0]));return sequence[0]
    api=NS(add_usd_reference_from_string=add,remove_usd=lambda stage,h:events.append(('remove',h)),
           apply_usd_changes=lambda stage,ordinal:events.append(('apply',ordinal)))
    return InstanceGraphPopulation(NS(population=api,stage=object(),ordinal=1,failed=False)),events

def packet(i,text=''):
    return dict(owner_id=i,projection_path=f'/World/_DirectGeometry/Object_{i}',hidden_source_path=f'/World/Owner{i}',layer_body='#usda 1.0\n'+text)

def test_two_owners_update_only_changed_handle_and_one_population_apply():
    graphs,events=fixture()
    graphs.apply(graphs.prepare([packet(1),packet(2)]),ordinal=2)
    unchanged=graphs.entries[2];events.clear()
    graphs.apply(graphs.prepare([packet(1,'# current'),packet(2)]),ordinal=3)
    assert events==[('add',3),('remove',1),('apply',3)]
    assert graphs.entries[2] is unchanged and len(graphs.entries)==2
    events.clear();assert not graphs.apply(graphs.prepare([packet(1,'# current')]),ordinal=4)
    assert not events
    graphs.clear();assert events==[('remove',3),('remove',2)] and not graphs.entries

def test_parse_failure_retains_previous_owners_and_removes_candidates():
    graphs,events=fixture();graphs.apply(graphs.prepare([packet(1)]),ordinal=2)
    previous=graphs.entries[1];events.clear()
    with pytest.raises(ValueError):graphs.apply(graphs.prepare([packet(1,'# changed'),packet(2,'BAD')]),ordinal=3)
    assert events==[('add',2),('remove',2)] and graphs.entries[1] is previous
    assert not graphs.owner.failed

def test_apply_failure_invalidates_owner():
    graphs,_=fixture()
    def fail(*args,**kwargs):raise RuntimeError('apply failed')
    graphs.owner.population.apply_usd_changes=fail
    with pytest.raises(RuntimeError):graphs.apply(graphs.prepare([packet(1)]),ordinal=2)
    assert graphs.owner.failed

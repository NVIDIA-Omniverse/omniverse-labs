from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'runtime'))
from subset_dependency_updates import subset_dependency_rows


def setup():
    raw=np.array([11,12],dtype=np.uint64)
    paths=NS(intern_token=lambda s:s,token_to_string=lambda _: 'Mesh',
             intern_path=lambda p:{'/M/R':11,'/M/B':12,'/M/Other':13}[p],
             create_path_list_from_strings=lambda p:nullcontext(p))
    def group(name,value,semantic=0,array=False):
        return NS(attribute=name,is_delete=False,prim_count=1,is_array=array,raw=NS(semantic=semantic),
                  array=lambda _:value,tensor=lambda _:NS(dtype=NS(lanes=1)))
    queries=[];released=[]
    groups=[group('usd-prim-type',np.array([1],dtype=np.uint64)),group('geomSubsets',raw,4,True)]
    def query(p):queries.append(p);return nullcontext(p)
    stage=NS(query_from_path_list=query,release_group=lambda g:released.append(g),
             read_attributes=lambda *args:nullcontext(NS(wait=lambda:None,groups=lambda:iter(groups))))
    ovs=NS(AttributeSemantic=NS(RELATIONSHIP_PATH_ID=4),OrdinalRange=NS(latest=lambda n:n))
    return stage,ovs,paths,queries,released,raw


def test_two_subset_updates_force_one_parent_write_and_copy_borrowed_data():
    stage,ovs,paths,queries,released,raw=setup()
    rows=subset_dependency_rows(stage,ovs,paths,[('/M/R','indices'),('/M/B','material:binding')],
                                {'/M/R':'GeomSubset','/M/B':'GeomSubset'},ordinal=8)
    assert queries==[['/M']] and len(released)==2 and len(rows)==1
    assert rows[0][:2]==('/M','geomSubsets') and rows[0][3:]==(True,4)
    raw[:]=99
    np.testing.assert_array_equal(rows[0][2],[11,12])


def test_unrelated_mesh_or_camera_values_do_not_read_parents():
    stage,ovs,paths,queries,*_=setup()
    assert subset_dependency_rows(stage,ovs,paths,[('/M','points')],{'/M':'Mesh'},ordinal=8)==()
    assert not queries


def test_explicit_parent_relationship_in_same_batch_is_not_duplicated():
    stage,ovs,paths,queries,*_=setup()
    rows=subset_dependency_rows(stage,ovs,paths,[('/M/R','indices'),('/M','geomSubsets')],{'/M/R':'GeomSubset','/M':'Mesh'},ordinal=8)
    assert not rows and not queries


def test_unlinked_subset_rejects_before_any_write():
    stage,ovs,paths,*_=setup()
    with pytest.raises(ValueError,match='absent from populated'):
        subset_dependency_rows(stage,ovs,paths,[('/M/Other','indices')],{'/M/Other':'GeomSubset'},ordinal=8)

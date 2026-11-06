# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'runtime'))
from inline_structure_population import InlineStructurePopulation, apply


def fixture():
    events=[];count=[0]
    def add(stage,text,target):
        count[0]+=1;events.append(('add',count[0]));return count[0]
    api=NS(add_usd_reference_from_string=add,
        remove_usd=lambda stage,handle:events.append(('remove',handle)),
        apply_usd_changes=lambda stage,ordinal:events.append(('apply',ordinal)))
    owner=NS(population=api,stage=object(),ordinal=1,failed=False)
    return InlineStructurePopulation(owner),events


def test_one_current_source_create_replace_remove():
    source,events=fixture()
    assert source.replace('#usda 1.0\ndef Xform "A" {}',('/A',),ordinal=2)
    assert source.replace('#usda 1.0\ndef Xform "B" {}',('/B',),ordinal=3)
    assert [event[0] for event in events]==['add','apply','add','remove','apply']
    assert source.handle==2 and source.roots==('/B',)
    assert source.replace('#usda 1.0\n',(),ordinal=4)
    assert source.handle is None and events[-2:]==[('remove',2),('apply',4)]


def test_generation_clear_removes_owned_handle():
    source,events=fixture()
    source.replace('#usda 1.0\ndef Xform "A" {}',('/A',),ordinal=2)
    source.clear()
    assert source.handle is None and source.text=='' and not source.roots
    assert events[-1]==('remove',1)


def test_source_mutation_failure_invalidates_projection():
    source,events=fixture()
    source.replace('#usda 1.0\ndef Xform "A" {}',('/A',),ordinal=2)
    def fail(*args,**kwargs):raise RuntimeError('apply failed')
    source.owner.population.apply_usd_changes=fail
    with pytest.raises(RuntimeError):source.replace('#usda 1.0\ndef Xform "B" {}',('/B',),ordinal=3)
    assert source.owner.failed
    with pytest.raises(RuntimeError,match='requires scene reload'):
        source.replace('#usda 1.0\n',(),ordinal=4)


@pytest.mark.parametrize('packet',[
    dict(experimental=False,inline_population={}),
    dict(experimental=True,inline_population=dict(usda='bad',owned_roots=['/A'])),
    dict(experimental=True,inline_population=dict(usda='#usda 1.0',owned_roots=['/A','/A/B'])),
    dict(experimental=True,inline_population=dict(usda='#usda 1.0\ndef Xform "A" {}',owned_roots=[])),
    dict(experimental=True,inline_population=dict(usda='#usda 1.0',owned_roots=['/A'],superseded_attributes=[dict(prim_path='/B',attribute_name='points')])),
])
def test_packet_validation_precedes_any_owner_access(packet):
    with pytest.raises(ValueError):apply(NS(),packet)

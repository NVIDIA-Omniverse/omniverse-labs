# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'runtime'))
from inline_visibility_population import InlineVisibilityPopulation
from renderer_values import prepare_value


def row(path, token):
    return prepare_value(dict(prim_path=path,attribute_name='visibility',value_type='Token',value=token))


def fixture():
    events=[];handles=[]
    def add(stage,text,target):
        handle=len(handles)+1;handles.append(handle);events.append(('add',handle,text,target));return handle
    api=NS(add_usd_reference_from_string=add,
           remove_usd=lambda stage,h:events.append(('remove',h)),
           apply_usd_changes=lambda stage,ordinal:events.append(('apply',ordinal)))
    owner=NS(population=api,stage=object(),failed=False,ordinal=1)
    return InlineVisibilityPopulation(owner),events


def test_replace_one_current_handle_and_no_floor_or_renderer():
    owner,events=fixture()
    initial=owner.prepare([row('/World/A','invisible')])
    assert not owner.values
    assert owner.apply(initial,ordinal=2)
    assert owner.owner.ordinal==2
    next=owner.prepare([row('/World/B','invisible'),row('/World/A','inherited')])
    assert next.values=={'/World/A':'inherited','/World/B':'invisible'}
    owner.apply(next,ordinal=3)
    assert [event[0] for event in events]==['add','apply','add','remove','apply']
    assert owner.handle==2
    assert not owner.apply(next,ordinal=4)
    owner.clear()
    assert owner.handle is None and not owner.values and owner.text==''
    assert events[-1]==('remove',2)


def test_invalid_visibility_rejected_before_source_mutation():
    owner,events=fixture()
    for token in ['visible','yes','']:
        with pytest.raises(ValueError):owner.prepare([row('/World/A',token)])
    wrong=prepare_value(dict(prim_path='/World/A',attribute_name='visibility',value_type='String',value='invisible'))
    with pytest.raises(ValueError):owner.prepare([wrong])
    assert not events


def test_failure_requires_explicit_recovery():
    owner,events=fixture()
    def fail(*args,**kwargs):raise RuntimeError('population failed')
    owner.owner.population.apply_usd_changes=fail
    with pytest.raises(RuntimeError):owner.apply(owner.prepare([row('/World/A','invisible')]),ordinal=2)
    assert owner.owner.failed
    with pytest.raises(RuntimeError,match='requires scene reload'):
        owner.apply(owner.prepare([row('/World/A','inherited')]),ordinal=3)


def test_worker_visibility_and_values_share_one_publication(monkeypatch):
    from test_public_sdk_edit_batching import backend_fixture
    import renderer_values
    import numpy as np
    backend,events,_=backend_fixture()
    real_tensor=renderer_values.tensor_value
    def tensor(item,*args,**kwargs):
        if item[1]=='visibility':return np.array([1],dtype=np.uint64),False,0
        return real_tensor(item,*args,**kwargs)
    monkeypatch.setattr(renderer_values,'tensor_value',tensor)
    owner=NS(stage=backend.stage,failed=False,ordinal=4,visibility=None)
    owner.population=NS(
        add_usd_reference_from_string=lambda *args:events.append(('inline',)) or 7,
        remove_usd=lambda *args:events.append(('remove_inline',)),
        apply_usd_changes=lambda *args,**kw:events.append(('population',)))
    backend.population_owner=owner
    backend.update([dict(prim_path='/A',attribute_name='visibility',value_type='Token',value='invisible'),
                    dict(prim_path='/A',attribute_name='inputs:intensity',value_type='Float',value=8.)])
    names=[event[0] for event in events]
    assert names.count('floor')==names.count('renderer')==names.count('population')==1
    assert names.index('population')<names.index('submit')<names.index('floor')<names.index('renderer')
    assert all('visibility' not in event[2] for event in events if event[0]=='submit')
    assert owner.visibility.values=={'/A':'invisible'}

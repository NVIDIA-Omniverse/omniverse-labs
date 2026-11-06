"""Token changes only when direct projection may need restoration."""
from types import SimpleNamespace as NS
import pytest
from test_direct_sdk_client import client, start
from ovrtx_blender_example import scene_population_delta
from ovrtx_blender_example.direct_sdk_client import _State
from ovrtx_blender_example.ovrtx_value_updates import OvrtxAttributeValue
from ovrtx_blender_example.ovrtx_runtime_client import RenderClientError


def test_values_keep_token_worker_epoch_and_new_state_change_it(client):
    identity=start(client);before=client.direct_projection_token(identity)
    client.update_attribute_values(identity,(OvrtxAttributeValue('/M','roughness',.5,'Float'),))
    assert client.direct_projection_token(identity)==before
    client.runtime.epoch+=1
    assert client.direct_projection_token(identity)!=before
    after=client.direct_projection_token(identity)
    old=client.states[identity]
    client.states[identity]=_State(old.spec,simulation_id=identity)
    assert client.direct_projection_token(identity)!=after


def test_only_successful_population_acceptance_advances_revision(client,monkeypatch):
    identity=start(client);before=client.direct_projection_token(identity)
    monkeypatch.setattr(scene_population_delta,'compile_population_delta',lambda *a,**k:NS(path=None,candidate_snapshot=None,
        preparation_ms=0,phase_timings={},changed_prims=0,changed_properties=0,created_prims=0,deleted_prims=0,fallback_reason=''))
    original=client.runtime.request
    client.runtime.request=lambda op,p:({'scene_updated':False,'rollback_succeeded':True} if op=='update_scene' else original(op,p))
    with pytest.raises(RenderClientError):client.update_scene(identity,client.states[identity].spec)
    assert client.direct_projection_token(identity)==before
    client.runtime.request=lambda op,p:({'scene_updated':True} if op=='update_scene' else original(op,p))
    client.update_scene(identity,client.states[identity].spec)
    assert client.states[identity].population_revision==1
    assert client.direct_projection_token(identity)!=before


def test_uncertain_and_pending_recovery_are_not_certified(client):
    identity=start(client);state=client.states[identity]
    state.failure=RuntimeError('uncertain');assert client.direct_projection_token(identity) is None
    state.failure=None;state.recovery_pending=True;assert client.direct_projection_token(identity) is None

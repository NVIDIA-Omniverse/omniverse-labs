from ovrtx_blender_example.direct_projection_ack import ProjectionAck

class NoScan(dict):
    def keys(self):raise AssertionError('warm no-op cannot scan current projection')


def test_first_load_replay_then_constant_time_noop():
    ack=ProjectionAck();payload=object();current={'geometry':payload}
    ack.changed('geometry');ticket=ack.select(current,('worker',1,'population',1))
    assert ticket.full_replay and ticket.rows[0] is payload
    assert ack.acknowledge(ticket,ticket.token)
    assert ack.select(NoScan(current),ticket.token).rows==()


def test_small_change_never_reconstructs_large_unchanged_payload():
    ack=ProjectionAck();current={'mesh':object(),'roughness':1}
    ticket=ack.select(current,'population1');ack.acknowledge(ticket,'population1')
    current['roughness']=2;ack.changed('roughness')
    ticket=ack.select(NoScan(current),'population1')
    assert ticket.rows==(2,) and not ticket.full_replay


def test_new_worker_or_population_replays_every_current_value():
    ack=ProjectionAck();current={'mesh':object(),'roughness':.5}
    ticket=ack.select(current,('worker1',1));ack.acknowledge(ticket,ticket.token)
    assert len(ack.select(current,('worker2',1)).rows)==2
    assert len(ack.select(current,('worker1',2)).rows)==2


def test_failure_leaves_pending_and_superseded_ack_does_not_clear_new_value():
    ack=ProjectionAck();current={'roughness':.3};ack.changed('roughness')
    ticket=ack.select(current,'population')
    assert ack.pending  # Failed native packet: caller never acknowledges.
    current['roughness']=.7;ack.changed('roughness')
    assert ack.acknowledge(ticket,'population')
    assert ack.select(current,'population').rows==(.7,)


def test_changed_token_during_ack_forces_full_replay():
    ack=ProjectionAck();current={'mesh':object()};ack.changed('mesh')
    ticket=ack.select(current,'old')
    assert not ack.acknowledge(ticket,'new')
    assert ack.pending and ack.select(current,'new').full_replay


def test_unknown_token_never_claims_reuse_and_retirement_is_bounded():
    ack=ProjectionAck();current={'a':1};ack.changed('a')
    ticket=ack.select(current,None)
    assert ticket.full_replay and not ack.acknowledge(ticket,None)
    ack.forget(['a']);assert ack.pending=={} and ack.versions=={}

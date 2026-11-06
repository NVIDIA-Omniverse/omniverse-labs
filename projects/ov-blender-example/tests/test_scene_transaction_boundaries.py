# SPDX-License-Identifier: Apache-2.0
"""Every native replacement boundary must restore or quarantine the scene."""
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'runtime'))
from public_sdk_backend import RendererBackend


def transaction(tmp_path, failure, *, rollback_failure=False):
    backend = RendererBackend.__new__(RendererBackend)
    source = tmp_path / 'accepted.usda'
    source.write_text('#usda 1.0\n')
    calls = []
    failed = False

    def boundary(name):
        nonlocal failed
        calls.append(name)
        if name == failure and (not failed or rollback_failure):
            failed = True
            raise RuntimeError(name)

    owner = SimpleNamespace(path=source, ordinal=2, overlays=[], failed=False)

    def replace(path, *, ordinal):
        owner.ordinal = ordinal
        boundary('population')
        owner.path = Path(path)
        return {}

    owner.replace = replace
    owner.restore = replace
    owner.snapshot_inline_sources = lambda: {'fixture': 'accepted'}
    owner.restore_inline_sources = lambda sources: boundary('sources')
    backend.population_owner = owner
    backend.pose_population = SimpleNamespace(accepted_generation_replaced=lambda **kw: boundary('pose'))
    backend.scene_config = {'scene_path': str(source), 'stage_metadata': {}, 'label': 'accepted',
                            'sensor_paths': ['/Render/Camera'], 'render_var': 'LdrColor'}
    backend.desired_values = {('/World/A', 'inputs:intensity'): {'prim_path': '/World/A', 'attribute_name': 'inputs:intensity', 'value': 4}}
    backend.ordinal = 2
    backend.pending = {}
    backend.native_stage_id = 'durable'
    backend.native_pose_fanout = object()
    backend.simulation_time_ns = 0
    backend.stage = SimpleNamespace(advance_write_floor=lambda ordinal: SimpleNamespace(wait=lambda: boundary('floor')))
    backend.renderer = SimpleNamespace(update_from_stage=lambda ordinal: boundary('synchronize'), reset=lambda **kw: boundary('reset'))
    backend._apply_scene_config = lambda params: boundary('configure')

    def replay(values):
        boundary('replay')
        backend.desired_values.update({(v['prim_path'], v['attribute_name']): v for v in values})

    backend.update = replay
    return backend, calls


@pytest.mark.parametrize('failure', ['population', 'floor', 'synchronize', 'reset', 'configure', 'pose'])
def test_rejection_restores_same_path_and_replays_acknowledged_values(tmp_path, failure):
    backend, calls = transaction(tmp_path, failure)
    accepted = backend.scene_config
    values = dict(backend.desired_values)
    result = backend.update_scene({**accepted, 'label': 'candidate'})
    assert result['scene_updated'] is False and result['rollback_succeeded'] is True
    assert backend.scene_config is accepted and backend.desired_values == values
    assert calls.count('population') == 2  # Same filename still needs restoration.
    assert calls[-1] == 'replay' and backend.native_pose_fanout is None
    assert backend.native_stage_id == 'durable'
    assert backend.update_scene({**accepted, 'label': 'next'})['scene_updated']


@pytest.mark.parametrize('failure', ['population', 'floor', 'synchronize', 'reset', 'configure', 'pose'])
def test_failed_restoration_quarantines_every_mutating_dispatch(tmp_path, failure):
    backend, calls = transaction(tmp_path, failure, rollback_failure=True)
    accepted = backend.scene_config
    result = backend.update_scene({**accepted, 'label': 'candidate'})
    assert result['rollback_succeeded'] is False and backend.scene_config is accepted
    before = list(calls)
    for op in ('render', 'begin_render', 'update', 'resize', 'update_scene', 'retarget_camera', 'structural_update'):
        with pytest.raises(RuntimeError, match='reload required'):
            backend.dispatch(op, {})
    assert calls == before


def test_population_failed_flag_cannot_be_reported_as_restored(tmp_path):
    backend, calls = transaction(tmp_path, 'population')
    backend.population_owner.failed = True
    result = backend.update_scene(dict(backend.scene_config))
    assert result['rollback_succeeded'] is False
    assert calls == ['population']


def test_replay_failure_quarantines_predecessor_instead_of_false_success(tmp_path):
    backend, calls = transaction(tmp_path, 'configure')
    backend.update = lambda values: (_ for _ in ()).throw(RuntimeError('replay rejected'))
    result = backend.update_scene(dict(backend.scene_config))
    assert result['rollback_succeeded'] is False and 'replay rejected' in result['error']


def test_inline_replay_must_precede_renderer_sync_and_value_replay(tmp_path):
    backend, calls = transaction(tmp_path, 'configure')
    assert backend.update_scene(dict(backend.scene_config))['rollback_succeeded']
    restore = calls.index('sources')
    assert calls[restore + 1:] == ['floor', 'synchronize', 'reset', 'pose', 'configure', 'replay']


def test_inline_replay_failure_cannot_claim_usable_predecessor(tmp_path):
    backend, calls = transaction(tmp_path, 'configure')
    backend.population_owner.restore_inline_sources = lambda sources: (_ for _ in ()).throw(RuntimeError('inline replay'))
    result = backend.update_scene(dict(backend.scene_config))
    assert not result['rollback_succeeded'] and 'inline replay' in result['error']
    assert 'replay' not in calls

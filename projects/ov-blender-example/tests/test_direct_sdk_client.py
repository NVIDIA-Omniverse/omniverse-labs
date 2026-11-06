# SPDX-License-Identifier: Apache-2.0
"""Contract checks at the existing typed-client/public-SDK process boundary."""
from contextlib import contextmanager
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'addon'))
from ovrtx_blender_example import direct_sdk_client as client_module
from ovrtx_blender_example import ovstream_shm
from ovrtx_blender_example.direct_runtime_session import RuntimeWorkerError
from ovrtx_blender_example.ovrtx_runtime_client import RenderClientError
from ovrtx_blender_example.ovrtx_value_updates import OvrtxAttributeValue


class FakeRuntime:
    def __init__(self, output_root):
        self.output_root = output_root
        self.epoch = 1
        self.pid = 4242
        self.depth = 0
        self.calls = []
        self.reply = {}
        self.errors = {}
        self.returncode = None

    @contextmanager
    def transaction(self):
        self.depth += 1
        try:
            yield self
        finally:
            self.depth -= 1

    def request(self, op, params):
        self.calls.append((op, params))
        if op in self.errors:
            raise self.errors[op]
        return {'_runtime_epoch': self.epoch, '_runtime_pid': self.pid,
                **(self.reply if op == 'render' else {}),
                **({'updated_count': len(params['values'])} if op == 'update' else {}),
                **({'released': True} if op == 'release_frame' else {}),
                **({'cancelled': True} if op == 'cancel_raw_transfer' else {})}

    def recover(self):
        self.epoch += 1
        self.calls.append(('recover', {}))

    def close(self):
        self.calls.append(('close', {}))


@pytest.fixture
def client(tmp_path, monkeypatch):
    runtime = FakeRuntime(tmp_path)
    monkeypatch.setattr(client_module, '_session', lambda kind: runtime)
    result = client_module.PublicSdkRenderClient()
    monkeypatch.setattr(result, '_snapshot_source', lambda path, **kw: str(path))
    monkeypatch.setattr(result, '_snapshot_inline', lambda text: ('fixture-inline', text))
    result._test_monkeypatch = monkeypatch
    return result


def start(client, render_var='HdrColor'):
    spec = SimpleNamespace(ovrtx_scene_composition=SimpleNamespace(composed_scene_path='/scene.usda'),
                           stage_metadata={}, sensor_paths=('/Render/Product',), width=1, height=2, render_var=render_var)
    return client.start_session(spec)


def test_scene_replacement_refuses_uncertain_predecessor_without_rpc(client):
    identity = start(client)
    state = client.states[identity]
    state.failure = RenderClientError('rollback was not acknowledged')
    calls = list(client.runtime.calls)
    with pytest.raises(RenderClientError, match='rollback was not acknowledged'):
        client.update_scene(identity, state.spec)
    assert client.runtime.calls == calls


def test_scene_replacement_waits_for_recovered_state_replay(client):
    identity = start(client)
    state = client.states[identity]
    state.recovery_pending = True
    calls = list(client.runtime.calls)
    with pytest.raises(RenderClientError, match='replay'):
        client.update_scene(identity, state.spec)
    assert client.runtime.calls == calls


def test_population_root_change_invalidates_all_acknowledged_descendants():
    values = {('/World/Mesh', 'omni:xform'): {'value': 1}}
    assert client_module._unaffected_population_values(
        values, {}, {}, effective_changed_paths=['/']) == {}


def test_real_client_replacement_retains_immutable_predecessor_through_ack(client, tmp_path, monkeypatch):
    from pxr import Usd
    monkeypatch.setattr(client, '_snapshot_source', client_module.PublicSdkRenderClient._snapshot_source.__get__(client))
    source = tmp_path / 'scene.usda'
    source.write_text('#usda 1.0\ndef Cube "Cube" {\n double size = 2\n}\n')
    spec = SimpleNamespace(ovrtx_scene_composition=SimpleNamespace(composed_scene_path=source),
        stage_metadata={}, sensor_paths=('/Render/Product',), width=1, height=2, render_var='HdrColor')
    identity = client.start_session(spec)
    accepted = Path(client.states[identity].native_scene_path)
    source.write_text('#usda 1.0\ndef Cube "Cube" {\n double size = 5\n}\n')
    def rejected(op, params):
        assert op == 'update_scene'
        old_stage, candidate_stage = Usd.Stage.Open(str(accepted)), Usd.Stage.Open(params['scene_path'])
        assert old_stage.GetPrimAtPath('/Cube').GetAttribute('size').Get() == 2
        assert candidate_stage.GetPrimAtPath('/Cube').GetAttribute('size').Get() == 5
        return {'scene_updated': False, 'rollback_succeeded': True}
    monkeypatch.setattr(client.runtime, 'request', rejected)
    with pytest.raises(RenderClientError):
        client.update_scene(identity, spec)
    assert set(client._scene_store().snapshots) == {str(accepted)}
    monkeypatch.setattr(client.runtime, 'request', lambda op, params: {'scene_updated': True, 'population_update': 'replacement'})
    client.update_scene(identity, spec)
    assert not accepted.exists()
    assert len(client._scene_store().snapshots) == 1
    population = client.states[identity].native_population_path
    monkeypatch.setattr(client.runtime, 'request', lambda op, params: {'camera_retargeted': True})
    for revision in range(5):
        camera_source = tmp_path / ('camera-' + str(revision) + '.usda')
        camera_source.write_text('#usda 1.0\ndef Cube "Cube" {\n double size = ' + str(6 + revision) + '\n}\n')
        next_spec = SimpleNamespace(**{**vars(spec), 'ovrtx_scene_composition':
            SimpleNamespace(composed_scene_path=camera_source)})
        client.retarget_camera(identity, next_spec, [])
        assert Path(population).is_file()
        assert len(client._scene_store().snapshots) == 2  # one population and one current recovery source
    client.shutdown()
    assert not tuple(client._scene_store().root.iterdir())


def test_typed_asset_mutation_keeps_accepted_bytes_until_update_ack(client, tmp_path, monkeypatch):
    identity = start(client)
    texture = tmp_path / 'texture.bin'
    texture.write_bytes(b'old')
    value = OvrtxAttributeValue('/Texture', 'inputs:file', str(texture), 'Asset')
    client.update_attribute_values(identity, [value])
    before = client.runtime.calls[-1][1]['values'][0]['value']
    texture.write_bytes(b'new')
    previous = client.runtime.request
    def update(op, params):
        assert Path(before).read_bytes() == b'old'
        assert Path(params['values'][0]['value']).read_bytes() == b'new'
        return previous(op, params)
    monkeypatch.setattr(client.runtime, 'request', update)
    client.update_attribute_values(identity, [value])
    assert not Path(before).exists()
    assert len(client._scene_store().snapshots) == 1


def test_collecting_and_repeated_asset_updates_keep_only_current_versions(client, tmp_path):
    identity = start(client)
    texture = tmp_path / 'texture.bin'
    texture.write_bytes(b'accepted')
    row = OvrtxAttributeValue('/Texture', 'inputs:file', str(texture), 'Asset')
    client.update_attribute_values(identity, [row])
    accepted = client.runtime.calls[-1][1]['values'][0]['value']
    state = client.states[identity]
    state.collecting = True
    for revision in range(30):
        texture.write_bytes(str(revision).encode())
        client.update_attribute_values(identity, [row])
        assert Path(accepted).read_bytes() == b'accepted'
        assert len(client._scene_store().snapshots) <= 2
    state.collecting = False
    client.replay_desired_values(identity)
    for _ in range(30):
        client.update_attribute_values(identity, [row])
        assert len(client._scene_store().snapshots) == 1


def test_rejected_camera_candidates_are_released_and_original_input_reusable(client, tmp_path, monkeypatch):
    identity = start(client)
    monkeypatch.setattr(client, '_snapshot_source', client_module.PublicSdkRenderClient._snapshot_source.__get__(client))
    source = tmp_path / 'candidate.usda'
    spec = client.states[identity].spec
    spec.ovrtx_scene_composition.composed_scene_path = source
    monkeypatch.setattr(client.runtime, 'request', lambda *args: (_ for _ in ()).throw(RuntimeError('rejected camera')))
    for revision in range(30):
        source.write_text('#usda 1.0\ndef Cube "C" {\n double size = ' + str(revision) + '\n}\n')
        with pytest.raises(RenderClientError, match='rejected camera'):
            client.retarget_camera(identity, spec, [])
        assert not client._scene_store().snapshots
        assert not tuple(client._scene_store().root.iterdir())


def test_failed_initial_load_tears_down_before_releasing_candidate_assets(client, tmp_path, monkeypatch):
    monkeypatch.setattr(client, '_snapshot_source', client_module.PublicSdkRenderClient._snapshot_source.__get__(client))
    source = tmp_path / 'candidate.usda'
    source.write_text('#usda 1.0\ndef Cube "C" {}\n')
    spec = SimpleNamespace(ovrtx_scene_composition=SimpleNamespace(composed_scene_path=source),
        stage_metadata={}, sensor_paths=('/P',), width=1, height=2, render_var='HdrColor')
    client.runtime.errors['load'] = RuntimeError('load rejected')
    for _ in range(10):
        with pytest.raises(RenderClientError, match='load rejected'):
            client.start_session(spec)
        assert client.runtime.calls[-1][0] == 'close'
        assert not tuple(client._scene_store().root.iterdir())


def test_inline_graph_capture_does_not_rewrite_callers_mutable_texture_path(client, tmp_path, monkeypatch):
    from pxr import Sdf
    identity = start(client)
    monkeypatch.setattr(client, '_snapshot_inline', client_module.PublicSdkRenderClient._snapshot_inline.__get__(client))
    texture = tmp_path / 'texture.bin'
    original = '#usda 1.0\nover "M" {\n asset texture = @' + texture.as_posix() + '@\n}\n'
    updates = [dict(material_path='/M', layer_body=original)]
    def acknowledge(op, params):
        captured = Sdf.Layer.CreateAnonymous()
        captured.ImportFromString(params['updates'][0]['layer_body'])
        assert Path(captured.GetAttributeAtPath('/M.texture').default.path).read_bytes() == texture.read_bytes()
        return {'material_graph_updated': True, 'material_count': 1}
    monkeypatch.setattr(client.runtime, 'request', acknowledge)
    for revision in range(10):
        texture.write_bytes(str(revision).encode())
        client.update_material_graphs(identity, updates)
        assert updates[0]['layer_body'] == original
        assert len(client._scene_store().snapshots) == 1


def test_population_after_direct_camera_switch_retires_actual_bootstrap_override(client, monkeypatch):
    from ovrtx_blender_example import scene_population_delta
    identity = start(client)
    current = client.states[identity].spec
    current.camera_prim_path = '/World/B'
    current.ovrtx_scene_composition.presentation_layers = ({
        'source': 'viewport_camera_projection', 'target_path': '/World/A', 'path': '/old-camera.usda'},)
    desired = SimpleNamespace(**vars(current))
    captured = []
    def compile(previous, candidate, **kwargs):
        captured.append(kwargs['retired_camera_projection'])
        return SimpleNamespace(path=None, candidate_snapshot=None, preparation_ms=0,
            phase_timings={}, changed_prims=0, changed_properties=0, created_prims=0,
            deleted_prims=0, fallback_reason='')
    monkeypatch.setattr(scene_population_delta, 'compile_population_delta', compile)
    original = client.runtime.request
    client.runtime.request = lambda op, params: ({'scene_updated': True} if op == 'update_scene' else original(op, params))
    client.update_scene(identity, desired)
    assert captured == [{'camera_path': '/World/A', 'layer_path': '/old-camera.usda'}]


def test_scene_compatibility_reuses_acknowledged_metadata_without_reopening_current(client, monkeypatch):
    identity = start(client, 'LdrColor')
    current = client.states[identity].spec
    desired = SimpleNamespace(**vars(current))
    desired.stage_metadata = {}
    # Mutating the caller's dictionary cannot change the acknowledged state.
    current.stage_metadata['upAxis'] = 'Y'
    original = client._load_params
    reads = []

    def load_params(spec):
        reads.append(spec)
        return original(spec)

    monkeypatch.setattr(client, '_load_params', load_params)
    assert client.can_update_scene(current, desired)
    assert reads == [desired]
    # An unrelated spec must not inherit this session's metadata cache.
    assert not client.can_update_scene(SimpleNamespace(**vars(current)), desired)
    assert len(reads) == 3


@pytest.mark.parametrize('accepted', [False, True])
def test_scene_metadata_snapshot_changes_only_after_acknowledged_replacement(client, accepted):
    identity = start(client, 'LdrColor')
    previous = client.states[identity].spec
    desired = SimpleNamespace(**vars(previous))
    desired.stage_metadata = {'upAxis': 'Z'}
    original = client.runtime.request
    client.runtime.request = lambda op, params: (
        {'scene_updated': accepted, 'rollback_succeeded': True}
        if op == 'update_scene' else original(op, params))
    if accepted:
        client.update_scene(identity, desired)
        desired.stage_metadata['upAxis'] = 'Y'
        assert client.states[identity].stage_metadata == {'upAxis': 'Z'}
        assert client.states[identity].spec is desired
    else:
        with pytest.raises(RenderClientError):
            client.update_scene(identity, desired)
        assert client.states[identity].stage_metadata == {}
        assert client.states[identity].spec is previous


def hdr(client, epoch=1):
    from ovrtx_blender_example import ovstream_raw_frame
    linear = struct.pack('<8e', .125, .25, .5, 1, 2, 4, 8, 1)
    rgba = bytes([1, 2, 3, 255, 4, 5, 6, 255])
    shm(client, client._test_monkeypatch)
    from ovrtx_blender_example.frame_protocol import payload_header
    source = dict(**payload_header('/Render/Product', 'HdrColor'), schema=1,
        encoding='raw_bytes_over_bgra8', byte_order='little', storage_width=1,
        storage_height=4, byte_count=16, planes=[dict(name='linear_rgba16f', format='rgba16f',
        width=1, height=2, offset_bytes=0, row_bytes=8, byte_count=16)])
    client.runtime.reply.update(render_var='HdrColor', raw_frame=source, chunk={}, transport='ovstream_shm_raw')
    class Assembler:
        def __init__(self, *args): pass
        def append(self, *args):
            assert client.runtime.depth > 0
            return True
        def finish(self):
            return SimpleNamespace(planes=(SimpleNamespace(plane=SimpleNamespace(name='rgba8')),), payload=lambda name, **kw: bytes(linear if name == 'linear_rgba16f' else rgba))
    client._test_monkeypatch.setattr(ovstream_raw_frame, 'RawFrameAssembler', Assembler)
    return linear, rgba


def render(client, identity, samples=1, render_var='HdrColor'):
    return client.render_result(identity, selected_sensor_paths=('/Render/Product',),
                                render_var=render_var, additional_samples=samples)


def shm(client, monkeypatch, *, copy_error=None, dimensions=(1, 2)):
    events = []
    class Reader:
        def __init__(self, **kwargs):
            events.append(('attach', kwargs))
        def copy_frame(self, descriptor):
            assert client.runtime.depth > 0, 'shared-frame copy must hold owner transaction'
            if copy_error:
                raise copy_error
            return SimpleNamespace(width=dimensions[0], height=dimensions[1], rgba8=bytes([1, 2, 3, 255] * 2))
        def close(self):
            events.append(('close', {}))
    monkeypatch.setattr(ovstream_shm, 'OvstreamShmReader', Reader)
    client.runtime.reply = {'width': 1, 'height': 2, 'origin': 'top_left',
                            'sensor_path': '/Render/Product', 'render_var': 'LdrColor',
                            'shm': {'stream_name': 'owned-frame', 'epoch': 'stream-epoch-1',
                                    'sequence': 1, 'producer_pid': client.runtime.pid,
                                    'width': 1, 'height': 2, 'capture_timestamp_ns': 10,
                                    'copy_token': 'held-frame-1'}}
    return events


def releases(client):
    return [params for op, params in client.runtime.calls if op == 'release_frame']


def test_hdr_retains_exact_half_float_and_owns_copied_bytes(client):
    identity = start(client)
    linear, rgba = hdr(client)
    result = render(client, identity)
    assert result.linear_rgba16f == linear
    assert result.linear_rgba16f_bytes() == linear[8:] + linear[:8]
    assert result.rgba8 == rgba
    assert result.rgba8_bytes() == rgba[4:] + rgba[:4]
    assert result.frame_format == 'rgba16f'
    assert result.product_id == '/Render/Product'
    assert result.render_var == 'HdrColor'
    client.runtime.reply.clear()
    assert result.linear_rgba16f == linear
    assert result.linear_rgba16f_bytes() == linear[8:] + linear[:8]
    assert result.rgba8 == rgba
    assert result.rgba8_bytes() == rgba[4:] + rgba[:4]


def test_hdr_rejects_file_payload_protocol(client):
    identity = start(client)
    hdr(client)
    del client.runtime.reply['raw_frame']
    with pytest.raises(RenderClientError, match='exact raw SHM'):
        render(client, identity)


def test_successful_shm_copy_releases_once_inside_transaction(client, monkeypatch):
    identity = start(client, 'LdrColor')
    shm(client, monkeypatch)
    original_request = client.runtime.request
    def check_request(op, params):
        if op == 'release_frame':
            assert client.runtime.depth > 0
        return original_request(op, params)
    client.runtime.request = check_request
    result = render(client, identity, render_var='LdrColor')
    assert result.rgba8 == bytes([1, 2, 3, 255] * 2)
    assert result.product_id == '/Render/Product'
    assert result.render_var == 'LdrColor'
    assert releases(client) == [{'copy_token': 'held-frame-1'}]


@pytest.mark.parametrize('failure', ['copy', 'dimensions', 'sensor', 'render_var', 'missing_sensor', 'missing_render_var', 'metadata_dimensions'])
def test_failed_frame_is_released_and_not_published(client, monkeypatch, failure):
    identity = start(client, 'LdrColor')
    shm(client, monkeypatch,
        copy_error=ovstream_shm.ShmFrameError('copy failed') if failure == 'copy' else None,
        dimensions=(2, 2) if failure == 'dimensions' else (1, 2))
    if failure == 'sensor':
        client.runtime.reply['sensor_path'] = '/Wrong'
    elif failure == 'render_var':
        client.runtime.reply['render_var'] = 'Wrong'
    elif failure == 'missing_sensor':
        del client.runtime.reply['sensor_path']
    elif failure == 'missing_render_var':
        del client.runtime.reply['render_var']
    elif failure == 'metadata_dimensions':
        client.runtime.reply['width'] = 0
    with pytest.raises((RenderClientError, ovstream_shm.ShmFrameError)):
        render(client, identity, render_var='LdrColor')
    assert releases(client) == [{'copy_token': 'held-frame-1'}]
    assert client.states[identity].samples == 0


def test_stale_shm_producer_is_rejected_even_with_self_consistent_descriptor(client, monkeypatch):
    identity = start(client, 'LdrColor')
    events = shm(client, monkeypatch)
    client.runtime.reply['shm']['producer_pid'] -= 1
    with pytest.raises(RenderClientError, match='producer|worker|identity'):
        render(client, identity, render_var='LdrColor')
    assert not events, 'reject a stale producer before attaching to its memory'


def test_recovery_resets_sample_counters_at_new_worker_epoch(client):
    identity = start(client)
    hdr(client)
    first = render(client, identity, samples=3)
    assert (first.completed_samples, first.session_completed_samples) == (3, 3)
    second = render(client, identity, samples=2)
    assert (second.completed_samples, second.session_completed_samples) == (5, 5)
    client.runtime.epoch = 2
    hdr(client, epoch=2)
    recovered = render(client, identity, samples=1)
    assert (recovered.completed_samples, recovered.session_completed_samples) == (1, 1)


def test_rejected_update_does_not_advance_samples_or_report_success(client):
    identity = start(client)
    hdr(client)
    render(client, identity, samples=3)
    client.runtime.errors['update'] = RuntimeError('unsupported attribute')
    value = OvrtxAttributeValue('/World/A', 'omni:visibility', False, 'Bool')
    with pytest.raises(RenderClientError, match='unsupported attribute'):
        client.update_attribute_values(identity, [value])
    assert client.states[identity].samples == 3
    assert client.states[identity].session_samples == 3


def test_partial_update_ack_does_not_claim_complete_success(client):
    identity = start(client)
    original = client.runtime.request
    def partial(op, params):
        return {'updated_count': 0} if op == 'update' else original(op, params)
    client.runtime.request = partial
    value = OvrtxAttributeValue('/World/A', 'omni:visibility', False, 'Bool')
    with pytest.raises(RenderClientError, match='acknowledge|incomplete|partial|count'):
        client.update_attribute_values(identity, [value])


def test_failed_write_blocks_render_until_explicit_recovery_and_replay(client):
    identity = start(client)
    hdr(client)
    first = OvrtxAttributeValue('/World/A', 'omni:visibility', True, 'Bool')
    client.update_attribute_values(identity, [first])
    client.runtime.errors['update'] = RuntimeError('connection lost after write')
    second = OvrtxAttributeValue('/World/A', 'omni:visibility', False, 'Bool')
    with pytest.raises(RenderClientError):
        client.update_attribute_values(identity, [second])
    calls_before = len(client.runtime.calls)
    with pytest.raises(RenderClientError):
        render(client, identity)
    assert len(client.runtime.calls) == calls_before
    del client.runtime.errors['update']
    assert client.recover_value_updates(identity, defer_replay=True)
    with pytest.raises(RenderClientError, match='replay'):
        render(client, identity)
    assert client.begin_recovery_update(identity)
    latest = OvrtxAttributeValue('/World/A', 'omni:visibility', True, 'Bool')
    client.update_attribute_values(identity, [latest])
    client.end_recovery_update(identity, commit=True)
    updates = [params for op, params in client.runtime.calls if op == 'update']
    assert updates[-1]['values'] == [{'prim_path': '/World/A', 'attribute_name': 'omni:visibility',
                                     'value_type': 'Bool', 'value': True}]
    hdr(client, epoch=2)
    assert render(client, identity).completed_samples == 1


@pytest.mark.parametrize('deferred', [False, True])
def test_explicit_recovery_never_replays_failed_desired_value(client, deferred):
    identity = start(client)
    accepted = OvrtxAttributeValue('/World/A', 'omni:visibility', True, 'Bool')
    client.update_attribute_values(identity, [accepted])
    client.runtime.errors['update'] = RuntimeError('uncertain write')
    with pytest.raises(RenderClientError):
        client.update_attribute_values(identity, [
            OvrtxAttributeValue('/World/A', 'omni:visibility', False, 'Bool')])
    del client.runtime.errors['update']
    assert client.recover_value_updates(identity, defer_replay=deferred)
    if deferred:
        assert client.begin_recovery_update(identity)
        client.end_recovery_update(identity, commit=True)
    update = [params for op, params in client.runtime.calls if op == 'update'][-1]
    assert update['values'][0]['value'] is True


@pytest.mark.parametrize('cancel', [False, True])
def test_deferred_fresh_value_does_not_survive_cancel_or_failed_replay(client, cancel):
    identity = start(client)
    client.update_attribute_values(identity, [
        OvrtxAttributeValue('/World/A', 'omni:visibility', True, 'Bool')])
    state = client.states[identity]
    state.failure = RenderClientError('worker died')
    assert client.recover_value_updates(identity, defer_replay=True)
    assert client.begin_recovery_update(identity)
    client.update_attribute_values(identity, [
        OvrtxAttributeValue('/World/A', 'omni:visibility', False, 'Bool')])
    if cancel:
        client.end_recovery_update(identity, commit=False)
    else:
        client.runtime.errors['update'] = RuntimeError('fresh replay uncertain')
        with pytest.raises(RenderClientError):
            client.end_recovery_update(identity, commit=True)
        del client.runtime.errors['update']
        client.recover_value_updates(identity)
    assert state.desired[('/World/A', 'omni:visibility')]['value'] is True


def test_copy_error_is_not_hidden_by_release_error(client, monkeypatch):
    identity = start(client, 'LdrColor')
    shm(client, monkeypatch, copy_error=ovstream_shm.ShmFrameError('original copy error'))
    client.runtime.errors['release_frame'] = RuntimeError('release also failed')
    with pytest.raises(ovstream_shm.ShmFrameError, match='original copy error'):
        render(client, identity, render_var='LdrColor')
    assert releases(client) == [{'copy_token': 'held-frame-1'}]


@pytest.mark.parametrize('deactivated', [True, False])
def test_public_worker_restart_uses_scene_owner_without_starting_sil(monkeypatch, deactivated):
    from ovrtx_blender_example import engine, scene_generation_sessions
    monkeypatch.setattr(engine, '_WORKER_RESTART', None)
    monkeypatch.setenv(client_module.PYTHON_ENV, '/external/python')
    calls = []
    monkeypatch.setattr(engine, '_visit_active_viewport_engines', lambda callback: ([], []))
    monkeypatch.setattr(scene_generation_sessions, 'pause_preparation', lambda: (calls.append('pause') or True))
    monkeypatch.setattr(scene_generation_sessions, 'deactivate_all_ovrtx', lambda: (calls.append('deactivate') or deactivated))
    monkeypatch.setattr(scene_generation_sessions, 'resume_preparation', lambda: calls.append('resume'))
    def forbidden(*args, **kwargs):
        raise AssertionError('Public SDK restart must not use installed SIL services')
    assert not hasattr(engine, "runtime_services")
    engine.restart_ovrtx_workers()
    assert engine._WORKER_RESTART["completion"].wait(2)
    result = engine.worker_restart_status()
    assert calls == ['pause', 'deactivate', 'resume']
    assert result['status'] == ('restarted' if deactivated else 'teardown_unconfirmed')
    if deactivated:
        assert result['backend'] == 'public_sdk'
        assert result['restart_pending'] is False


def test_scene_update_commits_only_after_acknowledgement(client):
    identity = start(client)
    old = client.states[identity].spec
    new = SimpleNamespace(**vars(old))
    new.ovrtx_scene_composition = SimpleNamespace(composed_scene_path='/next.usda')
    original = client.runtime.request
    client.runtime.request = lambda op, params: ({'scene_updated': False, 'rollback_succeeded': True} if op == 'update_scene' else original(op, params))
    with pytest.raises(RenderClientError):
        client.update_scene(identity, new)
    assert client.states[identity].spec is old
    assert client.states[identity].failure is None
    client.runtime.request = lambda op, params: ({'scene_updated': True} if op == 'update_scene' else original(op, params))
    client.update_scene(identity, new)
    assert client.states[identity].spec is new


@pytest.mark.parametrize('initial,desired', [('LdrColor', 'HdrColor'), ('HdrColor', 'LdrColor')])
def test_color_format_switch_requires_session_rekey(client, initial, desired):
    identity = start(client, initial)
    current = client.states[identity].spec
    same = SimpleNamespace(**vars(current))
    assert client.can_update_scene(current, same)
    changed = SimpleNamespace(**{**vars(current), 'render_var': desired})
    assert not client.can_update_scene(current, changed)


def test_unconfirmed_release_blocks_next_frame(client, monkeypatch):
    identity = start(client, 'LdrColor')
    events = shm(client, monkeypatch)
    client.runtime.errors['release_frame'] = RuntimeError('lost acknowledgement')
    with pytest.raises(RenderClientError):
        render(client, identity, render_var='LdrColor')
    assert client.states[identity].failure is not None
    assert client._reader is None and client._reader_key is None
    assert events[-1][0] == 'close'
    with pytest.raises(RenderClientError, match='lost acknowledgement'):
        render(client, identity, render_var='LdrColor')


def test_failed_mapping_discards_reader_before_next_frame(client, monkeypatch):
    identity = start(client, 'LdrColor')
    events = shm(client, monkeypatch, copy_error=OSError('mapping failed'))
    with pytest.raises(OSError):
        render(client, identity, render_var='LdrColor')
    assert client._reader is None
    assert events[-1][0] == 'close'
    shm(client, monkeypatch)
    assert render(client, identity, render_var='LdrColor').rgba8


def test_async_poll_is_nonblocking_until_frame_is_ready(client, monkeypatch):
    identity = start(client, 'LdrColor')
    shm(client, monkeypatch)
    original = client.runtime.request
    polls = 0
    def request(op, params):
        nonlocal polls
        if op == 'begin_render':
            return {'ticket': 'ticket-1', '_runtime_epoch': 1, '_runtime_pid': 4242}
        if op == 'poll_render':
            polls += 1
            if polls == 1:
                return {'ready': False}
            return {'ready': True, **original('render', {})}
        return original(op, params)
    client.runtime.request = request
    pending = client.begin_render_sample(identity, selected_sensor_paths=('/Render/Product',), render_var='LdrColor')
    assert client.poll_render_sample(pending) is None
    assert not releases(client)
    frame = client.poll_render_sample(pending)
    assert frame.rgba8
    assert frame.session_completed_samples == 1
    assert frame.native_timings['simulation_id'] == identity
    assert releases(client) == [{'copy_token': 'held-frame-1'}]
    with pytest.raises(RenderClientError, match='consumed'):
        client.poll_render_sample(pending)


def test_cancel_keeps_native_fence_until_ack_without_copying_obsolete_pixels(client):
    identity = start(client, 'LdrColor')
    complete = False
    calls = []
    def request(op, params):
        calls.append((op, params))
        if op == 'begin_render':
            return {'ticket': 'old-frame', '_runtime_epoch': 1, '_runtime_pid': 4242}
        assert op == 'cancel_render'
        return {'cancelled': True, 'native_complete': complete}
    client.runtime.request = request
    pending = client.begin_render_sample(identity, selected_sensor_paths=('/Render/Product',), render_var='LdrColor')
    pending.cancel()
    assert pending.cancelled and client.has_pending_render_cancellations
    assert client.pending_native_render_count == 1
    for _ in range(3):
        assert client.drain_render_cancellations() is False
    assert [p['timeout_ns'] for op, p in calls if op == 'cancel_render'] == [0, 8_000_000, 8_000_000, 8_000_000]
    complete = True
    assert client.drain_render_cancellations() is True
    assert client.pending_native_render_count == 0
    assert {op for op, _ in calls} == {'begin_render', 'cancel_render'}


@pytest.mark.parametrize('retired', ['epoch', 'process_exit'])
def test_cancelled_native_ticket_is_never_replayed_after_worker_retirement(client, retired):
    identity = start(client, 'LdrColor')
    client.runtime.request = lambda op, params: {'ticket': 'old-frame', '_runtime_epoch': 1, 'native_complete': False}
    pending = client.begin_render_sample(identity, selected_sensor_paths=('/Render/Product',), render_var='LdrColor')
    pending.cancel()
    if retired == 'epoch': client.runtime.epoch += 1
    else: client.runtime.returncode = 3
    def forbidden(*args): raise AssertionError('old cancellation replayed')
    client.runtime.request = forbidden
    assert client.drain_render_cancellations()
    assert client.pending_native_render_count == 0


def test_failed_cancel_rpc_retains_fence_for_recovery(client):
    identity = start(client, 'LdrColor')
    client.runtime.request = lambda op, params: {'ticket': 'old-frame', '_runtime_epoch': 1}
    pending = client.begin_render_sample(identity, selected_sensor_paths=('/Render/Product',), render_var='LdrColor')
    def fail(*args): raise RenderClientError('cancel transport failed')
    client.runtime.request = fail
    with pytest.raises(RenderClientError, match='cancel transport failed'): pending.cancel()
    assert client.has_pending_render_cancellations and client.pending_native_render_count == 1


def test_async_completion_prearms_before_owned_copy_without_relabelling_predecessor(client, monkeypatch):
    identity = start(client, 'LdrColor')
    shm(client, monkeypatch)
    original = client.runtime.request
    order, tickets = [], []
    def request(op, params):
        order.append(op)
        if op == 'begin_render':
            token = 'ticket-' + str(len(tickets) + 1)
            tickets.append(token)
            return {'ticket': token, '_runtime_epoch': 1, '_runtime_pid': 4242}
        if op == 'poll_render':
            return {'ready': True, **original('render', {}), 'simulation_time_ns': 16_666_667}
        return original(op, params)
    client.runtime.request = request
    copy = client._copy_frame
    def copy_frame(*args):
        order.append('owned_copy')
        return copy(*args)
    monkeypatch.setattr(client, '_copy_frame', copy_frame)
    first = client.begin_render_sample(identity, selected_sensor_paths=('/Render/Product',), render_var='LdrColor')
    successors = []
    def before_copy():
        assert client.runtime.depth > 0
        successors.append(client.begin_render_sample(identity,
            selected_sensor_paths=('/Render/Product',), render_var='LdrColor'))
    frame = client.poll_render_sample(first, before_copy=before_copy)
    assert order == ['begin_render', 'poll_render', 'begin_render', 'owned_copy', 'release_frame']
    assert frame.session_completed_samples == 1
    assert frame.simulation_time_ns == 16_666_667
    assert successors[0].session_samples == 2
    assert client.states[identity].time_ns == 33_333_334
    assert frame.native_timings['render_to_owned_frame_ms'] >= frame.native_timings['acquire_copy_ms']
    assert frame.native_timings['frame_ack_ms'] >= 0


def test_completion_hook_failure_still_releases_predecessor_hold(client, monkeypatch):
    identity = start(client, 'LdrColor')
    shm(client, monkeypatch)
    original = client.runtime.request
    def request(op, params):
        if op == 'begin_render':
            return {'ticket': 'ticket-1', '_runtime_epoch': 1, '_runtime_pid': 4242}
        if op == 'poll_render':
            return {'ready': True, **original('render', {})}
        return original(op, params)
    client.runtime.request = request
    pending = client.begin_render_sample(identity, selected_sensor_paths=('/Render/Product',), render_var='LdrColor')
    def fail():
        raise RenderClientError('prearm failed')
    with pytest.raises(RenderClientError, match='prearm failed'):
        client.poll_render_sample(pending, before_copy=fail)
    assert releases(client) == [{'copy_token': 'held-frame-1'}]


def test_raw_chunks_release_only_the_last_copied_token(client, monkeypatch):
    from ovrtx_blender_example import ovstream_raw_frame
    identity = start(client)
    hdr(client)
    observed = []
    class Assembler:
        def __init__(self, *args): self.layout = SimpleNamespace(byte_count=16)
        def append(self, reader, held, layout, chunk):
            assert client.runtime.depth > 0
            observed.append(held.copy_token)
            return len(observed) == 2
        def finish(self):
            return SimpleNamespace(planes=(), payload=lambda name, **kw: bytes(16 if name == 'linear_rgba16f' else 8))
    monkeypatch.setattr(ovstream_raw_frame, 'RawFrameAssembler', Assembler)
    original = client.runtime.request
    def request(op, params):
        if op == 'next_raw_chunk':
            assert params == {'copy_token': 'held-frame-1'}
            return {'shm': {**client.runtime.reply['shm'], 'copy_token': 'held-frame-2'},
                    'raw_frame': {}, 'chunk': {}}
        return original(op, params)
    client.runtime.request = request
    frame = render(client, identity)
    assert frame.linear_rgba16f == bytes(16)
    assert frame.native_timings['raw_chunk_count'] == 2
    assert frame.native_timings['chunk_handoff_ms'] >= 0
    assert observed == ['held-frame-1', 'held-frame-2']
    assert releases(client) == [{'copy_token': 'held-frame-2'}]


def test_raw_copy_failure_cancels_transfer_and_reconnects(client, monkeypatch):
    from ovrtx_blender_example import ovstream_raw_frame
    identity = start(client)
    hdr(client)
    class Assembler:
        def __init__(self, *args): pass
        def append(self, *args): raise OSError('failed raw mapping')
    monkeypatch.setattr(ovstream_raw_frame, 'RawFrameAssembler', Assembler)
    with pytest.raises(OSError, match='failed raw mapping'):
        render(client, identity)
    assert client._reader is None
    assert not releases(client)
    assert ('cancel_raw_transfer', {'copy_token': 'held-frame-1'}) in client.runtime.calls
    assert client.states[identity].failure is None


@pytest.mark.parametrize('mutation', ['product', 'aov', 'format', 'extra_plane', 'plane_dimensions',
                                     'response_dimensions', 'runtime', 'session', 'version'])
def test_complete_raw_response_rejected_before_mapping_or_allocation(client, monkeypatch, mutation):
    identity = start(client)
    hdr(client)
    result = client.runtime.reply
    source = result['raw_frame']
    if mutation == 'product': source['product_id'] = '/Other/Product'
    if mutation == 'aov': source['aov'] = 'LdrColor'
    if mutation == 'format': source['planes'][0]['name'] = 'rgba8'
    if mutation == 'extra_plane': source['planes'][0]['name'] = 'semantic_ids'
    if mutation == 'plane_dimensions':
        source['planes'][0].update(width=2, height=1, row_bytes=16)
    if mutation == 'response_dimensions': result.update(width=2, height=1)
    if mutation == 'runtime': result['shm']['producer_pid'] += 1
    if mutation == 'session':
        result['_runtime_session'] = 'authenticated-session'
        result['shm']['epoch'] = 'foreign-session'
    if mutation == 'version': source['version']['major'] = 2
    monkeypatch.setattr(ovstream_shm, 'OvstreamShmReader',
                        lambda **_: pytest.fail('mapped an invalid frame'))
    with pytest.raises((ValueError, RenderClientError, ovstream_shm.ShmFrameError)):
        render(client, identity)
    assert not releases(client)
    assert ('cancel_raw_transfer', {'copy_token': 'held-frame-1'}) in client.runtime.calls


def test_resize_dimensions_change_only_after_successful_acknowledgment(client):
    identity = start(client)
    client.resize_output(identity, 5, 3)
    assert client.states[identity].output_size == (5, 3)
    original = client._call
    def reject(op, params):
        if op == 'resize': raise RuntimeWorkerError('resize refused')
        return original(op, params)
    client._call = reject
    with pytest.raises(RuntimeWorkerError): client.resize_output(identity, 7, 4)
    assert client.states[identity].output_size == (5, 3)


def test_direct_owned_raw_copy_metadata_failure_never_releases_or_returns_frame(client,monkeypatch):
    from ovrtx_blender_example.frame_protocol import payload_header
    identity=start(client)
    shm(client,monkeypatch)
    client.runtime.reply.update(render_var='HdrColor',transport='ovstream_shm_raw',
        raw_frame=dict(**payload_header('/Render/Product', 'HdrColor'),schema=1,encoding='raw_bytes_over_bgra8',byte_order='little',storage_width=1,storage_height=4,byte_count=16,
                       planes=[dict(name='linear_rgba16f',format='rgba16f',width=1,height=2,offset_bytes=0,row_bytes=8,byte_count=16)]),
        chunk=dict(transfer_id='a'*32,index=0,count=1,offset_bytes=0,byte_count=16,total_bytes=16))
    client.runtime.reply['shm'].update(width=1,height=4)
    copied=[]
    class PoisonedReader:
        def __init__(self,**kwargs):pass
        def copy_bytes_into(self,held,destination,**kwargs):
            assert client.runtime.depth>0
            destination[:]=b'x'*16;copied.append(True)
            raise ovstream_shm.ShmFrameError('SHM producer advanced during held copy')
        def close(self):copied.append('closed')
    monkeypatch.setattr(ovstream_shm,'OvstreamShmReader',PoisonedReader)
    with pytest.raises(ovstream_shm.ShmFrameError,match='advanced'):render(client,identity)
    assert copied==[True,'closed']
    assert not releases(client)
    assert ('cancel_raw_transfer',{'copy_token':'held-frame-1'}) in client.runtime.calls
    assert client.states[identity].samples==0


def test_external_python_preserves_venv_symlink(tmp_path, monkeypatch):
    from ovrtx_blender_example import direct_runtime_session
    # A filesystem seam models a venv symlink even on Windows accounts that
    # cannot create links (WinError 1314). Resolving this path deliberately
    # selects the base interpreter, so an accidental .resolve() fails here.
    entrypoint = tmp_path / 'venv with spaces' / 'bin' / 'python'
    target = Path(sys.executable)
    original_is_file, original_resolve = Path.is_file, Path.resolve
    monkeypatch.setattr(Path, 'is_file', lambda path: original_is_file(target if path == entrypoint else path))
    monkeypatch.setattr(Path, 'resolve', lambda path, **kwargs: original_resolve(target if path == entrypoint else path, **kwargs))
    assert entrypoint.is_file() and entrypoint.resolve() == target.resolve()
    monkeypatch.setenv(client_module.PYTHON_ENV, str(entrypoint))
    monkeypatch.setenv(client_module.OUTPUT_ENV, str(tmp_path))
    monkeypatch.setattr(direct_runtime_session, 'RuntimeSession', lambda **kwargs: kwargs)
    assert client_module._session('renderer')['python'] == str(entrypoint)


def test_frame_epoch_must_match_authenticated_runtime_session(client, monkeypatch):
    monkeypatch.setattr(ovstream_shm, 'HeldFrame', lambda **values: SimpleNamespace(**values))
    with pytest.raises(RenderClientError, match='authenticated worker session'):
        client._copy_frame({'_runtime_pid': 4242, '_runtime_session': 'new-session',
                            'shm': {'producer_pid': 4242, 'epoch': 'old-session'}}, 1, 2, 'LdrColor')


def semantic_reply(client):
    import base64
    label = b'class: class_a;'
    mapping = struct.pack('<6I', 4294967295, 0, 0, 0, len(label), 24) + label + struct.pack('<I', 1)
    client.runtime.reply.update(epoch='stream-epoch-1', ordinal=2, simulation_time_ns=17, frame_start_time=0.25, frame_end_time=0.25)
    client.runtime.reply['semantic'] = {key: client.runtime.reply[key] for key in ('width', 'height', 'sensor_path', 'epoch', 'ordinal', 'simulation_time_ns', 'frame_start_time', 'frame_end_time', 'origin')}
    client.runtime.reply['semantic']['camera_prim_path'] = ''
    client.runtime.reply['semantic'].update(id_dtype='uint32', id_map_dtype='uint8', id_map_base64=base64.b64encode(mapping).decode())
    return mapping

def consume_future_semantic_fixture(client, identity):
    # Retain decoder coverage for the candidate-only format without claiming
    # that the MVP public rendering entrypoint admits semantic capture.
    with client.runtime.transaction():
        reply = {'_runtime_epoch': client.runtime.epoch, '_runtime_pid': client.runtime.pid,
                 **client.runtime.reply}
        return client._consume_render_result(client.states[identity], reply,
            ('/Render/Product',), 'HdrColor', 1, 0)


def test_future_semantic_decoder_copies_uint32_and_acknowledges_same_owned_frame(client, monkeypatch):
    from ovrtx_blender_example import ovstream_raw_frame
    identity = start(client)
    linear, _ = hdr(client)
    client.request = SimpleNamespace(semantic_output_path='/output/frame')
    semantic_reply(client)
    pixels = struct.pack('<2I', 0, 4294967295)

    class Assembler:

        def __init__(self, *args):
            pass

        def append(self, *args):
            return True

        def finish(self):
            return SimpleNamespace(epoch='stream-epoch-1', planes=(), payload=lambda name, **kw: pixels if name == 'semantic_ids' else linear)
    monkeypatch.setattr(ovstream_raw_frame, 'RawFrameAssembler', Assembler)
    result = consume_future_semantic_fixture(client, identity)
    assert result.linear_rgba16f == linear
    assert result.semantic_capture.ids == pixels
    assert result.semantic_capture.sensor_path == '/Render/Product'
    assert releases(client) == [{'copy_token': 'held-frame-1'}]
    assert not any(op == 'render' for op, _ in client.runtime.calls)
    client.runtime.reply.clear()
    assert result.semantic_capture.ids == pixels

def test_future_semantic_decoder_missing_plane_cancels_hold_without_publishing(client):
    identity = start(client)
    hdr(client)
    client.request = SimpleNamespace(semantic_output_path='/output/frame')
    with pytest.raises(ValueError, match='Missing native semantic'):
        consume_future_semantic_fixture(client, identity)
    assert ('cancel_raw_transfer', {'copy_token': 'held-frame-1'}) in client.runtime.calls
    assert client.states[identity].samples == 0


def test_repeated_acknowledged_values_skip_rpc_and_preserve_accumulation(client):
    identity = start(client)
    values = [OvrtxAttributeValue('/World/A', 'inputs:intensity', 10.0, 'Float'),
              OvrtxAttributeValue('/World/B', 'visibility', 'inherited', 'Token')]
    client.update_attribute_values(identity, values)
    state = client.states[identity]; state.samples = 12
    before = len(client.runtime.calls)
    result = client.update_attribute_values(identity, values)
    assert result.updated_count == 2
    assert len(client.runtime.calls) == before and state.samples == 12
    assert client.last_value_update_timings['sent_value_count'] == 0
    changed = OvrtxAttributeValue('/World/A', 'inputs:intensity', 20.0, 'Float')
    client.update_attribute_values(identity, [changed, values[1]])
    assert client.runtime.calls[-1][1]['values'] == [
        {'prim_path': '/World/A', 'attribute_name': 'inputs:intensity', 'value_type': 'Float', 'value': 20.0}]
    assert state.samples == 0


def test_failed_attempt_is_not_a_deduplication_baseline(client):
    identity = start(client)
    first = OvrtxAttributeValue('/World/A', 'inputs:intensity', 10.0, 'Float')
    client.update_attribute_values(identity, [first])
    client.runtime.errors['update'] = RuntimeError('lost acknowledgment')
    second = OvrtxAttributeValue('/World/A', 'inputs:intensity', 20.0, 'Float')
    with pytest.raises(RenderClientError): client.update_attribute_values(identity, [second])
    row = {'prim_path': '/World/A', 'attribute_name': 'inputs:intensity', 'value_type': 'Float', 'value': 20.0}
    assert client.states[identity].accepted_values.changed([row], epoch=client.runtime.epoch) == [row]
    with pytest.raises(RenderClientError): client.update_attribute_values(identity, [first])


def test_population_replacement_invalidates_acknowledged_value_cache(client):
    identity = start(client)
    value = OvrtxAttributeValue('/World/A', 'inputs:intensity', 10.0, 'Float')
    client.update_attribute_values(identity, [value])
    original = client.runtime.request
    client.runtime.request = lambda op, params: ({'scene_updated': True} if op == 'update_scene' else original(op, params))
    client.update_scene(identity, client.states[identity].spec)
    before = len(client.runtime.calls)
    client.update_attribute_values(identity, [value])
    assert len(client.runtime.calls) == before + 1


def test_camera_motion_keeps_world_input_semantics_after_first_ack(client):
    identity = start(client)
    def packet(x):
        return [OvrtxAttributeValue('/World/Camera', 'omni:xform',
            ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (x, 0, 0, 1)), 'Matrix4d'),
            OvrtxAttributeValue('/World/Camera', 'omni:resetXformStack', True, 'Bool')]
    client.update_attribute_values(identity, packet(0))
    client.update_attribute_values(identity, packet(1))
    rows = client.runtime.calls[-1][1]['values']
    assert [row['attribute_name'] for row in rows] == ['omni:xform', 'omni:resetXformStack']
    assert rows[1]['value'] is True
    before = len(client.runtime.calls)
    client.update_attribute_values(identity, packet(1))
    assert len(client.runtime.calls) == before


def test_same_asset_path_is_not_assumed_to_have_same_content(client):
    identity = start(client)
    value = OvrtxAttributeValue('/World/Texture', 'inputs:file', '/texture.exr', 'Asset')
    client.update_attribute_values(identity, [value])
    before = len(client.runtime.calls)
    client.update_attribute_values(identity, [value])
    assert len(client.runtime.calls) == before + 1


def test_worker_epoch_invalidates_deduplication(client):
    identity = start(client)
    value = OvrtxAttributeValue('/World/A', 'inputs:intensity', 10.0, 'Float')
    client.update_attribute_values(identity, [value])
    client.runtime.epoch += 1
    before = len(client.runtime.calls)
    client.update_attribute_values(identity, [value])
    assert len(client.runtime.calls) == before + 1


def test_parent_fanout_receipt_updates_client_recovery_and_dedup_state(client):
    from ovrtx_blender_example.ovrtx_value_updates import OvrtxTransformValue
    def matrix(x):
        return ((1.,0.,0.,0.), (0.,1.,0.,0.), (0.,0.,1.,0.), (float(x),0.,0.,1.))
    identity = start(client)
    child_path = '/World/Parent/Child'
    old_child = OvrtxTransformValue(child_path, matrix(5))
    client.update_transforms(identity, [old_child])
    original = client.runtime.request
    inherited_row = {'prim_path': child_path, 'attribute_name': 'omni:xform',
                     'value_type': 'Matrix4d', 'value': [list(row) for row in matrix(25)]}
    def response(op, params):
        result = original(op, params)
        if op == 'update' and params['values'][0]['prim_path'] == '/World/Parent':
            result['accepted_derived_values'] = [inherited_row,
                {'prim_path': child_path, 'attribute_name': 'omni:resetXformStack', 'value_type': 'Bool', 'value': True}]
        return result
    client.runtime.request = response
    client.update_transforms(identity, [OvrtxTransformValue('/World/Parent', matrix(20))])
    assert client.states[identity].desired[(child_path, 'omni:xform')] == inherited_row
    before = len(client.runtime.calls)
    client.update_transforms(identity, [OvrtxTransformValue(child_path, matrix(25))])
    assert len(client.runtime.calls) == before
    client.update_transforms(identity, [old_child])
    assert len(client.runtime.calls) == before + 1
    assert client.runtime.calls[-1][1]['values'][0]['value'][3][0] == 5


def test_changed_parent_still_sends_explicit_unchanged_child_override(client):
    from ovrtx_blender_example.ovrtx_value_updates import OvrtxTransformValue
    def matrix(x):
        return ((1.,0.,0.,0.), (0.,1.,0.,0.), (0.,0.,1.,0.), (float(x),0.,0.,1.))
    identity = start(client)
    child = OvrtxTransformValue('/World/Parent/Child', matrix(5))
    parent = OvrtxTransformValue('/World/Parent', matrix(0))
    client.update_transforms(identity, [parent, child])
    client.update_transforms(identity, [OvrtxTransformValue('/World/Parent', matrix(20)), child])
    rows = client.runtime.calls[-1][1]['values']
    assert any(row['prim_path'] == child.prim_path and row['attribute_name'] == 'omni:xform' for row in rows)


def test_population_delta_retains_only_untouched_value_certificates():
    a=("/Mesh", "attribute", "default")
    b=("/Lights", "attribute", "default")
    values={("/Mesh", "omni:xform"): {"value": 1}, ("/Lights/Sun", "intensity"): {"value": 2}}
    assert client_module._unaffected_population_values(values,{a:1,b:2},{a:3,b:2}) == {("/Lights/Sun", "intensity"): {"value": 2}}
    assert client_module._unaffected_population_values(values,{a:1,b:2},{a:1,b:3}) == {("/Mesh", "omni:xform"): {"value": 1}}


def test_effective_overlay_retention_ignores_raw_ancestor_history():
    ancestor = ('/World', '/World', 'specifier')
    values = {('/World/Mesh', 'omni:xform'): {'value': 1},
              ('/World/Cam', 'omni:xform'): {'value': 2},
              ('/World/Mat/Shader', 'inputs:base_color'): {'value': 3}}
    before, after = {ancestor: ('over', 'def')}, {ancestor: ('def',)}
    keep = client_module._unaffected_population_values
    assert keep(values, before, after, effective_changed_paths=()) == values
    assert keep(values, before, after, effective_changed_paths=('/World/Cam',)) == {
        key: value for key, value in values.items() if key[0] != '/World/Cam'}
    assert keep(values, before, after, effective_changed_paths=('/World/Mat',)) == {
        key: value for key, value in values.items() if not key[0].startswith('/World/Mat/')}
    assert keep(values, before, after, effective_changed_paths=('/World',)) == {}
    assert keep(values, before, after) == {}  # Missing compiler proof fails closed.


def test_material_graph_ack_invalidates_only_changed_root_without_population_token_bump(client):
    identity = start(client)
    state = client.states[identity]
    changed = dict(prim_path='/Materials/A/Shader', attribute_name='inputs:roughness', value_type='Float', value=.5)
    other = dict(changed, prim_path='/Materials/B/Shader')
    state.desired = {(row['prim_path'], row['attribute_name']): row for row in [changed, other]}
    state.accepted_values.acknowledge([changed, other], epoch=client.runtime.epoch)
    token = client.direct_projection_token(identity)
    original = client.runtime.request
    client.runtime.request = lambda op, params: ({'material_graph_updated': True, 'material_count': len(params['updates'])}
        if op == 'material_graph_update' else original(op, params))
    updates = [dict(material_path='/Materials/A', layer_body='#usda 1.0\ndef Material "A" {}\n')]
    client.update_material_graphs(identity, updates)
    assert list(state.desired.values()) == [other]
    assert state.accepted_values.changed([changed, other], epoch=client.runtime.epoch) == [changed]
    assert client.direct_projection_token(identity) == token


def test_material_graph_rejection_does_not_advance_client_certificates(client):
    identity = start(client)
    state = client.states[identity]
    row = dict(prim_path='/M/S', attribute_name='inputs:roughness', value_type='Float', value=.5)
    state.desired = {('/M/S', 'inputs:roughness'): row}
    state.accepted_values.acknowledge([row], epoch=client.runtime.epoch)
    client.runtime.request = lambda op, params: {'material_graph_updated': False}
    from ovrtx_blender_example.ovrtx_runtime_client import ValueUpdateRecoveryRequired
    with pytest.raises(ValueUpdateRecoveryRequired):
        client.update_material_graphs(identity, [dict(material_path='/M', layer_body='#usda 1.0\ndef Material "M" {}\n')])
    assert state.desired == {('/M/S', 'inputs:roughness'): row}
    assert state.accepted_values.changed([row], epoch=client.runtime.epoch) == []


def test_instance_graph_ack_invalidates_changed_projection_only(client):
    identity=start(client);state=client.states[identity]
    changed=dict(prim_path='/World/_DirectGeometry/Object_1/Instances',attribute_name='positions',value_type='Point3fArray',value=[[0,0,0]])
    other=dict(changed,prim_path='/World/_DirectGeometry/Object_2/Instances')
    state.desired={(r['prim_path'],r['attribute_name']):r for r in [changed,other]}
    state.accepted_values.acknowledge([changed,other],epoch=client.runtime.epoch)
    token=client.direct_projection_token(identity);original=client.runtime.request
    client.runtime.request=lambda op,params:({'instance_graph_updated':True,'owner_count':len(params['updates'])} if op=='instance_graph_update' else original(op,params))
    client.update_instance_graphs(identity,[dict(owner_id=1,projection_path='/World/_DirectGeometry/Object_1',hidden_source_path='/World/GN',layer_body='#usda 1.0\n')])
    assert list(state.desired.values())==[other]
    assert state.accepted_values.changed([changed,other],epoch=client.runtime.epoch)==[changed]
    assert client.direct_projection_token(identity)==token

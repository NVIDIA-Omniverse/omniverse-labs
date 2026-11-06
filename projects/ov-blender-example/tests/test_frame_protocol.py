# SPDX-License-Identifier: Apache-2.0
"""Complete control/layout/chunk messages; all fixtures execute on CPU."""
from dataclasses import replace
import importlib.util
import io
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'addon'))
from ovrtx_blender_example import frame_protocol as p
from ovrtx_blender_example.direct_runtime_session import RuntimeSession
from ovrtx_blender_example.ovstream_raw_frame import RawFrameAssembler, RawFrameLayout
from ovrtx_blender_example.ovstream_shm import HeldFrame, ShmFrameError


def message():
    return dict(**p.payload_header('/Render/Product', 'HdrColor'), schema=1,
        encoding='raw_bytes_over_bgra8', byte_order='little', storage_width=3,
        storage_height=4, byte_count=48, planes=[dict(name='linear_rgba16f',
        format='rgba16f', width=3, height=2, offset_bytes=0, row_bytes=24, byte_count=48)])


def chunk(index=0):
    return dict(transfer_id='a'*32, index=index, count=2, offset_bytes=index*24,
                byte_count=24, total_bytes=48)


def held(index=0):
    return HeldFrame('stream', 'a'*64, index+1, 123, 3, 2, 10+index, f'token{index}')


@pytest.mark.parametrize('minor', [0, 1, 65535])
def test_supported_peer_negotiates_exact_selected_semantics(minor, tmp_path):
    peer = p.offer(); peer['typed_minor'] = minor
    peer['optional']['future_diagnostic'] = 1
    assert p.negotiate(peer) == p.offer()
    runtime = RuntimeSession(sys.executable, __file__, 'renderer', tmp_path)
    runtime._session_nonce = 'a'*64; runtime._epoch = 1
    calls = []
    runtime._owner = SimpleNamespace(contains=lambda pid: pid == 123, authenticate=calls.append)
    runtime._accept_hello(dict(protocol=2, session='a'*64, epoch=1, pid=123,
        capabilities=dict(p.CAPABILITIES), application=p.negotiate(peer)))
    assert runtime.pid == 123 and calls == [123]


@pytest.mark.parametrize('field,value', [('control', 0), ('control', 2), ('typed_major', 0),
    ('typed_major', 2), ('carrier_abi', 1), ('carrier_abi', 3), ('typed_minor', True),
    ('typed_minor', -1), ('required', {}), ('required', {'foreign_required': 1}),
    ('optional', {'control': 1})])
def test_incompatible_complete_offers_fail_closed(field, value):
    peer = p.offer(); peer[field] = value
    with pytest.raises(p.ProtocolError): p.negotiate(peer)


def test_required_capability_extension_cannot_hide_in_optional_metadata():
    peer = p.offer(); peer['required']['future_semantics'] = 1
    with pytest.raises(p.ProtocolError, match='CAPABILITY_MISMATCH'): p.negotiate(peer)
    source = message(); source['required_capabilities']['future_semantics'] = 1
    with pytest.raises(ShmFrameError, match='CAPABILITY_MISMATCH'): RawFrameLayout.from_mapping(source)


@pytest.mark.parametrize('path,value', [
    (('version','major'), 0), (('version','major'), 2), (('version','minor'), 1),
    (('carrier_abi',), 1), (('byte_order',), 'big'), (('product_id',), '../Product'),
    (('aov',), 'Depth'), (('byte_count',), 2**80), (('storage_width',), True),
    (('storage_height',), 2**64), (('planes',0,'offset_bytes'), 8),
    (('planes',0,'row_bytes'), 28), (('planes',0,'byte_count'), 47),
    (('planes',0,'width'), 2**80), (('planes',0,'height'), True),
    (('planes',0,'format'), 'bgra8'), (('new_required_field',), 1),
])
def test_malformed_descriptor_is_rejected_before_owned_allocation(monkeypatch, path, value):
    source = message(); item = source
    for part in path[:-1]: item = item[part]
    item[path[-1]] = value
    import ovrtx_blender_example.ovstream_raw_frame as raw
    monkeypatch.setattr(raw, 'bytearray', lambda *_: pytest.fail('allocated malformed frame'), raising=False)
    with pytest.raises(ShmFrameError): RawFrameAssembler(held(), source, chunk())


def test_schema_only_legacy_payload_is_not_an_implicit_downgrade():
    source = message(); del source['version']
    with pytest.raises(ShmFrameError, match='missing required'): RawFrameLayout.from_mapping(source)


def test_optional_extensions_are_ignorable_without_changing_payload():
    source = message(); source['extensions'] = {'future_label': 'diagnostic'}
    source['planes'][0]['extensions'] = {'future_label': 'plane'}
    assert RawFrameLayout.from_mapping(source) == RawFrameLayout.from_mapping(message())


def test_unicode_usd_product_identity_is_preserved():
    from pxr import Sdf
    source = message(); source['product_id'] = '/Réndér/产品'
    assert Sdf.Path.IsValidPathString(source['product_id'])
    p.validate_product(source, product_id=source['product_id'], aov='HdrColor', width=3, height=2)
    assert RawFrameLayout.from_mapping(source).product_id == '/Réndér/产品'


@pytest.mark.parametrize('change', [dict(copy_token='token0'), dict(sequence=1),
    dict(producer_pid=124), dict(epoch='foreign'), dict(stream_name='foreign'),
    dict(capture_timestamp_ns=9), dict(sequence=True), dict(width=2**80)])
def test_contiguous_chunks_reject_stale_tokens_and_wrong_producers_before_read(change):
    assembler = RawFrameAssembler(held(), message(), chunk())
    reader = SimpleNamespace(copy_bytes=lambda _: SimpleNamespace(data=bytes(range(24))))
    assert not assembler.append(reader, held(), message(), chunk())
    reader.copy_bytes = lambda _: pytest.fail('read invalid chunk')
    with pytest.raises(ShmFrameError): assembler.append(reader, replace(held(1), **change), message(), chunk(1))


def test_writer_and_reader_share_identical_layout_policy_before_gpu_allocation(monkeypatch):
    spec = importlib.util.spec_from_file_location('protocol_publisher', ROOT/'runtime/ovstream_frames.py')
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module); spec.loader.exec_module(module)
    assert module._frame_protocol()['payload_header']('/Render/Product', 'HdrColor') == p.payload_header('/Render/Product', 'HdrColor')
    writer = module.ShmPublisher.__new__(module.ShmPublisher)
    writer._lock = threading.Lock(); writer._closed = False; writer._held_token = None
    writer.device = object()
    writer._start = lambda *_: pytest.fail('allocated GPU/SHM for invalid descriptor')
    monkeypatch.setattr(module.wp, 'array', lambda **_: object())
    source = SimpleNamespace(device=writer.device, dtype=module.wp.uint8, ndim=3,
                             shape=(2,3,4), strides=(16,4,1), ptr=1)
    with pytest.raises(ValueError, match='product identifier'):
        writer.publish_raw_planes_cuda([('rgba8','rgba8',source)], product_id='../foreign')


def test_worker_rejects_incompatible_offer_before_socket_duplication_or_native_imports(monkeypatch):
    spec = importlib.util.spec_from_file_location('protocol_worker', ROOT/'runtime/public_sdk_worker.py')
    worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)
    offer = p.offer(); offer['typed_major'] = 900
    challenge = dict(protocol=2, session='a'*64, epoch=1, port=1234, application=offer)
    monkeypatch.setattr(worker.sys, 'stdin', SimpleNamespace(buffer=io.BytesIO(json.dumps(challenge).encode()+b'\n')))
    monkeypatch.setattr(worker.socket, 'create_connection', lambda *_a, **_k: pytest.fail('accepted incompatible worker'))
    with pytest.raises(ValueError, match='PROTOCOL_MISMATCH'):
        worker.receive_control_socket(shared_stdin=True)


def test_complete_writer_descriptor_interoperates_with_owned_reader_on_cpu(monkeypatch):
    spec = importlib.util.spec_from_file_location('protocol_cpu_writer', ROOT/'runtime/ovstream_frames.py')
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module); spec.loader.exec_module(module)
    writer = module.ShmPublisher.__new__(module.ShmPublisher)
    writer._lock = threading.Lock(); writer._closed = False; writer._held_token = None
    writer.device = object(); writer._dimensions = (3, 2)
    writer._take_raw_payload = lambda size: bytes(range(size))
    monkeypatch.setattr(module.wp, 'array', lambda **_: object())
    monkeypatch.setattr(module.wp, 'get_stream', lambda _: None)
    monkeypatch.setattr(module.wp, 'launch', lambda *_a, **_k: None)
    def publish(_):
        transfer = writer._raw_transfer
        return dict(raw_frame=transfer.layout, chunk=dict(transfer_id=transfer.identity,
            index=0, count=transfer.count, offset_bytes=0, byte_count=24, total_bytes=24))
    writer._publish_raw_chunk = publish
    # Odd width, padded borrowed-source stride; metadata describes packed rows.
    source = SimpleNamespace(device=writer.device, dtype=module.wp.uint8, ndim=3,
                             shape=(2,3,4), strides=(16,4,1), ptr=1)
    reply = writer.publish_raw_planes_cuda([('rgba8','rgba8',source)], product_id='/Render/Product', aov='LdrColor')
    assert reply['raw_frame']['planes'][0]['row_bytes'] == 12
    assert reply['raw_frame']['planes'][0]['byte_count'] == 24
    p.validate_product(reply['raw_frame'], product_id='/Render/Product', aov='LdrColor', width=3, height=2)
    assembler = RawFrameAssembler(held(), reply['raw_frame'], reply['chunk'])
    reader = SimpleNamespace(copy_bytes=lambda _: SimpleNamespace(data=writer._raw_transfer.payload))
    assert assembler.append(reader, held(), reply['raw_frame'], reply['chunk'])
    copied = assembler.finish()
    assert copied.payload('rgba8', pixel_format='rgba8', width=3, height=2) == bytes(range(24))

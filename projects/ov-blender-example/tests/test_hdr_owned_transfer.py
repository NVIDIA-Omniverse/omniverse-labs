"""HDR-only publication, owned view lifetimes, and export orientation."""
import importlib.util
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'addon'))
from ovrtx_blender_example.ovrtx_runtime_client import RenderResult, RenderClientError
from ovrtx_blender_example.ovstream_raw_frame import RawFrameAssembler
from test_ovstream_raw_frame import layout, chunk, held_chunk, ChunkReader


def test_staging_default_off_and_explicit_on(monkeypatch):
    from ovrtx_blender_example import direct_sdk_client
    monkeypatch.setattr(direct_sdk_client.sys, 'platform', 'win32')
    monkeypatch.delenv('OVRTX_HDR_GPU_FLOAT32', raising=False)
    assert not direct_sdk_client._prepare_hdr_upload('HdrColor')
    monkeypatch.setenv('OVRTX_HDR_GPU_FLOAT32', '1')
    assert direct_sdk_client._prepare_hdr_upload('HdrColor')
    assert not direct_sdk_client._prepare_hdr_upload('LdrColor')
    monkeypatch.setenv('OVRTX_HDR_GPU_FLOAT32', '0')
    assert not direct_sdk_client._prepare_hdr_upload('HdrColor')


def test_assembled_planes_share_owned_readonly_storage_after_assembler_dies():
    assembler = RawFrameAssembler(held_chunk(0), layout(), chunk(0))
    for i in range(3):
        assembler.append(ChunkReader(), held_chunk(i), layout(), chunk(i))
    result = assembler.finish()
    hdr, ldr = (plane.data for plane in result.planes)
    assert hdr.readonly and ldr.readonly
    assert hdr.obj is ldr.obj
    del assembler, result
    assert bytes(hdr) == bytes(range(48))
    assert bytes(ldr) == bytes(range(48, 72))
    with pytest.raises(TypeError):
        hdr[0] = 9


def test_hdr_only_export_flips_rows_only_at_boundary():
    payload = struct.pack('<8e', .125, .25, .5, 1, 2, 4, 8, 1)
    frame = RenderResult(1, 2, b'', 1, 1, 10, linear_rgba16f=memoryview(payload),
                         frame_format='rgba16f', rgba8_origin='top_left')
    assert frame.linear_rgba16f_bytes(origin='top_left') == payload
    assert frame.linear_rgba16f_bytes() == payload[8:] + payload[:8]
    with pytest.raises(RenderClientError, match='displayed viewport'):
        frame.rgba8_bytes()


def test_worker_hdr_publication_never_requests_or_maps_ldr(monkeypatch):
    spec = importlib.util.spec_from_file_location('hdr_backend_test',
        Path(__file__).parents[1] / 'runtime/public_sdk_backend.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    mapped = SimpleNamespace(unmap=lambda: calls.append('unmap'))
    value = SimpleNamespace(map=lambda **kw: mapped)
    calls = []
    publisher = SimpleNamespace(publish_raw_planes_cuda=lambda planes, **options: calls.append(planes) or {'raw_frame': {}})
    monkeypatch.setitem(sys.modules, 'warp', SimpleNamespace(from_dlpack=lambda x: x))
    monkeypatch.setitem(sys.modules, 'ovstream_frames', SimpleNamespace(ShmPublisher=object))
    backend = object.__new__(module.RendererBackend)
    backend.publisher = publisher
    backend.rtx = SimpleNamespace(Device=SimpleNamespace(CUDA='cuda'))
    backend.epoch, backend.width, backend.height = 'epoch', 1, 2
    backend.ordinal, backend.native_stage_id, backend.simulation_time_ns = 7, 'native', 0
    products = {'/Render/Product': SimpleNamespace(frames=[SimpleNamespace(render_vars={'HdrColor': value})])}
    result = backend._publish_products(products, {'render_var': 'HdrColor'}, ['/Render/Product'], 1)
    assert [p[:2] for p in calls[0]] == [('linear_rgba16f', 'rgba16f')]
    assert calls[1] == 'unmap'
    assert result['render_var'] == 'HdrColor'


def test_pending_publication_uses_frozen_shape_and_ordinal(monkeypatch):
    spec = importlib.util.spec_from_file_location('hdr_pending_test',
        Path(__file__).parents[1] / 'runtime/public_sdk_backend.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    backend = object.__new__(module.RendererBackend)
    backend.pending = {'ticket': {'cancelled': False, 'products': object(),
        'params': {}, 'sensors': ['/Color'], 'samples': 1,
        'frame_metadata': {'width': 12, 'height': 6, 'ordinal': 7, 'native_stage_id': 'old'}}}
    backend._publish_products = lambda *args, frame_metadata: frame_metadata
    backend.width, backend.height, backend.ordinal = 99, 88, 10
    assert backend.poll_render({'ticket': 'ticket'}) == {
        'ready': True, 'width': 12, 'height': 6, 'ordinal': 7, 'native_stage_id': 'old'}
    assert not backend.pending

# SPDX-License-Identifier: Apache-2.0
"""LDR uses typed RGBA bytes at every size and releases borrowed CUDA maps."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('width,height', [(65, 9), (1280, 720), (1024, 1024), (1920, 1080)])
@pytest.mark.parametrize('publication_fails', [False, True])
def test_ldr_preserves_raw_rgba_contract_and_unmaps(monkeypatch, width, height, publication_fails):
    spec = importlib.util.spec_from_file_location('ldr_backend_test',
        Path(__file__).parents[1] / 'runtime/public_sdk_backend.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    mapped = SimpleNamespace(unmap=lambda: calls.append('unmap'))
    value = SimpleNamespace(map=lambda **kw: mapped)
    raw_reply = {'shm': {'copy_token': 'owned-frame'}, 'raw_frame': {'encoding': 'raw_bytes_over_bgra8'}}

    def publish(planes, **metadata):
        assert metadata == {'product_id': '/Render/Product', 'aov': 'LdrColor'}
        assert planes == [('rgba8', 'rgba8', mapped)]
        calls.append('publish')
        if publication_fails:
            raise RuntimeError('publication failed')
        return raw_reply

    publisher = SimpleNamespace(publish_raw_planes_cuda=publish,
        publish_rgba_cuda=lambda *args: pytest.fail('LDR must not convert RGBA to BGRA'))
    monkeypatch.setitem(sys.modules, 'warp', SimpleNamespace(from_dlpack=lambda x: x))
    monkeypatch.setitem(sys.modules, 'ovstream_frames', SimpleNamespace(ShmPublisher=object))
    backend = object.__new__(module.RendererBackend)
    backend.publisher = publisher
    backend.rtx = SimpleNamespace(Device=SimpleNamespace(CUDA='cuda'))
    backend.epoch, backend.width, backend.height = 'epoch', width, height
    backend.ordinal, backend.native_stage_id, backend.simulation_time_ns = 7, 'native', 0
    products = {'/Render/Product': SimpleNamespace(frames=[SimpleNamespace(render_vars={'LdrColor': value})])}
    if publication_fails:
        with pytest.raises(RuntimeError, match='publication failed'):
            backend._publish_products(products, {'render_var': 'LdrColor'}, ['/Render/Product'], 1)
    else:
        result = backend._publish_products(products, {'render_var': 'LdrColor'}, ['/Render/Product'], 1)
        assert result['transport'] == 'ovstream_shm_raw'
        assert result['raw_frame'] == raw_reply['raw_frame']
        assert result['shm']['copy_token'] == 'owned-frame'
        assert result['render_var'] == 'LdrColor'
        assert (result['width'], result['height']) == (width, height)
    assert calls == ['publish', 'unmap']

# SPDX-License-Identifier: Apache-2.0
"""Build 29 follow-up: the worker uses PCI device order and checks CUDA/Vulkan GPU identity.

This machine has one GPU, so two-GPU behaviour runs against fake Warp devices
and a fake OVRTX renderer.
"""
from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'runtime'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'addon'))
import public_sdk_backend
from public_sdk_backend import RendererBackend
from ovrtx_blender_example.direct_runtime_session import worker_environment

VERSION = '0.6.0.1791947691'
LAPTOP_IGPU = {'status': 'known', 'name': 'Intel(R) Iris(R) Xe Graphics',
               'uuid': '8680a07a-0c00-0000-0002-000000000000'}
DGPU_UUID = 'GPU-4f1c2d3e-5a6b-7c8d-9e0f-112233445566'
OTHER_UUID = 'GPU-0a0b0c0d-1111-2222-3333-444455556666'


class ReachedStage(Exception):
    pass


def fake_warp(monkeypatch, devices):
    """A Warp module whose ``cuda:N`` devices are ``devices[N]`` (name, uuid)."""
    warp = ModuleType('warp')
    def get_device(alias):
        ordinal = int(alias.split(':')[1])
        name, uuid = devices[ordinal]
        return SimpleNamespace(is_cuda=True, ordinal=ordinal, name=name, uuid=uuid,
                               pci_bus_id=f'00000000:{ordinal + 1:02X}:00')
    warp.get_device = get_device
    monkeypatch.setitem(sys.modules, 'warp', warp)


def load_until_stage(tmp_path, monkeypatch, *, active='0', vulkan=None):
    """Run ``load`` past renderer creation up to the OVStage scene; return the backend."""
    destroyed = []
    owner = RendererBackend.__new__(RendererBackend)
    owner.output_root = tmp_path
    owner.active_cuda_gpus = active
    owner.gpu_identity = None
    owner.renderer = owner.stage = owner.publisher = None
    owner.pending = {}
    owner.native_stage_id = None
    owner.epoch = 'e'
    owner.rtx = SimpleNamespace(RendererConfig=lambda **kw: kw,
                                Renderer=lambda config: SimpleNamespace(destroy=lambda: destroyed.append(True)))
    def stage(name):
        raise ReachedStage()
    owner.ovstage = SimpleNamespace(Stage=stage)
    owner.destroyed = destroyed
    monkeypatch.setattr(public_sdk_backend, '_ovrtx_version', lambda: VERSION)
    monkeypatch.setattr(public_sdk_backend, '_versions', lambda: {'ovrtx': VERSION})
    if vulkan is not None:
        monkeypatch.setattr(public_sdk_backend, 'vulkan_device_identity', lambda renderer: vulkan)
    monkeypatch.delenv('OV_BLENDER_OVRTX_CACHE_ROOT', raising=False)
    monkeypatch.delenv('UJITSO_DATASTORE_CONFIG', raising=False)
    return owner


def test_worker_env_sets_pci_bus_order_when_unset():
    base = {'PATH': r'C:\Windows', 'PYTHONPATH': r'C:\elsewhere', 'OCIO': 'blender.ocio'}
    environment = worker_environment(base, {'OV_BLENDER_OVRTX_CACHE_ROOT': r'W:\cache'})
    assert environment['CUDA_DEVICE_ORDER'] == 'PCI_BUS_ID'
    assert environment['OV_BLENDER_OVRTX_CACHE_ROOT'] == r'W:\cache'
    assert environment['PATH'] == r'C:\Windows'
    assert 'PYTHONPATH' not in environment and 'OCIO' not in environment
    # An empty value is unset, and the caller's mapping is not modified.
    assert worker_environment({'CUDA_DEVICE_ORDER': ''}, {})['CUDA_DEVICE_ORDER'] == 'PCI_BUS_ID'
    assert 'CUDA_DEVICE_ORDER' not in base


def test_worker_env_keeps_user_device_order():
    assert worker_environment({'CUDA_DEVICE_ORDER': 'FASTEST_FIRST'}, {})['CUDA_DEVICE_ORDER'] == 'FASTEST_FIRST'
    # The session's own environment wins over the inherited one.
    assert worker_environment({}, {'CUDA_DEVICE_ORDER': 'FASTEST_FIRST'})['CUDA_DEVICE_ORDER'] == 'FASTEST_FIRST'


def test_cuda_identity_follows_active_cuda_index(monkeypatch):
    fake_warp(monkeypatch, [('NVIDIA RTX A2000', OTHER_UUID), ('NVIDIA RTX PRO 3000 Laptop GPU', DGPU_UUID)])
    monkeypatch.setenv('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    identity = public_sdk_backend.cuda_device_identity('1,0')
    assert identity == {'status': 'known', 'index': 1, 'name': 'NVIDIA RTX PRO 3000 Laptop GPU',
                        'uuid': DGPU_UUID, 'pci_bus_id': '00000000:02:00', 'device_order': 'PCI_BUS_ID'}
    missing = public_sdk_backend.cuda_device_identity('5')
    assert missing['status'] == 'unavailable' and "'5'" in missing['reason']


def test_identity_mismatch_fails_with_both_names(tmp_path, monkeypatch):
    fake_warp(monkeypatch, [('NVIDIA RTX PRO 3000 Laptop GPU', DGPU_UUID)])
    owner = load_until_stage(tmp_path, monkeypatch, vulkan=LAPTOP_IGPU)
    with pytest.raises(RuntimeError) as failure:
        owner.dispatch('load', {'sensor_paths': ['/Render/Camera']})
    message = str(failure.value)
    assert 'NVIDIA RTX PRO 3000 Laptop GPU' in message and 'Intel(R) Iris(R) Xe Graphics' in message
    assert DGPU_UUID in message and 'OVRTX_ACTIVE_CUDA_GPUS' in message
    # The session start failed before any scene; the renderer was released.
    assert owner.destroyed == [True] and owner.renderer is None and owner.gpu_identity is None
    # Same GPU (Vulkan reports the UUID without the CUDA prefix): the load continues.
    same = {'status': 'known', 'name': 'NVIDIA RTX PRO 3000 Laptop GPU', 'uuid': DGPU_UUID[4:].upper()}
    owner = load_until_stage(tmp_path, monkeypatch, vulkan=same)
    with pytest.raises(ReachedStage):
        owner.dispatch('load', {'sensor_paths': ['/Render/Camera']})
    assert owner.gpu_identity['check'] == 'match'


def test_identity_unavailable_is_recorded_not_fatal(tmp_path, monkeypatch):
    fake_warp(monkeypatch, [('NVIDIA RTX A6000', DGPU_UUID)])
    owner = load_until_stage(tmp_path, monkeypatch)
    with pytest.raises(ReachedStage):
        owner.dispatch('load', {'sensor_paths': ['/Render/Camera']})
    health = owner.dispatch('health', {})
    identity = health['gpu_identity']
    assert identity['cuda']['name'] == 'NVIDIA RTX A6000' and identity['cuda']['uuid'] == DGPU_UUID
    assert identity['vulkan']['status'] == 'unavailable'
    assert VERSION in identity['vulkan']['reason']
    assert identity['check'] == 'unavailable' and identity['reason'].startswith('Vulkan: OVRTX')
    # Without Warp's device either, both reasons are recorded and the load still continues.
    monkeypatch.setitem(sys.modules, 'warp', None)
    owner = load_until_stage(tmp_path, monkeypatch)
    with pytest.raises(ReachedStage):
        owner.dispatch('load', {'sensor_paths': ['/Render/Camera']})
    assert owner.gpu_identity['cuda']['status'] == 'unavailable'
    assert owner.gpu_identity['reason'].startswith('CUDA: Warp could not read')

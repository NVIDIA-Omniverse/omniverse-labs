"""Renderer cache configuration runs before native initialization."""
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from public_sdk_backend import RendererBackend


def test_load_binds_typed_and_environment_cache_to_owned_output(tmp_path, monkeypatch):
    class ReachedNativeInitialization(Exception):
        pass
    configurations=[]
    def initialize(config):
        configurations.append(config)
        raise ReachedNativeInitialization()
    owner=RendererBackend.__new__(RendererBackend)
    owner.output_root=tmp_path/'worker-output'
    owner.active_cuda_gpus='0'
    owner.unload=lambda:None
    owner.rtx=SimpleNamespace(RendererConfig=lambda **kw:kw,Renderer=initialize)
    monkeypatch.setenv('UJITSO_DATASTORE_CONFIG','grpcdns://foreign-cache:1234')
    with pytest.raises(ReachedNativeInitialization):
        owner.dispatch('load',{'sensor_paths':['/Render/Camera']})
    expected='local://'+(owner.output_root/'cache/DerivedDataCache').resolve().as_posix()
    assert configurations[0]['datastore_cache']==expected
    assert os.environ['UJITSO_DATASTORE_CONFIG']==expected
    assert (owner.output_root/'cache/DerivedDataCache').is_dir()

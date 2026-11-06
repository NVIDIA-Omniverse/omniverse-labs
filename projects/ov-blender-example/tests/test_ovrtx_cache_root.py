# SPDX-License-Identifier: Apache-2.0
"""OVRTX shader caches persist with the profile's SDK store, not a worker epoch."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'addon'))

from ovrtx_blender_example import direct_sdk_client, public_sdk_install  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_installer_names_a_versioned_cache_beside_the_sidecars(tmp_path):
    installer = public_sdk_install.Installer()
    assert installer.ovrtx_cache_root() is None
    requirements = ROOT / 'runtime/requirements-public-sdk.txt'
    installer.configure(tmp_path / 'ovrtx-sdk', sys.executable, requirements)
    from ovrtx_blender_example.public_sdk_lock import versions
    expected = (tmp_path / 'ovrtx-sdk').resolve() / 'cache' / ('ovrtx-' + versions(requirements)['ovrtx'])
    assert installer.ovrtx_cache_root() == expected


def test_worker_environment_passes_the_store_cache(monkeypatch, tmp_path):
    monkeypatch.delenv(direct_sdk_client.OVRTX_CACHE_ENV, raising=False)
    monkeypatch.setattr(public_sdk_install, 'ovrtx_cache_root', lambda: tmp_path / 'cache')
    assert direct_sdk_client._worker_environment() == {direct_sdk_client.OVRTX_CACHE_ENV: str(tmp_path / 'cache')}
    monkeypatch.setattr(public_sdk_install, 'ovrtx_cache_root', lambda: None)
    assert direct_sdk_client._worker_environment() == {}


def test_explicit_cache_environment_takes_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv(direct_sdk_client.OVRTX_CACHE_ENV, str(tmp_path / 'explicit'))
    monkeypatch.setattr(public_sdk_install, 'ovrtx_cache_root', lambda: tmp_path / 'store')
    assert direct_sdk_client._worker_environment() == {}


def test_backend_prefers_the_configured_cache_root():
    source = (ROOT / 'runtime/public_sdk_backend.py').read_text(encoding='utf-8')
    assert "os.environ.get('OV_BLENDER_OVRTX_CACHE_ROOT', '')" in source
    assert 'cache_root_path=str(cache_root)' in source

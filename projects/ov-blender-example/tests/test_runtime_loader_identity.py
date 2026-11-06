"""Loader overrides cannot bypass inventory verification before native imports."""
import builtins
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'runtime'))
import runtime_identity as identity
import public_sdk_backend as backend


PATHS = {
    'OVPHYSX_LIB': 'ovphysx/lib/ovphysx.dll',
    'OVSTAGE_LIBRARY_PATH_HINT': 'ovstage/bin',
    'OVRTX_LIBRARY_PATH_HINT': 'ovrtx/bin',
    'OVSTREAM_LIB_PATH': 'ovstream/ovstream.dll',
    'OMNI_PLUGINS_BASE_PATH': 'ovrtx/bin',
    'OMNI_USD_PLUGINS_BASE_PATH': 'ovrtx/bin',
}


@pytest.fixture
def site(tmp_path):
    site = tmp_path / 'site with spaces'
    for name in PATHS.values():
        path = site / name
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'locked')
        else:
            path.mkdir(parents=True, exist_ok=True)
    return site


def test_unset_and_canonical_paths(site):
    identity.validate_loader_overrides({}, site)
    identity.validate_loader_overrides({key: '' for key in PATHS}, site)
    identity.validate_loader_overrides({key: str(site / path) for key, path in PATHS.items()}, site)
    identity.validate_loader_overrides({'OVSTREAM_LIB_PATH': str(site / 'ovstream')}, site)


@pytest.mark.parametrize('key', PATHS)
@pytest.mark.parametrize('value', ['relative/path', ' ', '\x00', 'missing', 'foreign', 'multi'])
def test_foreign_and_malformed_paths_fail_closed(site, tmp_path, key, value):
    if value in ('missing', 'foreign'):
        path = tmp_path / value
        if value == 'foreign':
            path.mkdir()
        value = str(path)
    elif value == 'multi':
        value = str(site / PATHS[key]) + ';' + str(tmp_path)
    with pytest.raises(ValueError, match=key):
        identity.validate_loader_overrides({key: value}, site)


@pytest.mark.parametrize('key', ['OVSTAGE_ALLOW_CWD_LIBRARY_SEARCH', 'OVRTX_ALLOW_CWD_LIBRARY_SEARCH'])
def test_cwd_search_only_allows_disabled(site, key):
    for value in ('', '0'):
        identity.validate_loader_overrides({key: value}, site)
    for value in ('1', 'false', ' ', '-1'):
        with pytest.raises(ValueError, match=key):
            identity.validate_loader_overrides({key: value}, site)


@pytest.mark.parametrize('key', ['PXR_PLUGINPATH_NAME'])
def test_unsupported_plugin_overrides(site, key):
    identity.validate_loader_overrides({key: ''}, site)
    with pytest.raises(ValueError, match=key):
        identity.validate_loader_overrides({key: str(site)}, site)


@pytest.mark.parametrize('package', ['ovstage', 'ovrtx'])
def test_path_fallback_cannot_replace_bundled_loader(site, tmp_path, package):
    canonical = site / package / 'bin'
    name = package + '-dynamic.dll'
    (canonical / name).write_bytes(b'locked')
    identity.validate_loader_overrides({'PATH': str(canonical)}, site)
    foreign = tmp_path / 'foreign'
    foreign.mkdir()
    (foreign / name).write_bytes(b'foreign')
    with pytest.raises(ValueError, match='foreign native loader'):
        identity.validate_loader_overrides({'PATH': str(foreign)}, site)
    development = site.parent / 'bin'
    development.mkdir()
    (development / name).write_bytes(b'foreign')
    with pytest.raises(ValueError, match='foreign native loader'):
        identity.validate_loader_overrides({}, site)


@pytest.mark.parametrize('package', ['ovstage', 'ovrtx'])
def test_existing_module_hint_is_checked(site, monkeypatch, package):
    name = package.upper() + '_LIBRARY_PATH_HINT'
    module = SimpleNamespace(**{name: str(site / package / 'bin')})
    monkeypatch.setitem(sys.modules, package + '._src.bindings', module)
    identity.validate_loader_overrides({}, site)
    setattr(module, name, 'foreign')
    with pytest.raises(ValueError, match='foreign native module'):
        identity.validate_loader_overrides({}, site)


def test_cached_inventory_does_not_hide_changed_loader_state(monkeypatch):
    calls = []
    monkeypatch.setattr(identity, '_verified_inventory', lambda: calls.append(True) or {'verified': True})
    monkeypatch.setattr(identity, 'validate_loader_overrides', lambda: None)
    assert identity.verified_identity() == {'verified': True}
    def reject():
        raise ValueError('foreign override')
    monkeypatch.setattr(identity, 'validate_loader_overrides', reject)
    with pytest.raises(ValueError, match='foreign override'):
        identity.verified_identity()
    assert calls == [True]


@pytest.mark.parametrize('kind', ['RendererBackend', 'PhysicsBackend'])
def test_direct_backend_rejects_before_native_imports(monkeypatch, tmp_path, kind):
    native_imports = []
    original = builtins.__import__
    def observe(name, *args, **kwargs):
        if name in ('numpy', 'ovrtx', 'ovstage', 'ovphysx'):
            native_imports.append(name)
        return original(name, *args, **kwargs)
    def reject():
        raise ValueError('loader override rejected')
    monkeypatch.setattr(identity, 'verified_identity', reject)
    monkeypatch.setattr(builtins, '__import__', observe)
    with pytest.raises(ValueError, match='loader override rejected'):
        getattr(backend, kind)(tmp_path)
    assert native_imports == []

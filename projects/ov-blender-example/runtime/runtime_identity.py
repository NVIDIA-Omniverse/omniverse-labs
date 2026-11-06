# SPDX-License-Identifier: Apache-2.0
"""External worker identity before SDK imports; shared by physics admission."""
from functools import lru_cache
from pathlib import Path
import runpy
import os
import sys
import sysconfig


def validate_loader_overrides(environ=None, site=None):
    """Reject paths that bypass the verified Windows wheel inventory, before imports.

    The SDKs search their bundled DLLs first unless these explicit overrides
    redirect them. PATH remains available for OS/driver dependencies; the locked
    inventory check requires every bundled SDK DLL to exist before loading.
    """
    environ = os.environ if environ is None else environ
    site = Path(site or sysconfig.get_path('purelib')).resolve()
    allowed = {
        'OVPHYSX_LIB': (site / 'ovphysx/lib/ovphysx.dll',),
        'OVSTAGE_LIBRARY_PATH_HINT': (site / 'ovstage/bin',),
        'OVRTX_LIBRARY_PATH_HINT': (site / 'ovrtx/bin',),
        'OVSTREAM_LIB_PATH': (site / 'ovstream', site / 'ovstream/ovstream.dll'),
        'OMNI_PLUGINS_BASE_PATH': (site / 'ovrtx/bin',),
        'OMNI_USD_PLUGINS_BASE_PATH': (site / 'ovrtx/bin',),
    }
    for key, paths in allowed.items():
        value = environ.get(key, '')
        if not value:
            continue
        try:
            path = Path(value)
            valid = path.is_absolute() and path.exists() and path.resolve() in {
                candidate.resolve() for candidate in paths}
        except (OSError, ValueError, TypeError):
            valid = False
        if not valid:
            raise ValueError(f'{key} redirects native loading outside the locked runtime; unset it or use its canonical bundled path')
    for key in ('OVSTAGE_ALLOW_CWD_LIBRARY_SEARCH', 'OVRTX_ALLOW_CWD_LIBRARY_SEARCH'):
        if environ.get(key, '') not in ('', '0'):
            raise ValueError(f'{key} enables unverified native-library search; unset it')
    # The Stage/RTX loaders continue through PATH/dev candidates if a bundled
    # DLL fails to load. Do not let that failure select another SDK instead.
    for package in ('ovstage', 'ovrtx'):
        canonical = (site / package / 'bin').resolve()
        candidates = [site.parent / 'bin']
        candidates.extend(Path(value) for value in environ.get('PATH', '').split(os.pathsep) if value)
        for directory in candidates:
            if (directory / (package + '-dynamic.dll')).is_file() and directory.resolve() != canonical:
                raise ValueError(f'{package} has a foreign native loader in PATH/development search; remove that directory')
        # A direct backend caller may already have imported another SDK. Its
        # public module hint must obey the same invariant as environment hints.
        bindings = sys.modules.get(package + '._src.bindings')
        hint = getattr(bindings, package.upper() + '_LIBRARY_PATH_HINT', None)
        if hint and (not Path(hint).is_absolute() or Path(hint).resolve() != canonical):
            raise ValueError(f'{package} has a foreign native module library hint')
    # This plugin override is unnecessary for the bundled SDKs and can load
    # native plugins outside the hashed wheel inventory.
    for key in ('PXR_PLUGINPATH_NAME',):
        if environ.get(key):
            raise ValueError(f'{key} is not supported by the locked runtime; unset it')


def verified_identity():
    # Recheck mutable loader state even when the expensive inventory is cached.
    validate_loader_overrides()
    return _verified_inventory()


@lru_cache(maxsize=1)
def _verified_inventory():
    root = Path(__file__).resolve().parents[1]
    module = root / 'addon/ovrtx_blender_example/public_sdk_lock.py'
    if not module.is_file():
        module = root / 'ovrtx_blender_example/public_sdk_lock.py'
    policy = runpy.run_path(str(module))
    return policy['inspect_installed'](policy['load_lock'](root / 'runtime/public-sdk-lock.json'))

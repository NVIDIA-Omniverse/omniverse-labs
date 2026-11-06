# SPDX-License-Identifier: Apache-2.0
"""ASCII-safe native locations for the Windows runtime; stdlib only, no SDK imports.

The SDKs' native loaders and caches use narrow (ANSI) Windows paths. With the runtime
under a non-ASCII folder, OVRTX cannot load CRenderApi or its carb plugins, OVStage
cannot load ovstage.dll, and a non-ASCII cache_root_path sends the driver shader
cache to a mis-encoded sibling folder; an 8.3 short spelling does not help, because
the packaged CPython canonicalizes its executable to the long name before it loads
anything. A settings folder whose runtime store is not ASCII therefore keeps that
runtime under an ASCII folder owned by this install, recorded in the store's
``native-root.json`` and removed with the runtime. Python-side paths stay Unicode.
The paired installer ships a copy of this module beside its scripts.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import uuid

RECEIPT = 'native-root.json'
OWNER = 'owner.json'
FOLDER = 'ovrtx-blender'
_MAX_RECORD = 16384


class UnsupportedNativePath(ValueError):
    """No ASCII-safe native location can be used for a Windows runtime."""


def _windows(windows):
    return os.name == 'nt' if windows is None else windows


def ascii_safe(path, *, windows=None):
    """Whether native SDK code can use ``path`` as spelled."""
    return not _windows(windows) or str(path).isascii()


def require_ascii_native(path, what, *, windows=None):
    """Refuse a native path that cannot be remapped (an explicit external interpreter)."""
    if not ascii_safe(Path(path).expanduser().absolute(), windows=windows):
        raise UnsupportedNativePath(
            f'The {what} path contains non-ASCII characters, which the Windows SDK runtime '
            f'cannot load from: {path}. Choose an interpreter in a folder whose path is ASCII.')


def bases(environ=None):
    """Where mapped roots may live: %ProgramData%, then the system drive."""
    environ = os.environ if environ is None else environ
    result = []
    for value in (environ.get('ProgramData') or environ.get('PROGRAMDATA'),
                  (environ.get('SystemDrive') or environ.get('SYSTEMDRIVE') or 'C:') + '\\'):
        if value:
            base = Path(value) / FOLDER
            if base not in result:
                result.append(base)
    return result


def _long(path):
    """Extended-length spelling: runtime trees exceed MAX_PATH for processes without long-path support."""
    value = str(Path(path).absolute())
    if os.name != 'nt' or value.startswith('\\\\?\\'):
        return Path(value)
    return Path('\\\\?\\UNC\\' + value[2:] if value.startswith('\\\\') else '\\\\?\\' + value)


def _key(path):
    return os.path.normcase(str(Path(path).absolute()))


def mapped_name(store):
    """One mapped root per settings store; the name does not reveal the store path."""
    return hashlib.sha256(_key(store).encode('utf-8')).hexdigest()[:16]


def _read(path):
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if (stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400
            or not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_RECORD):
        raise UnsupportedNativePath(f'Unexpected native-root record: {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def _write(path, record):
    temporary = path.with_name(path.name + '-' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def _valid_root(value, environ):
    """A recorded root must be a 16-hex folder directly under one of ``bases``."""
    if not isinstance(value, str) or not value.isascii():
        return None
    root = Path(value)
    if (not root.is_absolute() or not re.fullmatch('[0-9a-f]{16}', root.name)
            or _key(root.parent) not in {_key(base) for base in bases(environ)}):
        return None
    for item in (root, *root.parents):
        try:
            info = item.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            return None
    return root


def recorded_root(store, *, windows=None, environ=None):
    """The mapped root recorded for ``store``, or None when it has none (or it is stale).

    A settings folder moved as a whole keeps its runtime: the root's owner record
    still names the old location, which no longer exists, so the mapping follows the
    move. A copied folder (the old location still exists) gets its own root on install.
    The record is honoured for an ASCII store too: a folder moved from a non-ASCII path
    keeps its root rather than orphaning it.
    """
    if not _windows(windows):
        return None
    store = Path(store).absolute()
    try:
        record = _read(store / RECEIPT)
        if not isinstance(record, dict) or record.get('schema') != 1:
            return None
        root = _valid_root(record.get('native_root'), environ)
        if root is None:
            return None
        owner = _read(root / OWNER)
        if not isinstance(owner, dict) or owner.get('schema') != 1:
            return None
        recorded = owner.get('settings_store')
        if not isinstance(recorded, str) or _key(record.get('settings_store', '')) != _key(recorded):
            return None
        if _key(recorded) != _key(store):
            if Path(recorded).exists():
                return None
            # Whole-folder relocation: adopt the root for the new location.
            _write(root / OWNER, dict(owner, settings_store=str(store)))
            _write(store / RECEIPT, dict(record, settings_store=str(store)))
        return root
    except (OSError, ValueError, UnicodeError):
        return None


def native_storage_root(store, *, create=False, windows=None, environ=None):
    """The runtime folder for the settings store ``store`` that native code can use.

    ``store`` itself when its path is ASCII (or off Windows), unless the store records a
    mapped root it owns: a settings folder moved from a non-ASCII to an ASCII path keeps
    using that root until Remove Runtime deletes it. Otherwise the recorded mapped root;
    with ``create`` a new one is made under the first writable ASCII base and recorded.
    Without ``create`` and without a record, return the root an install would try first
    (not created). Raises UnsupportedNativePath when none can be made.
    """
    store = Path(store).absolute()
    if not _windows(windows):
        return store
    recorded = recorded_root(store, windows=windows, environ=environ)
    if ascii_safe(store, windows=windows):
        return store if recorded is None else recorded
    if recorded is not None and (not create or recorded.is_dir()):
        return recorded
    candidates = [base / mapped_name(store) for base in bases(environ)]
    if not create:
        return next((root for root in candidates if root.is_dir()), candidates[0] if candidates else store)
    reasons = []
    for root in candidates:
        if not str(root).isascii():
            reasons.append(f'{root} is not ASCII')
            continue
        try:
            if _valid_root(str(root), environ) is None:
                raise UnsupportedNativePath('it is a link or reparse point')
            root.mkdir(parents=True, exist_ok=True)
            owner = _read(root / OWNER)
            if owner is not None and _key(owner.get('settings_store', '')) != _key(store):
                raise UnsupportedNativePath('it belongs to another settings folder')
            _write(root / OWNER, dict(schema=1, settings_store=str(store)))
            store.mkdir(parents=True, exist_ok=True)
            _write(store / RECEIPT, dict(schema=1, native_root=str(root), settings_store=str(store),
                                         reason='settings folder path is not ASCII'))
            return root
        except (OSError, ValueError) as error:
            reasons.append(f'{root}: {error}')
    raise UnsupportedNativePath(
        'The settings folder path contains non-ASCII characters, which the Windows SDK '
        f'runtime cannot load from ({store}), and no ASCII folder for the runtime could be '
        'created (' + '; '.join(reasons or ['no candidate folder']) + '). Make '
        f'%ProgramData%\\{FOLDER} writable, or install into a settings folder whose path is ASCII.')


def remove_mapped_root(store, root, *, windows=None, environ=None):
    """Delete a mapped root this install owns, then its record; return what was removed.

    Only a root recorded for ``store`` whose owner record names ``store`` is deleted,
    and never through a link or reparse point. The empty ``ovrtx-blender`` base goes too.
    """
    store, root = Path(store).absolute(), Path(root).absolute()
    if not _windows(windows) or root == store:
        return []
    if recorded_root(store, windows=windows, environ=environ) != root:
        raise UnsupportedNativePath('The runtime folder is not the one recorded for this settings folder')
    removed = []
    def failed(error):
        raise error
    for parent, directories, files in os.walk(_long(root), onerror=failed, followlinks=False):
        for name in (*directories, *files):
            info = (Path(parent) / name).lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
                raise UnsupportedNativePath(f'Refusing to remove a link inside the runtime folder: {Path(parent) / name}')
    shutil.rmtree(_long(root))
    removed.append(str(root))
    try:
        root.parent.rmdir()
        removed.append(str(root.parent))
    except OSError:
        pass  # Other settings folders' roots remain.
    (store / RECEIPT).unlink(missing_ok=True)
    removed.append(str(store / RECEIPT))
    return removed

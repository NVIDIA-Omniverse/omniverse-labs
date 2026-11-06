# SPDX-License-Identifier: Apache-2.0
"""Owned immutable USD and asset versions for acknowledged renderer rollback.

Capture only at population changes, never in camera/render polling. Keep the
current full scene and the bounded set of native source references alive until
their replacement is acknowledged. USD performs composition and asset rewriting.
"""
from dataclasses import dataclass
import glob
import hashlib
import os
from pathlib import Path
import re
import shutil
import uuid


def _ordinary_path(value):
    """Keep extended Windows spelling out of USD and ownership/cache keys."""
    value = os.fspath(value)
    if os.name == 'nt':
        value = os.path.normpath(value)
        if value.startswith('\\\\?\\UNC\\'):
            value = '\\\\' + value[8:]
        elif value.startswith('\\\\?\\'):
            value = value[4:]
    return Path(value)


def _io_path(value):
    """Filesystem access only; SDK resolvers receive ordinary owned paths."""
    path = _ordinary_path(value)
    if os.name != 'nt':
        return path
    value = os.path.abspath(path)
    return Path('\\\\?\\UNC\\' + value[2:] if value.startswith('\\\\') else '\\\\?\\' + value)


def _canonical_path(value):
    return _ordinary_path(_io_path(value).resolve())


_TEXTURE_SUFFIXES = frozenset(('.exr', '.hdr', '.png', '.jpg', '.jpeg', '.tif', '.tiff',
                             '.bmp', '.tga', '.dds', '.tx', '.ktx', '.ktx2', '.webp'))


def _texture_names(value, files):
    """Compact texture names without changing module/dependency identifiers.

    Full source names and bytes still participate in the directory SHA-256.
    Tile identifiers remain in both the authored pattern and each copied file.
    """
    name = Path(value).name
    suffix = Path(name).suffix
    unchanged = name, {item: item.name for item in files}
    if suffix.lower() not in _TEXTURE_SUFFIXES:
        return unchanged
    tokens = re.findall(r'<UDIM>|<UVTILE>', name)
    if not tokens:
        return ('texture' + suffix, {files[0]: 'texture' + suffix}) if len(files) == 1 else unchanged
    parts = re.split(r'(<UDIM>|<UVTILE>)', name)
    expression = ''.join(r'(\d{4})' if part == '<UDIM>' else
                         r'(u-?\d+_v-?\d+)' if part == '<UVTILE>' else re.escape(part)
                         for part in parts)
    matches = {item: re.fullmatch(expression, item.name) for item in files}
    if not all(matches.values()):
        return unchanged
    return ('texture.' + '.'.join(tokens) + suffix,
            {item: 'texture.' + '.'.join(match.groups()) + suffix for item, match in matches.items()})


@dataclass(frozen=True)
class SceneSnapshot:
    path: str
    assets: frozenset[str]


class ImmutableSceneStore:
    def __init__(self, root):
        self.root = _canonical_path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.snapshots = {}
        self.asset_cache = {}
        self.cleanup_errors = []

    def capture(self, path, *, flatten=True):
        from pxr import Sdf, Usd
        if len(self.cleanup_errors) >= 64:
            raise OSError('Scene snapshot cleanup backlog requires releasing locked output files')
        if flatten:
            stage = Usd.Stage.Open(str(path))
            if stage is None:
                raise ValueError('Cannot snapshot accepted scene: ' + str(path))
            # USD caches clean parsed layers. An external editor may replace a
            # shared layer at the same path between accepted generations.
            for source in stage.GetUsedLayers():
                if source.realPath and not source.dirty:
                    source.Reload(force=True)
            sources = [_io_path(source.realPath) for source in stage.GetUsedLayers() if source.realPath]
            signatures = [(item, item.stat()) for item in sources]
            layer = stage.Flatten()
            if any((item.stat().st_size, item.stat().st_mtime_ns) != (before.st_size, before.st_mtime_ns)
                   for item, before in signatures):
                raise ValueError('Scene source changed during immutable capture')
        else:
            source = Sdf.Layer.FindOrOpen(str(path))
            if source is None:
                raise ValueError('Cannot snapshot scene overlay: ' + str(path))
            layer = Sdf.Layer.CreateAnonymous('immutable-overlay.usda')
            layer.TransferContent(source)
        return self._capture_layer(layer)

    def capture_inline(self, text):
        from pxr import Sdf
        layer = Sdf.Layer.CreateAnonymous('immutable-inline.usda')
        if not layer.ImportFromString(text):
            raise ValueError('Cannot snapshot inline scene source')
        snapshot = self._capture_layer(layer)
        return snapshot, Sdf.Layer.FindOrOpen(snapshot.path).ExportToString()

    def capture_value(self, row):
        from pxr import Sdf
        layer = Sdf.Layer.CreateAnonymous('immutable-value.usda')
        prim = Sdf.CreatePrimInLayer(layer, '/Value')
        array = row['value_type'].lower().endswith('[]')
        attribute = Sdf.AttributeSpec(prim, 'asset', Sdf.ValueTypeNames.AssetArray if array else Sdf.ValueTypeNames.Asset)
        attribute.default = ([Sdf.AssetPath(value) for value in row['value']] if array else Sdf.AssetPath(row['value']))
        snapshot = self._capture_layer(layer)
        captured = Sdf.Layer.FindOrOpen(snapshot.path)
        value = captured.GetAttributeAtPath('/Value.asset').default
        return snapshot.path, {**row, 'value': [item.path for item in value] if array else value.path}

    def _capture_layer(self, layer):
        from pxr import UsdUtils
        assets = set()

        def immutable_asset(value):
            if not value:
                return value
            # Flatten resolves authored relative assets. Inline producers use
            # absolute paths; unresolved SDK/builtin assets retain resolver IDs.
            if Path(value).is_absolute():
                value = _ordinary_path(value).as_posix()
            pattern = value.replace('<UDIM>', '*').replace('<UVTILE>', '*')
            # Keep authored member names (including linked UDIM tiles). Resolving
            # symlinks here would substitute unrelated target basenames.
            files = sorted(_ordinary_path(item) for item in glob.glob(str(_io_path(pattern)))
                           if _io_path(item).is_file())
            if not files:
                if Path(value).is_absolute() and not _canonical_path(value).is_relative_to(self.root):
                    # Preserve a missing local asset as missing even if someone
                    # creates the original filename after this ACK. Resolver
                    # identifiers (for example builtin MDL modules) stay intact.
                    return (self.root / ('missing-' + hashlib.sha256(value.encode()).hexdigest()) / Path(value).name).as_posix()
                return value
            if all(item.is_relative_to(self.root) and _canonical_path(item) == item for item in files):
                assets.update(str(item) for item in files)
                return value
            def signatures_for_files():
                rows = []
                for item in files:
                    info = _io_path(item).stat()
                    # Equal-size writes can share the same filesystem clock
                    # tick. Metadata alone cannot certify immutable contents.
                    with _io_path(item).open('rb') as source:
                        content = hashlib.file_digest(source, 'sha256').hexdigest()
                    rows.append((str(item), str(_canonical_path(item)), info.st_size,
                                 info.st_mtime_ns, info.st_ctime_ns, content))
                return tuple(rows)
            signatures = signatures_for_files()
            cached = self.asset_cache.get((value, signatures))
            if cached is not None and all(Path(item).is_file() for item in cached[1]):
                assets.update(cached[1])
                return cached[0]
            temporary = self.root / ('asset-' + uuid.uuid4().hex)
            temporary.mkdir()
            try:
                digest = hashlib.sha256()
                pattern_name, names = _texture_names(value, files)
                for original in files:
                    target = temporary / names[original]
                    digest.update(original.name.encode('utf-8'))
                    with _io_path(original).open('rb') as source, target.open('wb') as output:
                        for chunk in iter(lambda: source.read(1024 * 1024), b''):
                            digest.update(chunk)
                            output.write(chunk)
                # Refuse a torn capture; leave the live predecessor untouched.
                after = signatures_for_files()
                if after != signatures:
                    raise ValueError('Scene asset changed during immutable capture: ' + value)
                directory = self.root / ('assets-' + digest.hexdigest())
                if directory.exists():
                    shutil.rmtree(temporary)
                else:
                    temporary.replace(directory)
                owned = frozenset(str(directory / names[item]) for item in files)
                result = (directory / pattern_name).as_posix()
                self.asset_cache[(value, signatures)] = result, owned
                assets.update(owned)
                return result
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)

        UsdUtils.ModifyAssetPaths(layer, immutable_asset)
        # Binary USD keeps large geometry bounded and preserves typed values.
        temporary = self.root / ('scene-' + uuid.uuid4().hex + '.usdc')
        try:
            if not layer.Export(str(temporary)):
                raise ValueError('Cannot serialize immutable scene snapshot')
            digest = hashlib.sha256()
            with temporary.open('rb') as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b''):
                    digest.update(chunk)
            destination = self.root / ('scene-' + digest.hexdigest() + '.usdc')
            if not destination.exists():
                temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        snapshot = SceneSnapshot(str(destination), frozenset(assets))
        self.snapshots[snapshot.path] = snapshot
        return snapshot

    def retain(self, paths):
        """Release only files made by this owner after native source ACK."""
        keep = set(paths)
        self.cleanup_errors = []
        live_assets = set().union(*(item.assets for path, item in self.snapshots.items() if path in keep))
        for path, item in tuple(self.snapshots.items()):
            if path not in keep:
                try:
                    Path(path).unlink(missing_ok=True)
                    self.snapshots.pop(path)
                except OSError as error:
                    self.cleanup_errors.append(str(error))
        for directory in self.root.glob('assets-*'):
            if not any(str(item) in live_assets for item in directory.iterdir()):
                try:
                    shutil.rmtree(directory)
                except OSError as error:
                    self.cleanup_errors.append(str(error))
        self.asset_cache = {key: value for key, value in self.asset_cache.items()
                            if set(value[1]) <= live_assets}

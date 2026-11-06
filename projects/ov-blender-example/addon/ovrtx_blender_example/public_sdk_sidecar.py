# SPDX-License-Identifier: Apache-2.0
"""Verify and stage a user-selected local public SDK sidecar, without imports.

The package contains its own relocatable Python. No SDK is installed into
Blender, and selecting a local archive never needs network access.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
import sys
import uuid
import zipfile

MAX_BYTES = 16 * 1024**3
MAX_FILES = 100_000
MAX_MANIFEST_BYTES = 16 * 1024**2
MAX_GENERATED_CACHE_FILES = 512
MAX_GENERATED_CACHE_BYTES = 1024**3
MAX_GENERATED_CACHE_FILE_BYTES = 512 * 1024**2


def _filesystem_path(path):
    """Blender's Windows process may lack long-path awareness; keep receipts canonical."""
    if os.name != 'nt':
        return path
    value = str(path.absolute())
    if value.startswith('\\\\?\\'):
        return Path(value)
    return Path('\\\\?\\UNC\\' + value[2:] if value.startswith('\\\\') else '\\\\?\\' + value)


def runtime_storage_root(extension_storage, *, config_root=None, local_appdata=None, windows=None):
    """Keep explicit installations profile-owned and Windows plugin paths short."""
    extension_storage = Path(extension_storage).resolve()
    if windows is None:
        windows = os.name == 'nt'
    if config_root:
        # The paired installer owns this location on both platforms; keeping
        # it beside config also preserves whole-profile relocation.
        return Path(config_root).absolute().parent / 'ovrtx-sdk'
    if not windows:
        return extension_storage / 'public-sdk'
    identity = hashlib.sha256(str(extension_storage).casefold().encode()).hexdigest()[:16]
    if local_appdata:
        base = Path(local_appdata).absolute() / 'OVRTX' / 'sdk'
    else:
        raise ValueError('Windows SDK installation needs a local profile storage directory')
    return base / identity


def platform_id():
    architecture = platform.machine().lower()
    if sys.platform == 'linux' and architecture in ('aarch64', 'arm64'):
        return 'linux-arm64'
    if sys.platform in ('win32', 'linux') and architecture in ('amd64', 'x86_64'):
        return 'windows-x64' if sys.platform == 'win32' else 'linux-x64'
    raise ValueError('Public SDK sidecars support Linux arm64/x64 and Windows x64')


def package_path(name):
    if (not isinstance(name, str) or not name or any(c in name for c in '\\:<>"|?*')
            or any(ord(c) < 32 for c in name)):
        raise ValueError('Invalid sidecar member path')
    path = PurePosixPath(name)
    if path.is_absolute() or any(p in ('', '.', '..') for p in name.split('/')):
        raise ValueError('Unsafe sidecar member path')
    if any(p.endswith((' ', '.')) or re.fullmatch(r'(?i)(con|prn|aux|nul|com[1-9¹²³]|lpt[1-9¹²³])(?:\..*)?', p) for p in path.parts):
        raise ValueError('Sidecar path uses a reserved Windows name')
    return path


def expected_versions(requirements):
    from .public_sdk_lock import versions
    return versions(requirements)


def _manifest_json(data):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate key in sidecar manifest')
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=unique_object)


def validate_manifest(manifest, requirements):
    if (not isinstance(manifest, dict) or manifest.get('schema') != 1 or manifest.get('backend') != 'public_sdk'
            or manifest.get('platform') != platform_id()):
        raise ValueError('Sidecar format/backend/platform does not match this add-on and host')
    if manifest.get('requirements') != expected_versions(requirements):
        raise ValueError('Sidecar SDK versions do not match this add-on')
    from .public_sdk_lock import load_lock, lock_identity, validate_integrity_record, LOCK_NAME
    lock = load_lock(Path(requirements).with_name(LOCK_NAME))
    if manifest.get('runtime_identity') != lock_identity(lock):
        raise ValueError('Sidecar complete runtime lock/ABI identity does not match this add-on')
    python = str(package_path(manifest.get('python')))
    if not python.startswith('python/'):
        raise ValueError('Sidecar interpreter must be inside its own python directory')
    files = manifest.get('files')
    if not isinstance(files, dict) or not files or len(files) > MAX_FILES or python not in files:
        raise ValueError('Sidecar manifest lacks a complete file roster/interpreter')
    folded = set()
    for name, digest in files.items():
        package_path(name)
        if name.endswith('.pth') or PurePosixPath(name).name in ('sitecustomize.py', 'usercustomize.py'):
            raise ValueError('Sidecar contains an unsupported Python startup hook')
        if name.casefold() in folded or not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
            raise ValueError('Duplicate sidecar path or invalid file digest')
        folded.add(name.casefold())
    if 'sidecar.json' in folded:
        raise ValueError('Sidecar manifest cannot list itself as a payload')
    for name in files:
        if any(str(parent).casefold() in folded for parent in PurePosixPath(name).parents):
            raise ValueError('Sidecar file conflicts with a parent directory')
    # A self-consistent sidecar file roster is not provenance. Bind every SDK
    # byte to the shipped lock before the packaged interpreter may execute.
    packages = validate_integrity_record(lock, manifest.get('runtime_integrity', {}))
    site = str(package_path(manifest.get('site_packages')))
    if not site.startswith('python/'):
        raise ValueError('Sidecar site-packages must be owned by its interpreter')
    for package in packages.values():
        for relative, expected in package.items():
            if files.get(site + '/' + relative) != expected:
                raise ValueError('Sidecar package bytes disagree with trusted wheel inventory')
    return manifest


def validate_pair(data, requirements):
    """Bind all executable bytes, including Python, to the selected addon."""
    marker = Path(requirements).parent.parent / 'public-sdk.json'
    if marker.is_file():
        expected = json.loads(marker.read_text()).get('sidecar_manifest_sha256')
        if expected is not None and hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('Sidecar manifest does not match the selected add-on package')


def no_reparse_owned(path, root):
    path, root = Path(path).absolute(), Path(root).absolute()
    if '..' in path.parts or '..' in root.parts:
        raise ValueError('Owned sidecar path contains parent traversal')
    path.relative_to(root)
    for item in (path, *path.parents):
        try:
            info = _filesystem_path(item).lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('Owned sidecar path contains a link/reparse point')
    return path


def _installed_files(directory, root):
    """Use long Windows paths and propagate every directory traversal error."""
    scan_root = _filesystem_path(directory)
    def failed(error):
        raise error
    for parent, directories, files in os.walk(scan_root, topdown=True, onerror=failed, followlinks=False):
        relative_parent = Path(parent).relative_to(scan_root)
        for name in (*directories, *files):
            path = directory/relative_parent/name
            no_reparse_owned(path, root)
            mode = _filesystem_path(path).stat().st_mode
            if stat.S_ISREG(mode):
                yield path
            elif not stat.S_ISDIR(mode):
                raise ValueError('Unexpected installed sidecar filesystem entry')


def installed_cache_inventory(directory, root, manifest):
    """Classify bounded generated GPU/texture data, never shipped provenance.

    Only installed qualified candidates may accumulate these locked-version cache
    formats. Archive validation stays exact. Shader cache data contains GPU code;
    it is mutable vendor output and does not inherit the wheel's trusted hashes.
    """
    allowed = set(manifest['files']) | {'sidecar.json'}
    cache_prefix = manifest['site_packages'] + '/ovrtx/bin/cache/'
    from . import public_sdk_materialx_cache as materialx
    platform = manifest.get('platform')
    materialx_platform = {'windows-x64': 'windows', 'linux-x64': 'linux'}.get(platform, 'unsupported')
    materialx_prefix = str(PurePosixPath(manifest['python']).parent) + f'/cache/rtx.materialx/v2/{materialx_platform}/release/local/'
    supported = (platform in {'windows-x64', 'linux-x64'} and
        manifest.get('runtime_identity', {}).get('versions', {}).get('ovrtx') == '0.6.0.1791947691')
    files = {}; total = 0; materialx_records = {}
    for path in _installed_files(directory, root):
        relative = path.relative_to(directory).as_posix()
        if relative in allowed:
            continue
        tail = relative[len(cache_prefix):] if relative.startswith(cache_prefix) else ''
        shader = re.fullmatch(r'nv_shadercache/vulkan/GLCache/[0-9a-f]{32}/[0-9a-f]{16}/[0-9a-f]{16}\.(bin|toc)', tail)
        # Observed locked Windows/Linux ShaderDb output; only the obfuscated SPV
        # formats measured during real rendering are admitted here. ShaderDb
        # appends .l to SPV library caches; these omit the reflection hash.
        shaderdb = re.fullmatch(r'shadercache/vulkan/gpu_foundation/[0-9a-f]{32}\.v(\.l)?(\.desc)?', tail)
        library = bool(shaderdb and shaderdb.group(1))
        descriptor = bool(shaderdb and shaderdb.group(2))
        metadata = tail == 'texturecache/14/_cache.json'
        texture = re.fullmatch(r'texturecache/14/[^/\\:\x00-\x1f]{1,160}_14[0-9]_[0-9]{1,10}_[0-9]{1,10}_(?:[0-9]{1,20}_[0-9]{1,20}|[0-9a-fA-F]{1,128})_[0-9]{1,10}\.ovtex', tail)
        materialx_tail = relative[len(materialx_prefix):] if relative.startswith(materialx_prefix) else ''
        materialx_kind, materialx_limit = materialx.cache_kind(materialx_tail)
        # Both hosts produced the same bounded ShaderDb and MaterialX formats.
        # Linux texture payloads remain unqualified; only metadata was observed.
        admitted = shader or shaderdb or metadata or materialx_kind or (
            platform == 'windows-x64' and texture)
        if not supported or not admitted:
            raise ValueError('Unexpected installed sidecar file: '+relative)
        io_path = _filesystem_path(path)
        size = io_path.stat().st_size
        total += size
        if (len(files) >= MAX_GENERATED_CACHE_FILES or total > MAX_GENERATED_CACHE_BYTES
                or size > MAX_GENERATED_CACHE_FILE_BYTES or (metadata and size > 4096) or (descriptor and size > 256)
                or (materialx_kind and size > materialx_limit)
                or (shader and shader.group(1) == 'toc' and size > 16*1024**2)):
            raise ValueError('Generated vendor cache exceeds installed cache bounds')
        limit = min(MAX_GENERATED_CACHE_FILE_BYTES, MAX_GENERATED_CACHE_BYTES - (total-size),
            materialx_limit if materialx_kind else 256 if descriptor else 4096 if metadata else 16*1024**2 if shader and shader.group(1) == 'toc' else MAX_GENERATED_CACHE_FILE_BYTES)
        digest = hashlib.sha256(); observed = 0; metadata_bytes = bytearray()
        with io_path.open('rb') as stream:
            while chunk := stream.read(min(1024**2, limit-observed+1)):
                if observed == 0 and chunk.startswith((b'MZ', b'\x7fELF')):
                    raise ValueError('Executable disguised as generated cache data')
                observed += len(chunk)
                if observed > limit:
                    raise ValueError('Generated vendor cache exceeds installed cache bounds')
                digest.update(chunk)
                if metadata or descriptor or materialx_kind:
                    metadata_bytes.extend(chunk)
        if observed != size or io_path.stat().st_size != size:
            raise ValueError('Generated vendor cache changed during validation')
        if metadata and json.loads(metadata_bytes) != {'name':'Texture Cache','maxsize':100}:
            raise ValueError('Unexpected texture cache metadata')
        if descriptor:
            pattern = rb'ShaderDb Disk Cache version 1\.81\nslang-2026\.5\.2\n1\n0\n[0-9a-f]{32}\n'
            if not library:
                pattern += rb'[0-9a-f]{32}\n'
            if not re.fullmatch(pattern,metadata_bytes):
                raise ValueError('Unexpected ShaderDb cache descriptor')
        if shaderdb:
            partner = relative[:-5] if descriptor else relative+'.desc'
            if not _filesystem_path(directory/partner).is_file():
                raise ValueError('Incomplete ShaderDb cache pair')
        if materialx_kind:
            materialx_records[relative] = materialx.inspect_cache(materialx_tail, bytes(metadata_bytes))
        files[relative] = dict(bytes=size, sha256=digest.hexdigest())
    materialx.validate_pairs(materialx_records)
    return dict(files=files, total_bytes=total, trusted_artifact_provenance=False)


def installed_python(directory, root, requirements, *, verify_files=True):
    """Locate an installed interpreter; validate payloads during installation."""
    manifest_path = no_reparse_owned(directory/'sidecar.json', root)
    if _filesystem_path(manifest_path).stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError('Sidecar manifest is too large')
    data = _filesystem_path(manifest_path).read_bytes()
    validate_pair(data, requirements)
    manifest = validate_manifest(_manifest_json(data), requirements)
    if verify_files:
        installed_cache_inventory(directory, root, manifest)
        for name, expected in manifest['files'].items():
            path = no_reparse_owned(directory.joinpath(*package_path(name).parts), root)
            with _filesystem_path(path).open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                    raise ValueError('Installed sidecar file hash mismatch: '+name)
    return no_reparse_owned(directory.joinpath(*package_path(manifest['python']).parts), root)


def materialize_local(source, storage_root, requirements, *, cancelled=lambda: False,
                      progress=lambda message, completed, total: None):
    """Copy/hash into a fresh candidate; caller verifies Python and publishes it.

    Never execute code from an unchecked archive, mutate the selected package,
    replace an active runtime, follow links, or extract paths outside candidate.
    """
    source, store = Path(source).expanduser().absolute(), Path(storage_root).absolute()
    archive = None
    candidate = None
    try:
        try:
            source_info = source.lstat()
        except FileNotFoundError as error:
            raise ValueError('Choose a sidecar ZIP or extracted sidecar folder') from error
        if source.is_symlink() or getattr(source_info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('Sidecar source is a link/reparse point')
        if source.is_file():
            archive = zipfile.ZipFile(source)
            members = archive.infolist()
            if len(members) > MAX_FILES + 1:
                raise ValueError('Sidecar has too many files')
            names, folded = {}, set()
            for entry in members:
                name = entry.filename
                mode = entry.external_attr >> 16
                if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)):
                    raise ValueError('Sidecar archive contains a link or special file')
                if entry.is_dir():
                    package_path(name.rstrip('/'))
                    key = name.rstrip('/').casefold()
                    if key in folded:
                        raise ValueError('Duplicate sidecar directory')
                    folded.add(key)
                    continue
                package_path(name)
                if stat.S_IFMT(mode) not in (0, stat.S_IFREG):
                    raise ValueError('Sidecar archive contains a link or special file')
                if name.casefold() in folded or entry.flag_bits & 1:
                    raise ValueError('Duplicate or encrypted sidecar member')
                folded.add(name.casefold()); names[name] = entry
            if 'sidecar.json' not in names or names['sidecar.json'].file_size > MAX_MANIFEST_BYTES:
                raise ValueError('Sidecar manifest is absent or too large')
            manifest_bytes = archive.read(names['sidecar.json'])
            validate_pair(manifest_bytes, requirements)
            manifest = _manifest_json(manifest_bytes)
            validate_manifest(manifest, requirements)
            if set(names) != set(manifest['files']) | {'sidecar.json'}:
                raise ValueError('Sidecar file roster differs from its manifest')
            total = sum(entry.file_size for entry in names.values())
            def open_member(name):
                return archive.open(names[name])
            def mode_for(name):
                return names[name].external_attr >> 16
        elif source.is_dir():
            marker = source / 'sidecar.json'
            if marker.is_symlink() or marker.stat().st_size > MAX_MANIFEST_BYTES:
                raise ValueError('Invalid sidecar manifest')
            manifest_bytes = marker.read_bytes()
            validate_pair(manifest_bytes, requirements)
            manifest = _manifest_json(manifest_bytes)
            validate_manifest(manifest, requirements)
            names = {}; identities = {}; total = 0
            for path in source.rglob('*'):
                info = path.lstat()
                if path.is_symlink() or getattr(info, 'st_file_attributes', 0) & 0x400:
                    raise ValueError('Sidecar folder contains a link/reparse point')
                if path.is_dir():
                    continue
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError('Sidecar contains a special file')
                name = path.relative_to(source).as_posix(); package_path(name)
                names[name] = path; total += info.st_size
                identities[name] = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
                if len(names) > MAX_FILES + 1 or total > MAX_BYTES:
                    raise ValueError('Sidecar exceeds size budget')
            if set(names) != set(manifest['files']) | {'sidecar.json'}:
                raise ValueError('Sidecar file roster differs from its manifest')
            def open_member(name):
                fd = os.open(names[name], os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_BINARY', 0))
                try:
                    info = os.fstat(fd)
                    if (not stat.S_ISREG(info.st_mode) or
                            (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != identities[name]):
                        raise ValueError('Sidecar source changed while preparing installation')
                    return os.fdopen(fd, 'rb')
                except BaseException:
                    os.close(fd)
                    raise
            def mode_for(name):
                return names[name].stat().st_mode
        else:
            raise ValueError('Choose a sidecar ZIP or extracted sidecar folder')
        if total > MAX_BYTES:
            raise ValueError('Sidecar exceeds the 16 GiB uncompressed budget')
        _filesystem_path(store).mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(store).free < total + 1024**3:
            raise ValueError('Not enough disk space to stage sidecar and retain 1 GiB free')
        candidate = store / 'sidecars' / uuid.uuid4().hex
        _filesystem_path(candidate).mkdir(parents=True)
        copied = 0
        for name, digest in manifest['files'].items():
            if cancelled():
                raise RuntimeError('Sidecar installation cancelled')
            destination = _filesystem_path(candidate.joinpath(*package_path(name).parts))
            destination.parent.mkdir(parents=True, exist_ok=True)
            sha = hashlib.sha256()
            with open_member(name) as incoming, destination.open('xb') as outgoing:
                while chunk := incoming.read(1024 * 1024):
                    if cancelled():
                        raise RuntimeError('Sidecar installation cancelled')
                    copied += len(chunk)
                    if copied > total or copied > MAX_BYTES:
                        raise ValueError('Sidecar expanded beyond its declared size')
                    sha.update(chunk); outgoing.write(chunk)
            if sha.hexdigest() != digest:
                raise ValueError(f'Sidecar checksum mismatch: {name}')
            if os.name != 'nt':
                destination.chmod(0o755 if mode_for(name) & 0o111 else 0o644)
            progress('Verifying sidecar package', copied, total)
        _filesystem_path(candidate / 'sidecar.json').write_bytes(manifest_bytes)
        return {'python': str(candidate.joinpath(*PurePosixPath(manifest['python']).parts)),
                'root': str(candidate), 'manifest': manifest}
    except BaseException:
        if candidate is not None:
            shutil.rmtree(_filesystem_path(candidate), ignore_errors=True)
        raise
    finally:
        if archive is not None:
            archive.close()

# SPDX-License-Identifier: Apache-2.0
"""Untrusted package validation without executing packaged Python or SDKs."""
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import zipfile
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'addon'))
from ovrtx_blender_example import public_sdk_sidecar as sidecar
from ovrtx_blender_example import public_sdk_lock as policy
from ovrtx_blender_example import public_sdk_materialx_cache as materialx


def materialx_pair(source, manifest):
    cache_platform = {'windows-x64': 'windows', 'linux-x64': 'linux'}[manifest['platform']]
    root = source/Path(manifest['python']).parent/f'cache/rtx.materialx/v2/{cache_platform}/release/local'
    module = 'omni_mtlx_v2_' + 'b'*32
    mdl = root/module/'materialx.mdl'; mdl.parent.mkdir(parents=True)
    code = b'mdl 1.10;\nexport material example() = material();\n'
    mdl.write_bytes(code)
    body = b'nodes 1\nnode 61 62\nports 0\n'
    metadata = root/'.metadata'/('a'*32+'.metadata'); metadata.parent.mkdir()
    data = ('rtx.materialx.metadata.v2\nlookup '+ 'a'*32+'\ngenerator '+'c'*32+
            '\nmodule '+module+'\nmdl '+materialx.persistent_digest('omni.materialx.persistent-mdl.v2',code)+
            '\nmetadata '+materialx.persistent_digest('omni.materialx.persistent-metadata.v2',body)+'\n').encode()+body
    metadata.write_bytes(data)
    return root, metadata, mdl


@pytest.mark.parametrize('fault', ['', 'lookup', 'schema', 'generator', 'node_count', 'odd_hex',
    'duplicate', 'trailing', 'body_digest', 'mdl_digest', 'missing_mdl', 'missing_metadata',
    'mdl_header', 'exe', 'nul', 'metadata_size', 'mdl_size', 'wrong_root', 'unknown_file',
    'nested', 'file_count', 'total_bytes', 'platform', 'version'])
@pytest.mark.parametrize('platform', ['windows-x64', 'linux-x64'])
def test_materialx_pairs_are_bounded_untrusted_installed_cache(package, monkeypatch, fault, platform):
    source, store, requirements, manifest = package
    manifest['platform']=platform
    manifest['runtime_identity']['versions']['ovrtx']='0.6.0.1791947691'
    root, metadata, mdl = materialx_pair(source,manifest)
    data = metadata.read_bytes()
    if fault=='lookup':data=data.replace(b'lookup '+b'a'*32,b'lookup '+b'd'*32)
    if fault=='schema':data=data.replace(b'metadata.v2',b'metadata.v3')
    if fault=='generator':data=data.replace(b'generator '+b'c'*32,b'generator BAD')
    if fault=='node_count':data=data.replace(b'nodes 1',b'nodes 99999')
    if fault=='odd_hex':data=data.replace(b'node 61 62',b'node 6 62')
    if fault=='duplicate':data=data.replace(b'nodes 1\nnode 61 62',b'nodes 2\nnode 61 62\nnode 61 62')
    if fault=='trailing':data+=b'unexpected\n'
    if fault=='body_digest':data=data.replace(b'node 61 62',b'node 61 63')
    if fault=='mdl_digest':mdl.write_bytes(mdl.read_bytes()+b'// changed\n')
    if fault=='metadata_size':data+=b'x'*(materialx.MAX_METADATA_BYTES+1)
    metadata.write_bytes(data)
    if fault=='missing_mdl':mdl.unlink()
    if fault=='missing_metadata':metadata.unlink()
    if fault=='mdl_header':mdl.write_bytes(b'import os\n')
    if fault=='exe':mdl.write_bytes(b'MZnot-mdl')
    if fault=='nul':mdl.write_bytes(b'mdl 1.10;\n\x00')
    if fault=='mdl_size':monkeypatch.setattr(materialx,'MAX_MDL_BYTES',16)
    if fault=='wrong_root':root.rename(root.with_name('unapproved'))
    if fault=='unknown_file':(root/'sitecustomize.py').write_text('pass')
    if fault=='nested':
        (metadata.parent/'nested').mkdir();metadata.rename(metadata.parent/'nested'/metadata.name)
    if fault=='file_count':monkeypatch.setattr(sidecar,'MAX_GENERATED_CACHE_FILES',1)
    if fault=='total_bytes':monkeypatch.setattr(sidecar,'MAX_GENERATED_CACHE_BYTES',16)
    if fault=='platform':manifest['platform']='linux-x64' if platform=='windows-x64' else 'windows-x64'
    if fault=='version':manifest['runtime_identity']['versions']['ovrtx']='unknown'
    if fault:
        with pytest.raises(ValueError):sidecar.installed_cache_inventory(source,source,manifest)
    else:
        result=sidecar.installed_cache_inventory(source,source,manifest)
        assert len(result['files'])==2 and result['trusted_artifact_provenance'] is False
        with pytest.raises(ValueError):sidecar.materialize_local(source,store,requirements)
        preserved(store)


def test_materialx_shipped_bytes_remain_pinned(package):
    source,store,requirements,manifest=package
    root,metadata,mdl=materialx_pair(source,manifest)
    for path in (metadata,mdl):manifest['files'][path.relative_to(source).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    (source/'sidecar.json').write_text(json.dumps(manifest))
    result=sidecar.materialize_local(source,store,requirements)
    directory=Path(result['root'])
    sidecar.installed_python(directory,store,requirements)
    (directory/mdl.relative_to(source)).write_bytes(mdl.read_bytes()+b'// changed\n')
    with pytest.raises(ValueError,match='file hash mismatch'):
        sidecar.installed_python(directory,store,requirements)


def test_installed_interpreter_lookup_does_not_scan_payload(package, monkeypatch):
    source, store, requirements, manifest = package
    result = sidecar.materialize_local(source, store, requirements)
    directory = Path(result['root'])
    def unexpected(*args, **kwargs):
        pytest.fail('Startup must not scan or hash installed payloads')
    monkeypatch.setattr(sidecar, 'installed_cache_inventory', unexpected)
    monkeypatch.setattr(hashlib, 'file_digest', unexpected)
    assert sidecar.installed_python(directory, store, requirements, verify_files=False) == directory / manifest['python']


def test_materialx_reparse_parent_refused(package,monkeypatch):
    source,store,requirements,manifest=package
    root,metadata,mdl=materialx_pair(source,manifest)
    original=Path.lstat
    def altered(path,*args,**kwargs):
        info=original(path,*args,**kwargs)
        if path in (root,sidecar._filesystem_path(root)):
            return SimpleNamespace(st_mode=info.st_mode,st_file_attributes=0x400)
        return info
    monkeypatch.setattr(Path,'lstat',altered)
    with pytest.raises(ValueError,match='link/reparse'):
        sidecar.installed_cache_inventory(source,source,manifest)


def test_windows_native_root_scope_allows_spaces_and_preserves_portable_unicode(tmp_path):
    from ovrtx_blender_example import native_paths
    native_paths.require_ascii_native(tmp_path/'artist with spaces', 'runtime interpreter', windows=True)
    native_paths.require_ascii_native(tmp_path/'artist \u03a9', 'runtime interpreter', windows=False)
    with pytest.raises(ValueError, match='non-ASCII characters'):
        native_paths.require_ascii_native(tmp_path/'artist \u03a9', 'runtime interpreter', windows=True)


@pytest.mark.parametrize('name', [
    'nv_shadercache/vulkan/GLCache/'+'a'*32+'/'+'b'*16+'/'+'c'*16+'.bin',
    'nv_shadercache/vulkan/GLCache/'+'a'*32+'/'+'b'*16+'/'+'c'*16+'.toc',
    'texturecache/14/wood_140_2_3_123_456_0.ovtex',
    'texturecache/14/wood_140_2_3_abcdef_0.ovtex',
    'texturecache/14/_cache.json',
])
@pytest.mark.parametrize('platform', ['windows-x64', 'linux-x64'])
def test_installed_only_vendor_cache_is_separate_from_trusted_payload(package, name, platform):
    source, store, requirements, manifest = package
    manifest['platform'] = platform
    manifest['runtime_identity']['versions']['ovrtx'] = '0.6.0.1791947691'
    path = source/manifest['site_packages']/'ovrtx/bin/cache'/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'{"name":"Texture Cache","maxsize":100}' if name.endswith('.json') else b'cache-data')
    if platform == 'linux-x64' and name.endswith('.ovtex'):
        with pytest.raises(ValueError, match='Unexpected installed sidecar'):
            sidecar.installed_cache_inventory(source, source, manifest)
        return
    result = sidecar.installed_cache_inventory(source, source, manifest)
    assert result['trusted_artifact_provenance'] is False
    assert list(result['files']) == [path.relative_to(source).as_posix()]


@pytest.mark.parametrize('library', [False, True])
@pytest.mark.parametrize('fault', ['', 'compiler', 'flags', 'header', 'extra', 'oversize', 'unpaired', 'missing_binary', 'executable', 'unknown_suffix', 'nested', 'binary_bound'])
@pytest.mark.parametrize('platform', ['windows-x64', 'linux-x64'])
def test_observed_shaderdb_pair_is_bounded_and_exact(package, fault, library, monkeypatch, platform):
    source, store, requirements, manifest = package
    manifest['platform']=platform
    manifest['runtime_identity']['versions']['ovrtx']='0.6.0.1791947691'
    directory=source/manifest['site_packages']/'ovrtx/bin/cache/shadercache/vulkan/gpu_foundation'
    directory.mkdir(parents=True)
    binary=directory/('a'*32+('.v.l' if library else '.v'));descriptor=Path(str(binary)+'.desc')
    binary.write_bytes(b'compiled-shader-data')
    data=b'ShaderDb Disk Cache version 1.81\nslang-2026.5.2\n1\n0\n'+b'b'*32+b'\n'
    if not library:data+=b'c'*32+b'\n'
    if fault=='compiler':data=data.replace(b'2026.5.2',b'2026.5.3')
    if fault=='flags':data=data.replace(b'\n1\n0\n',b'\n0\n0\n')
    if fault=='header':data=data.replace(b'version 1.81',b'version 1.9')
    if fault=='extra':data+=b'sitecustomize.py\n'
    if fault=='oversize':data+=b'x'*257
    descriptor.write_bytes(data)
    if fault=='unpaired':descriptor.unlink()
    if fault=='missing_binary':binary.unlink()
    if fault=='binary_bound':monkeypatch.setattr(sidecar,'MAX_GENERATED_CACHE_FILE_BYTES',16)
    if fault=='executable':binary.write_bytes(b'MZexecutable')
    if fault=='unknown_suffix':binary.rename(binary.with_suffix('.dll'))
    if fault=='nested':
        (directory/'nested').mkdir();binary.rename(directory/'nested'/binary.name)
    if fault:
        with pytest.raises(ValueError):sidecar.installed_cache_inventory(source,source,manifest)
    else:
        result=sidecar.installed_cache_inventory(source,source,manifest)
        assert len(result['files'])==2 and result['trusted_artifact_provenance'] is False
        # Generated cache acceptance never changes the package extraction rule.
        with pytest.raises(ValueError):sidecar.materialize_local(source,store,requirements)


@pytest.mark.parametrize('name', [
    'texturecache/14/sitecustomize.py','texturecache/14/tool.exe','texturecache/14/foreign.dll',
    'texturecache/14/unknown.ovtex','texturecache/14/nested/wood_140_2_3_123_456_0.ovtex',
    'texturecache/14/_cache.json.pyc','DerivedDataCache/largecachedata_0',
    'nv_shadercache/vulkan/GLCache/unknown.bin',
])
def test_cache_camouflage_and_unknown_shapes_refused(package, name):
    source, store, requirements, manifest = package
    manifest['platform'] = 'windows-x64'
    path=source/manifest['site_packages']/'ovrtx/bin/cache'/name
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'payload')
    with pytest.raises(ValueError,match='Unexpected installed sidecar'):
        sidecar.installed_cache_inventory(source,source,manifest)


@pytest.mark.parametrize('limit', ['MAX_GENERATED_CACHE_FILES','MAX_GENERATED_CACHE_BYTES','MAX_GENERATED_CACHE_FILE_BYTES'])
def test_generated_cache_bounds_fail_closed(package,monkeypatch,limit):
    source, store, requirements, manifest = package
    manifest['platform']='windows-x64'
    path=source/manifest['site_packages']/'ovrtx/bin/cache/texturecache/14/_cache.json'
    path.parent.mkdir(parents=True);path.write_text('{"name":"Texture Cache","maxsize":100}')
    monkeypatch.setattr(sidecar,limit,0)
    with pytest.raises(ValueError,match='cache bounds'):
        sidecar.installed_cache_inventory(source,source,manifest)


@pytest.mark.parametrize('platform', ['windows-x64', 'linux-x64'])
@pytest.mark.parametrize('magic', [b'MZ', b'\x7fELF'])
def test_cache_metadata_and_executable_disguise_refused(package, platform, magic):
    source, store, requirements, manifest = package
    manifest['platform']=platform
    path=source/manifest['site_packages']/'ovrtx/bin/cache/texturecache/14/_cache.json'
    path.parent.mkdir(parents=True);path.write_text('{"name":"Texture Cache","maxsize":100,"foreign":true}')
    with pytest.raises(ValueError,match='texture cache metadata'):
        sidecar.installed_cache_inventory(source,source,manifest)
    path.unlink()
    disguised=source/manifest['site_packages']/('ovrtx/bin/cache/nv_shadercache/vulkan/GLCache/'+ 'a'*32+'/'+ 'b'*16+'/'+ 'c'*16+'.bin')
    disguised.parent.mkdir(parents=True)
    disguised.write_bytes(magic+b'fake-native')
    with pytest.raises(ValueError,match='Executable disguised'):
        sidecar.installed_cache_inventory(source,source,manifest)


def test_archive_cache_extras_still_rejected_and_shipped_cache_hash_stays_required(package,tmp_path):
    source,store,requirements,manifest=package
    relative=manifest['site_packages']+'/ovrtx/bin/cache/texturecache/14/_cache.json'
    path=source/relative
    path.parent.mkdir(parents=True);path.write_bytes(b'{"name":"Texture Cache","maxsize":100}')
    with pytest.raises(ValueError):
        sidecar.materialize_local(archive(source,tmp_path),store,requirements)
    preserved(store)
    manifest['files'][relative]=hashlib.sha256(path.read_bytes()).hexdigest()
    (source/'sidecar.json').write_text(json.dumps(manifest))
    result=sidecar.materialize_local(source,store,requirements)
    directory=Path(result['root'])
    sidecar.installed_python(directory,store,requirements)
    (directory/relative).write_bytes(path.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='file hash mismatch'):
        sidecar.installed_python(directory,store,requirements)


def test_generated_cache_reparse_refused_without_windows_privilege(package,monkeypatch):
    source,store,requirements,manifest=package
    path=source/manifest['site_packages']/'ovrtx/bin/cache/texturecache/14/_cache.json'
    path.parent.mkdir(parents=True);path.write_text('{"name":"Texture Cache","maxsize":100}')
    original=Path.lstat
    def lstat(value,*args,**kwargs):
        result=original(value,*args,**kwargs)
        if value in (path, sidecar._filesystem_path(path)):
            return SimpleNamespace(st_mode=result.st_mode,st_file_attributes=0x400)
        return result
    monkeypatch.setattr(Path,'lstat',lstat)
    with pytest.raises(ValueError,match='link/reparse'):
        sidecar.installed_cache_inventory(source,source,manifest)


def test_growing_cache_stream_cannot_bypass_initial_stat_budget(package,monkeypatch):
    source,store,requirements,manifest=package
    manifest['platform']='windows-x64'
    tail='nv_shadercache/vulkan/GLCache/'+'a'*32+'/'+'b'*16+'/'+'c'*16+'.bin'
    path=source/manifest['site_packages']/'ovrtx/bin/cache'/tail
    path.parent.mkdir(parents=True);path.write_bytes(b'x'*64)
    original=Path.stat
    def stat(value,*args,**kwargs):
        result=original(value,*args,**kwargs)
        if value in (path, sidecar._filesystem_path(path)):
            return SimpleNamespace(st_mode=result.st_mode,st_size=1,st_file_attributes=getattr(result,'st_file_attributes',0))
        return result
    monkeypatch.setattr(Path,'stat',stat)
    monkeypatch.setattr(sidecar,'MAX_GENERATED_CACHE_FILE_BYTES',16)
    with pytest.raises(ValueError,match='cache bounds'):
        sidecar.installed_cache_inventory(source,source,manifest)


def test_cache_directory_scan_errors_are_not_silently_skipped(package,monkeypatch):
    source,store,requirements,manifest=package
    def walk(path,**options):
        options['onerror'](PermissionError('owned cache scan denied'))
        yield
    monkeypatch.setattr(sidecar.os,'walk',walk)
    with pytest.raises(PermissionError,match='scan denied'):
        sidecar.installed_cache_inventory(source,source,manifest)


def test_long_cache_path_keeps_canonical_names_and_uses_safe_io(package):
    source,store,requirements,manifest=package
    manifest['platform']='windows-x64'
    name='t'*150+'_140_2_3_123_456_0.ovtex'
    relative=manifest['site_packages']+'/ovrtx/bin/cache/texturecache/14/'+name
    path=source/relative
    sidecar._filesystem_path(path.parent).mkdir(parents=True)
    sidecar._filesystem_path(path).write_bytes(b'texture-cache')
    assert len(str(path))>260
    result=sidecar.installed_cache_inventory(source,source,manifest)
    assert list(result['files'])==[relative]


@pytest.mark.parametrize('architecture', ['', 'x86', 'ppc64'])
def test_sidecar_rejects_unsupported_architecture(monkeypatch, architecture):
    monkeypatch.setattr(sidecar.platform, 'machine', lambda: architecture)
    with pytest.raises(ValueError, match='support Linux arm64/x64'):
        sidecar.platform_id()


@pytest.mark.parametrize('windows', [False, True])
def test_runtime_root_is_short_and_profile_isolated(tmp_path, windows):
    config = tmp_path/'profiles'/'artist'/'config'
    first = sidecar.runtime_storage_root(tmp_path/'extension-one', config_root=config, windows=windows)
    second = sidecar.runtime_storage_root(tmp_path/'extension-two', config_root=config, windows=windows)
    assert first == tmp_path/'profiles'/'artist'/'ovrtx-sdk'
    assert first == second
    assert first != sidecar.runtime_storage_root(tmp_path/'extension-one', config_root=tmp_path/'profiles/other/config', windows=windows)
    assert first == sidecar.runtime_storage_root(tmp_path/'extension-one', config_root=config, windows=windows)
    default = sidecar.runtime_storage_root(tmp_path/'extension-one', local_appdata=tmp_path, windows=True)
    assert default.parent == tmp_path/'OVRTX'/'sdk'
    assert sidecar.runtime_storage_root(tmp_path/'extension-one', windows=False) == tmp_path/'extension-one/public-sdk'


@pytest.fixture
def package(tmp_path):
    source=tmp_path/'source';source.mkdir()
    requirements=tmp_path/'requirements.txt'
    runtime = Path(__file__).resolve().parents[1] / 'runtime'
    requirements.write_bytes((runtime / 'requirements-public-sdk.txt').read_bytes())
    lock = policy.load_lock(runtime / policy.LOCK_NAME)
    inventory = {'example.py': hashlib.sha256(b'value=42\n').hexdigest()}
    for row in lock['platforms'][sidecar.platform_id()]['artifacts'].values():
        row['files_sha256'] = hashlib.sha256(policy.canonical(inventory)).hexdigest()
    requirements.with_name(policy.LOCK_NAME).write_text(json.dumps(lock))
    files={'python/bin/python':b'not executed\n','python/lib/example.py':b'value=42\n'}
    manifest={'schema':1,'backend':'public_sdk','platform':sidecar.platform_id(),
              'python':'python/bin/python','requirements':policy.versions(requirements),
              'runtime_identity':policy.lock_identity(lock), 'site_packages': 'python/lib',
              'runtime_integrity': {'identity': policy.lock_identity(lock),
                  'packages': {name: inventory for name in policy.REQUIRED}},
              'files':{name:hashlib.sha256(data).hexdigest() for name,data in files.items()}}
    for name,data in files.items():
        path=source/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    (source/'python/bin/python').chmod(0o755)
    (source/'sidecar.json').write_text(json.dumps(manifest))
    store=tmp_path/'storage';store.mkdir();old=store/'prior-runtime';old.mkdir();(old/'keep').write_text('old verified runtime')
    (store/'ready.json').write_text('{"old":"unchanged"}')
    return source,store,requirements,manifest


def archive(source,tmp_path,extras=()):
    path=tmp_path/'sidecar.zip'
    with zipfile.ZipFile(path,'w') as output:
        for file in sorted(source.rglob('*')):
            if file.is_file():output.write(file,file.relative_to(source).as_posix())
        for name,content,mode in extras:
            info=zipfile.ZipInfo(name);info.external_attr=mode<<16;output.writestr(info,content)
    return path


def preserved(store):
    assert (store/'ready.json').read_text()=='{"old":"unchanged"}'
    assert (store/'prior-runtime/keep').read_text()=='old verified runtime'
    assert not (store/'sidecars').exists() or not list((store/'sidecars').iterdir())


@pytest.mark.parametrize('zipped',[False,True])
def test_local_package_stages_verified_fresh_candidate_without_replacing_ready(package,tmp_path,zipped):
    source,store,requirements,manifest=package
    selected=archive(source,tmp_path) if zipped else source
    updates=[]
    result=sidecar.materialize_local(selected,store,requirements,progress=lambda *args:updates.append(args))
    candidate=Path(result['root']);assert candidate.parent==store/'sidecars'
    assert Path(result['python']).read_bytes()==(source/'python/bin/python').read_bytes()
    assert result['manifest']==manifest and updates
    assert (store/'ready.json').read_text()=='{"old":"unchanged"}'
    if os.name!='nt':assert Path(result['python']).stat().st_mode&0o111


@pytest.mark.parametrize('change', ['relocated', 'interpreter_bytes', 'self_consistent_manifest',
                                   'wrong_interpreter', 'sidecars_reparse', 'nested_reparse', 'python_reparse',
                                   'startup_hook', 'unexpected_bytecode'])
def test_ready_receipt_relocation_validates_paired_owned_interpreter(package,tmp_path,monkeypatch,change):
    from ovrtx_blender_example import public_sdk_install as installer
    source,store,old_requirements,manifest=package
    requirements=tmp_path/'paired/runtime/requirements.txt';requirements.parent.mkdir(parents=True)
    requirements.write_bytes(old_requirements.read_bytes())
    requirements.with_name(policy.LOCK_NAME).write_bytes(old_requirements.with_name(policy.LOCK_NAME).read_bytes())
    (requirements.parent.parent/'public-sdk.json').write_text(json.dumps({
        'sidecar_manifest_sha256':hashlib.sha256((source/'sidecar.json').read_bytes()).hexdigest()}))
    selected=sidecar.materialize_local(source,store,requirements)
    directory=Path(selected['root']);python=Path(selected['python'])
    pins,digest=installer._pins(requirements)
    receipt=dict(schema=2,source_kind='local_package',requirements_sha256=digest,versions=pins,
                 environment=str(directory),python=str(python),
                 environment_relative=directory.relative_to(store).as_posix(),
                 python_relative=python.relative_to(store).as_posix())
    (store/'ready.json').write_text(json.dumps(receipt))
    relocated=tmp_path/('Relocated space' if os.name == 'nt' else 'Relocated space \u03a9');store.rename(relocated)
    directory=relocated/receipt['environment_relative'];python=relocated/receipt['python_relative']
    if change in ('interpreter_bytes','self_consistent_manifest'):
        python.write_bytes(b'tampered interpreter')
        if change=='self_consistent_manifest':
            manifest['files'][manifest['python']]=hashlib.sha256(python.read_bytes()).hexdigest()
            (directory/'sidecar.json').write_text(json.dumps(manifest))
    if change=='wrong_interpreter':
        other=directory/'python/other.exe';other.write_text('not executed')
        receipt['python_relative']=other.relative_to(relocated).as_posix()
        (relocated/'ready.json').write_text(json.dumps(receipt))
    if change in ('startup_hook','unexpected_bytecode'):
        extra=directory/('python/lib/sitecustomize.py' if change=='startup_hook' else 'python/lib/__pycache__/example.cpython-313.pyc')
        extra.parent.mkdir(parents=True,exist_ok=True);extra.write_bytes(b'unlisted executable content')
    if change.endswith('_reparse'):
        target={'sidecars_reparse':directory.parent,'nested_reparse':python.parent,'python_reparse':python}[change]
        original=Path.lstat
        def lstat(path,*args,**kwargs):
            result=original(path,*args,**kwargs)
            return SimpleNamespace(st_mode=result.st_mode,st_file_attributes=0x400) if path in (target,sidecar._filesystem_path(target)) else result
        monkeypatch.setattr(Path,'lstat',lstat)
    if change == 'relocated':
        def unexpected_scan(*args, **kwargs):
            pytest.fail('Restoring the installed runtime must not scan payloads')
        monkeypatch.setattr(sidecar, 'installed_cache_inventory', unexpected_scan)
        monkeypatch.setattr(hashlib, 'file_digest', unexpected_scan)
    restored=installer.Installer();restored.configure(relocated,sys.executable,requirements)
    # Startup resolves the installed selection. Payload validation belongs to
    # installation, rather than a full file scan on every Blender launch.
    payload_changes = ('interpreter_bytes', 'startup_hook', 'unexpected_bytecode')
    assert restored.ready_python()==(str(python) if change=='relocated' or change in payload_changes else None)
    if change in payload_changes:
        with pytest.raises(ValueError):
            sidecar.installed_python(directory, relocated, requirements)


@pytest.mark.parametrize('zipped',[False,True])
def test_hash_mismatch_removes_only_candidate(package,tmp_path,zipped):
    source,store,requirements,_=package;(source/'python/lib/example.py').write_text('tampered')
    with pytest.raises(ValueError,match='checksum'):
        sidecar.materialize_local(archive(source,tmp_path) if zipped else source,store,requirements)
    preserved(store)


def test_self_consistent_tampered_manifest_is_not_trusted(package):
    source, store, requirements, manifest = package
    (source / 'python/lib/example.py').write_bytes(b'changed SDK code')
    manifest['files']['python/lib/example.py'] = hashlib.sha256(b'changed SDK code').hexdigest()
    (source / 'sidecar.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='trusted wheel inventory'):
        sidecar.materialize_local(source, store, requirements)
    preserved(store)


def test_long_windows_destination_extracts_and_receipts_stay_canonical(package, tmp_path):
    source, store, requirements, manifest = package
    name = 'python/resources/' + '/'.join(['long-resource-segment-' + 'a'*40]*4) + '/data.bin'
    payload = b'long path resource'
    manifest['files'][name] = hashlib.sha256(payload).hexdigest()
    (source/'sidecar.json').write_text(json.dumps(manifest))
    selected = archive(source, tmp_path, [(name, payload, stat.S_IFREG|0o644)])
    result = sidecar.materialize_local(selected, store, requirements)
    candidate = Path(result['root'])
    assert not str(candidate).startswith('\\\\?\\')
    assert sidecar._filesystem_path(candidate/name).read_bytes() == payload
    assert result['manifest'] == manifest


@pytest.mark.parametrize('name',['../escape','/absolute','python/../escape','python//x','python/./x',
    'C:/escape','C:escape','\\\\server\\share','python\\file','python/data:stream',
    'python/con.txt','python/LPT²','python/com¹.dll','python/trailing.','python/trailing ',
    'python/a?b','python/a*b','python/a|b','python/<x>','python/"x"','python/a\nb'])
def test_unsafe_and_windows_paths_rejected(name):
    with pytest.raises(ValueError):sidecar.package_path(name)


@pytest.mark.parametrize('extra',[('../escape',b'x',stat.S_IFREG|0o644),
    ('python/link',b'/outside',stat.S_IFLNK|0o777),('python/link/',b'/outside',stat.S_IFLNK|0o777),
    ('python/bin/PYTHON',b'duplicate',stat.S_IFREG|0o644)])
def test_malicious_archive_rejected_before_candidate(package,tmp_path,extra):
    source,store,requirements,_=package
    with pytest.raises(ValueError):sidecar.materialize_local(archive(source,tmp_path,[extra]),store,requirements)
    preserved(store)


def test_exact_duplicate_archive_member_rejected(package,tmp_path):
    source,store,requirements,_=package
    with pytest.warns(UserWarning,match='Duplicate'):
        selected=archive(source,tmp_path,[('python/bin/python',b'duplicate',stat.S_IFREG|0o644)])
    with pytest.raises(ValueError,match='Duplicate'):sidecar.materialize_local(selected,store,requirements)
    preserved(store)


@pytest.mark.parametrize('mutation',[{'platform':'unsupported-platform'},{'requirements':{'ovstream':'0.0'}},{'schema':2}])
def test_platform_versions_schema_rejected_before_execution(package,mutation):
    source,store,requirements,manifest=package;manifest.update(mutation)
    (source/'sidecar.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):sidecar.materialize_local(source,store,requirements)
    preserved(store)


def test_cancel_during_copy_preserves_prior_runtime(package):
    source,store,requirements,_=package
    cancel=False
    def progress(*_):
        nonlocal cancel
        cancel=True
    with pytest.raises(RuntimeError,match='cancelled'):
        sidecar.materialize_local(source,store,requirements,cancelled=lambda:cancel,progress=progress)
    preserved(store)


def test_folder_file_replacement_during_copy_is_rejected(package):
    source,store,requirements,_=package
    def progress(*_):
        target=source/'python/lib/example.py';replacement=source/'swap';replacement.write_text('foreign')
        replacement.replace(target)
    with pytest.raises(ValueError,match='changed'):
        sidecar.materialize_local(source,store,requirements,progress=progress)
    preserved(store)


def test_folder_symlink_is_not_followed(package,tmp_path):
    source,store,requirements,_=package
    target=source/'python/lib/example.py';target.unlink()
    try:target.symlink_to(tmp_path/'foreign')
    except OSError:pytest.skip('Host cannot create symlinks')
    with pytest.raises(ValueError,match='link'):sidecar.materialize_local(source,store,requirements)
    preserved(store)


def test_windows_reparse_attribute_rejected_without_symlink_privilege(package, monkeypatch):
    from types import SimpleNamespace
    source, store, requirements, _ = package
    target = source / 'python/lib/example.py'
    original = Path.lstat
    def lstat(path, *args, **kwargs):
        info = original(path, *args, **kwargs)
        if path in (target,sidecar._filesystem_path(target)):
            return SimpleNamespace(st_file_attributes=0x400, st_mode=info.st_mode)
        return info
    monkeypatch.setattr(Path, 'lstat', lstat)
    with pytest.raises(ValueError, match='link/reparse'):
        sidecar.materialize_local(source, store, requirements)
    preserved(store)


def test_manifest_file_directory_collision_rejected(package):
    _,_,requirements,manifest=package
    manifest['files']['python']=hashlib.sha256(b'prefix').hexdigest()
    with pytest.raises(ValueError,match='parent directory'):sidecar.validate_manifest(manifest,requirements)


def test_duplicate_manifest_keys_rejected(package):
    source,store,requirements,_=package
    text=(source/'sidecar.json').read_text().replace('"schema": 1','"schema": 0, "schema": 1')
    (source/'sidecar.json').write_text(text)
    with pytest.raises(ValueError,match='Duplicate key'):sidecar.materialize_local(source,store,requirements)
    preserved(store)


def test_expansion_budget_rejects_before_copy(package,monkeypatch):
    source,store,requirements,_=package
    monkeypatch.setattr(sidecar,'MAX_BYTES',1)
    with pytest.raises(ValueError,match='budget'):sidecar.materialize_local(source,store,requirements)
    preserved(store)


@pytest.mark.parametrize('system,architecture,expected', [
    ('linux', 'aarch64', 'linux-arm64'), ('linux', 'arm64', 'linux-arm64'),
    ('linux', 'x86_64', 'linux-x64'), ('win32', 'AMD64', 'windows-x64')])
def test_sidecar_uses_exact_native_platform(monkeypatch, system, architecture, expected):
    monkeypatch.setattr(sidecar.sys, 'platform', system)
    monkeypatch.setattr(sidecar.platform, 'machine', lambda: architecture)
    assert sidecar.platform_id() == expected


def test_sidecar_windows_arm64_remains_unadmitted(monkeypatch):
    monkeypatch.setattr(sidecar.sys, 'platform', 'win32')
    monkeypatch.setattr(sidecar.platform, 'machine', lambda: 'ARM64')
    with pytest.raises(ValueError, match='support Linux arm64/x64'):
        sidecar.platform_id()


def test_x64_archive_cannot_be_installed_on_arm64(package, monkeypatch):
    source, store, requirements, manifest = package
    manifest['platform'] = 'linux-x64'
    (source/'sidecar.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(sidecar.sys, 'platform', 'linux')
    monkeypatch.setattr(sidecar.platform, 'machine', lambda: 'aarch64')
    with pytest.raises(ValueError, match='platform does not match'):
        sidecar.materialize_local(source, store, requirements)
    preserved(store)

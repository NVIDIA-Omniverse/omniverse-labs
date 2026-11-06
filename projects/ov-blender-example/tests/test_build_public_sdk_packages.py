# SPDX-License-Identifier: Apache-2.0
"""Small developer packaging fixtures; never install or execute runtime code."""
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile
from types import SimpleNamespace

import pytest

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/build_public_sdk_packages.py'
spec=importlib.util.spec_from_file_location('public_sdk_packager_test',SCRIPT)
packager=importlib.util.module_from_spec(spec);spec.loader.exec_module(packager)
_fixtures=importlib.util.spec_from_file_location('runtime_license_fixtures',Path(__file__).with_name('test_runtime_licenses.py'))
license_fixtures=importlib.util.module_from_spec(_fixtures);_fixtures.loader.exec_module(license_fixtures)
ADAPTER_FILES=('public_sdk_backend.py','incremental_physics_backend.py','inline_physics_geometry.py',
               'renderer_values.py','physics_adapter_identity.py','runtime_identity.py',
               'public_sdk_worker.py','physics_adapter_identity.json')


@pytest.fixture
def source(tmp_path,monkeypatch):
    public=tmp_path/'public';addon=public/'addon';module=addon/'ovrtx_blender_example';module.mkdir(parents=True)
    (addon/'__init__.py').write_text('from .ovrtx_blender_example import register\n')
    (addon/'blender_manifest.toml').write_text('id = "ovrtx_blender_example"\nblender_version_min = "5.1.0"\n[permissions]\nnetwork = "Connect to a managed local ovrtx worker over loopback gRPC"\n')
    (module/'__init__.py').write_text('def register(): pass\n')
    (module/'__pycache__').mkdir();(module/'__pycache__/discard.pyc').write_bytes(b'cache')
    runtime=public/'runtime';runtime.mkdir()
    original = SCRIPT.parents[1] / 'runtime'
    # The physics adapter sources and their build-checked identity record.
    for name in ADAPTER_FILES:
        (runtime/name).write_bytes((original/name).read_bytes())
    (runtime/'requirements-public-sdk.txt').write_bytes((original/'requirements-public-sdk.txt').read_bytes())
    (runtime/'public-sdk-lock.json').write_bytes((original/'public-sdk-lock.json').read_bytes())
    for name in packager.ADDON_LICENSE_FILES:
        (public/name).write_bytes((SCRIPT.parents[1]/name).read_bytes())
    monkeypatch.setattr(packager,'PUBLIC',public)
    return public


@pytest.mark.parametrize('automatic',[False,True])
def test_addon_zip_layout_marker_and_checksum(source,tmp_path,automatic):
    output=tmp_path/'addon.zip';packager.build_addon(output,automatic)
    with zipfile.ZipFile(output) as archive:
        names=set(archive.namelist())
        assert {'__init__.py','blender_manifest.toml','ovrtx_blender_example/__init__.py',
                'runtime/public_sdk_worker.py','runtime/requirements-public-sdk.txt','public-sdk.json',
                'runtime/physics_adapter_identity.json'}<=names
        assert archive.read('runtime/physics_adapter_identity.json')==(source/'runtime/physics_adapter_identity.json').read_bytes()
        assert archive.read('LICENSE')==(source/'LICENSE').read_bytes() and b'Apache License' in archive.read('LICENSE')
        assert archive.read('THIRD_PARTY_NOTICES.md')==(source/'THIRD_PARTY_NOTICES.md').read_bytes()
        assert not any('__pycache__' in name or name.endswith('.pyc') for name in names)
        marker=json.loads(archive.read('public-sdk.json'))
        assert marker=={'schema':1,'backend':'public_sdk','install_mode':'automatic' if automatic else 'local_package'}
        assert b'5.1.0' in archive.read('blender_manifest.toml')
        import tomllib
        assert len(tomllib.loads(archive.read('blender_manifest.toml').decode())['permissions']['network']) <= 64
    assert output.with_suffix('.zip.sha256').read_text().startswith(hashlib.sha256(output.read_bytes()).hexdigest())
    with pytest.raises(ValueError,match='immutable'):packager.build_addon(output)


@pytest.mark.parametrize('change',['missing','stale_record','changed_adapter'])
def test_addon_build_rejects_missing_or_stale_physics_adapter_record(source,tmp_path,change):
    record=source/'runtime/physics_adapter_identity.json'
    if change=='missing':record.unlink()
    elif change=='stale_record':record.write_bytes(record.read_bytes().replace(b'"schema": 1',b'"schema": 2'))
    else:
        path=source/'runtime/incremental_physics_backend.py'
        path.write_bytes(path.read_bytes()+b'\nphysics_revision = 2\n')
    output=tmp_path/'addon.zip'
    with pytest.raises(ValueError,match='physics_adapter_identity.json is missing or stale'):packager.build_addon(output)
    assert not output.exists()


def test_unicode_archive_checksum_and_cp1252_redirected_output(source,tmp_path,monkeypatch):
    import io,sys
    buffer=io.BytesIO(); stream=io.TextIOWrapper(buffer,encoding='cp1252')
    output=tmp_path/'addon \u03a9.zip'
    with monkeypatch.context() as patch:
        patch.setattr(sys,'stdout',stream)
        packager.build_addon(output)
        stream.flush()
    checksum=output.with_suffix('.zip.sha256').read_text(encoding='utf-8')
    assert checksum==hashlib.sha256(output.read_bytes()).hexdigest()+'  '+output.name+'\n'
    assert json.loads(buffer.getvalue().decode('cp1252'))==str(output)


@pytest.mark.parametrize('filename,content',[('bad.dll',b'blob'),('bad.pyd',b'blob'),('bad.exe',b'blob'),
    ('libbad.so.1.2',b'blob'),('bad.dylib',b'blob'),('sdk.whl',b'blob'),
    ('hidden.data',b'\x7fELFnative'),('hidden.py',b'MZbinary')])
def test_addon_zip_rejects_native_payloads(source,tmp_path,filename,content):
    (source/'runtime'/filename).write_bytes(content)
    output=tmp_path/'addon.zip'
    with pytest.raises(ValueError,match='native SDK'):packager.build_addon(output)
    assert not output.exists()


def test_addon_source_symlinks_are_not_copied(source,tmp_path):
    target=source/'runtime/link.py'
    try:target.symlink_to(source/'runtime/public_sdk_worker.py')
    except OSError:pytest.skip('Host cannot create symlinks')
    with pytest.raises(ValueError,match='symlinks'):packager.build_addon(tmp_path/'addon.zip')


def test_addon_link_refusal_with_portable_filesystem_fixture(source, tmp_path, monkeypatch):
    original = Path.is_symlink
    target = source / 'runtime/public_sdk_worker.py'
    monkeypatch.setattr(Path, 'is_symlink', lambda path: path == target or original(path))
    with pytest.raises(ValueError, match='symlinks'):
        packager.build_addon(tmp_path / 'addon.zip')


def test_archived_site_path_uses_actual_spelling_when_sysconfig_differs(tmp_path):
    actual=tmp_path/'python/lib/site-packages';actual.mkdir(parents=True)
    assert packager.existing_relative(tmp_path/'python/Lib/site-packages',tmp_path)=='python/lib/site-packages'


def test_sidecar_package_roster_pins_and_relocated_layout_without_sdk_execution(source,tmp_path,monkeypatch):
    base=tmp_path/'standalone-python';(base/'bin').mkdir(parents=True)
    executable=base/'bin/python';executable.write_bytes(b'fake interpreter never executed')
    cache = base/'bin/cache/shader'
    cache.mkdir(parents=True)
    (cache/'generated.bin').write_bytes(b'host-specific renderer cache')
    sdk_file=tmp_path/'sdk-native.dll';sdk_file.write_bytes(b'fake native payload')
    # The base interpreter's pip pieces must not reach the shipped runtime.
    for name in ('bin/pip3','lib/ensurepip/__init__.py'):
        (base/name).parent.mkdir(parents=True,exist_ok=True);(base/name).write_bytes(b'pip')
    blender_license=license_fixtures.blender_license(tmp_path/'license.md')
    lock = packager.public_sdk_lock.load_lock(source / 'runtime/public-sdk-lock.json')
    (base / packager.public_sdk_lock.RECEIPT_NAME).write_text(json.dumps(
        {'identity': packager.public_sdk_lock.lock_identity(lock), 'packages': {name: {} for name in lock['versions']}}))
    executions=[]
    def query(python,code):
        if code==packager.runtime_licenses.PROBE_CODE:
            return license_fixtures.PROBE
        if str(python)==str(executable):
            return {'prefix':str(base),'executable':str(executable),'version':[3,13],'purelib':str(base/'lib/site-packages')}
        if str(python)=='sdk-python':
            if 'sys.prefix' in code:
                return str(base)
            return {name: {'version':version,'files':[[str(sdk_file),name+'/sdk-native.dll']]}
                    for name, version in lock['versions'].items()}
        target=Path(python).parent.parent
        return {'prefix':str(target),'purelib':str(target/'lib/site-packages')}
    monkeypatch.setattr(packager,'query_python',query)
    monkeypatch.setattr(packager.shutil,'disk_usage',lambda _:SimpleNamespace(free=32*1024**3))
    host_platform = packager.platform_id()
    monkeypatch.setattr(packager, 'platform_id', lambda: host_platform)
    monkeypatch.setattr(packager.subprocess,'run',lambda args,**kw:executions.append(args))
    output=tmp_path/'sidecar.zip';packager.build_sidecar(output,executable,'sdk-python',blender_license=blender_license)
    with zipfile.ZipFile(output) as archive:
        manifest=json.loads(archive.read('sidecar.json'))
        assert not any('pip' in name for name in archive.namelist())
        licensed=packager.runtime_licenses.validate(archive.namelist(),archive.read)
        assert {'Python','OpenSSL'}<={row['name'] for row in licensed['components']}
        assert all('licenses/'+path in manifest['files'] for row in licensed['components'] for path in row['license_files'])
        assert manifest['python']=='python/bin/python'
        assert manifest['requirements']==lock['versions']
        assert manifest['runtime_identity'] == packager.public_sdk_lock.lock_identity(lock)
        assert set(archive.namelist())==set(manifest['files'])|{'sidecar.json'}
        for name,digest in manifest['files'].items():assert hashlib.sha256(archive.read(name)).hexdigest()==digest
        assert archive.read('python/lib/site-packages/ovstream/sdk-native.dll')==sdk_file.read_bytes()
    assert len(executions)==2 and str(executions[0][2]).endswith('materialize_sdk.py')
    assert str(executions[1][1]).endswith('public_sdk_environment.py')

# SPDX-License-Identifier: Apache-2.0
"""Installed-package selection and main-thread setup wiring; no downloads/GPU."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'addon'))
import ovrtx_blender_example as addon
from ovrtx_blender_example import bundled_runtime as bundle
from ovrtx_blender_example import direct_sdk_client as client
from ovrtx_blender_example import preflight,public_sdk_install as install


@pytest.fixture
def package(tmp_path,monkeypatch):
    monkeypatch.delenv(client.PYTHON_ENV,raising=False)
    root=tmp_path/'extension';root.mkdir();(root/'runtime').mkdir()
    (root/'public-sdk.json').write_text(json.dumps({'schema':1,'backend':'public_sdk','install_mode':'local_package'}))
    (root/'runtime/public_sdk_worker.py').write_text('# worker fixture')
    (root/'runtime/requirements-public-sdk.txt').write_text('ovstream==0.4.5\n')
    monkeypatch.setattr(bundle,'public_sdk_package_root',lambda:root)
    monkeypatch.setattr(addon,'_PUBLIC_SDK_SETUP_ERROR','')
    monkeypatch.setattr(install,'status',lambda:{'state':'unconfigured','message':'Waiting for setup','progress':0.,'error':''})
    monkeypatch.setattr(install,'ready_python',lambda:None)
    return root


def test_marker_selects_sdk_while_pending_without_sil_defaults(package,monkeypatch):
    assert client.enabled()
    assert bundle.defaults().worker_command==''
    assert client.worker_script()==package/'runtime/public_sdk_worker.py'
    with pytest.raises(client.RenderClientError,match='Waiting for setup'):client._session('renderer')


def test_explicit_interpreter_override_still_works(package,monkeypatch,tmp_path):
    interpreter=tmp_path/'custom-python';interpreter.touch()
    monkeypatch.setenv(client.PYTHON_ENV,str(interpreter))
    assert bundle.public_sdk_python()==str(interpreter)
    assert str(interpreter) in bundle.defaults().worker_command


def test_source_without_marker_never_selects_a_legacy_backend(monkeypatch):
    monkeypatch.delenv(client.PYTHON_ENV,raising=False)
    monkeypatch.setattr(bundle,'public_sdk_package_root',lambda:None)
    assert client.enabled() and bundle.public_sdk_python() is None
    assert client.worker_script().name=='public_sdk_worker.py'


def test_nested_installed_marker_takes_priority_over_legacy_root_override(tmp_path,monkeypatch):
    root=tmp_path/'extension';package=root/'ovrtx_blender_example';package.mkdir(parents=True)
    (root/'public-sdk.json').write_text('{"schema":1,"backend":"public_sdk"}')
    monkeypatch.setattr(bundle,'__file__',str(package/'bundled_runtime.py'))
    monkeypatch.setenv('OV_BLENDER_EXAMPLE_BUNDLED_ADDON_ROOT',str(tmp_path/'legacy'))
    assert bundle.public_sdk_package_root()==root
    assert bundle.public_sdk_worker_script()==root/'runtime/public_sdk_worker.py'


def configure_context(package,monkeypatch,tmp_path,*,online=False):
    calls=[]
    timers=SimpleNamespace(is_registered=lambda _:False,register=lambda *a,**kw:calls.append(('timer',a,kw)))
    monkeypatch.setattr(addon,'bpy',SimpleNamespace(app=SimpleNamespace(online_access=online,timers=timers),
        utils=SimpleNamespace(user_resource=lambda kind:str(tmp_path/'config'))))
    monkeypatch.setattr(addon,'_extension_storage_root',lambda **_:tmp_path/'user-storage')
    interpreter=tmp_path/'bundled-python';interpreter.touch()
    monkeypatch.setattr(addon,'_bundled_python_path',lambda:interpreter)
    monkeypatch.setattr(install,'configure',lambda *args:calls.append(('configure',args)))
    monkeypatch.setattr(install,'start',lambda **kw:calls.append(('start',kw)) or True)
    return calls


@pytest.mark.parametrize('explicit_profile', [False, True])
def test_windows_runtime_discovery_uses_resolved_blender_profile(monkeypatch, tmp_path, explicit_profile):
    from ovrtx_blender_example import public_sdk_sidecar as sidecar
    profile = tmp_path/'Blender Foundation'/'Blender'/'5.3'
    config = profile/'config'
    if explicit_profile:
        monkeypatch.setenv('BLENDER_USER_CONFIG', str(config))
    else:
        monkeypatch.delenv('BLENDER_USER_CONFIG', raising=False)
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path/'local-appdata'))
    monkeypatch.setattr(addon, 'bpy', SimpleNamespace(utils=SimpleNamespace(user_resource=lambda kind:str(config))))
    monkeypatch.setattr(addon, '_extension_storage_root', lambda **_:profile/'extensions/.user/user_default/ovrtx_blender_example')
    storage_root = sidecar.runtime_storage_root
    monkeypatch.setattr(sidecar, 'runtime_storage_root', lambda *args, **kwargs:storage_root(*args, **kwargs, windows=True))
    assert addon._public_sdk_storage_root() == profile/'ovrtx-sdk'


def test_missing_blender_config_directory_is_actionable(monkeypatch):
    monkeypatch.setattr(addon, 'bpy', SimpleNamespace(utils=SimpleNamespace(user_resource=lambda kind:'')))
    with pytest.raises(RuntimeError, match='profile configuration directory'):
        addon._public_sdk_storage_root()


def test_local_registration_configures_but_never_installs_or_starts_worker(package,monkeypatch,tmp_path):
    calls=configure_context(package,monkeypatch,tmp_path)
    assert not addon._start_public_sdk_setup()
    assert [x[0] for x in calls]==['configure','timer']
    assert calls[0][1][0]==addon._public_sdk_storage_root()


def test_local_install_forwards_chosen_package_without_online_permission(package,monkeypatch,tmp_path):
    calls=configure_context(package,monkeypatch,tmp_path)
    assert addon._start_public_sdk_setup(source='/chosen/runtime.zip')
    assert ('start',{'online_access':False,'source':'/chosen/runtime.zip'}) in calls


@pytest.mark.parametrize('online',[False,True])
def test_automatic_package_delegates_online_permission_without_changing_it(package,monkeypatch,tmp_path,online):
    (package/'public-sdk.json').write_text(json.dumps({'schema':1,'backend':'public_sdk','install_mode':'automatic'}))
    calls=configure_context(package,monkeypatch,tmp_path,online=online)
    assert addon._start_public_sdk_setup()
    assert ('start',{'online_access':online,'source':''}) in calls


def test_malformed_marker_stays_selected_without_download(package,monkeypatch,tmp_path):
    (package/'public-sdk.json').write_text('{}')
    calls=configure_context(package,monkeypatch,tmp_path)
    assert not addon._start_public_sdk_setup()
    assert client.enabled() and addon._public_sdk_status()['state']=='failed'
    assert not any(call[0]=='start' for call in calls)


def test_public_preflight_does_not_require_running_sil_services(package,monkeypatch,tmp_path):
    interpreter=tmp_path/'python';interpreter.touch()
    monkeypatch.setattr(install,'ready_python',lambda:str(interpreter))
    monkeypatch.setattr(install,'status',lambda:{'state':'ready','message':'Ready','progress':1.,'error':''})
    monkeypatch.setattr(preflight,'_driver_version_check',lambda:preflight.PreflightCheck('driver','Driver',True,'ok'))
    result=addon.preflight_preferences(SimpleNamespace())
    assert result['status']=='pass'
    assert 'runtime_services' not in {x['key'] for x in result['checks']}


def test_bundled_interpreter_comes_from_blender_not_shell(package,monkeypatch,tmp_path):
    binary=tmp_path/'blender';binary.touch()
    directory=tmp_path/'5.2/python/bin';directory.mkdir(parents=True)
    interpreter=directory/('python.exe' if sys.platform=='win32' else f'python{sys.version_info.major}.{sys.version_info.minor}')
    interpreter.touch()
    monkeypatch.setattr(addon,'bpy',SimpleNamespace(app=SimpleNamespace(binary_path=str(binary),version=(5,2,0))))
    assert addon._bundled_python_path()==interpreter


def test_local_operator_skips_online_dialog_and_uses_existing_source(monkeypatch,tmp_path):
    from test_addon_preflight import _load_addon_with_fake_bpy
    module=_load_addon_with_fake_bpy(monkeypatch)
    monkeypatch.setattr(module.bundled_runtime,'public_sdk_package_root',lambda:tmp_path)
    monkeypatch.setattr(module,'_public_sdk_local_package_mode',lambda:True)
    monkeypatch.setattr(module,'_public_sdk_status',lambda:{'state':'unconfigured'})
    monkeypatch.setattr(module,'runtime_bundle_status',lambda:{'state':'unconfigured'})
    monkeypatch.setattr(module,'get_addon_preferences',lambda *_:SimpleNamespace(runtime_source='/chosen/runtime.zip'))
    calls=[];monkeypatch.setattr(module,'_start_public_sdk_setup',lambda context,**kw:calls.append(kw) or True)
    module.bpy.app.online_access=False
    context=SimpleNamespace(window_manager=SimpleNamespace(invoke_confirm=lambda *_a,**_k:pytest.fail('Local install must not request online access')))
    operator=module.OvrtxExampleInstallRuntime()
    assert operator.invoke(context,None)=={'FINISHED'}
    assert calls==[{'source':'/chosen/runtime.zip'}]

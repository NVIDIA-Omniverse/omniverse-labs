# SPDX-License-Identifier: Apache-2.0
"""Unsupported requests fail before renderer work or partial publication."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def backend_module():
    spec = importlib.util.spec_from_file_location('mvp_backend',
        Path(__file__).parents[1] / 'runtime/public_sdk_backend.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('params', [
    {'sensor_paths': ['/A', '/B']}, {'sensor_paths': ['/A', '/A']},
    {'render_var': 'Depth'}, {'semantic_outputs': True},
    {'render_vars': ['LdrColor', 'Depth']}, {'additional_aovs': ['Depth']},
    {'aovs': ['Depth']},
])
@pytest.mark.parametrize('route', ['load', 'update_scene', 'render', 'begin_render', '_publish_products'])
def test_unsupported_outputs_fail_before_native_work(backend_module, params, route):
    backend = object.__new__(backend_module.RendererBackend)
    backend.sensors = ['/A', '/B'] if 'sensor_paths' in params else ['/A']
    # Deliberately no renderer, publisher, Warp or native state: selection must
    # fail before any of those are used, including the final publication guard.
    with pytest.raises(ValueError, match='future work'):
        if route in ('load','update_scene'):
            backend.dispatch(route, params)
        elif route == '_publish_products':
            backend._publish_products({}, params, params.get('sensor_paths', backend.sensors), 1)
        else:
            getattr(backend, route)(params)


@pytest.mark.parametrize('color', ['LdrColor', 'HdrColor'])
def test_supported_single_product_boundary(backend_module, color):
    backend_module._validate_output_selection(['/A'], {'render_var': color, 'semantic_outputs': False})


@pytest.mark.parametrize('route', ['render_result', 'begin_render_sample'])
@pytest.mark.parametrize('sensors,color,semantic', [
    (('/A', '/B'), 'LdrColor', ''), (('/A',), 'Depth', ''),
    (('/A',), 'HdrColor', 'capture'),
])
def test_client_rejects_before_runtime_submission(route, sensors, color, semantic):
    from ovrtx_blender_example import direct_sdk_client as client
    cls = next(value for value in vars(client).values()
               if isinstance(value, type) and value.__module__ == client.__name__
               and hasattr(value, '_validate_mvp_outputs'))
    adapter = object.__new__(cls)
    adapter.request = SimpleNamespace(semantic_output_path=semantic)
    adapter.states = {'scene': SimpleNamespace(failure=None, recovery_pending=False,
        spec=SimpleNamespace(sensor_paths=('/A', '/B')))}
    args = dict(selected_sensor_paths=sensors, render_var=color)
    if route == 'render_result':
        args['additional_samples'] = 1
    with pytest.raises(client.RenderClientError, match='future work'):
        getattr(adapter, route)('scene', **args)


@pytest.mark.parametrize('sensors,color,semantic', [
    (('/A','/B'), 'LdrColor', ''), (('/A',), 'Depth', ''),
    (('/A',), 'HdrColor', 'semantic'),
])
def test_invalid_session_rejected_before_usd_access_or_runtime_load(sensors,color,semantic):
    from ovrtx_blender_example.direct_sdk_client import PublicSdkRenderClient, RenderClientError
    adapter=object.__new__(PublicSdkRenderClient)
    adapter.request=SimpleNamespace(semantic_output_path=semantic)
    adapter.states={}
    with pytest.raises(RenderClientError,match='future work'):
        adapter.start_session(SimpleNamespace(sensor_paths=sensors,render_var=color))


@pytest.mark.parametrize('route', ['capture_owner_for_scene','start_sequence'])
def test_candidate_timeline_unavailable_before_accessing_blender(route):
    from ovrtx_blender_example import render_sequence
    with pytest.raises(ValueError,match='not admitted'):
        getattr(render_sequence,route)(None)


@pytest.mark.parametrize('selected,declared', [
    (('/A','/B'),('/A','/B')), (('/A',),('/A','/B')), (('/B',),('/A',)),
])
def test_request_never_silently_selects_first_product(selected,declared):
    from ovrtx_blender_example.render_requests import RenderRequest
    with pytest.raises(ValueError,match='Multiple|not declared'):
        RenderRequest(sensor_paths=declared,selected_sensor_paths=selected).render_product_path


@pytest.mark.parametrize('enabled,white_point', [(True,10.0),(False,3.0)])
def test_candidate_native_exposure_rejected_before_stage_read(enabled,white_point):
    from ovrtx_blender_example.direct_sdk_client import PublicSdkRenderClient, RenderClientError
    from ovrtx_blender_example.native_exposure import NativeExposure
    adapter=object.__new__(PublicSdkRenderClient)
    adapter.request=SimpleNamespace(native_exposure=NativeExposure(enabled,white_point))
    with pytest.raises(RenderClientError,match='native exposure controls are not admitted'):
        adapter._load_params(SimpleNamespace(sensor_paths=('/A',),render_var='HdrColor'))


def test_candidate_capture_operators_do_not_register(monkeypatch):
    from ovrtx_blender_example import hdr_output,render_sequence
    registrations=[]
    blender=SimpleNamespace(utils=SimpleNamespace(register_class=registrations.append))
    for module in (hdr_output,render_sequence):
        monkeypatch.setattr(module,'bpy',blender)
        module.register()
    assert registrations==[]


@pytest.mark.parametrize('fields', [
    {'dlss_quality':'quality'},
    {'native_exposure':SimpleNamespace(enabled=True,white_point_scale=10.0)},
    {'semantic_output_path':'capture'},
    {'sensor_paths':('/A','/B')},
    {'selected_sensor_paths':('/A','/B')},
    {'sensor_paths':('/A',),'selected_sensor_paths':('/B',)},
])
def test_cold_and_live_request_rejected_without_work_or_identity_change(monkeypatch,fields):
    from ovrtx_blender_example import direct_sdk_client as client
    calls=[]
    monkeypatch.setattr(client,'_session',lambda kind:calls.append(kind))
    request=SimpleNamespace(**fields)
    with pytest.raises(client.RenderClientError):
        client.PublicSdkRenderClient(request)
    assert not calls
    adapter=object.__new__(client.PublicSdkRenderClient)
    accepted=SimpleNamespace(dlss_quality='INHERIT')
    adapter.request=accepted
    with pytest.raises(client.RenderClientError):
        adapter.configure_session(request)
    assert adapter.request is accepted

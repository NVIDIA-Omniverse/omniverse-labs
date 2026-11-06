"""Camera transitions preserve population, while incompatible changes refresh it."""
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from pxr import Gf, Usd, UsdGeom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'addon'))
sys.path.insert(0, str(ROOT / 'runtime'))
from ovrtx_blender_example.camera_retarget import camera_values
from ovrtx_blender_example import ovrtx_session
from ovrtx_blender_example.render_requests import RenderRequest
from public_sdk_backend import RendererBackend
from renderer_values import prepare_value, tensor_value
import numpy as np


def requests(tmp_path):
    path = tmp_path / 'cameras.usda'
    stage = Usd.Stage.CreateNew(str(path))
    parent = UsdGeom.Xform.Define(stage, '/World/Rig')
    parent.AddTranslateOp().Set(Gf.Vec3d(3, 4, 5))
    for name in ('A', 'B'):
        cam = UsdGeom.Camera.Define(stage, '/World/Rig/' + name)
        cam.AddTranslateOp().Set(Gf.Vec3d(1, 0, 0))
    stage.GetRootLayer().Save()
    a = RenderRequest(input_usd_path=str(path), camera_prim_path='/World/Rig/A',
                      sensor_paths=('/Render/Product',), selected_sensor_paths=('/Render/Product',))
    return a, replace(a, camera_prim_path='/World/Rig/B')


def rows(a, b):
    return camera_values(a, b, ovrtx_session.build_spec(a), ovrtx_session.build_spec(b))


def test_a_b_a_restores_world_pose_and_product_target(tmp_path):
    a, b = requests(tmp_path)
    for old, new in ((a, b), (b, a)):
        result = rows(old, new)
        assert result is not None
        indexed = {(r['prim_path'], r['attribute_name']): r['value'] for r in result}
        assert indexed[('/Render/Product', 'camera')] == [new.camera_prim_path]
        for target in (a.camera_prim_path, b.camera_prim_path):
            assert indexed[(target, 'omni:xform')][3][:3] == [4, 4, 5]
            assert indexed[(target, 'omni:resetXformStack')] is True


def test_explicit_pose_only_applies_to_selected_camera(tmp_path):
    a, b = requests(tmp_path)
    matrix = np.eye(4); matrix[3, :3] = [9, 8, 7]
    result = rows(a, replace(b, camera_matrix=tuple(map(tuple, matrix))))
    indexed = {(r['prim_path'], r['attribute_name']): r['value'] for r in result}
    assert indexed[(b.camera_prim_path, 'omni:xform')][3][:3] == [9, 8, 7]
    assert indexed[(a.camera_prim_path, 'omni:xform')][3][:3] == [4, 4, 5]


@pytest.mark.parametrize('change', [
    {'camera_prim_path': '/World/Missing'},
    {'input_usd_path': 'different.usda'},
    {'sensor_paths': ('/Render/Other',)},
    {'width': 123},
])
def test_incompatible_changes_require_population_refresh(tmp_path, change):
    a, b = requests(tmp_path)
    assert rows(a, replace(b, **change)) is None


def worker():
    backend = RendererBackend.__new__(RendererBackend)
    backend.pending = {}; backend.sensors = ['/Render/Product']; backend.native_stage_id = 7
    backend._reap_cancelled = lambda: None
    calls = []
    backend.update = lambda values, **kw: calls.append((values, kw)) or {'updated_count': len(values)}
    return backend, calls


def transaction():
    return [{'prim_path': '/Render/Product', 'attribute_name': 'camera', 'value_type': 'Relationship', 'value': ['/World/B']},
            {'prim_path': '/World/B', 'attribute_name': 'omni:resetXformStack', 'value_type': 'Bool', 'value': True}]


def test_worker_requires_native_camera_and_product_types_before_mutation():
    backend, calls = worker()
    result = backend.retarget_camera({'values': transaction()})
    assert result['camera_retargeted'] is True
    assert calls[0][1] == {'required_types': {'/Render/Product': 'RenderProduct', '/World/B': 'Camera'}}


@pytest.mark.parametrize('case', ['pending', 'missing-camera', 'non-camera-attribute', 'missing-product', 'unbounded'])
def test_worker_rejects_invalid_transactions_before_mutation(case):
    backend, calls = worker(); values = transaction()
    if case == 'pending': backend.pending = {1: object()}
    if case == 'missing-camera': values.pop()
    if case == 'non-camera-attribute': values[1]['attribute_name'] = 'inputs:base_color'
    if case == 'missing-product': values.pop(0)
    if case == 'unbounded': values *= 100
    with pytest.raises(ValueError): backend.retarget_camera({'values': values})
    assert calls == []


def test_relationship_uses_native_path_id_and_strict_schema():
    row = transaction()[0]
    ovs = SimpleNamespace(AttributeSemantic=SimpleNamespace(RELATIONSHIP_PATH_ID=4))
    paths = SimpleNamespace(intern_path=lambda path: 123)
    data, array, role = tensor_value(prepare_value(row), ovs, paths, np.uint64,
                                    existing_semantic=4, existing_is_array=True, existing_width=1)
    assert data.tolist() == [123] and array and role == 4
    for kwargs in ({'existing_semantic': 2}, {'existing_is_array': False}, {'existing_width': 2}):
        with pytest.raises(ValueError): tensor_value(prepare_value(row), ovs, paths, np.uint64, **kwargs)


@pytest.mark.parametrize('target', [['/'], ['relative']])
def test_relationship_rejects_invalid_target(target):
    row = transaction()[0]; row['value'] = target
    with pytest.raises(ValueError): prepare_value(row)


@pytest.mark.parametrize('target', [[], ['/A', '/B']])
def test_camera_retarget_retains_single_sensor_contract_for_general_relationship_values(target):
    backend, calls = worker()
    rows = transaction(); rows[0]['value'] = target
    with pytest.raises(ValueError, match='exactly one camera'):
        backend.retarget_camera({'values': rows})
    assert calls == []


def test_projection_kind_offsets_and_native_exposure_survive_retarget(tmp_path):
    from ovrtx_blender_example.render_requests import CameraProjectionState
    from ovrtx_blender_example.native_exposure import NativeExposure, ENABLED_ATTRIBUTE, TARGET_ATTRIBUTE
    a, b = requests(tmp_path)
    projection = CameraProjectionState(source='test', focal_length=50,
        horizontal_aperture=10, vertical_aperture=8, projection='orthographic',
        horizontal_aperture_offset=2, vertical_aperture_offset=3, clipping_range=(.1, 100))
    b = replace(b, camera_projection=projection, native_exposure=NativeExposure(True, 4))
    result = rows(a, b)
    indexed = {(r['prim_path'], r['attribute_name']): r['value'] for r in result}
    assert indexed[(b.camera_prim_path, 'projection')] == 'orthographic'
    assert indexed[(a.camera_prim_path, 'projection')] == 'perspective'
    assert indexed[(b.camera_prim_path, 'horizontalApertureOffset')] == 2
    assert indexed[(b.camera_prim_path, 'verticalApertureOffset')] == 3
    for path in (b.camera_prim_path, '/Render/Product'):
        assert indexed[(path, ENABLED_ATTRIBUTE)] is True
        assert indexed[(path, TARGET_ATTRIBUTE)] == 4


def test_output_resize_requires_existing_live_route(tmp_path):
    a, b = requests(tmp_path)
    a = replace(a, output_size_value_route=True)
    b = replace(b, output_size_value_route=True, width=512, height=256)
    assert rows(a, b) is not None


def test_changed_generation_source_skips_second_composition(tmp_path, monkeypatch):
    a, b = requests(tmp_path)
    b = replace(b, input_usd_path=str(tmp_path / 'new-generation.usda'))
    previous, desired = ovrtx_session.build_spec(a), ovrtx_session.build_spec(b)
    def unexpected_build(_request):
        raise AssertionError('Changed population source cannot use camera retarget')
    monkeypatch.setattr(ovrtx_session, 'build_spec', unexpected_build)
    assert camera_values(a, b, previous, desired) is None

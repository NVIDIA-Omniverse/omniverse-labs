from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
from pxr import Usd, UsdGeom

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'addon'))
from ovrtx_blender_example import direct_camera_updates as direct, ovrtx_session, ovrtx_session_controller
from ovrtx_blender_example.render_requests import RenderRequest, CameraProjectionState
from ovrtx_blender_example.scene_generation import BlenderId, BlenderPrimPath

MATRIX = tuple(tuple(float(i == j) for j in range(4)) for i in range(4))


def requests(tmp_path):
    path = tmp_path / 'camera.usda'
    stage = Usd.Stage.CreateNew(str(path))
    for name in ('A', 'B'): UsdGeom.Camera.Define(stage, '/World/' + name)
    stage.GetRootLayer().Save()
    request = RenderRequest(input_usd_path=str(path), current_scene_generation=True,
        camera_prim_path='/World/A', scene_camera_matrix=MATRIX,
        camera_projection=CameraProjectionState('test', 50, 36, 24, clipping_range=(.1, 100)))
    spec = ovrtx_session.build_spec(request)
    catalog = direct.request_catalog(spec.ovrtx_scene_composition.composed_scene_path)
    return request, spec, catalog


def test_camera_switch_resize_quality_packet_matches_request_without_composition(tmp_path):
    request, spec, catalog = requests(tmp_path)
    projection = replace(request.camera_projection, focal_length=35, projection='orthographic',
                         horizontal_aperture_offset=2, vertical_aperture_offset=-1, f_stop=2.8, focus_distance=4)
    edited = replace(request, camera_prim_path='/World/B', width=512, height=256,
                     camera_projection=projection, rtpt_quality={'rtpt_max_bounces': 4})
    rows = direct.request_rows(request, edited, spec, catalog)
    indexed = {(r['prim_path'], r['attribute_name']): r['value'] for r in rows}
    assert indexed[('/World/B', 'projection')] == 'orthographic'
    assert indexed[('/World/B', 'focalLength')] == 35
    assert indexed[('/World/B', 'horizontalApertureOffset')] == 2
    assert indexed[('/World/B', 'fStop')] == 2.8
    assert indexed[(spec.sensor_paths[0], 'camera')] == ['/World/B']
    assert indexed[(spec.sensor_paths[0], 'resolution')] == (512, 256)
    assert indexed[(spec.sensor_paths[0], 'omni:rtx:rtpt:maxBounces')] == 6


def test_native_only_camera_pose_does_not_author_usd_transform_or_rekey_composition(tmp_path):
    request, spec, catalog = requests(tmp_path)
    native = replace(request, scene_camera_matrix=None, native_scene_camera_matrix=MATRIX)
    bootstrap = ovrtx_session.build_spec(native)
    moved = tuple((row if index != 3 else (7., 8., 9., 1.)) for index, row in enumerate(MATRIX))
    edited = replace(native, native_scene_camera_matrix=moved)
    assert ovrtx_session.build_spec(edited).ovrtx_scene_composition.digest == bootstrap.ovrtx_scene_composition.digest
    rows = direct.request_rows(native, edited, bootstrap, direct.request_catalog(bootstrap.ovrtx_scene_composition.composed_scene_path))
    assert next(r['value'] for r in rows if r['attribute_name'] == 'omni:xform') == moved
    for layer in bootstrap.ovrtx_scene_composition.presentation_layers:
        if layer.get('source') == 'viewport_camera_projection':
            assert 'xformOp:transform' not in Path(layer['path']).read_text()


@pytest.mark.parametrize('edit', [dict(input_usd_path='different.usda'), dict(camera_prim_path='/World/Missing'),
    dict(sensor_paths=('/Render/New',)), dict(semantic_output_path='new.semantic'), dict(dlss_enabled=False),
    dict(color_presentation={'render_var': 'HdrColor'})])
def test_unpopulated_or_resource_identity_changes_keep_population_route(tmp_path, edit):
    request, spec, catalog = requests(tmp_path)
    assert direct.request_rows(request, replace(request, **edit), spec, catalog) is None


def test_inactive_camera_compiler_uses_cached_owner_without_scene_scan():
    data = SimpleNamespace(type='PERSP', lens=35, sensor_width=36, sensor_height=24,
        sensor_fit='AUTO', clip_start=.1, clip_end=100, shift_x=.2, shift_y=0, dof=None)
    obj = SimpleNamespace(name_full='Inactive', type='CAMERA', data=data, matrix_world=MATRIX)
    obj.evaluated_get = lambda graph: obj
    key = BlenderId('OBJECT', 42)
    mapping = BlenderPrimPath('Inactive', 'CAMERA', '/World/Inactive', '/World/Inactive/Camera')
    class NoScan:
        def __iter__(self): pytest.fail('Plan scanned the whole scene')
    scene = SimpleNamespace(objects=NoScan(), render=SimpleNamespace(resolution_x=400, resolution_y=200, resolution_percentage=100))
    result = direct.plan(scene, None, {key}, {key: (mapping, obj)}, depsgraph=object())
    assert not result.fallback
    values = {v['attribute_name']: v['value'] for v in result.values}
    assert values['focalLength'] == 35
    assert values['horizontalApertureOffset'] == 7.2


def test_controller_fast_request_skips_build_and_commits_after_native_ack(tmp_path, monkeypatch):
    request, spec, catalog = requests(tmp_path)
    calls = []
    class Client:
        supports_direct_render_parameters = True
        def configure_session(self, request): pass
        def update_attribute_values(self, identity, values):
            calls.append(('write', len(values)))
            return SimpleNamespace(timings={})
        def resize_output(self, identity, width, height): calls.append(('resize', width, height))
        def accept_render_parameters(self, identity, desired): calls.append(('commit', desired.camera_prim_path))
    controller = ovrtx_session_controller.OvrtxSessionController()
    controller._client = Client(); controller._simulation_id = 'test'; controller._spec = spec
    controller._camera_request = request
    controller._direct_camera_catalog = catalog
    controller._direct_camera_catalog_source = spec.ovrtx_scene_composition.composed_scene_path
    monkeypatch.setattr(ovrtx_session, 'build_spec', lambda *_: pytest.fail('Direct request composed USD'))
    edited = replace(request, camera_prim_path='/World/B', width=512, height=256)
    result = controller.ensure(edited)
    assert result.value_state_preserved and not result.session_started
    assert calls[0][0] == 'write' and calls[-1] == ('commit', '/World/B')
    assert controller._spec.width == 512


def _direct_controller(request, spec, catalog):
    class Client:
        supports_direct_render_parameters = True
        fail = ''
        writes = 0
        restored = None
        def configure_session(self, request): pass
        def update_attribute_values(self, identity, values):
            self.writes += 1
            if self.fail == 'update':
                raise RuntimeError('native ACK unavailable')
            return SimpleNamespace(timings={})
        def resize_output(self, identity, width, height):
            if self.fail == 'resize':
                raise RuntimeError('native ACK unavailable')
        def accept_render_parameters(self, identity, desired):
            if self.fail == 'accept':
                raise RuntimeError('native ACK unavailable')
        def retarget_camera(self, identity, desired, rows):
            if self.fail == 'retarget':
                raise RuntimeError('native restore ACK unavailable')
            self.restored = rows
            return {}
    controller = ovrtx_session_controller.OvrtxSessionController()
    controller._client = Client()
    controller._simulation_id = 'test'
    controller._spec = spec
    controller._runtime_binding = (request.worker_command, request.native_client_module)
    controller._camera_request = request
    controller._direct_camera_catalog = catalog
    controller._direct_camera_catalog_source = spec.ovrtx_scene_composition.composed_scene_path
    return controller


def test_acknowledged_orthographic_request_stops_resync_and_preserves_bootstrap(tmp_path, monkeypatch):
    request, spec, catalog = requests(tmp_path)
    controller = _direct_controller(request, spec, catalog)
    ortho = replace(request, camera_projection=replace(request.camera_projection,
        projection='orthographic', horizontal_aperture=201.6, vertical_aperture=114.528))
    assert controller.would_replace(ortho) == 'scene_composition_changed'
    controller.ensure(ortho)
    assert controller._camera_request == ortho
    assert controller._spec.ovrtx_scene_composition == spec.ovrtx_scene_composition
    with monkeypatch.context() as frozen:
        frozen.setattr(ovrtx_session, 'build_spec', lambda *_: pytest.fail('Accepted request rebuilt USD'))
        for _ in range(4):
            assert controller.would_replace(replace(ortho)) == ''
    assert controller._client.writes == 1
    # Returning to the bootstrap perspective must still reach the native path:
    # the loaded USD alone is no longer the accepted live projection identity.
    assert controller.would_replace(request) == 'scene_composition_changed'
    controller.ensure(request)
    assert controller.would_replace(request) == ''
    assert controller._client.writes == 2


@pytest.mark.parametrize('edit', ['camera', 'quality', 'source', 'binding', 'dlss', 'output'])
def test_changed_direct_request_still_invalidates_after_orthographic_ack(tmp_path, edit):
    request, spec, catalog = requests(tmp_path)
    controller = _direct_controller(request, spec, catalog)
    ortho = replace(request, camera_projection=replace(request.camera_projection, projection='orthographic'))
    controller.ensure(ortho)
    (tmp_path/'new-generation.usda').write_bytes(Path(request.input_usd_path).read_bytes())
    changed = {'camera': replace(ortho, camera_prim_path='/World/B'),
               'quality': replace(ortho, rtpt_quality={'rtpt_max_bounces': 4}),
               'source': replace(ortho, input_usd_path=str(tmp_path/'new-generation.usda')),
               'binding': replace(ortho, worker_command='other-worker'),
               'output': replace(ortho, width=512),
               'dlss': replace(ortho, dlss_enabled=not ortho.dlss_enabled)}[edit]
    assert controller.would_replace(changed)
    assert controller._camera_request == ortho
    assert controller._client.writes == 1


@pytest.mark.parametrize('failure', ['update', 'resize', 'accept'])
def test_unacknowledged_orthographic_request_cannot_clear_resync(tmp_path, failure):
    request, spec, catalog = requests(tmp_path)
    controller = _direct_controller(request, spec, catalog)
    ortho = replace(request, width=512,
                    camera_projection=replace(request.camera_projection, projection='orthographic'))
    controller._client.fail = failure
    with pytest.raises(RuntimeError, match='ACK unavailable'):
        controller.ensure(ortho)
    assert controller._camera_request == request
    assert controller.would_replace(ortho)


def test_failed_followup_keeps_last_direct_ack_identity(tmp_path):
    request, spec, catalog = requests(tmp_path)
    controller = _direct_controller(request, spec, catalog)
    ortho = replace(request, camera_projection=replace(request.camera_projection, projection='orthographic'))
    controller.ensure(ortho)
    quality = replace(ortho, rtpt_quality={'rtpt_max_bounces': 4})
    controller._client.fail = 'accept'
    with pytest.raises(RuntimeError, match='ACK unavailable'):
        controller.ensure(quality)
    assert controller._camera_request == ortho
    assert controller.would_replace(quality) == 'scene_composition_changed'
    controller._client.fail = ''
    controller.ensure(quality)
    assert controller.would_replace(quality) == ''


@pytest.mark.parametrize('fail_restore', [False, True])
@pytest.mark.parametrize('authored_bootstrap', [False, True])
def test_orthographic_override_to_authored_camera_restores_native_schema(tmp_path, fail_restore, authored_bootstrap):
    request, spec, catalog = requests(tmp_path)
    authored = replace(request, camera_matrix=None, camera_projection=None, scene_camera_matrix=None)
    initial = authored if authored_bootstrap else request
    spec = ovrtx_session.build_spec(initial)
    catalog = direct.request_catalog(spec.ovrtx_scene_composition.composed_scene_path)
    controller = _direct_controller(initial, spec, catalog)
    ortho = replace(request, camera_matrix=MATRIX,
        camera_projection=replace(request.camera_projection, projection='orthographic',
                                  horizontal_aperture=201.6, vertical_aperture=114.528))
    controller.ensure(ortho)
    assert controller.would_replace(authored)
    if fail_restore:
        controller._client.fail = 'retarget'
        with pytest.raises(RuntimeError, match='restore ACK unavailable'):
            controller.ensure(authored)
        assert controller._camera_request == ortho
        assert controller._spec.ovrtx_scene_composition == spec.ovrtx_scene_composition
        assert controller.would_replace(authored)
        controller._client.fail = ''
    result = controller.ensure(authored)
    assert result.value_state_preserved and not result.session_started
    assert controller._client.restored is not None
    rows = {(r['prim_path'], r['attribute_name']): r['value'] for r in controller._client.restored}
    stage = Usd.Stage.Open(request.input_usd_path)
    camera = stage.GetPrimAtPath(request.camera_prim_path)
    for name in ('projection', 'horizontalAperture', 'verticalAperture', 'focalLength'):
        assert rows[(request.camera_prim_path, name)] == camera.GetAttribute(name).Get()
    assert rows[(request.camera_prim_path, 'omni:xform')] == [list(row) for row in MATRIX]
    assert rows[(request.camera_prim_path, 'omni:resetXformStack')] is True
    assert controller._camera_request == authored
    assert controller.would_replace(authored) == ''

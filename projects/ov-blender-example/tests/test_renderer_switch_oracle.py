"""Mutated genuine-shape records must fail for the defect they represent."""
import copy
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('switch_oracle', Path(__file__).parents[1]/'scripts/renderer_switch_oracle.py')
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


def good():
    frame = {'status': 'frame', 'published_ns': 30, 'render_started_ns': 20,
             'snapshot_matches': True, 'projection_clip': [.1, 250],
             'samples': 32,
             'width': 64, 'height': 64, 'controller_id': 1, 'simulation_id': 'sim',
             'pixels': {'tiles': [.3]*64}}
    names = ['initial', 'clip-visible', 'clip-hidden', 'clip-restored', 'two-panes', 'pane-closed']
    names += [f'{action}-{i}' for i in range(1, 7) for action in ('return', 'navigate')]
    steps = [{'name': name, 'boundary_ns': 10, 'panes': 2 if name == 'two-panes' else 1,
              'expected_clip': [[.1, 250]],
              'frames': [copy.deepcopy(frame) for _ in range(2 if name == 'two-panes' else 1)]} for name in names]
    steps[2]['frames'][0]['pixels']['tiles'] = [.05]*64
    for index, step in enumerate(steps):
        if step['name'] in {f'{action}-{i}' for action in ('return', 'navigate') for i in range(3, 7)}:
            step['panes'] = 2
            step['frames'].append(copy.deepcopy(step['frames'][0]))
        mode = 'PERSP' if step['name'].startswith('navigate-') and int(step['name'].split('-')[1]) % 2 == 0 else 'CAMERA'
        step['view_modes'] = [mode]*step['panes']
        for pane, item in enumerate(step['frames']):
            item['engine_id'] = pane+1
            item['output_time_ns'] = index+1
            item['session_revision'] = 1
    return {'schema_version': 1, 'steps': steps,
            'switches': [{'alternative': e, 'active_work': True} for e in ('BLENDER_EEVEE', 'BLENDER_WORKBENCH') for _ in range(3)],
            'workers': [123], 'worker_identities': [{'pid': 123, 'sha256': 'a'*64}],
            'worker_images_match_installed': True, 'loaded_native_clients': [{'sha256': 'b'*64}],
            'teardown': {'ordinary_exit': True, 'threads_stopped': True, 'worker_graceful': True, 'orphans': []}}


def test_complete_record_passes():
    assert oracle.validate(good()) == []


def test_original_builder_error_survives_later_aborted_slot():
    log = '[ovrtx] loading\n[ovrtx] RenderClientError: clippingRange.type unsupported\n' \
          '[ovrtx] RenderClientError: ABORTED already written\n'
    assert oracle.first_runtime_error(log) == 'RenderClientError: clippingRange.type unsupported'


@pytest.mark.parametrize('field,value', [('published_ns', 10), ('render_started_ns', 10),
                                       ('status', 'resyncing'), ('width', 0)])
def test_stale_or_unpresentable_frames_rejected(field, value):
    record = good()
    record['steps'][0]['frames'][0][field] = value
    assert 'stale_frame' in oracle.validate(record)


@pytest.mark.parametrize('field,value,error', [('workers', [123, 456], 'worker_restart_or_missing'),
    ('failed_slots', 1, 'failed_slot'), ('aborted_errors', 2000, 'aborted_write'),
    ('first_failure', {'error': 'clippingRange'}, 'first_failure')])
def test_failures_are_not_hidden_by_later_frames(field, value, error):
    record = good()
    record[field] = value
    assert error in oracle.validate(record)


@pytest.mark.parametrize('change', ['orphan', 'forced_exit', 'thread'])
def test_unclean_teardown_rejected(change):
    record = good()
    record['teardown'].update({'orphan': {'orphans': [123]}, 'forced_exit': {'ordinary_exit': False},
                               'thread': {'threads_stopped': False}}[change])
    assert 'unclean_teardown' in oracle.validate(record)


def test_noise_and_permanently_clipped_images_do_not_prove_clipping():
    record = good()
    record['steps'][2]['frames'][0]['pixels']['tiles'] = [.301]*64
    assert 'clipping_not_observable' in oracle.validate(record)
    record = good()
    record['steps'][3]['frames'][0]['pixels']['tiles'] = [.05]*64
    assert 'clipping_not_observable' in oracle.validate(record)


def test_two_panes_must_share_controller_and_simulation():
    record = good()
    record['steps'][4]['frames'][1]['controller_id'] = 2
    assert 'unshared_runtime' in oracle.validate(record)


def test_every_engine_trip_and_post_return_edit_required():
    record = good()
    record['switches'].pop()
    record['steps'].pop()
    errors = oracle.validate(record)
    assert 'round_trip_count' in errors and 'missing_steps' in errors


def test_idle_switch_does_not_satisfy_active_work():
    record = good()
    record['switches'][0]['active_work'] = False
    assert 'switch_without_active_work' in oracle.validate(record)


def test_fresh_pixels_with_stale_camera_metadata_rejected():
    record = good()
    record['steps'][0]['frames'][0]['projection_clip'] = [.1, 100]
    assert 'camera_clip_mismatch' in oracle.validate(record)
    record['steps'][0]['frames'][0]['snapshot_matches'] = False
    assert 'stale_frame' in oracle.validate(record)


def test_negative_control_requires_original_failure_after_first_real_frame():
    record = good()
    record.update(first_failure={'kind': 'failed_slot', 'name': 'return-1'},
                  first_underlying_error='attribute_values.clippingRange.type must be Matrix4d',
                  aborted_errors=10)
    assert oracle.baseline_reproduced(record)
    for field, value in [('first_underlying_error', 'missing DLL'), ('aborted_errors', 0),
                          ('steps', []), ('first_failure', {'kind': 'harness_exception'})]:
        changed = copy.deepcopy(record)
        changed[field] = value
        assert not oracle.baseline_reproduced(changed)


def test_new_publication_cannot_disguise_cached_native_output():
    record = good()
    record['steps'][1]['frames'][0]['output_time_ns'] = 1
    assert 'stale_native_output' in oracle.validate(record)
    record['steps'][1]['frames'][0]['session_revision'] = 2
    assert 'stale_native_output' not in oracle.validate(record)


def test_blender_exit_zero_does_not_hide_forced_worker_shutdown():
    record = good()
    record['teardown']['worker_graceful'] = False
    assert 'worker_shutdown_unconfirmed' in oracle.validate(record)


@pytest.mark.parametrize('sidecar', [
    {'thread': {'failure': 'No active OVRTX session'}},
    {'thread': {'leaked_thread': True}},
    {'thread': {'status': 'failed'}},
    {'teardown': {'errors': ['shutdown rejected']}},
    {'stop_confirmed': False},
    {'thread': {'alive': False}, 'teardown': {'ran': False}},
])
def test_graceful_worker_exit_does_not_hide_failed_pane_cleanup(sidecar):
    record = good()
    record['teardown']['sidecars'] = [sidecar]
    assert 'unclean_teardown' in oracle.validate(record)


def test_stopped_pane_with_unconfirmed_stop_is_terminal_but_pending_stop_is_not():
    sidecar = {'thread': {'alive': False, 'failure': ''},
               'stop_confirmed': False, 'teardown': {'ran': True, 'errors': []}}
    assert oracle.sidecar_failed(sidecar)
    sidecar['stop_confirmed'] = None
    assert not oracle.sidecar_failed(sidecar)
    sidecar['thread']['alive'] = True
    sidecar['stop_confirmed'] = False
    assert not oracle.sidecar_failed(sidecar)


def test_single_pane_cannot_claim_shared_viewport_coverage():
    record = good()
    for step in record['steps']:
        step['panes'] = 1
        step['frames'] = step['frames'][:1]
    assert 'scenario_pane_count' in oracle.validate(record)


def test_camera_only_run_cannot_claim_perspective_coverage():
    record = good()
    for step in record['steps']:
        step['view_modes'] = ['CAMERA']*step['panes']
    assert 'view_mode_coverage' in oracle.validate(record)


def test_duplicate_pane_frame_and_lost_shared_controller_rejected():
    record = good()
    record['steps'][4]['frames'][1]['engine_id'] = 1
    record['steps'][5]['frames'][0]['controller_id'] = 9
    assert 'duplicate_or_missing_pane_identity' in oracle.validate(record)
    assert 'pane_close_lost_shared_controller' in oracle.validate(record)


def test_reused_pid_cannot_hide_new_worker_creation_identity():
    record = good()
    record['worker_identities'].append({'pid': 123, 'sha256': 'a'*64, 'created': 2})
    assert 'worker_image_identity_unconfirmed' in oracle.validate(record)
    record = good()
    record['worker_identity_changes'] = [{'expected': {'pid': 123, 'created': 1},
                                         'observed': {'pid': 123, 'created': 2}}]
    assert 'worker_restart_or_missing' in oracle.validate(record)


def test_installed_folder_hash_cannot_substitute_for_loaded_identity():
    record = good()
    record['loaded_native_clients'] = []
    assert 'loaded_package_identity_unconfirmed' in oracle.validate(record)


def test_unrefined_pixels_cannot_conclude_clipping_honor():
    record = good()
    record['steps'][2]['frames'][0]['samples'] = 2
    assert 'clipping_not_refined' in oracle.validate(record)


def test_each_raw_write_rejection_is_counted_once():
    assert oracle.count_write_rejections('ABORTED: x has already been written\nABORTED: y\n') == 2
    record = good()
    record['worker_images_match_installed'] = False
    assert 'loaded_package_identity_unconfirmed' in oracle.validate(record)

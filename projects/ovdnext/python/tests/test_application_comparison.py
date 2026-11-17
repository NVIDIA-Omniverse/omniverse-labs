# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure evidence-policy regressions; these are not rendering benchmarks."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'benchmarks' / 'compare_applications.py'
spec = importlib.util.spec_from_file_location('compare_applications', SOURCE)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)
IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]


def inspector_evidence(side):
    return dict(schema_version=1, complete=True, coverage='closed-property-arrays',
        refresh_policy='current-frame', group_size=10, max_array_to_load=1000,
        expanded_property_ranges=[], array_controls=[], diagnostics=[],
        optional_reads=dict(mechanism='owned-presentation' if side == 'python' else 'direct-native',
            read_requests=[] if side == 'python' else None,
            record_requests=[] if side == 'python' else None,
            publication_count=1 if side == 'python' else None),
        backend_policy={'arrays': 'bounded-owned-window' if side == 'python' else 'direct-native-full-copy'})


@pytest.fixture
def reports(tmp_path):
    settings = dict(face_culling='none', planes='none', background_rgba=[.8, .8, .8, 1],
                    geometry_scale=1, plane_scale=1, particle_size=5, show_particles=True,
                    sleeping_objects='all', trigger_shapes='all', wireframe='none', transparency='none',
                    transparency_alpha=.5, actual_msaa_samples=0,
                    overlays={**dict.fromkeys(comparison.OVERLAYS, 'none'), 'orbit_center': False})
    record = dict(object_id=22, shape_id=23, geometry_id=24, scene_id=30, kind=2, element_index=0,
                  actor_class=4, shape_flags=1, actor_flags=0, sleeping=False, double_sided=False,
                  particle_radius_mode=0, transform=IDENTITY[:], dimensions=[1, 2, 3],
                  bounds=[-1, -2, -3, 1, 2, 3], override_color=None, resource_key=None)
    sample = dict(repetition=0, frame=30, inspector_frame=30, selected_native_ids=[22],
                  hidden_native_ids=[], submitted_native_ids=[22], submitted_shape_ids=[23],
                  visible_shape_records=[[22, 23, 24, 2]],
                  submitted_geometry=dict(schema_version=1, complete=True, mapping_supported=True, diagnostics=[]),
                  source_geometry=dict(schema_version=1, complete=True, diagnostics=[], records=[record], resources=[]))
    common = dict(workload='paused-next', warmup_passes=0, measured_passes=1, effective_settings=settings,
                  view_matrix=IDENTITY[:], projection_matrix=IDENTITY[:],
                  gpu=dict(renderer='NVIDIA L40/PCIe/SSE2', vendor='NVIDIA Corporation', driver_version='570.158.01', api='glx'),
                  surface=dict(condition='same-geometry-fbo', window_logical=[1280., 800.], window_pixels=[1280, 800],
                               geometry_fbo_pixels=[1280, 800], gl_viewport=[0, 0, 1280, 800], device_pixel_ratio=1),
                  completion_policy=dict(mode='completed-update', pre_sample_finish=True, viewport_phase_finish=False,
                                         post_ui_finish=True, post_ui_context='composition', post_ui_context_verified=True,
                                         swap_interval_requested=1),
                  job=dict(capture_sha256='a' * 64, frames=[30], scene_instance_id=30, selected_id=22,
                           excluded_actor_ids=[], fov=60, near=.01),
                  samples=[sample])
    py = {'captures': [copy.deepcopy(common)]}
    qt = copy.deepcopy(common)
    qt['capture_sha256'] = 'a' * 64
    py['captures'][0]['samples'][0]['completed_iteration_ms'] = 4
    qt['samples'][0]['completed_update_ms'] = 2
    for side, case in (('python', py['captures'][0]), ('qt', qt)):
        case['inspector_workload'] = inspector_evidence(side)
        case['samples'][0]['inspector_workload'] = inspector_evidence(side)
    images = []
    # A real tiny PNG encoded with the standard library; policy checks its file
    # bytes and producer-declared digest, not an external UI/image dependency.
    import struct
    import zlib
    def chunk(kind, payload):
        return struct.pack('>I', len(payload)) + kind + payload + struct.pack('>I', zlib.crc32(kind + payload))
    for index, name in enumerate(('python.png', 'qt.png')):
        data = (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0))
                + chunk(b'IDAT', zlib.compress(bytes([0, index, 0, 0]))) + chunk(b'IEND', b''))
        (tmp_path / name).write_bytes(data)
        images.append(dict(path=name, sha256=hashlib.sha256(data).hexdigest()))
    py['captures'][0]['screenshot_sha256'] = images[0]['sha256']
    qt['images'] = {'final-window.png': images[1]['sha256']}
    review = dict(schema_version=1, accepted=True, condition='same-geometry-fbo', scope='Policy test, not performance evidence',
                  reviewer='test fixture', python_sha256='b' * 64, qt_sha256='c' * 64, artifacts=images)
    return py, qt, review, tmp_path


def run(reports, **kwargs):
    py, qt, review, directory = reports
    return comparison.compare(py, qt, review=review, python_hash='b' * 64, qt_hash='c' * 64,
                              review_directory=directory, **kwargs)


def rejected(result, path):
    assert not result['comparable']
    assert result['timing_ratios'] is None
    assert any(path in issue['path'] for issue in result['issues']), result['issues']


def test_complete_evidence_only_compares_completed_totals(reports):
    py, qt, _, _ = reports
    py['captures'][0]['samples'][0].update(draw_calls=1, geometry_records=[dict(vertex_count=10)], render_submit_ms=500)
    qt['samples'][0].update(render_queue_counts={'opaque': 9}, geometry_records=[dict(vertex_count=50)], transport_update_ms=.01)
    result = run(reports)
    assert result['comparable'], result['issues']
    assert result['equivalent_viewport_work']
    assert result['timing_ratios']['python_over_qt'] == 2


def test_closed_arrays_allow_different_read_mechanisms_not_missing_evidence(reports):
    py, qt, _, _ = reports
    py['captures'][0]['inspector_workload']['optional_reads']['publication_count'] = 4
    result = run(reports)
    assert result['comparable'], result['issues']
    observed = result['diagnostics']['inspector_workload']
    assert observed['python']['case']['optional_reads']['publication_count'] == 4
    assert observed['qt']['case']['optional_reads']['mechanism'] == 'direct-native'
    assert observed['python']['case']['backend_policy'] != observed['qt']['case']['backend_policy']
    del qt['samples'][0]['inspector_workload']
    rejected(run(reports), 'samples[0].inspector_workload')


@pytest.mark.parametrize('where', ('case', 'sample'))
@pytest.mark.parametrize('field,value', [
    ('schema_version', True), ('schema_version', 2), ('complete', False),
    ('coverage', 'all-arrays'), ('refresh_policy', 'unknown'),
    ('group_size', 9), ('group_size', True), ('group_size', 10000),
    ('max_array_to_load', -1), ('max_array_to_load', 100000),
    ('expanded_property_ranges', None), ('array_controls', None), ('backend_policy', {}),
    ('diagnostics', ['observed expansion is unsupported']), ('optional_reads', None)])
def test_incomplete_or_malformed_inspector_evidence_rejects(reports, where, field, value):
    case = reports[1]
    target = case if where == 'case' else case['samples'][0]
    target['inspector_workload'][field] = value
    rejected(run(reports), 'inspector_workload.' + field)


@pytest.mark.parametrize('field,value', [('group_size', 20), ('max_array_to_load', 0)])
def test_matching_source_geometry_does_not_excuse_different_array_policy(reports, field, value):
    qt = reports[1]
    for target in (qt, qt['samples'][0]):
        target['inspector_workload'][field] = value
    rejected(run(reports), 'inspector_workload.' + field)


def test_both_equal_policy_changes_during_samples_reject(reports):
    for case in (reports[0]['captures'][0], reports[1]):
        case['samples'][0]['inspector_workload']['group_size'] = 20
    rejected(run(reports), 'samples[0].inspector_workload.group_size')


def test_expanded_ranges_remain_unsupported_even_when_both_match(reports):
    for case in (reports[0]['captures'][0], reports[1]):
        for target in (case, case['samples'][0]):
            target['inspector_workload']['expanded_property_ranges'] = [
                dict(instance_id=22, property_id=41, first=0, count=10)]
    rejected(run(reports), 'inspector_workload.expanded_property_ranges')


def test_observed_closed_arrays_can_resize_without_changing_policy(reports):
    for case in (reports[0]['captures'][0], reports[1]):
        case['inspector_workload']['array_controls'] = [dict(expanded=False, total=10)]
        case['samples'][0]['inspector_workload']['array_controls'] = [dict(expanded=False, total=20)]
    assert run(reports)['comparable']


@pytest.mark.parametrize('control', [None, {}, {'expanded': 0}, {'expanded': True},
    {'expanded': False, 'observed_expansion_states': [True]},
    {'expanded': False, 'observed_expansion_states': [0]}])
def test_array_control_observations_cannot_contradict_closed_coverage(reports, control):
    reports[1]['samples'][0]['inspector_workload']['array_controls'] = [control]
    rejected(run(reports), 'inspector_workload.array_controls')


@pytest.mark.parametrize('side,key,value', [
    ('python', 'read_requests', [[22, 41, 72000, 256]]),
    ('python', 'record_requests', [[22, 41, 0]]),
    ('python', 'read_requests', None), ('python', 'publication_count', True),
    ('python', 'publication_count', -1), ('python', 'mechanism', 'direct-native'),
    ('qt', 'read_requests', []), ('qt', 'record_requests', []),
    ('qt', 'publication_count', 0), ('qt', 'mechanism', 'owned-presentation')])
def test_actual_optional_read_scope_is_required(reports, side, key, value):
    case = reports[0]['captures'][0] if side == 'python' else reports[1]
    case['samples'][0]['inspector_workload']['optional_reads'][key] = value
    rejected(run(reports), 'optional_reads.' + key)


def test_backend_policy_cannot_change_mid_pass(reports):
    reports[1]['samples'][0]['inspector_workload']['backend_policy'] = {'arrays': 'different-copy-policy'}
    rejected(run(reports), 'inspector_workload.backend_policy')


def test_declared_current_policy_does_not_accept_matching_stale_frames(reports):
    for case in (reports[0]['captures'][0], reports[1]):
        case['workload'] = 'normal-playback'
        case['inspection_policy'] = 'frozen-at-pass-start'
        case['pass_starts'] = [dict(repetition=0, inspector_frame=29)]
        case['samples'][0]['inspector_frame'] = 29
    rejected(run(reports), 'inspector_workload.refresh_policy')


def test_fresh_and_frozen_inspection_are_different_even_at_same_first_frame(reports):
    for case in (reports[0]['captures'][0], reports[1]):
        case['workload'] = 'normal-playback'
    qt = reports[1]
    qt['inspection_policy'] = 'frozen-at-pass-start'
    qt['pass_starts'] = [dict(repetition=0, inspector_frame=30)]
    for target in (qt, qt['samples'][0]):
        target['inspector_workload']['refresh_policy'] = 'frozen-at-pass-start'
    rejected(run(reports), 'inspector_workload.refresh_policy')


@pytest.mark.parametrize('field,value', [('face_culling', 'normal'), ('planes', 'auto'),
    ('background_rgba', [0, 0, 0, 1]), ('particle_size', 3), ('actual_msaa_samples', 4), ('sleeping_objects', 'selected')])
def test_settings_difference_blocks(reports, field, value):
    reports[1]['effective_settings'][field] = value
    rejected(run(reports), 'effective_settings.' + field)


@pytest.mark.parametrize('field,value', [('mode', 'phase-completion'), ('viewport_phase_finish', True),
    ('post_ui_context', 'geometry'), ('post_ui_context_verified', False)])
def test_incomplete_or_extra_gpu_barrier_blocks(reports, field, value):
    reports[1]['completion_policy'][field] = value
    rejected(run(reports), 'completion_policy.' + field)


def test_both_missing_is_not_equal_and_unmatched_frames_still_check_geometry(reports):
    py, qt, _, _ = reports
    del py['captures'][0]['effective_settings']['planes']
    del qt['effective_settings']['planes']
    del qt['samples'][0]['source_geometry']
    del qt['samples'][0]['repetition']
    result = run(reports)
    rejected(result, 'effective_settings.planes')
    rejected(result, 'source_geometry')
    rejected(result, 'samples[0]')


@pytest.mark.parametrize('field,value', [('transform', [2] + IDENTITY[1:]), ('dimensions', [1, 2, 4]),
    ('shape_flags', 3), ('override_color', [1, 0, 0, 1]), ('kind', 9), ('shape_id', 999)])
def test_canonical_source_difference_or_invalid_record_blocks(reports, field, value):
    reports[1]['samples'][0]['source_geometry']['records'][0][field] = value
    rejected(run(reports), 'source_geometry')


def test_source_resource_hash_and_shape_coverage_are_not_inferred(reports):
    source = reports[1]['samples'][0]['source_geometry']
    source['records'][0]['resource_key'] = 'missing'
    rejected(run(reports), 'source_geometry')
    source['records'][0]['resource_key'] = None
    reports[1]['samples'][0]['submitted_shape_ids'] = None
    reports[1]['samples'][0]['visible_shape_records'] = None
    rejected(run(reports), 'submitted_shape_ids')
    rejected(run(reports), 'visible_shape_records')


@pytest.mark.parametrize('field,value', [('selected_native_ids', []),
    ('submitted_native_ids', []), ('submitted_shape_ids', []), ('inspector_frame', 29)])
def test_selection_visibility_and_property_freshness_are_workload(reports, field, value):
    reports[1]['samples'][0][field] = value
    rejected(run(reports), field)


def test_submitted_row_multiplicity_and_native_membership_are_checked(reports):
    reports[1]['samples'][0]['visible_shape_records'] *= 2
    rejected(run(reports), 'visible_shape_records')


def test_measured_repetitions_are_explicit_and_ordered(reports):
    reports[1]['measured_passes'] = 2
    reports[1]['samples'].append(copy.deepcopy(reports[1]['samples'][0]))
    rejected(run(reports), 'samples')
    reports[1]['samples'][1]['repetition'] = 1
    rejected(run(reports), 'measured_passes')
    reports[0]['captures'][0]['measured_passes'] = 2
    reports[0]['captures'][0]['samples'].append(copy.deepcopy(reports[0]['captures'][0]['samples'][0]))
    reports[0]['captures'][0]['samples'][1]['repetition'] = 1
    assert run(reports)['comparable']
    reports[1]['samples'].reverse()
    rejected(run(reports), 'samples')


def test_same_window_labels_unequal_pixels_and_requires_same_lens(reports):
    py, qt, review, _ = reports
    for case in (py['captures'][0], qt):
        case['surface']['condition'] = 'same-window'
    review['condition'] = 'same-window'
    qt['surface']['geometry_fbo_pixels'] = [719, 538]
    qt['surface']['gl_viewport'] = [0, 0, 719, 538]
    qt['projection_matrix'][0] = 2
    result = run(reports)
    assert result['comparable'], result['issues']
    assert not result['equivalent_viewport_work']
    assert any('not equivalent viewport work' in note for note in result['notes'])
    qt['job']['fov'] = 70
    rejected(run(reports), 'camera.fov')


def test_same_fbo_rejects_projection_and_pixel_mismatch(reports):
    reports[1]['surface']['geometry_fbo_pixels'] = [719, 538]
    reports[1]['projection_matrix'][0] = 2
    result = run(reports)
    rejected(result, 'geometry_fbo_pixels')
    rejected(result, 'projection_matrix')


@pytest.mark.parametrize('mutation', ['missing', 'stale', 'changed_image', 'wrong_report_image', 'duplicate_image', 'non_image'])
def test_quality_evidence_is_bound_to_inputs_and_images(reports, mutation):
    py, qt, review, directory = reports
    if mutation == 'missing':
        reports = py, qt, None, directory
    elif mutation == 'stale':
        review['python_sha256'] = 'd' * 64
    elif mutation == 'changed_image':
        (directory / 'python.png').write_bytes(b'changed')
    elif mutation == 'wrong_report_image':
        py['captures'][0]['screenshot_sha256'] = 'e' * 64
    elif mutation == 'duplicate_image':
        review['artifacts'][1] = review['artifacts'][0]
    elif mutation == 'non_image':
        path = directory / 'python.png'
        path.write_bytes(b'not an image')
        review['artifacts'][0]['sha256'] = py['captures'][0]['screenshot_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    rejected(run(reports), 'visual_quality')


def test_programmatic_missing_input_digests_cannot_accept_null_review(reports):
    py, qt, review, directory = reports
    review['python_sha256'] = review['qt_sha256'] = None
    rejected(comparison.compare(py, qt, review=review, review_directory=directory), 'visual_quality')


def test_bool_ids_nonfinite_values_and_software_rendering_rejected(reports):
    reports[1]['samples'][0]['selected_native_ids'] = [True]
    reports[1]['effective_settings']['particle_size'] = float('nan')
    reports[1]['gpu']['renderer'] = 'llvmpipe'
    result = run(reports)
    rejected(result, 'selected_native_ids')
    rejected(result, 'particle_size')
    rejected(result, 'gpu.renderer')


@pytest.mark.parametrize('content', ['{"x":NaN}', '{"x":1e9999}', '{"x":1,"x":2}'])
def test_strict_json_reader_rejects_nonfinite_and_duplicate_keys(tmp_path, content):
    path = tmp_path / 'bad.json'
    path.write_text(content)
    with pytest.raises(ValueError):
        comparison.read_json(path)


def test_invalid_overlays_is_a_reported_rejection(reports):
    reports[1]['effective_settings']['overlays'] = None
    rejected(run(reports), 'effective_settings.overlays')


def test_viewport_and_window_scale_must_be_internally_consistent(reports):
    for case in (reports[0]['captures'][0], reports[1]):
        case['surface']['gl_viewport'] = [0, 0, 640, 400]
        case['surface']['device_pixel_ratio'] = 2
    result = run(reports)
    rejected(result, 'surface.gl_viewport')
    rejected(result, 'surface.device_pixel_ratio')


def test_both_frontends_dropping_requested_selection_cannot_pass(reports):
    for case in (reports[0]['captures'][0], reports[1]):
        case['samples'][0]['selected_native_ids'] = []
    rejected(run(reports), 'selected_native_ids')


def test_both_stale_inspectors_require_explicit_normal_playback_freeze(reports):
    py, qt, _, _ = reports
    for case in (py['captures'][0], qt):
        case['samples'][0]['inspector_frame'] = 29
    rejected(run(reports), 'inspector_frame')
    for case in (py['captures'][0], qt):
        case['workload'] = 'normal-playback'
        case['inspection_policy'] = 'frozen-at-pass-start'
        case['inspector_workload']['refresh_policy'] = 'frozen-at-pass-start'
        case['samples'][0]['inspector_workload']['refresh_policy'] = 'frozen-at-pass-start'
        case['pass_starts'] = [dict(repetition=0, inspector_frame=29)]
    assert run(reports)['comparable']
    assert any('frozen inspection' in note for note in run(reports)['notes'])
    qt['pass_starts'][0]['inspector_frame'] = 28
    rejected(run(reports), 'inspector_frame')


@pytest.mark.parametrize('timings', [(1e308, 1e-308), (1e308, 1e308)])
def test_finite_input_timings_cannot_emit_nonfinite_ratios(reports, timings):
    py, qt, _, _ = reports
    py['captures'][0]['samples'][0]['completed_iteration_ms'] = timings[0]
    qt['samples'][0]['completed_update_ms'] = timings[1]
    if timings[0] == timings[1]:
        for case in (py['captures'][0], qt):
            case['job']['frames'] = [30, 31]
            case['samples'].append(copy.deepcopy(case['samples'][0]))
            case['samples'][1].update(frame=31, inspector_frame=31)
    rejected(run(reports), 'timing_ratios')


def test_mesh_source_requires_owned_resource_even_if_both_omit_it(reports):
    for case in (reports[0]['captures'][0], reports[1]):
        case['samples'][0]['source_geometry']['records'][0]['kind'] = 5
        case['samples'][0]['visible_shape_records'][0][3] = 5
    rejected(run(reports), 'resource_key')


def test_viewport_screenshot_is_valid_review_provenance(reports):
    case = reports[0]['captures'][0]
    case['viewport_screenshot_sha256'] = case.pop('screenshot_sha256')
    assert run(reports)['comparable']


def test_missing_sample_array_is_reported_without_crashing(reports):
    reports[1]['samples'] = None
    rejected(run(reports), 'samples')


def test_cli_rejects_without_review_and_preserves_existing_evidence(reports, tmp_path):
    import subprocess
    import sys
    py, qt, _, _ = reports
    a, b, output = (tmp_path / name for name in ('python.json', 'qt.json', 'comparison.json'))
    a.write_text(json.dumps(py))
    b.write_text(json.dumps(qt))
    command = [sys.executable, str(SOURCE), str(a), str(b), '--output', str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 2
    rejected(json.loads(output.read_text()), 'visual_quality')
    before = output.read_bytes()
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 2
    assert 'Refusing to overwrite' in result.stderr
    assert output.read_bytes() == before


def cascade_tag_storage(reports):
    """Same excluded plane actor, explicit Python descendant tags and Qt root."""
    for case in (reports[0]['captures'][0], reports[1]):
        case['job']['excluded_actor_ids'] = [25]
        source = case['samples'][0]['source_geometry']
        plane = copy.deepcopy(source['records'][0])
        plane.update(object_id=25, shape_id=26, geometry_id=27, kind=4)
        source['records'].append(plane)
    reports[0]['captures'][0]['samples'][0]['hidden_native_ids'] = [25, 26, 27]
    reports[1]['samples'][0]['hidden_native_ids'] = [25]


def test_different_cascade_tag_storage_normalizes_with_authoritative_coverage(reports):
    cascade_tag_storage(reports)
    result = run(reports)
    assert result['comparable'], result['issues']
    assert result['timing_ratios']['python_over_qt'] == 2
    assert any('cascade tag storage' in note for note in result['notes'])
    assert result['diagnostics']['visibility']['python']['samples'][0]['hidden_native_ids'] == [25, 26, 27]
    assert result['diagnostics']['visibility']['qt']['samples'][0]['hidden_native_ids'] == [25]


@pytest.mark.parametrize('field,value', [('complete', False), ('mapping_supported', False),
    ('schema_version', 2), ('diagnostics', ['Unscoped queue item'])])
def test_matching_actors_and_source_cannot_replace_complete_mapping(reports, field, value):
    cascade_tag_storage(reports)
    reports[1]['samples'][0]['submitted_geometry'][field] = value
    rejected(run(reports), 'submitted_geometry.' + field)


def test_old_unsupported_qt_witness_remains_rejected(reports):
    cascade_tag_storage(reports)
    sample = reports[1]['samples'][0]
    del sample['submitted_geometry']
    sample.update(submitted_shape_mapping_supported=False, submitted_shape_ids=None, visible_shape_records=None)
    result = run(reports)
    rejected(result, 'submitted_geometry')
    rejected(result, 'submitted_shape_ids')
    rejected(result, 'visible_shape_records')


def test_legacy_unsupported_flag_cannot_contradict_new_complete_mapping(reports):
    cascade_tag_storage(reports)
    reports[1]['samples'][0]['submitted_shape_mapping_supported'] = False
    rejected(run(reports), 'submitted_shape_mapping_supported')


@pytest.mark.parametrize('roots', [[], [26], None])
def test_exclusion_roots_must_match_even_when_submitted_geometry_agrees(reports, roots):
    cascade_tag_storage(reports)
    reports[1]['job']['excluded_actor_ids'] = roots
    rejected(run(reports), 'job.excluded_actor_ids')


def test_dropped_duplicate_native_invocation_is_not_hidden_by_id_sets(reports):
    cascade_tag_storage(reports)
    for case in (reports[0]['captures'][0], reports[1]):
        sample = case['samples'][0]
        duplicate = copy.deepcopy(sample['source_geometry']['records'][0])
        duplicate['element_index'] = 1
        sample['source_geometry']['records'].append(duplicate)
        sample['visible_shape_records'].append(sample['visible_shape_records'][0][:])
    assert run(reports)['comparable']
    reports[1]['samples'][0]['visible_shape_records'].pop()
    result = run(reports)
    rejected(result, 'visible_shape_records')
    assert not any('submitted_native_ids' in issue['path'] or 'submitted_shape_ids' in issue['path'] for issue in result['issues'])


def test_dropped_shape_under_same_actor_is_not_hidden_by_actor_ids(reports):
    cascade_tag_storage(reports)
    for case in (reports[0]['captures'][0], reports[1]):
        sample = case['samples'][0]
        extra = copy.deepcopy(sample['source_geometry']['records'][0])
        extra.update(shape_id=28, geometry_id=29)
        sample['source_geometry']['records'].append(extra)
        sample['visible_shape_records'].append([22, 28, 29, 2])
        sample['submitted_shape_ids'].append(28)
    assert run(reports)['comparable']
    sample = reports[1]['samples'][0]
    sample['visible_shape_records'].pop()
    sample['submitted_shape_ids'].pop()
    result = run(reports)
    rejected(result, 'visible_shape_records')
    rejected(result, 'submitted_shape_ids')
    assert not any('submitted_native_ids' in issue['path'] for issue in result['issues'])

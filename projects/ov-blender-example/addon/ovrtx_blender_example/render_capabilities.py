# SPDX-License-Identifier: Apache-2.0
"""Explicit admission boundary for candidate-only render conveniences.

Blender F12, render animation, scene-linear EXR output, Blender display exposure,
the baseline DLSS preset toggle and physics playback/baking remain supported.
These checks only constrain additions absent from the compatibility baseline.
"""


def require_candidate_capture(feature):
    raise ValueError(
        f'{feature} is not admitted in the Windows MVP; use Blender Render '
        'Image or Render Animation. Candidate capture operators are future work')


def validate_candidate_settings(request):
    from .ovrtx_runtime_client import RenderClientError
    if str(getattr(request, 'dlss_quality', 'INHERIT')) != 'INHERIT':
        raise RenderClientError('Candidate DLSS quality presets are not admitted; '
                                'use the supported dlss_enabled preset control')
    exposure = getattr(request, 'native_exposure', None)
    if exposure is not None and (exposure.enabled or exposure.white_point_scale != 10.0):
        raise RenderClientError('Candidate native exposure controls are not admitted; '
                                'use Blender color management with scene-linear HDR')


def validate_request(request):
    """Gate cold and live requests before accepting any new request identity."""
    from .ovrtx_runtime_client import RenderClientError
    validate_candidate_settings(request)
    if getattr(request, 'semantic_output_path', ''):
        raise RenderClientError('Semantic/additional AOV output is future work for the Windows MVP')
    declared=tuple(getattr(request,'sensor_paths',()))
    selected=tuple(getattr(request,'selected_sensor_paths',()))
    if len(declared)>1 or len(selected)>1:
        raise RenderClientError('Multiple RenderProducts are future work; request exactly one product')
    if selected and declared and selected[0] not in declared:
        raise RenderClientError('Selected output is not declared by the current scene')

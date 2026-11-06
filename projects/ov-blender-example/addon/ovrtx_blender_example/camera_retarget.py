# SPDX-License-Identifier: Apache-2.0
"""Retarget an existing public-SDK product without replacing its population.

Port of the research build's bounded camera transaction. Non-camera identity
must remain accepted; every target already exists in that population.
"""
from dataclasses import replace
from pathlib import Path

from . import ovrtx_session
from .camera_value_conversion import SUPPORTED_USD_ATTRIBUTES
from .render_requests import camera_projection_usd_attributes

CAMERA_ATTRIBUTES = {**SUPPORTED_USD_ATTRIBUTES, 'projection': 'Token',
                     'horizontalApertureOffset': 'Float', 'verticalApertureOffset': 'Float'}
from .native_exposure import ENABLED_ATTRIBUTE, TARGET_ATTRIBUTE


def camera_values(previous_request, request, previous, desired):
    if previous_request is None:
        return None
    # Immutable generation roots are part of the composition digest. Changing
    # that source necessarily requires population reconciliation; rebuilding a
    # second composition with the old camera cannot make it reusable.
    if (Path(previous.ovrtx_scene_composition.source_scene_path).expanduser().resolve() !=
            Path(desired.ovrtx_scene_composition.source_scene_path).expanduser().resolve()):
        return None
    if previous.render_var != desired.render_var or previous.sensor_paths != desired.sensor_paths:
        return None
    normalized = replace(request, **{name: getattr(previous_request, name) for name in (
        'camera_prim_path', 'camera_matrix', 'camera_projection', 'scene_camera_matrix',
        'camera_value_route_classes', 'native_exposure')})
    if not ovrtx_session.reuse_decision(previous, ovrtx_session.build_spec(normalized)).reuse:
        return None
    from pxr import Usd, UsdGeom
    # One accepted base, not two opens of its full presentation compositions.
    stage = Usd.Stage.Open(previous.ovrtx_scene_composition.source_scene_path)
    if stage is None:
        return None
    selected = request.camera_prim_path
    targets = {selected, previous_request.camera_prim_path}
    rows = []
    for target in sorted(targets):
        prim = stage.GetPrimAtPath(target)
        if not UsdGeom.Camera(prim):
            return None
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        if target == selected:
            matrix = (request.camera_matrix if request.camera_matrix is not None else
                      request.native_scene_camera_matrix if request.native_scene_camera_matrix is not None else
                      request.scene_camera_matrix if request.scene_camera_matrix is not None else matrix)
        rows.extend(({'prim_path': target, 'attribute_name': 'omni:xform', 'value_type': 'Matrix4d',
                      'value': [[float(v) for v in row] for row in matrix]},
                     {'prim_path': target, 'attribute_name': 'omni:resetXformStack', 'value_type': 'Bool', 'value': True}))
        attributes = {name: (kind, prim.GetAttribute(name).Get())
                      for name, kind in CAMERA_ATTRIBUTES.items()}
        if target == selected and request.camera_projection is not None:
            attributes.update({name: (CAMERA_ATTRIBUTES[name], value)
                               for name, value in camera_projection_usd_attributes(request.camera_projection).items()})
        exposure = request.native_exposure if target == selected else previous_request.native_exposure
        attributes.update({ENABLED_ATTRIBUTE: ('Bool', exposure.enabled),
                           TARGET_ATTRIBUTE: ('Float', exposure.white_point_scale)})
        for name, (kind, value) in sorted(attributes.items()):
            if value is not None:
                if not isinstance(value, (str, int, float, bool)):
                    value = list(value)
                rows.append({'prim_path': target, 'attribute_name': name, 'value_type': kind, 'value': value})
    for product in desired.sensor_paths:
        rows.extend(({'prim_path': product, 'attribute_name': 'camera', 'value_type': 'Relationship', 'value': [selected]},
                     {'prim_path': product, 'attribute_name': ENABLED_ATTRIBUTE, 'value_type': 'Bool',
                      'value': request.native_exposure.enabled},
                     {'prim_path': product, 'attribute_name': TARGET_ATTRIBUTE, 'value_type': 'Float',
                      'value': request.native_exposure.white_point_scale}))
    return rows

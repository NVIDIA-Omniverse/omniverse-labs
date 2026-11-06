# SPDX-License-Identifier: Apache-2.0
"""Existing-camera/product packets; bootstrap USD supplies identities once."""
from dataclasses import dataclass
from types import SimpleNamespace
from . import render_requests
from .camera_retarget import CAMERA_ATTRIBUTES
from .native_exposure import ENABLED_ATTRIBUTE, TARGET_ATTRIBUTE


@dataclass(frozen=True)
class Result:
    values: tuple
    fallback: tuple
    next_contract: object


def capture_contract(stage, scene, generation):
    from .scene_generation import blender_id
    objects = {blender_id(obj, 'OBJECT'): obj for obj in scene.objects}
    return {key: (mapping, objects[key]) for key, mapping in generation.blender_prim_paths.items()
            if key.kind == 'OBJECT' and mapping.blender_id_type == 'CAMERA'
            and key in objects
            and stage.GetPrimAtPath(mapping.schema_path)
            and str(stage.GetPrimAtPath(mapping.schema_path).GetTypeName()) == 'Camera'}


def projection_values(path, projection):
    attributes = render_requests.camera_projection_usd_attributes(projection)
    return [dict(prim_path=path, attribute_name=name, value_type=CAMERA_ATTRIBUTES[name], value=value)
            for name, value in attributes.items()]


def plan(scene, generation, changed_ids, contract, *, depsgraph):
    rows, reasons = [], []
    width = max(1, int(scene.render.resolution_x * scene.render.resolution_percentage / 100))
    height = max(1, int(scene.render.resolution_y * scene.render.resolution_percentage / 100))
    for key in changed_ids:
        item = contract.get(key)
        if item is None:
            reasons.append('camera_not_populated'); continue
        mapping, obj = item
        if obj.type != 'CAMERA' or obj.name_full != mapping.blender_id_name:
            reasons.append('camera_identity_changed'); continue
        evaluated = obj.evaluated_get(depsgraph)
        try:
            projection = render_requests.camera_projection_from_scene_camera(
                SimpleNamespace(camera=evaluated), None, width, height)
        except Exception as exc:
            reasons.append('camera_projection_unsupported:' + str(exc)); continue
        if projection is None:
            reasons.append('camera_projection_unavailable'); continue
        rows.extend(projection_values(mapping.schema_path, projection))
        matrix = tuple(tuple(float(evaluated.matrix_world[j][i]) for j in range(4)) for i in range(4))
        rows.extend((dict(prim_path=mapping.object_path, attribute_name='omni:xform', value_type='Matrix4d', value=matrix),
                     dict(prim_path=mapping.object_path, attribute_name='omni:resetXformStack', value_type='Bool', value=True)))
    return Result(tuple(rows), tuple(reasons), contract)


def request_catalog(source):
    from pxr import Usd
    stage = Usd.Stage.Open(str(source))
    if stage is None: return None
    cameras, products = set(), set()
    for prim in stage.Traverse():
        kind = str(prim.GetTypeName())
        if kind == 'Camera': cameras.add(str(prim.GetPath()))
        elif kind == 'RenderProduct': products.add(str(prim.GetPath()))
    return cameras, products


def request_rows(previous, request, spec, catalog):
    """Admit only current-authoring requests with unchanged population inputs."""
    if not previous or not request.current_scene_generation or not previous.current_scene_generation:
        return None
    for name in ('input_usd_path', 'worker_command', 'native_client_module', 'sensor_paths',
                 'material_scene_layer', 'light_scene_layer', 'dlss_enabled'):
        if getattr(previous, name) != getattr(request, name): return None
    from .ovrtx_session import _render_var_from_request
    if _render_var_from_request(request) != spec.render_var: return None
    if bool(previous.semantic_output_path) != bool(request.semantic_output_path): return None
    if request.camera_projection is None or catalog is None: return None
    cameras, products = catalog
    if request.camera_prim_path not in cameras or not set(spec.sensor_paths) <= products:
        return None
    rows = projection_values(request.camera_prim_path, request.camera_projection)
    matrix = (request.camera_matrix if request.camera_matrix is not None else
              request.native_scene_camera_matrix if request.native_scene_camera_matrix is not None else
              request.scene_camera_matrix)
    if matrix is None: return None
    rows.extend((dict(prim_path=request.camera_prim_path, attribute_name='omni:xform', value_type='Matrix4d', value=matrix),
                 dict(prim_path=request.camera_prim_path, attribute_name='omni:resetXformStack', value_type='Bool', value=True)))
    from .properties import RTPT_RENDER_SETTINGS
    for product in spec.sensor_paths:
        rows.extend((dict(prim_path=product, attribute_name='camera', value_type='Relationship', value=[request.camera_prim_path]),
                     dict(prim_path=product, attribute_name='resolution', value_type='Int2', value=(max(1, request.width), max(1, request.height)))))
        for name, setting in RTPT_RENDER_SETTINGS.items():
            rows.append(dict(prim_path=product, attribute_name=setting.attribute,
                             value_type='Bool' if setting.dtype == 'bool' else 'Int',
                             value=setting.to_wire(request.rtpt_quality.get(name, setting.default))))
    for path in (request.camera_prim_path, *spec.sensor_paths):
        rows.extend((dict(prim_path=path, attribute_name=ENABLED_ATTRIBUTE, value_type='Bool', value=request.native_exposure.enabled),
                     dict(prim_path=path, attribute_name=TARGET_ATTRIBUTE, value_type='Float', value=request.native_exposure.white_point_scale)))
    return rows

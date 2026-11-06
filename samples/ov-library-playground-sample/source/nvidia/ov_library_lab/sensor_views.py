# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Display native OVRTX camera tensors, retaining raw measurements for export."""
import json
import struct
import numpy as np

OUTPUTS = {'Color': 'LdrColor', 'Distance': 'DistanceToCameraSD',
           'Normals': 'NormalSD', 'Classes': 'SemanticSegmentation'}


def author_lenses(camera):
    from pxr import Gf, Sdf
    prim = camera.GetPrim()
    for schema, model in (('OmniLensDistortionOpenCvFisheyeAPI', 'opencvFisheye'),
                          ('OmniLensDistortionOpenCvPinholeAPI', 'opencvPinhole')):
        prim.AddAppliedSchema(schema)
        for key, value in (('fx',220.), ('fy',220.), ('cx',320.), ('cy',180.),
                           ('k1',0.), ('k2',0.), ('k3',0.), ('k4',0.)):
            prim.CreateAttribute(f'omni:lensdistortion:{model}:{key}', Sdf.ValueTypeNames.Float).Set(value)
        prim.CreateAttribute(f'omni:lensdistortion:{model}:imageSize', Sdf.ValueTypeNames.Int2).Set(Gf.Vec2i(640,360))
    prim.CreateAttribute('omni:lensdistortion:model', Sdf.ValueTypeNames.Token).Set('opencvPinhole')


def configure_lens(camera, fisheye):
    prim = camera.GetPrim()
    values = {'omni:lensdistortion:model': 'opencvFisheye' if fisheye else 'opencvPinhole'}
    focal_pixels = 640*camera.GetFocalLengthAttr().Get()/camera.GetHorizontalApertureAttr().Get()
    for axis in ('fx', 'fy'):
        values['omni:lensdistortion:opencvPinhole:'+axis] = focal_pixels
    for name, value in values.items():
        attr = prim.GetAttribute(name)
        if attr.Get() != value:
            attr.Set(value)


def author_labels(stage):
    from pxr import Sdf
    for path, label in (('/World/CarDisplay/Turntable', 'Car'), ('/World/Rendering', 'Robot'),
                        ('/World/Environment', 'Building')):
        prim = stage.GetPrimAtPath(path)
        if prim:
            prim.AddAppliedSchema('SemanticsAPI:class')
            prim.CreateAttribute('semantic:class:params:semanticType', Sdf.ValueTypeNames.String).Set('class')
            prim.CreateAttribute('semantic:class:params:semanticData', Sdf.ValueTypeNames.String).Set(label)


def decode_labels(values):
    data = np.ascontiguousarray(values).tobytes()
    if len(data) < 4:
        return {}
    count = struct.unpack_from('<I', data, len(data)-4)[0]
    if count*24 > len(data)-4:
        raise ValueError('Invalid semantic ID map')
    labels = {}
    for index in range(count):
        identity, _, _, _, length, offset = struct.unpack_from('<6I', data, index*24)
        if offset+length > len(data)-4:
            raise ValueError('Invalid semantic label offset')
        labels[identity] = data[offset:offset+length].decode('utf-8').rstrip('\x00')
    return labels


def colorize(name, values, labels=None):
    """Fixed scales and label-derived colors, never infer sensors from RGB."""
    if name == 'LdrColor':
        return values
    result = np.full((*values.shape[:2], 4), 255, dtype=np.uint8)
    if name == 'DistanceToCameraSD':
        distance = values[..., 0]
        valid = np.isfinite(distance) & (distance > 0)
        unit = np.clip(np.nan_to_num(distance, nan=0, posinf=20)/20., 0, 1)
        result[..., :3] = np.stack((unit, 1-np.abs(2*unit-1), 1-unit), axis=-1)*255
        result[~valid, :3] = 0
    elif name == 'NormalSD':
        result[..., :3] = np.clip(np.nan_to_num(values[..., :3])*.5+.5, 0, 1)*255
    else:
        ids = values[..., 0]
        result[..., :3] = (28, 34, 40)
        for identity, label in (labels or {}).items():
            label = label.casefold()
            color = ((85, 210, 245) if 'class: car;' in label else (255, 180, 75) if 'class: robot;' in label
                     else (143, 164, 187) if 'class: building;' in label else (130, 110, 170))
            result[ids == identity, :3] = color
    return result


def export(runtime, folder):
    folder.mkdir(parents=True, exist_ok=True)
    if runtime.sensor_raw is None or runtime.sensor_raw_name != runtime.sensor_output:
        raise RuntimeError('Wait for a sensor image before exporting.')
    np.save(folder/'measurement.npy', runtime.sensor_raw)
    (folder/'measurement.json').write_text(json.dumps({
        'sourceName': runtime.sensor_raw_name, 'units': 'metres' if runtime.sensor_raw_name == 'DistanceToCameraSD' else 'native output',
        'labels': runtime.sensor_labels, 'shape': runtime.sensor_raw.shape,
        'camera': runtime.sensor_raw_camera,
        'description': 'Unmodified native OVRTX tensor; display colors are separate.'}, indent=2), encoding='utf-8')

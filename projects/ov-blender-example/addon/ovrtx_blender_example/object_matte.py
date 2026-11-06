# SPDX-License-Identifier: Apache-2.0
"""Durable matte and holdout opinions; omission preserves source opinions."""


def opinions(obj):
    rendering = getattr(getattr(obj, 'ov', None), 'rendering', None)
    result = {}
    for field, name in (('matte', 'primvars:isMatteObject'), ('holdout', 'primvars:holdoutObject')):
        value = getattr(rendering, field, 'INHERIT')
        if value not in ('INHERIT', 'ON', 'OFF'):
            raise ValueError('Invalid object matte opinion')
        if value != 'INHERIT':
            result[name] = value == 'ON'
    return result


def opinion_records(scene, mappings, identity):
    from pxr import Sdf, Usd, UsdGeom
    from .usd_opinion_records import AddOnUsdOpinionRecord

    records = []
    for obj in getattr(scene, 'objects', ()):
        values = opinions(obj)
        if not values:
            continue
        mapping = mappings.get(identity(obj, 'OBJECT'))
        if mapping is None:
            raise ValueError('Matte object has no exported USD identity')
        if obj.type != 'MESH':
            raise ValueError('Matte and holdout require a mesh object')
        layer = Sdf.Layer.CreateAnonymous('.usda')
        stage = Usd.Stage.Open(layer)
        prim = stage.OverridePrim(mapping.schema_path)
        for name, value in values.items():
            UsdGeom.PrimvarsAPI(prim).CreatePrimvar(name.removeprefix('primvars:'), Sdf.ValueTypeNames.Bool, UsdGeom.Tokens.constant).Set(value)
        records.append(AddOnUsdOpinionRecord(mapping.schema_path, layer, {'source': 'object_matte'}))
    return tuple(records)

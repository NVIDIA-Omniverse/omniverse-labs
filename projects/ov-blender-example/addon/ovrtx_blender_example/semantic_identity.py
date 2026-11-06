# SPDX-License-Identifier: Apache-2.0
"""Durable class ownership, independent of renderer-generated numeric IDs."""
from dataclasses import dataclass

@dataclass(frozen=True)
class SemanticIdentity:
    class_id: str
    name: str
    source_id: str

def identity(obj):
    get = getattr(obj, 'get', lambda key, default=None: default)
    label = str(get('ovrtx_semantic_class', '')).strip()
    if not label:
        return None
    if any((character in label for character in ';\n\r')):
        raise ValueError('Semantic class cannot contain separators')
    source = str(get('ovrtx_semantic_source_id', obj.name_full))
    return SemanticIdentity(label, str(get('ovrtx_semantic_name', label)), source)

def scene_labels(scene):
    labels = {}
    for obj in getattr(scene, 'objects', ()):
        item = identity(obj)
        if item is None:
            continue
        if item.class_id in labels and labels[item.class_id]['name'] != item.name:
            raise ValueError('Conflicting names for semantic class ' + item.class_id)
        record = labels.setdefault(item.class_id, dict(id=item.class_id, name=item.name, source_ids=[]))
        record['source_ids'].append(item.source_id)
    for item in labels.values():
        item['source_ids'] = sorted(set(item['source_ids']))
    return tuple((labels[key] for key in sorted(labels)))

def opinion(obj, prim_path):
    item = identity(obj)
    if item is None:
        return None
    from pxr import Sdf, Usd
    from .usd_opinion_records import AddOnUsdOpinionRecord
    layer = Sdf.Layer.CreateAnonymous('.usda')
    stage = Usd.Stage.Open(layer)
    prim = stage.OverridePrim(prim_path)
    prim.SetMetadata('apiSchemas', Sdf.TokenListOp.Create(prependedItems=['SemanticsAPI:class']))
    prim.CreateAttribute('semantic:class:params:semanticType', Sdf.ValueTypeNames.String).Set('class')
    prim.CreateAttribute('semantic:class:params:semanticData', Sdf.ValueTypeNames.String).Set(item.class_id)
    return AddOnUsdOpinionRecord(prim_path, layer, dict(semantic_class=item.class_id, source_id=item.source_id))

def opinion_records(scene, mappings, blender_id):
    records = []
    for obj in getattr(scene, 'objects', ()):
        mapping = mappings.get(blender_id(obj, 'OBJECT'))
        if mapping is not None:
            record = opinion(obj, str(mapping.object_path))
            if record is not None:
                records.append(record)
    return tuple(records)


def fingerprint(obj):
    item = identity(obj)
    return (item.class_id, item.name, item.source_id) if item else ()

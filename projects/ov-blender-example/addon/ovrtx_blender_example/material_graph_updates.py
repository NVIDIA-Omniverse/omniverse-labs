# SPDX-License-Identifier: Apache-2.0
"""Bounded material graph compilation requests, without Blender USD exports.

Uses the existing authored MaterialX converter; unsupported graphs report its
actual blocking reasons rather than pretending arbitrary Blender graphs match.
The worker's USD population projects/compiles graph resources in the same stage.
"""
from dataclasses import dataclass, replace
import hashlib
import json
import re
from . import materialx_openpbr_conversion as conversion
from .material_graph_fingerprint import material_graph_signature
from .material_direct_updates import MaterialState, _values
from .usd_value_edit_support import PRINCIPLED_PREVIEW_SURFACE_VALUE_SPECS

@dataclass(frozen=True)
class GraphPlan:
    updates: tuple
    fallback: tuple
    handled: frozenset
    next_contract: object


def material_layer(material, material_path):
    """Return one complete current MaterialX subtree at an existing exact root."""
    from pxr import Sdf
    path = Sdf.Path(material_path)
    if not path.IsAbsolutePath() or not path.IsPrimPath() or path == Sdf.Path.absoluteRootPath:
        raise ValueError('Material graph requires an absolute material prim root')
    classified = conversion._classify_material(material, ('/GraphConversionBindingProof',), (), {})
    if classified['status'] != conversion._STATUS_GENERATED:
        return None, tuple(classified.get('blocking_reasons', ('unsupported material graph',)))
    parts = material_path.split('/')[1:]
    generated = '/' + conversion._SCOPE + '/' + parts[-1]
    lines = conversion._material_block_lines(parts[-1], classified['openpbr_values'])
    lines = [line.replace(generated + '/', material_path + '/') for line in lines]
    # Existing stock materials may retain a universal Preview output. Make both
    # surface contexts target the new graph, so the old network cannot win.
    surface = material_path + '/ND_open_pbr_surface_surfaceshader.outputs:out'
    lines.insert(2, '    token outputs:surface.connect = <' + surface + '>')
    # Installed renderer caches compiled node input layouts by native prim path.
    # Same-path coordinate rewires were accepted but visibly ignored; new paths
    # rendered the distinct constants correctly. Content-address every shader
    # in this current graph, while keeping the material/binding root stable.
    content = '\n'.join(lines)
    digest = hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
    names = re.findall(r'def Shader "([^"]+)"', content)
    for name in names:
        addressed = name + '_g' + digest
        content = content.replace('def Shader "' + name + '"',
                                  'def Shader "' + addressed + '"')
        content = content.replace(material_path + '/' + name + '.outputs:',
                                  material_path + '/' + addressed + '.outputs:')
    lines = content.splitlines()
    wrapped = []
    for index, parent in enumerate(parts[:-1]):
        wrapped.extend([' ' * index + 'over ' + json.dumps(parent) + ' {'])
    depth = len(parts) - 1
    wrapped.extend(' ' * depth + line for line in lines)
    wrapped.extend(' ' * index + '}' for index in reversed(range(depth)))
    text = '#usda 1.0\n' + '\n'.join(wrapped) + '\n'
    validate_material_layer(material_path, text)
    return text, ()


def validate_material_layer(material_path, text):
    """Validate isolated scope before any native mutation; no scene stage scans."""
    from pxr import Sdf, Usd
    layer = Sdf.Layer.CreateAnonymous('inline-material-validation.usda')
    if not layer.ImportFromString(text):
        raise ValueError('Invalid inline material USD')
    stage = Usd.Stage.Open(layer)
    root = stage.GetPrimAtPath(material_path)
    if not root or root.GetTypeName() != 'Material':
        raise ValueError('Inline graph must define its declared material root')
    for prim in stage.TraverseAll():
        path = str(prim.GetPath())
        inside = path == material_path or path.startswith(material_path + '/')
        ancestor = material_path.startswith(path + '/')
        if not inside and (not ancestor or prim.GetTypeName() or prim.GetAuthoredProperties()):
            raise ValueError('Inline graph escapes declared material subtree')
        if inside and str(prim.GetTypeName()) not in {'Material', 'Shader', 'NodeGraph', 'Scope'}:
            raise ValueError('Inline graph contains nonmaterial scene geometry')
        for attr in prim.GetAttributes():
            for target in attr.GetConnections():
                target_prim = str(target.GetPrimPath())
                if target_prim != material_path and not target_prim.startswith(material_path + '/'):
                    raise ValueError('Inline graph connection escapes declared material subtree')
    return True


def plan(scene, generation, changed_identities, contract):
    """Compile only nominated material graphs from cached original handles."""
    from .material_direct_updates import _key
    affected = frozenset(changed_identities)
    if contract.generation_key != _key(generation):
        return GraphPlan((), ('stale material graph contract',), frozenset(), contract)
    updates = []; errors = []; states = dict(contract.materials)
    preview = {'inputs:' + n for n, _ in PRINCIPLED_PREVIEW_SURFACE_VALUE_SPECS.values()}
    for key in affected:
        if key.kind != 'MATERIAL':
            errors.append('graph route only handles material identities'); continue
        material = contract.material_handles.get(key)
        root = contract.material_paths.get(key)
        if material is None or root is None:
            errors.append('graph material is not in accepted native stage'); continue
        text, reasons = material_layer(material, root)
        if reasons:
            errors.extend(reasons); continue
        updates.append(dict(material_path=root, layer_body=text))
        values = tuple(v for v in _values(material) if v[0] not in preview)
        shader = re.search(r'token outputs:surface.connect = <([^>]+)\.outputs:out>', text).group(1)
        states[key] = MaterialState(material_graph_signature(material), values,
                                    {name: shader for name, _, _ in values})
    if errors:
        return GraphPlan((), tuple(dict.fromkeys(errors)), frozenset(), contract)
    return GraphPlan(tuple(updates), (), affected, replace(contract, materials=states))

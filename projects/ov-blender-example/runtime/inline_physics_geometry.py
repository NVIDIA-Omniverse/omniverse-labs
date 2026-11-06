# SPDX-License-Identifier: Apache-2.0
"""Bounded current collider geometry opinion on the existing physics population."""
import json
import re
import numpy as np
from renderer_values import prepare_value

TYPES = {'points': 'Point3fArray', 'faceVertexCounts': 'IntArray',
         'faceVertexIndices': 'IntArray', 'extent': 'Float3Array',
         'size': 'Double', 'radius': 'Double', 'height': 'Double'}
USD_TYPES = {'points': 'point3f[]', 'faceVertexCounts': 'int[]',
             'faceVertexIndices': 'int[]', 'extent': 'float3[]',
             'size': 'double', 'radius': 'double', 'height': 'double'}


def prepare(rows, current, read, *, allow_dynamic=False):
    if not rows:
        raise ValueError('Collider geometry update requires values')
    values = dict(current)
    targets = set()
    for row in rows:
        path, name = row.get('prim_path'), row.get('attribute_name')
        if not isinstance(path, str) or not re.fullmatch(r'(?:/[A-Za-z_][A-Za-z_0-9]*)+', path):
            raise ValueError('Collider geometry requires a canonical prim path')
        if name not in TYPES or row.get('value_type') != TYPES[name]:
            raise ValueError('Unsupported collider geometry attribute/type')
        parent = path
        while parent:
            if read(parent, 'physics:rigidBodyEnabled') is not None:
                if not allow_dynamic or bool(np.asarray(read(parent, 'physics:kinematicEnabled')).reshape(-1)[0]):
                    raise ValueError('Live geometry updates require a static or standalone dynamic collider')
            parent = parent.rsplit('/', 1)[0]
        target = path, name
        if target in targets:
            raise ValueError('Collider geometry targets must be unique')
        targets.add(target)
        # No schema creation or speculative new prims in the existing-geometry lane.
        if read(path, name) is None:
            raise ValueError('Collider geometry attribute is not populated')
        data = prepare_value(row)[5]
        if name in {'size', 'radius', 'height'} and float(data[0]) <= 0:
            raise ValueError('Collider dimension must be positive')
        values[target] = np.asarray(data).tolist()
    for path in {p for p, n in targets if n in {'points', 'faceVertexCounts', 'faceVertexIndices'}}:
        def value(name):
            return np.asarray(values.get((path, name), read(path, name)))
        points, counts, indices = value('points'), value('faceVertexCounts').reshape(-1), value('faceVertexIndices').reshape(-1)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points):
            raise ValueError('Collider points require nonempty vec3 array')
        if np.any(counts < 3) or int(counts.sum()) != len(indices) or np.any(indices < 0) or np.any(indices >= len(points)):
            raise ValueError('Invalid collider mesh topology')
    tree = {}
    for (path, name), value in sorted(values.items()):
        node = tree
        for part in path.split('/')[1:]:
            node = node.setdefault(part, {})
        node.setdefault(None, {})[name] = value
    lines = ['#usda 1.0']
    def numeric(value):
        if isinstance(value, list):
            return '(' + ', '.join(numeric(x) for x in value) + ')'
        return repr(value)
    def emit(node, depth):
        for part, child in node.items():
            if part is None:
                for name, value in child.items():
                    text = '[' + ', '.join(numeric(v) for v in value) + ']' if TYPES[name].endswith('Array') else numeric(value[0])
                    lines.append(' ' * depth + USD_TYPES[name] + ' ' + name + ' = ' + text)
            else:
                lines.append(' ' * depth + 'over ' + json.dumps(part) + ' {')
                emit(child, depth + 1)
                lines.append(' ' * depth + '}')
    emit(tree, 0)
    return values, '\n'.join(lines) + '\n'


def apply(backend, params):
    if backend.stage is None or backend.physx is None or backend._geometry_failed:
        raise RuntimeError('Collider update requires a healthy loaded physics owner')
    values, text = prepare(params.get('values', []), backend._geometry_values, backend._stage_value, allow_dynamic=True)
    if text == backend._geometry_text:
        return dict(status='ok', updated_count=0, simulation_time_ns=backend.time_ns, ordinal=backend.ordinal)
    before = {name: backend._columns(name) for name in ('position', 'orientation', 'linearVelocity', 'angularVelocity')}
    stage_id, physics_id, clock = id(backend.stage), id(backend.physx), backend.time_ns
    api = backend.ovstage.population
    candidate = api.add_usd_reference_from_string(backend.stage, text, '/')
    backend.ordinal += 1
    try:
        if backend._geometry_handle is not None:
            api.remove_usd(backend.stage, backend._geometry_handle)
        api.apply_usd_changes(backend.stage, ordinal=backend.ordinal)
        backend.stage.advance_write_floor(backend.ordinal).wait()
        backend.physx.update_from_ovstage(backend.ordinal, backend.ordinal)
        backend.physx.wait_all()
        if (id(backend.stage), id(backend.physx), backend.time_ns) != (stage_id, physics_id, clock):
            raise RuntimeError('Collider update replaced physics state')
        after = {name: backend._columns(name) for name in before}
        if any(set(after[name]) != set(before[name]) for name in before):
            raise RuntimeError('Collider update changed body membership')
        changed = {path for name in before for path in before[name] if before[name][path] != after[name][path]}
        nominated = {path for path in before['position'] if any(target == path or target.startswith(path + '/') for target, name in values)}
        if changed - nominated:
            raise RuntimeError('Collider update changed unrelated body motion')
        if changed:
            backend.write('write_poses', dict(reset=True, values=[dict(prim_path=path,
                translate=before['position'][path], orient=before['orientation'][path]) for path in changed]))
            backend.write('write_velocities', dict(values=[dict(prim_path=path,
                linear=before['linearVelocity'][path], angular=before['angularVelocity'][path]) for path in changed]))
        if any(backend._columns(name) != prior for name, prior in before.items()):
            raise RuntimeError('Collider update could not preserve body motion')
    except Exception:
        backend._geometry_failed = True
        raise
    backend._geometry_handle, backend._geometry_values, backend._geometry_text = candidate, values, text
    return dict(status='ok', updated_count=len(params['values']), simulation_time_ns=clock,
                ordinal=backend.ordinal, stage_identity=stage_id, physics_identity=physics_id,
                motion_preserved=True)

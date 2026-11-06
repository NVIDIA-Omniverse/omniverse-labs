# SPDX-License-Identifier: Apache-2.0
"""Explicitly experimental, bounded structural OVStage packets for SDK probes.

No Blender classification automatically selects this route. It tests native
consumption independently of USD generation. The caller owns affected Blender
identities and must discard/repopulate this projection before normal fallback.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import ExitStack
import time
import numpy as np

try:
    from .renderer_values import prepare_value, tensor_value, tensor_width
except ImportError:
    from renderer_values import prepare_value, tensor_value, tensor_width

MAX_ITEMS = 256


def _path(path):
    if (not isinstance(path, str) or not path.startswith('/') or path == '/'
            or any(part in ('', '.', '..') for part in path[1:].split('/'))
            or any(char in path for char in '.{}[]')):
        raise ValueError('Structural packet requires an absolute prim path')
    return path


def _under(path, root):
    return path == root or path.startswith(root + '/')


def _bounded_list(value, name):
    if not isinstance(value, list) or len(value) > MAX_ITEMS:
        raise ValueError(f'{name} requires at most {MAX_ITEMS} entries')
    return value


def _prepare(params):
    if params.get('experimental') is not True:
        raise ValueError('Structural writes require experimental=true and an isolated test scene')
    if set(params) - {'experimental', 'clones', 'creates', 'deletes', 'values'}:
        raise ValueError('Unknown structural packet option')
    clones, creates, deletes, values = [
        _bounded_list(params.get(name, []), name)
        for name in ('clones', 'creates', 'deletes', 'values')]
    destination_sources = {}
    for clone in clones:
        if set(clone) != {'source', 'targets'}:
            raise ValueError('Clone requires source and targets')
        source = _path(clone['source'])
        targets = _bounded_list(clone['targets'], 'clone targets')
        if not targets:
            raise ValueError('Clone requires at least one destination')
        for target in targets:
            target = _path(target)
            if target in destination_sources or _under(target, source) or _under(source, target):
                raise ValueError('Clone destinations must be new, distinct and outside their source')
            destination_sources[target] = source
    metadata = []
    owned_values = list(values)
    created_paths = set()
    for create in creates:
        if set(create) - {'prim_path', 'prim_type', 'schemas', 'values'}:
            raise ValueError('Unknown create option')
        path, kind = _path(create['prim_path']), create['prim_type']
        if not isinstance(kind, str) or not kind or not kind.isidentifier():
            raise ValueError('Create requires an explicit USD prim type token')
        if kind == 'PointInstancer':
            raise ValueError('Direct PointInstancer creation is unsupported by the validated renderer route; use USD population')
        if path in created_paths or any(_under(path, root) for root in destination_sources):
            raise ValueError('Duplicate or overlapping structural destination')
        created_paths.add(path)
        metadata.append(dict(prim_path=path, attribute_name='usd-prim-type', value_type='Token', value=kind))
        if 'schemas' in create:
            metadata.append(dict(prim_path=path, attribute_name='usd-schemas',
                                 value_type='TokenArray', value=create['schemas']))
        for row in _bounded_list(create.get('values', []), 'create values'):
            if 'prim_path' in row and row['prim_path'] != path:
                raise ValueError('Create value escapes its prim')
            owned_values.append({**row, 'prim_path': path})
    deletions = tuple(_path(path) for path in deletes)
    if len(set(deletions)) != len(deletions):
        raise ValueError('Duplicate deletion')
    if any(_under(path, deleted) for path in (*created_paths, *destination_sources) for deleted in deletions):
        raise ValueError('Cannot create and delete overlapping destinations')
    if len(metadata) + len(owned_values) + len(destination_sources) + len(deletions) > MAX_ITEMS:
        raise ValueError('Structural packet exceeds total item budget')
    rows = []
    for row in metadata + owned_values:
        path = _path(row['prim_path'])
        if any(_under(path, deleted) for deleted in deletions):
            raise ValueError('Cannot write into a deleted subtree')
        if row.get('value_type', '').lower() == 'connection':
            targets = _bounded_list(row['value'], 'connection targets')
            for edge in targets:
                if not isinstance(edge, dict) or set(edge) != {'prim_path', 'attribute_name'}:
                    raise ValueError('Connection target requires prim_path and attribute_name')
                _path(edge['prim_path'])
                if not isinstance(edge['attribute_name'], str) or not edge['attribute_name']:
                    raise ValueError('Connection output attribute must be nonempty')
            if not isinstance(row.get('attribute_name'), str) or not row['attribute_name']:
                raise ValueError('Connection destination attribute must be nonempty')
            rows.append((dict(row), None))
        else:
            rows.append((dict(row), prepare_value(row)))
    if not (rows or clones or deletions):
        raise ValueError('Empty structural packet')
    return clones, created_paths, destination_sources, deletions, rows


def _columns(backend, paths, path, names=None):
    """Read one nominated prim, retaining only copied schema/short metadata."""
    result = {}
    with ExitStack() as owners:
        if names is None:
            # Explicit path lists target absent prims for UPSERT and do not
            # evaluate attribute discovery. Only the query API promises that.
            ovs = backend.ovstage
            scope = ovs.Filter([ovs.Predicate('usd-path', ovs.FilterOp.IN, [path])])
            query = owners.enter_context(backend.stage.query(filter=scope))
            query.wait()
            discovered = query.result()
            if not discovered.total_prim_count:
                return result
            # Installed discovery omits synthetic reserved metadata columns;
            # they remain directly readable and are useful to the probe caller.
            tokens = list(dict.fromkeys([*discovered.attributes,
                paths.intern_token('usd-prim-type'), paths.intern_token('usd-schemas')]))
        else:
            plist = owners.enter_context(paths.create_path_list_from_strings([path]))
            query = owners.enter_context(backend.stage.query_from_path_list(plist))
            tokens = [paths.intern_token(name) for name in dict.fromkeys(names)]
        if tokens:
            with backend.stage.read_attributes(query, tokens,
                    backend.ovstage.OrdinalRange.latest(backend.ordinal)) as read:
                read.wait()
                for group in read.groups():
                    try:
                        if group.is_delete:
                            continue
                        name = paths.token_to_string(group.attribute)
                        if group.prim_count != 1 or name in result:
                            raise ValueError(f'Ambiguous structural schema: {path}.{name}')
                        data = np.asarray(group.array(0))
                        result[name] = dict(dtype=data.dtype, semantic=int(group.raw.semantic),
                            is_array=bool(group.is_array), width=tensor_width(group.tensor(0)),
                            shape=list(data.shape), preview=data.reshape(-1)[:16].tolist())
                    finally:
                        backend.stage.release_group(group)
    return result


def inspect(backend, params):
    """Bounded native-column inspection; never scans the scene or writes."""
    paths_to_read = [_path(path) for path in _bounded_list(params.get('prim_paths', []), 'prim_paths')]
    names = params.get('attributes')
    if names is not None:
        _bounded_list(names, 'attributes')
    result = {}
    with backend.ovstage.PathDictionary(backend.stage) as paths:
        for path in paths_to_read:
            columns = _columns(backend, paths, path, names)
            for name, desc in columns.items():
                desc['dtype'] = str(desc['dtype'])
                if (name in {'usd-prim-type', 'usd-schemas'} or
                        desc['semantic'] == int(backend.ovstage.AttributeSemantic.TOKEN_ID)):
                    desc['resolved'] = [paths.token_to_string(int(item)) for item in desc['preview']]
                elif desc['semantic'] == int(backend.ovstage.AttributeSemantic.RELATIONSHIP_PATH_ID):
                    desc['resolved'] = [paths.path_to_string(int(item)) for item in desc['preview']]
                elif desc['semantic'] == int(backend.ovstage.AttributeSemantic.CONNECTION_PATH_ID):
                    preview = desc['preview']
                    desc['resolved'] = [dict(prim_path=paths.path_to_string(int(preview[i])),
                        attribute_name=paths.token_to_string(int(preview[i + 1])))
                        for i in range(0, len(preview) - 1, 2)]
                elif (desc['semantic'] == int(getattr(backend.ovstage.AttributeSemantic, 'ASSET_STRING', 1))
                      and desc['dtype'] == 'uint64' and desc['width'] == 2):
                    # Native USD population stores an asset's authored/resolved
                    # string IDs as a pair, distinct from raw-byte asset edits.
                    try:
                        names = [paths.token_to_string(int(item)) for item in desc['preview']]
                        desc['asset_paths'] = [dict(authored=names[i], resolved=names[i + 1])
                                               for i in range(0, len(names) - 1, 2)]
                    except Exception as exc:
                        desc['asset_decode_error'] = str(exc)
            result[path] = columns
    return {'native_stage_id': backend.native_stage_id, 'ordinal': backend.ordinal, 'columns': result}


def apply(backend, params):
    """Validate a complete probe packet, publish one ordinal, then pull once.

    The SDK batch is not transactional. Any failure after the first mutation
    marks this projection uncertain and requires a reload, rather than silently
    rendering partially applied structure.
    """
    if 'inline_population' in params:
        try:
            from .inline_structure_population import apply as apply_inline_structure
        except ImportError:
            from inline_structure_population import apply as apply_inline_structure
        return apply_inline_structure(backend, params)
    started = time.perf_counter()
    clones, created, destinations, deleted, rows = _prepare(params)
    backend._require_idle_native_render()
    ovs = backend.ovstage
    timings = {}
    with ovs.PathDictionary(backend.stage) as paths:
        requested = defaultdict(set)
        for row, prepared in rows:
            requested[row['prim_path']].add(row['attribute_name'])
        for path in {*created, *destinations, *destinations.values(), *deleted}:
            requested[path].add('usd-prim-type')
        schemas = {path: _columns(backend, paths, path, ['usd-prim-type', *names])
                   for path, names in requested.items()}
        for path in {*created, *destinations}:
            if schemas[path]:
                raise ValueError(f'Structural destination already exists: {path}')
        for path in {*destinations.values(), *deleted}:
            if 'usd-prim-type' not in schemas[path]:
                raise ValueError(f'Structural source does not exist: {path}')
        encoded = []
        for row, prepared in rows:
            path, name = row['prim_path'], row['attribute_name']
            fresh = path in created or any(_under(path, root) for root in destinations)
            if not fresh and 'usd-prim-type' not in schemas[path]:
                raise ValueError(f'Structural value target does not exist: {path}')
            schema = schemas[path].get(name)
            # Clone descendants inherit their source schema; inspect that exact
            # source property, not a whole subtree or historical USD snapshot.
            for root, source in destinations.items():
                if _under(path, root):
                    source_path = source + path[len(root):]
                    source_columns = _columns(backend, paths, source_path, ['usd-prim-type', name])
                    if 'usd-prim-type' not in source_columns:
                        raise ValueError(f'Clone value source is absent: {source_path}')
                    schema = source_columns.get(name)
                    break
            if prepared is None:
                data = np.asarray([[paths.intern_path(edge['prim_path']),
                                    paths.intern_token(edge['attribute_name'])]
                                   for edge in row['value']], dtype=np.uint64).reshape(-1, 2)
                semantic, array = ovs.AttributeSemantic.CONNECTION_PATH_ID, True
                if schema and (schema['dtype'] != np.dtype(np.uint64) or not schema['is_array']
                               or schema['width'] != 2 or schema['semantic'] != int(semantic)):
                    raise ValueError(f'{path}.{name} is not a native connection column; inspect its actual storage name')
                data = ovs.make_dltensor(data,
                    dtype=ovs.DLDataType(ovs.DLDataTypeCode.kDLUInt, 64, 2), shape=[len(data)])
            else:
                data, array, semantic = tensor_value(prepared, ovs, paths,
                    None if schema is None else schema['dtype'],
                    existing_semantic=None if schema is None else schema['semantic'],
                    existing_is_array=None if schema is None else schema['is_array'],
                    existing_width=None if schema is None else schema['width'])
            encoded.append((path, name, data, array, semantic))
        timings['preflight_ms'] = (time.perf_counter() - started) * 1000
        mutation_started = time.perf_counter()
        backend.ordinal += 1
        try:
            for clone in clones:
                backend.stage.clone_async(clone['source'], clone['targets'], backend.ordinal).wait()
            # Establish type/schema first, reset boundaries next, then values.
            for phase in (lambda name: name in {'usd-prim-type', 'usd-schemas'},
                          lambda name: name == 'omni:resetXformStack',
                          lambda name: name not in {'usd-prim-type', 'usd-schemas', 'omni:resetXformStack'}):
                backend._write_phase(paths, [item for item in encoded if phase(item[1])])
            if deleted:
                with paths.create_path_list_from_strings(list(deleted)) as plist:
                    with backend.stage.query_from_path_list(plist) as query:
                        backend.stage.delete_attributes(query, [], backend.ordinal).wait()
            timings['mutation_ms'] = (time.perf_counter() - mutation_started) * 1000
            phase_started = time.perf_counter()
            backend.stage.advance_write_floor(backend.ordinal, ovs.Scope.ALL).wait()
            timings['publication_ms'] = (time.perf_counter() - phase_started) * 1000
            phase_started = time.perf_counter()
            backend.renderer.update_from_stage(backend.ordinal)
            backend.renderer.reset(time=backend.simulation_time_ns / 1e9)
            timings['renderer_pull_reset_ms'] = (time.perf_counter() - phase_started) * 1000
        except Exception:
            backend.experimental_structure_uncertain = True
            raise
    # Reading a clone source does not modify it; keep its accepted overrides.
    touched = set(created) | set(destinations) | set(deleted) | {row['prim_path'] for row, _ in rows}
    try:
        from .native_pose_fanout import successor_fanout
    except ImportError:
        from native_pose_fanout import successor_fanout
    # The stage survives: untouched synthetic world inputs keep their identity.
    backend.native_pose_fanout = successor_fanout(backend.native_pose_fanout, touched)
    backend.sample_count = 0
    backend.desired_values = {key: value for key, value in backend.desired_values.items()
                              if not any(_under(key[0], path) for path in touched)}
    timings['total_ms'] = (time.perf_counter() - started) * 1000
    return dict(experimental=True, production_enabled=False, structural_updated=True,
        native_stage_id=backend.native_stage_id, ordinal=backend.ordinal,
        clone_count=len(destinations), create_count=len(created), delete_count=len(deleted),
        written_attributes=len(encoded), timings=timings,
        source_projection_stale=True)

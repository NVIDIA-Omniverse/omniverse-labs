# SPDX-License-Identifier: Apache-2.0
"""Invalidate RTX's parent mesh when existing GeomSubset data changes.

Installed OVStage population publishes `geomSubsets` as a Mesh relationship to
its subset children. The renderer consumes that parent's dirty column, while a
write to child indices alone leaves its cached per-face materials unchanged.
Reauthor the current relationship at the same ordinal as child writes. This runs
in the worker after value dedup, so an equal relationship cannot be dropped by a
client's accepted-value cache. No persistent cache or scene traversal is needed.
"""
from collections import defaultdict
import numpy as np


def subset_dependency_rows(stage, ovs, paths, prepared, prim_types, *, ordinal):
    parents = defaultdict(set)
    provided = {(row[0], row[1]) for row in prepared}
    for row in prepared:
        path, name = row[:2]
        if prim_types.get(path) == 'GeomSubset' and name in {'indices', 'material:binding'}:
            parent = path.rsplit('/', 1)[0]
            if (parent, 'geomSubsets') not in provided:
                parents[parent].add(path)
    if not parents:
        return ()
    type_token = paths.intern_token('usd-prim-type')
    relation_token = paths.intern_token('geomSubsets')
    expected_semantic = ovs.AttributeSemantic.RELATIONSHIP_PATH_ID
    rows = []
    for parent, children in parents.items():
        relation = None
        parent_type = None
        with paths.create_path_list_from_strings([parent]) as plist, stage.query_from_path_list(plist) as query:
            with stage.read_attributes(query, [type_token, relation_token], ovs.OrdinalRange.latest(ordinal)) as read:
                read.wait()
                for group in read.groups():
                    try:
                        if group.is_delete:
                            continue
                        if group.prim_count != 1:
                            raise ValueError('Ambiguous subset parent: ' + parent)
                        array = np.asarray(group.array(0))
                        if group.attribute == type_token:
                            parent_type = paths.token_to_string(int(array.reshape(-1)[0]))
                        elif group.attribute == relation_token:
                            if (not group.is_array or group.tensor(0).dtype.lanes != 1
                                    or int(group.raw.semantic) != int(expected_semantic)
                                    or array.dtype != np.dtype('uint64')):
                                raise ValueError('Unexpected native geomSubsets schema: ' + parent)
                            relation = np.array(array, dtype=np.uint64, copy=True).reshape(-1)
                    finally:
                        stage.release_group(group)
        if parent_type != 'Mesh' or relation is None:
            raise ValueError('Subset update needs an existing populated Mesh parent: ' + parent)
        links = set(map(int, relation))
        if any(int(paths.intern_path(child)) not in links for child in children):
            raise ValueError('Subset is absent from populated geomSubsets relationship: ' + parent)
        rows.append((parent, 'geomSubsets', relation, True, expected_semantic))
    return tuple(rows)

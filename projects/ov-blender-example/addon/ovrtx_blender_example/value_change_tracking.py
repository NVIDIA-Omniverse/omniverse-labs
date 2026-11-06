# SPDX-License-Identifier: Apache-2.0
"""Exact, acknowledged property state shared by all public-SDK edit families.

An attempted value is never an accepted value. The ledger belongs to one native
population/worker epoch; replacement or recovery invalidates it. Asset values
remain writes until callers can supply resource-content identity, since an
unchanged filename does not prove unchanged texture bytes.
"""
from copy import deepcopy


_POSE_ATTRIBUTES = frozenset({'omni:xform', 'xformOp:transform'})
_POSE_STATE_ATTRIBUTES = _POSE_ATTRIBUTES | {'omni:resetXformStack'}


def _under_changed_parent(path, parents):
    while '/' in path:
        path = path.rsplit('/', 1)[0]
        if path in parents:
            return True
    return False


class AcknowledgedValueLedger:
    def __init__(self):
        self._epoch = None
        self._rows = {}
        self._descendants = {}

    def clear(self):
        self._epoch = None
        self._rows.clear()
        self._descendants.clear()

    def invalidate_subtrees(self, roots, *, epoch):
        """Drop only affected certificates after an acknowledged scoped rebuild."""
        self._bind(epoch)
        roots = tuple(roots)
        if any(not isinstance(root, str) or not root.startswith('/') or root == '/'
               for root in roots):
            raise ValueError('Certificate invalidation requires absolute nonroot paths')
        removed = {key for key in self._rows
                   if any(key[0] == root or key[0].startswith(root.rstrip('/') + '/')
                          for root in roots)}
        for key in removed:
            del self._rows[key]
        for parent in tuple(self._descendants):
            self._descendants[parent].difference_update(removed)
            if not self._descendants[parent]:
                del self._descendants[parent]
        return len(removed)

    def _bind(self, epoch):
        if epoch != self._epoch:
            self.clear()
            self._epoch = epoch

    def changed(self, rows, *, epoch):
        self._bind(epoch)
        # Preserve sequential semantics if a caller deliberately supplies the
        # same property twice; only acknowledged or earlier planned values
        # justify removing a repeated write.
        pending = {}
        changed_parents = {row['prim_path'] for row in rows
                           if row['attribute_name'] in _POSE_ATTRIBUTES
                           and self._rows.get((row['prim_path'], row['attribute_name'])) != row}
        result = []
        for row in rows:
            key = (row['prim_path'], row['attribute_name'])
            kind = row.get('value_type', '').lower().removesuffix('array').removesuffix('[]')
            explicit_descendant = (row['attribute_name'] in _POSE_STATE_ATTRIBUTES
                                   and _under_changed_parent(row['prim_path'], changed_parents))
            # The native pose packet requires its reset boundary to establish
            # world-input semantics on every matrix change. Dropping an
            # unchanged True flag makes camera updates take the conservative
            # renderer-reset path and parent updates repopulate the USD stage.
            pose_boundary = (row['attribute_name'] == 'omni:resetXformStack'
                             and row['prim_path'] in changed_parents
                             and key not in pending)
            if kind == 'asset' or explicit_descendant or pose_boundary or pending.get(key, self._rows.get(key)) != row:
                result.append(deepcopy(row))
                pending[key] = row
        return result

    def acknowledge(self, rows, *, epoch):
        self._bind(epoch)
        changed_parents = {row['prim_path'] for row in rows if row['attribute_name'] in _POSE_ATTRIBUTES}
        # Native parent propagation may move previously explicit descendants.
        # Drop their certificates until their accepted derived receipt (or an
        # explicit child override in this transaction) certifies the new value.
        for parent in changed_parents:
            for key in self._descendants.get(parent, ()):
                self._rows.pop(key, None)
        for row in rows:
            key = (row['prim_path'], row['attribute_name'])
            self._rows[key] = deepcopy(row)
            if key[1] in _POSE_STATE_ATTRIBUTES:
                ancestor = key[0]
                while '/' in ancestor:
                    ancestor = ancestor.rsplit('/', 1)[0]
                    self._descendants.setdefault(ancestor, set()).add(key)

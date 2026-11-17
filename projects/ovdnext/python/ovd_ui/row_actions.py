# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable tree action identities and local flags, separate from native tags.

Qt's InstanceView gives containers, reference leaves and query rows their own
ITEM_HIDDEN/ITEM_TRANSPARENT flags. Selecting such a row is not selecting its
descendants. Only an explicit action resolves a subtree into native roots.
This module owns no UI widgets, native sessions or capture reads.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class RowIdentity:
    capture_key: object
    key: str
    kind: str
    class_name: str = ""
    instance_id: int | None = None
    owner_id: int | None = None
    source_instance_id: int | None = None
    property_id: int | None = None
    query_id: int | None = None
    element_index: int | None = None
    element_count: int = 0
    segment_epoch: int | None = None

    @property
    def token(self):
        # Names, current array length and canonical parent can change while a
        # row's source identity remains the same. Never use a writer handle.
        return (self.capture_key, self.key, self.kind, self.instance_id,
                self.source_instance_id, self.property_id, self.query_id,
                self.element_index, self.segment_epoch)

    @property
    def native(self):
        return self.kind == "object" and self.instance_id is not None and self.instance_id > 0


@dataclass(frozen=True)
class RowActionScope:
    capture_key: object
    segment_epoch: int | None
    action: str
    selected_rows: tuple[RowIdentity, ...]
    native_roots: tuple[int, ...]
    local_rows: tuple[RowIdentity, ...]

    @property
    def local_keys(self):
        return tuple(row.key for row in self.local_rows)

    @property
    def segment_scoped(self):
        return any(row.segment_epoch is not None for row in (*self.selected_rows, *self.local_rows))


class RowActionState:
    """Application-owned row selection and local flags.

    Call sync with each published model's capture/segment. apply returns False
    for a stale scope without changing anything; only after True should the
    caller apply scope.native_roots through InteractionState. Positive native
    tags remain exclusively in that existing state. revision changes only when
    this local state changes, never merely because another frame was published.
    """
    def __init__(self):
        self.capture_key = None
        self.segment_epoch = None
        self.selected = ()
        self._hidden = {}
        self._transparent = {}
        self.revision = 0

    def sync(self, capture_key, segment_epoch=None):
        if capture_key != self.capture_key:
            self.capture_key, self.segment_epoch = capture_key, segment_epoch
            self.selected = ()
            self._hidden.clear()
            self._transparent.clear()
            self.revision += 1
        elif segment_epoch != self.segment_epoch:
            self.segment_epoch = segment_epoch
            selected = tuple(row for row in self.selected if row.segment_epoch is None)
            changed = selected != self.selected
            self.selected = selected
            for flags in (self._hidden, self._transparent):
                transient = [key for key, row in flags.items() if row.segment_epoch is not None]
                changed |= bool(transient)
                for key in transient:
                    del flags[key]
            if changed:
                self.revision += 1

    def _current(self, row):
        return (row.capture_key == self.capture_key and
                (row.segment_epoch is None or row.segment_epoch == self.segment_epoch))

    def select(self, rows):
        rows = tuple(rows)
        if any(not self._current(row) for row in rows):
            return False
        unique = {}
        for row in rows:
            unique.setdefault(row.token, row)
        selected = tuple(unique.values())
        if tuple(row.token for row in selected) != tuple(row.token for row in self.selected):
            self.selected = selected
            self.revision += 1
        else:
            # Refresh descriptive metadata without an unnecessary UI rebuild.
            self.selected = selected
        return True

    def accepts(self, scope):
        return (scope.capture_key == self.capture_key and
                (not scope.segment_scoped or scope.segment_epoch == self.segment_epoch) and
                all(self._current(row) for row in (*scope.selected_rows, *scope.local_rows)))

    def apply(self, scope, enabled):
        if scope.action not in ("visibility", "transparency"):
            raise ValueError("Unknown row action")
        if not self.accepts(scope):
            return False
        if any(row.native for row in scope.local_rows):
            raise ValueError("Native tags belong to InteractionState")
        flags = self._hidden if scope.action == "visibility" else self._transparent
        tagged = not enabled if scope.action == "visibility" else bool(enabled)
        changed = False
        for row in scope.local_rows:
            if tagged:
                changed |= row.token not in flags
                flags[row.token] = row
            else:
                changed |= row.token in flags
                flags.pop(row.token, None)
        if changed:
            self.revision += 1
        return True

    def is_hidden(self, row, native_hidden=()):
        if not self._current(row):
            return False
        return row.instance_id in native_hidden if row.native else row.token in self._hidden

    def is_transparent(self, row, native_transparent=()):
        if not self._current(row):
            return False
        return row.instance_id in native_transparent if row.native else row.token in self._transparent

    def show_all(self):
        if self._hidden:
            self._hidden.clear()
            self.revision += 1

    def hide_all(self, identities):
        """Hide currently represented local rows; caller hides native IDs separately.

        Does not construct lazy pages or impose flags on future rows/objects.
        A newly published row starts with its own default flag, as in Qt.
        """
        rows = tuple(identities)
        if any(not self._current(row) for row in rows):
            return False
        changed = False
        for row in rows:
            if not row.native:
                changed |= row.token not in self._hidden
                self._hidden[row.token] = row
        if changed:
            self.revision += 1
        return True

    def clear_transparency(self):
        if self._transparent:
            self._transparent.clear()
            self.revision += 1

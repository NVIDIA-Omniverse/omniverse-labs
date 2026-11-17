# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Selection, history and presentation tags over native creation IDs.

The sticky/alive selection split, subtree tags and per-frame persistence follow
PVDNext/instanceview.h/.cpp. Transparency modes match pvdsettings.h values and
pvdrenderer.cpp (tags are consulted only in Tagged mode). Rendering and widgets
consume snapshots; this module never reads external pointers or mutates captures.

Call begin_capture for every new displayed Session, even when reopening the same
path. Native positive creation IDs remain valid across resampling, so segment
changes preserve their sticky selection and tags; they never transfer to new IDs
with reused writer handles. Negative synthetic query IDs are scoped to a segment
and cleared when its epoch changes. Expansion keys should use stable model keys,
not row indices. Frame updates never create selection-history entries.
"""
from collections import deque
from dataclasses import dataclass
from enum import IntEnum
import math
from numbers import Integral


class TransparencyMode(IntEnum):
    NONE = 0
    ALL = 1
    SELECTED = 2
    UNSELECTED = 3
    TAGGED = 4


def _ids(values):
    result, seen = [], set()
    for value in values:
        # Native bindings return built-in ints. Avoid the numbers.Integral ABC
        # lookup for that common case while retaining NumPy/integer subclasses
        # and the existing strict rejection of booleans and coercible strings.
        if type(value) is not int:
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError("Interaction identity must be a native integer ID")
            value = int(value)
        if value in (0, -1):
            continue
        if value not in seen:
            result.append(value)
            seen.add(value)
    return tuple(result)


@dataclass(frozen=True)
class InteractionSnapshot:
    sticky: tuple[int, ...]
    selected: tuple[int, ...]
    primary: int | None
    hidden: frozenset[int]
    transparent: frozenset[int]
    expanded: frozenset
    revision: int


class InteractionState:
    def __init__(self, *, history_limit=128):
        if not isinstance(history_limit, int) or history_limit < 2:
            raise ValueError("Selection history must retain at least two entries")
        self.history_limit = history_limit
        self.transparency_mode = TransparencyMode.NONE
        self.transparency_amount = .5  # UI amount: 0 opaque, 1 fully transparent.
        self.revision = 0
        self.begin_capture(None)

    def begin_capture(self, key):
        self.capture_key = key
        self.segment_epoch = None
        self.known = set()
        self.alive = set()
        self.sticky = ()
        self._sticky_members = set()
        self.hidden = set()
        self.transparent = set()
        self.expanded = set()
        self._children = {}
        self._owners = {}
        self._owner_inputs = {}
        self._scene_ids = set()
        self._history = [()]
        self._history_index = 0
        self._anchor = None
        self.revision += 1
        return self.snapshot

    @property
    def selected(self):
        return tuple(object_id for object_id in self.sticky if object_id in self.alive)

    @property
    def primary(self):
        selected = self.selected
        return selected[-1] if selected else None

    @property
    def snapshot(self):
        return InteractionSnapshot(self.sticky, self.selected, self.primary, frozenset(self.hidden),
                                   frozenset(self.transparent), frozenset(self.expanded), self.revision)

    @property
    def can_back(self):
        return self._history_index > 0

    @property
    def can_forward(self):
        return self._history_index + 1 < len(self._history)

    def update_frame(self, known_ids, alive_ids, *, segment_epoch=None, children=None, owners=None, scene_ids=None):
        """Apply authoritative identities/lifetimes, with optional ownership graph.

        children may be a mapping or a callback returning native child IDs on
        demand. It must encode ownership/child relationships, not arbitrary back
        references (joints, parents, references to other scenes). Cycles are safe.
        owners maps an object ID to every owning scene ID, including shared actors.
        Known IDs accumulate for the capture so dead-object intent is recoverable.
        """
        before = self.snapshot
        previous_alive = self.alive
        known, alive = set(_ids(known_ids)), set(_ids(alive_ids))
        if not alive <= known:
            raise ValueError("Alive IDs must be present in the native instance set")
        if (self.segment_epoch is not None and segment_epoch is not None
                and self.segment_epoch != segment_epoch):
            self.sticky = tuple(value for value in self.sticky if value > 0)
            self._sticky_members = set(self.sticky)
            self.hidden = {value for value in self.hidden if value > 0}
            self.transparent = {value for value in self.transparent if value > 0}
            self.known = {value for value in self.known if value > 0}
            history, position = [], 0
            for index, item in enumerate(self._history):
                item = tuple(value for value in item if value > 0)
                if not history or item != history[-1]:
                    history.append(item)
                if index <= self._history_index:
                    position = len(history) - 1
            self._history, self._history_index = history, position
            self._anchor = self._anchor if self._anchor is not None and self._anchor > 0 else None
        if segment_epoch is not None:
            self.segment_epoch = segment_epoch
        new_ids = bool(known - self.known)
        self.known.update(known)
        self.alive = alive
        if children is not None:
            self._children = children
        if owners is not None:
            normalized, immutable_inputs = {}, {}
            for key, value in owners.items():
                # Reuse only the very same immutable input already validated.
                # Equal values may have different types (True == 1); mutable
                # collections must be checked on every publication.
                immutable = type(value) is frozenset
                if immutable and value is self._owner_inputs.get(key):
                    normalized[key] = self._owners[key]
                else:
                    normalized[key] = frozenset(_ids(value))
                if immutable:
                    immutable_inputs[key] = value
            self._owners, self._owner_inputs = normalized, immutable_inputs
        if scene_ids is not None:
            self._scene_ids = set(_ids(scene_ids))
        after = self.snapshot
        if before != after or previous_alive != alive or new_ids or children is not None or owners is not None:
            self.revision += 1
        return self.snapshot

    def _record(self):
        if self._history[self._history_index] == self.sticky:
            return
        del self._history[self._history_index + 1:]
        self._history.append(self.sticky)
        if len(self._history) > self.history_limit:
            del self._history[:len(self._history) - self.history_limit]
        self._history_index = len(self._history) - 1

    def select(self, ids, *, record_history=True):
        """User selection is sticky; programmatic history/frame replay is not a new visit."""
        selected = tuple(value for value in _ids(ids) if value in self.known)
        if selected != self.sticky:
            self.sticky = selected
            self._sticky_members = set(selected)
            self.revision += 1
            if record_history:
                self._record()
        return self.snapshot

    def clear_selection(self, *, record_history=True, reset_history=False):
        self._anchor = None
        result = self.select((), record_history=record_history and not reset_history)
        if reset_history:
            self._history, self._history_index = [()], 0
        return result

    def pick(self, object_id, *, additive=False, toggle=False, extend=False, order=()):
        """Replace, Ctrl-toggle, or Shift-range in visible tree order.

        Empty viewport clicks clear selection unless a multi-select modifier is
        held. The range anchor stays at the last non-Shift pick. ``additive``
        combines a Shift range with the current visible selection.
        """
        if object_id is None or object_id in (0, -1):
            return self.snapshot if additive or toggle or extend else self.clear_selection()
        object_id, = _ids((object_id,))
        if object_id not in self.alive:
            return self.snapshot
        visible = tuple(value for value in _ids(order) if value in self.alive)
        if extend and self._anchor in visible and object_id in visible:
            start, stop = sorted((visible.index(self._anchor), visible.index(object_id)))
            interval = visible[start:stop + 1]
            chosen = tuple(dict.fromkeys((*self.selected, *interval))) if additive else interval
            # The clicked row becomes primary even for an upward Shift range.
            chosen = tuple(value for value in chosen if value != object_id) + (object_id,)
        else:
            self._anchor = object_id
            if toggle and object_id in self.selected:
                chosen = tuple(value for value in self.selected if value != object_id)
            elif additive or toggle:
                chosen = tuple(value for value in self.selected if value != object_id) + (object_id,)
            else:
                chosen = (object_id,)
        return self.select(chosen)

    def back(self):
        if self.can_back:
            self._history_index -= 1
            self.select(self._history[self._history_index], record_history=False)
            self._anchor = self.sticky[-1] if self.sticky else None
        return self.snapshot

    def forward(self):
        if self.can_forward:
            self._history_index += 1
            self.select(self._history[self._history_index], record_history=False)
            self._anchor = self.sticky[-1] if self.sticky else None
        return self.snapshot

    def set_expanded(self, key, expanded=True):
        before = key in self.expanded
        if expanded:
            self.expanded.add(key)
        else:
            self.expanded.discard(key)
        if before != bool(expanded):
            self.revision += 1
        return self.snapshot

    def _descendants(self, roots, *, stop_at_scenes=False):
        pending, visited = deque(_ids(roots)), set()
        while pending:
            value = pending.popleft()
            if value in visited or value not in self.known:
                continue
            visited.add(value)
            if stop_at_scenes and value in self._scene_ids:
                continue
            children = self._children(value) if callable(self._children) else self._children.get(value, ())
            pending.extend(child for child in _ids(children) if child not in visited)
        return visited

    def set_visible(self, ids, visible=True, *, cascade=True):
        changed = set(self.hidden)
        affected = self._descendants(ids, stop_at_scenes=True) if cascade else set(_ids(ids)) & self.known
        if visible:
            self.hidden.difference_update(affected)
        else:
            self.hidden.update(affected)
        if changed != self.hidden:
            self.revision += 1
        return self.snapshot

    def hide_selected(self):
        return self.set_visible(self.selected, False)

    def show_selected(self):
        return self.set_visible(self.selected, True)

    def show_all(self):
        if self.hidden:
            self.hidden.clear()
            self.revision += 1
        return self.snapshot

    def hide_all(self):
        if self.hidden != self.known:
            self.hidden = set(self.known)
            self.revision += 1
        return self.snapshot

    def isolate(self, ids=None):
        roots = self.selected if ids is None else tuple(value for value in _ids(ids) if value in self.known)
        if not roots:
            return self.snapshot
        keep = self._descendants(roots)
        # Scene isolation also works when the UI supplies only ownership rather
        # than eagerly populating every scene→actor tree branch.
        keep.update(value for value, owners in self._owners.items() if owners.intersection(roots))
        keep.update(owner for value in tuple(keep) for owner in self._owners.get(value, ()))
        hidden = self.known - keep
        if hidden != self.hidden:
            self.hidden = hidden
            self.revision += 1
        return self.snapshot

    def is_visible(self, object_id, *, owner_ids=None):
        if object_id not in self.alive or object_id in self.hidden:
            return False
        owners = self._owners.get(object_id, ()) if owner_ids is None else owner_ids
        return not owners or any(owner in self.alive and owner not in self.hidden for owner in owners)

    def set_transparent(self, ids, tagged=True, *, cascade=True):
        before = set(self.transparent)
        affected = self._descendants(ids) if cascade else set(_ids(ids)) & self.known
        if tagged:
            self.transparent.update(affected)
        else:
            self.transparent.difference_update(affected)
        if before != self.transparent:
            self.revision += 1
        return self.snapshot

    def tag_selected(self, tagged=True):
        return self.set_transparent(self.selected, tagged)

    def clear_transparency(self):
        if self.transparent:
            self.transparent.clear()
            self.revision += 1
        return self.snapshot

    def set_transparency(self, mode, amount=None):
        mode = TransparencyMode(mode)
        amount = self.transparency_amount if amount is None else float(amount)
        if not math.isfinite(amount) or not 0 <= amount <= 1:
            raise ValueError("Transparency amount must be between 0 and 1")
        if (mode, amount) != (self.transparency_mode, self.transparency_amount):
            self.transparency_mode, self.transparency_amount = mode, amount
            self.revision += 1
        return self.snapshot

    def alpha_for(self, object_id):
        mode = self.transparency_mode
        selected = object_id in self._sticky_members and object_id in self.alive
        transparent = (mode == TransparencyMode.ALL
                       or mode == TransparencyMode.SELECTED and selected
                       or mode == TransparencyMode.UNSELECTED and not selected
                       or mode == TransparencyMode.TAGGED and object_id in self.transparent)
        return min(.999, 1. - self.transparency_amount) if transparent else 1.

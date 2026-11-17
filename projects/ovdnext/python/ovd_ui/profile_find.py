# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Original Find criteria over one owned native profile result, never a file scan.

Qt searches its loaded event statistics, including hidden events. Native zone
identity is retained in addition to the original context/thread/event tuple.
Context captions use the snapshot's owned scene-name map, while queries and
selection retain exact IDs. This lookup is independent of the inline filter.
"""
from dataclasses import dataclass
import math
import re
from _pvd_native import CROSS_THREAD_ID

from .profiling import EventKey


CRITERIA = ("Name", "Context ID", "Thread ID", "Event ID")
# QChar::isSpace / QString::trimmed: Python additionally strips U+001C–001F,
# which Qt's integer conversion rejects. Keep the recorded query literal.
_QT_SPACE = "\t\n\v\f\r \x85\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000"


def unsigned_decimal(text):
    """QString::toULongLong's base-10 domain, without float or width truncation."""
    text = str(text).strip(_QT_SPACE)
    if not re.fullmatch(r"\+?[0-9]+", text):
        return None
    digits = text.lstrip("+").lstrip("0") or "0"
    if len(digits) > 20:
        return None
    value = int(digits)
    return value if value <= (1 << 64) - 1 else None


@dataclass(frozen=True)
class FoundProfileEvent:
    key: EventKey
    name: str
    context_name: str = ''

    @property
    def cells(self):
        key = self.key
        return (self.name, self.context_name or str(key.context_id),
                "Cross thread" if key.thread_id == CROSS_THREAD_ID else str(key.thread_id),
                str(key.event_id))


def find_profile_events(snapshot, expression, criterion=0):
    if criterion not in range(len(CRITERIA)):
        raise ValueError("Unknown profile search criterion")
    if snapshot is None or not snapshot.supported:
        return ()
    expression = str(expression)
    value = unsigned_decimal(expression) if criterion else None
    if criterion and value is None:
        return ()
    field = ("name", "context_id", "thread_id", "event_id")[criterion]
    names=getattr(snapshot,'context_names',{})
    return tuple(FoundProfileEvent(EventKey.of(row), row.name,names.get(row.context_id,''))
                 for row in snapshot.summaries
                 if (expression in row.name if criterion == 0 else getattr(row, field) == value))


def profile_find_sort_key(row, column=0):
    """Qt FindSortModel compares signed64 decimal cells before UTF-16 text."""
    value = row.cells[column]
    stripped = value.strip(_QT_SPACE)
    if re.fullmatch(r"[+-]?[0-9]+", stripped):
        digits = stripped.lstrip("+-").lstrip("0") or "0"
        if len(digits) <= 19:
            number = int(digits) * (-1 if stripped.startswith("-") else 1)
            if -(1 << 63) <= number < (1 << 63):
                return 0, number
    return 1, value.encode("utf-16-be", "surrogatepass")


def activate_profile_event(model, snapshot, key):
    """Select one native identity and reveal its interval; reject stale results.

Both lane modes use this same ProfileModel. Visibility tags are untouched.
Like Qt selectEvent/zoomTo, keep the zoom when the matching span occupies
10–90% of it, otherwise show twice that span, and scroll to its beginning.
All measurements come from the already-owned native scopes.
"""
    if snapshot is None or snapshot is not model.snapshot or not model.supported:
        return False
    if not any(EventKey.of(row) == key for row in snapshot.summaries):
        return False
    model.select([key])
    scopes = [scope for scope in snapshot.scopes if EventKey.of(scope) == key]
    if not scopes or not model.view_range or snapshot.duration_ms <= 0:
        return True
    # Qt's task selection deduplicates repeated start timestamps.
    starts = {}
    for scope in scopes:
        starts.setdefault(scope.start_tick, scope)
    first = min(scope.start_ms for scope in starts.values())
    last = max(scope.end_ms for scope in starts.values())
    duration = snapshot.duration_ms
    old_span = model.view_range[1] - model.view_range[0]
    span = last - first
    if not all(math.isfinite(value) for value in (first, last, duration, old_span)):
        return True
    width = old_span if .1 * old_span <= span <= .9 * old_span else 2 * span
    width = min(duration, max(duration / 1_000_000, width))
    first = min(duration - width, max(0., first))
    model.view_range = (first, first + width)
    model.marked_range = None
    return True

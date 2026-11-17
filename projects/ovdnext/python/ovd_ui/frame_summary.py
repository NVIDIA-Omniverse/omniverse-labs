# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Current published native-object counts, independent of tree presentation.

This repairs Qt's unpopulated lazy Frame summary, with an explicit narrower
scope: each alive positive native creation ID counts once. Synthetic queries,
contacts, reference aliases, UI groups and render visibility are not counts.
No Session seeks, property reads or capture parsing occur here.
"""
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class TypeCount:
    type_name: str
    count: int


def native_object_counts(adapter):
    """Read only the adapter's already-owned publication metadata."""
    if adapter is None:
        return ()
    counts = Counter(adapter.class_name(identity)
                     for identity, instance in adapter.instances.items()
                     if identity > 0 and instance.alive)
    return tuple(TypeCount(name, count) for name, count in sorted(counts.items()))


def sorted_counts(rows, column="type", descending=False):
    if column not in ("type", "count"):
        raise ValueError("Summary sort column must be type or count")
    # Numeric counts; alphabetical ties stay stable in either direction.
    if column == "count":
        return tuple(sorted(rows, key=lambda row: (-row.count if descending else row.count, row.type_name)))
    return tuple(sorted(rows, key=lambda row: row.type_name, reverse=descending))

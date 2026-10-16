# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-ID discovery and catalog matching, independent of USD and rendering."""
from .config import source_key


def source_ids(asset_info):
    """Read the sample's sourceIds.<issuing-system>.id convention."""
    entries = asset_info.get('sourceIds', {})
    if not isinstance(entries, dict):
        return []
    return [{'source': source, 'id': entry['id']}
            for source, entry in entries.items()
            if isinstance(entry, dict) and isinstance(entry.get('id'), str) and entry['id']]


def match_bindings(authored, catalog):
    """Return all prim placements for each known ID, plus visible unmatched IDs.

    authored contains (prim path, asset-info dictionary) pairs. Prims with no
    source ID are unbound and do not produce an unmatched diagnostic.
    """
    known = {source_key(entry) for entry in catalog}
    matched, unmatched = {}, []
    for path, asset_info in authored:
        for entry in source_ids(asset_info):
            key = source_key(entry)
            if key in known:
                matched.setdefault(key, []).append(path)
            else:
                unmatched.append(dict(entry, path=path, state='unmatched'))
    return matched, unmatched

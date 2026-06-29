---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
name: stage-queries
description: >
  Discovering prims on the runtime stage and inspecting their attribute schemas. Use
  when user asks to find prims by type, filter by attribute, list all prims, or look up
  attribute types before reading or writing them.
license: LicenseRef-NvidiaProprietary
version: "0.3.0"
author: NVIDIA ovrtx
tags:
  - ovrtx
  - usd
  - queries
tools:
  - Read
  - Grep
---

# Stage Queries

## When to Use

Use this skill when the user asks to find prims by type, filter by attribute, list all prims, or look up attribute types before reading or writing them.

## Inputs

Resolve inputs in this order: existing repository files and referenced snippets, explicit user request, then broader agent context.

- Target API surface: Python, C/C++, or both.
- Query goal: list all prims, filter by prim type, require/exclude attributes, or inspect attribute schemas before read/write.
- Filter sets: `require_all`, `require_any`, `exclude`, attribute filter mode, and whether names must be resolved through dictionaries in C.
- Desired execution mode: Python sync/async query or C async query.
- Repository source snippets referenced below. Treat these snippets as the API source of truth.

## Prerequisites

- Use an ovrtx checkout that contains the referenced examples and docs tests.
- Read the relevant `> **Source:**` snippet before writing or explaining API usage.
- Use this before attribute read/write skills when the target prim list or schema is unknown.

## Instructions

1. Identify whether the caller needs all prims, prims by type, prims with required attributes, or a filtered include/exclude query.
2. Read the matching Python or C query snippet before choosing filter fields or result traversal.
3. For async queries, preserve the two-phase wait/fetch lifecycle before reading result dictionaries or C result arrays.
4. In C, resolve path and attribute IDs through the path dictionary before presenting names to users.
5. When changing code, run the stage-query docs test that owns the snippet whenever practical.

## Output Format

- For explanations, cite the relevant API names, source snippets, and caveats.
- For code changes, summarize the files changed, snippets affected, and validation run.

## Scripts

This skill has no scripts.

## Limitations

- The referenced snippets remain the source of truth; update or add tested snippets before documenting new API usage.

## Overview

`query_prims` finds prims on the runtime stage that match a set of filters and returns them grouped by attribute schema. Each group carries a `prim_list_handle` that can be plugged directly into subsequent `read_attribute` / `write_attribute` calls — so a typical workflow is:

1. Query to discover prims and/or their attribute schemas.
2. Decide what to read/write based on the result.
3. Reuse the returned prim list to read or write without re-resolving paths.

Filters combine as **AND** (`require_all`) × **OR** (`require_any`) × **NOT** (`exclude`). Each filter is a `(kind, name)` pair where `kind` is `FilterKind.PRIM_TYPE` (match by USD type, e.g. `"Mesh"`, `"Camera"`) or `FilterKind.HAS_ATTRIBUTE` (match by attribute existence, e.g. `"points"`).

`AttributeFilterMode` controls how much attribute metadata the query returns:
- `NONE` — no attribute descriptors, lightweight prim discovery.
- `ALL` — every attribute on each matched prim.
- `SPECIFIC` — only the attributes named in `attribute_names`.

## Python

### Discover every prim (no attribute descriptors)

> **Source:** `tests/docs/python/test_stage_query.py` snippet `doc-query-prims-basic`

### Filter by USD prim type

> **Source:** `tests/docs/python/test_stage_query.py` snippet `doc-query-prims-by-type`

### Request specific attribute descriptors

`AttributeInfo` exposes `dtype`, `is_array`, and `semantic`. Relationship-valued attributes such as `material:binding` surface with `Semantic.PATH_ID` (a raw path handle); resolve through the renderer's path dictionary when you need the string form.

> **Source:** `tests/docs/python/test_stage_query.py` snippet `doc-query-prims-with-attributes`

### Combine OR, NOT, and attribute-reporting filters

> **Source:** `tests/docs/python/test_stage_query.py` snippet `doc-query-require-any-exclude`

### Async query

Queries follow the two-phase `Operation` / `PendingFetch` lifecycle: `.wait()` resolves once the enqueued work finishes; `.fetch()` retrieves the result dict.

> **Source:** `tests/docs/python/test_stage_query.py` snippet `doc-query-prims-async`

## C

### Basic query

> **Source:** `tests/docs/c/test_stage_query.cpp` snippet `doc-query-prims-basic-c`

### Filter by prim type

> **Source:** `tests/docs/c/test_stage_query.cpp` snippet `doc-query-prims-by-type-c`

### Filter by attribute existence

> **Source:** `tests/docs/c/test_stage_query.cpp` snippet `doc-query-has-attribute-c`

### Combine OR and NOT filters

> **Source:** `tests/docs/c/test_stage_query.cpp` snippet `doc-query-require-any-exclude-c`

### Empty specific-attribute lists

> **Source:** `tests/docs/c/test_stage_query.cpp` snippet `doc-query-specific-empty-attributes-c`

### Path dictionary round-trip

The renderer's path dictionary resolves `ovx_primpath_t` / `ovx_token_t` handles to strings and back. It is valid for the lifetime of the renderer — no release call is required.

> **Source:** `tests/docs/c/test_stage_query.cpp` snippet `doc-path-dictionary-resolve-c`

There is no public Python wrapper for `ovrtx_get_path_dictionary()` today; the Python `query_prims` already returns path strings directly (resolved internally).

## Key Types / Functions

| Python | C |
|--------|---|
| `renderer.query_prims(...)` | `ovrtx_query_prims(renderer, &desc, &query_handle)` + `ovrtx_fetch_query_results(...)` |
| `renderer.query_prims_async(...)` → `Operation[PendingFetch[...]]` | same C trio; wait + fetch are the two phases |
| `FilterKind.PRIM_TYPE` / `HAS_ATTRIBUTE` | `OVRTX_FILTER_PRIM_TYPE` / `OVRTX_FILTER_HAS_ATTRIBUTE` |
| `AttributeFilterMode.NONE` / `ALL` / `SPECIFIC` | `OVRTX_ATTRIBUTE_FILTER_NONE` / `ALL` / `SPECIFIC` |
| `AttributeInfo(name, dtype, is_array, semantic)` | `ovrtx_attribute_desc_t` |
| (not exposed) | `ovrtx_get_path_dictionary(renderer, &pd)` |

C result shape:
- `ovrtx_query_result_t.groups` — one `ovrtx_query_prim_group_t` per attribute-schema bucket.
- Each group's `prim_list_handle` is a persistent `ovx_primpath_list_t` that plugs into `ovrtx_binding_desc_t::prims_list_handle`.
- Attribute names are returned as tokens — resolve via the path dictionary's `get_strings_from_tokens`.

## Troubleshooting

- In C, the pointers in `ovrtx_query_result_t` and `ovrtx_query_prim_group_t` become invalid after `ovrtx_release_query_results()`. Copy anything you want to keep.
- `AttributeFilterMode.SPECIFIC` with an empty `attribute_names` list returns no descriptors. Use `AttributeFilterMode.ALL` to dump everything, or `NONE` for lightweight counting.
- Relationship-valued attributes (`rel material:binding = ...`) surface with `Semantic.PATH_ID`, not `PATH_STRING`. Resolve through the path dictionary if you need strings.
- Attribute names in query results are tokens — in Python these are pre-resolved to strings; in C, call `path_dictionary_get_strings_from_tokens` yourself.
- An empty filter (no `require_all` / `require_any` / `exclude`) matches every prim. Pair with `AttributeFilterMode.NONE` for the cheapest full-stage walk.

## References

- Use the `> **Source:**` directives in this skill to locate tested snippets before reusing API patterns.
- Keep related skills, docs, and snippets synchronized when changing the workflow.

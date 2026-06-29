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
name: picking-selection
description: >
  Viewport picking, marquee selection, pick-hit decoding, pickability, and selection
  outline drawing. Use when implementing click picking, drag selection, printing picked
  prim names, or highlighting selected prims.
license: LicenseRef-NvidiaProprietary
version: "0.3.0"
author: NVIDIA ovrtx
tags:
  - ovrtx
  - picking
  - selection
tools:
  - Read
  - Grep
---

# Picking and Selection

## When to Use

Use this skill when implementing click picking, drag selection, printing picked prim names, or highlighting selected prims.

## Inputs

Resolve inputs in this order: existing repository files and referenced snippets, explicit user request, then broader agent context.

- Target API surface: Python, C/C++, USD, or a combination.
- RenderProduct path and resolution, plus the UI click or marquee coordinates in RenderProduct pixel space.
- Desired pick result handling: path ID decoding, picked prim names, pickability filtering, or selected-path storage.
- Selection outline requirements: group IDs, colors, fill mode, and whether the picking RenderProduct must be pinned to CUDA-visible GPU 0.
- Repository source snippets referenced below. Treat these snippets as the API source of truth.

## Prerequisites

- Use an ovrtx checkout that contains the referenced examples and docs tests.
- Read the relevant `> **Source:**` snippet before writing or explaining API usage.
- Confirm the picking RenderProduct is pinned to CUDA-visible GPU 0; use `render-product-device-pinning` when that USD authoring is missing.
- Use `reading-render-output` only for general render-var readback; pick hit decoding follows this skill's snippets.

## Instructions

1. Identify whether the task is click picking, marquee selection, pickability filtering, path decoding, or selection-outline styling.
2. Ensure the RenderProduct used for picking is running on CUDA-visible GPU 0 before adding pick queries.
3. Read the matching Python or C snippet for enqueueing the pick query, mapping the hit buffer, and resolving prim path IDs.
4. Keep selection outline group assignment separate from pick-hit decoding; use the style/group snippets only after the selected paths are known.
5. When changing code, run the picking-selection docs test that owns the snippet whenever practical.

## Output Format

- For explanations, cite the relevant API names, source snippets, and caveats.
- For code changes, summarize the files changed, snippets affected, and validation run.

## Scripts

This skill has no scripts.

## Limitations

- Picking currently only works for RenderProducts running on CUDA-visible GPU 0.
- Outline dashing or stippling is not supported by the underlying RTX outline pipeline.
- The referenced snippets remain the source of truth; update or add tested snippets before documenting new API usage.

## Overview

Picking is a RenderProduct-space query consumed by the next renderer step. The
result is a synthetic render var named `ovrtx_pick_hit` / `OVRTX_RENDER_VAR_PICK_HIT`.
Each hit stores a path-dictionary id, not a string.

Current limitation: picking only works for RenderProducts running on
CUDA-visible GPU 0. In multi-GPU scenes, restrict picking RenderProducts to
CUDA-visible GPU 0 by authoring `uint[] deviceIds = [0]`. `deviceIds` is an
allow-list of indices into `CUDA_VISIBLE_DEVICES`; ovrtx may choose any
CUDA-visible GPU from the list.

Selection drawing is separate from picking. Enable the selection outline pass at
renderer creation, then write non-zero `omni:selectionOutlineGroup` values to the
prims that should be outlined. Write group `0` to clear an outline.

Selection styling has two layers:
- Global renderer-creation state controls outline width and fill mode.
- Runtime per-group state controls outline and fill colors. Prims opt into a
  style by writing that group's id to `omni:selectionOutlineGroup`.

Fill colors are visible only when the renderer is created with a fill mode that
uses per-group fill color, such as `GROUP_FILL_COLOR` /
`OVRTX_SELECTION_FILL_MODE_GROUP_FILL_COLOR`.

## Workflow

1. Load the stage and identify the RenderProduct path. For picking, ensure that RenderProduct is restricted to CUDA-visible GPU 0 in USD.
2. Convert UI/window input into RenderProduct pixel coordinates.
3. Queue a pick query before the next step. Use a 1x1 rectangle for click picking and a larger rectangle for marquee selection.
4. Step the renderer for the same RenderProduct.
5. Find and CPU-map the synthetic pick-hit render var.
6. Validate the `magic` / `version` params and read named tensors such as `primPath`.
7. Resolve each `primPath` id through the renderer path dictionary.
8. Deduplicate resolved paths, then print or otherwise report the picked prim names.
9. Optionally write group `0` to clear the previous selection and group `1` to outline the new selection.

## Python

### Create a renderer with outline drawing enabled

> **Source:** `tests/docs/python/test_picking_selection.py` snippet `doc-create-selection-outline-renderer-python`

### Enqueue a click or marquee pick

> **Source:** `tests/docs/python/test_picking_selection.py` snippet `doc-enqueue-pick-query-python`

### Decode the pick-hit buffer

> **Source:** `tests/docs/python/test_picking_selection.py` snippet `doc-read-pick-hit-buffer-python`

### Resolve picked prim names

> **Source:** `tests/docs/python/test_picking_selection.py` snippet `doc-resolve-picked-prim-paths-python`

After resolving path ids, deduplicate them before printing names or writing
selection outline groups. Print the resolved strings, not the numeric path ids.

### Mark prims pickable or unpickable

> **Source:** `tests/docs/python/test_picking_selection.py` snippet `doc-set-pickable-python`

### Draw selection outlines

> **Source:** `tests/docs/python/test_picking_selection.py` snippet `doc-set-selection-outline-group-python`

### Clear selection outlines

> **Source:** `tests/docs/python/test_picking_selection.py` snippet `doc-clear-selection-outline-group-python`

### Create a styled selection renderer

> **Source:** `tests/docs/python/test_picking_selection.py` snippet `doc-create-styled-selection-renderer-python`

Configure global style state at renderer creation:
- `selection_outline_width` controls outline thickness in pixels (`0..15`).
- `selection_fill_mode` controls whether selected prim interiors are filled and
  which color source they use.

Changing these global settings requires recreating the renderer.

### Set per-group selection styles

> **Source:** `tests/docs/python/test_picking_selection.py` snippet `doc-set-selection-group-styles-python`

Use `Renderer.set_selection_group_styles()` to assign outline and fill RGBA
colors to group ids. The operation is stream-ordered and affects subsequent
steps. Later writes to the same group id win.

### Assign styled groups to prims

> **Source:** `tests/docs/python/test_picking_selection.py` snippet `doc-assign-selection-style-groups-python`

Write `OVRTX_ATTR_NAME_SELECTION_OUTLINE_GROUP` per prim to choose which styled
selection group that prim uses. This is the per-prim part of selection styling.

## C

### Create a renderer with outline drawing enabled

> **Source:** `tests/docs/c/test_picking_selection.cpp` snippet `doc-create-selection-outline-renderer-c`

### Enqueue a click or marquee pick

> **Source:** `tests/docs/c/test_picking_selection.cpp` snippet `doc-enqueue-pick-query-c`

### Decode the pick-hit buffer

> **Source:** `tests/docs/c/test_picking_selection.cpp` snippet `doc-read-pick-hit-buffer-c`

### Resolve picked prim names

> **Source:** `tests/docs/c/test_picking_selection.cpp` snippet `doc-resolve-picked-prim-paths-c`

The helper used by the C snippet shows the path dictionary calls needed to turn
an `ovx_primpath_t` into a UTF-8 path.

> **Source:** `tests/docs/c/helpers.h` snippet `doc-resolve-primpath-helper-c`

### Mark prims pickable or unpickable

> **Source:** `tests/docs/c/test_picking_selection.cpp` snippet `doc-set-pickable-c`

### Draw selection outlines

> **Source:** `tests/docs/c/test_picking_selection.cpp` snippet `doc-set-selection-outline-group-c`

### Clear selection outlines

> **Source:** `tests/docs/c/test_picking_selection.cpp` snippet `doc-clear-selection-outline-group-c`

### Create a styled selection renderer

> **Source:** `tests/docs/c/test_picking_selection.cpp` snippet `doc-create-styled-selection-renderer-c`

Configure global style state at renderer creation with
`ovrtx_config_entry_selection_outline_width()` and
`ovrtx_config_entry_selection_fill_mode()`. Changing these global settings
requires recreating the renderer.

### Set per-group selection styles

> **Source:** `tests/docs/c/test_picking_selection.cpp` snippet `doc-set-selection-group-styles-c`

Use `ovrtx_set_selection_group_styles()` to assign outline and fill RGBA colors
to group ids. The operation is stream-ordered and affects subsequent steps.
Later writes to the same group id win.

### Assign styled groups to prims

> **Source:** `tests/docs/c/test_picking_selection.cpp` snippet `doc-assign-selection-style-groups-c`

Use `ovrtx_set_selection_outline_group()` to write each prim's
`omni:selectionOutlineGroup` value. The group id selects which per-group style
that prim uses.

## Key Types / Functions

| Python | C |
|--------|---|
| `Renderer.enqueue_pick_query()` | `ovrtx_enqueue_pick_query()` |
| `Renderer.resolve_prim_path_id()` | `ovrtx_get_path_dictionary()` plus path dictionary utilities |
| `RendererConfig(selection_outline_enabled=True)` | `ovrtx_config_entry_selection_outline_enabled(true)` |
| `RendererConfig(selection_outline_width=...)` | `ovrtx_config_entry_selection_outline_width(...)` |
| `RendererConfig(selection_fill_mode=SelectionFillMode.GROUP_FILL_COLOR)` | `ovrtx_config_entry_selection_fill_mode(OVRTX_SELECTION_FILL_MODE_GROUP_FILL_COLOR)` |
| `Renderer.set_selection_group_styles()` | `ovrtx_set_selection_group_styles()` |
| `SelectionGroupStyle` | `ovrtx_selection_group_style_t` |
| `Renderer.write_attribute(..., OVRTX_ATTR_NAME_SELECTION_OUTLINE_GROUP, uint8)` | `ovrtx_set_selection_outline_group()` |
| `Renderer.write_attribute(..., OVRTX_ATTR_NAME_PICKABLE, uint8)` | `ovrtx_set_pickable()` |
| `OVRTX_RENDER_VAR_PICK_HIT` | `OVRTX_RENDER_VAR_PICK_HIT` |
| `OVRTX_PICK_FLAG_GIZMO`, `OVRTX_PICK_FLAG_INCLUDE_TRACKED_INFO` | `OVRTX_PICK_FLAG_GIZMO`, `OVRTX_PICK_FLAG_INCLUDE_TRACKED_INFO` |

Pick-hit render-var layout:
- Params: `magic`, `version`, `hitCount`.
- Tensors: `primPath`, `objectType`, `geometryInstanceId`, `worldPositionM`, `worldNormal`.
- Validate `OVRTX_PICK_HIT_MAGIC` and `OVRTX_PICK_HIT_VERSION` before reading tensors.
- C path resolution needs `ovx/path_dictionary/path_dictionary_utils.h`; selection and pickability helpers need `ovrtx/ovrtx_attributes.h`.

## UI Integration Notes

- Convert from window or framebuffer coordinates to RenderProduct pixel coordinates before enqueueing a query. Do not pass raw window pixels unless the RenderProduct exactly matches the window.
- `left` and `top` are inclusive. `right` and `bottom` are exclusive, so a click uses `right = left + 1` and `bottom = top + 1`.
- Use a small drag threshold so tiny mouse movement is still treated as a click.
- For drag selection, clamp both endpoints to the RenderProduct extent, then use the min values as `left/top` and `max + 1` as `right/bottom`.
- Keep camera controls on a non-picking mouse button when adding click or marquee selection to an interactive viewport.

## Troubleshooting

- The pick rectangle is in RenderProduct pixel coordinates, not window coordinates. UI integrations must convert from window or framebuffer coordinates before enqueueing.
- Enqueue the pick query before the step that should produce pick results.
- Picking currently requires the target RenderProduct to run on CUDA-visible GPU 0. Use the `render-product-device-pinning` skill to author a CUDA-visible GPU 0-only `deviceIds` allow-list on picking RenderProducts.
- If multiple pick queries target the same RenderProduct before one step, the last query wins.
- The synthetic pick-hit render var appears only on a step that consumed a pick query.
- `ovrtx_pick_hit` can only be mapped on CPU/default. Always unmap the output and destroy the step result after use.
- Hit records contain path ids. Resolve them before printing names or writing selection groups.
- Selection drawing requires both renderer config and non-zero per-prim group ids. Write group `0` to prims from the previous selection before writing group `1` to the next selection.
- Selection fill color has no visible effect unless the renderer's fill mode uses per-group fill color.
- Outline dashing/stippling is not supported by the underlying RTX outline pipeline.

## Related Skills

- `render-product-device-pinning` for authoring `deviceIds` allow-lists on
  RenderProducts, including the CUDA-visible GPU 0-only list required by
  current picking support.

## References

- Use the `> **Source:**` directives in this skill to locate tested snippets before reusing API patterns.
- Keep related skills, docs, and snippets synchronized when changing the workflow.

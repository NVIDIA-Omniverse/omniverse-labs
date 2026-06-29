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
name: loading-usd
description: >
  Loading USD scenes into the renderer from files, URLs, or inline USDA strings. Use
  when user asks to load a USD scene, compose USD content, add cameras or
  RenderProducts to an existing USD layer, add referenced content, or create runtime
  geometry.
license: LicenseRef-NvidiaProprietary
version: "0.3.0"
author: NVIDIA ovrtx
tags:
  - ovrtx
  - usd
  - loading
tools:
  - Read
  - Grep
---

# Loading USD

## When to Use

Use this skill when the user asks to load a USD scene, compose USD content, add cameras or RenderProducts to an existing USD layer, add referenced content, or create runtime geometry.

## Inputs

Resolve inputs in this order: existing repository files and referenced snippets, explicit user request, then broader agent context.

- Target API surface: Python, C/C++, or both.
- Application lifecycle stage: renderer creation, scene loading, stepping, warmup, output readback, or cleanup.
- Repository source snippets referenced below. Treat these snippets as the API source of truth.

## Prerequisites

- Use an ovrtx checkout that contains the referenced examples and docs tests.
- Read the relevant `> **Source:**` snippet before writing or explaining API usage.
- For code changes, preserve renderer lifecycle ordering and cleanup semantics for the selected language.

## Instructions

1. Identify the requested language and lifecycle stage before choosing an example.
2. Read the referenced snippet that matches the requested stage and language.
3. Preserve the normal ovrtx order: create or initialize the renderer, load or compose USD, step or wait for work, read outputs when needed, then release C resources explicitly.
4. Apply the async, status-query, error-handling, and warmup skills when the workflow crosses those concerns.
5. When changing code, run the narrow example or docs test that owns the snippet whenever practical.

## Output Format

- For explanations, cite the relevant API names, source snippets, and caveats.
- For code changes, summarize the files changed, snippets affected, and validation run.

## Scripts

This skill has no scripts.

## Limitations

- The referenced snippets remain the source of truth; update or add tested snippets before documenting new API usage.

## Overview

Before rendering, you must load USD content into the renderer. ovrtx supports three input modes:

1. **File path or URL** -- open an existing `.usd`/`.usda`/`.usdc` file as the root layer.
2. **Inline USDA string** -- open runtime-generated USD content as the root layer. The inline root layer may use USD `subLayers` to compose a base scene and additional authored prims.
3. **USD references** -- compose additional file or string content under a specific path prefix.

There are two common composition patterns:

- **Inline root with `subLayers`** -- use when an existing scene layer does not contain required prims such as Cameras, RenderProducts, or RenderVars. Build one inline root USDA layer that sublayers the original scene and authors the extra prims, then load it with `open_usd_from_string()` / `ovrtx_open_usd_from_string()`.
- **Additive references** -- use when a root stage is already open and you need to add removable referenced content at a new prim path. Use `add_usd_reference()` / `add_usd_reference_from_string()`.

## Python

### Open from file or URL

> **Source:** `examples/python/minimal/main.py` snippet `add-usd`

### Open from inline USDA string

Useful for creating RenderProducts, cameras, or runtime geometry without editing the original scene:

> **Source:** `tests/docs/python/test_sensor_configuration.py` snippet `doc-add-render-config-layer`

### Compose a base scene plus extra prims

When a USD layer has the scene content but lacks render configuration or sensors, compose a new inline root layer: add `subLayers = [@existing_scene.usda@]`, author the missing Camera / RenderProduct / RenderVar prims in that same inline layer, and load it with `open_usd_from_string()`.

> **USDA source:** `tests/docs/usd/data/inline_sublayers_camera_renderproduct.usda` snippet `doc-usda-inline-sublayers-camera-renderproduct`
>
> **Source:** `tests/docs/python/test_sensor_configuration.py` snippet `doc-add-render-config-layer`
>
> **Query check:** `tests/docs/python/test_stage_query.py` snippet `doc-query-inline-sublayer-composition`

### Add a USD reference with a path prefix

Use `add_usd_reference()` or `add_usd_reference_from_string()` after a root stage is open when you need additive content that can be removed later by handle.

### Compose multiple inputs

Use inline `subLayers` for one composed root stage, or reference APIs for incremental additions. See the planet-system example for the reference-addition pattern.

### Remove USD

Use `remove_usd(handle)` with handles returned by `add_usd_reference()` / `add_usd_reference_from_string()`. Root layers opened with `open_usd*` are replaced by the next `open_usd*` call or cleared with `reset_stage()`.

## C

### Open from file or URL

> **Source:** `examples/c/minimal/main.cpp` snippet `load-usd-and-wait`

### Poll for completion

Loading is asynchronous in C. Poll until done:

> **Source:** `examples/c/minimal/main.cpp` snippet `load-usd-and-wait`

Or block indefinitely:

> **Source:** `examples/c/minimal/main.cpp` snippet `step-renderer`
>
> The step snippet demonstrates blocking with `ovrtx_timeout_infinite`.

### Add a USD reference with a path prefix

> Use `ovrtx_add_usd_reference_from_file` or `ovrtx_add_usd_reference_from_string` with a path prefix.

### Open from inline USDA content

> **Source:** `tests/docs/c/test_sensor_configuration.cpp` snippet `doc-add-render-config-layer-c`
>
> Use `ovrtx_open_usd_from_file` for files/URLs and `ovrtx_open_usd_from_string` for inline root USDA, including inline roots with `subLayers`.

### Compose a base scene plus extra prims

When a USD layer has the scene content but lacks render configuration or sensors, pass one inline root USDA string to `ovrtx_open_usd_from_string()`. That inline root can sublayer the existing scene and author missing Camera / RenderProduct / RenderVar prims.

> **USDA source:** `tests/docs/usd/data/inline_sublayers_camera_renderproduct.usda` snippet `doc-usda-inline-sublayers-camera-renderproduct`
>
> **Source:** `tests/docs/c/test_sensor_configuration.cpp` snippet `doc-add-render-config-layer-c`

### Remove USD

> C: `ovrtx_enqueue_result_t result = ovrtx_remove_usd(renderer, usd_handle);`

### Update time-sampled attributes

For animated USD scenes, re-evaluate every time-sampled attribute at a specific stage time. Time is in **seconds**; the runtime converts to USD timecodes via the stage's `timeCodesPerSecond` metadata.

> **Source:** `tests/docs/python/test_base.py` snippet `doc-update-from-usd-time-async`

### Reset stage to empty

Clear all USD content from the runtime stage:

> Python: `renderer.reset_stage()` / `renderer.reset_stage_async()`
>
> C: `ovrtx_reset_stage(renderer)`

## Key Types / Functions

| Python | C |
|--------|---|
| `renderer.open_usd(path)` | `ovrtx_open_usd_from_file(renderer, file)` |
| `renderer.open_usd_from_string(usda)` | `ovrtx_open_usd_from_string(renderer, content)` |
| `renderer.add_usd_reference(path, prefix)` | `ovrtx_add_usd_reference_from_file(renderer, file, prefix, &handle)` |
| `renderer.add_usd_reference_from_string(usda, prefix)` | `ovrtx_add_usd_reference_from_string(renderer, content, prefix, &handle)` |
| `renderer.remove_usd(handle)` | `ovrtx_remove_usd(renderer, handle)` |
| `renderer.update_from_usd_time(usd_time)` | `ovrtx_update_stage_from_usd_time(renderer, usd_time)` |
| `renderer.reset_stage()` / `reset_stage_async()` | `ovrtx_reset_stage(renderer)` |

## Troubleshooting

- **Only one root layer is allowed.** Calling `open_usd()` followed by `open_usd_from_string()` (or vice versa) replaces the root stage. To combine a scene file with extra prims such as Cameras and RenderProducts, use a single `open_usd_from_string()` root layer with `subLayers`:

  > **Source:** `tests/docs/python/test_sensor_configuration.py` snippet `doc-add-render-config-layer`

- **Reference additions are a separate pattern.** Use `add_usd_reference*` when a root stage is already open and you want to place additional referenced content at a new absolute path. The `prefix_path` must not collide with existing prim paths, and inline reference layers must set `defaultPrim`.
- Authored attributes that are not part of the normal population schema are ignored unless the root layer sets `customLayerData.populateAllAuthoredAttributes = true`. Use this only for workflows that need generic authored attributes, because populating every authored attribute can dramatically increase memory usage when assets contain many properties the runtime will never read or write.
- In Python, `open_usd()` blocks until loaded. Use the `_async` variants if you need non-blocking behavior.
- In C, `ovrtx_open_usd_from_file()` is always asynchronous -- you must poll or wait.
- Load errors (e.g., file not found) are reported through `ovrtx_op_wait_result_t::error_op_ids`, not the immediate return value.

## References

- Use the `> **Source:**` directives in this skill to locate tested snippets before reusing API patterns.
- Keep related skills, docs, and snippets synchronized when changing the workflow.

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
name: attribute-bindings
description: >
  Creating persistent attribute bindings for efficient repeated writes to the same
  prims and attribute. Use when user asks about persistent bindings, repeated writes,
  efficient animation loops, bind_attribute, or updating transforms every frame with
  caller-owned tensors. Use mapping-attributes when the hot path needs zero-copy
  direct writes into ovrtx buffers.
license: LicenseRef-NvidiaProprietary
version: "0.3.0"
author: NVIDIA ovrtx
tags:
  - ovrtx
  - attributes
  - bindings
tools:
  - Read
  - Grep
---

# Attribute Bindings

## When to Use

Use this skill when the user asks about persistent bindings, repeated writes, efficient animation loops, `bind_attribute`, or updating transforms every frame with caller-owned data. Use `mapping-attributes` instead when the user specifically needs zero-copy direct access to ovrtx internal buffers, CUDA/Warp kernels writing in place, or map/unmap lifetimes.

## Inputs

Resolve inputs in this order: existing repository files and referenced snippets, explicit user request, then broader agent context.

- Target API surface: Python, C/C++, or both.
- Prim paths, attribute name, element type, semantic conversion, and whether the attribute is scalar or array-valued.
- Repeated-write cadence, sync/async behavior, caller-owned data location, and whether CUDA stream/event synchronization is needed.
- Whether the user needs binding writes, binding maps, or one-shot writes.
- Repository source snippets referenced below. Treat these snippets as the API source of truth.

## Prerequisites

- Use an ovrtx checkout that contains the referenced examples and docs tests.
- Read the relevant `> **Source:**` snippet before writing or explaining API usage.
- Use `writing-attributes` for one-shot or infrequent writes where descriptor creation cost does not matter.
- Use `mapping-attributes` for zero-copy direct buffer access; this skill can still apply when creating a persistent binding first and then mapping through it repeatedly.

## Instructions

1. Identify whether the user needs repeated `write()` calls, repeated `write_async()` calls, or repeated `map()` calls through a stable binding.
2. Read the matching source snippet and copy its create/use/destroy lifecycle rather than inventing equivalent calls.
3. Validate prim list, attribute name, dtype, shape, semantic, and array/scalar rules before proposing or editing code.
4. Keep the binding alive across repeated updates, and explicitly destroy or unbind it when the hot path ends.
5. Choose `mapping-attributes` when avoiding the data copy matters more than avoiding descriptor recreation.
6. When changing code, run the narrow docs test or example that owns the snippet whenever practical.

## Output Format

- For explanations, cite the relevant API names, source snippets, and caveats.
- For code changes, summarize the files changed, snippets affected, and validation run.

## Scripts

This skill has no scripts.

## Limitations

- The referenced snippets remain the source of truth; update or add tested snippets before documenting new API usage.

## Overview

When writing the same attribute to the same set of prims every frame (e.g., animating transforms), creating a persistent binding avoids the overhead of rebuilding the binding descriptor each time. Create the binding once, then call `write()` or `map()` on it repeatedly. For "update transforms every frame", start here unless the request explicitly calls for zero-copy direct buffer writes or GPU kernels that operate on ovrtx-owned memory.

## Python

### Direct write (no binding)

For one-shot or infrequent writes, `renderer.write_attribute()` rebuilds the binding descriptor each call:

> **Source:** `tests/docs/python/test_attribute_bindings.py` snippet `doc-write-attribute-async-data-access`

### Create binding and write

For repeated writes to the same attribute/prims, create a persistent binding once:

> **Source:** `tests/docs/python/test_attribute_bindings.py` snippet `doc-bind-attribute-write`

### Reuse binding for repeated writes

> **Source:** `tests/docs/python/test_attribute_bindings.py` snippet `doc-binding-write-async`

### Array attribute binding

> **Source:** `tests/docs/python/test_attribute_bindings.py` snippet `doc-bind-array-attribute`

### Binding with explicit semantic

> **Source:** `tests/docs/python/test_attribute_bindings.py` snippet `doc-bind-attribute-write`

### Use binding for mapping (zero-copy)

> **Source:** `tests/docs/python/test_attribute_bindings.py` snippet `doc-map-bound-attribute`

## C

### Create and use a persistent binding

> **Source:** `tests/docs/c/test_attribute_bindings.cpp` snippets `doc-create-attribute-binding-c`, `doc-write-bound-attribute-c`, `doc-destroy-attribute-binding-c`

## Key Types / Functions

| Python | C |
|--------|---|
| `renderer.bind_attribute(...)` | `ovrtx_create_attribute_binding(renderer, &desc, &handle)` |
| `renderer.bind_array_attribute(...)` | same, with `is_array=true` in desc |
| `binding.write(data, data_access=DataAccess.SYNC, cuda_stream=, cuda_event=)` | `ovrtx_write_attribute(renderer, &binding_ref, &buffer, access)` |
| `binding.map(device=...)` | `ovrtx_map_attribute(renderer, &binding_ref, mapping_desc, &out)` |
| `binding.unbind()` | `ovrtx_destroy_attribute_binding(renderer, handle)` |

Binding flags (C only, set via `desc.flags`):
- `OVRTX_BINDING_FLAG_NONE` -- default
- `OVRTX_BINDING_FLAG_OPTIMIZE` -- optimize internal structures for frequent high-volume writes

## Troubleshooting

- **Tensor lifetime:** When using `binding.map()`, the returned `mapping.tensor` is only valid inside the `with` block (Python) or before `unmap` (C). All tensor access **must** happen while the mapping is active — accessing it after unmap is undefined behavior.
- The canonical transform attribute name is `"omni:xform"`. The legacy name `"omni:fabric:localMatrix"` (used in examples above) is also accepted. New code should prefer `"omni:xform"`.
- Always call `unbind()` (Python) or `ovrtx_destroy_attribute_binding()` (C) when done. In Python, the binding auto-unbinds on garbage collection, but explicit cleanup is preferred.
- The binding locks in the prim list, attribute name, and element type. You cannot change these after creation.
- In C, `OVRTX_BINDING_FLAG_OPTIMIZE` should be used for the primary hot-path binding. The last binding created with this flag takes priority.
- In Python, `bind_attribute` is synchronous (blocks until the binding is ready). Use `bind_attribute_async` for non-blocking creation.

## References

- Use the `> **Source:**` directives in this skill to locate tested snippets before reusing API patterns.
- Keep related skills, docs, and snippets synchronized when changing the workflow.

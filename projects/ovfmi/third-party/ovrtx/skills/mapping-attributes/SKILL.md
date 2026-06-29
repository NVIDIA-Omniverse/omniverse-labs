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
name: mapping-attributes
description: >
  Zero-copy attribute map/unmap for direct memory access to ovrtx internal buffers.
  Use when user asks about zero-copy writes, map attribute, direct memory access,
  Warp/CUDA kernel writes into mapped tensors, or GPU attribute updates. Use
  attribute-bindings for repeated writes with caller-owned tensors when a copy is
  acceptable.
license: LicenseRef-NvidiaProprietary
version: "0.3.0"
author: NVIDIA ovrtx
tags:
  - ovrtx
  - attributes
  - mapping
tools:
  - Read
  - Grep
---

# Mapping Attributes

## When to Use

Use this skill when the user asks about zero-copy writes, `map_attribute`, direct memory access, Warp/CUDA kernel writes into mapped tensors, or GPU attribute updates. Use `attribute-bindings` instead for repeated writes to the same prims/attribute when the caller already owns the data tensor and a copy into ovrtx is acceptable.

## Inputs

Resolve inputs in this order: existing repository files and referenced snippets, explicit user request, then broader agent context.

- Target API surface: Python, C/C++, or both.
- Prim paths, attribute name, element type, semantic conversion, and whether the attribute is mappable.
- Mapping target: CPU tensor, linear CUDA memory, or a bound attribute reused for repeated map/unmap cycles.
- Synchronization and lifetime requirements: stream/event, map duration, whether data must outlive the mapping, and whether array attributes are involved.
- Repository source snippets referenced below. Treat these snippets as the API source of truth.

## Prerequisites

- Use an ovrtx checkout that contains the referenced examples and docs tests.
- Read the relevant `> **Source:**` snippet before writing or explaining API usage.
- Array attributes such as `float3[] points` are not mappable; use `writing-attributes` or `attribute-bindings` for array writes.
- Use `attribute-bindings` first when the user wants repeated updates but does not need zero-copy direct buffer access.

## Instructions

1. Identify the concrete map/unmap target, language, prim list, attribute name, memory target, and synchronization requirement.
2. Read the matching source snippet and copy its map/write/unmap lifecycle rather than inventing equivalent calls.
3. Validate dtype, shape, semantic, mappability, and ownership rules before proposing or editing code.
4. Keep mapped tensor views alive only until unmap, and copy anything that must outlive the mapping.
5. For repeated map/unmap cycles, create a persistent binding first and map through it to avoid recreating the descriptor.
6. When changing code, run the narrow docs test or example that owns the snippet whenever practical.

## Output Format

- For explanations, cite the relevant API names, source snippets, and caveats.
- For code changes, summarize the files changed, snippets affected, and validation run.

## Scripts

This skill has no scripts.

## Limitations

- Array attributes such as `float3[] points` are not supported for map/unmap because their lengths vary per prim.
- The referenced snippets remain the source of truth; update or add tested snippets before documenting new API usage.

## Overview

Mapping gives you direct access to RTX's internal Fabric buffer for an attribute. Instead of copying data in (like `write_attribute`), you write directly into the buffer, then unmap to apply changes. This is the most efficient path for per-frame updates, especially with GPU compute (e.g., Warp kernels).

The pattern is: **map -> write into tensor -> unmap**.

## Python

### CPU mapping with NumPy

> **Source:** `tests/docs/python/test_attribute_bindings.py` snippet `doc-map-attribute-cpu`

### Context manager (recommended)

> **Source:** `tests/docs/python/test_attribute_bindings.py` snippet `doc-map-attribute-cpu`

### GPU mapping with Warp

From the planet-system example -- map on CUDA, compute with a Warp kernel, unmap with stream sync:

> **Source:** `tests/docs/python/test_attribute_bindings.py` snippet `doc-map-attribute-cuda`

### Using a persistent binding for mapping

More efficient for repeated map/unmap cycles (avoids recreating the binding descriptor):

> **Source:** `tests/docs/python/test_attribute_bindings.py` snippet `doc-map-bound-attribute`

## C

### Map, write, unmap

> **Source:** `tests/docs/c/test_attribute_bindings.cpp` snippet `doc-map-attribute-cpu-c`

### GPU mapping with CUDA sync

> **Source:** `examples/c/vulkan-interop/src/main.cpp` snippet `map-rendered-output-cuda-array`

## Key Types / Functions

| Python | C |
|--------|---|
| `renderer.map_attribute(...)` | `ovrtx_map_attribute(renderer, &binding, desc, &out)` |
| `renderer.unmap_attribute(mapping)` | `ovrtx_unmap_attribute(renderer, handle, sync)` |
| `binding.map(device=...)` | same, with binding handle |
| `mapping.tensor` | `out_mapping.dl` (DLTensor) |
| `mapping.unmap(stream=...)` | `ovrtx_unmap_attribute` with `cuda_sync.stream` |
| `mapping.unmap(event=...)` | `ovrtx_unmap_attribute` with `cuda_sync.wait_event` |
| `renderer.unmap_attribute_async(mapping, ...)` | `ovrtx_unmap_attribute` (always async in C) |

Tensor layout follows the API-specific attribute rules (see the `writing-attributes` skill for the full table):
- Python mapping tensors are NumPy-style: `dtype="float64", shape=(4, 4)` returns a tensor with shape `(N, 4, 4)` and scalar lanes.
- C mapping tensors are lane-based: a 4x4 double matrix attribute maps as `shape=[N]`, `dtype={kDLFloat, 64, 16}`.
- The Python `semantic=` path reshapes automatically: `Semantic.XFORM_MAT4x4` → `(N, 4, 4)`, points → `(N, 3)`, colors → `(N, 4)` for RGBA or `(N, 3)` for RGB.

> **Source:** `tests/docs/python/test_attribute_shapes.py` snippets `doc-shape-scalar-int32`, `doc-shape-float3-array`, `doc-shape-mat4-array`.

## Troubleshooting

- **Tensor lifetime:** The tensor from `mapping.tensor` is only valid while the mapping is active (inside the `with` block in Python, or before `unmap` in C). All reads, writes, and kernel launches that access the tensor **must** happen before the mapping is released. Accessing it after the `with` block exits is undefined behavior. If you need data to outlive the mapping, copy it while still inside the block.
- The canonical transform attribute name is `"omni:xform"`. The legacy name `"omni:fabric:localMatrix"` (used in examples above) is also accepted. New code should prefer `"omni:xform"`.
- Array attributes (e.g., `float3[] points`) are **not supported** for mapping because they have variable lengths per prim. Use `write_array_attribute()` instead.
- Data must be fully written before calling `unmap()`. For CUDA, pass `stream` or `event` so ovrtx knows when the GPU write is complete.
- `event` and `stream` are mutually exclusive on unmap -- use one or the other.
- Providing `event`/`stream` for a CPU-mapped attribute raises an error.
- Multiple mappings can be outstanding on the same attribute. The logical order of application depends on the order of `unmap()` calls.

## References

- Use the `> **Source:**` directives in this skill to locate tested snippets before reusing API patterns.
- Keep related skills, docs, and snippets synchronized when changing the workflow.

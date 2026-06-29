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
name: reading-attributes
description: >
  Reading scalar or array attributes from prims into CPU or GPU tensors. Use when user
  asks to read an attribute value, fetch mesh data (points, faceVertexCounts, etc.),
  inspect a render setting, or sample transforms.
license: LicenseRef-NvidiaProprietary
version: "0.3.0"
author: NVIDIA ovrtx
tags:
  - ovrtx
  - attributes
  - reading
tools:
  - Read
  - Grep
---

# Reading Attributes

## When to Use

Use this skill when the user asks to read an attribute value, fetch mesh data (points, faceVertexCounts, etc.), inspect a render setting, or sample transforms.

## Inputs

Resolve inputs in this order: existing repository files and referenced snippets, explicit user request, then broader agent context.

- Target API surface: Python, C/C++, or both.
- Prim paths or prim-list handle, attribute name, expected USD value type, and scalar versus array read mode.
- Target consumer: CPU NumPy, C DLTensor, GPU-aware DLPack consumer, or metadata/schema inspection.
- Whether the caller needs raw storage, shape/dtype discovery, or values copied out for later use.
- Repository source snippets referenced below. Treat these snippets as the API source of truth.

## Prerequisites

- Use an ovrtx checkout that contains the referenced examples and docs tests.
- Read the relevant `> **Source:**` snippet before writing or explaining API usage.
- Use `stage-queries` first if the user needs to discover prims or attribute schemas before reading values.

## Instructions

1. Identify the target language, prim paths or prim-list handle, attribute name, scalar/array mode, memory target, and sync/async requirement.
2. Read the matching source snippet and copy its lifecycle pattern rather than inventing equivalent calls.
3. Validate dtype, shape, semantic, and ownership rules before proposing or editing code.
4. Keep bindings, mapped buffers, operation handles, and C strings alive only for the lifetimes documented by the referenced examples.
5. When changing code, run the narrow docs test or example that owns the snippet whenever practical.

## Output Format

- For explanations, cite the relevant API names, source snippets, and caveats.
- For code changes, summarize the files changed, snippets affected, and validation run.

## Scripts

This skill has no scripts.

## Limitations

- Runtime string arrays, asset arrays, and timecode attributes are not supported.
- Read semantics must be `OVRTX_SEMANTIC_NONE`; semantic conversion is write-side only.
- The referenced snippets remain the source of truth; update or add tested snippets before documenting new API usage.

## Overview

`read_attribute` returns **one value per prim** (a scalar read). `read_array_attribute` returns a **variable-length array per prim**. Both produce DLPack tensors — NumPy, Warp, PyTorch, and anything else that speaks DLPack can consume them without a copy.

The read sees the stage as-if all prior stream-ordered operations have completed, so a write followed immediately by a read is safe.

For mapping (zero-copy writes into ovrtx's internal buffer), see the `mapping-attributes` skill. For queueing work instead of blocking, every `read_*` method has an `_async` twin that returns an `Operation` — see the `async-operations` skill.

## USD Type Lookup

Use this table to answer "how do I read USD type X?" For array-valued USD attributes, use `read_array_attribute` in Python and set `binding.binding_desc.attribute_type.is_array = true` in C. Python exposes shape dimensions; C uses `DLDataType::lanes`.

| USD type(s) | Python read result | C binding dtype | Notes |
|---|---|---|---|
| `bool` / `bool[]` | `(N,)` or `(M,)`, `np.bool_` | `{kDLBool, 8, 1}` | |
| `uchar` / `uchar[]` | `np.uint8` | `{kDLUInt, 8, 1}` | |
| `int`, `int2`, `int3`, `int4` and arrays | `np.int32`, trailing shape `()`, `(2,)`, `(3,)`, `(4,)` | `{kDLInt, 32, lanes}` | |
| `uint` / `uint[]` | `np.uint32` | `{kDLUInt, 32, 1}` | |
| `int64` / `int64[]` | `np.int64` | `{kDLInt, 64, 1}` | |
| `uint64` / `uint64[]` | `np.uint64` | `{kDLUInt, 64, 1}` | |
| `half`, `half2`, `half3`, `half4` and arrays | `np.float16`, trailing shape by component count | `{kDLFloat, 16, lanes}` | |
| `float`, `float2`, `float3`, `float4` and arrays | `np.float32`, trailing shape by component count | `{kDLFloat, 32, lanes}` | Authored scalar USD `float3` currently populates as zero; prefer role-bearing `point3f`, `normal3f`, `vector3f`, or `color3f` for values read from USD. |
| `double`, `double2`, `double3`, `double4` and arrays | `np.float64`, trailing shape by component count | `{kDLFloat, 64, lanes}` | |
| `point3*`, `normal3*`, `vector3*`, `color3*`, `color4*`, `texCoord2f` and arrays | numeric dtype by suffix (`h/f/d`), trailing role dimensions | same numeric dtype/lanes as storage | Roles are not surfaced as tensor metadata; they still matter for USD population and schema intent. |
| `quat*` and arrays | numeric dtype by suffix, shape `(N, 4)` or `(M, 4)` | `{kDLFloat, bits, 4}` | Runtime component order is `(i, j, k, real)`, while USDA authoring order is `(real, i, j, k)`. |
| `matrix2d`, `matrix3d`, `matrix4d`, `frame4d` and arrays | `np.float64`, flattened trailing shape `(4,)`, `(9,)`, `(16,)`, `(16,)` | `{kDLFloat, 64, 4/9/16}` | Generic authored matrix attrs are flattened. Transform-specific APIs/snippets may reshape 4x4 xforms to `(N, 4, 4)`. |
| `extent`, `_worldExtent` | `np.float64`, shape `(N, 6)` | `{kDLFloat, 64, 6}` | `extent` is local-space; `_worldExtent` is world-space. |
| `string` | `uint8` byte array, decode as UTF-8 | `{kDLUInt, 8, 1}` with `is_array=true` | Scalar USD strings are represented as byte arrays. This is not `string[]`; string arrays are not supported. Use `token[]` for string-like arrays. |
| `token` / `token[]` | raw `uint64` token IDs | `{kDLUInt, 64, 1}` | Python reads return raw IDs; C can resolve them through the path dictionary. Python has no public path-dictionary resolver for attribute values today. |
| `asset` | `uint64` lanes/columns of 2: token plus second lane | `{kDLUInt, 64, 2}` | Scalar asset only. Asset arrays are not supported. Python reads return the raw token pair; C can resolve the token through the path dictionary. |
| `relationship` | only supported schema relationships | path IDs / path-list semantics | Custom relationships are ignored today. For material bindings and shader connections, use the relationship-specific skills/snippets rather than generic custom relationship reads. |
| `timecode` / `timecode[]` | unsupported | unsupported | |

## Python

### Scalar read

Returns a `ManagedDLTensor` with shape `(N,)` for N input prims. Convert with `np.from_dlpack()` for a zero-copy numpy view.

> **Source:** `tests/docs/python/test_attribute_read.py` snippet `doc-read-attribute-scalar`

### Scalar read into a pre-allocated destination

Pass `dest=` with a DLPack-compatible tensor (NumPy array, Warp array, etc.). The read writes directly into `dest`; the returned tensor aliases the same memory. The `dest` dtype must match how the runtime stores the attribute.

> **Source:** `tests/docs/python/test_attribute_read.py` snippet `doc-read-attribute-dest-tensor`

### Array read

Returns a `dict[prim_path, ManagedDLTensor]`. Iteration order matches the input `prim_paths`.

> **Source:** `tests/docs/python/test_attribute_read.py` snippet `doc-read-array-attribute`

### Async read

`read_attribute_async` returns an `Operation`; call `.wait()` then `.fetch()` to pull the `ManagedDLTensor` out.

> **Source:** `tests/docs/python/test_attribute_read.py` snippet `doc-read-attribute-async`

### GPU destination (CUDA)

Allocate the destination on the GPU via any DLPack-compatible allocator (e.g. Warp). Pass the CUDA stream handle so the read is stream-ordered with your GPU work.

> **Source:** `tests/docs/python/test_attribute_read.py` snippet `doc-read-attribute-cuda-dest`

### Supported authored attribute read snippets

Python raw snippets:

| Type/pattern | Snippet |
|---|---|
| `bool` | `doc-read-usd-bool` |
| `int` | `doc-read-usd-int` |
| `float` | `doc-read-usd-float` |
| `point3f` | `doc-read-usd-point3f` |
| `point3f[]` | `doc-read-usd-point3f-array` |
| `normal3f` | `doc-read-usd-normal3f` |
| `vector3f` | `doc-read-usd-vector3f` |
| `color3f` | `doc-read-usd-color3f` |
| `matrix4d` | `doc-read-usd-matrix4d` |
| `quatf` | `doc-read-usd-quatf` |
| `string` | `doc-read-usd-string` |

The snippets listed in this table live in `tests/docs/python/test_all_attributes.py`.

No Python raw snippets are provided for token or asset reads yet. `read_attribute` returns raw numeric token IDs for token payloads and raw `(token, 0)` pairs for scalar asset payloads, and there is no public Python wrapper for resolving those payload IDs back to strings today. `query_prims` resolves prim paths and attribute names internally, but token-valued attribute payloads remain raw IDs.

### Local and world-space extents

`extent` is the authored local-space extent. `_worldExtent` is populated as the transformed world-space extent.

> **Source:** `tests/docs/python/test_all_attributes.py` snippet `doc-extent-world-extent`

## C

### Scalar read

> **Source:** `tests/docs/c/test_attribute_read.cpp` snippet `doc-read-attribute-scalar-c`

### Array read

Set `binding.binding_desc.attribute_type.is_array = true` — `ovrtx_make_binding_desc` defaults it to `false`.

> **Source:** `tests/docs/c/test_attribute_read.cpp` snippet `doc-read-array-attribute-c`

### Supported authored attribute read snippets in C

C raw snippets:

| Type/pattern | Snippet |
|---|---|
| scalar numeric | `doc-read-usd-float-c` |
| lane-3 scalar | `doc-read-usd-point3f-c` |
| lane-3 array | `doc-read-usd-point3f-array-c` |
| lane-16 matrix | `doc-read-usd-matrix4d-c` |
| quaternion | `doc-read-usd-quatf-c` |
| token | `doc-read-usd-token-c` |
| token array | `doc-read-usd-token-array-c` |
| string bytes | `doc-read-usd-string-c` |
| scalar asset token pair | `doc-read-usd-asset-c` |

The snippets listed in this table live in `tests/docs/c/test_all_attributes.cpp`.

C token and asset snippets include path-dictionary resolution because the C API exposes `ovrtx_get_path_dictionary()`.

### Local and world-space extents in C

> **Source:** `tests/docs/c/test_all_attributes.cpp` snippet `doc-extent-world-extent-c`

## Key Types / Functions

| Python | C |
|--------|---|
| `renderer.read_attribute(name, prim_paths, dest=, cuda_stream=, cuda_event=)` → `ManagedDLTensor` | `ovrtx_read_attribute(renderer, &binding, &read_dest, &read_handle)` + `ovrtx_fetch_read_result(...)` + `ovrtx_release_read_result(...)` |
| `renderer.read_array_attribute(name, prim_paths)` → `dict[path, ManagedDLTensor]` | same trio with `attribute_type.is_array = true` |
| `renderer.read_attribute_async(...)` / `read_array_attribute_async(...)` → `Operation[PendingFetch[...]]` | same trio; wait + fetch are the two phases |
| `PrimMode.EXISTING_ONLY` / `MUST_EXIST` | `OVRTX_BINDING_PRIM_MODE_EXISTING_ONLY` / `MUST_EXIST` |

C result layout (`ovrtx_read_output_t`):
- Scalar reads: `buffer_count == 1`, single tensor with shape `[prim_count]`. Multi-component values use `dtype.lanes` (for example, a 4x4 double matrix is `shape=[prim_count]`, `dtype={kDLFloat, 64, 16}`).
- Array reads: `buffer_count == prim_count`, one tensor per prim (variable length). Multi-component array elements also use `dtype.lanes` (for example, a `point3f[]` with 10 points is `shape=[10]`, `dtype={kDLFloat, 32, 3}`).
- When a caller-supplied `read_dest` tensor was passed in: `buffer_count == 0` (data landed in your tensor).
- Python reshapes lane-based C attribute tensors into NumPy-style trailing dimensions before exposing them through `ManagedDLTensor`.

## Troubleshooting

- **Semantics are write-only.** `ovrtx_read_attribute` rejects any semantic other than `OVRTX_SEMANTIC_NONE` with `"Semantic conversion is not supported for read_attribute"`. Always use `NONE` on the read binding and ask for the raw storage dtype (e.g. `{kDLFloat, 64, 16}` for a 4x4 matrix).
- **Generic authored USD attributes require opt-in population.** Root-layer `customLayerData.populateAllAuthoredAttributes = true` asks the runtime to populate authored attributes beyond the normal schema set. Use it only when needed: populating everything can dramatically increase memory usage on assets with many unused properties. See `loading-usd` for the layer metadata tradeoff.
- **Schema-owned attributes fix the element type.** `omni:rtx:rtpt:maxBounces` is stored as `uint32` even if you wrote it as `int32`. When allocating a `dest` tensor, match the runtime's dtype (`np.uint32`) — not what you wrote.
- **`PrimMode.EXISTING_ONLY`** (default) silently skips prims that don't exist on the stage. Use `MUST_EXIST` if you want an error instead. `CREATE_NEW` is not supported for reads.
- **Release in C, rely on GC in Python.** `ovrtx_release_read_result` must be called after `ovrtx_fetch_read_result`. In Python `ManagedDLTensor` releases automatically when the tensor (and every DLPack consumer of it) is dropped.
- **Array reads need `is_array=true` on the binding.** The `ovrtx_make_binding_desc` helper defaults `is_array=false`; set it explicitly after the call for array attributes.
- **C binding descriptors borrow path storage.** `ovrtx_make_binding_desc` stores the `ovx_string_t*` prim path array you pass; it does not copy it. Keep the `ovx_string_t` objects and array alive until the read has been enqueued, waited, fetched, and released. For persistent bindings, keep them alive until `ovrtx_create_attribute_binding` finishes.
- **Unsupported authored types are tested as absent.** `tests/docs/data/all-attributes.usda` deliberately authors `string[]`, `asset[]`, custom relationships, `timecode`, and `timecode[]`; the all-attributes tests assert they are not populated by the current runtime.

## Related skills

- `stage-queries` — discover prims and their attribute schemas before reading.
- `writing-attributes` — write values that can then be read back.
- `mapping-attributes` — zero-copy mapping (no fetch step).
- `async-operations` — polling, timeouts, the two-phase `Operation`/`PendingFetch` lifecycle.

## References

- Use the `> **Source:**` directives in this skill to locate tested snippets before reusing API patterns.
- Keep related skills, docs, and snippets synchronized when changing the workflow.

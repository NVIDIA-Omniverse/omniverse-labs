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
name: writing-attributes
description: >
  Writing scalar and array attribute data to prims. Use when user asks to write an
  attribute, set a property, change a material, set a color, or modify mesh data.
license: LicenseRef-NvidiaProprietary
version: "0.3.0"
author: NVIDIA ovrtx
tags:
  - ovrtx
  - attributes
  - writing
tools:
  - Read
  - Grep
---

# Writing Attributes

## When to Use

Use this skill when the user asks to write an attribute, set a property, change a material, set a color, or modify mesh data.

## Inputs

Resolve inputs in this order: existing repository files and referenced snippets, explicit user request, then broader agent context.

- Target API surface: Python, C/C++, or both.
- Prim paths, attribute name, USD value type, scalar versus array shape, and desired values.
- Data source location, dtype, semantic conversion, and sync/async behavior when those details are part of the task.
- Whether the write is one-shot, repeated with a stable target, or needs direct mapped-buffer access.
- Repository source snippets referenced below. Treat these snippets as the API source of truth.

## Prerequisites

- Use an ovrtx checkout that contains the referenced examples and docs tests.
- Read the relevant `> **Source:**` snippet before writing or explaining API usage.
- Use `attribute-bindings` for repeated writes to the same prims/attribute, and `mapping-attributes` for zero-copy direct writes.
- Use `binding-materials` when the request is specifically to set `material:binding`.

## Instructions

1. Identify the target language, prim paths, attribute name, value kind, data shape, memory target, and sync/async requirement.
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
- Use `token[]` for string-like arrays.
- The referenced snippets remain the source of truth; update or add tested snippets before documenting new API usage.

## Overview

Beyond transforms, you can write arbitrary attributes to prims: colors, visibility, mesh geometry, string tokens, and more. ovrtx uses DLTensor as the data format, and the dtype must exactly match the USD attribute schema.

There are two categories:
- **Scalar attributes** -- one value per prim (e.g., a color, a transform)
- **Array attributes** -- variable-length per prim (e.g., mesh points, face vertex counts)

## Tensor layout

Python and C expose different DLTensor layouts for multi-component attributes:

### Python — NumPy-style shapes

Python accepts and returns NumPy-style shapes with `dtype.lanes=1`. Multi-component values live in trailing shape dimensions:

| USD type | Python shape | dtype |
|---|---|---|
| `int` / `float` scalar for N prims | `(N,)` | `int32` / `float32` |
| `float3` / `point3f` scalar for N prims | `(N, 3)` | `float32` |
| `float4` / `color4f` scalar for N prims | `(N, 4)` | `float32` |
| generic `matrix4d` attribute for N prims | `(N, 16)` | `float64` |
| 4x4 transform semantic for N prims | `(N, 4, 4)` | `float64` |
| `int[]` array with M elements on one prim | `(M,)` | `int32` |
| `float3[]` / `point3f[]` with M elements on one prim | `(M, 3)` | `float32` |

> **Source:** `tests/docs/python/test_attribute_shapes.py` snippets `doc-shape-scalar-int32`, `doc-shape-float3-array`, `doc-shape-mat4-array`.

### C — lane-based attributes

The C API uses `DLDataType::lanes` for multi-component attribute reads and writes. The shape counts logical attribute elements; the lane count holds the vector or matrix component count:

| USD type | C DLTensor shape | C `DLDataType` |
|---|---|---|
| `int` / `float` scalar for N prims | `[N]` | `{kDLInt/kDLFloat, bits, 1}` |
| `float3` / `point3f` scalar for N prims | `[N]` | `{kDLFloat, 32, 3}` |
| 4x4 double matrix for N prims | `[N]` | `{kDLFloat, 64, 16}` |
| `int[]` array with M elements on one prim | `[M]` | `{kDLInt, 32, 1}` |
| `float3[]` / `point3f[]` with M elements on one prim | `[M]` | `{kDLFloat, 32, 3}` |

> **Source:** `tests/docs/c/test_attribute_shapes.cpp` snippets `doc-shape-scalar-int32-c`, `doc-shape-float3-array-c`, `doc-shape-mat4-array-c`.

For a C `point3f[]` attribute with 10 points, write or read one tensor with `shape=[10]` and `dtype={kDLFloat, 32, 3}`. Rendered output/AOV tensors are not attribute tensors; in C they use channel-last shapes such as `[height, width, channels]` with `dtype.lanes=1`.

## USD Type Lookup

Use this table to answer "how do I write USD type X?" For array-valued USD attributes, use `write_array_attribute` in Python and set `binding.binding_desc.attribute_type.is_array = true` in C. Python uses normal shape dimensions; C uses `DLDataType::lanes`.

| USD type(s) | Python value to write | C tensor dtype | Notes |
|---|---|---|---|
| `bool` / `bool[]` | `np.bool_`, shape `(N,)` or per-prim `(M,)` | `{kDLBool, 8, 1}` | |
| `uchar` / `uchar[]` | `np.uint8` | `{kDLUInt, 8, 1}` | |
| `int`, `int2`, `int3`, `int4` and arrays | `np.int32`, trailing shape `()`, `(2,)`, `(3,)`, `(4,)` | `{kDLInt, 32, lanes}` | |
| `uint` / `uint[]` | `np.uint32` | `{kDLUInt, 32, 1}` | |
| `int64` / `int64[]` | `np.int64` | `{kDLInt, 64, 1}` | |
| `uint64` / `uint64[]` | `np.uint64` | `{kDLUInt, 64, 1}` | |
| `half`, `half2`, `half3`, `half4` and arrays | `np.float16`, trailing shape by component count | `{kDLFloat, 16, lanes}` | |
| `float`, `float2`, `float3`, `float4` and arrays | `np.float32`, trailing shape by component count | `{kDLFloat, 32, lanes}` | Direct runtime writes to scalar `float3` work, but authored scalar USD `float3` population is bugged in the current runtime. Prefer role-bearing `point3f`, `normal3f`, `vector3f`, or `color3f` for USD-authored data. |
| `double`, `double2`, `double3`, `double4` and arrays | `np.float64`, trailing shape by component count | `{kDLFloat, 64, lanes}` | |
| `point3*`, `normal3*`, `vector3*`, `color3*`, `color4*`, `texCoord2f` and arrays | numeric dtype by suffix (`h/f/d`), trailing role dimensions | same numeric dtype/lanes as storage | Roles are schema intent; tensor storage is numeric. |
| `quat*` and arrays | numeric dtype by suffix, shape `(N, 4)` or `(M, 4)` | `{kDLFloat, bits, 4}` | Write runtime order `(i, j, k, real)`, not USDA order `(real, i, j, k)`. |
| `matrix2d`, `matrix3d`, `matrix4d`, `frame4d` and arrays | `np.float64`, flattened trailing shape `(4,)`, `(9,)`, `(16,)`, `(16,)` | `{kDLFloat, 64, 4/9/16}` | Generic authored matrix attrs are flattened. Transform semantic writes may use `(N, 4, 4)` Python arrays or `{kDLFloat,64,16}` C tensors. |
| `extent`, `_worldExtent` | `np.float64`, shape `(N, 6)` | `{kDLFloat, 64, 6}` | Usually read-only from population; `extent` is local-space, `_worldExtent` is world-space. |
| `string` | UTF-8 `np.uint8` byte array via `write_array_attribute` | `{kDLUInt, 8, 1}` with `is_array=true` | String arrays are not supported. Use `token[]` for string-like arrays. |
| `token` / `token[]` | Python strings; pass `is_token=True` for `token[]` | `{kDLUInt, 64, 1}` raw IDs with `OVRTX_SEMANTIC_TOKEN_ID`, or string helpers with `OVRTX_SEMANTIC_TOKEN_STRING` | |
| `asset` | `np.uint64` shape `(N, 2)` token pair | `{kDLUInt, 64, 2}` | Scalar asset only. Asset arrays are not supported. Python has no public helper to create asset tokens from strings today; C can use the path dictionary. |
| `relationship` | supported relationship helpers, usually path arrays | path string/path ID semantics | Custom relationships are ignored today. For material bindings and shader connections, use the relationship-specific helpers/snippets rather than generic custom relationship writes. |
| `timecode` / `timecode[]` | unsupported | unsupported | |

## Python

### Array attribute write (mesh points)

> **Source:** `tests/docs/python/test_attribute_shapes.py` snippet `doc-shape-float3-array`

### Array attribute write (same pattern for other array schemas)

> **Source:** `tests/docs/python/test_attribute_shapes.py` snippet `doc-shape-float3-array`

### Prim modes

> **Source:** `tests/docs/python/test_attribute_bindings.py` snippet `doc-bind-attribute-write`

### Token array attribute

> **Source:** `tests/docs/python/test_attribute_bindings.py` snippet `doc-write-token-array`

### Path/relationship array attribute

> **Source:** `tests/docs/python/test_base.py` snippet `doc-bind-material`

### Supported authored attribute write snippets

Python raw snippets:

| Type/pattern | Snippet |
|---|---|
| `bool` | `doc-write-usd-bool` |
| `int` | `doc-write-usd-int` |
| `float` | `doc-write-usd-float` |
| `point3f` | `doc-write-usd-point3f` |
| `point3f[]` | `doc-write-usd-point3f-array` |
| `normal3f` | `doc-write-usd-normal3f` |
| `vector3f` | `doc-write-usd-vector3f` |
| `color3f` | `doc-write-usd-color3f` |
| `matrix4d` | `doc-write-usd-matrix4d` |
| `quatf` | `doc-write-usd-quatf` |
| `string` | `doc-write-usd-string` |
| `token` | `doc-write-usd-token` |
| `token[]` | `doc-write-usd-token-array` |

The snippets listed in this table live in `tests/docs/python/test_all_attributes.py`.

Python can write token attributes directly from strings. For scalar asset attributes, Python can write the raw `(token, 0)` pair only if you already have a valid token ID; there is no public Python path-dictionary token-creation helper for asset strings today. The C snippet below shows the fully supported path-dictionary flow.

## C

### Generic scalar write in C

> **Source:** `tests/docs/c/test_attribute_bindings.cpp` snippet `doc-write-bound-attribute-c`

### Supported authored attribute write snippets in C

C raw snippets:

| Type/pattern | Snippet |
|---|---|
| scalar numeric | `doc-write-usd-float-c` |
| lane-3 scalar | `doc-write-usd-point3f-c` |
| lane-3 array | `doc-write-usd-point3f-array-c` |
| lane-16 matrix | `doc-write-usd-matrix4d-c` |
| quaternion | `doc-write-usd-quatf-c` |
| token | `doc-write-usd-token-c` |
| token array | `doc-write-usd-token-array-c` |
| string bytes | `doc-write-usd-string-c` |
| scalar asset token pair | `doc-write-usd-asset-c` |

The snippets listed in this table live in `tests/docs/c/test_all_attributes.cpp`.

## Key Types / Functions

| Python | C |
|--------|---|
| `renderer.write_attribute(...)` | `ovrtx_write_attribute(renderer, &binding, &buffer, access)` → returns `ovrtx_enqueue_result_t` |
| `renderer.write_array_attribute(..., is_token=True)` | same, with `is_array=true` in binding desc |

Semantics (Python: `from ovrtx import Semantic`; C: `ovrtx_attribute_semantic_t`):
- `Semantic.NONE` / `OVRTX_SEMANTIC_NONE` -- generic data. In Python, element shape comes from trailing tensor dimensions. In C, element component count comes from `DLDataType::lanes`. Use for colors, visibility, and other non-structured attributes.
- `Semantic.XFORM_MAT4x4` / `OVRTX_SEMANTIC_XFORM_MAT4x4` -- 4x4 double matrix
- `OVRTX_SEMANTIC_XFORM_POS3d_ROT4f_SCALE3f` -- decomposed transform (C only)
- `OVRTX_SEMANTIC_XFORM_POS3d_ROT3x3f` -- decomposed transform (C only)
- `Semantic.PATH_STRING` / `OVRTX_SEMANTIC_PATH_STRING` -- prim path strings
- `Semantic.TOKEN_STRING` / `OVRTX_SEMANTIC_TOKEN_STRING` -- token strings
- `Semantic.TOKEN_ID` / `OVRTX_SEMANTIC_TOKEN_ID` -- raw token IDs that are already present in the runtime path dictionary. In C, create them with `path_dictionary_create_tokens_from_strings()` before writing raw token arrays.

In C, `ovrtx_write_attribute` returns `ovrtx_enqueue_result_t` which contains both `.status` (check for `OVRTX_API_ERROR`) and `.op_index` (for async tracking via `ovrtx_wait_op`).

Data access modes (Python: `from ovrtx import DataAccess`):
- `DataAccess.SYNC` -- copies data during the call, safe to free after return
- `DataAccess.ASYNC` -- data accessed later during stream execution, must keep alive; pass `cuda_stream=` or `cuda_event=` for GPU synchronization. Not allowed with string data.

## Troubleshooting

- Array attribute dtype must exactly match the USD schema. Using numpy's default `float64` for a `float3[]` attribute (which expects `float32`) will cause errors.
- In the current runtime, authored scalar USD `float3` values may be created but populated as zero by `populateAllAuthoredAttributes`. If a value needs to come from USD, author it as a role-bearing type such as `vector3f`, `point3f`, `normal3f`, or `color3f`. Direct runtime writes to scalar `float3` still work.
- Quaternion tensors use ovrtx/Fabric lane order `(i, j, k, real)`. USDA `quat*` values are authored as `(real, i, j, k)`, so reading `quatd`, `quatf`, or `quath` attributes reorders the components into `(i, j, k, real)`, and writes should use that runtime tensor order.
- String attributes (`Semantic.PATH_STRING`, `Semantic.TOKEN_STRING`) only support synchronous data access (`DataAccess.SYNC`).
- String arrays are not supported as runtime attributes. If a workflow needs an array of string-like labels or categories, author and write a `token[]` instead and pass `is_token=True` when writing `list[list[str]]` data.
- Asset arrays are not supported as runtime attributes. Scalar asset attributes may be represented internally, but there is no general string-list replacement for `asset[]`.
- Timecode attributes are not supported as runtime attributes, including both scalar `timecode` and `timecode[]`.
- Custom relationships are not populated by the generic authored-attribute path. Specific relationships used by supported schemas, such as `material:binding` and shader connections, are handled by their schema/population code paths; arbitrary custom relationships are ignored today.
- Unsupported authored types are covered by negative tests in `tests/docs/python/test_all_attributes.py` and `tests/docs/c/test_all_attributes.cpp`; if one starts populating, keep this documentation and those tests in sync.
- For array attributes in Python, pass a list of tensors (one per prim), not a single tensor. NumPy arrays, Warp arrays, and any `__dlpack__`-compatible objects are accepted directly.
- When creating attributes from scratch (`prim_mode=PrimMode.CREATE_NEW`), `semantic=` is required for structured types (transforms, packed structs) so the C library knows what USD attribute type to create.
- For `write_array_attribute` with `list[list[str]]` data, strings default to path/relationship arrays. Set `is_token=True` to write token arrays for custom token-list attributes.
- `ovrtx_make_binding_desc` borrows the C prim path array. Keep any `ovx_string_t` prim paths and the array passed to the binding descriptor alive until the write operation has completed, or until `ovrtx_create_attribute_binding` finishes for persistent bindings.
- `dirty_bits` is a bitvector with 1 bit per prim -- the byte array size must be `(prim_count + 7) / 8`.
- In C, `dirty_bits` support three combination modes via `ovrtx_write_bits_t` in the `ovrtx_input_buffer_t.dirty_bits_mode` field: `OVRTX_DIRTY_MASK_REPLACE` (default -- replace existing mask), `OVRTX_DIRTY_MASK_OR` (merge with existing), `OVRTX_DIRTY_MASK_AND` (intersect with existing).

C convenience helpers for string attributes (`#include <ovrtx/ovrtx_attributes.h>`):
- `ovrtx_set_path_attributes(renderer, paths, count, attr_name, path_values)` -- write path/relationship attributes. Each prim gets a single-element array (relationships are always arrays in USD).
- `ovrtx_set_token_attributes(renderer, paths, count, attr_name, token_values)` -- write token string attributes (one per prim).

> **Source:** `tests/docs/c/test_attribute_helpers.cpp` snippet `doc-set-token-attributes-c`

## References

- Use the `> **Source:**` directives in this skill to locate tested snippets before reusing API patterns.
- Keep related skills, docs, and snippets synchronized when changing the workflow.

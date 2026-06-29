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
name: cloning-prims
description: >
  Cloning USD subtrees to create copies at new paths. Use when user asks to clone,
  duplicate, copy a prim, or create instances of existing geometry.
license: LicenseRef-NvidiaProprietary
version: "0.3.0"
author: NVIDIA ovrtx
tags:
  - ovrtx
  - usd
  - prims
tools:
  - Read
  - Grep
---

# Cloning Prims

## When to Use

Use this skill when the user asks to clone, duplicate, copy a prim, or create instances of existing geometry.

## Inputs

Resolve inputs in this order: existing repository files and referenced snippets, explicit user request, then broader agent context.

- Target API surface: Python, C/C++, or both.
- Source prim path to clone and one or more destination prim paths to create.
- Desired execution mode: Python sync, Python async, or C async clone.
- Whether the request is actually a clone, USD reference, instance, or new-geometry authoring workflow.
- Repository source snippets referenced below. Treat these snippets as the API source of truth.

## Prerequisites

- Use an ovrtx checkout that contains the referenced examples and docs tests.
- Read the relevant `> **Source:**` snippet before writing or explaining API usage.
- Confirm the source prim already exists on the runtime stage and destination paths do not conflict with existing prims unless replacement is intended.

## Instructions

1. Identify the source prim path and every destination path the caller wants to create.
2. Confirm whether the workflow needs a synchronous Python clone, Python async clone, or C async clone.
3. Read the matching snippet and preserve its source-path, destination-list, wait, and error-checking pattern.
4. Do not use cloning when the caller actually needs USD instancing, references, or authoring new geometry from scratch; route to the loading or writing skills instead.
5. When changing code, run the stage-mutation docs test that owns the clone snippet whenever practical.

## Output Format

- For explanations, cite the relevant API names, source snippets, and caveats.
- For code changes, summarize the files changed, snippets affected, and validation run.

## Scripts

This skill has no scripts.

## Limitations

- The referenced snippets remain the source of truth; update or add tested snippets before documenting new API usage.

## Overview

`clone_usd` creates copies of an existing prim subtree at one or more new target paths in the runtime stage. This is useful for duplicating geometry, creating arrays of objects, or spawning instances from a template.

## Python

### Clone to a single target

> **Source:** `tests/docs/python/test_stage_mutation.py` snippet `doc-clone-usd`

### Clone to multiple targets

> **Source:** `tests/docs/python/test_stage_mutation.py` snippet `doc-clone-usd`

### Async clone

> **Source:** `tests/docs/python/test_stage_mutation.py` snippet `doc-clone-usd-async`

## C

### Clone to multiple targets

> **Source:** `tests/docs/c/test_stage_mutation.cpp` snippet `doc-clone-usd-c`

### Wait for completion

> **Source:** `tests/docs/c/test_stage_mutation.cpp` snippet `doc-clone-usd-c`

## Key Types / Functions

| Python | C |
|--------|---|
| `renderer.clone_usd(source, targets)` | `ovrtx_clone_usd(renderer, source, targets, count)` |
| `renderer.clone_usd_async(source, targets)` | `ovrtx_clone_usd()` (always async in C) |

## Troubleshooting

- The source path **must exist** in the stage.
- The target paths **must not already exist** in the stage.
- Cloning copies the entire subtree under the source path, including all children.
- In Python, `clone_usd()` blocks until the operation completes. Use `clone_usd_async()` for non-blocking behavior.
- In C, `ovrtx_clone_usd()` is always asynchronous. You **must** wait on the returned `op_index` with `ovrtx_wait_op()` before using the cloned prims (e.g., writing attributes or stepping).

## References

- Use the `> **Source:**` directives in this skill to locate tested snippets before reusing API patterns.
- Keep related skills, docs, and snippets synchronized when changing the workflow.

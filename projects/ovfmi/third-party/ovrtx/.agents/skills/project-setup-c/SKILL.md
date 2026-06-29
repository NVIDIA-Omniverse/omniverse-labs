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
name: project-setup-c
description: >
  Setting up a new CMake C/C++ project that uses ovrtx. Use when user asks to create a
  new C project, set up CMake with ovrtx, scaffold a C++ app, or configure build
  dependencies.
license: LicenseRef-NvidiaProprietary
version: "0.3.0"
author: NVIDIA ovrtx
tags:
  - ovrtx
  - c
  - setup
tools:
  - Read
  - Grep
  - Shell
  - Write
---

# Project Setup (C)

## When to Use

Use this skill when the user asks to create a new C project, set up CMake with ovrtx, scaffold a C++ app, or configure build dependencies.

## Inputs

Resolve inputs in this order: existing repository files and referenced snippets, explicit user request, then broader agent context.

- Project language, package manager, build system, and target platform from the user request.
- Whether the project should use packaged ovrtx or the local dev-mode build.
- Repository source snippets and example projects referenced below. Treat these snippets as the API source of truth.

## Prerequisites

- Use an ovrtx checkout that contains the referenced examples and docs tests.
- Confirm Python, CMake, compiler, and package-index requirements before giving setup commands.
- Avoid modifying tracked dependency files unless the user explicitly asks for project scaffolding changes.

## Instructions

1. Identify the requested project type, platform, package source, and expected run command.
2. Read the minimal example and setup snippets before writing scaffolding.
3. Use the repository's existing Python or CMake conventions instead of inventing a new layout.
4. Include renderer creation and first-frame validation only after the project dependency setup is complete.
5. When changing code, run the narrow example setup or docs test whenever practical.

## Output Format

- For explanations, cite the relevant API names, source snippets, and caveats.
- For code changes, summarize the files changed, snippets affected, and validation run.

## Scripts

This skill has no scripts.

## Limitations

- The referenced snippets remain the source of truth; update or add tested snippets before documenting new API usage.

## Overview

ovrtx provides a C API with a CMake config for easy integration. The recommended approach uses CMake FetchContent to download the ovrtx binary package from GitHub Releases. A convenience macro in `ovrtx.cmake` handles fetching, finding, and runtime setup.

## Project Structure

```
my-ovrtx-app/
  CMakeLists.txt
  cmake/
    ovrtx.cmake       # Copy from examples/c/cmake/ovrtx.cmake
  main.cpp
```

## CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.15)
project(my-ovrtx-app)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Fetch ovrtx library
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
include(ovrtx)
ovrtx_fetch()

add_executable(my-ovrtx-app main.cpp)
target_link_libraries(my-ovrtx-app PRIVATE ovrtx::ovrtx)

# Setup runtime dependencies (rpath on Linux, DLL copying on Windows)
ovrtx_setup_runtime(my-ovrtx-app)
```

## ovrtx.cmake

Copy `examples/c/cmake/ovrtx.cmake` into your project's `cmake/` directory. The key macro it provides:

- `ovrtx_fetch()` -- downloads the ovrtx package via FetchContent and makes `ovrtx::ovrtx` available.
- `ovrtx_setup_runtime(TARGET)` -- configures rpath (Linux) or copies DLLs and creates junctions (Windows) so the executable can find ovrtx runtime dependencies.

Update the `FetchContent_Declare` URL inside `ovrtx.cmake` to point to the appropriate GitHub Releases package for your platform.

## Minimal main.cpp

> **Source:** `examples/c/minimal/main.cpp` snippet `check-error-helper`
>
> Followed by: `examples/c/minimal/main.cpp` snippet `create-renderer`
>
> Followed by: `examples/c/minimal/main.cpp` snippet `load-usd-and-wait`
>
> See the full minimal example for the complete flow including step, fetch, map, and cleanup.

## Build and Run

```bash
cmake -B build -G Ninja
cmake --build build
./build/my-ovrtx-app
```

## Runtime Packaging

The ovrtx binary distribution includes runtime dependencies under `bin/`:

```
bin/
  libovrtx-dynamic.so / ovrtx-dynamic.dll
  cache/
  library/
  libs/
  mdl/
  plugins/
  rendering-data/
  usd_plugins/
```

`ovrtx_setup_runtime()` configures rpath (Linux) or copies DLLs + creates junctions (Windows) so the executable can find these at runtime.

With dynamic linking, ovrtx expects these runtime directories to be available next to `ovrtx-dynamic.dll` / `libovrtx-dynamic.so` under the same `bin/` layout.

Set `binary_package_root_path` only when:
- static linking ovrtx, or
- your install/deploy layout breaks apart the default `bin/` directory structure.

For these cases, use `ovrtx_config_entry_binary_package_root_path()` in the
same config-entry pattern shown by the support API docs test:

> **Source:** `tests/docs/c/test_support_api.cpp` snippet `doc-version-and-config-c`

## Headers

| Header | Purpose |
|--------|---------|
| `<ovrtx/ovrtx.h>` | Main API: create/destroy renderer, add USD, step, fetch results, map output |
| `<ovrtx/ovrtx_types.h>` | All type definitions (handles, structs, enums) |
| `<ovrtx/ovrtx_config.h>` | Config entry builders (`ovrtx_config_entry_*` helpers) |

## Troubleshooting

- `binary_package_root_path` is only needed for static linking or custom layouts that split the default `bin/` structure.
- `ovrtx_setup_runtime()` must be called for each executable target that uses ovrtx.
- On Linux, the build rpath is set automatically. For installed/packaged binaries, ensure the install rpath points to the ovrtx `bin/` directory.
- CMake >= 3.15 is required. Ninja is recommended as the generator.

## References

- Use the `> **Source:**` directives in this skill to locate tested snippets before reusing API patterns.
- Keep related skills, docs, and snippets synchronized when changing the workflow.

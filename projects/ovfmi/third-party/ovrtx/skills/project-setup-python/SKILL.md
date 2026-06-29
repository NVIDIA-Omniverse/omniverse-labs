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
name: project-setup-python
description: >
  Setting up a new Python project that uses ovrtx. Use when user asks to create a new
  Python project, set up ovrtx in Python, create a pyproject.toml, or scaffold a Python
  app.
license: LicenseRef-NvidiaProprietary
version: "0.3.0"
author: NVIDIA ovrtx
tags:
  - ovrtx
  - python
  - setup
tools:
  - Read
  - Grep
  - Shell
  - Write
---

# Project Setup (Python)

## When to Use

Use this skill when the user asks to create a new Python project, set up ovrtx in Python, create a pyproject.toml, or scaffold a Python app.

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

ovrtx is distributed as a Python package on PyPI. The quickest way to get started is with `uv` (recommended) or `pip`. This skill shows how to scaffold a minimal project.

## Project Structure

```
my-ovrtx-app/
  pyproject.toml
  main.py
```

## Setup with uv (Recommended)

```bash
mkdir my-ovrtx-app && cd my-ovrtx-app
uv init
uv add ovrtx
```

This creates a `pyproject.toml` and `uv.lock`. Then add `numpy` (required for array/tensor operations):

```bash
uv add numpy
```

The resulting `pyproject.toml` will look like:

```toml
[project]
name = "my-ovrtx-app"
version = "0.1.0"
requires-python = ">=3.10,<3.14"
dependencies = [
    "ovrtx",
    "numpy",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### Adding optional dependencies

For image output:

```bash
uv add pillow
```

For GPU compute with Warp:

```bash
uv add warp-lang
```

For visualization with rerun:

```bash
uv add rerun-sdk
```

## Setup with pip

```bash
pip install ovrtx
```

## Minimal main.py

> **Source:** `examples/python/minimal/main.py` snippet `create-renderer`
>
> Followed by: `examples/python/minimal/main.py` snippet `add-usd`
>
> Followed by: `examples/python/minimal/main.py` snippet `step`
>
> Followed by: `examples/python/minimal/main.py` snippet `read-render-output`

### Run

```bash
uv run main.py
# or with pip-installed ovrtx:
python main.py
```

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `ovrtx` | Core renderer |
| `pillow` | Image I/O (PNG, JPEG) |
| `numpy` | Array manipulation, DLPack interop |
| `warp-lang` | GPU compute kernels |
| `rerun-sdk` | Real-time visualization |

## Troubleshooting

- ovrtx requires Python 3.10–3.13.
- ovrtx requires an NVIDIA GPU with up-to-date drivers.
- The first `Renderer()` call may take several seconds as the runtime initializes.
- USD files can be loaded from local paths, `file://` URIs, or `https://` URLs.

## References

- Use the `> **Source:**` directives in this skill to locate tested snippets before reusing API patterns.
- Keep related skills, docs, and snippets synchronized when changing the workflow.

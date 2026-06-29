<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-NvidiaProprietary

NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
property and proprietary rights in and to this material, related
documentation and any modifications thereto. Any use, reproduction,
disclosure or distribution of this material and related documentation
without an express license agreement from NVIDIA CORPORATION or
its affiliates is strictly prohibited.
-->
# ovrtx Skills Directory

This directory contains structured skill files for AI coding agents (Cursor, Copilot, etc.) working on the ovrtx codebase. Each skill is a self-contained reference for a specific ovrtx task, with code snippets for both the Python and C APIs.

## Structure

Each subdirectory contains a single `SKILL.md` file with YAML frontmatter:

```
skills/
  renderer-creation/SKILL.md
  loading-usd/SKILL.md
  stepping-and-rendering/SKILL.md
  ...
```

## SKILL.md Format

```markdown
---
name: skill-name
description: >
  What this skill covers. Use when user asks to [trigger phrases].
license: LicenseRef-NvidiaProprietary
version: "0.3.0"
author: NVIDIA ovrtx
tags:
  - ovrtx
  - topic
tools:
  - Read
  - Grep
---

# Skill Title

## When to Use
Activation conditions and trigger phrases.

## Inputs
Expected user request, repository context, and snippet source precedence.

## Prerequisites
Environment or API assumptions needed before following the skill.

## Instructions
Numbered, agent-directed steps.

## Python
Python-specific source references.

## C
C/C++-specific source references.

## Key Types / Functions
Quick reference of the API surface involved.

## Output Format
Expected response or implementation summary.

## Limitations
Known constraints, unsupported cases, or missing example coverage.

## Troubleshooting
Gotchas and things to watch out for.

## References
Related skills and source-reference conventions.
```

## Code Snippet References

Skills reference live code in test and example files instead of duplicating snippets inline. This keeps code in skills accurate as the API evolves.

### Marker format in source files

Python API snippets (`tests/docs/python/test_*.py`):
```python
# [snippet:doc-write-attribute-cpu]
source_np = np.eye(4, dtype=np.float64)
renderer.write_attribute(...)
# [/snippet:doc-write-attribute-cpu]
```

C API snippets (`tests/docs/c/test_*.cpp`):
```cpp
// [snippet:doc-create-renderer-c]
ovrtx_config_t config {};
ovrtx_result_t result = ovrtx_create_renderer(&config, &renderer);
// [/snippet:doc-create-renderer-c]
```

Pure USDA snippets (`tests/docs/data/*.usda` or `tests/docs/usd/data/*.usda`):
```usda
# [snippet:doc-render-product-usda]
def "Render" {
    def RenderProduct "Camera" {
        rel camera = </World/Camera>
    }
}
# [/snippet:doc-render-product-usda]
```

Snippet names are kebab-case, prefixed with `doc-`, and unique across the `tests/docs/` tree. Full examples under `examples/` may still be referenced for complete application structure, but skill-level API and USDA patterns should come from the tested docs snippets above.

### Reference format in SKILL.md

Replace inline code blocks with a blockquote directive:

```markdown
> **Source:** `tests/docs/python/test_attribute_read.py` snippet `doc-read-attribute-scalar`
```

Agents read the referenced file between the `# [snippet:name]` and `# [/snippet:name]` markers to get the current code.

### Reference format in RST docs

Use Sphinx `literalinclude` with marker-based selectors:

```rst
.. literalinclude:: ../../examples/c/minimal/main.cpp
   :language: cpp
   :start-after: // [snippet:create-renderer]
   :end-before: // [/snippet:create-renderer]
```

## Adding a New Skill

1. **Write docs tests first.** Add focused tests under `tests/docs/python` for Python APIs, `tests/docs/c` for C APIs, or `tests/docs/usd` for pure USDA constructs. Wrap every illustrative section in `doc-*` snippet markers.
2. Create a new directory under `skills/` named after the skill (use kebab-case; underscores are allowed when preserving version-number readability, such as `update-0_2-0_3`).
3. Add a `SKILL.md` file inside it following the format above.
4. **Do not write inline code blocks for API usage.** Reference the test/example snippets using `> **Source:** ...` blockquotes. This ensures code in skills always matches real, tested code.
5. For C API patterns, add markers to `tests/docs/c`. Use `examples/c/` only when the skill is specifically explaining full application structure rather than one API pattern.

## Updating Skills

When you make changes to the ovrtx API surface, examples, or conventions that affect an existing skill, update the corresponding `SKILL.md` to keep it accurate.

## Modifying Tests or Examples

- **Preserve snippet markers.** If you move or restructure marked code, update the markers to stay around the illustrative section.
- **Do not remove markers** without also removing or updating every `> **Source:**` reference in `skills/` and every `literalinclude` in `docs/` that points to them.
- **Add markers to new tests.** Every new test function that demonstrates an API workflow should have snippet markers around its illustrative code. If the workflow maps to an existing skill, add a reference there. If not, consider creating a new skill.

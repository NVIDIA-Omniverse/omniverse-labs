---
name: agentic-codebase-cartography
description: Map unfamiliar or multi-repository codebases before refactoring, integration, or feature work. Use when Codex needs to analyze submodules, ownership boundaries, APIs, duplicate implementations, build/test surfaces, risk areas, and a practical refactor sequence before changing code.
---

# Agentic Codebase Cartography

## Purpose

Build an accurate map before making architectural changes. Treat the repository as evidence, not as confirmation of names, READMEs, or old plans.

## Workflow

1. Start with filesystem and VCS shape.
   - Run `git status --short` at the workspace root and in relevant submodules.
   - Run `rg --files` or `find` to identify languages, build systems, bindings, shaders, generated files, tests, and docs.
   - Record dirty files so later changes do not accidentally overwrite user or prior-agent work.

2. Identify product roles.
   - Classify each repo/module as data layer, renderer/backend, viewer/frontend, binding layer, test suite, benchmark suite, or staging code.
   - Prefer what the build and entrypoints actually use over what docs say.
   - Note who owns behavior: scene parsing, material extraction, GPU upload, window/input, UI, screenshots, tests.

3. Trace executable surfaces.
   - Find `build.sh`, `run.sh`, CMake targets, package entrypoints, Python `__main__.py`, CLI flags, tests, and examples.
   - Run or inspect `--help` paths before assuming usage.
   - Follow wrappers to the real binary/library/module.

4. Trace contract surfaces.
   - Identify public headers, Python bindings, ABI structs, shared libraries, plugin APIs, and CLI contracts.
   - Check whether downstream callers duplicate structures or call optional symbols.
   - Treat contract drift as a higher risk than local implementation mess.

5. Locate overlap and duplication.
   - Search for repeated filenames and concepts across repos: renderer, scene, material, camera, shader, backend, viewer, compat.
   - Compare implementations by ownership, not just filename.
   - Mark duplicated code as either intentional portability, temporary transition, or accidental fork.

6. Separate live, dormant, and dead paths.
   - Live: built or imported by current default commands.
   - Dormant: not default but intentionally supported, documented, or tested.
   - Dead: reference copies, staging paths, stale compatibility fallbacks, old generated code, or scripts that only target removed architecture.

7. Summarize with refactor leverage.
   - Give a short module map.
   - List canonical contracts.
   - List deletion candidates with evidence.
   - List risky unknowns requiring a build/run trace.
   - Propose the smallest sequence that removes duplication without blocking feature work.

## Evidence Patterns

Use these searches early:

```bash
rg -n "add_executable|add_library|target_link|project\\(" .
rg -n "__main__|argparse|click|if __name__ == .__main__." .
rg -n "legacy|fallback|deprecated|compat|TODO|Phase|temporary|for reference" .
rg -n "ctypes|extern \"C\"|typedef struct|public API|ABI|binding" .
rg -n "shader|material|scene|renderer|viewer|backend|adapter" .
```

## Deliverable Shape

Prefer this structure:

- **Map:** repo/module -> role -> owned behaviors.
- **Contracts:** public API or binary/module boundary -> consumers.
- **Overlap:** duplicated implementation -> canonical owner -> recommended action.
- **Risks:** uncertain or high-blast-radius paths -> validation needed.
- **Next Work:** ordered refactor steps, each independently buildable.

## Guardrails

- Do not delete or refactor based only on file names.
- Do not trust phase docs more than current build outputs.
- Do not “clean up” compatibility paths until the current supported version policy is clear.
- Do not assign agents overlapping write ownership.
- Do not let exploration end without a concrete integration/test plan.

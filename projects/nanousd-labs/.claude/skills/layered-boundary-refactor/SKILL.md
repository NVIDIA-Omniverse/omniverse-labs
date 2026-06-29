---
name: layered-boundary-refactor
description: Refactor disjoint or duplicated projects into clear layered architecture with canonical contracts and thin adapters. Use when viewers, renderers, bindings, data libraries, plugins, or services overlap and need maintainable ownership boundaries without losing behavior.
---

# Layered Boundary Refactor

## Purpose

Move behavior to the layer that should own it. Keep adapters thin, contracts explicit, and each step buildable.

## Layer Model

Use this ownership split unless the project already has a stronger convention:

- **Data/core layer:** parsing, composition, data queries, authoring, stable data ABI.
- **Backend/renderer/service layer:** domain execution, resource ownership, performance-critical internals, backend-specific implementation.
- **Binding layer:** exact projection of the public contract into Python/JS/Rust/etc.; no hidden business logic.
- **Viewer/UI/client layer:** user interaction, windowing, presentation, CLI flags, screenshots, orchestration.
- **Tests/benchmarks:** validate the layer that owns the behavior.

## Refactor Workflow

1. Declare the canonical contract.
   - Pick the smallest public header, interface, protocol, or package API that downstream code should consume.
   - Write down required structs, functions, flags, lifecycle, error handling, and ownership rules.
   - Remove parallel “almost the same” contracts where possible.

2. Make one consumer thin.
   - Replace duplicate implementation code with calls into the canonical layer.
   - Keep only user interaction, CLI parsing, input handling, and adapter glue.
   - Rename transitional names once the old path is gone; avoid permanent `thin`, `new`, or `legacy` names.

3. Prove parity before deletion.
   - Build both old and new paths if the old path still exists.
   - Run smoke tests on representative fixtures.
   - Compare behavior with logs, screenshots, outputs, or API results.
   - Only then remove the duplicate path.

4. Delete transition scaffolding.
   - Remove fallback executables, old docs, duplicate shaders/assets, staging packages, reference copies, and QA scripts aimed only at the removed path.
   - Update build scripts so failure is explicit instead of silently falling back.
   - Update docs to describe the new steady state, not the migration.

5. Rebuild downstream.
   - Rebuild the canonical implementation first.
   - Rebuild adapters/clients after.
   - Run at least one end-to-end smoke path through the new boundary.

## Boundary Heuristics

Move code downward when:

- Multiple clients duplicate it.
- It owns resources or lifecycle details hidden from callers.
- Correctness depends on private backend details.
- The viewer/client must be fixed every time the backend changes.

Keep code upward when:

- It is presentation or input behavior.
- It is user-facing CLI/UI policy.
- It selects among backends without knowing their internals.
- It adapts one public contract into another.

## Adapter Rules

- Adapter code should be boring: parse args, call API, translate errors, update UI.
- Adapter code should not parse scene internals, upload GPU resources, compile shaders, or duplicate material logic.
- Adapter builds should link to the canonical library and fail if the library is missing.
- Adapter docs should point to the canonical layer for feature support.

## Deletion Checklist

Before deleting a duplicate path:

- Current default build no longer compiles it.
- Current default run path no longer calls it.
- Docs no longer recommend it.
- Tests cover the replacement path.
- Any useful test assets or small fixtures were moved to the owning repo.
- The removed path does not provide a unique supported capability.

## Common Failure Modes

- Keeping fallback logic after the fallback has stopped being maintained.
- Leaving duplicate tests that validate deleted architecture.
- Moving code without moving ownership of bugs and tests.
- Treating compatibility wrappers as harmless; they hide stale contracts.
- Renaming binaries without updating scripts, docs, and smoke tests.

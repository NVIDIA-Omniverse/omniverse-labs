---
name: renderer-feature-validation-matrix
description: Create and maintain renderer feature validation matrices for correctness, performance, backend parity, and test coverage. Use when Codex plans renderer milestones, delegates subagent work, compares against Kit/Hydra/reference output, or decides whether a renderer feature is implemented, degraded, missing, or untested.
---

# Renderer Feature Validation Matrix

## Purpose

Keep renderer work from drifting into vague “looks better” progress. Every major renderer capability should have an owner, a fixture, a reference, an observable output, acceptance criteria, and a backend status. The matrix is also allowed to authorize broad pivots: rows can describe subsystem replacement, not just small feature checkboxes.

## Workflow

1. Inventory feature dimensions.
   - USD ingest: payloads, variants, purpose, transforms, cameras, units, instancing, subsets, visibility, primvars, and animation.
   - Materials/textures: Preview Surface, MaterialX, bindings, inherited opinions, UV transforms, UDIMs, normal maps, ORM packing, color spaces, alpha, opacity, transmission, emissive, metallic, and roughness.
   - Lighting/transport: dome, distant, sphere, disk, rect, emissive meshes, exposure, intensity units, shadows, transparent shadows, GI, reflections, transmission, MIS, max depth, and accumulation.
   - Renderer architecture: batching, instancing, streaming, texture residency, AS build/refit, pass scheduling, readback, telemetry, and memory budgets.
   - Viewer/test path: camera framing, first frame, render modes, screenshots, debug AOVs, goldens, reference captures, and CI availability.

2. Define a row before assigning implementation.
   - Each row needs a feature, priority, source scene, reduced fixture, reference renderer/image, observable output, test command, pass criteria, backend status, owner, and residual risk.
   - Split rows when correctness, quality, and performance need different proof.
   - For a stalled renderer subsystem, create an architecture-pivot row that names the old path to bypass or retire, the target replacement class, and the evidence that justifies the broader scope.
   - Mark unsupported features explicitly as `missing`, `degraded`, `stubbed`, or `blocked`, not as unknown.

3. Use the matrix to delegate work.
   - Give worker/subagent prompts scoped to rows or independent row groups.
   - Treat those scopes as ownership boundaries for coordination, not as a prohibition on a broad replacement when the row requires one.
   - Avoid delegating broad renderer quality goals without a fixture and acceptance criterion.
   - Require changed files, tests run, images captured, telemetry, and remaining matrix updates in every worker handoff.

4. Gate milestone completion.
   - A milestone is complete only when required rows have passing evidence on required backends.
   - A beauty-image improvement does not close rows for load coverage, material terms, shadow transport, color management, or performance unless those outputs were checked directly.
   - If a backend cannot run locally, require a handoff row with exact build/run commands and expected artifacts.

5. Keep the matrix current.
   - Add rows for every newly discovered bug, missing feature, or reference mismatch.
   - Update pass criteria after deliberate product decisions, not after incidental output changes.
   - Link reduced repro scenes and artifacts so future agents can rerun the proof.

## Suggested Columns

Use a Markdown table or CSV with these columns:

```text
Priority | Area | Feature | Scene | Reduced fixture | Reference | Backend(s) | Observable | Command | Pass criteria | Status | Owner | Notes
```

Recommended statuses:

```text
passing | failing | missing | degraded | stubbed | blocked | untested | handoff
```

## Minimum Rows For Physical Renderer Parity

- Stage traversal and load counts match the reference for selected payloads, variants, and purposes.
- Camera, units, exposure, tone map, and color-management settings are reproducible.
- Base color, normal, metallic, roughness, opacity, emissive, and texture color-space AOVs are validated independently.
- Direct lighting and shadow visibility are validated for each supported light type.
- Transparent/alpha-cutout shadow transport is validated with a reduced fixture.
- Emissive mesh sampling is validated separately from analytic lights.
- Indirect GI, reflections, and transmission are validated with deterministic samples and recorded seeds.
- Instancing, material overrides, and per-instance transforms are validated on both tiny and large scenes.
- Large-scene performance has preflight estimates, resource telemetry, and opt-in stress captures.
- Each backend has a status for support, parity, known degradation, and platform handoff.

## Anti-Patterns

- Treating a screenshot comparison as proof for every renderer subsystem.
- Closing a feature because it works on a tiny scene but has no authored-scene coverage.
- Leaving unsupported USD features invisible in logs and matrices.
- Combining “implemented”, “tested”, and “matches reference” into one checkbox.
- Delegating work without giving the worker a row-level acceptance criterion.
- Splitting a failed architecture into ever-smaller rows that cannot falsify or replace the suspect subsystem.

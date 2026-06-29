---
name: renderer-implementation-roadmap
description: Sequence a nanoUSD renderer build from nothing into a correct, scalable backend using only the renderer skill set. Use when Codex starts a new renderer or backend, plans renderer milestones and phase order, decides what must be correct before the next layer, or needs the contracts-first build order and reduced-fixture corpus that turn the other renderer skills into a buildable plan with no reference renderer to copy from.
---

# Renderer Implementation Roadmap

## Purpose

The other renderer skills are *aspect* skills — scene extraction, PBR validation,
performance, parity triage. This is the *sequencing* skill. It orders them into a
build a fresh agent can follow with no reference renderer to copy from, and it
states the gate that must hold before each next layer. Contracts before backends;
one coherent vertical slice before breadth; every phase has a correctness gate.
This is the renderer counterpart to a spec implementation roadmap: the build order
is itself a deliverable, because the wrong order forces rewrites.

Read this first. Each phase below names its goal, the skill(s) that drive it, the
gate that closes it, and **the failure you cause by skipping or reordering it** —
so the order is self-justifying, not arbitrary.

## Build order

**Phase 0 — Map and target.** Skills: `agentic-codebase-cartography`,
`renderer-target-architecture-design`. Write down target scene scale, resolution,
frame time, sample/noise budget, memory budget, supported features, the reference
renderer, and the architecture *class* (hardware RT vs raster, resident scene,
render graph, temporal accumulation, …).
*Gate:* a written target and a named reference exist.
*Skip-failure:* you build a brute-force/flattened path you will throw away before
the goal is reachable — the exact trap `renderer-target-architecture-design` exists
to prevent.

**Phase 1 — Freeze the contracts before any backend.** Skills:
`renderer-capability-abi`, `contract-first-ffi`, `nanousd-renderer-scene-extraction`.
Define, as one canonical copy each: the public C ABI (capability bitmask +
backend-info query); the RHI seam that separates the platform-agnostic core from
the backend; the scene-extraction structs; and the cross-cutting conventions
(coordinate/transform convention, vertex layout, color-space policy).
*Gate:* the ABI conformance check is green and a stub backend links against all
three contracts.
*Skip-failure:* backend #2 forces a rewrite of the core, and the "shared" files
drift apart (observed in practice: a scene loader that diverged thousands of lines
between backends, and a capability ABI present in only one of three headers).

**Phase 2 — Backend skeleton, portable build, headless triangle.** Skills:
`portable-build-runtime-paths`, `renderer-backend-port`. The smallest end-to-end
path: init the device, compile one shader, draw one triangle, read pixels back
headless.
*Gate:* a deterministic headless image (known clear color + triangle) that renders
after relocation — no `cwd`, absolute-path, username, or display assumptions.
*Skip-failure:* you debug GPU bring-up and scene bugs simultaneously with nothing
reproducible to bisect.

**Phase 3 — Scene extraction (CPU, no shading).** Skill:
`nanousd-renderer-scene-extraction`. Compose USD into the renderer scene structs
with counts, bounds, instancing preserved, and unsupported-feature diagnostics —
kept separate from GPU upload.
*Gate:* load counts, bounds, and diagnostics match the reference on the reduced
corpus **before any shading**.
*Skip-failure:* white or exploded scenes you misdiagnose as shader bugs (see that
skill's failure signatures) — the cost of shading on top of a wrong scene.

**Phase 4 — Raster vertical slice.** Forward-raster the extracted meshes with one
light.
*Gate:* a tiny fixture renders nonblank with correct silhouette and a correct
normals AOV.
*Skip-failure:* you start ray tracing with no known-good geometry/transform path to
diff against.

**Phase 5 — Hardware ray tracing (if in target).** Skills:
`renderer-performance-architecture` (acceleration-structure section),
`renderer-backend-port`. BLAS-per-prototype, TLAS over instances, refit rather than
rebuild.
*Gate:* RT primary matches the raster slice on the same fixture, plus a refit smoke
test (move one instance, re-render, verify it moved).
*Skip-failure:* raster and RT silently disagree about which geometry exists — the
`renderer-performance-architecture` anti-pattern.

**Phase 6 — Lights, materials, IBL.** Skills:
`nanousd-renderer-scene-extraction` (materials/lights),
`physically-based-renderer-validation`. UsdLux lights, UsdPreviewSurface/MaterialX,
dome/IBL.
*Gate:* per-term AOVs (base color, normal, roughness, metallic, direct, shadow) are
validated independently before any beauty comparison.
*Skip-failure:* you tune exposure or material defaults to hide a broken term — the
core anti-pattern of `physically-based-renderer-validation` and `graphics-debugging-lab`.

**Phase 7 — Validation harness (continuous from here).** Skills:
`renderer-image-comparison-testing`, `renderer-feature-validation-matrix`. Stand up
the golden-image harness and the feature matrix as the live ledger.
*Gate:* backend goldens exist and, where parity is a contract, reference deltas —
with term AOVs stored beside beauty.
*Skip-failure:* "looks better" progress with no regression net.

**Phase 8 — Performance architecture for scale.** Skill:
`renderer-performance-architecture`. Instancing, residency, async batched uploads,
draw batching, pass gating, telemetry; readback off the interactive path.
*Gate:* a representative-scale probe is within budget and the correctness fixtures
are still green.
*Skip-failure:* a fast path that omits geometry/shadows/GI and a "speedup" that is
really a correctness regression.

## Continuous spines (not phases)

- `renderer-feature-validation-matrix` — track every capability's status, owner,
  fixture, and reference from Phase 1 onward; it is the ledger the gates write to.
- `renderer-stalled-parity-triage` — invoke the moment beauty/metrics stop improving.
- `verification-led-development` — the smallest meaningful build/test/screenshot at
  each gate.
- `graphics-debugging-lab` — when one stage misbehaves, isolate it before tuning.
- Defer features by *precondition, not by date*: when a phase's technique has no
  cost-justifying trigger yet (a denoiser on noise-free output, intersection-function
  buffers with one intersection function, BLAS compaction on small scenes), record the
  exact condition that should revive it — "deferred" without its trigger becomes either
  gold-plating or a forgotten gap.
- The no-sacred-cows rule from `renderer-target-architecture-design` overrides the
  whole order: if a gate keeps failing with the same residual, pivot the
  architecture — do not shrink scope and retry.

## Canonical reduced-fixture corpus

Build these tiny, deterministic fixtures early and keep them as regression cases.
Record **what each one proves**, not expected pixel counts (those move with sampling
and tone mapping; a frozen count rots). Mirror the reference corpus that ships under
the renderer repos' `tests/assets/` rather than re-inventing names.

| Fixture | Proves |
|---|---|
| reset-xform-stack scene | `resetXformStack` composition and transform order |
| z-up + centimeters scene | up-axis and metersPerUnit applied exactly once at the boundary |
| face-varying UV seam | face-varying attribute vertex expansion and seam handling |
| typed-gprim (cube/cone/capsule) | procedural-primitive coverage and axis convention |
| MaterialX UV-tiling bind | material binding + UV transform/tiling resolution |
| white diffuse room + one analytic light | direct lighting and shadow visibility |
| metallic/roughness grid | BRDF response under neutral light |
| ORM + normal + UV-transform case | texture packing, tangent space, color-space classification |
| alpha-cutout blocker between light and surface | transparent/cutout shadow transport |
| emissive mesh behind a blocker | emitter sampling and visibility (not indirect-only discovery) |
| instanced/prototype with material override | instancing, per-instance transform and material override |
| one large authored scene (warehouse/vehicle) | integration and scale, opt-in and telemetry-first |

## Anti-Patterns

- Standing up a backend before the public ABI, RHI seam, and scene structs exist —
  guarantees a rewrite at backend #2.
- Shading before load counts, bounds, and AOVs are proven on the reduced corpus.
- Treating the phases as independent parallel tasks instead of gated layers.
- Copying contract values into a plan document instead of pointing at the canonical
  header — recreates the drift this roadmap is built to avoid.
- Shrinking scope after a gate repeatedly fails instead of pivoting the architecture.

## Handoff

Report the current phase, which gates pass on which backends, the feature-matrix
link, the reduced-fixture results, and the single smallest command that advances the
next gate.

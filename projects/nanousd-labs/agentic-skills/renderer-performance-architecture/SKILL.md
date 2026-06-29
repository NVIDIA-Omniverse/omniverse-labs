---
name: renderer-performance-architecture
description: Design, review, or refactor high-performance renderer architecture for large USD scenes. Use when Codex needs to reason about target-quality/SOTA renderer architecture, instancing, batching, draw calls, frame-time budgets, geometry streaming, GPU buffer residency, texture/material upload, acceleration structures, shadows, forward/deferred render paths, readback, telemetry, performance validation, or replacing dead-end renderer paths.
---

# Renderer Performance Architecture

## Purpose

Design renderer changes so large scenes remain structured, measurable, and scalable. Preserve scene intent first, then choose backend-specific GPU representations deliberately.
Performance work has no sacred cows: when telemetry and visual evidence show the current architecture cannot reach the target, replace the architecture instead of optimizing around it.

## Workflow

0. Start from the target architecture.
   - Record the architecture class required by the goal before implementing: hardware RT/ray query, resident scene, render graph, temporal accumulation, guided denoising, light tree/reservoir/path guiding, GPU-driven submission, streaming textures, or another proven technique.
   - Use simple paths only as diagnostics, fixtures, compatibility fallbacks, or target-shaped vertical slices. Do not build feature coverage on a path already known to miss the visual, memory, or frame-time target.
   - If multiple measured slices fail because the current pass layout, sampler, denoiser, scene residency, or AS/update model is wrong, plan and execute the larger replacement. Do not keep a slow or visually wrong path as the center of gravity because it is already wired up.

1. Establish the correctness gate.
   - Name the reference renderer or analytic fixture that defines correct output before making a performance claim.
   - Confirm load counts, material/light counts, camera metadata, tone mapping, and key AOVs for the workload.
   - If the scene is physically wrong, classify each issue as a correctness bug, missing feature, quality gap, or performance bottleneck before optimizing.
   - Do not optimize a fast path that omits required geometry, materials, lights, transparency, shadows, or GI without labeling it as degraded.

2. Define the performance claim.
   - Identify the target workload: many instances, many unique meshes, huge textures, many lights, RT shadows, tiled environments, interactive viewing, offline capture, or animation playback.
   - Set budgets for load time, first image, frame time, draw calls, upload bytes, texture memory, acceleration-structure build time, readback time, and peak RSS/VRAM.
   - Record current telemetry before changing architecture.

3. Preserve source structure.
   - Keep USD prototypes, PointInstancers, native instances, material bindings, paths, visibility, transforms, time samples, and payload state distinct during ingest.
   - Build a prototype table and instance table. Do not make a flat mesh array the only internal representation.
   - Treat flattening as an explicit fallback with telemetry, not the default for large scenes.

4. Separate dirty types.
   - Topology changes can rebuild geometry and acceleration structures.
   - Transform, visibility, material parameter, camera, light, render-mode, and time changes should have narrower update paths.
   - Avoid full scene reloads during playback, camera motion, visibility toggles, and capture option changes.

5. Design geometry storage for residency.
   - Use compact vertex layouts and split rarely used attributes into separate streams.
   - Preserve shared prototype buffers and store transforms/material overrides in instance buffers.
   - Use chunked allocation, 64-bit size accounting, explicit overflow checks, and resource-class memory telemetry.
   - Plan for partial rebuilds and streaming before adding large asset test cases.

6. Make uploads asynchronous and batched.
   - Prefer persistent staging rings or reusable upload heaps.
   - Batch copies, texture uploads, mip generation, and buffer updates into fewer command submissions.
   - Do not use queue-wide idle waits or per-resource blocking waits in normal large-scene paths.
   - Surface upload progress and cancellation for long asset operations.

7. Batch draw submission.
   - Group by pipeline, render state, material, vertex layout, prototype, and index range.
   - Use instanced draws for repeated prototypes.
   - Use multi-draw indirect, indirect command buffers, or backend equivalents where available.
   - Store per-instance data in GPU buffers instead of recording CPU push constants for every object.

8. Treat acceleration structures as managed resources.
   - Reuse prototype BLASes across instances.
   - Build all BLAS from one pooled allocation with batched builds; keep memory-allocation and submit counts O(1) / O(n÷batch), not O(n). If the pool allocation fails from VRAM fragmentation, binary-search the size (halve and retry) instead of giving up, and keep a fixed safety margin for driver overhead.
   - When VRAM forces a partial BLAS subset, do not pack smallest-first to maximize count — that omits the few huge BLASes (terrain, roads) that dominate the image. Reserve budget for the largest *visible* BLASes, then fill the remainder with small meshes.
   - A persistent AS/geometry cache must be content-keyed (a geometry hash), not index-keyed — USD parse order is nondeterministic across cold loads, so an index key misses constantly; memcmp-verify a hash hit, exclude per-instance attributes (transform, color) from the key, and treat a baked-in material id as a separate group key.
   - Build/refit TLASes asynchronously when possible.
   - Partition AS work only when it reduces traversal, rebuild, or memory cost.
   - Report partial RT coverage explicitly; never let raster and RT silently disagree about omitted geometry.
   - Keep primary, shadow, reflection, and tiled/per-env rays consistent about which TLAS they query.
   - Keep alpha cutout, opacity, transmission, two-sidedness, and visibility policy consistent between raster, primary RT, shadow RT, and GI paths.

9. Make materials and textures demand-driven.
   - Cache material bindings by prim path/ancestor chain.
   - Deduplicate textures by resolved identity plus sampler/color-space semantics.
   - Preserve material term semantics while optimizing: base color, data textures, normals, roughness, metallic, opacity, emissive, and texture transforms cannot share an undifferentiated upload path.
   - Apply memory budgets to decoded images, UDIM atlases, mip chains, and GPU texture residency.
   - Upload visible or high-priority textures first and use explicit placeholders for pending assets.

10. Gate expensive passes.
   - Use a render graph or pass scheduler for AOVs, shadows, debug views, resolve/copy work, and readback.
   - Produce only requested outputs.
   - For real-time path tracing with hard HDRI sunlight, many emitters, or noisy GI, separate direct-light reuse, GI reuse/path guiding, and guided denoising into measurable pass classes rather than hiding them inside one monolithic beauty dispatch.
   - Prefer clustered, tiled, forward+, or deferred lighting when light count makes simple forward shading expensive.
   - Keep debug and validation modes available but out of the normal hot path.
   - Keep light sampling tables, PDFs, MIS weights, and pass filters synchronized so performance filtering does not bias transport.

11. Remove readback from interactive rendering.
   - Interactive viewing should render without fetching pixels to the CPU.
   - Screenshot and golden-image paths should use explicit readiness/fence APIs.
   - Use readback rings or backend equivalents and reuse CPU buffers.

12. Validate with scale-aware tests.
   - Keep tiny deterministic correctness scenes.
   - Add medium scenes that exercise instancing, materials, textures, curves, lights, and transforms.
   - Make large showcase scenes opt-in, preflighted, and telemetry-first.
   - Record counts, timings, memory, AOV/image metrics, sample settings, and reference-renderer deltas so performance regressions are explainable without hiding correctness regressions.

## Review Checklist

- Does the ingest path preserve prototypes and instances until the backend consumes them?
- Is the optimized path still compared against a reference or analytic correctness fixture?
- Does a time/transform/visibility edit avoid full geometry/material reload?
- Are large uploads batched without queue-wide idle waits?
- Are draw calls proportional to batches or visible prototypes, not raw instance count?
- Are textures decoded/uploaded under a budget with deduplication and clear color-space policy?
- Are BLAS/TLAS builds asynchronous or at least isolated from ordinary frame rendering?
- Do primary, shadow, reflection, and GI rays agree on geometry, visibility, opacity, and alpha policy?
- Are light sampling PDFs and MIS weights valid after culling, proxying, or filtering lights?
- Are AOVs, shadows, and readbacks produced only when requested?
- Are hard limits reported as structured errors instead of silent truncation?
- Are memory and timing metrics split by geometry, texture, material, AS, render targets, upload, and readback?
- Is backend parity explicit: supported, degraded, stubbed, or unavailable?

## Common Anti-Patterns

- Expanding PointInstancers into full mesh records before renderer import.
- Uploading a second full vertex buffer for a material mode instead of using compact streams.
- Recording one draw call per mesh or instance after already merging geometry.
- Waiting for the whole GPU queue after each staging copy, texture upload, BLAS batch, screenshot, or readback.
- Scanning entire asset directories for every material load.
- Stitching unbounded UDIM atlases without decoded-memory estimates.
- Rebuilding the entire renderer scene for time changes.
- Copying depth, normals, segmentation, or debug buffers when the current request does not need them.
- Letting RT omit geometry silently while raster still draws it.
- Running large golden captures without preflight estimates and incremental telemetry.
- Claiming a speedup while material terms, shadows, transparency, or GI are known to diverge from the reference.
- Extending a known-insufficient fallback instead of moving work onto the required target architecture.
- Treating narrow implementation scope as safer after evidence shows the existing architecture is the risk.

## Handoff

When finishing performance architecture work, report:

- Which workload and budget the change targets.
- Which target architecture class the change advances, and which temporary paths remain.
- What scene structure is preserved or intentionally flattened.
- Expected effects on draw count, upload bytes, AS build time, texture memory, readback, and frame time.
- Exact telemetry or tests used to validate the claim.
- Remaining backend-specific gaps and any platform that still needs hand-off validation.

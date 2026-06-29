---
name: renderer-target-architecture-design
description: Design renderer features from the target-quality architecture first, including radical subsystem replacement when incremental slices are exhausted. Use when Codex implements or plans path tracing, denoising, lighting, sampling, acceleration structures, scene residency, instancing, texture systems, material systems, render graphs, or major renderer features where state-of-the-art/proven production techniques are needed.
---

# Renderer Target Architecture Design

## Purpose

Start from the architecture capable of reaching the target. Small steps are useful only when they are vertical slices, probes, or scaffolding for that architecture. Avoid shipping intentionally weak implementations that will predictably be replaced before the stated goals can be met. No subsystem is sacred: when evidence shows the current architecture cannot reach the visual, scalability, or frame-time target, prefer a confident replacement over another local patch.

## Workflow

1. State the target, not just the next patch.
   - Record target scene scale, resolution, frame time, samples/noise, memory budget, update rate, supported features, and reference quality.
   - Decide whether the goal is interactive, offline, viewport-quality, reference parity, or diagnostic-only.
   - Name the class of proven solution expected for that goal, such as hardware RT/ray queries, resident scene data, render graph scheduling, temporal accumulation, guided denoising, light trees/reservoirs, path guiding, tiled/clustered lighting, bindless resources, streaming textures, or GPU-driven draws.

2. Reject dead-end simplicity.
   - A simple implementation is acceptable only if it is a fixture, diagnostic probe, compatibility fallback, or vertical slice of the target architecture.
   - If the simple path cannot scale to the target workload, gate it as degraded, document its removal/integration plan, and avoid building more features on top of it.
   - Do not choose brute-force loops, flattened scene copies, CPU readbacks, per-object/per-light submissions, per-frame full reloads, or scalar post tweaks when the target requires a different architectural class.
   - Do not preserve code, scripts, metrics, or defaults merely because they already exist. If they encode the wrong architecture, replace or retire them.

3. Design the minimal target-shaped slice.
   - Preserve the final data model even in the first slice: instances stay instances, materials stay structured, lights keep PDFs/metadata, history buffers keep frame indices, and resources keep residency ownership.
   - Implement one coherent end-to-end path through the final architecture before broad coverage. It may be narrow in feature coverage while still touching many files or replacing a subsystem.
   - Add capability flags only for real backend differences, not to hide a half-replacement architecture.
   - "Minimal" means the smallest slice that can prove or falsify the target architecture, not the smallest possible edit. If a broader rewrite is the first slice that can test the right hypothesis, take the broader rewrite.

4. Escalate scope when evidence warrants it.
   - After repeated local slices fail with the same residual pattern, stop iterating around the old design and name the architecture-level pivot.
   - Accept large, multi-file changes when the hypothesis is coherent, evidence-backed, and directly aligned with the target. Keep them reviewable with clear ownership, fixed scripts, and before/after artifacts.
   - It is valid to discard or bypass existing renderer paths, denoisers, samplers, caches, or comparison scripts when they are blocking the target state. Leave a short migration/removal note rather than preserving both paths indefinitely.

5. Choose SOTA/proven techniques by bottleneck. Search latest research for inspiration.
   - Geometry scale: shared BLAS/TLAS, GPU-driven or batched submission, compact buffers, streaming, and stable IDs.
   - Many lights or HDR environments: importance sampling, MIS, light trees/reservoirs, alias tables, spatial/temporal reuse, or path guiding instead of uniform brute force.
   - Noisy indirect transport: better proposal distributions, accumulation history, variance estimates, adaptive sampling, and guided reconstruction instead of only clamping.
   - Hard direct light, HDRI sun lobes, or many emissive surfaces: preserve direct-light samples as stable lighting estimates before denoising. Use light trees, reservoirs, temporal/spatial reuse, final visibility, and unbiased or explicitly biased reconstruction controls before trying more light-scale or clamp sweeps.
   - For reservoir or candidate-reuse slices, store only candidates that are valid for the estimator under test. Raw light proposals are useful diagnostics, but quality-path reuse candidates should carry enough PDF/MIS/count/visibility or contribution metadata to be re-evaluated, merged, and rejected without hidden energy gain.
   - Noisy GI in large interiors: preserve first secondary-surface candidates, PDFs, radiance, normal, visibility, and history when feasible. If indirect AOVs are dominated by fireflies, the target slice should be GI reuse/path guiding/variance-aware reconstruction, not only more spp or roughness gating.
   - Denoising: use normal/albedo/depth/motion/history guidance where available; color-only blur is a diagnostic baseline, not the target for hard lighting.
   - Production denoising requires a signal contract: diffuse/specular or direct/indirect split, demodulated radiance where useful, hit distance, roughness, view depth, disocclusion/history confidence, and stable surface IDs or planes. Design these buffers before judging denoiser quality.
   - Interactive iteration: resident resources, incremental updates, asynchronous uploads, GPU timing, and readback only on capture paths.

6. Validate target viability early.
   - Run a tiny deterministic fixture for correctness.
   - Run a representative-scale probe for memory layout, dispatch size, frame time, and driver stability.
   - Compare equal-time or equal-budget quality when changing sampling, denoising, or transport.
   - For opt-in reuse/reconstruction passes, require both a resident-frame timing probe and a beauty/contact-sheet checkpoint against the current non-reuse baseline before promotion. A finite candidate AOV or lower noise in one buffer is not enough if regional beauty metrics or signed residuals regress.
   - Track whether each slice reduces target risk or just creates a temporary result.

7. Handoff target debt explicitly.
   - Report which parts are target architecture, which are temporary scaffolding, and what must not be built on.
   - Leave matrix rows for the remaining target techniques and removal of degraded paths.
   - If current hardware/API support blocks the target, mark the feature blocked or degraded instead of substituting a known-insufficient design.

## Anti-Patterns

- Building a brute-force or flattened implementation first when the target requires acceleration, residency, or reuse.
- Letting a diagnostic CPU/readback path become the default production path.
- Adding feature coverage to a path already known to miss the frame-time or memory target.
- Calling a placeholder “initial support” without a target-shaped data model and migration plan.
- Optimizing a dead-end path because it is easier than implementing the needed architectural class.
- Treating "small scope" as a virtue after evidence shows only a larger architecture change can move the target.
- Confusing narrow feature coverage with narrow edit scope; the first useful target-shaped slice may be a broad rewrite.
- Keeping a familiar subsystem alive after its assumptions have been falsified by artifacts.
- Treating denoising, clamping, or exposure tuning as a substitute for transport and sampling architecture.
- Treating AOV-visible direct light that disappears in beauty as a scalar parameter problem before proving the direct, residual, clamp, and denoiser input buffers.
- Adding more color-only denoiser variants after the target requires separate direct/indirect or diffuse/specular histories, hit distances, and temporal confidence.

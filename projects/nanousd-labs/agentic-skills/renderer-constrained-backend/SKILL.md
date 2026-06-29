---
name: renderer-constrained-backend
description: Build a renderer backend to a hard resource or capability budget — low VRAM, raster-only (no hardware ray tracing), or a portable/embedded GPU like OpenGL ES — where features must be baked, capped, or degraded rather than ported wholesale from the high-end backend. Use when Codex builds a low-end/portable backend, fits a GPU-memory budget, bakes materials instead of codegen, or makes a no-RT fallback match a reference renderer.
---

# Renderer Constrained Backend

## Purpose

A low-end backend — tight VRAM, no hardware ray tracing, a portable GPU such as OpenGL ES — is a deliberate *target*, not a degraded copy of the reference renderer. Its constraints change the architecture, not just a quality knob: what you bake vs. generate, what you cap, what you approximate. Each of those must be a code invariant and stay comparable to the reference, or the backend is "correct but visibly different" — which derails parity for reasons that read as bugs. Pairs with `renderer-target-architecture-design` (name the target first) and `renderer-realtime-sensor` (the fidelity/throughput/memory dial).

## Budget is an invariant enforced at upload, not an aspiration

- A "<100 MB GPU" target is met by a hard, central cap applied at load time — clamp texture resolution and box-downsample anything over the cap, exposed as one tunable. Warehouse-scale USD ships 2K–4K textures that blow a low-VRAM budget instantly; nothing downstream guarantees the budget unless the clamp does. Make the budget a code invariant, not a README line.

## Material strategy differs from the high-end backend — bake, don't codegen

- A constrained raster backend resolves MaterialX by walking the node graph and *baking* its `<image>` and procedural nodes into fixed UsdPreviewSurface-shaped texture slots — deliberately not generating a shader per material. The high-end backend's per-material shader codegen would inherit pipeline-count explosion and a uniform-model mismatch on the constrained target. These are *different* material architectures; do not mirror the codegen path onto the constrained backend.

## Degrade to match the reference's policy, not a local invention

- When the backend can't do a feature (no RT shadows, no GI), its fallback must replicate the *reference renderer's documented fallback policy* — e.g. the same fallback dome + distant light gated on authored-light count and IBL presence — so cross-backend images stay comparable; approximate RT shadows with shadow maps. A fallback the backend invents locally produces images that differ for reasons that look like bugs in `renderer-image-comparison-testing`.
- Report every degraded feature honestly through the capability ABI (`renderer-capability-abi`): advertise only what the backend truly does, and mark the rest `degraded`/`stubbed` in the feature matrix — never silently render something different.

## Validate against the budget and the reference

- Prove the budget: load the largest target asset and assert peak GPU memory stays under the cap, with the texture-clamp telemetry visible. Prove parity: compare the constrained backend's degraded output against the *reference's fallback* (not against the high-end backend's RT beauty), so the test exercises the intended degradation rather than an unfair target. See `renderer-feature-validation-matrix` for the per-backend degradation rows.

## Anti-Patterns

- Treating the budget as an asset-authoring assumption instead of a load-time clamp.
- Porting the high-end backend's per-material shader codegen onto the constrained target.
- Inventing a local no-RT lighting fallback instead of matching the reference's policy.
- Comparing the constrained backend against the high-end backend's RT beauty instead of the reference's fallback.
- Advertising a capability the backend only approximates.

## Handoff

Report the resource/capability budget and how it is enforced (the clamp + its tunable), the material strategy (bake vs codegen) and why, which features are degraded and against which reference fallback they were validated, and the peak-memory + parity evidence. Cross-reference `renderer-capability-abi` for honest capability reporting and `renderer-realtime-sensor` for the target dial.

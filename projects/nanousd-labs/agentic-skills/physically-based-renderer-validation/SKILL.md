---
name: physically-based-renderer-validation
description: Design, debug, or validate physically based renderer behavior against a reference renderer. Use when Codex works on PBR materials, USD light semantics, shadows, transparency, GI, emissive sampling, tone mapping, AOVs, stochastic sampling, path tracing, ray tracing, or Omniverse Kit/Hydra/OVRTX visual parity.
---

# Physically Based Renderer Validation

## Purpose

Make physical correctness an explicit contract, not a subjective beauty-image judgment. A renderer is not physically credible until material terms, light transport, camera/exposure, color management, and stochastic sampling can be inspected and compared against a trusted reference. Validation should raise renderer requirements, not force fixes to stay local; when the evidence points at the sampler, reconstruction contract, scene extraction policy, or denoiser architecture, make that subsystem replaceable.

## Workflow

1. Define the parity contract.
   - Name the reference renderer, reference image, scene file, camera, render mode, resolution, samples per pixel, seed, exposure, tone map, color space, and environment.
   - Decide whether the goal is exact parity, physically plausible parity, backend stability, or a known degraded mode.
   - Record which visual differences are accepted and why.

2. Preserve authored scene semantics before shading.
   - Confirm units, up-axis, transforms, cameras, visibility, purpose, payloads, variants, instancing, material bindings, subsets, and inherited opinions.
   - Confirm light type, intensity, exposure, color, shaping, normalization, radius/area, and texture/environment semantics.
   - Confirm texture resolution, UVs, primvars, texture transforms, UDIMs, wrap modes, and color-space classification.
   - Treat unsupported USD or MaterialX nodes as visible capability gaps, not silent white/default materials.

3. Decompose the renderer before tuning beauty output.
   - Add or use AOV/debug views for base color, normals, tangents, metallic, roughness, opacity, emissive, F0, direct light, shadow visibility, indirect light, path length, depth, and tonemapped output.
   - Compare each term against the reference or a reduced analytic scene.
   - Do not compensate for a broken term by changing unrelated shader constants, exposure, or material defaults.

4. Audit transport invariants.
   - Light sampling tables must contain only candidates the shader can actually evaluate.
   - Sampling probabilities, MIS weights, and energy scaling must remain consistent after filtering invisible, unsupported, disabled, or proxy lights.
   - Shadow rays must respect alpha cutout, opacity, transmission, two-sidedness, and material-sidedness policy.
   - Emissive meshes need explicit sampling or a documented fallback; indirect discovery alone is not enough for low-noise validation.
   - Accumulation must track seed, frame index, spp, max depth, Russian roulette policy, and denoising/filtering state.
   - Direct lighting, GI, and denoising should have separable evidence. If direct/visibility AOVs show the expected signal but beauty loses it, validate the exact direct/residual buffers that are clamped, accumulated, filtered, and composited.
   - Many-light or HDRI-sun scenes need target-grade direct-light sampling for low-spp parity: correct PDFs/MIS plus light trees, reservoirs, temporal/spatial reuse, final visibility, and material/depth/normal gates when brute-force samples are insufficient.
   - Noisy indirect GI needs its own sampling/reuse strategy. Track secondary-surface candidates, PDFs, radiance, visibility, and history when possible; do not treat depth-2+ fireflies as a denoiser-only problem.
   - Denoiser validation must include the denoiser signal contract: diffuse/specular or direct/indirect split, demodulated radiance if used, normal/depth/albedo/roughness/motion guides, hit distance, variance, and history/disocclusion confidence.

5. Build reduced reference scenes.
   - White diffuse room with one analytic light for direct/shadow validation.
   - Metallic/roughness grid with neutral lighting for BRDF response.
   - Textured ORM/normal/UV-transform case for material packing and tangent-space checks.
   - Transparent or alpha-cutout blocker between a light and surface for shadow transport.
   - Emissive mesh behind a blocker for emitter sampling and visibility.
   - Instanced/prototype material override case for binding and per-instance data — also the fixture that validates segmentation / instance-ID AOV contracts: an identity such as `segmentation = mesh_id + 1` may hold only because today's scenes have one BLAS per mesh (`instance_id == mesh_id`); the first shared-prototype scene (many meshes → one BLAS) is the real test, so validate ID/AOV semantics here before claiming they hold.
   - A large authored scene, such as a warehouse or vehicle showcase, for integration and scale.

6. Validate in dependency order.
   - First prove load counts, material/light counts, texture resolution, and camera metadata.
   - Then prove unlit/material AOVs.
   - Then prove direct lighting and shadow visibility.
   - Then prove indirect GI, reflections, transmission, and tonemapping.
   - Only then optimize frame time, memory, noise, denoising, and interactive behavior.

7. Report failures as renderer requirements.
   - Separate correctness bugs from missing features, quality limitations, and performance bottlenecks.
   - Attach the smallest scene, captured images, AOVs, logs, telemetry, and reference-renderer settings.
   - If several term-level hypotheses preserve the same wrong beauty/residual structure, stop reducing the patch size. Promote the failure to an architecture requirement and plan a no-sacred-cows replacement or bypass of the suspect subsystem.
   - If parity is still incomplete, leave an explicit validation gap instead of claiming success from a plausible beauty image.

## Failure Signatures

- Flat shaded large scenes usually indicate missing authored materials, missing environment/direct lights, disabled shadows, broken normals/tangents, or tone mapping hiding contrast.
- Metallic objects that look diffuse usually indicate missing metallic texture/channel, wrong color-space conversion for ORM data, incorrect F0/BRDF math, or absent reflection/indirect contribution.
- Noisy direct-light regions usually indicate poor emitter sampling, mismatched PDFs, missing MIS, or treating large emissive geometry as only indirectly discoverable.
- Shadows that disappear in authored scenes often indicate a separate shadow acceleration structure omitting geometry, alpha/transparency not implemented, disabled light candidates, or inconsistent instance visibility.
- Kit/reference renders with much more geometry usually indicate payload/variant/purpose traversal, instancing, transform stack, or material-subset coverage gaps before shading begins.
- Direct sun or shadow bands visible in direct/visibility AOVs but absent in beauty usually indicate sampling/reconstruction/denoiser architecture, not missing light source semantics.
- A stable residual pattern across several local changes usually means the active hypothesis class is wrong; reset to a broader architecture pivot rather than another constant or clamp tweak.

## Handoff

When handing off renderer correctness work, include:

- Reference renderer/settings and local renderer settings.
- Scene and camera used.
- AOVs captured and what each proved.
- Bugs fixed versus features still missing.
- Metrics for image difference, load counts, timings, memory, samples, and noise where relevant.

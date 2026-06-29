---
name: graphics-debugging-lab
description: Debug rendering, viewer, shader, material, camera, lighting, and GPU integration bugs with reproducible visual evidence. Use when scenes render blank, only appear after camera motion, show incorrect shading, diverge across backends, crash in GUI mode, or need pixel/screenshot verification.
---

# Graphics Debugging Lab

## Purpose

Turn visual bugs into small, repeatable experiments. Capture images, logs, camera/light/material state, and shader assumptions until the defect has one owner.

## Reproduction Ladder

1. Prove launch and load.
   - Run the smallest CLI path.
   - Capture stdout/stderr.
   - Confirm scene counts, bounds, materials, lights, textures, and selected render mode.

2. Reduce the scene.
   - Prefer a tiny asset with one or two meshes, one material, one light, and known bounds.
   - Keep the original asset available as a regression case.
   - If the bug depends on camera or light position, capture those values.

3. Capture pixels.
   - Add or use a screenshot/headless mode.
   - Capture multiple modes if available: raster, shadow, RT, normals, depth, segmentation, material debug.
   - For physically based defects, capture term AOVs before beauty output: base color, normals, roughness, metallic, opacity, emissive, direct, shadow visibility, indirect, depth, and tonemapped output.
   - For path-traced reconstruction defects, also capture unclamped/clamped direct, residual before and after denoise, filtered direct, variance, history/confidence, and hit distance when those buffers exist.
   - For GUI-only bugs, add a delayed window capture or run under offscreen/Xvfb when possible.

4. Isolate pipeline stage.
   - Scene/load issue: counts, transforms, bounds, up-axis, instancing, material binding.
   - Camera issue: view/projection matrices, initial dirty flag, resize, first-frame render.
   - Material issue: authored values, texture slots, sRGB/data texture classification, default fallbacks.
   - Shader issue: coordinate spaces, normal/tangent transforms, face winding, push constants, SSBO layout.
   - GPU issue: descriptor lifetime, pipeline layout, barriers, cache, swapchain, staging/readback.

5. Add a targeted debug view.
   - Use early shader returns for normals, base color, roughness, metallic, F0, light terms, ambient, NdotL.
   - Add CPU-side dumps for material structs and descriptor counts.
   - Avoid guessing from beauty output alone.

6. Compare against the reference deliberately.
   - Capture the same camera, resolution, exposure, tone map, color space, light/environment, sample count, and seed as the reference renderer when parity is the goal.
   - If the reference has more geometry, materials, shadows, or contrast, first prove load/material/light coverage before changing shading constants.
   - If direct or visibility AOVs contain expected light/shadows but beauty does not, inspect clamping, accumulation, split buffers, and denoiser inputs before changing light orientation or intensity.
   - Separate bugs from missing features and performance-driven degradations.

7. Validate the fix visually and mechanically.
   - Re-run the original repro.
   - Re-run the reduced scene.
   - Check screenshots are nonblank and stable across at least two camera/light positions when relevant.
   - Keep the debug mode only if it will be reused; otherwise remove it after diagnosis.

## Shader Debugging Checks

Audit these first for banding, inverted light, unstable shading, or view-dependent artifacts:

- Object/world/view space consistency.
- Normal matrix direction and transpose/inverse convention.
- Row-major vs column-major matrix multiplication.
- Face-varying normals vs vertex normals.
- Tangent basis normalization and handedness.
- sRGB conversion applied only to color textures, never normal/roughness/metallic/AO data.
- Host and GLSL struct sizes and field order.
- Push constant layout and alignment.
- Light direction convention: vector to light vs from light.

## Viewer Debugging Checks

For blank-first-frame or “renders only after pan” bugs:

- Initial render must happen after scene load and swapchain/widget creation.
- Dirty flags must be set after load, resize, camera seed, backend switch, material/environment load.
- Paint callbacks must not race active painters or render from stale buffers.
- GUI embedding must have a deterministic first image path, not only input-driven redraw.

For intermittent fallback/bounding-box display:

- Trace state transitions between loading, bbox placeholder, render-ready, render-failed.
- Log backend exceptions; do not swallow them into “unrendered” UI state.
- Rebuild acceleration/material descriptors after scene reload and variant/time changes.

## Material/PBR Debugging Checks

- Print authored material values before GPU upload.
- Confirm texture index meanings match shader slots.
- Confirm no-material fallback is visually neutral and explicit.
- Compare direct light, IBL diffuse, IBL specular, emissive, SSS/transmission separately.
- Clamp roughness to sane minimums but do not hide bad authored data without logging.
- Treat unknown nodegraphs as unsupported, not as valid white/default materials.

## Physically Based Transport Checks

- Confirm direct-light candidate tables match the lights or emitters the shader can actually evaluate.
- Confirm PDFs, MIS weights, power/intensity scaling, and light selection probabilities after filtering unsupported or invisible lights.
- Confirm shadow rays use the same geometry, instance visibility, alpha cutout, opacity, transmission, and sidedness policy as primary rays.
- Confirm emissive mesh sampling exists when authored scenes rely on area emitters; indirect-only discovery is usually too noisy for validation.
- Confirm environment, dome, distant, sphere, disk, and rect lights preserve authored units, exposure, normalization, color, shaping, and texture orientation.
- Confirm accumulation resets on camera/material/light changes and records seed, spp, max depth, and denoiser/filter state.
- Confirm direct-light and indirect-light buffers entering denoising are the same estimates validated by AOVs; otherwise add diagnostic AOVs for the actual pre/post-clamp and pre/post-filter buffers.
- Confirm tone mapping and color management are applied once, in the expected order, and never used to hide incorrect energy.

## Evidence to Keep

- Command used.
- Scene path and reduced fixture path.
- Camera, target, light/environment settings.
- Reference renderer/settings when visual parity is the goal.
- Before/after screenshots or image hashes.
- AOVs or term-level captures when diagnosing PBR, shadows, or GI.
- Logs showing load counts and GPU pipeline creation.
- Root cause tied to a specific layer.

## Anti-Patterns

- Debugging screenshots without controlling camera/light.
- Fixing shader beauty output by changing material loader defaults blindly.
- Tuning exposure or tone mapping before proving material, direct, shadow, and indirect terms.
- Hand-rolling visual comparisons when a screenshot or pixel check can be automated.
- Treating GUI and headless results as interchangeable without testing both paths.

---
name: renderer-image-comparison-testing
description: Design, implement, and run golden-image renderer comparison harnesses for graphics and viewer projects. Use when Codex needs to capture renderer output, compare Vulkan/OpenGL/Metal or other backends against golden images, refresh baselines, choose image-difference thresholds, wire tests into CI/CTest, handle display/GPU constraints, inspect visual diffs, or report residual risk for renderer correctness.
---

# Renderer Image Comparison Testing

## Purpose

Make renderer correctness testable with repeatable images, explicit baselines, useful failure artifacts, and honest environment reporting.

## Workflow

1. Define the comparison claim.
   - Decide whether the test proves one backend is stable against its own baseline, or whether multiple backends should match a reference backend.
   - Prefer backend-specific goldens by default; use cross-backend comparison only when output parity is an intended contract.
   - For physically based renderer work, name the reference renderer/settings separately from backend-specific regression goldens.
   - Name the visual surface under test: viewer screenshot path, native renderer CLI, offscreen render path, or material/shader debug mode.

2. Choose deterministic scenes.
   - Start with a tiny fixture: one or two meshes, fixed camera, fixed light/environment, fixed resolution, known bounds.
   - Add focused cases for PBR, textures, alpha, normals/tangents, curves, instancing, large transforms, and missing-material fallbacks.
   - Add physical parity cases for metallic/roughness grids, transparent or alpha-cutout shadows, emissive mesh sampling, camera/exposure/tone mapping, and environment lighting.
   - Keep reduced repro scenes after fixing visual bugs; they become regression cases.

3. Capture through the product path.
   - Exercise the same path users rely on unless the goal is explicitly a lower-layer renderer unit test.
   - For viewers, prefer a screenshot flag that constructs the real widget/render backend, frames the scene, pumps events, and reads the rendered framebuffer.
   - Avoid comparing stale buffers; confirm the captured image is from the current frame, current scene, selected backend, and requested resolution.

4. Control visual state.
   - Fix resolution, camera, target, focal length, time, render mode, clear color, lighting, tone mapping, color management, and environment assets.
   - For stochastic renderers, also fix and record samples per pixel, max depth, seed, accumulation frame, denoiser/filter state, and any adaptive-sampling setting.
   - Record backend name, scene path, dimensions, command, return code, stdout/stderr, renderer settings, and reference settings with every run.
   - Add a nonblank or minimum-signal check before trusting threshold comparisons.

5. Compare terms before beauty when validating physical accuracy.
   - Capture debug AOVs for base color, normals, roughness, metallic, opacity, emissive, direct light, shadow visibility, indirect light, depth, and tonemapped output when available.
   - Compare AOVs against the reference or reduced analytic expectation before accepting a beauty-image match.
   - Store term-level outputs and diffs beside beauty outputs so failures can be assigned to load, material, light, transport, tonemap, or readback layers.

6. Compare images with actionable metrics.
   - Validate format and dimensions before pixel comparison.
   - Compute RMS, mean absolute error, and maximum channel delta.
   - Emit a visual diff image beside the current output so failures can be inspected quickly.
   - Use strict thresholds for deterministic raster paths and looser thresholds only when driver, platform, or stochastic differences are expected and understood.
   - For stochastic paths, compare both image error and noise/stability metrics across repeated fixed-seed and changed-seed captures.

7. Manage goldens deliberately.
   - Store goldens under a stable tree such as `tests/golden/<backend>/<case>.png` or `.ppm`.
   - Store generated outputs, logs, summaries, and diffs under an ignored tree such as `tests/output/`.
   - Require an explicit `--update` or equivalent command to refresh goldens.
   - Review image diffs before committing refreshed baselines; never auto-update goldens during a normal regression run.
   - Version physical-reference metadata with the golden: reference renderer, scene revision, camera, exposure, tone map, color space, spp, seed, and feature limitations.

8. Handle backend availability.
   - Distinguish unavailable backends from failed renderers.
   - Skip unavailable backends only for known setup failures, such as missing optional bindings or platform-only backends.
   - Provide a strict mode for CI jobs where a requested backend must be present.
   - Do not hide crashes, bad images, missing output files, shader failures, or display initialization failures as skips.

9. Integrate with CI conservatively.
   - Register a cheap smoke test by default, such as harness `--help` or a parser self-check.
   - Make full image comparison opt-in when it requires GPU hardware, a display server, or platform-specific renderers.
   - Accept explicit display configuration (`DISPLAY`, `XAUTHORITY`, `QT_QPA_PLATFORM`, `xvfb-run`, or platform equivalents) instead of baking in a developer workstation path.
   - Write machine-readable summaries so CI can archive current images, diffs, logs, and metrics.

10. Triage failures by layer.
   - Launch failure: dependency, display, backend discovery, or platform problem.
   - Load failure: scene parser, asset path, material binding, texture discovery, or stage traversal problem.
   - Blank image: camera/framing, first-frame rendering, dirty flags, framebuffer readback, or sync problem.
   - High diff with valid image: shader math, material defaults, texture color space, lighting, shadow visibility, transport, tone mapping, or backend parity issue.
   - Intermittent diff: uninitialized data, async render timing, nondeterministic camera/light state, stochastic sampling, or driver variance.

## Harness Checklist

- Provide flags for backend list, case list, resolution, threshold, timeout, output directory, golden directory, update mode, strict backend mode, and reference backend.
- Provide flags or manifests for reference renderer metadata, AOV list, spp, seed, max depth, exposure, tone map, and color-management settings.
- Return nonzero for real failures and missing goldens unless the user explicitly allows missing baselines during first capture.
- Capture per-case logs even when the render process fails.
- Store a JSON summary with status, paths, metrics, and reasons.
- Keep generated outputs out of git while allowing golden images to be committed.
- Prefer a simple deterministic image format when dependencies matter; use a real parser and validate headers instead of ad hoc byte slicing.
- Make missing display or GPU requirements clear in docs and final handoff.

## Command Patterns

Use commands like these, adapted to the repo:

```bash
python3 tests/render_goldens.py --backend vulkan --update
python3 tests/render_goldens.py --backend vulkan,opengl,metal
python3 tests/render_goldens.py --backend opengl,metal --reference-backend vulkan
python3 tests/render_goldens.py --backend vulkan --strict-backends --display :99
ctest --test-dir build --output-on-failure -L golden
```

For first-time setup, run with `--allow-missing` only to validate capture and artifact paths before goldens exist.

## Final Handoff

Report:

- Harness files added or changed.
- Supported backends, cases, image format, metrics, and threshold.
- Reference renderer/settings and any term-level AOVs compared.
- Where goldens live and where generated outputs are written.
- Exact commands run and whether they captured real renderer output or only smoke-tested the harness.
- Environment limitations such as unavailable Metal hardware, blocked Wayland/X11, missing Xvfb, or skipped optional bindings.

## Anti-Patterns

- Updating goldens automatically after a failing comparison.
- Comparing images from a code path users do not exercise.
- Treating backend-specific legitimate differences as bugs without defining parity as a goal.
- Accepting a beauty match when material, shadow, direct, indirect, or tonemap AOVs still disagree.
- Using only pass/fail output without writing the current image, diff image, log, and metrics.
- Skipping crashes or missing images as if the backend were merely unavailable.
- Hiding local display assumptions in the harness.

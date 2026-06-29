---
name: renderer-stalled-parity-triage
description: Reset renderer/reference parity work when beauty images, scalar metrics, repeated local tweaks, or narrow target slices stop improving. Use when signed residuals stay structured, AOVs disagree with beauty, noise or denoising hides lighting/shadow detail, equal-spp comparisons are misleading, reference metadata is uncertain, or several exposure/material/light/denoiser/transport experiments have been rejected without progress.
---

# Renderer Stalled Parity Triage

## Purpose

Stop unbounded beauty-image tuning when evidence says the problem is structural, stochastic, architectural, or under-specified. Turn a stalled comparison into testable hypotheses with clear stop conditions. If narrow hypotheses keep failing, switch to a no-sacred-cows architecture pivot instead of making the next edit smaller.

## Trigger

Use this workflow when any of these are true:

- Beauty output has not materially improved after two or more targeted experiments.
- Several target-shaped vertical slices have been rejected and the same residual structure remains.
- Local patches are making the code harder to reason about while preserving the same visual failure.
- RMS/MAE improves but signed residuals, crops, or AOVs still show the same visual failure.
- Residuals are spatially structured: missing bands, contact shadows, highlights, instances, silhouettes, texture regions, or lighting gradients.
- Noise, fireflies, clamping, or denoising changes the apparent result more than the semantic change under test.
- The reference renderer's camera, post settings, color management, light policy, or feature support is not fully known.
- Frame time or driver instability prevents rapid iteration on the current path.

## Workflow

1. Freeze the current baseline.
   - Record command, scene, camera, reference, resolution, spp, max depth, seed, exposure, tone map, denoise/filter, texture/resource caps, backend, frame time, and artifact paths.
   - Save raw/linear output when possible, plus tonemapped beauty, signed residual, and the most relevant AOVs.
   - List recent rejected experiments and the evidence that rejected each one.

2. Read the residual as a map, not a score.
   - Divide the image into named regions or feature classes: foreground, background, high-error clusters, shadow bands, highlights, thin geometry, transparent regions, interiors/exteriors, sky/environment, and material groups.
   - Mark whether each region is too bright, too dark, too noisy, too smooth, spatially shifted, missing structure, or wrong color.
   - Use crops and local statistics so small but important features are not averaged away.

3. Separate semantic mismatch from convergence and reconstruction.
   - Compare raw/no-denoise and denoised outputs.
   - Compare direct, visibility, environment, indirect, normal, depth, albedo/basecolor, material ID, and opacity/transmission AOVs when available.
   - If a term AOV contains the missing structure but beauty loses it, capture the exact buffers entering reconstruction: unclamped and clamped direct, residual before denoise, residual after firefly clamp, direct after filtering, final composite, variance, and any history/confidence data.
   - If a reuse or reservoir candidate AOV looks plausible but beauty regresses, classify the pass as scaffolding until candidate validity, target PDFs, final visibility, merge weights, and the residual/direct composite contract are proven together.
   - For path-traced lighting failures, classify direct illumination, GI, and denoising separately. Direct-light instability points toward sampling/reuse/final visibility; indirect fireflies point toward GI reuse/path guiding/variance controls; denoiser blur points toward missing guides, hit distances, or signal splits.
   - Run fixed-seed and changed-seed captures to distinguish deterministic bias from variance.
   - Prefer equal-time or equal-budget comparisons for path-traced changes; equal-spp comparisons can mislead when one method changes variance or frame time.

4. Build a hypothesis ledger.
   - For each proposed change, write the predicted region/AOV effect, smallest test, pass signal, fail signal, and expected frame-time impact.
   - Do not repeat a rejected experiment unless new evidence invalidates its rejection.
   - Change one meaningful axis at a time unless the hypothesis requires a coupled change.
   - When an experiment improves an internal AOV but not beauty, keep the artifact and rejection reason in the ledger. Do not continue tuning local weights unless the next hypothesis predicts a specific regional change in the signed residual.
   - If the ledger contains multiple rejected variants of the same subsystem, mark that subsystem as suspect. The next hypothesis should be allowed to replace it, not just adjust it.
   - If replacing the suspect subsystem is the first test that can falsify the new hypothesis, do that replacement instead of inserting another transitional knob.

5. Classify the next pivot.
   - If load/material/light/count AOVs are wrong, return to scene extraction or material/light fixtures.
   - If direct/visibility AOVs imply plausible light but the residual is spatially blocked or opened in the wrong places, check geometry visibility semantics before light tuning: inherited visibility, purpose, shadow flags, opacity, and authored versus schema-fallback sidedness such as Mesh `doubleSided`.
   - If visibility AOVs contain the missing structure but beauty does not, inspect sampling allocation, MIS, clamping, denoising, and accumulation.
   - If direct terms are plausible but indirect is noisy or firefly-prone, prioritize transport variance reduction, path guiding, sample allocation, or reconstruction.
   - If direct and indirect term AOVs both show useful signal but beauty is flat, stop local tuning and plan a target-shaped reconstruction slice: stable direct-light estimates, stable GI estimates, and a denoiser input contract with normal/depth/albedo/roughness/motion/hit-distance/history guides.
   - If raw output is plausible but tonemapped output is not, prioritize reference post settings, exposure, color management, and output transforms.
   - If denoising erases structure or spreads outliers, prioritize guided spatial/temporal denoising with normal/albedo/depth/history inputs.
   - If iteration is too slow, prioritize resident rendering, reduced probes, hardware RT/ray query, pass gating, or smaller representative fixtures before more beauty sweeps.
   - If reference metadata is missing, mark parity blocked or degraded until it is recovered or an explicit approximation contract is written.
   - If local fixes keep preserving the same wrong image, pursue a radical but evidence-backed change: replace the sampler, reconstruction contract, denoiser, scene extraction policy, or fixed checkpoint shape. Keep the old path only as a comparison baseline or rollback point.

6. Exit with a decision.
   - Name the best current hypothesis, the rejected hypotheses, and the next smallest command.
   - State whether the next move is correctness, sampling/convergence, denoising/reconstruction, reference metadata, performance architecture, or a broader subsystem replacement.
   - When the decision is a broader replacement, name the old path to bypass or retire and the minimum artifact that will prove the new architecture is worth keeping.
   - Leave a matrix row or handoff note so the next agent does not restart the same scalar sweep.

## Anti-Patterns

- Continuing exposure, tone map, light-scale, material-mode, or denoiser sweeps after the residual pattern has stopped changing.
- Claiming progress from a scalar metric while the same regional failure remains visible.
- Comparing only equal spp when one method is much slower or has different variance.
- Treating denoising or clamping as a substitute for plausible transport AOVs.
- Reusing a direct AOV as proof of beauty correctness when the beauty path clamps, subtracts, filters, or composites a different buffer.
- Ignoring unknown reference renderer settings while tuning local constants to match one image.
- Keeping rejected experiments only in prose without a clear reason not to repeat them.
- Treating the existing renderer architecture, denoiser, sampler, or comparison setup as a constraint after evidence says it is the problem.
- Responding to repeated failed slices by shrinking scope again instead of changing the hypothesis class.

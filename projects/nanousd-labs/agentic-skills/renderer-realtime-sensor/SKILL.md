---
name: renderer-realtime-sensor
description: Design real-time, headless, multi-camera renderers that produce reproducible sensor data for simulation and RL training. Use when Codex builds or reviews a renderer feeding a physics/RL loop (IsaacLab, Newton, MuJoCo), works on determinism/reproducibility, sensor AOVs (depth/segmentation/normals), multi-camera tiling, or the fidelity-versus-throughput target for a training sensor rather than an offline beauty render.
---

# Real-Time Sensor Renderer

## Purpose

A renderer that feeds a simulation or RL training loop is a *sensor*, not a picture maker. Its job is to produce reproducible, semantically-labeled observations at the rate the loop consumes them. This skill is for that path: place the renderer on the fidelity/throughput/determinism spectrum, make that placement a stated target, and hold the architecture to it. It does **not** assume throughput beats fidelity — a high-fidelity sensor is a legitimate target — but it insists the choice be explicit and that reproducibility be a contract, not an accident.

This is the sibling of `renderer-target-architecture-design` (target-first thinking) and `physically-based-renderer-validation` (reference parity). Use those when the output is judged by an eye; use this when the output is consumed by an algorithm.

## Name the target first

Before architecture, place the renderer on the spectrum and write it down:

- **Offline / beauty** — reference parity, fidelity over speed. (Use the PBR-validation path, not this one.)
- **Interactive / viewport** — responsive, good-enough fidelity.
- **Sensor / throughput** — many cameras, fixed budget, fidelity only as high as the task needs.
- **Diagnostic** — AOVs and probes, not a final image.

The fidelity ↔ throughput ↔ latency tradeoff is the user's to set, not the renderer's to assume. Record the chosen point and the budget it implies (cameras, resolution, ms/step, memory). Every later decision is justified against *that* target. Do not silently optimize a chosen-fidelity sensor into a fast-but-wrong one, nor burden a chosen-throughput sensor with beauty work it never asked for.

## Determinism is a contract

For a training sensor, reproducibility is usually a hard requirement: the same inputs must produce the same observation, run to run and — within a stated tolerance — backend to backend. Nondeterminism shows up as unexplainable variance in reward curves, not as a visual bug.

- Fix and record every stochastic input: seed, sample count, accumulation frame, time/sub-step, camera and light state. A sensor that needs many samples to converge is already in tension with a throughput target — prefer low-variance or analytic shading on that path.
- Treat floating-point and GPU-scheduling nondeterminism as first-class, not noise. Reductions, atomics, and unordered accumulation can break run-to-run equality even with a fixed seed.
- Define cross-backend tolerance explicitly (exact, or bounded RMS / max-delta) and decide whether determinism is per-backend or across backends. Deliberate variation (domain randomization) must be *seeded and reproducible*, not ambient.
- Color management, sRGB encoding, and matrix/transform conventions must be identical across backends, or "the same scene" yields different observations (see `nanousd-renderer-scene-extraction` for the transform-convention trap).

## Sensor AOVs are first-class outputs

Depth, segmentation, and normals are the product, not debug views — design them up front with the same rigor as color.

- Give every AOV defined units and ranges (linear eye depth vs normalized, world vs view normals, segmentation as stable IDs) and document them where the consumer reads them.
- Segmentation IDs must be stable across frames and tie back to USD paths/prims, so a label means the same thing every step. Decide instance vs prototype vs semantic ID up front.
- Keep AOVs consistent with the beauty path's geometry, visibility, and instancing — a depth/seg buffer that disagrees with color is a silent corruption of the training signal.
- Produce only the AOVs the consumer requested; gating the rest is also where the throughput budget is won or lost.

## Architecture for the chosen target

- **Many cameras, few dispatches.** Tiled / multi-camera rendering in one submission, with a per-environment output layout the consumer can read without a de-tile pass. One submit, one sync, one hand-off.
- **Fast-mode shading is a target, not a hack.** When the target is throughput, a flat/analytic shading path with no stochastic sampling is the *correct* architecture, gated as such — not an apology.
- **Keep the consumer's data on the device.** A loop that bounces pixels GPU→CPU→GPU has already lost; the output contract is "where the algorithm reads it" (a device tensor), not a saved image. Cross-API/runtime memory sharing belongs in the `renderer-gpu-interop` sibling skill.
- **Per-frame updates, not reloads.** Transforms and visibility from the physics state update narrowly each step; never rebuild the scene per frame. The physics→render contract belongs in the `renderer-simulation-loop` sibling skill.

## No sacred cows — scoped to the chosen target

The "replace the subsystem when evidence warrants" stance from `renderer-target-architecture-design` still applies, but to *whichever target was chosen*. A throughput pivot must not quietly wreck a chosen-fidelity sensor, and a fidelity pivot must not blow a chosen-throughput budget. Replacement is justified when the architecture cannot reach the *stated* target — not when it fails a target nobody picked.

## Validate as a sensor, not a picture

- **Reproducibility:** identical inputs, two runs (and two backends if cross-backend parity is the contract) → equal within the stated tolerance. Fixed-seed and changed-seed captures separate bias from variance.
- **AOVs:** depth/segmentation/normals validated against a reduced analytic scene for units, ranges, and stable IDs before anything downstream trusts them.
- **Budget:** cameras × resolution at the target ms/step and memory, with telemetry — a sensor that misses its rate is failing its target even if it looks right.
- Compare equal-time, not equal-spp, when changing sampling: a slower "better" sensor may be the wrong trade for the chosen target.

## Anti-Patterns

- Assuming throughput beats fidelity (or the reverse) instead of recording it as a target choice.
- Treating run-to-run or cross-backend nondeterminism as acceptable noise in a training sensor.
- Shipping depth/segmentation/normals as afterthoughts with undocumented units or unstable IDs.
- A de-tile or GPU→CPU→GPU bounce on the hot path of a throughput sensor.
- Rebuilding the scene each physics step.
- Letting AOVs diverge from the beauty path's geometry, visibility, or instancing.
- Optimizing a chosen-fidelity sensor into a fast-but-wrong one because a throughput maxim said so.

## Handoff

Report the chosen target and budget (cameras, resolution, ms/step, memory), the determinism contract (seeds, tolerance, per- or cross-backend), the AOV set with units and ID semantics, what runs on the device vs the CPU, and the reproducibility + budget evidence. Flag interop and sim-loop specifics for the `renderer-gpu-interop` and `renderer-simulation-loop` sibling skills.

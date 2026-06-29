---
name: renderer-simulation-loop
description: Drive a renderer from a physics/RL simulation loop (IsaacLab, Newton, MuJoCo) — per-step transform/visibility updates from physics state, time handling, frame pacing, and the consumer contract. Use when Codex integrates a renderer into a training/sim loop, plumbs physics transforms or time into the scene, or debugs bodies-render-at-origin, stale-frame, per-env update, or pacing problems.
---

# Renderer In The Simulation Loop

## Purpose

A renderer inside a physics/RL loop is *updated, not reloaded* every step, and is fed by a physics engine whose conventions are not the renderer's. This skill is the contract between the two: how per-step state flows in, how time is handled, how frames are paced, and the failure modes that silently feed the policy a wrong or empty observation. Pairs with `renderer-realtime-sensor` (the sensor-output side) and `renderer-gpu-interop` (the zero-copy hand-off).

## The transform convention is the single most expensive bug

- Physics→render transform plumbing must agree on **both** row-vector vs column-vector **and** row-major vs column-major. The catastrophic, silent failure: applying a transpose under the wrong assumption picks up the always-zero last row of a rigid transform, **zeroing every instance's translation** — every body renders stacked at the origin, the sensor sees nothing, and there is no error. In the field this shows up as "the TLAS has N instances but the depth AOV reports 0 ray hits."
- Establish the producer's convention once (e.g. column-vector row-major, as numpy/Warp emit), apply **one named conversion** at the renderer boundary, and verify with a translate-and-re-render test — never a static frame, which passes tautologically. See `nanousd-renderer-scene-extraction` for the generic transform trap; this is its sharpest sim-loop manifestation.

## Per-step updates flow through narrow dirty paths, never a reload

- A transform / visibility / color / time edit sets a **narrow** dirty bit and updates in place; never rebuild the scene or reload geometry per step (see `renderer-performance-architecture` for the dirty-type separation). The non-obvious residue at multi-environment scale: when the TLAS is *partitioned per environment*, globals shared across all envs are refitted only via env 0's slot — because identical entries already populate every env's partition, env-0's transform copy is correct for all of them. Mishandling this re-scatters or drops shared static geometry.

## Time is an explicit input with an "authored default" sentinel

- A renderer on a physics clock needs a current-time input, but its **default must mean "evaluate at the USD-authored default time,"** not literally `t = 0` — defaulting to 0.0 silently re-times every time-sampled attribute. Use a NaN (or equivalent) sentinel for "unset → authored default," and re-evaluate time-varying transforms / geometry / visibility whenever the time changes.

## Frame-skip is a throughput lever — a target choice, valid only if dirty bits accumulate

- Because the scene barely changes between physics steps, rendering every Nth call and returning the cached frame can roughly halve trace cost with no measured policy impact. It is safe only if skipped frames (a) leave the buffer ping-pong indices *untouched* and (b) **accumulate** pending dirty bits ("latest wins"), so the next real render applies the latest state. Whether to enable it is a *target decision* (see `renderer-realtime-sensor`), validated against the policy — not assumed.
- **Pacing:** end-of-frame should `commit` without blocking; pay the GPU wait *lazily* in the readback/map call, and have the next begin-frame wait on the prior in-flight buffer. This overlaps host and GPU work without exposing fences to the caller; a naive commit-and-wait-every-frame serializes the two and halves throughput.

## Validate against the loop, not a single frame

- Drive the renderer from a scripted physics trajectory and assert the *observation tracks it* — a moved body moves in the depth/segmentation AOV, not just in color. Run the reproducibility checks from `renderer-realtime-sensor` across loop steps. The translate→re-render→expect-motion test is the cheapest guard against the origin-collapse bug.

## Anti-Patterns

- A flat `memcpy` of physics transforms across a row/column-vector or major boundary (silent origin collapse).
- Rebuilding or reloading the scene per physics step.
- Defaulting current time to `0.0` instead of an authored-default sentinel.
- Frame-skip that resets ping-pong indices or drops accumulated dirty bits.
- Commit-and-wait every frame instead of deferring the wait to the consumer.
- Validating animation from a single static frame.

## Handoff

Report the physics→render transform convention and the one conversion point, which per-step edits map to which narrow dirty paths (including any per-env partition rule), the time-input semantics, whether frame-skip is enabled and how it was validated, and the moved-body-moves-in-the-AOV evidence. Defer interop/teardown to `renderer-gpu-interop` and sensor AOV/determinism to `renderer-realtime-sensor`.

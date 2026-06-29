---
name: verification-led-development
description: Plan and execute code changes with focused verification at each layer. Use when implementing fixes or refactors that need builds, tests, smoke runs, screenshots, logs, benchmarks, or clear residual-risk reporting before handoff.
---

# Verification-Led Development

## Purpose

Make every meaningful change earn trust with the cheapest relevant signal. Prefer small, layer-owned checks over broad, slow rituals that do not exercise the changed behavior. Small verification scope does not imply small implementation scope: a broad rewrite is valid when it is the smallest coherent way to test or reach the target architecture, as long as it has staged checks and clear rollback/baseline evidence.

## Verification Workflow

1. Define the claim.
   - State what should now be true.
   - Identify the owner layer.
   - Identify the smallest artifact that can prove it.
   - State whether the change is a local patch, a target-shaped slice, or a subsystem replacement, and why that scope is justified by the evidence.

2. Choose checks by risk.
   - Build check for compile/link/interface changes.
   - Import/syntax check for Python/package changes.
   - Unit or smoke test for local behavior.
   - End-to-end run for cross-layer integration.
   - Screenshot/pixel/log check for visual behavior.
   - AOV/reference-image check for renderer material, shadow, lighting, GI, tone mapping, or stochastic sampling changes.
   - Benchmark only when performance is part of the claim.

3. Run checks in dependency order.
   - Core library first.
   - Bindings second.
   - Viewer/client last.
   - Do not test an adapter against a stale library unless that is the intended compatibility case.

4. Inspect outputs, not just exit codes.
   - Read warnings that may signal ABI drift or stale build paths.
   - Verify the expected target was produced.
   - Confirm logs show the intended code path.
   - Confirm screenshots are nonblank and from the expected scene/mode.
   - For renderer captures, confirm the output records camera, resolution, color management, tone map, sample count, seed, and backend.

5. Report residual risk.
   - Say what passed.
   - Say what was not run and why.
   - Mention environment assumptions such as display, GPU, network, cache, or unavailable hardware.

## Check Selection

Use this mapping:

- Build script changed: run that script from the repo root.
- CMake target changed: clean or configure build; ensure old targets disappeared when deletion was intended.
- Native API changed: rebuild library and run a binding import/smoke.
- Python binding changed: `python -m py_compile` or `compileall`, then import/use the installed package.
- Viewer CLI changed: run `--help` and one noninteractive path.
- Renderer output changed: run a screenshot or benchmark smoke on a known small scene.
- Physically based renderer output changed: run term-level AOV or reference-golden checks before relying on beauty images.
- GUI state changed: run offscreen capture when available, then note if live GUI was not exercised.

## Useful Commands

Prefer commands that match repository conventions:

```bash
./build.sh --no-clean
./build.sh
./run.sh --help
python3 -m compileall -q path/to/package
ctest --test-dir build --output-on-failure
```

For visual smoke paths, prefer explicit output files:

```bash
python3 test/bench_render.py scene.usda
QT_QPA_PLATFORM=offscreen ./build/nanousdview --screenshot /tmp/smoke.ppm scene.usda
```

## Verification Integrity

- Do not mark work done because compilation reached a target before a later packaging step failed.
- Do not trust old build directories after deleting targets; clean at least once.
- Do not ignore warnings about missing modules, stale symbols, or fallback targets.
- Do not use tests that exercise removed legacy paths as proof of the new path.
- Do not use performance wins as proof of renderer correctness; physical terms and reference parity need separate evidence.
- Do not shrink implementation scope just because broad changes are harder to review when repeated small changes have already falsified the local hypothesis.
- Do not leave long-running servers/processes open unless the user asked for them.

## Final Handoff

Keep the handoff short and concrete:

- Changed files or areas.
- Behavior now provided.
- Commands run and outcomes.
- Known gaps or checks not run.
- Next action when it directly follows from the work.

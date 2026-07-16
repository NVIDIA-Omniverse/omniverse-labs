---
name: nanousd-real-to-sim
description: Turn Gaussian splat PLY or PlayCanvas SOG/LOD assets into registered interactive NanoUSD robotics scenes on Apple Silicon. Use when Codex must inspect a splat, select objects through stable IDs, build colliders and support graphs, fit drawer or door joints, propose hidden interiors, generate voxel collision meshes, compile USDA, verify gates, or prepare deterministic traces for visual-agent RLVR.
---

# NanoUSD Gaussian Real-to-Sim

## Purpose

Operate the local `nanousd-rts` lab as an evidence-producing scene author. Keep
measured Gaussian appearance, physical proxies, and generated completion priors
separate. Treat the M5 lane as a fast development oracle and use Isaac/PhysX for
final dynamics promotion.

## Locate the lab

Resolve the experiment relative to this checkout:

```bash
LAB=projects/nanousd-labs/nanousd-metal-renderer/experiments/agentic_real_to_sim
RENDERER=projects/nanousd-labs/nanousd-metal-renderer
export PYTHONPATH="$RENDERER/experiments/agentic_real_to_sim/src"
PYTHON=projects/nanousd-labs/.venv/bin/python
```

Do not embed a username, another worktree, or a Cybernetic Physics checkout in
source or generated manifests. `NANOUSD_METAL_RENDERER_ROOT` is the only renderer
root override.

## Workflow

1. Run `nanousd-rts doctor`.
   - Build NanoUSD and `nanousd-metal-renderer` if the local library or Gaussian
     smoke binary is missing.
   - Run `test_gaussian_render`; the stable ID count must be nonzero.

2. Ingest one immutable visual source.
   - Use a standard 3DGS PLY directly.
   - For SOG/LOD directories, start at the lowest useful LOD and record the
     `splat-transform` conversion.
   - Never rewrite `workspace/source/source.ply`.

3. Render before authoring physics.
   - Produce RGB, depth, normal, and stable Gaussian-ID AOVs.
   - Adjust the camera when the image is blank or exterior-only; do not infer
     object bounds from an unverified frame.
   - Keep zero as background and map nonzero ID `n` to immutable PLY row `n - 1`.
   - Treat these frames as evidence renders, not beauty output. For SOG/LOD inputs,
     retain the original streamed hierarchy for the final visual viewer instead of
     judging source quality from the reduced NanoUSD working LOD.

4. Create semantic nodes from evidence.
   - Prefer a stable-ID render mask for irregular objects.
   - Use an AABB for coarse static structure or when no segmentation exists.
   - Keep the node selection and collider update atomic.
   - Mark static support surfaces as `support`, enclosing furniture as `shell`,
     and collision-blocking objects as `solid`.

5. Build the support graph.
   - Run `infer-support`, inspect all unresolved relations, and use `set-support`
     for explicit corrections.
   - Reject cycles.
   - Use containment for drawers/doors inside furniture and surface support for
     furniture/objects on floors or tables.

6. Fit articulation candidates.
   - Fit prismatic joints for drawers and revolute joints for doors/lids.
   - Inspect confidence, axis sign, origin, and limits.
   - Override uncertain values explicitly; heuristics are candidates, not truth.
   - Run a sweep immediately after each fit.

7. Complete hidden geometry without laundering provenance.
   - Run `propose-completions` only after fitting the joint.
   - Compare multiple generated candidates.
   - Every generated PLY must remain marked `measured=false`.
   - Split articulation completions into a world-attached static cavity and a
     joint-attached moving backside, liner, bin, rack, or drawer box.
   - When an unobserved source aperture would remain as a black/noisy void, record
     an explicit accepted-completion occlusion volume and remove those splats only
     from the streamed background derivative. Never include that volume in the
     measured moving-object extraction.
   - Accept one candidate explicitly. Acceptance must update the collider and pass
     the articulation sweep; otherwise leave all candidates unpromoted.

8. Build dense physical proxies when useful.
   - Run `voxelize` on the full scene or a stable-ID node selection.
   - Use external fill for closed interior scans, floor fill for exterior scenes,
     and carve only with a known-free seed.
   - Preserve both the raw collision GLB and the registered GLB.
   - Require the fixed `splat-transform GLB (-x,-y,z) -> NanoUSD PLY (x,y,z)`
     adapter and a passing registration residual.

9. Compile and validate USDA.
   - Compile registered colliders, bodies, support relationships, and Physics
     joints.
   - Preserve source PLY, source checksum, selection asset, and accepted completion
     asset as custom provenance.
   - Render the USDA through NanoUSD and require nonzero meshes and pixels.

10. Verify and preview.
    - Run `verify`; any failed hard gate blocks promotion.
    - Open the dependency-free physics preview and scrub every joint in
      front/top/side views.
    - For SOG/LOD sources, run `experience-preview`, then `serve-preview`; do not
      open the streamed viewer through `file://`.
    - Keep original streamed SOG visual truth and reduced NanoUSD evidence/physics
      truth as separate registered lanes.
    - Check that stability was not achieved by freezing movable parts.

For the pinned Home Scan kitchen profile, author and scrub the complete visible
kitchen set before handoff:

```bash
"$PYTHON" -m nanousd_rts author-home-kitchen /tmp/nanousd-home-scan-rts
"$PYTHON" -m nanousd_rts verify /tmp/nanousd-home-scan-rts
"$PYTHON" -m nanousd_rts experience-preview /tmp/nanousd-home-scan-rts --budget 16
"$PYTHON" -m nanousd_rts serve-preview /tmp/nanousd-home-scan-rts \
  --host 127.0.0.1 --port 8765
```

This deterministic profile contains both refrigerator doors, the oven door, upper
and base cabinet doors, and kitchen drawers. Confirm every entry appears in the
joint selector, then visually open at least one appliance, one hinged cabinet, and
one drawer.

## Agent plans

Use `run-plan` for bounded multi-turn execution. Plans may contain only tools from
`nanousd-rts tools`; arbitrary shell actions are rejected. A failed action stops the
plan and writes `trace/plan-result.json`.

For RLVR, expose:

- the JSON tool catalog as actions;
- compact render/AOV observations and failed gates as turn feedback;
- `trace/operations.jsonl` as the rollout trace;
- verification gates plus external robot-task results as rewards.

Use `RealToSimEpisode` as the dependency-free local environment seam. Keep its
`trainer_reward` evaluator-side; send only `observation` back to the policy.
Attach per-turn `dense_score` values to Kevin-style future credit and use the
fail-closed `terminal_reward` for submission.

Pair fidelity and stability rewards. Keep source integrity, provenance, graph
validity, and simulator load success as hard gates.

## Checks

```bash
PYTHONPATH="$RENDERER/experiments/agentic_real_to_sim/src" \
  "$PYTHON" -m unittest discover \
  -s "$RENDERER/experiments/agentic_real_to_sim/tests" -v

cmake --build "$RENDERER/build" \
  --target nusd_renderer test_gaussian_render --parallel
"$RENDERER/build/test_gaussian_render"
```

Run one synthetic `demo` and one real-asset ingest/render before handing off.

## Handoff

Report the source hash and LOD, camera, node/selection counts, support edges,
joint candidates and overrides, accepted generated completions, voxel/GLB
registration transform and residual, USDA hash/load result, hard-gate vector,
interactive preview path, measured M5 timings, and the remaining Isaac/PhysX
promotion work.

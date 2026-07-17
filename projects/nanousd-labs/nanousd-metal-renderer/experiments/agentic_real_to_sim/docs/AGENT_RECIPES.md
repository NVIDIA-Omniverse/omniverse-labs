# Agent recipes

This is the human-facing operating guide for assigning work to Codex or another
tool-using agent. It intentionally uses an evidence-first loop: the source stays
immutable, physical proxies stay explicit, and anything invented to complete an
unobserved interior remains non-measured.

## The contract to give every agent

Put these constraints directly in the request:

1. Name the source path and a writable workspace path.
2. Require RGB, depth, normal, and stable-ID evidence before semantic selection.
3. Require the agent to state which representation is measured, physical, or
   generated after every material change.
4. For each movable part, require measured-only closed, half, and open evidence.
5. Require `verify` and the hard-gate vector before handoff.
6. Ask for source Gaussian count, streamed live budget, measured LOD0 articulation
   count, and generated-completion count separately.

The default end state is a local Metal development oracle. Passing it does not
claim Isaac/PhysX contact fidelity or robot-task success; those belong to the
external promotion lane.

## Recipe: request an independent result review

Use this as a separate, fresh-context assignment after implementation. Do not
include the implementing agent's diagnosis or a statement of what should pass.

```text
Independently review the real-to-sim result at <WORKSPACE> / commit <COMMIT>.
Use only the artifacts and commands below; do not assume any implementation
intent. Return exactly `pass`, `fail`, or `inconclusive`, followed by paths and
observations that support the verdict.

Acceptance rubric:
- source/source.ply remains immutable;
- measured-only closed, half, and open poses are coherent before generated
  interiors are considered;
- joint sweeps, segmentation review, and verify all pass;
- every generated completion/material is labelled measured=false;
- mesh, map, and binding hashes are present and valid;
- report source count, live budget, LOD0 count, and completion count separately.

Reproduce with:
<COMMANDS>

Inspect these paths:
<ARTIFACT_PATHS>

Known limitations that must remain explicit:
<LIMITS>
```

Save the verdict in `evidence/independent-review/`. If no fresh reviewer can be
launched, save this packet there and mark the independent verdict as pending;
local implementation checks are not a substitute.

## Recipe: author a scan into an interactive scene

Use this prompt when starting from a PLY or SOG/LOD scan:

```text
Use $nanousd-real-to-sim to author <SOURCE> into <WORKSPACE> for a robotics
interaction task. Preserve the source exactly. First run doctor, ingest, and render
RGB/depth/normal/stable-ID evidence. Build named semantic nodes from stable source
rows, then author support edges, colliders, and revolute/prismatic joints.

For every articulated node: run a joint sweep, create measured-only closed/half/
open visual evidence in the streamed preview, and accept only independent,
coherent movement. If a front is sparse or trim-heavy, use tangent occupancy and
positive/negative labels to refine it rather than guessing from a single view.

Only after the measured front is accepted, propose hidden interiors. Keep cavities
world-attached and moving backsides/liners joint-attached; mark them measured=false.
Compile USDA, render it, run verify, and hand back the interactive preview URL,
hard-gate vector, source/live/LOD0/generated counts, plus all remaining Isaac
promotion work.
```

Expected evidence:

- `evidence/render/*` AOVs and stable source IDs;
- selections under `selections/` with immutable source provenance;
- pose triplets and `evidence/segmentation/review.json`;
- `evidence/verification/report.json` with every local gate passing;
- `preview/index.html` and `exports/scene.usda`.

## Recipe: diagnose an interaction visually

Use this when a door looks static, a drawer moves the wrong object, or an opening
shows a black void:

```text
Investigate <NODE> in <WORKSPACE>. Do not modify the immutable source. Start with
the streamed preview in ?segmentation-review=1, where generated completions are
suppressed. Capture closed, half, and open poses with the selected front framed.
Compare them against its neighbours and report the measured pose deltas.

If the wrong splats move, refine only the selection: preserve positive and negative
reference labels, choose the planar peak by tangent occupancy, reauthor the LOD0
front, and recapture the triplet. If the measured front passes but the aperture is
unobserved, create a separately labelled generated cavity/interior. Re-run sweeps,
segmentation review, and verify; explain whether the fix affected measured,
physical, or generated truth.
```

## Recipe: compare learned materials

Use this after accepting a completion, not as a substitute for geometry review:

```text
For accepted completion <NODE> in <WORKSPACE>, run fit-mesh-pbr to create the
deterministic mesh/UV request. In the isolated Python 3.12 materials environment,
run official MatFuse and StableMaterials with the same seed. Generate the comparison
page, inspect both bundles, and import the selected one through
external-pbr-atlas-v1. Preserve MatFuse specular and StableMaterials height as their
native quantities; do not rename them metallic or AO. Hash-check mesh, maps,
request, and mesh bindings. Report provider/model revision, prompt, seed, output
maps, and explicit measured=false provenance for the completion.
```

## Recipe: turn the loop into an RLVR episode

Use this when the agent will be trained rather than manually supervised:

```text
Create a RealToSimEpisode for <WORKSPACE> with required nodes <NODES>, required
interactive nodes <MOVABLE_NODES>, and required mesh/PBR nodes <FIDELITY_NODES>.
Expose only nanousd-rts tools as policy actions. Feed compact render/AOV summaries,
tool outcomes, and failed gates to the policy; keep required-node checks and reward
evaluation hidden. Score source provenance, selection/collider association, support
graph validity, joint sweeps, visual segmentation acceptance, completion provenance,
and mesh/PBR binding completeness. Make terminal reward zero if any hard gate fails
or an interaction is frozen.
```

For a concrete Python seam, see `RealToSimEpisode` in the README's RLVR/Tinker
bridge. The policy receives public observations; `trainer_reward` stays evaluator
side so the target requirements are not leaked.

## Handoff checklist

Ask the agent to return this compact record:

| Item | What good looks like |
| --- | --- |
| Source | Immutable hash, LOD, source count, and transform |
| Selection | Node IDs, source-row selections, positive/negative references |
| Articulation | Joint type/axis/origin/limits, sweep report, pose triplet |
| Completion | Candidate ID, acceptance state, measured=false boundary |
| Materials | Provider, model revision, seed, map/binding hashes |
| Validation | Hard-gate vector, known local limitations, Isaac promotion work |
| Handoff | Preview URL, USDA path, trace, and evidence directory |

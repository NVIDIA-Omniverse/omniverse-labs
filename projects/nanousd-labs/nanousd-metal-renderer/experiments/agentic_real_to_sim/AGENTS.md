# AGENTS.md — agentic real-to-sim

Follow the repository root `AGENTS.md` and this experiment-specific contract.

## Evidence-first delivery

- Preserve `source/source.ply`; never overwrite, recolor, or relabel measured
  source Gaussians.
- Keep measured fronts, physical proxies, and generated completions distinct in
  code, manifests, screenshots, and handoff language.
- Do not let a generated cavity hide an unreviewed measured-front extraction.
- Run the narrowest relevant checks, then `verify`, before claiming a promotion
  candidate is ready.

## Independent verification gate

For a non-trivial visual, physics, material, or workflow change, use a fresh
subagent/reviewer that has not seen the implementation conversation. Give it only:

1. the exact artifact paths or commit/diff;
2. the acceptance rubric and expected invariants;
3. commands to reproduce checks and inspect the preview;
4. known limits that must remain honestly labelled.

Do **not** give it the implementer's diagnosis, expected conclusion, or a summary
of what is supposed to be fixed. Ask it to return `pass`, `fail`, or `inconclusive`
with artifact-backed reasons. Save its report under `evidence/independent-review/`
or include it in the PR handoff. A self-review does not satisfy this gate.

If the current environment cannot launch a subagent, create the same artifact-only
review packet and mark independent verification as pending. Do not say the result
was independently verified.

For articulation, the reviewer must inspect measured-only closed/half/open evidence
before inspecting generated interiors. For materials, it must verify map/binding
hashes and `measured=false` provenance. For viewer quality, it must distinguish
source Gaussian count, live budget, measured LOD0 articulation, and generated
completion counts.

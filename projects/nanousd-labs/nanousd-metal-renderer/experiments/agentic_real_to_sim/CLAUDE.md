# CLAUDE.md — agentic real-to-sim

Read `AGENTS.md` in this directory before modifying the experiment.

## Required final pass

Before declaring a real-to-sim result ready, ask a fresh-context reviewer/subagent
to evaluate raw artifacts, not the implementation narrative. Provide only the
workspace/commit paths, a short acceptance rubric, expected invariants, and
reproduction commands. Keep the implementer's diagnosis and desired outcome out of
the reviewer prompt.

Require an explicit `pass`, `fail`, or `inconclusive` verdict with links/paths to
the evidence it used. Store the report in `evidence/independent-review/` or attach
it to the PR handoff. If a fresh reviewer cannot be launched, leave this verdict
pending and say so plainly.

For Home Scan articulation: review measured-only closed, half, and open poses first;
then inspect generated interiors, map/binding hashes, provenance, sweeps, and the
hard-gate vector. Never report generated texture or hidden geometry as measured scan
quality.

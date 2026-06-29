---
name: ai-agent-orchestration
description: Coordinate multiple AI agents or subagents on software work without duplicating effort or causing conflicts. Use when Codex is explicitly asked to delegate exploration, implementation, review, testing, integration, or parallel analysis across a codebase.
---

# AI Agent Orchestration

## Purpose

Use additional agents to increase throughput and independence, not to outsource the critical path blindly. Bounded worker ownership is a coordination tool, not a claim that the solution must be small; broad architecture pivots can be split into disjoint ownership areas under one explicit design.

## Orchestration Workflow

1. Decompose the task.
   - Identify the immediate blocker that the lead agent should handle locally.
   - Identify sidecar tasks that can run independently.
   - Identify write scopes that can be made disjoint.
   - If evidence calls for a subsystem replacement, decompose by architecture boundary rather than shrinking the goal into unrelated local tweaks.

2. Assign ownership.
   - Give each worker a bounded module, file set, or question.
   - Tell workers they are not alone in the codebase.
   - Tell workers not to revert edits made by others.
   - Require workers to report files changed, tests run, and remaining risk.

3. Keep exploration concrete.
   - Good: “Find every path that imports `_stageView_pixar_orig.py` and report whether it is safe to delete.”
   - Bad: “Understand the whole repo.”
   - Good: “Patch only `nanousdview/python/.../usdviewq` to remove Hydra debugger artifacts.”
   - Bad: “Clean up legacy code wherever you see it.”

4. Avoid duplicate work.
   - Do not ask several agents the same unresolved question unless independent confirmation is the goal.
   - Do not redo a worker’s task locally while they are running.
   - Use local time for non-overlapping work.

5. Integrate deliberately.
   - Review returned diffs quickly.
   - Run repo-level searches for stale references.
   - Run focused tests that cross the boundary between worker changes and local changes.
   - Resolve naming/documentation consistency in the lead thread.

## Good Delegation Targets

- Inventory one submodule while the lead inspects another.
- Remove a bounded class of dead files in one package.
- Add tests for a module the lead is not editing.
- Compare shader/material implementations and report mismatches.
- Validate a skill, doc, or patch from a fresh perspective.

## Poor Delegation Targets

- The next action blocking the lead.
- A tightly coupled API redesign without a written contract.
- Broad refactors across shared files without a written contract, owner boundaries, and integration check.
- Tasks requiring user approval or live environment control unless planned.
- Anything where a merge conflict is more expensive than doing the work locally.

## Worker Prompt Template

Use a prompt like:

```text
You are not alone in the codebase. Do not revert edits made by others.
Your ownership scope is: <files/modules>.
Goal: <specific outcome>.
Constraints: <what not to touch>.
Verification: <commands or checks to run if possible>.
Report: files changed, tests run, remaining risks.
```

## Integration Checklist

- Worker touched only assigned scope.
- Returned changes match the architectural plan.
- No stale imports, build targets, docs, or generated references remain.
- Local and worker changes compose in one build/test run.
- Final report credits outcomes, not agent activity.

## Failure Modes

- Delegating vague “analysis” and getting unusable summaries.
- Letting workers edit overlapping files without ownership boundaries.
- Waiting on sidecar work while no local progress happens.
- Accepting a worker patch without running the system-level check.
- Reporting parallelism instead of verified software behavior.

---
name: extend-ovrtx-interactive-edits
description: Extend the OVRTX Blender add-on with new interactive transform, material, light, world, UV, camera, physics, or semantic edit families. Use when implementing Blender depsgraph extraction, typed USD value conversion, live runtime submission, durable write routing, topology fallback, user diagnostics, and tests.
license: "Apache-2.0"
metadata:
  author: "Max Bickley"
  version: "0.1"
  team: "omniverse"
  domain: "physical-ai"
  tags:
    - blender
    - omniverse
    - ovrtx
---
# Extend OVRTX interactive edits

## When to Use

Use when implementing Blender depsgraph extraction, typed USD value conversion, live runtime submission, durable write routing, topology fallback, user diagnostics, and tests.

## Boundary

Work in the distributed source checkout and use documented runtime interfaces.
Prove a missing runtime capability with the escalation probe before declaring
an add-on-only implementation impossible.

## Preserve the architecture

Route every edit through this pipeline:

`Blender depsgraph signal` → `InteractiveEdit` → `EditPlan` →
`InteractiveEditWorkflow` → `RuntimeScheduler` or durable USD write →
`OvrtxSessionUpdatePort`.

Do not call the native client from a Blender callback, UI operator, conversion
policy, or depsgraph builder. Do not mutate Blender data from the render thread.
Blender RNA reads and writes stay on Blender's main thread; runtime calls stay
on the serialized viewport session-owner thread. Queue work through the
scheduler and use the existing main-thread handoff in `engine.py` for results.

## Code map

- `addon/ovrtx_blender_example/blender_interactive_edit_builders.py`:
  extend `build_interactive_edits_from_depsgraph` and add a narrowly named
  builder beside `object_transform_edit`, `material_value_edits_from_resolver`,
  `light_value_edits_from_prim`, `world_value_edits_from_prim`, or
  `uv_value_edits_from_resolver`. Resolve authored prim identity and attach
  provenance; never infer a target from display names alone.
- `addon/ovrtx_blender_example/value_edit_conversion.py`: add or register
  a `ValueEditConversionPolicy` in
  `default_value_edit_conversion_policies`. Put domain-specific mapping in a
  focused `*_value_conversion.py` module. Return typed `UsdAttributeValue`
  records and one normalized classification: `supported`, `unsupported`,
  `non_rendering`, or `topology`.
- `addon/ovrtx_blender_example/interactive_edit_planner.py`: extend
  `InteractiveEditPlanner.plan` and `_value_update_kind` only when the new
  family needs a distinct routing rule. Keep value, topology, authority, and
  persistence separate in `InteractiveEdit`, `EditPlanImpact`, and `EditPlan`.
- `addon/ovrtx_blender_example/write_target_resolution.py`: use
  `resolve_write_target` for durable opinions. Fail closed on anonymous,
  session-only, ambiguous, mismatched, or topology-owning layer stacks.
- `addon/ovrtx_blender_example/interactive_edit_workflow.py`: preserve
  `InteractiveEditWorkflow.preview_edit` as the single choice point between
  update, composition replacement, durable write, and unsupported outcomes.
  Extend `_plan_diagnostics` or `_topology_fallback_diagnostics` when the new
  route has observable state.
- `addon/ovrtx_blender_example/runtime_scheduler.py`: queue accepted
  `EditIntent` values in `RuntimeScheduler.submit_edit`; apply them during
  `tick_viewport`. Keep view-authoritative updates in `ViewUpdateStream` and
  simulation-authoritative initial conditions in `SimUpdateStream`.
- `addon/ovrtx_blender_example/ovrtx_value_updates.py`: use
  `OvrtxTransformValue`, `OvrtxAttributeValue`, and `OvrtxSessionUpdatePort`.
  Add a new typed value and protocol method only when neither transform nor
  attribute semantics can represent the operation accurately.
- `addon/ovrtx_blender_example/topology_edit_fallback.py`: add a stable
  reason to `topology_reasons_for_edit` and preserve canonical coalescing and
  `topology_rekey_diagnostics` when an edit changes graph shape, prim type,
  relationships, membership, or authored object topology.
- `addon/ovrtx_blender_example/engine.py`: keep
  `_live_interactive_edit_depsgraph_handler`,
  `submit_depsgraph_interactive_edits_to_active_viewports`,
  `register_interactive_edit_bridge`, and the persistent file-load lifecycle as
  the integration boundary. Extend `submit_interactive_edit` only if the
  existing generic path cannot carry the result. Surface state through
  `interactive_edit_bridge_diagnostics` and viewport diagnostics rather than
  ad-hoc logging.

## Instructions

1. Define the exact Blender source field, authored USD target, USD type, data
   authority, and whether the change is a value or topology edit.
2. Build one `InteractiveEdit` per independently writable target. Include
   `blender_property_path`, exact `usd_prim_path`, `usd_property_name`, typed
   value metadata, and verified identity. Deduplicate repeated depsgraph signals.
3. Convert Blender values without reading global context. Reject linked,
   ambiguous, unsupported, or unproven values explicitly; do not silently
   approximate them.
4. Plan live preview separately from persistence. A live update may remain
   preview-only; a durable write requires a verified layer target. Never turn a
   missing write target into permission to modify exported source USD.
5. Submit value edits to the scheduler. Preserve edit wake-up, batching,
   refinement reset, failure propagation, and session identity. Ensure the
   native call executes only inside the session-owner render loop.
6. Route topology edits to scene-generation replacement or a verified selected
   write. Emit a stable topology reason and confirm composition identity changes
   once; never retry a rejected value update as an untyped topology mutation.
7. Add change-only user reporting and structured diagnostics containing source
   field, target identity, classification, route, status, reason, typed update
   counts, and topology re-key state. Do not report success before runtime
   acknowledgement.

## Required tests

Extend these existing suites:

- `tests/test_blender_interactive_edit_builders.py`: Blender source
  extraction, evaluated/original identity, exact prim mapping, deduplication,
  selection changes, and fail-closed cases.
- `tests/test_value_edit_conversion.py` plus a focused
  `test_<domain>_value_conversion.py`: units, typing, boundary values, linked
  inputs, unsupported values, and topology classification.
- `tests/test_interactive_edit_planner.py`: value/topology, authority,
  persistence, missing identity, and unsupported routing.
- `tests/test_interactive_edit_workflow.py`: scheduler update, write
  target, composition fallback, rejection, and diagnostics.
- `tests/test_runtime_scheduler.py` and
  `tests/test_latest_view_render_loop.py`: batching, edit wake, owner-thread
  application, ordering relative to simulation, refinement reset, and failures.
- `tests/test_write_target_resolution.py` and
  `tests/test_topology_edit_fallback.py`: layer ownership, ambiguous
  stacks, stable reasons, coalescing, and one-time re-key.
- `tests/test_interactive_edit_bridge_persistence.py`: registration and
  file-load survival when the handler path changes.

Add `tests/test_<feature>_interactive_edits.py` for end-to-end behavior.
Use fake Blender IDs and a fake typed update port to prove:

1. one supported drag produces the expected typed target and one queued update;
2. repeated callback noise coalesces without duplicate native writes;
3. a topology change replaces composition instead of entering the value lane;
4. unsupported or unresolved targets fail closed with actionable diagnostics;
5. runtime rejection is visible and is never recorded as applied; and
6. no runtime method runs on the Blender callback thread.

Run the focused tests first, then the complete documented test suite. Include
one small generic scene with a mesh, material, light,
and camera for Blender-level acceptance; do not depend on project-specific
assets.

## Runtime capability gate

Probe the documented runtime interface with the smallest typed update against one
known prim in a temporary scene. Record client/runtime versions, request type,
target, response, and diagnostics. Report a runtime capability gap only if the
interface cannot express the required operation, rejects a documented
supported typed operation, or cannot acknowledge application sufficiently to
avoid false success. Missing extraction, conversion, scheduling, fallback, UI,
or diagnostics remains add-on work.

## Handoff

Return a focused add-on commit, the tests above, generic acceptance evidence,
and an explicit `addon-only` or `runtime-capability-gap` conclusion.

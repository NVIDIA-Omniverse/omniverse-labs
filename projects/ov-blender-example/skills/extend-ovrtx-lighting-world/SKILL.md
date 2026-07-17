---
name: extend-ovrtx-lighting-world
description: Extend the OVRTX Blender add-on's light and World conversion, initial USD presentation, USD prim resolution, and interactive live edits. Use when adding Blender light fields or families, dome/World policies, topology classification, typed USD attributes, depsgraph routing, retained edits, or tests.
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
    - lighting
---
# Extend OVRTX Lighting and World

## When to Use

Use when adding Blender light fields or families, dome/World policies, topology classification, typed USD attributes, depsgraph routing, retained edits, or tests.

## Boundaries

- Change only add-on code and tests, using documented runtime interfaces.
- Keep initial authored values and live edits on one conversion policy. Do not add a render-only correction that disagrees with interactive updates.
- Route family, graph, texture, assignment, or prim-presence changes as topology. Use typed value writes only for fields whose USD prim and attribute already exist.
- Fail closed on missing, ambiguous, wrong-family, or unauthored USD targets.

## Instructions

Read the relevant lane end to end before editing:

1. Conversion policy:
   - `light_value_conversion.py`: `exported_light_family`, `authored_light_form`, `classify_field`, `usd_attribute_values`, unit/area/color helpers, `EDIT_VALUE_ATTRIBUTES_BY_FIELD`, and topology/unsupported reasons.
   - `world_dome_conversion.py`: `classify_field`, `world_dome_spec`, `usd_attribute_values`, `DEFAULT_DOME_OWNER_PATH`, graph/texture/assignment topology reasons.
   - `usd_value_edit_support.py` and `value_edit_conversion.py`: supported USD value types and shared classification/value records.
2. Initial authored presentation:
   - `light_scene_layer.py`: `scene_layer_from_lights` indexes stock-exported light prims and authors values through the light conversion policy.
   - `ovrtx_scene_composition.py`: composes presentation layers without changing the base generation.
   - World dome authoring and composition call sites found with `rg 'world_dome_conversion|scene_layer_from_lights' addon/ovrtx_blender_example`.
3. Target resolution:
   - `light_usd_prim.py`: `resolve_light_usd_prim`, supported families, authored form, and fail-closed match sources.
   - `world_dome_usd_prim.py`: `resolve_world_dome_usd_prim` and required dome attributes.
   - `usd_prim_resolver.py` and `write_target_resolution.py`: current-generation lookup and durable attribute ownership.
4. Live edit construction and execution:
   - `blender_interactive_edit_builders.py`: `light_value_edits_from_prim`, `world_value_edits_from_prim`, `_light_form_topology_edit`, `_world_topology_edit`, `_world_presence_topology_edit`, `_world_edits_for_update`, and `build_interactive_edits_from_depsgraph`.
   - `interactive_edit_planner.py`, `interactive_edit_workflow.py`, and `ovrtx_value_updates.py`: classify, retain, route, and apply typed writes.
   - `scene_generation.py` and `scene_generation_sessions.py`: topology regeneration and retained-value rebinding across generation handoff.

## Implementation

1. Add policy tests that state Blender input, classification, USD family/attribute/type/value, metadata, and unsupported/topology reason.
2. Extend `SUPPORTED_USD_ATTRIBUTES`, `EDIT_VALUE_ATTRIBUTES_BY_FIELD`, `classify_field`, and `usd_attribute_values` together. Keep clamping, units, normalization, and conversion-policy metadata deterministic.
3. For a new light family or form, update `exported_light_family`, `authored_light_form`, `USD_LIGHT_FAMILIES`, resolver authored-form detection, and `light_scene_layer._USD_LIGHT_TYPES` as one change.
4. Update initial presentation and composition. Verify it calls the same `usd_attribute_values` used by live edits and records every authored property.
5. Update prim resolution and durable write-target checks. Require the expected family and authored attribute; never select by display name alone when identity/path evidence exists.
6. Update live builders. Emit one typed value edit per converted USD attribute. Emit a topology edit for light-family/form changes, World node graphs/textures, World assignment, or dome presence divergence.
7. Update depsgraph recognition for light objects, light datablocks, World datablocks, node trees, or scenes only as needed. Avoid re-emitting World values for unrelated scene updates.
8. Verify retained edits rebind only when Blender identity, USD prim, family, and attribute remain compatible after regeneration.

## Tests

Use these focused suites:

- `tests/test_light_value_conversion.py`: light classification and numeric policy.
- `tests/test_blender_interactive_edit_builders.py` and `tests/test_update_stream.py`: World conversion, typed edits, and update-stream evidence.
- Add `tests/test_world_dome_conversion.py` or `tests/test_light_scene_layer.py` when the change needs direct policy or initial-layer coverage not already represented by those integration tests.
- `tests/test_light_usd_prim.py` and `tests/test_world_dome_usd_prim.py`: target resolution and failure reasons.
- `tests/test_blender_interactive_edit_builders.py`: depsgraph-to-edit routing and topology/value separation.
- `tests/test_interactive_edit_planner.py`, `tests/test_interactive_edit_workflow.py`, and `tests/test_scene_generation_sessions.py`: write, retention, regeneration, and replay.

Confirm exact filenames with `rg --files tests | rg 'light|world|interactive_edit|scene_generation'`, then run:

```bash
python -m pytest -q \
  tests/test_light_value_conversion.py \
  tests/test_light_usd_prim.py \
  tests/test_world_dome_usd_prim.py \
  tests/test_blender_interactive_edit_builders.py \
  tests/test_interactive_edit_planner.py \
  tests/test_interactive_edit_workflow.py
python -m pytest -q tests
```

Run the documented validator with a clean Blender profile when an installed runtime is available. Require both initial-render evidence and a same-session live-edit probe; neither substitutes for source conversion and resolver tests.

## Acceptance

- Initial presentation and live edits produce identical typed USD values from the same Blender state.
- Each supported edit resolves one existing prim, family, durable layer, and authored attribute.
- Topology changes regenerate instead of entering the typed-write lane.
- World absence, complex graphs, and environment textures have explicit deterministic outcomes.
- Candidate generation replay preserves accepted values and rejects incompatible rebinding.
- Runtime binaries remain unchanged.

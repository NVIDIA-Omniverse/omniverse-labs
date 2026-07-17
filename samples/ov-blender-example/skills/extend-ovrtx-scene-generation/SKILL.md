---
name: extend-ovrtx-scene-generation
description: Extend the OVRTX Blender add-on's authored-scene generation for evaluated geometry, mesh topology, modifiers, Geometry Nodes, and object instances. Use when changing Blender-to-USD export coverage, identity mapping, topology fingerprints, sparse generation reconciliation, or tests.
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
# Extend OVRTX Scene Generation

## When to Use

Use when changing Blender-to-USD export coverage, identity mapping, topology fingerprints, sparse generation reconciliation, or tests.

## Boundaries

- Work in the distributed source checkout and use documented runtime interfaces.
- Preserve stock Blender USD export as the geometry authority. Add-on code may prepare temporary evaluated objects, normalize exported USD, validate identity, and compose add-on opinions.
- Keep temporary authoring reversible: restore modes, selection, active object, datablocks, and generated node groups in `finally` blocks.

## Instructions

Read these files before editing:

1. `addon/ovrtx_blender_example/scene_generation.py`
   - `_stock_export`: fixed stock USD export contract (`evaluation_mode="VIEWPORT"`, tessellated subdivision, triangulation, relative assets).
   - `_temporary_export_identities`: temporary object, mesh, and material session-UID properties.
   - `_temporary_particle_hair_curves`: model for temporary evaluated-geometry preparation and cleanup.
   - `_normalize_stock_export_geometry`: deterministic post-export normalization.
   - `_validated_blender_prim_paths`, `_schema_paths`: fail-closed Blender-ID-to-USD mapping.
   - `_topology_fingerprints`: determines whether dirty mesh/light IDs require regeneration.
   - `_sparse_object_closure`, `_selected_stock_export`, `_write_scene_topology_delta`: sparse replacement, dependency closure, instancing siblings, removals, and mapping rewrites.
   - `_compile_add_on_opinions`: post-export add-on opinions; do not move evaluated mesh topology here.
2. `addon/ovrtx_blender_example/scene_generation_sessions.py`
   - `affected_blender_ids`, `topology_identity_changes`, `mark_scene_dirty`, and generation handoff establish dirty-ID and publication semantics.
3. `addon/ovrtx_blender_example/generation_runtime_adapters.py`
   - Preserve candidate activation, retained-value replay, rollback, and immutable generation handoff.
4. `addon/ovrtx_blender_example/topology_edit_fallback.py` and `blender_interactive_edit_builders.py`
   - Route topology-changing depsgraph edits to regeneration; never disguise topology as a typed value write.

## Implementation

1. Define the evaluated result and identity contract in tests. Cover an unchanged object, a modifier or Geometry Nodes result, changed vertex/index topology, shared mesh datablocks, collection/object instances, instance transforms, removal, and parent/material closure as applicable.
2. Extend temporary export preparation around `_stock_export` only if stock `evaluation_mode="VIEWPORT"` cannot express the case directly. Never mutate the saved `.blend` state.
3. Extend `_temporary_export_identities` and `_validated_blender_prim_paths` together when a new temporary or evaluated object needs durable correlation. Require one object identity and one schema path; reject missing or ambiguous mappings.
4. Extend `_topology_fingerprints` using deterministic evaluated data that changes exactly when authored topology must be replaced. Include instance source/collection identity and topology-affecting modifier state where required; exclude ordinary world transforms already handled by value edits.
5. Extend `_sparse_object_closure` so every selected replacement includes required parents, shared materials, instance sources, collection members, and affected siblings. Preserve explicit removal handling.
6. Update `_write_scene_topology_delta`, `_rewritten_mapping`, and `_updated_generation_mappings` only if the generated prim layout changes. Keep generation namespaces and predecessor composition intact.
7. Preserve `_validate_composed_generation`, generation digests, candidate rejection, and predecessor rollback. A later runtime pass must not repair an invalid authored generation.

## Tests

Add focused tests beside:

- `tests/test_scene_generation.py`: export preparation, fingerprints, identity mapping, sparse closure, topology deltas, and rollback.
- `tests/test_scene_generation_sessions.py`: depsgraph dirty IDs, generation reuse, concurrent edits, activation, and failed-candidate behavior.
- `tests/test_topology_edit_fallback.py`: topology classification and regeneration routing.
- `tests/test_scene_generation_contract.py`: Blender-backed authored USD evidence when the feature needs real evaluated geometry.

Run focused tests first, then the full suite:

```bash
python -m pytest -q \
  tests/test_scene_generation.py \
  tests/test_scene_generation_sessions.py \
  tests/test_topology_edit_fallback.py
python -m pytest -q tests
```

When Blender and an installed runtime are available, run the documented validator with a clean Blender profile. Require authored USD inspection before treating a runtime render as proof. Record runtime unavailability separately from source-test failure.

## Acceptance

- The generated USD contains the evaluated geometry/topology and preserves relative dependencies.
- Every supported Blender object maps uniquely to its object and schema prim.
- Unchanged topology reuses the generation; changed or removed topology creates a validated candidate.
- Sparse replacement carries all dependencies and leaves no stale predecessor prim active.
- Candidate failure leaves the predecessor live and retains pending edits.
- The installed runtime is consumed unchanged.

---
name: extend-ovrtx-semantic-aovs
description: Extend the OVRTX Blender add-on with stable semantic and instance identity authoring, native semantic render-variable requests, typed buffer parsing, ID maps, tight boxes, UI controls, and validation. Use when adding semantic or instance AOV support.
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
    - aovs
---
# Extend OVRTX semantic AOVs

## When to Use

Use when adding semantic or instance AOV support.

## Boundary

Implement stable authoring, product selection, parsing, and presentation in a
feature branch of the distributed source checkout. Treat renderer IDs as
runtime outputs, not persistent scene identity, and use documented typed
readback interfaces.

## Code map

- Add `semantic_identity.py` to inventory evaluated Blender instances, choose
  stable owners, and produce deterministic class/instance records. Keep this
  policy separate from renderer-generated numeric IDs.
- Use `addon/ovrtx_blender_example/scene_generation.py` and the existing
  geometry/material prim builders to attach semantic records to the same
  generated prim identities. If semantic opinions are overlays, contribute them
  through `ovrtx_scene_composition.py` rather than editing exported source USD.
- `addon/ovrtx_blender_example/properties.py` and `ui.py`: expose product
  selection and identity diagnostics.
- `addon/ovrtx_blender_example/render_requests.py`: replace a single
  color-only `render_var` assumption with normalized requested product
  descriptors while preserving the existing LdrColor/HdrColor path.
- `addon/ovrtx_blender_example/ovrtx_scene_composition.py`: author each
  RenderVar's exact `sourceName` and `orderedVars` membership; include product
  identity in composition/session digests.
- `addon/ovrtx_blender_example/ovrtx_runtime_client.py`: the current
  `render_result` branches accept only LdrColor/HdrColor. Add generic typed
  selection/decoding beside `_render_var_paths`; return a new typed AOV result
  carrying bytes/array, shape, dtype, product path, frame/time, and status.
- Add `semantic_aov.py` for active-count parsing, ID-map reconciliation,
  palettes, tight boxes, and derived overlays. Do not put semantic parsing in
  the native binding adapter.
- `addon/ovrtx_blender_example/ovrtx_session_controller.py` and viewport
  handoff code: preserve product identity and never feed integer/record buffers
  into the RGBA upload path.

## Instructions

1. Define stable class and instance ownership across Blender objects, linked
   data, collections, evaluated instances, USD composition, and flattening.
2. Author typed semantic schemas or supported primvars deterministically and
   retain an object/prim/label map. Never derive persistent IDs from traversal
   order or display colors.
3. Expose supported native products such as StableIdSegmentation,
   SemanticSegmentation, SemanticIdMap, and SemanticBoundingBox2DTight. Record
   product path, type, shape, dtype, orientation, frame, and camera.
4. Parse raw buffers losslessly, including active-count versus capacity rules.
   Keep raw integer maps separate from palettes, boxes, labels, and overlays.
5. Test identity propagation, request composition, buffer parsing, and stale
   frame/camera rejection. Validate a tiny occluded fixture, then a
   representative multi-object scene at first/middle/last frames.

## Required tests

Add `tests/test_semantic_identity.py`, `test_semantic_aov.py`, and
`test_semantic_aov_runtime.py`. Extend `test_ovrtx_runtime_client.py`,
`test_render_requests.py`, `test_generated_presentation_defs.py`,
`test_ovrtx_session.py`, and `test_viewport_handoff.py`. Fixtures must cover two
classes, two instances of one class, one occluder, padded tight-box capacity,
an ID map, unsupported products, stale frames, and lossless integer IDs.

## Client/server escalation gate

Use a direct-client fixture with known typed semantics and RenderVars. Escalate
only if a documented native product is unavailable, its typed buffer or ID map
cannot be retrieved through the official API, or the worker returns incorrect
data on the minimal fixture. Missing Blender identity policy, USD authoring,
product UI, parsing, or overlays is add-on work.

## Handoff

Return the focused commit, semantic fixture, unit and runtime tests, raw products,
ID maps, aligned previews, provenance report, and the boundary decision.
